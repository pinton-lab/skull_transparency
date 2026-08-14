"""Forward (transducer -> target) solve: transcranial vs free-field target pressure.

The rest of the package runs the OUTWARD direction: a virtual point source at the target
radiates out through the skull, and reciprocity turns that single solve into a whole-skull
transparency map and a placement. This module runs the direction the experiment itself
uses -- a focused bowl at a chosen placement drives INTO the head -- and answers the
question a placement alone cannot: how much pressure actually arrives at the target, and
how much of it the skull costs you.

Physics contract
----------------
The capability is a PAIR of solves that are identical in every respect except the medium:

  * same grid: the sim tree's own ``(Nx, Ny, Nz)`` interior box and pitch ``dX``;
  * same source: the same bowl-surface voxels (one Dirichlet pressure source cell each);
  * same drive: the same pulse at the same face amplitude ``p0`` (Pa), with the same
    geometric focusing delays -- computed in water for the same target, so the delay law
    never sees the skull;
  * same recording: the same focal box of point recorders around the target, the same time
    step and the same recorded frames;
  * (a) TRANSCRANIAL uses the subject medium (``c.f32`` [+ ``rho.f32`` / ``alpha.f32``]).
    The transducer stands off in water and the beam crosses bone on its way in;
  * (b) FREE FIELD replaces that medium with uniform water at ``c0``
    (:func:`water_medium_like`) -- the same coupling water, now all the way to the target.

Because only the medium differs, the ratio ``peak|p|_transcranial / peak|p|_freefield`` is
the INSERTION LOSS of the skull for THAT placement (linear ratio, and dB via
``20*log10(ratio)``). It folds together transmission through the bone, absorption inside
it, aberration/defocusing of the beam, and reflection back off the tables.

Interpretation
--------------
The number belongs to the placement, not to the skull: re-aim the bowl, move it to another
window, or change the frequency and it changes. Always report it together with the
placement (apex, aim, target) that produced it. The companion outputs -- the focal shift
(distance from the geometric target to where the peak actually landed) and the -6 dB FWHM
of each case -- say whether the skull merely attenuated the beam or also moved and smeared
the focus, which a single ratio hides. When the shift is an appreciable fraction of the
focal spot, quote ``transmission_at_target`` (the same ratio taken at the target voxel
itself) instead of the peak-to-peak ``transmission``.

Pressures are reported in Pa for a face drive of ``p0`` Pa (default 1 Pa), so they scale
linearly with the drive: multiply by your source amplitude to get absolute pressure.

Typical use::

    from skull_transparency.forward import run_forward_pair
    cmp = run_forward_pair("run/", "result/placement.json", "run/forward", gpu=0)
    print(cmp.transmission, cmp.transmission_db, cmp.focal_shift_mm)

or, equivalently, ``skull-transparency forward --sim run/ --placement result/placement.json
--out run/forward``.

Both solves are GPU work (the CUDA solver invoked exactly as the time-reversal launchers
invoke it); :func:`write_forward_pair` writes the two decks without running anything.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Coupling-medium defaults (m/s, kg/m^3) used to build the free-field twin.
WATER_C_MS = 1540.0
WATER_RHO_KGM3 = 1000.0

#: Grid axis the producer (:mod:`skull_transparency.sim.prepare`) rotates the approach
#: vector onto: the transducer sits at high +Z and fires toward the target. Used only to
#: seat a default bowl when the caller gives no placement.
_APPROACH_AXIS = np.array([0.0, 0.0, 1.0])

_CFL = 0.2          # the launchers' fixed Courant number (see sim/launchers.py)


# ---------------------------------------------------------------------------
# Free-field medium
# ---------------------------------------------------------------------------

def water_medium_like(c_map, c0: float = WATER_C_MS, *, rho_map=None, alpha_map=None,
                      rho_water: float | None = None, alpha_water: float | None = None):
    """Build the ALL-WATER twin of a subject medium: same grid, no skull.

    Returns ``(c, rho, alpha)`` with exactly the contract of
    :func:`skull_transparency.sim._common.rebuild_medium` -- ``c`` float64 (m/s), ``rho``
    float32 (kg/m^3), ``alpha`` float64 (dB/MHz/cm) or ``None`` -- so the same deck writer
    consumes either medium unchanged.

    Every voxel is set to the reference sound speed ``c0`` (m/s), i.e. the medium the
    transducer already couples through in the transcranial run is extended all the way to
    the target. Nothing else about the run changes, which is what makes the pair a clean
    insertion-loss measurement.

    Density: ``rho_water`` if given; else, when the subject run derives density from ``c``
    (``rho_map is None``), the same derivation evaluated at ``c0`` -- so the free-field
    water is bit-for-bit the water of the transcranial run; else 1000 kg/m^3, the value
    :mod:`skull_transparency.sim.prepare` fills outside the head.

    Absorption: ``None`` when the subject run has no explicit map, because the c-porosity
    model then returns water's 0.4 dB/MHz/cm everywhere on its own -- again identical to
    the coupling water of the transcranial run. With an explicit map, a uniform
    ``alpha_water`` is used (default: the median of the supplied map over the non-bone
    voxels, ``c <= c0``, i.e. the coupling water's own absorption).
    """
    from .sim._common import rho_from_c

    c_map = np.asarray(c_map)
    c0 = float(c0)
    c = np.full(c_map.shape, c0, dtype=np.float64)

    if rho_water is None:
        rho_val = float(rho_from_c(np.array([c0]))[0]) if rho_map is None else WATER_RHO_KGM3
    else:
        rho_val = float(rho_water)
    rho = np.full(c_map.shape, rho_val, dtype=np.float32)

    if alpha_map is None:
        alpha = None
    else:
        if alpha_water is None:
            water = np.asarray(alpha_map, float)[np.asarray(c_map) <= c0]
            alpha_water = float(np.median(water)) if water.size else 0.0
        alpha = np.full(c_map.shape, float(alpha_water), dtype=np.float64)
    return c, rho, alpha


# ---------------------------------------------------------------------------
# Source geometry: a focused bowl seated in the sim grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BowlSource:
    """A focused bowl seated in the sim grid, in DOMAIN-VOXEL coordinates.

    ``points_vox`` are the unique voxels of the bowl face -- one Dirichlet pressure source
    cell each. ``apex_vox`` is the face centre, ``aim`` the unit acoustic axis (apex ->
    target), and ``focus_vox = apex + roc*aim`` the bowl's own geometric focus; it
    coincides with the target when the apex stands exactly one radius of curvature off.
    The drive delays always steer to ``target_vox``, whatever the standoff.
    """
    points_vox: np.ndarray            # (M,3) int voxel coords on the bowl face
    apex_vox: np.ndarray              # (3,)
    aim: np.ndarray                   # (3,) unit, apex -> target
    focus_vox: np.ndarray             # (3,) apex + roc*aim
    target_vox: np.ndarray            # (3,) the steering point (the sim tree's target)
    roc_mm: float
    aperture_mm: float
    dx_mm: float

    @property
    def n_points(self) -> int:
        return int(len(self.points_vox))

    @property
    def standoff_mm(self) -> float:
        """Apex-to-target distance (mm)."""
        return float(np.linalg.norm(self.apex_vox - self.target_vox) * self.dx_mm)

    @property
    def focus_to_target_mm(self) -> float:
        """Geometric focus vs steering target (mm); 0 when standoff == ROC."""
        return float(np.linalg.norm(self.focus_vox - self.target_vox) * self.dx_mm)

    def to_dict(self) -> dict:
        return {
            "n_points": self.n_points,
            "apex_vox": np.asarray(self.apex_vox, float).tolist(),
            "aim": np.asarray(self.aim, float).tolist(),
            "focus_vox": np.asarray(self.focus_vox, float).tolist(),
            "target_vox": np.asarray(self.target_vox, float).tolist(),
            "roc_mm": float(self.roc_mm),
            "aperture_mm": float(self.aperture_mm),
            "dx_mm": float(self.dx_mm),
            "standoff_mm": self.standoff_mm,
            "focus_to_target_mm": self.focus_to_target_mm,
        }


def bowl_source(target_vox, apex_vox, dx_mm: float, *, roc_mm: float, aperture_mm: float,
                grid_shape=None, density: float = 1.0) -> BowlSource:
    """Build a :class:`BowlSource` from a face centre (``apex_vox``) and a target, both in
    domain-voxel coordinates.

    The cap point cloud is :func:`skull_transparency.transducer.build_cap` (the same bowl
    geometry the placement objective scores), rounded to unique grid voxels -- the solver
    drives one cell per voxel, so duplicates would only double-set the same cell.
    ``density`` thins the cap sampling (1.0 = continuous surface at grid density).

    ``grid_shape`` (Nx,Ny,Nz), when given, is checked: a bowl that pokes out of the
    interior grid cannot be driven, so this raises rather than silently clipping it into a
    dented dish.
    """
    from .transducer import build_cap

    target_vox = np.asarray(target_vox, float).ravel()
    apex_vox = np.asarray(apex_vox, float).ravel()
    aim = target_vox - apex_vox
    n = float(np.linalg.norm(aim))
    if n < 1e-9:
        raise ValueError("bowl apex coincides with the target; give an apex one radius of "
                         "curvature (or more) away from it.")
    aim = aim / n
    roc_vox = float(roc_mm) / float(dx_mm)
    half_angle_deg = float(np.degrees(np.arcsin(min(1.0, (aperture_mm / 2.0) / roc_mm))))
    pts, focus = build_cap(apex_vox, aim, roc_vox, half_angle_deg, density=density)
    pts = np.unique(np.rint(pts).astype(np.int64), axis=0)

    if grid_shape is not None:
        hi = np.asarray(grid_shape, np.int64) - 1
        out = ((pts < 0) | (pts > hi)).any(axis=1)
        if out.any():
            raise ValueError(
                f"{int(out.sum())} of {len(pts)} bowl voxels fall outside the interior grid "
                f"{tuple(int(s) for s in grid_shape)}. The sim tree is too small for this "
                "standoff/aperture: rebuild it with a larger --surround-mm / --standoff-mm, "
                "or move the apex closer to the target.")
    return BowlSource(points_vox=pts, apex_vox=apex_vox, aim=aim, focus_vox=focus,
                      target_vox=target_vox, roc_mm=float(roc_mm),
                      aperture_mm=float(aperture_mm), dx_mm=float(dx_mm))


def _placement_geometry(placement) -> dict:
    """Normalise a placement input to ``{apex_mm, target_mm, roc_mm, aperture_mm, frame}``.

    Accepts a :class:`~skull_transparency.placement.BowlPlacement`, either placement dict
    (``BowlPlacement.to_dict`` or the ``placement.json`` written by
    :func:`skull_transparency.neuromod.to_placement_dict`), or a path to such a JSON file.
    Missing entries come back as ``None``.
    """
    if placement is None:
        return {"apex_mm": None, "target_mm": None, "roc_mm": None,
                "aperture_mm": None, "frame": None}
    if isinstance(placement, (str, Path)):
        placement = json.loads(Path(placement).read_text())
    if not isinstance(placement, dict):                       # a BowlPlacement instance
        d = placement.to_dict()
        d["frame"] = getattr(placement, "world_frame", None)
        placement = d
    p = placement
    apex = p.get("apex_mni_mm") or p.get("xdc_center_lps") or p.get("apex_mm")
    target = p.get("target_mni_mm") or p.get("target_lps") or p.get("target_mm")
    roc = p.get("focal_length_mm") or p.get("apex_to_target_mm")
    radius = p.get("bowl_radius_mm")
    return {
        "apex_mm": None if apex is None else np.asarray(apex, float),
        "target_mm": None if target is None else np.asarray(target, float),
        "roc_mm": None if roc is None else float(roc),
        "aperture_mm": None if radius is None else 2.0 * float(radius),
        "frame": p.get("frame"),
    }


def bowl_source_from_placement(sim_dir, placement=None, *, apex_mm=None, target_mm=None,
                               apex_vox=None, target_vox=None, roc_mm=None,
                               aperture_mm=None, density: float = 1.0) -> BowlSource:
    """Seat a bowl in ``sim_dir``'s grid from a placement (or explicit coordinates).

    Resolution order, most explicit first:

      1. ``apex_vox`` / ``target_vox`` -- domain-voxel coordinates, used verbatim;
      2. ``apex_mm`` / ``target_mm`` -- world-frame millimetres, mapped through the tree's
         ``registration.json`` (the frame the producer recorded, e.g. MNI RAS mm);
      3. ``placement`` -- a ``BowlPlacement``/placement dict/``placement.json`` path, whose
         apex, target, focal length and bowl radius fill in whatever is still missing;
      4. defaults -- the target is the tree's own ``meta['dent_grid']`` and the bowl is
         seated one radius of curvature out along the grid ``+Z`` approach axis, which is
         the aim the tree was built for (``prepare`` rotates ``--approach`` onto ``+Z``).

    The device geometry defaults to the tree's own ``meta['transducer']`` block, then to
    the CTX-500. A placement in a world frame other than the tree's own is refused rather
    than silently mis-mapped.
    """
    from .registration import Registration
    from .sim import _common as C
    from .transducer import ROC_MM, APERTURE_MM

    sim_dir = Path(sim_dir)
    meta = C.load_meta(sim_dir)
    dx_mm = float(meta["dX_m"]) * 1e3
    gshape = C.grid_shape(meta)
    dev = meta.get("transducer") or {}
    pl = _placement_geometry(placement)

    roc_mm = float(roc_mm if roc_mm is not None else
                   (pl["roc_mm"] or dev.get("roc_mm") or ROC_MM))
    aperture_mm = float(aperture_mm if aperture_mm is not None else
                        (pl["aperture_mm"] or dev.get("aperture_mm") or APERTURE_MM))

    reg = None
    reg_path = sim_dir / "registration.json"

    def _to_vox(p_mm, what):
        nonlocal reg
        if reg is None:
            if not reg_path.exists():
                raise FileNotFoundError(
                    f"{reg_path} not found; world-mm {what} needs the tree's registration. "
                    "Pass voxel coordinates instead.")
            reg = Registration.from_json(reg_path)
        frame = pl.get("frame")
        if frame and frame != reg.world_frame:
            raise ValueError(
                f"placement is in frame {frame!r} but the sim tree's world frame is "
                f"{reg.world_frame!r}; mapping across frames is not supported here. Pass "
                "voxel coordinates, or re-run `place` against this tree's bundle.")
        return np.asarray(reg.mni_to_fullres(np.asarray(p_mm, float)), float).ravel()

    if target_vox is None:
        tgt_mm = target_mm if target_mm is not None else pl["target_mm"]
        target_vox = (_to_vox(tgt_mm, "target") if tgt_mm is not None
                      else np.asarray(meta["dent_grid"], float))
    target_vox = np.asarray(target_vox, float).ravel()

    if apex_vox is None:
        ap_mm = apex_mm if apex_mm is not None else pl["apex_mm"]
        apex_vox = (_to_vox(ap_mm, "apex") if ap_mm is not None
                    else target_vox + (roc_mm / dx_mm) * _APPROACH_AXIS)
    apex_vox = np.asarray(apex_vox, float).ravel()

    return bowl_source(target_vox, apex_vox, dx_mm, roc_mm=roc_mm, aperture_mm=aperture_mm,
                       grid_shape=gshape, density=density)


# ---------------------------------------------------------------------------
# Deck writing (one forward run) -- reuses the launcher primitives verbatim
# ---------------------------------------------------------------------------

def _geometric_drive(src_vox, target_vox, dX_m: float, c0: float, omega0: float,
                     dT: float, p0: float):
    """Focusing drive for a bowl: one pulse per source cell, delayed so that all cells
    arrive at the target together IN WATER.

    Identical recipe to the ``geo`` mode of
    :func:`skull_transparency.sim.launchers.launch_subset_focalbox` (the same
    :func:`~skull_transparency.sim.mlcompat.unit_pulse` and the same
    ``tau = (max_i d_i - d_i)/c0`` delay law, rounded to the time step). The delay law is
    computed in water and never sees the skull, so the transcranial and free-field runs
    are driven identically -- an aberration-corrected drive would confound the pair.

    Returns ``(icmat (M,nTic) float32 in Pa, duration_s, d_max_m)``.
    """
    from .sim.mlcompat import matlab_round, unit_pulse

    src = np.asarray(src_vox, float)
    d = np.linalg.norm(src - np.asarray(target_vox, float), axis=1) * float(dX_m)   # m
    p_unit, npn = unit_pulse(dT, omega0)
    tau = (d.max() - d) / c0
    shift = matlab_round(tau / dT).astype(np.int64)
    nTic = int(shift.max()) + npn
    icmat = np.zeros((src.shape[0], nTic), dtype=np.float64)
    for i in range(src.shape[0]):
        icmat[i, shift[i]:shift[i] + npn] = p_unit
    icmat *= float(p0)                       # face amplitude, Pa (peak of the unit pulse == 1)
    duration = 1.4 * d.max() / c0            # reach the focus and let the pulse pass through
    return icmat.astype(np.float32), float(duration), float(d.max())


def write_forward_deck(out_dir, meta: dict, source: BowlSource, c, rho, alpha, *,
                       box_half_mm: float = 12.0, modT: int = 2, p0: float = 1.0,
                       attenuation: bool = False, beta: float = 5.5,
                       write_maps: bool = True) -> dict:
    """Write ONE forward solver deck into ``out_dir`` and return its parameters.

    The deck is written with the same primitives as the time-reversal launchers
    (:func:`skull_transparency.sim.launch_core.launch_core` for the medium/coefficient
    files, :mod:`skull_transparency.sim.fwio` for the scalars). It records ONLY a focal box
    of point recorders around the target -- ``modX``/``modY``/``modZ`` are deliberately not
    written, so the solver skips the whole-volume ``genout_mod`` dump and the run costs a
    few hundred MB instead of tens of GB.

    ``box_half_mm`` is the half-width of that recording box (mm) about the target: wide
    enough to hold the focus (the peak must not land on its boundary) but not so wide that
    it reaches into bone, where the recorders would read the elevated in-bone pressure
    rather than the focus. ``modT`` is the temporal decimation of the recording (a frame
    every ``modT`` steps) and ``p0`` the drive amplitude at the bowl face (Pa).
    ``write_maps=False`` writes the coordinate/drive/scalar files only (useful for
    inspecting a deck without spending the multi-GB medium maps).

    ``c``/``rho``/``alpha`` are the medium as :func:`sim._common.rebuild_medium` returns it
    -- pass the subject medium for the transcranial run and
    :func:`water_medium_like` for the free-field twin. Nothing else may differ between the
    two decks.
    """
    import os

    from .sim import fwio, _common as C
    from .sim.launch_core import launch_core
    from .sim.launchers import _write_icmat

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = np.asarray(c, np.float64)
    Nx, Ny, Nz = c.shape
    dX = float(meta["dX_m"]); c0 = float(meta["C0"]); f0 = float(meta["F0"])
    omega0 = 2 * np.pi * f0
    lam = c0 / f0
    ppw = lam / dX
    dT = dX / c0 * _CFL

    src = np.asarray(source.points_vox, np.int64)
    if src.size == 0 or (src < 0).any() or (src >= np.array([Nx, Ny, Nz])).any():
        raise ValueError(f"bowl-face voxels fall outside the interior grid ({Nx},{Ny},{Nz}); "
                         "rebuild the sim tree with more surround, or move the apex in.")
    target_vox = np.asarray(source.target_vox, float)
    icmat, duration, dmax_m = _geometric_drive(src, target_vox, dX, c0, omega0, dT, p0)
    nTic = int(icmat.shape[1])

    # incoords: one driven cell per bowl voxel (col4 = cell id, col5 = 1), as the subset
    # launchers lay out their driven array elements.
    incoords = np.empty((src.shape[0], 5), dtype=np.float64)
    incoords[:, 0:3] = src
    incoords[:, 3] = np.arange(1, src.shape[0] + 1)
    incoords[:, 4] = 1

    fb = int(round(float(box_half_mm) / (dX * 1e3)))
    if fb < 2:
        raise ValueError(f"box_half_mm={box_half_mm} is under two voxels at dx="
                         f"{dX * 1e3:.3f} mm; widen the recording box.")
    box = C.focal_box(target_vox, (Nx, Ny, Nz), fb)
    if len(box) == 0:
        raise ValueError("the focal box is empty: the target lies outside the grid.")
    outcoords = box

    cwd = os.getcwd(); os.chdir(out_dir)
    try:
        fwio.writeVabs("int", int(modT), "modT")     # no modX/Y/Z -> no genout_mod volume dump
        launch_core(c0, omega0, Nx * dX, Ny * dX, Nz * dX, duration, p0, ppw, _CFL,
                    c, rho, incoords, outcoords, nTic, write_maps=write_maps,
                    attenuation=attenuation, alpha_map=alpha, betaval=beta)
        fwio.writeVabs("int", nTic, "nTic")
        fwio.writeVabs("int", 0, "ncoords_add")
        _write_icmat(icmat)
    finally:
        os.chdir(cwd)

    np.save(out_dir / "box_vox.npy", box[:, :3].astype(np.int32))
    nT = int(round(duration * c0 / lam * ppw / _CFL))
    info = {
        "n_box": int(len(box)), "fb_vox": fb, "box_half_mm": float(box_half_mm),
        "modT": int(modT), "nTic": nTic, "nT": nT,
        "n_frames_expected": max(0, (nT - 1) // int(modT)),
        "duration_s": float(duration), "dT_s": float(dT), "dt_frame_s": float(dT * modT),
        "dx_mm": float(dX * 1e3), "c0_ms": c0, "f0_hz": f0, "p0_pa": float(p0),
        "attenuation": bool(attenuation), "beta": float(beta),
        "grid_shape": [Nx, Ny, Nz], "target_vox": target_vox.tolist(),
        "source_max_distance_mm": float(dmax_m * 1e3),
        "source": source.to_dict(),
    }
    (out_dir / "forward_deck.json").write_text(json.dumps(info, indent=1))
    return info


# ---------------------------------------------------------------------------
# Focal-box analysis
# ---------------------------------------------------------------------------

def _parabolic_peak(y0, y1, y2):
    """Peak of the parabola through ``(-1,y0), (0,y1), (1,y2)`` when it lies between the
    outer samples, else ``y1``.

    The traces are decimated in time by ``modT``, so the true crest of the waveform almost
    never falls exactly on a recorded sample; the fit recovers most of that lost amplitude.
    Both runs are corrected the same way, so the ratio is not skewed by the correction.
    """
    y0, y1, y2 = (np.asarray(v, float) for v in (y0, y1, y2))
    den = y0 - 2.0 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(den < 0, (y0 - y2) / (2.0 * den), 0.0)
    x = np.where(np.isfinite(x) & (np.abs(x) <= 0.5), x, 0.0)
    return y1 + x * (y2 - y0) / 2.0 + x * x * den / 2.0


def peak_pressure(traces) -> np.ndarray:
    """Peak ``|p|`` (Pa) per recorder from a ``(n_frames, n_points)`` trace block, with the
    sub-sample refinement of :func:`_parabolic_peak`."""
    a = np.abs(np.asarray(traces, float))
    if a.ndim != 2:
        raise ValueError(f"traces must be (n_frames, n_points), got shape {a.shape}")
    n = a.shape[0]
    if n < 3:
        return a.max(axis=0) if n else np.zeros(a.shape[1])
    i = a.argmax(axis=0)
    ic = np.clip(i, 1, n - 2)
    ch = np.arange(a.shape[1])
    return np.maximum(a[i, ch], _parabolic_peak(a[ic - 1, ch], a[ic, ch], a[ic + 1, ch]))


def _box_volume(values, box_vox):
    """Scatter per-recorder ``values`` onto the focal box's own lattice.

    Returns ``(vol, axes)`` where ``vol`` is ``(nx,ny,nz)`` (NaN where the box was clipped
    by the grid edge) and ``axes`` are the sorted voxel indices along each grid axis.
    """
    box = np.asarray(box_vox)
    axes = [np.unique(box[:, a]) for a in range(3)]
    idx = [np.searchsorted(axes[a], box[:, a]) for a in range(3)]
    vol = np.full([len(a) for a in axes], np.nan, dtype=float)
    vol[idx[0], idx[1], idx[2]] = np.asarray(values, float)
    return vol, axes


def _fwhm_1d(profile, step_mm: float, level: float = 0.5) -> float:
    """Full width (mm) where ``profile`` stays above ``level`` times its maximum, measured
    through that maximum with linear interpolation on the two crossings.

    ``level=0.5`` on a pressure profile is the -6 dB width (half the peak pressure, a
    quarter of the peak intensity). Returns ``inf`` when the profile never falls to the
    level inside the recorded box -- the box is then too small to size the focus.
    """
    y = np.nan_to_num(np.asarray(profile, float), nan=0.0)
    n = y.size
    if n == 0 or y.max() <= 0:
        return float("nan")
    thr = level * y.max()
    im = int(np.argmax(y))
    l = im
    while l > 0 and y[l] > thr:
        l -= 1
    r = im
    while r < n - 1 and y[r] > thr:
        r += 1
    if y[l] > thr or y[r] > thr:
        return float("inf")
    cross = lambda a, b: a + (y[a] - thr) / (y[a] - y[b] + 1e-30) * (b - a)
    return float((cross(r, r - 1) - cross(l, l + 1)) * step_mm)


@dataclass
class FocalMetrics:
    """What one recorded focal box says about the beam that made it.

    ``peak_pa`` is the largest peak ``|p|`` (Pa) anywhere in the box and ``peak_vox`` is
    where it landed; ``peak_at_target_pa`` is the peak ``|p|`` of the recorder sitting on
    the geometric target itself, which is the honest number when the beam missed.
    ``focal_shift_mm`` is the distance between the two. ``fwhm_mm`` are the -6 dB widths
    (mm) along the three grid axes through the peak (``inf`` when the box is too small to
    contain the -6 dB contour); the bowl axis is generally oblique to the grid, so read the
    three together rather than as axial/lateral.

    ``on_box_edge`` flags that the peak landed on the boundary of the recorded box, i.e.
    the true maximum may lie outside it -- widen ``box_half_mm`` and re-run before quoting
    such a number.
    """
    peak_pa: float
    peak_vox: tuple
    peak_at_target_pa: float
    focal_shift_mm: float
    fwhm_mm: tuple
    target_vox: tuple
    dx_mm: float
    n_points: int = 0
    on_box_edge: bool = False
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "peak_pa": float(self.peak_pa),
            "peak_vox": [float(v) for v in self.peak_vox],
            "peak_at_target_pa": float(self.peak_at_target_pa),
            "focal_shift_mm": float(self.focal_shift_mm),
            "fwhm_mm": [float(v) for v in self.fwhm_mm],
            "target_vox": [float(v) for v in self.target_vox],
            "dx_mm": float(self.dx_mm),
            "n_points": int(self.n_points),
            "on_box_edge": bool(self.on_box_edge),
        }


def focal_metrics(traces, box_vox, target_vox, dx_mm: float, *, label: str = "") -> FocalMetrics:
    """Reduce one recorded focal box to :class:`FocalMetrics`.

    ``traces`` is the ``(n_frames, n_points)`` block of the box recorders (Pa) and
    ``box_vox`` their ``(n_points, 3)`` voxel coordinates, in the same channel order.
    """
    pk = peak_pressure(traces)
    box = np.asarray(box_vox, float)
    vol, axes = _box_volume(pk, box)
    flat = int(np.nanargmax(vol))
    i, j, k = np.unravel_index(flat, vol.shape)
    peak_vox = np.array([axes[0][i], axes[1][j], axes[2][k]], float)
    target_vox = np.asarray(target_vox, float).ravel()
    shift_mm = float(np.linalg.norm(peak_vox - target_vox) * dx_mm)
    fwhm = (_fwhm_1d(vol[:, j, k], dx_mm), _fwhm_1d(vol[i, :, k], dx_mm),
            _fwhm_1d(vol[i, j, :], dx_mm))
    edge = bool(i in (0, vol.shape[0] - 1) or j in (0, vol.shape[1] - 1)
                or k in (0, vol.shape[2] - 1))
    if edge:
        warnings.warn(f"the {label or 'focal'} peak sits on the boundary of the recorded box; "
                      "the true maximum may lie outside it -- widen box_half_mm and re-run.")
    at = [int(np.argmin(np.abs(np.asarray(axes[a], float) - target_vox[a]))) for a in range(3)]
    return FocalMetrics(peak_pa=float(vol[i, j, k]), peak_vox=tuple(peak_vox),
                        peak_at_target_pa=float(np.nan_to_num(vol[at[0], at[1], at[2]])),
                        focal_shift_mm=shift_mm, fwhm_mm=fwhm,
                        target_vox=tuple(target_vox), dx_mm=float(dx_mm),
                        n_points=int(len(box)), on_box_edge=edge, label=label)


def load_focal_box(run_dir, *, max_load_bytes: int = 4 << 30):
    """Load a solved forward run: ``(traces (n_frames, n_points), box_vox, deck_info)``.

    ``traces`` are the focal-box recorders in Pa (the solver writes ``genout.dat``
    frame-major, one frame every ``modT`` steps).
    """
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "forward_deck.json").read_text())
    box = np.load(run_dir / "box_vox.npy").astype(np.int64)
    g = run_dir / "genout.dat"
    if not g.exists():
        raise FileNotFoundError(f"{g} not found -- this forward deck has not been solved "
                                "(run it with run_solver=True / without --no-run).")
    nbytes = g.stat().st_size
    if nbytes > max_load_bytes:
        raise MemoryError(
            f"{g} is {nbytes / 2**30:.1f} GiB; re-run with a smaller recording box "
            "(--box-mm) or a coarser --modt, or raise max_load_bytes.")
    n_box = int(len(box))
    traces = np.fromfile(g, dtype="<f4").reshape(-1, n_box)
    return traces, box, info


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

@dataclass
class ForwardComparison:
    """Transcranial vs free-field pressure at one target, for ONE transducer placement.

    ``transmission`` is ``p_transcranial_pa / p_freefield_pa`` (linear) and
    ``transmission_db`` is ``20*log10`` of it -- the insertion loss of the skull along that
    beam, so it is normally <= 1 (negative dB). ``transmission_at_target`` is the same ratio
    taken at the target voxel itself rather than at each case's own peak -- the one to quote
    once the skull has moved the focus. ``focal_shift_mm`` / ``fwhm_mm`` are the
    transcranial case (the free-field ones sit in ``free_field``); pressures are in Pa for
    the deck's face drive ``p0`` (Pa) and scale linearly with it.

    A value above 1 is not automatically a bug: a skull segment can act as a weak lens and
    concentrate the beam, and a recording box that reaches into bone will read the elevated
    in-bone pressure. Check ``focal_shift_mm`` and ``FocalMetrics.on_box_edge`` before
    believing it, and remember the number describes this placement, not the skull at large.
    """
    p_transcranial_pa: float
    p_freefield_pa: float | None
    transmission: float | None
    transmission_db: float | None
    focal_shift_mm: float
    fwhm_mm: tuple
    freefield_focal_shift_mm: float | None
    freefield_fwhm_mm: tuple | None
    transcranial: FocalMetrics
    free_field: FocalMetrics | None = None
    transmission_at_target: float | None = None    # same ratio, but at the target voxel
    transmission_at_target_db: float | None = None
    dx_mm: float = 0.0
    target_vox: tuple = ()
    run_dirs: dict = field(default_factory=dict)
    source: dict | None = None

    @classmethod
    def from_metrics(cls, transcranial: FocalMetrics, free_field: FocalMetrics | None = None,
                     *, run_dirs=None, source=None) -> "ForwardComparison":
        """Assemble from the two per-case metrics (the free-field case may be absent when
        the pair was run with ``free_field=False``).

        Two ratios are formed: ``transmission`` compares the two PEAK pressures (what the
        beam can deliver at all), ``transmission_at_target`` compares the pressure at the
        target voxel itself (what it delivers where you aimed). They differ exactly by how
        far the skull moved the focus, so quote the second one whenever
        ``focal_shift_mm`` is an appreciable fraction of the focal spot.
        """
        ratio_of = lambda a, b: (a / b) if (b and b > 0) else None
        db_of = lambda r: (20.0 * np.log10(r)) if (r and r > 0) else None
        p_tc = float(transcranial.peak_pa)
        p_ff = None if free_field is None else float(free_field.peak_pa)
        ratio = ratio_of(p_tc, p_ff)
        at = (None if free_field is None
              else ratio_of(float(transcranial.peak_at_target_pa),
                            float(free_field.peak_at_target_pa)))
        return cls(
            p_transcranial_pa=p_tc, p_freefield_pa=p_ff,
            transmission=(None if ratio is None else float(ratio)),
            transmission_db=(None if db_of(ratio) is None else float(db_of(ratio))),
            focal_shift_mm=float(transcranial.focal_shift_mm),
            fwhm_mm=tuple(transcranial.fwhm_mm),
            freefield_focal_shift_mm=(None if free_field is None
                                      else float(free_field.focal_shift_mm)),
            freefield_fwhm_mm=(None if free_field is None else tuple(free_field.fwhm_mm)),
            transcranial=transcranial, free_field=free_field,
            transmission_at_target=(None if at is None else float(at)),
            transmission_at_target_db=(None if db_of(at) is None else float(db_of(at))),
            dx_mm=float(transcranial.dx_mm), target_vox=tuple(transcranial.target_vox),
            run_dirs=dict(run_dirs or {}), source=source)

    def to_dict(self) -> dict:
        return {
            "schema": "skull_transparency.forward_comparison/1",
            "p_transcranial_pa": float(self.p_transcranial_pa),
            "p_freefield_pa": (None if self.p_freefield_pa is None
                               else float(self.p_freefield_pa)),
            "transmission": (None if self.transmission is None else float(self.transmission)),
            "transmission_db": (None if self.transmission_db is None
                                else float(self.transmission_db)),
            "focal_shift_mm": float(self.focal_shift_mm),
            "fwhm_mm": [float(v) for v in self.fwhm_mm],
            "freefield_focal_shift_mm": (None if self.freefield_focal_shift_mm is None
                                         else float(self.freefield_focal_shift_mm)),
            "freefield_fwhm_mm": (None if self.freefield_fwhm_mm is None
                                  else [float(v) for v in self.freefield_fwhm_mm]),
            "transcranial": self.transcranial.to_dict(),
            "free_field": (None if self.free_field is None else self.free_field.to_dict()),
            "transmission_at_target": (None if self.transmission_at_target is None
                                       else float(self.transmission_at_target)),
            "transmission_at_target_db": (None if self.transmission_at_target_db is None
                                          else float(self.transmission_at_target_db)),
            "dx_mm": float(self.dx_mm),
            "target_vox": [float(v) for v in self.target_vox],
            "run_dirs": {k: str(v) for k, v in self.run_dirs.items()},
            "source": self.source,
        }

    def summary(self) -> str:
        """One-paragraph human summary (the CLI prints this)."""
        lines = [f"peak pressure    transcranial {self.p_transcranial_pa:.4g} Pa"]
        if self.transmission is not None:
            lines[0] += f"   free field {self.p_freefield_pa:.4g} Pa"
            lines.append(f"transmission     {self.transmission:.4f}  "
                         f"({self.transmission_db:+.2f} dB insertion loss for this placement)")
        if self.transmission_at_target is not None:
            lines.append(f"  at the target  {self.transmission_at_target:.4f}  "
                         f"({self.transmission_at_target_db:+.2f} dB; "
                         f"{self.transcranial.peak_at_target_pa:.4g} Pa vs "
                         f"{self.free_field.peak_at_target_pa:.4g} Pa)")
        lines.append(f"focal shift      {self.focal_shift_mm:.2f} mm from the target"
                     + ("" if self.freefield_focal_shift_mm is None else
                        f"  (free field {self.freefield_focal_shift_mm:.2f} mm)"))
        lines.append("-6 dB FWHM (mm)  transcranial "
                     + " x ".join(f"{v:.2f}" for v in self.fwhm_mm)
                     + ("" if self.freefield_fwhm_mm is None else
                        "   free field " + " x ".join(f"{v:.2f}" for v in self.freefield_fwhm_mm)))
        return "\n".join(lines)


def compare_focal_boxes(traces_transcranial, traces_freefield, box_vox, target_vox,
                        dx_mm: float, *, run_dirs=None, source=None) -> ForwardComparison:
    """Compare two focal boxes recorded on the SAME box with the SAME source and drive.

    ``traces_freefield=None`` yields a transcranial-only comparison (no ratio). This is the
    pure post-processing half of :func:`run_forward_pair` and needs no solver.
    """
    tc = focal_metrics(traces_transcranial, box_vox, target_vox, dx_mm, label="transcranial")
    ff = (None if traces_freefield is None
          else focal_metrics(traces_freefield, box_vox, target_vox, dx_mm, label="free_field"))
    return ForwardComparison.from_metrics(tc, ff, run_dirs=run_dirs, source=source)


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------

def _subject_medium(sim_dir, meta):
    """``(c, rho, alpha)`` of the subject medium described by ``meta`` (the c-map plus the
    optional density / absorption maps the producer wrote)."""
    from .sim import _common as C
    return C.rebuild_medium(str(sim_dir), {
        "kind": "maps", "file": meta.get("c_file", "halle_c.f32"),
        "N": list(C.grid_shape(meta)),
        "rho_file": meta.get("rho_file"), "alpha_file": meta.get("alpha_file")})


def write_forward_pair(sim_dir, placement=None, out_dir="forward", *, free_field: bool = True,
                       box_half_mm: float = 12.0, modT: int = 2, p0: float = 1.0,
                       roc_mm=None, aperture_mm=None, density: float = 1.0,
                       apex_vox=None, target_vox=None, apex_mm=None, target_mm=None,
                       source: BowlSource | None = None, run_solver: bool = False,
                       gpu: int = 0, write_maps: bool = True, log=None) -> dict:
    """Write (and optionally solve) the transcranial + free-field forward decks.

    Returns ``{"source": BowlSource, "transcranial": (dir, info), "free_field": (dir, info)
    or None}``. The two decks differ ONLY in their medium maps: same grid, same bowl
    voxels, same drive, same focal box (see the module docstring). The medium maps are
    written fresh for each -- they must never be hardlinked between the two runs, which is
    exactly what would silently turn the pair into two identical solves.
    """
    from .sim import _common as C
    from .sim.launchers import _maybe_run

    sim_dir = Path(sim_dir)
    out_dir = Path(out_dir)
    meta = C.load_meta(sim_dir)
    if source is None:
        source = bowl_source_from_placement(
            sim_dir, placement, apex_mm=apex_mm, target_mm=target_mm, apex_vox=apex_vox,
            target_vox=target_vox, roc_mm=roc_mm, aperture_mm=aperture_mm, density=density)

    c, rho, alpha = _subject_medium(sim_dir, meta)
    attenuation = bool(meta.get("attenuation") or meta.get("alpha_file"))
    beta = float(meta.get("beta", 5.5))

    # A bowl seated inside the head would report a huge insertion loss that is really a
    # placement error, so say so at launch rather than at read-out. 200 m/s above the
    # reference speed is well clear of soft tissue and below any bone, human or rodent.
    src = np.asarray(source.points_vox, np.int64)
    hard = float(meta.get("C0", WATER_C_MS)) + 200.0
    in_bone = int((c[src[:, 0], src[:, 1], src[:, 2]] > hard).sum())
    if in_bone:
        warnings.warn(f"{in_bone} of {len(src)} bowl-face voxels sit in tissue faster than "
                      f"{hard:.0f} m/s (i.e. in bone) -- the transducer is not standing off "
                      "in water; check the placement.")

    if log:
        log(f"forward: {source.n_points} bowl voxels, apex {np.round(source.apex_vox, 1)} vox, "
            f"standoff {source.standoff_mm:.1f} mm, target {np.round(source.target_vox, 1)} vox")

    media = [("transcranial", (c, rho, alpha))]
    if free_field:
        media.append(("free_field", water_medium_like(
            c, float(meta["C0"]),
            rho_map=(rho if meta.get("rho_file") else None), alpha_map=alpha)))

    decks = {"free_field": None}
    for label, medium in media:
        d = out_dir / label
        info = write_forward_deck(d, meta, source, *medium, box_half_mm=box_half_mm,
                                  modT=modT, p0=p0, attenuation=attenuation, beta=beta,
                                  write_maps=write_maps)
        if log:
            log(f"  wrote {label} deck {d} ({info['n_box']} box recorders, "
                f"{info['n_frames_expected']} frames)")
        if run_solver:
            _maybe_run(str(d), None, True, gpu)
            if log:
                log(f"  solved {label}")
        decks[label] = (d, info)
    return {"source": source, **decks}


def run_forward_pair(sim_dir, placement=None, out_dir="forward", *, free_field: bool = True,
                     gpu: int = 0, box_half_mm: float = 12.0, modT: int = 2, p0: float = 1.0,
                     roc_mm=None, aperture_mm=None, density: float = 1.0,
                     apex_vox=None, target_vox=None, apex_mm=None, target_mm=None,
                     source: BowlSource | None = None, write_json: bool = True,
                     log=None) -> ForwardComparison:
    """Run the forward PAIR and compare: what the target sees through the skull, and what
    it would have seen with the skull replaced by water.

    ``sim_dir`` is a prepared sim tree (``skull-transparency prepare``); ``placement`` is a
    ``BowlPlacement``/placement dict/``placement.json`` path, or ``None`` to seat the bowl
    on the tree's own approach axis (see :func:`bowl_source_from_placement`). Both solves
    run on GPU ``gpu``; ``free_field=False`` skips the second one (and the ratio).

    Writes ``<out_dir>/forward.json`` (the :meth:`ForwardComparison.to_dict` payload) and
    leaves both solved decks in place for inspection. See the module docstring for the
    physics contract this pair depends on and for how to read the ratio.
    """
    out_dir = Path(out_dir)
    decks = write_forward_pair(
        sim_dir, placement, out_dir, free_field=free_field, box_half_mm=box_half_mm,
        modT=modT, p0=p0, roc_mm=roc_mm, aperture_mm=aperture_mm, density=density,
        apex_vox=apex_vox, target_vox=target_vox, apex_mm=apex_mm, target_mm=target_mm,
        source=source, run_solver=True, gpu=gpu, log=log)

    src = decks["source"]
    tc_dir = decks["transcranial"][0]
    traces_tc, box, info = load_focal_box(tc_dir)
    run_dirs = {"transcranial": str(tc_dir)}
    traces_ff = None
    if decks["free_field"] is not None:
        ff_dir = decks["free_field"][0]
        traces_ff, box_ff, _ = load_focal_box(ff_dir)
        if not np.array_equal(box, box_ff):
            raise ValueError("the two forward runs recorded different focal boxes; the pair "
                             "is not comparable (re-run both decks from one call).")
        run_dirs["free_field"] = str(ff_dir)

    cmp = compare_focal_boxes(traces_tc, traces_ff, box, src.target_vox,
                              float(info["dx_mm"]), run_dirs=run_dirs, source=src.to_dict())
    if write_json:
        (out_dir / "forward.json").write_text(json.dumps(cmp.to_dict(), indent=1))
    if log:
        log(cmp.summary())
    return cmp
