#!/usr/bin/env python3
"""MOUSE case 1, the movie: the outward time-reversal wave leaving the cerebellum.

The paper's transparency map is a *time-collapsed* quantity -- one number per patch of
calvaria -- so it hides the thing it is actually measuring: a spherical wave born at the
target, sweeping outward, and being reshaped by every millimetre of bone it crosses. This
script renders that wave. It is the mouse counterpart of the human propagation animation
(manuscript Fig. 3): the sound-speed map in grey, the signed pressure field over it in a
diverging colormap, the target marked with a cross, and the elapsed time in microseconds.

WHY THIS NEEDS ITS OWN SOLVE
``run_mouse_tips_cerebellum.py`` runs the outward solve with ``recorder="shell"``, which
records ONLY the calvarial-surface standoff points. That is the right choice for the
transparency map (it is ~1000x less data), but it means there is no volume field to
animate -- the shell run stores a time-series per surface patch and nothing in between.
So this script re-runs the SAME outward solve with ``recorder="volume"``, which makes the
solver dump the decimated full field to ``genout_mod.dat``. Everything else -- the medium,
the source position, the grid, the pulse -- is imported from the case module, so the movie
shows exactly the wave the shipped transparency map was computed from.

COST. The decimation is ``launch_outward``'s fixed modX=modY=modZ=2, modT=8. On this
252x292x212 grid (dx = 0.128 mm) the padded, decimated field is 174x194x154 float32 =
20.8 MB per recorded frame, and the run records ~209 frames, so ``genout_mod.dat`` is
~4.3 GB. That is small enough to keep in /dev/shm for the few minutes the rendering
takes, and ``all`` deletes it afterwards. Only three planes through the target are ever
pulled out of it, a few MB in RAM.

DISPLAY. The source is a point in a 20 mm head, so |p| falls by more than an order of
magnitude between the target and the far skull: on a linear colour scale the first two
millimetres would be all you ever saw. The overlay is therefore compressed with a signed
power law (:data:`GAMMA`) about ONE global limit, the 99.5th percentile of |p| over every
displayed pixel and frame with a 1.5 mm ball around the source excluded. The limit being
global is what stops the scale flickering; the compression being monotonic and signed is
what keeps a wavefront reading as a red/blue oscillation with white (and transparent) at
zero. What survives past ~15 us is not noise -- it is genuine re-radiation from the
reverberating skull, falling off as 1/r and decaying with time -- so it is shown, but the
opacity floor (:data:`ALPHA_LO`) keeps it a wash rather than a competitor to the front.

Run (the solve takes well under a minute on one A6000):

    GPU=1 python make_propagation_movie.py              # solve + render + drop the dump
    GPU=1 python make_propagation_movie.py solve        # phases separately
    python make_propagation_movie.py render
    python make_propagation_movie.py sheet              # contact sheet only (fast iteration)
    python make_propagation_movie.py verify             # physics checks, no figures
    python make_propagation_movie.py clean              # remove the scratch tree

Environment knobs: ``GPU`` (default 1), ``SCRATCH_MOVIE`` (default /dev/shm/...),
``TSTRIDE`` (temporal stride for the movie, default 2), ``FPS`` (default 12),
``KEEP_DUMP=1`` (do not delete ``genout_mod.dat`` after ``all``).

Outputs, all next to this script:
  * ``propagation_mouse_tips_cerebellum.mp4``   -- the movie (3 anatomical panels);
  * ``propagation_mouse_tips_cerebellum.gif``   -- the same, sagittal only, for previews;
  * ``propagation_contact_sheet.png``           -- 9 sagittal frames, so the result can be
    checked at a glance without playing anything. This is the one small output that is
    committed; the mp4/gif are regenerated, not tracked.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")                       # render off-screen; no display is needed
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                                  # the case module lives here
sys.path.insert(0, str(HERE.parents[1] / "src"))               # the package, run from a checkout

# The case: medium ramp, cerebellum centroid, TIPS spec, bone threshold, surround. Imported
# rather than copied so the movie can never drift from the run that produced the map.
import run_mouse_tips_cerebellum as case
from skull_transparency.sim import _common as C
from skull_transparency.sim.launch_core import PAD            # 48 absorbing-layer voxels per side
from skull_transparency.sim.prepare import build_brain_center_run

# --------------------------------------------------------------------------- paths
#: Scratch for the volume-recorder solve. Deliberately NOT the case's own SCRATCH: this run
#: writes a multi-GB genout_mod that the shell run does not, and keeping the trees apart
#: means re-running the movie can never disturb a shipped bundle.
SCRATCH = Path(os.environ.get("SCRATCH_MOVIE", "/dev/shm/mouse_tips_cb_movie"))
MP4 = HERE / "propagation_mouse_tips_cerebellum.mp4"
GIF = HERE / "propagation_mouse_tips_cerebellum.gif"
SHEET = HERE / "propagation_contact_sheet.png"

# --------------------------------------------------------------------------- display
TSTRIDE = int(os.environ.get("TSTRIDE", 2))    # keep every Nth recorded frame in the movie
FPS = int(os.environ.get("FPS", 12))
PCTL = 99.5           # percentile of |p| (over all displayed pixels/frames) that sets the limit
GAMMA = 0.60          # signed power-law compression of p/vmax; 1.0 = linear
#: Where the overlay starts to appear and where it reaches full opacity, in COMPRESSED units.
#: Tuned against this run: the outgoing front is ~0.02-0.05 Pa once it is past the skull while
#: the late reverberation tail sits near 0.005 Pa, so a floor at 0.10 keeps the tail as a faint
#: wash (it is real, not noise) and lets the front read as the solid feature it is.
ALPHA_LO, ALPHA_HI = 0.10, 0.40
ALPHA_MAX = 0.90
NEAR_SOURCE_MM = 1.5  # radius excluded when setting vmax, so the source singularity does not
#                       swallow the whole colour scale (the wave, not the source, is the subject)
SHEET_N = 9           # frames on the contact sheet
#: Greyscale window (m/s) for the sound-speed background. The bottom deliberately sits WELL
#: below water (1540 m/s): the overlay is translucent where the field is weak, and a pale
#: half-transparent blue composited onto pure black reads as near-black hatching. Putting
#: water at ~20% grey instead keeps the weak far field looking weak. Bone (median 2716 m/s
#: here) still reaches ~90% white.
BG_VMIN, BG_VMAX = 1200.0, 2900.0


# ============================================================================ solve

def solve():
    """Re-run the outward solve with the volume recorder, so a full field is on disk.

    Identical to ``run_mouse_tips_cerebellum.solve`` except for ``recorder="volume"``: same
    sound-speed map, same cerebellum source, same 12 ppw grid, same pulse.
    """
    c_map, affine = case._c_map()
    center = case.cerebellum_center_ras_mm()
    print(f"  c-map {c_map.shape} @ {abs(affine[0, 0]) * 1e3:.0f} um -> grid dx "
          f"{case.TIPS.dx_mm:.3f} mm ({case.TIPS.ppw:.0f} ppw at {case.TIPS.f0_hz / 1e6:.0f} MHz)")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(c_map, affine, case.TIPS, SCRATCH, center_phys_mm=center,
                                 bone_threshold=case.BONE_THRESHOLD, surround_mm=case.SURROUND_MM,
                                 input_frame=case.INPUT_FRAME)
    meta = json.loads((Path(sim) / "meta.json").read_text())
    gs = meta.get("grid_shape", [meta["N"]] * 3)
    print(f"  sim tree {sim}  grid {gs[0]}x{gs[1]}x{gs[2]}")

    os.environ.setdefault("FULLWAVE2_BIN", "/celerina/gfp/mfs/fullwave2-ultra/bin/bench_3d_opt")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "1")
    from skull_transparency.sim.launchers import launch_outward
    outdir = launch_outward(str(sim), str(SCRATCH), run_solver=True, recorder="volume")
    if not (Path(outdir) / "SUCCESS").exists():
        raise SystemExit("solver did not write SUCCESS; check the run log")
    gm = Path(outdir) / "genout_mod.dat"
    print(f"SOLVE DONE -> {outdir}\n  genout_mod.dat = {gm.stat().st_size / 1e9:.2f} GB")
    return outdir


# ============================================================================ loading

class Field:
    """The recorded field plus everything needed to place it in space and time.

    Attributes carry units in their names. ``vol`` is a memory-map of the padded, decimated
    dump with shape ``(nframes, nXp, nYp, nZp)``; the interior grid starts at ``lo = PAD//mod``
    along every axis, and decimated index ``k`` samples full-res interior voxel ``k*mod``.
    """

    def __init__(self, scratch=SCRATCH):
        scratch = Path(scratch)
        self.meta = json.loads((scratch / "meta.json").read_text())
        ws = C.load_workspace(scratch / "outward" / "workspace.npz")
        s = ws["scalars"]

        self.grid = C.grid_shape(self.meta)                       # (Nx,Ny,Nz) interior, full res
        self.mod = int(s["modX"])
        assert int(s["modY"]) == self.mod == int(s["modZ"]), "anisotropic decimation not handled"
        self.dx_mm = float(self.meta["dX_m"]) * 1e3
        self.dxf_mm = self.dx_mm * self.mod                       # spacing of the recorded field
        self.c0_ms = float(s["c0"])
        self.bone_threshold = case.BONE_THRESHOLD

        # Time base, straight from the deck: the solver steps at dT = dx/c0*cfl and records
        # every modT-th step, so recorded frame k sits at t = k*modT*dT (up to the one-frame
        # convention, which cancels in any slope fitted below).
        self.dT_s = float(s["dX"]) / self.c0_ms * float(s["cfl"])
        self.modT = int(s["modT"])
        self.dt_frame_s = self.dT_s * self.modT

        self.target_vox = np.asarray(self.meta["dent_grid"], float)   # full-res interior voxels
        self.target_f = self.target_vox / self.mod                    # same point, field indices

        self.nf = tuple(len(range(0, n, self.mod)) for n in self.grid)
        self.lo = PAD // self.mod
        npad = tuple((n + 2 * PAD) // self.mod + ((n + 2 * PAD) % self.mod > 0) for n in self.grid)
        gm = scratch / "outward" / "genout_mod.dat"
        if not gm.exists():
            raise SystemExit(f"{gm} not found -- run `make_propagation_movie.py solve` first.")
        nbytes = gm.stat().st_size
        per = 4 * int(np.prod(npad))
        if nbytes % per:
            raise SystemExit(f"{gm} ({nbytes} B) is not a whole number of {npad} frames")
        self.nframes = nbytes // per
        self.vol = np.memmap(gm, dtype="<f4", mode="r", shape=(self.nframes,) + npad)
        self.gm_bytes = nbytes

        # Background: the posed sound-speed map the solver actually used, sampled on exactly
        # the recorder positions (voxels 0, mod, 2*mod, ...) so field and background align.
        c_file = self.meta.get("c_file", "c.f32")
        c = np.fromfile(scratch / c_file, dtype="<f4").reshape(*self.grid, order="F")
        self.c_full = c
        self.c_f = c[::self.mod, ::self.mod, ::self.mod]

    def t_s(self, k):
        """Elapsed physical time (s) of recorded frame ``k``."""
        return k * self.dt_frame_s

    def plane(self, axis, index):
        """All frames of one plane, ``(nframes, A, B)``, cropped to the interior grid.

        ``axis`` is 0/1/2 (sagittal/coronal/axial) and ``index`` is a FIELD index along it.
        """
        lo, nf = self.lo, self.nf
        j = lo + int(round(index))
        sl = [slice(None), slice(lo, lo + nf[0]), slice(lo, lo + nf[1]), slice(lo, lo + nf[2])]
        sl[axis + 1] = j
        return np.asarray(self.vol[tuple(sl)], dtype=np.float32)

    def mm_from_target(self, axis, n=None):
        """Coordinates (mm, signed, relative to the target) of the field samples on ``axis``."""
        n = self.nf[axis] if n is None else n
        return (np.arange(n) * self.mod - self.target_vox[axis]) * self.dx_mm

    def extent(self, a_axis, b_axis):
        """imshow ``extent`` (mm from target) for a panel spanning ``a_axis`` x ``b_axis``."""
        a = self.mm_from_target(a_axis)
        b = self.mm_from_target(b_axis)
        h = 0.5 * self.dxf_mm
        return (a[0] - h, a[-1] + h, b[0] - h, b[-1] + h)


# ============================================================================ physics

def verify(F=None, verbose=True):
    """Check that the wave does what a wave in this geometry must do, and say the numbers.

    Two independent checks, both reported with units:

      1. SPEED. Arrival times are read off along every direction in the sagittal plane whose
         path from the target is pure water, at radii from 1 mm out to the first bone. The
         arrival at a point is its first crossing of 20% of its own peak |p|. Pooling those
         (radius, arrival) pairs and fitting a straight line gives the propagation speed; the
         intercept absorbs the source pulse's own delay. It must come out at the water speed
         (1540 m/s), because that is the medium the fit is restricted to.

      2. FIRST BONE CONTACT. The distance from the target to the nearest bone voxel divided by
         the water speed, compared with the frame at which |p| on that voxel first rises. The
         geometric time must be corrected by the fit intercept from check 1 before comparing:
         the source is a finite transmit pulse, so "t = 0" is when it starts ramping, not when
         its front leaves the target.
    """
    F = F or Field()
    out = {}
    kx = int(round(F.target_f[0]))
    sag = F.plane(0, kx)                                   # (nframes, nfy, nfz)
    y_mm = F.mm_from_target(1)
    z_mm = F.mm_from_target(2)
    c_sag = F.c_f[kx]

    # ---- 1. speed along water-only rays in the sagittal plane
    radii = np.arange(1.0, 12.01, 0.25)                    # mm from the target
    rs, ts = [], []
    for ang in np.arange(0.0, 360.0, 5.0):
        dy, dz = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        for r in radii:
            iy = int(round((r * dy - y_mm[0]) / F.dxf_mm))
            iz = int(round((r * dz - z_mm[0]) / F.dxf_mm))
            if not (0 <= iy < len(y_mm) and 0 <= iz < len(z_mm)):
                break
            if c_sag[iy, iz] > F.bone_threshold:           # ray has reached bone: stop this ray
                break
            trace = np.abs(sag[:, iy, iz])
            pk = trace.max()
            if pk <= 0:
                continue
            k0 = int(np.argmax(trace > 0.2 * pk))
            rs.append(r)
            ts.append(F.t_s(k0))
    rs, ts = np.asarray(rs), np.asarray(ts)
    A = np.stack([rs * 1e-3, np.ones_like(rs)], 1)
    slope, icpt = np.linalg.lstsq(A, ts, rcond=None)[0]
    c_meas = 1.0 / slope
    out.update(c_measured_ms=c_meas, n_probes=len(rs), fit_intercept_us=icpt * 1e6)

    # ---- 2. distance to bone and the time the wave gets there
    bone = F.c_full > F.bone_threshold
    idx = np.argwhere(bone)
    d_mm = np.linalg.norm((idx - F.target_vox) * F.dx_mm, axis=1)
    d_min, d_med = float(d_mm.min()), float(np.median(d_mm))
    near = idx[int(np.argmin(d_mm))]
    kf = np.clip(np.round(near / F.mod).astype(int), 0, np.array(F.nf) - 1)
    trace = np.abs(F.plane(0, kf[0])[:, kf[1], kf[2]])
    k0 = int(np.argmax(trace > 0.2 * trace.max()))
    out.update(bone_min_mm=d_min, bone_median_mm=d_med, bone_max_mm=float(d_mm.max()),
               bone_expected_us=d_min / F.c0_ms * 1e3 + icpt * 1e6,
               bone_measured_us=F.t_s(k0) * 1e6,
               head_crossing_us=float(d_mm.max()) / F.c0_ms * 1e3)

    # ---- 3. span and dynamic range (on the three displayed planes)
    out.update(nframes=F.nframes, span_us=F.t_s(F.nframes - 1) * 1e6,
               dt_frame_us=F.dt_frame_s * 1e6)

    if verbose:
        print(f"  frames {F.nframes} @ {out['dt_frame_us']:.3f} us  ->  span "
              f"0 - {out['span_us']:.2f} us")
        print(f"  speed fit: {out['c_measured_ms']:.0f} m/s from {out['n_probes']} water probes "
              f"(deck c0 = {F.c0_ms:.0f} m/s, error {100 * (c_meas - F.c0_ms) / F.c0_ms:+.1f}%); "
              f"source turn-on {out['fit_intercept_us']:.2f} us")
        print(f"  target-to-bone: nearest {d_min:.2f} mm, median {d_med:.2f} mm, farthest "
              f"{out['bone_max_mm']:.2f} mm")
        print(f"  first bone contact: expected {out['bone_expected_us']:.2f} us "
              f"(= {d_min / F.c0_ms * 1e3:.2f} us of flight + turn-on), measured "
              f"{out['bone_measured_us']:.2f} us -- within "
              f"{abs(out['bone_measured_us'] - out['bone_expected_us']) / out['dt_frame_us']:.1f} frames")
        print(f"  the far side of the head is {out['bone_max_mm']:.1f} mm out, so the wave should "
              f"clear the skull by ~{out['head_crossing_us']:.1f} us of flight "
              f"(the run records {out['span_us']:.1f} us)")
    return out


# ============================================================================ rendering

def _window_vox():
    """Full-res voxel of the TIPS window centre from the shipped bundle, or None.

    Purely decorative -- the movie is about the wave -- so any failure to load the bundle
    (it may simply not have been built yet) is swallowed and the marker is dropped.
    """
    try:
        import skull_transparency as st
        bundle = st.load_bundle(case.OUT)
        tmap = st.compute_transparency_map(bundle)
        foot_mm, _ = case.footprint_radius_mm(case.TIPS, tmap)
        pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=case.TIPS.roc_mm,
                                                    bowl_radius_mm=foot_mm,
                                                    theta_max_deg=case.TIPS.acceptance_angle_deg))
        return np.asarray(pl.window_center_fullres_voxel, float)
    except Exception as exc:                       # noqa: BLE001 -- decoration only
        print(f"  (no window marker: {type(exc).__name__}: {exc})")
        return None


def _rgba(sl, vmax, cmap, gamma=GAMMA):
    """Signed pressure slice -> RGBA, compressed about zero and transparent where quiet."""
    u = np.clip(sl / vmax, -1.0, 1.0)
    s = np.sign(u) * np.abs(u) ** gamma                       # monotonic, signed, zero-preserving
    rgba = cmap(0.5 * (s + 1.0))
    rgba[..., 3] = np.clip((np.abs(s) - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0.0, 1.0) * ALPHA_MAX
    return rgba


#: The three anatomical panels: (title, fixed axis, in-plane axes, axis labels). The in-plane
#: arrays are transposed at draw time, so the SECOND in-plane axis is the vertical one.
PANELS = (
    ("sagittal", 0, (1, 2), "y  anterior + (mm)", "z  superior + (mm)"),
    ("coronal",  1, (0, 2), "x  right + (mm)",    "z  superior + (mm)"),
    ("axial",    2, (0, 1), "x  right + (mm)",    "y  anterior + (mm)"),
)


def _setup_axis(F, ax, name, fix_axis, ip, xl, yl, bg, win_vox):
    """Draw the static parts of one panel: skull, target cross, optional window marker."""
    ext = F.extent(ip[0], ip[1])
    ax.imshow(bg.T, cmap="gray", origin="lower", extent=ext, vmin=BG_VMIN, vmax=BG_VMAX,
              interpolation="bilinear", zorder=1)
    im = ax.imshow(np.zeros(bg.T.shape + (4,)), origin="lower", extent=ext,
                   interpolation="bilinear", zorder=2)
    # Bone outline on TOP of the field, so the skull stays readable in the frames where the
    # wave is bright enough to wash the greyscale out.
    ax.contour(np.linspace(ext[0], ext[1], bg.shape[0]), np.linspace(ext[2], ext[3], bg.shape[1]),
               bg.T, levels=[F.bone_threshold], colors="k", linewidths=0.45, alpha=0.65, zorder=3)
    ax.plot(0.0, 0.0, "+", color="#39ff5e", ms=11, mew=1.8, zorder=4)   # the cerebellum target
    if win_vox is not None:
        off = abs((win_vox[fix_axis] - F.target_vox[fix_axis]) * F.dx_mm)
        if off < 4.0:                                    # only mark it where it is really in-plane
            a = (win_vox[ip[0]] - F.target_vox[ip[0]]) * F.dx_mm
            b = (win_vox[ip[1]] - F.target_vox[ip[1]]) * F.dx_mm
            ax.plot(a, b, "o", mfc="none", mec="#ffd23f", ms=10, mew=1.8, zorder=4)
    ax.set_xlabel(xl, fontsize=8)
    ax.set_ylabel(yl, fontsize=8)
    ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=7)
    return im


def render(F=None, tstride=TSTRIDE, fps=FPS, sheet_only=False):
    """Render the movie, the GIF and the contact sheet; return a dict of what was written."""
    F = F or Field()
    kx, ky, kz = (int(round(v)) for v in F.target_f)
    print(f"  reading 3 planes through the target (field voxel {kx},{ky},{kz}) "
          f"from {F.gm_bytes / 1e9:.2f} GB ...")
    planes = {0: F.plane(0, kx), 1: F.plane(1, ky), 2: F.plane(2, kz)}
    bgs = {0: F.c_f[kx], 1: F.c_f[:, ky], 2: F.c_f[:, :, kz]}

    # One global colour limit for the whole movie: a high percentile of |p| over every
    # displayed pixel and frame, with the immediate neighbourhood of the point source cut out
    # (there |p| is a 1/r singularity that would otherwise set the scale for all 20 mm).
    vals = []
    for ax_i, (ip0, ip1) in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
        a = F.mm_from_target(ip0)[:, None]
        b = F.mm_from_target(ip1)[None, :]
        far = (a ** 2 + b ** 2) > NEAR_SOURCE_MM ** 2
        vals.append(np.abs(planes[ax_i][:, far]).ravel())
    allv = np.concatenate(vals)
    vmax = float(np.percentile(allv, PCTL))
    pmax = float(max(np.abs(planes[i]).max() for i in planes))
    print(f"  |p| on the displayed planes: max {pmax:.3g} Pa, p{PCTL} (>{NEAR_SOURCE_MM} mm "
          f"from source) {vmax:.3g} Pa  ->  dynamic range {pmax / vmax:.0f}x, "
          f"compressed with gamma = {GAMMA}")

    win_vox = _window_vox()
    cmap = plt.get_cmap("RdBu_r")
    frames = list(range(0, F.nframes, tstride))

    # ---- contact sheet: SHEET_N sagittal frames, spaced so the early, fast part is dense
    picks = np.unique(np.round(np.linspace(0.0, 1.0, SHEET_N) ** 1.5 * (F.nframes - 1)).astype(int))
    figs = plt.figure(figsize=(11.0, 8.6))
    for n, k in enumerate(picks):
        ax = figs.add_subplot(3, 3, n + 1)
        im = _setup_axis(F, ax, "", 0, (1, 2), "y  anterior + (mm)", "z  superior + (mm)",
                         bgs[0], win_vox)
        im.set_data(_rgba(planes[0][k].T, vmax, cmap))
        ax.set_title(f"t = {F.t_s(k) * 1e6:.2f} us", fontsize=9)
        if n % 3:
            ax.set_ylabel("")
        if n < 6:
            ax.set_xlabel("")
    figs.suptitle("Mouse, TIPS 1 MHz: outward time-reversal wave from the cerebellum "
                  "(sagittal through the target)", fontsize=11)
    figs.tight_layout(rect=(0, 0, 1, 0.96))
    figs.savefig(SHEET, dpi=92, pil_kwargs={"optimize": True})
    plt.close(figs)
    print(f"  contact sheet -> {SHEET}  ({SHEET.stat().st_size / 1e6:.2f} MB, "
          f"frames {list(picks)})")
    if sheet_only:
        return {"sheet": SHEET}

    # ---- the movie: three panels side by side, one global scale, time in the suptitle
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.0))
    ims = [_setup_axis(F, axs[i], nm, fx, ip, xl, yl, bgs[fx], win_vox)
           for i, (nm, fx, ip, xl, yl) in enumerate(PANELS)]
    sup = fig.suptitle("", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    import imageio.v2 as imageio
    writers = []
    try:
        writers.append(("mp4", imageio.get_writer(MP4, fps=fps, codec="libx264", quality=7,
                                                  macro_block_size=None)))
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (no mp4 encoder: {type(exc).__name__}: {exc}; PNG frames + sheet only)")
    gif_frames = []

    for k in frames:
        for i, (nm, fx, ip, xl, yl) in enumerate(PANELS):
            ims[i].set_data(_rgba(planes[fx][k].T, vmax, cmap))
        sup.set_text(f"Mouse, TIPS 1 MHz -- outward wave from the cerebellum      "
                     f"t = {F.t_s(k) * 1e6:6.2f} us")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        for _, w in writers:
            w.append_data(buf)
        if len(gif_frames) < 400:                # the GIF is a preview: sagittal third only
            gif_frames.append(buf[::2, : buf.shape[1] // 3 : 2].copy())
    for _, w in writers:
        w.close()
    plt.close(fig)

    made = {"sheet": SHEET}
    if writers:
        made["mp4"] = MP4
        print(f"  movie -> {MP4}  ({MP4.stat().st_size / 1e6:.2f} MB, {len(frames)} frames "
              f"@ {fps} fps, stride {tstride})")
    imageio.mimsave(GIF, gif_frames, duration=1.0 / fps, loop=0)
    made["gif"] = GIF
    print(f"  gif   -> {GIF}  ({GIF.stat().st_size / 1e6:.2f} MB)")
    return made


# ============================================================================ driver

def drop_dump():
    """Delete the multi-GB field dump once the frames exist (set KEEP_DUMP=1 to keep it)."""
    gm = SCRATCH / "outward" / "genout_mod.dat"
    if gm.exists() and not os.environ.get("KEEP_DUMP"):
        sz = gm.stat().st_size / 1e9
        gm.unlink()
        print(f"  removed {gm} ({sz:.2f} GB)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "clean":
        shutil.rmtree(SCRATCH, ignore_errors=True)
        print(f"removed {SCRATCH}")
        return
    if cmd in ("solve", "all"):
        solve()
    if cmd in ("verify", "sheet", "render", "all"):
        F = Field()
        verify(F)
        if cmd != "verify":
            render(F, sheet_only=(cmd == "sheet"))
    if cmd == "all":
        drop_dump()


if __name__ == "__main__":
    main()
