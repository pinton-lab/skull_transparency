#!/usr/bin/env python3
"""Interactive transducer positioning tool for a Field Bundle (napari 3-D + docked ortho slices).

A port of ``runs/rebuild_6ppw_graded/ctx500_position_tool.py`` (the human CTX-500 tool,
manuscript Appendix A) onto the **Field Bundle**, so it runs on the Saimiri brain-center map
-- or any other bundle -- instead of the hard-wired Halle graded medium.

Manually pose a focused bowl relative to the skull and the bundle's target. The face-centre
rides on a sphere centred on the target; by default the bowl points straight at the target
(geometric focus on target when the standoff == ROC). Optional tilt/yaw aim off-target.

VISUALS
  * 3-D napari canvas: skull surface coloured by the PLACEMENT-OBJECTIVE field (the same
    sqrt(J) moving-window map the position optimizer maximises) + cyan dish + green target +
    magenta geometric focus + yellow beam axis.
  * docked matplotlib panel: three orthogonal slices through the target (sagittal/coronal/
    axial) with the live cap footprint, target (+) and apex (x) overlaid.
  * live readout: pose, focus->target offset, cap clearance, and the PLACEMENT SCORE at the
    current window with its percentage of the global placement-score peak.

KEYBOARD
  arrows  Up/Down  -> move apex toward Superior / Inferior on the sphere (elevation)
          Left/Right -> move apex toward anatomical Left / Right (azimuth)
  . / ,   radius OUTWARD / INWARD (standoff from target, mm)
  t / g   tilt aim +/-  (off-target, about the beam's right axis)
  y / h   yaw  aim +/-  (off-target, about the beam's up axis)
  1..9    switch bundle/target (when several --bundle are given)
  [ / ]   smaller / larger angular & radial step
  r       reset pose for the current target
  e       export pose + full-density cap (.npy grid-voxel coords + .json) for the TR pipeline
  ?       print this help to the console

WHAT CHANGED vs THE HUMAN TOOL (every hard-wired human assumption is now derived)
  * frame      -- the bundle's ``Registration`` (world mm <-> grid voxel), not tuba.species.human
                  + ``crop_lo_ds``. The bundle's own ``world_frame`` label is carried through.
  * anatomy    -- S/A/L unit axes are DERIVED from ``R_mni_to_sim``. This matters: the Halle
                  domain has anatomical left at ``+axis0``, the Saimiri grid at ``-axis0``.
                  Copying the constants would have mirrored every azimuth.
  * medium     -- ``bundle/skull_fullres_c.npy`` (49 MB), not the 1.38 GB graded medium.
  * bone       -- the bundle's ``physics.bone_threshold`` (Saimiri 1700 m/s), not 1600.
  * objective  -- built from the bundle's own ``TransparencyMap`` (surf_vox/Pmax/rhat), not a
                  per-target ``surf_intensity.npz`` on disk. Same math as the human tool:
                  E = Pmax^2, w_inc = clip(cos,0,1)^2, J(centre) = sum over the legal footprint,
                  value = sqrt(J), PCA surface normals (a binary-mask gradient staircases).
  * objective  -- the moving window is the CONE footprint ``r*sin(half_angle)`` and only
                  ACCESSIBLE patches are legal (``skull_transparency.access``: foramen,
                  neck/pharynx cones, multi-layer beam paths, dish clearance), so the
                  glow field matches what the report's placement would choose.
  * bowl       -- ``--roc-mm`` / ``--aperture-mm`` (Saimiri default 35 / 30 mm, half-angle
                  25.4 deg), not the CTX-500 63.2 / 64 mm. Standoff limits scale with ROC.
  * readout    -- the live line now reports EVERY bone layer the beam crosses, not just
                  the first. On the vault the first bone is the window; off-vault on a
                  small skull the mandible/zygoma can precede it, and the score (read at
                  the first layer) would otherwise look inexplicably low.
  * targets    -- one per bundle; pass several ``--bundle`` to bind them to keys 1..9.

Usage:
  python saimiri_position_tool.py                          # interactive (needs DISPLAY)
  python saimiri_position_tool.py --build-cache            # mesh/bone cache, then exit
  python saimiri_position_tool.py --build-objfield         # placement-objective field, then exit
  python saimiri_position_tool.py --selftest               # headless geometry checks (no GUI)
  python saimiri_position_tool.py --smoke                  # GUI integration test, synthetic skull
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace

import numpy as np

os.environ.setdefault("QT_API", "pyqt6")
sys.path.insert(0, "/celerina/gfp/mfs/skull_transparency/src")

import skull_transparency as st                       # noqa: E402
from skull_transparency import transducer as tx       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "bundle")

# ----- bowl: small-NHP 1 MHz default (the Saimiri spec in build_saimiri_braincenter.py) -----
DEF_ROC_MM = 35.0
DEF_APERTURE_MM = 30.0

# placement-objective parameters (identical to the human tool)
THETA_MAX_DEG = 35.0                # incidence-legality cut
N_CAND = 9000                       # subsampled candidate window centres
CLIMLO_PCT, CLIMHI_PCT = 8.0, 99.0  # percentile clim on the normalised objective
SURF_SNAP_MM = 2.5                  # mesh vertex counts as "on the map" within this of a patch
GLOW_COLS = [[0.10, 0.10, 0.13], [0.42, 0.18, 0.30], [0.86, 0.27, 0.10],
             [1.0, 0.66, 0.10], [1.0, 0.97, 0.82]]
GLOW_CTRL = [0.0, 0.20, 0.48, 0.74, 1.0]

CONTROLS_LEGEND = (
    "TRANSDUCER POSITIONING - CONTROLS\n"
    "---------------------------------\n"
    "  Up / Down   move  up / down  (Superior / Inferior)\n"
    "  Left/Right  move  left / right (azimuth)\n"
    "  . / ,       radius  outward / inward\n"
    "  t / g       tilt  + / -   (aim off-target)\n"
    "  y / h       yaw   + / -   (aim off-target)\n"
    "  1 .. 9      switch bundle / target\n"
    "  [ / ]       step size  - / +\n"
    "  r           reset pose (re-seed current target)\n"
    "  e           export pose + full-density cap\n"
    "  ?           print this legend to the console\n"
    "\n"
    "skull colour = placement-objective field\n"
    "green=target  magenta=geom focus  cyan=apex/dish")


# ============================================================================ frame
class Frame:
    """World mm <-> grid voxel for ONE Field Bundle, plus its derived anatomical axes.

    Replaces the human tool's ``Frame`` (tuba.species.human ``mni2ds`` minus ``crop_lo_ds``).
    The bundle already carries a rigid ``Registration``, so the transform is exact and the
    world-frame LABEL travels with it -- no MNI assumption anywhere."""

    def __init__(self, bundle_dir: str, name: str | None = None):
        self.dir = os.path.abspath(bundle_dir)
        self.bundle = st.load_bundle(self.dir)
        self.reg = self.bundle.registration
        if self.reg is None:
            raise ValueError(f"{self.dir} has no registration.json; the tool needs the world<->grid map")
        self.dx_mm = float(self.reg.dx_mm)
        self.world_frame = getattr(self.reg, "world_frame", "mni_ras_mm")
        self.bone_c = float(self.bundle.physics.get("bone_threshold", 2200.0))
        self.shape = tuple(int(s) for s in self.bundle.skull_c().shape)
        self.name = name or str(self.bundle.target.get("name") or os.path.basename(self.dir))
        self.target_world = np.asarray(self.reg.target_mni_mm, float)

        # anatomical unit axes IN THE GRID FRAME, derived from the registration rotation.
        # World is RAS (+x Right, +y Anterior, +z Superior); R maps a world-mm displacement
        # to a grid-mm displacement, so its columns are the grid-frame anatomical directions.
        R = np.asarray(self.reg.R_mni_to_sim, float)
        self.r_hat = R @ np.array([1.0, 0.0, 0.0])
        self.a_hat = R @ np.array([0.0, 1.0, 0.0])
        self.s_hat = R @ np.array([0.0, 0.0, 1.0])
        self.l_hat = -self.r_hat                      # anatomical LEFT (the tool's azimuth +90)

    @property
    def axes(self) -> dict:
        return dict(s_hat=self.s_hat, a_hat=self.a_hat, l_hat=self.l_hat)

    def world2vox(self, world_mm) -> np.ndarray:
        return np.asarray(self.reg.mni_to_fullres(np.asarray(world_mm, float)), float)

    def vox2world(self, vox) -> np.ndarray:
        return np.asarray(self.reg.fullres_to_mni(np.asarray(vox, float)), float)

    @property
    def target_vox(self) -> np.ndarray:
        return np.asarray(self.reg.target_fullres_voxel, float)

    # ---- per-bundle artefact paths ----
    def cache_path(self, mod: int) -> str:
        return os.path.join(self.dir, f".position_tool_cache_mod{mod}.npz")

    def objfield_path(self, mod: int) -> str:
        return os.path.join(self.dir, f".position_tool_objfield_mod{mod}.npz")

    @property
    def export_dir(self) -> str:
        return os.path.join(os.path.dirname(self.dir), "manual_caps")


@dataclass(frozen=True)
class Bowl:
    """The focused bowl being positioned (the human tool's fixed CTX-500 constants, freed)."""
    roc_mm: float = DEF_ROC_MM
    aperture_mm: float = DEF_APERTURE_MM

    @property
    def half_angle_deg(self) -> float:
        return float(np.degrees(np.arcsin((self.aperture_mm / 2.0) / self.roc_mm)))

    def footprint_radius_mm(self, tmap=None, pctile: float = 30.0) -> float:
        """Moving-window radius of the placement objective, ``r*sin(half_angle)``.

        NOT the aperture radius. That default is right for a human (the skull sits about a
        focal length from the target) and ~2x too big here, which puts a fifth of the skull
        inside every candidate footprint and flattens the window search. Same correction the
        mouse TIPS case needed. Falls back to the aperture radius with no map to measure."""
        if tmap is None:
            return self.aperture_mm / 2.0
        r = float(np.percentile(np.asarray(tmap.rad_mm, float), pctile))
        return r * float(np.sin(np.radians(self.half_angle_deg)))

    @property
    def radius_limits_mm(self) -> tuple:
        """Standoff clamps, scaled from ROC (the human tool hard-coded 25..130 mm for ROC 63.2)."""
        return 0.40 * self.roc_mm, 2.06 * self.roc_mm


# ============================================================================ volume cache
def build_cache(frame: Frame, mod: int = 1) -> dict:
    """Bone mask + marching-cubes skull mesh from the bundle's own sound-speed volume.

    The human tool cached to avoid re-reading a 1.38 GB medium off the RAID; a bundle's
    ``skull_fullres_c.npy`` is 49 MB, so this is now just a marching-cubes cache."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    c = np.asarray(frame.bundle.skull_c())
    bone_ds = (c[::mod, ::mod, ::mod] > frame.bone_c)
    del c
    print(f"[cache] bone voxels (ds mod={mod}, c>{frame.bone_c:.0f}): {int(bone_ds.sum())}; "
          f"building mesh ...", flush=True)
    sm = gaussian_filter(bone_ds.astype(np.float32), 0.7)
    verts, faces, _, _ = marching_cubes(sm, level=0.5, step_size=1)
    print(f"[cache] mesh {len(verts)} verts / {len(faces)} faces", flush=True)
    out = dict(verts=verts.astype(np.float32), faces=faces.astype(np.int32),
               bone_ds=bone_ds.astype(np.uint8), mod=np.int32(mod),
               N=np.asarray(frame.shape, np.int32))
    np.savez_compressed(frame.cache_path(mod), **out)
    print(f"[cache] wrote {frame.cache_path(mod)}", flush=True)
    return out


def load_cache(frame: Frame, mod: int = 1, allow_build: bool = True) -> dict:
    p = frame.cache_path(mod)
    if os.path.exists(p):
        print(f"[cache] loading {p}", flush=True)
        z = np.load(p)
        return dict(verts=z["verts"], faces=z["faces"], bone_ds=z["bone_ds"],
                    mod=int(z["mod"]), N=z["N"])
    if not allow_build:
        raise FileNotFoundError(f"no cache at {p}; run with --build-cache first")
    return build_cache(frame, mod)


# ============================================================================ objective field
def pca_normals(sv: np.ndarray, k: int = 40, chunk: int = 300000) -> np.ndarray:
    """Un-oriented surface normals at each point of the dense cloud by local PCA: the normal is
    the smallest-eigenvalue eigenvector of the kNN covariance. Averaging a plane over k
    neighbours removes the grid quantization, so the incidence cosine is smooth (no staircase)
    -- unlike a gradient of the binary bone mask."""
    from scipy.spatial import cKDTree
    tree = cKDTree(sv)
    M = len(sv)
    n = np.empty((M, 3))
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        _, idx = tree.query(sv[s:e], k=k, workers=-1)
        X = sv[idx]
        X = X - X.mean(1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", X, X)
        _, V = np.linalg.eigh(cov)
        n[s:e] = V[:, :, 0]
    return n


def build_objfield(frame: Frame, bowl: Bowl, mod: int = 1, neck_cone_deg: float = 45.0):
    """Precompute the placement-objective field on the cached mesh verts for this bundle.

    Faithful to ``render_position_optimizer_3cases.optimizer_field`` / the human tool: per-patch
    energy E = Pmax^2, incidence weight w_inc = clip(cos,0,1)^2, window score
    J(centre) = sum over the footprint of legal patches of E*w_inc, placement value sqrt(J);
    interpolated to the mesh vertices. The only change is the SOURCE of (surf_vox, Pmax): the
    bundle's own TransparencyMap rather than a per-target surf_intensity.npz."""
    from scipy.spatial import cKDTree
    cache = load_cache(frame, mod, allow_build=False)
    verts, faces = cache["verts"].astype(float), cache["faces"]

    tmap = st.compute_transparency_map(
        frame.bundle, options=st.TransparencyOptions(bone_threshold=frame.bone_c))
    foot = bowl.footprint_radius_mm(tmap)
    access, acc = st.access_mask(tmap, frame.bundle, standoff_mm=bowl.roc_mm,
                                 neck_cone_deg=neck_cone_deg, cap_roc_mm=bowl.roc_mm,
                                 cap_aperture_mm=bowl.aperture_mm)
    print(f"  footprint radius {foot:.1f} mm (cone-derived, not the "
          f"{bowl.aperture_mm/2:.0f} mm aperture radius)")
    print("  " + acc.summary())
    sv = np.asarray(tmap.surf_vox, float)
    Pmax = np.asarray(tmap.Pmax, float)
    rhat = np.asarray(tmap.rhat, float)
    surf_world = frame.vox2world(sv)

    print(f"[objfield] PCA surface normals ({len(sv)} patches) ...", flush=True)
    nrm_un = pca_normals(sv)
    nrm = nrm_un * np.sign(np.sum(nrm_un * rhat, axis=1))[:, None]     # orient away from target
    cos_inc = np.clip(np.sum(rhat * nrm, axis=1), -1.0, 1.0)
    Tw = (Pmax ** 2) * np.clip(cos_inc, 0.0, 1.0) ** 2
    # incidence legality AND physical accessibility (foramen / neck / mandible /
    # dish clearance) -- the same mask the report's placement uses, so the glow field
    # the user steers by cannot recommend a window the report would reject.
    legal = (cos_inc >= np.cos(np.deg2rad(THETA_MAX_DEG))) & access
    legal_idx = np.where(legal)[0]
    if len(legal_idx) == 0:
        raise ValueError(f"no patch is both within {THETA_MAX_DEG} deg incidence and "
                         f"accessible; check bone_threshold / loosen neck_cone_deg")

    tree = cKDTree(surf_world)
    rng = np.random.default_rng(0)
    if len(legal_idx) > N_CAND:                      # bias candidates to high-Tw + uniform spread
        ntop = N_CAND // 2
        top = legal_idx[np.argsort(-Tw[legal_idx])[:ntop]]
        rest = rng.choice(np.setdiff1d(legal_idx, top), N_CAND - ntop, replace=False)
        cand = np.concatenate([top, rest])
    else:
        cand = legal_idx
    nbrs = tree.query_ball_point(surf_world[cand], foot, workers=-1)
    sqrtJ = np.sqrt(np.maximum(np.array([Tw[n].sum() for n in nbrs]), 0.0))
    peak = float(sqrtJ.max())
    ctree = cKDTree(surf_world[cand]); dist, idx = ctree.query(surf_world, k=6, workers=-1)
    wgt = 1.0 / (dist + 1e-6); wgt /= wgt.sum(1, keepdims=True)
    field = np.where(legal, (wgt * sqrtJ[idx]).sum(1), 0.0)
    fn = field / (field.max() + 1e-30)
    clo = float(np.percentile(fn[fn > 0], CLIMLO_PCT)); chi = float(np.percentile(fn[fn > 0], CLIMHI_PCT))
    if chi <= clo:
        chi = clo + 1e-9

    # interpolate (normalised colour field + raw score) to the mesh vertices (IDW, k=8)
    ptree = cKDTree(sv / mod); vdist, vidx = ptree.query(verts, k=8, workers=-1)
    vw = 1.0 / (vdist + 1e-6); vw /= vw.sum(1, keepdims=True)
    vfield = (vw * fn[vidx]).sum(1)
    vscore = (vw * field[vidx]).sum(1).astype(np.float32)
    snap_ds = SURF_SNAP_MM / (frame.dx_mm * mod)                 # mm -> downsampled voxels
    vgood = (vdist.min(1) < snap_ds) & ((fn[vidx] > 0).any(1))
    cohf = vgood[faces].all(axis=1)
    vv = np.clip(np.where(vgood, vfield, clo), clo, chi).astype(np.float32)

    # seed direction for the default pose = the argmax of the objective (the optimizer's window)
    best = int(np.argmax(field))
    seed_dir = (sv[best] - frame.target_vox)
    seed_dir = seed_dir / (np.linalg.norm(seed_dir) or 1.0)

    np.savez_compressed(frame.objfield_path(mod), vv=vv, cohf=cohf, vscore=vscore, vgood=vgood,
                        clo=clo, chi=chi, peak=peak, seed_dir=seed_dir.astype(np.float32),
                        footprint_mm=np.float64(foot), n_access=np.int64(acc.n_legal),
                        n_verts=np.int64(len(verts)), n_faces=np.int64(len(faces)))
    print(f"[objfield] {frame.name}: legal {int(legal.sum())} patches, {int(cohf.sum())} overlay "
          f"faces, peak score {peak:.4g}, clim {clo:.3g}..{chi:.3g}, seed window "
          f"{np.round(frame.vox2world(sv[best]), 2)} mm -> {frame.objfield_path(mod)}", flush=True)


def load_objfield(frame: Frame, mod: int):
    p = frame.objfield_path(mod)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return dict(vv=z["vv"], cohf=z["cohf"], vscore=z["vscore"], vgood=z["vgood"],
                clo=float(z["clo"]), chi=float(z["chi"]), peak=float(z["peak"]),
                seed_dir=z["seed_dir"].astype(float),
                n_verts=int(z["n_verts"]), n_faces=int(z["n_faces"]))


# ============================================================================ pose -> geometry
def seed_pose(frame: Frame, bowl: Bowl, obj=None) -> tx.CapPose:
    """Default pose: apex along the objective-field peak direction (i.e. the placement
    optimizer's own window), standoff = ROC so the geometric focus sits on the target.
    Falls back to straight-superior when no objective cache exists yet."""
    d = obj["seed_dir"] if obj is not None else frame.s_hat
    az, el = tx.anatomical_az_el(d, **frame.axes)
    return tx.CapPose(az_deg=az, el_deg=el, radius_mm=bowl.roc_mm)


def cap_for_pose(pose: tx.CapPose, frame: Frame, bowl: Bowl, density: float):
    """Cap point cloud (grid voxel) + geometric focus + apex + aim for a pose."""
    return tx.build_cap_pose(frame.target_vox, pose, bowl.roc_mm / frame.dx_mm, frame.dx_mm,
                             half_angle_deg=bowl.half_angle_deg, density=density, **frame.axes)


def beam_bone_profile(apex, aim, bone_ds, mod, dx_mm, max_mm):
    """Every bone layer the beam axis crosses between the apex and the target.

    The human tool only ever needed the FIRST bone hit, because a CTX-500 on the vault meets
    the acoustic window first. On a small skull an off-vault approach can cross the mandible,
    zygomatic arch and tympanic region before the calvarial window -- so the first hit is not
    the window, and a score read there is honest but easy to misread. Returning the whole
    stack makes the obstruction visible. Returns (n_layers, total_bone_mm, first_entry_mm)."""
    p = np.asarray(apex, float) / mod
    step = np.asarray(aim, float) / mod
    shp = np.array(bone_ds.shape)
    n_steps = int(max_mm / dx_mm) + 1
    inside = np.zeros(n_steps, bool)
    for i in range(n_steps):
        q = np.round(p).astype(int)
        if np.all((q >= 0) & (q < shp)) and bone_ds[q[0], q[1], q[2]]:
            inside[i] = True
        p = p + step
    if not inside.any():
        return 0, 0.0, float("nan")
    edges = np.diff(inside.astype(np.int8))
    n_layers = int((edges == 1).sum()) + int(inside[0])
    return n_layers, float(inside.sum() * dx_mm), float(int(np.argmax(inside)) * dx_mm)


def clearance(cap_vox: np.ndarray, bone_ds: np.ndarray, mod: int):
    """How many cap points sit inside bone / outside the grid (positioning aid)."""
    q = np.round(cap_vox / mod).astype(int)
    shp = np.array(bone_ds.shape)
    ib = np.all((q >= 0) & (q < shp), axis=1)
    n_oob = int((~ib).sum())
    qq = q[ib]
    n_bone = int(bone_ds[qq[:, 0], qq[:, 1], qq[:, 2]].sum()) if ib.any() else 0
    return n_bone, n_oob


# ============================================================================ ortho-slice widget
def slice_planes(frame: Frame):
    """(title, horiz-axis, vert-axis, slice-axis, +h,-h,+v,-v) for the three ortho panels.

    DERIVED from the frame's anatomical axes instead of hard-coded to the Halle axis_map, so
    the panel titles and the L/R/A/P/S/I corner labels stay correct on any subject."""
    def grid_axis(v):
        k = int(np.argmax(np.abs(v)))
        return k, (1.0 if v[k] >= 0 else -1.0)

    (kr, sr), (ka, sa), (ks, ss) = (grid_axis(frame.r_hat), grid_axis(frame.a_hat),
                                    grid_axis(frame.s_hat))
    lab = lambda sgn, pos, neg: (pos, neg) if sgn > 0 else (neg, pos)
    rp, rn = lab(sr, "R", "L")          # +/- along the grid axis carrying Right
    ap, an = lab(sa, "A", "P")
    sp, sn = lab(ss, "S", "I")
    return [("sagittal  (A-S @ target)", ka, ks, kr, ap, an, sp, sn),
            ("coronal   (R-S @ target)", kr, ks, ka, rp, rn, sp, sn),
            ("axial     (R-A @ target)", kr, ka, ks, rp, rn, ap, an)]


class SliceView:
    """Three orthogonal bone slices through the target with the live cap overlaid (matplotlib)."""

    def __init__(self, bone_ds, mod, dx_mm, planes):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.bone_ds = bone_ds
        self.mod = mod
        self.dx_mm = dx_mm
        self.planes = planes
        self.min_halfwidth_ds = 40.0 / (dx_mm * mod)          # always show >= 40 mm across
        self.fig = Figure(figsize=(11, 3.7), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.axes, self.bg, self.cap, self.tgt, self.apex, self.beam = [], [], [], [], [], []
        gs = self.fig.add_gridspec(1, 3, wspace=0.06, left=0.02, right=0.99, top=0.88, bottom=0.04)
        for ci, (title, ah, av, asl, *_lab) in enumerate(planes):
            ax = self.fig.add_subplot(gs[0, ci]); ax.set_facecolor("black")
            im = ax.imshow(np.zeros((2, 2)), origin="lower", cmap="bone", vmin=0, vmax=1,
                           aspect="equal", interpolation="nearest")
            cap = ax.scatter([], [], s=3, c="#16c6e6", edgecolor="none", zorder=4)
            tgt, = ax.plot([], [], "+", color="lime", ms=15, mew=2.6, zorder=6)
            apx, = ax.plot([], [], "x", color="orange", ms=9, mew=2.2, zorder=6)
            beam, = ax.plot([], [], "-", color="yellow", lw=1.2, alpha=0.8, zorder=5)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=6, colors="0.4")
            self._corner_labels(ax, _lab)
            self.axes.append(ax); self.bg.append(im); self.cap.append(cap)
            self.tgt.append(tgt); self.apex.append(apx); self.beam.append(beam)

    @staticmethod
    def _corner_labels(ax, lab):
        lhp, lhn, lvp, lvn = lab
        tb = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7)
        for xx, yy, ha, va, lb in [(0.98, 0.5, "right", "center", lhp), (0.02, 0.5, "left", "center", lhn),
                                   (0.5, 0.98, "center", "top", lvp), (0.5, 0.02, "center", "bottom", lvn)]:
            ax.text(xx, yy, lb, transform=ax.transAxes, color="0.15", fontsize=8, fontweight="bold",
                    ha=ha, va=va, bbox=tb, zorder=7)

    def set_target(self, target_vox):
        """Re-slice the bone volume through the (new) target and re-frame each panel."""
        mod = self.mod
        for ci, (title, ah, av, asl, *_lab) in enumerate(self.planes):
            sl = int(round(target_vox[asl] / mod))
            sl = min(max(sl, 0), self.bone_ds.shape[asl] - 1)
            img = np.take(self.bone_ds, sl, axis=asl)
            rem = [a for a in (0, 1, 2) if a != asl]
            disp = img.T if rem == [ah, av] else img
            self.bg[ci].set_data(disp.astype(float))
            self.bg[ci].set_extent([0, self.bone_ds.shape[ah], 0, self.bone_ds.shape[av]])

    def update(self, cap_vox, target_vox, apex_vox):
        """Overlay the FULL transducer (all cap points projected onto each plane) and auto-frame
        each panel to include the target, the cap and the apex."""
        mod = self.mod
        for ci, (title, ah, av, asl, *_lab) in enumerate(self.planes):
            ch, cv = cap_vox[:, ah] / mod, cap_vox[:, av] / mod
            th, tv = target_vox[ah] / mod, target_vox[av] / mod
            axh, axv = apex_vox[ah] / mod, apex_vox[av] / mod
            self.cap[ci].set_offsets(np.c_[ch, cv])
            self.tgt[ci].set_data([th], [tv])
            self.apex[ci].set_data([axh], [axv])
            self.beam[ci].set_data([th, axh], [tv, axv])
            xs = np.concatenate([ch, [th, axh]]); ys = np.concatenate([cv, [tv, axv]])
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            hw = max((xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2) * 1.12 + 10.0
            hw = max(hw, self.min_halfwidth_ds)
            self.axes[ci].set_xlim(cx - hw, cx + hw); self.axes[ci].set_ylim(cy - hw, cy + hw)
        self.canvas.draw_idle()


# ============================================================================ export
def export_pose(pose: tx.CapPose, frame: Frame, bowl: Bowl, score=None, pct=None):
    os.makedirs(frame.export_dir, exist_ok=True)
    pts, focus, apex, aim = cap_for_pose(pose, frame, bowl, density=1.0)    # full grid density
    cap = np.unique(np.round(pts).astype(int), axis=0).astype(float)
    stem = os.path.join(frame.export_dir, f"cap_{frame.name}_manual")
    npy = stem + ".npy"
    np.save(npy, cap)
    meta = dict(target=frame.name, bundle=frame.dir, frame=frame.world_frame,
                target_world_mm=frame.target_world.tolist(), target_vox=frame.target_vox.tolist(),
                az_deg=pose.az_deg, el_deg=pose.el_deg, radius_mm=pose.radius_mm,
                tilt_deg=pose.tilt_deg, yaw_deg=pose.yaw_deg,
                roc_mm=bowl.roc_mm, aperture_mm=bowl.aperture_mm,
                half_angle_deg=bowl.half_angle_deg,
                apex_vox=apex.tolist(), apex_world_mm=frame.vox2world(apex).tolist(),
                aim_hat=aim.tolist(), geom_focus_vox=focus.tolist(),
                geom_focus_to_target_mm=float(np.linalg.norm(focus - frame.target_vox) * frame.dx_mm),
                placement_score=(None if score is None else float(score)),
                placement_pct_of_peak=(None if pct is None else float(pct)),
                n_cap=int(len(cap)), cap_npy=npy)
    jsn = stem + "_pose.json"
    json.dump(meta, open(jsn, "w"), indent=1)
    print(f"[export] {len(cap)} cap pts -> {npy}\n[export] pose -> {jsn} "
          f"(focus-to-target {meta['geom_focus_to_target_mm']:.1f} mm)", flush=True)
    return npy, jsn


# ============================================================================ interactive scene
def build_scene(frames, bowl: Bowl, mod: int, caches=None):
    """Build the napari 3-D scene + docked ortho slices + key bindings for a list of Frames.
    Returns (viewer, state, refresh) WITHOUT starting the event loop (so smoke can reuse it)."""
    import napari
    from napari.utils import Colormap
    from scipy.spatial import cKDTree

    caches = caches or {}

    def get_cache(fr):
        if fr.dir not in caches:
            caches[fr.dir] = load_cache(fr, mod)
        return caches[fr.dir]

    def load_obj_valid(fr, cache):
        """Load the objective cache, but only if it matches this mesh (guards the smoke mesh)."""
        o = load_objfield(fr, mod)
        if o is not None and (o["n_verts"] != len(cache["verts"])
                              or o["n_faces"] != len(cache["faces"])):
            return None
        return o

    fr0 = frames[0]
    cache0 = get_cache(fr0)
    obj0 = load_obj_valid(fr0, cache0)
    state = {"frame": fr0, "cache": cache0, "obj": obj0, "step": 5.0, "density": 0.12,
             "pose": seed_pose(fr0, bowl, obj0), "tree": cKDTree(cache0["verts"])}

    def cap_disp():
        return cap_for_pose(state["pose"], state["frame"], bowl, density=state["density"])

    def window_score():
        """Placement score at the CURRENT window = objective field at the surface patch where
        the beam axis meets the outer skull. Returns (score, pct_of_global_peak)."""
        obj, fr = state["obj"], state["frame"]
        if obj is None:
            return None, None
        bone_ds = state["cache"]["bone_ds"].astype(bool)
        _, _, apex, aim = cap_disp()
        p = apex / mod
        step = aim / mod                                     # one full-res voxel per step
        shp = np.array(bone_ds.shape)
        n_steps = int((state["pose"].radius_mm + 20.0) / fr.dx_mm) + 1
        for _ in range(n_steps):
            q = np.round(p).astype(int)
            if np.all((q >= 0) & (q < shp)) and bone_ds[q[0], q[1], q[2]]:
                break
            p = p + step
        else:
            return 0.0, 0.0
        vi = int(state["tree"].query(p)[1])
        sc = float(obj["vscore"][vi])
        return sc, 100.0 * sc / (obj["peak"] + 1e-30)

    # ---- napari scene (everything in DOWNSAMPLED-voxel coords = verts frame) ----
    viewer = napari.Viewer(ndisplay=3, title=f"transducer positioning - {fr0.name}")
    flat = lambda rgb: Colormap([rgb + [1.0], rgb + [1.0]])
    glow = Colormap(colors=[c + [1.0] for c in GLOW_COLS], controls=GLOW_CTRL, name="glow")
    verts, faces = cache0["verts"], cache0["faces"]
    skull_layer = viewer.add_surface((verts, faces, np.ones(len(verts))), name="skull",
                                     colormap=flat([0.55, 0.55, 0.60]), contrast_limits=[0, 1],
                                     shading="smooth", opacity=0.13, blending="translucent")
    if obj0 is not None:
        obj_layer = viewer.add_surface((verts, faces[obj0["cohf"]], obj0["vv"]),
                                       name="placement objective", colormap=glow,
                                       contrast_limits=[obj0["clo"], obj0["chi"]],
                                       shading="smooth", opacity=0.92, blending="translucent")
    else:
        obj_layer = None
        print(f"[warn] no placement-objective cache for '{fr0.name}'. Run: "
              f"python {os.path.basename(__file__)} --build-objfield", flush=True)

    pts0, focus0, apex0, aim0 = cap_disp()
    capv, capf = tx.triangulate_cap(pts0 / mod)
    cap_layer = viewer.add_surface((capv, capf, np.ones(len(capv))), name="bowl",
                                   colormap=flat([0.10, 0.78, 0.92]), contrast_limits=[0, 1],
                                   shading="smooth", opacity=0.55, blending="translucent")
    tgt_layer = viewer.add_points(fr0.target_vox[None] / mod, name="target", size=6,
                                  face_color="lime", border_color="white", border_width=0.15,
                                  blending="translucent_no_depth")
    foc_layer = viewer.add_points(focus0[None] / mod, name="geom focus", size=5, face_color="magenta",
                                  border_color="white", border_width=0.15,
                                  blending="translucent_no_depth")
    apx_layer = viewer.add_points(apex0[None] / mod, name="apex", size=5, face_color="cyan",
                                  border_color="white", border_width=0.15,
                                  blending="translucent_no_depth")
    beam_layer = viewer.add_vectors(np.array([[apex0 / mod, (focus0 - apex0) / mod]]), name="beam",
                                    edge_color="yellow", edge_width=1.2)

    slices = SliceView(cache0["bone_ds"].astype(bool), mod, fr0.dx_mm, slice_planes(fr0))
    slices.set_target(fr0.target_vox)
    viewer.window.add_dock_widget(slices.canvas, name="ortho slices", area="bottom")

    from qtpy.QtWidgets import QLabel
    from qtpy.QtCore import Qt
    legend_txt = CONTROLS_LEGEND
    if len(frames) > 1:
        legend_txt += "\n\ntargets: " + "  ".join(f"{i+1}={f.name}" for i, f in enumerate(frames))
    legend = QLabel(legend_txt)
    legend.setStyleSheet("QLabel{font-family:monospace; font-size:11px; color:#e8e8e8;"
                         " background:#1b1b1f; padding:8px;}")
    legend.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    legend.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
    viewer.window.add_dock_widget(legend, name="controls", area="right")

    viewer.text_overlay.visible = True
    viewer.text_overlay.color = "white"

    def switch_to(fr):
        """Rebind every layer to another bundle's mesh/objective (multi-target sessions)."""
        cache = get_cache(fr)
        state.update(frame=fr, cache=cache, obj=load_obj_valid(fr, cache),
                     tree=cKDTree(cache["verts"]))
        state["pose"] = seed_pose(fr, bowl, state["obj"])
        v, f = cache["verts"], cache["faces"]
        skull_layer.data = (v, f, np.ones(len(v)))
        o = state["obj"]
        if obj_layer is not None:
            obj_layer.visible = o is not None
            if o is not None:
                obj_layer.data = (v, f[o["cohf"]], o["vv"])
                obj_layer.contrast_limits = [o["clo"], o["chi"]]
        slices.bone_ds = cache["bone_ds"].astype(bool)
        slices.planes = slice_planes(fr)
        viewer.title = f"transducer positioning - {fr.name}"

    def refresh(reframe=False):
        pose, fr = state["pose"], state["frame"]
        pts, focus, apex, aim = cap_disp()
        capv, capf = tx.triangulate_cap(pts / mod)
        cap_layer.data = (capv, capf, np.ones(len(capv)))
        foc_layer.data = focus[None] / mod
        apx_layer.data = apex[None] / mod
        tgt_layer.data = fr.target_vox[None] / mod
        beam_layer.data = np.array([[apex / mod, (focus - apex) / mod]])
        if reframe:
            slices.set_target(fr.target_vox)
        slices.update(pts, fr.target_vox, apex)
        bone_ds = state["cache"]["bone_ds"].astype(bool)
        n_bone, n_oob = clearance(pts, bone_ds, mod)
        foc_mm = float(np.linalg.norm(focus - fr.target_vox) * fr.dx_mm)
        score, pct = window_score()
        n_lay, bone_mm, first_mm = beam_bone_profile(apex, aim, bone_ds, mod, fr.dx_mm,
                                                     pose.radius_mm)
        sline = (f"placement score {score:.4g}  ({pct:.0f}% of peak {state['obj']['peak']:.3g})"
                 if score is not None else "placement score n/a (no objective cache)")
        bline = (f"beam crosses {n_lay} bone layer{'s' if n_lay != 1 else ''}, {bone_mm:.1f} mm "
                 f"total (first at {first_mm:.0f} mm from apex)" if n_lay else "beam crosses no bone")
        if n_lay > 1:
            bline += "  <- score is read at the FIRST layer"
        viewer.text_overlay.text = (
            f"target: {fr.name}   az {pose.az_deg:+.0f}  el {pose.el_deg:+.0f}   "
            f"R {pose.radius_mm:.1f} mm   tilt {pose.tilt_deg:+.0f}  yaw {pose.yaw_deg:+.0f}\n"
            f"focus->target {foc_mm:.1f} mm   cap in-bone {n_bone}  off-grid {n_oob}   "
            f"step {state['step']:.0f}\n{sline}\n{bline}")
        print(f"[pos] {fr.name} az{pose.az_deg:+.0f} el{pose.el_deg:+.0f} R{pose.radius_mm:.0f} "
              f"tilt{pose.tilt_deg:+.0f} yaw{pose.yaw_deg:+.0f} -> {sline}; {bline}", flush=True)

    # ---- key bindings (CapPose is frozen -> replace()) ----
    rmin, rmax = bowl.radius_limits_mm

    def bump(**kw):
        p = state["pose"]
        new = {k: v(getattr(p, k)) for k, v in kw.items()}
        state["pose"] = replace(p, **new)

    def bind(key, fn, reframe=False):
        @viewer.bind_key(key, overwrite=True)
        def _wrap(v, fn=fn, reframe=reframe):
            fn()
            refresh(reframe=reframe)

    s = lambda: state["step"]
    bind("Up", lambda: bump(el_deg=lambda x: min(89.0, x + s())))
    bind("Down", lambda: bump(el_deg=lambda x: max(-89.0, x - s())))
    bind("Left", lambda: bump(az_deg=lambda x: x + s()))
    bind("Right", lambda: bump(az_deg=lambda x: x - s()))
    bind(".", lambda: bump(radius_mm=lambda x: min(rmax, x + s())))
    bind(",", lambda: bump(radius_mm=lambda x: max(rmin, x - s())))
    bind("t", lambda: bump(tilt_deg=lambda x: x + s()))
    bind("g", lambda: bump(tilt_deg=lambda x: x - s()))
    bind("y", lambda: bump(yaw_deg=lambda x: x + s()))
    bind("h", lambda: bump(yaw_deg=lambda x: x - s()))
    bind("[", lambda: state.__setitem__("step", max(1.0, state["step"] - 1.0)))
    bind("]", lambda: state.__setitem__("step", min(20.0, state["step"] + 1.0)))
    bind("r", lambda: state.__setitem__("pose", seed_pose(state["frame"], bowl, state["obj"])),
         reframe=True)

    for i, fr in enumerate(frames[:9]):
        bind(str(i + 1), (lambda fr=fr: switch_to(fr)), reframe=True)

    @viewer.bind_key("e", overwrite=True)
    def _export(v):
        sc, pct = window_score()
        export_pose(state["pose"], state["frame"], bowl, score=sc, pct=pct)

    @viewer.bind_key("?", overwrite=True)
    def _help(v):
        print(CONTROLS_LEGEND)

    refresh(reframe=True)
    return viewer, state, refresh


def run_interactive(frames, bowl: Bowl, mod: int):
    import napari
    build_scene(frames, bowl, mod)
    print(__doc__)
    napari.run()


def screenshot(path: str, frames, bowl: Bowl, mod: int,
               swing: float = 40.0, tilt: float = -15.0, zoom: float = 1.35):
    """Build the scene and save a full-window screenshot (3-D canvas + ortho slices + controls
    + live readout). The camera looks obliquely down the beam onto the acoustic window."""
    import time
    from qtpy.QtCore import QCoreApplication
    viewer, state, refresh = build_scene(frames, bowl, mod)
    fr = state["frame"]
    state["pose"] = replace(state["pose"], radius_mm=state["pose"].radius_mm + 0.24 * bowl.roc_mm)
    refresh()
    viewer.window.resize(1740, 1180)
    pump = lambda n: [QCoreApplication.processEvents() or time.sleep(0.04) for _ in range(n)]
    pump(12)
    viewer.reset_view()
    _, _, apex, aim = cap_for_pose(state["pose"], fr, bowl, density=0.04)
    tgt_ds, apex_ds = fr.target_vox / mod, apex / mod
    up = np.asarray(fr.s_hat, float)
    vd = tx.rodrigues(aim, up, swing)
    horiz = np.cross(vd, up)
    horiz = horiz / np.linalg.norm(horiz) if np.linalg.norm(horiz) > 1e-6 else np.asarray(fr.a_hat, float)
    vd = tx.rodrigues(vd, horiz, tilt)
    viewer.camera.set_view_direction(view_direction=tuple(vd), up_direction=tuple(up))
    viewer.camera.center = tuple((tgt_ds + apex_ds) / 2.0)
    viewer.camera.zoom = float(viewer.camera.zoom) * zoom
    pump(10)
    viewer.window.screenshot(path=path, canvas_only=False, flash=False)
    print(f"[screenshot] wrote {path}", flush=True)
    viewer.close()
    return 0


# ============================================================================ tests
def _synthetic_cache(frame: Frame, mod: int) -> dict:
    """A fake hollow-ellipsoid 'skull' covering the real target voxel, for the GUI smoke test."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes
    shp = tuple(int(s) // mod for s in frame.shape)
    zz, yy, xx = np.indices(shp)
    c = np.array(shp) / 2.0
    rad = np.array(shp) / 2.6
    r = (((zz - c[0]) / rad[0]) ** 2 + ((yy - c[1]) / rad[1]) ** 2 + ((xx - c[2]) / rad[2]) ** 2)
    bone_ds = ((r < 1.0) & (r > 0.80)).astype(np.uint8)
    sm = gaussian_filter(bone_ds.astype(np.float32), 0.7)
    verts, faces, _, _ = marching_cubes(sm, level=0.5, step_size=2)
    return dict(verts=verts.astype(np.float32), faces=faces.astype(np.int32),
                bone_ds=bone_ds, mod=mod, N=np.asarray(frame.shape))


def smoke(frames, bowl: Bowl, mod: int):
    """GUI integration test on a synthetic skull: build the scene, drive a few 'keypresses'
    via refresh(), export, then close -- catches API misuse without needing the real mesh."""
    fr = frames[0]
    caches = {fr.dir: _synthetic_cache(fr, mod)}
    viewer, state, refresh = build_scene([fr], bowl, mod, caches=caches)
    print("[smoke] scene built; simulating control actions ...")
    p = state["pose"]
    state["pose"] = replace(p, el_deg=p.el_deg + 5); refresh()
    state["pose"] = replace(state["pose"], radius_mm=state["pose"].radius_mm + 10); refresh()
    state["pose"] = replace(state["pose"], tilt_deg=state["pose"].tilt_deg + 8); refresh()
    export_pose(state["pose"], fr, bowl)
    viewer.close()
    print("[smoke] PASS")
    return 0


def selftest(frames, bowl: Bowl):
    """Validate coordinate transforms + cap geometry WITHOUT a mesh or a GUI."""
    ok = True
    for fr in frames:
        # 1) the registration round-trips the bundle's own anchor
        v = fr.world2vox(fr.target_world)
        err = float(np.linalg.norm(v - fr.target_vox))
        print(f"[selftest] {fr.name}: world {np.round(fr.target_world,3).tolist()} -> vox "
              f"{np.round(v,2).tolist()} vs bundle {np.round(fr.target_vox,2).tolist()}  "
              f"|err| {err:.4f} vox")
        ok &= err < 1e-6

        # 2) anatomical axes are an orthonormal right-handed RAS triad in the grid frame
        M = np.stack([fr.r_hat, fr.a_hat, fr.s_hat])
        orth = float(np.abs(M @ M.T - np.eye(3)).max())
        det = float(np.linalg.det(M))
        print(f"[selftest] {fr.name}: axes R{np.round(fr.r_hat,3).tolist()} "
              f"A{np.round(fr.a_hat,3).tolist()} S{np.round(fr.s_hat,3).tolist()} "
              f"orth-err {orth:.2e} det {det:+.3f}; L = -R {np.allclose(fr.l_hat, -fr.r_hat)}")
        ok &= orth < 1e-9 and det > 0.99

        # 3) elevation +90 must move the apex superior, azimuth +90 anatomical left
        up = tx.pose_apex_aim(fr.target_vox, tx.CapPose(0.0, 90.0, bowl.roc_mm), fr.dx_mm,
                              **fr.axes)[0] - fr.target_vox
        lf = tx.pose_apex_aim(fr.target_vox, tx.CapPose(90.0, 0.0, bowl.roc_mm), fr.dx_mm,
                              **fr.axes)[0] - fr.target_vox
        cs, cl = float(up @ fr.s_hat / np.linalg.norm(up)), float(lf @ fr.l_hat / np.linalg.norm(lf))
        print(f"[selftest] {fr.name}: el+90 . S = {cs:+.4f}   az+90 . L = {cl:+.4f}  (both +1)")
        ok &= cs > 0.999 and cl > 0.999

        # 4) cap geometry: pole on apex, focus = apex+ROC*aim, all dirs within the half-angle
        roc_vox = bowl.roc_mm / fr.dx_mm
        pose = tx.CapPose(az_deg=0.0, el_deg=45.0, radius_mm=bowl.roc_mm)
        pts, focus, apex, aim = cap_for_pose(pose, fr, bowl, density=0.3)
        d_pole = float(np.linalg.norm(pts[0] - apex))
        d_focus = float(np.linalg.norm(focus - (apex + roc_vox * aim)))
        radii = np.linalg.norm(pts - focus, axis=1) * fr.dx_mm
        ang = np.degrees(np.arccos(np.clip(
            ((pts - focus) / np.linalg.norm(pts - focus, axis=1, keepdims=True)) @ (-aim), -1, 1)))
        foc_mm = float(np.linalg.norm(focus - fr.target_vox) * fr.dx_mm)
        print(f"[selftest] {fr.name}: npts {len(pts):5d}  pole-apex {d_pole:.2e}  "
              f"focus-def {d_focus:.2e}  radius {radii.min():.2f}-{radii.max():.2f}mm  "
              f"max-ang {ang.max():.1f}deg (half {bowl.half_angle_deg:.1f})  focus->target {foc_mm:.2e}mm")
        ok &= d_pole < 1e-6 and d_focus < 1e-6
        ok &= abs(radii.max() - bowl.roc_mm) < 1e-3 and abs(radii.min() - bowl.roc_mm) < 1e-3
        ok &= ang.max() < bowl.half_angle_deg + 1e-3 and foc_mm < 1e-3

        # 5) tilt moves the focus off target by ROC*sin(tilt)
        pose = tx.CapPose(az_deg=0.0, el_deg=45.0, radius_mm=bowl.roc_mm, tilt_deg=10.0)
        _, focus, _, _ = cap_for_pose(pose, fr, bowl, density=0.3)
        off = float(np.linalg.norm(focus - fr.target_vox) * fr.dx_mm)
        expect = bowl.roc_mm * np.sin(np.deg2rad(10.0))
        print(f"[selftest] {fr.name}: tilt 10deg -> focus off-target {off:.2f}mm "
              f"(expect ~{expect:.2f}mm)")
        ok &= abs(off - expect) < 0.05 * bowl.roc_mm

        # 6) the ortho-panel labels are derived, not assumed
        print(f"[selftest] {fr.name}: slice planes " +
              " | ".join(f"{t.split()[0]}: h {lb[0]}/{lb[1]} v {lb[2]}/{lb[3]}"
                         for t, _ah, _av, _as, *lb in slice_planes(fr)))

    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return 0 if ok else 1


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description="Interactive transducer positioning on a Field Bundle")
    ap.add_argument("--bundle", action="append", default=None,
                    help="Field Bundle directory (repeat for several targets -> keys 1..9)")
    ap.add_argument("--mod", type=int, default=1, help="downsample factor for mesh/slices")
    ap.add_argument("--roc-mm", type=float, default=DEF_ROC_MM, help="bowl radius of curvature")
    ap.add_argument("--aperture-mm", type=float, default=DEF_APERTURE_MM, help="full aperture diameter")
    ap.add_argument("--build-cache", action="store_true", help="build the mesh/bone cache, then exit")
    ap.add_argument("--build-objfield", action="store_true",
                    help="precompute the placement-objective field, then exit")
    ap.add_argument("--neck-cone-deg", type=float, default=45.0,
                    help="exclusion cone about every significant skull opening "
                         "(foramen magnum, basicranial gap) -- 0 disables")
    ap.add_argument("--selftest", action="store_true", help="headless geometry checks (no GUI)")
    ap.add_argument("--smoke", action="store_true", help="GUI integration test on a synthetic skull")
    ap.add_argument("--screenshot", metavar="PATH", help="save a full-window screenshot, then exit")
    args = ap.parse_args()

    bowl = Bowl(roc_mm=args.roc_mm, aperture_mm=args.aperture_mm)
    frames = [Frame(b) for b in (args.bundle or [DEFAULT_BUNDLE])]
    print(f"[bowl] ROC {bowl.roc_mm:.1f} mm, aperture {bowl.aperture_mm:.1f} mm "
          f"-> half-angle {bowl.half_angle_deg:.1f} deg "
          f"(footprint radius is measured per bundle, in --build-objfield)")
    for fr in frames:
        print(f"[bundle] {fr.name}: {fr.dir}  grid {fr.shape} dx {fr.dx_mm:.4f} mm  "
              f"bone>{fr.bone_c:.0f} m/s  frame {fr.world_frame}")

    if args.selftest:
        return selftest(frames, bowl)
    if args.smoke:
        return smoke(frames, bowl, args.mod)
    if args.build_cache:
        for fr in frames:
            build_cache(fr, args.mod)
        return 0
    if args.build_objfield:
        for fr in frames:
            build_objfield(fr, bowl, args.mod,
                           neck_cone_deg=(args.neck_cone_deg or None))
        return 0
    if args.screenshot:
        return screenshot(args.screenshot, frames, bowl, args.mod)
    run_interactive(frames, bowl, args.mod)
    return 0


if __name__ == "__main__":
    sys.exit(main())
