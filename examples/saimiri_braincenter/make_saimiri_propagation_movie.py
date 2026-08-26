#!/usr/bin/env python3
"""Saimiri brain-center, the movie: the outward time-reversal wave leaving the brain center.

The transparency map is a *time-collapsed* quantity -- one number per patch of calvaria --
so it hides the thing it measures: a spherical wave born at the target, sweeping outward,
reshaped by every millimetre of bone it crosses. This renders that wave. It is the squirrel
monkey counterpart of the human propagation animation (manuscript Fig. 3) and of
``examples/mouse_tips_cerebellum/make_propagation_movie.py``, whose display recipe -- one
global colour limit, signed power-law compression, opacity floor -- it follows.

WHY THIS NEEDS ITS OWN SOLVE. ``build_saimiri_braincenter.py`` deletes ``genout_mod.dat``
after ``extract`` (the bundle is all the map needs, and the dump is GBs). This re-runs the
SAME outward solve and keeps the dump. Everything -- medium, source, grid, pulse -- is
imported from the case module, so the movie shows exactly the wave the shipped map came from.

DISPLAY. The source is a point in a 40 mm head, so |p| falls by more than an order of
magnitude between the target and the far skull; on a linear scale the first two millimetres
would be all you ever saw. The overlay is compressed with a signed power law about ONE
global limit (the 99.5th percentile of |p| over every displayed pixel and frame, with a
2.5 mm ball around the source excluded). Global stops the scale flickering; signed and
monotonic keeps a wavefront reading as a red/blue oscillation with transparent at zero.
What survives late is genuine re-radiation from the reverberating skull, so it is shown, but
the opacity floor keeps it a wash rather than a competitor to the front.

Run (the solve is well under a minute on one A6000):

    GPU=1 python make_saimiri_propagation_movie.py           # solve + render + drop the dump
    GPU=1 python make_saimiri_propagation_movie.py solve
    python make_saimiri_propagation_movie.py render
    python make_saimiri_propagation_movie.py sheet           # contact sheet only (fast iteration)

Knobs: ``GPU`` (default 1), ``SCRATCH_MOVIE``, ``TSTRIDE`` (default 2), ``FPS`` (default 12),
``KEEP_DUMP=1``.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import build_saimiri_braincenter as case            # noqa: E402  (medium, source, spec, grid)
from skull_transparency.sim import _common as C     # noqa: E402
from skull_transparency.sim.launch_core import PAD  # noqa: E402  48 absorbing voxels per side
from skull_transparency.sim.prepare import build_brain_center_run   # noqa: E402

# --------------------------------------------------------------------------- paths
#: Deliberately NOT the case's own SCRATCH: this run writes a multi-GB genout_mod, and
#: keeping the trees apart means re-running the movie cannot disturb a shipped bundle.
SCRATCH = Path(os.environ.get("SCRATCH_MOVIE", "/dev/shm/saim_bc_movie"))
MP4 = HERE / "propagation_saimiri_braincenter.mp4"
GIF = HERE / "propagation_saimiri_braincenter.gif"
SHEET = HERE / "propagation_contact_sheet.png"

# --------------------------------------------------------------------------- display
TSTRIDE = int(os.environ.get("TSTRIDE", 2))
FPS = int(os.environ.get("FPS", 12))
PCTL = 99.5           # percentile of |p| (all displayed pixels/frames) that sets the limit
GAMMA = 0.60          # signed power-law compression of p/vmax; 1.0 = linear
ALPHA_LO, ALPHA_HI = 0.10, 0.40
ALPHA_MAX = 0.90
#: Radius excluded when setting vmax, so the 1/r source singularity does not swallow the
#: colour scale. Scaled from the mouse's 1.5 mm by head size (~20 mm -> ~40 mm across).
NEAR_SOURCE_MM = 2.5
SHEET_N = 9
#: Greyscale window (m/s) for the sound-speed background. The bottom sits WELL below water:
#: the overlay is translucent where the field is weak, and a pale half-transparent blue on
#: pure black reads as near-black hatching. Water at ~20% grey keeps weak field looking weak.
BG_VMIN, BG_VMAX = 1200.0, 2900.0


# ============================================================================ solve
def solve():
    """Re-run the outward solve keeping the volume dump. Identical to the case's own solve."""
    c_map, affine, center = case._load_inputs()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(c_map, affine, case.SPEC, SCRATCH, center_phys_mm=center,
                                 bone_threshold=case.BONE_THRESHOLD,
                                 surround_mm=case.SURROUND_MM, input_frame=case.INPUT_FRAME)
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
    """The recorded field plus everything needed to place it in space and time."""

    def __init__(self, scratch=SCRATCH):
        scratch = Path(scratch)
        self.meta = json.loads((scratch / "meta.json").read_text())
        ws = C.load_workspace(scratch / "outward" / "workspace.npz")
        s = ws["scalars"]

        self.grid = C.grid_shape(self.meta)
        self.mod = int(s["modX"])
        assert int(s["modY"]) == self.mod == int(s["modZ"]), "anisotropic decimation not handled"
        self.dx_mm = float(self.meta["dX_m"]) * 1e3
        self.dxf_mm = self.dx_mm * self.mod
        self.c0_ms = float(s["c0"])
        self.bone_threshold = case.BONE_THRESHOLD

        self.dT_s = float(s["dX"]) / self.c0_ms * float(s["cfl"])
        self.modT = int(s["modT"])
        self.dt_frame_s = self.dT_s * self.modT

        self.target_vox = np.asarray(self.meta["dent_grid"], float)
        self.target_f = self.target_vox / self.mod

        self.nf = tuple(len(range(0, n, self.mod)) for n in self.grid)
        self.lo = PAD // self.mod
        npad = tuple((n + 2 * PAD) // self.mod + ((n + 2 * PAD) % self.mod > 0) for n in self.grid)
        gm = scratch / "outward" / "genout_mod.dat"
        if not gm.exists():
            raise SystemExit(f"{gm} not found -- run `make_saimiri_propagation_movie.py solve` first.")
        nbytes = gm.stat().st_size
        per = 4 * int(np.prod(npad))
        if nbytes % per:
            raise SystemExit(f"{gm} ({nbytes} B) is not a whole number of {npad} frames")
        self.nframes = nbytes // per
        self.vol = np.memmap(gm, dtype="<f4", mode="r", shape=(self.nframes,) + npad)
        self.gm_bytes = nbytes

        c_file = self.meta.get("c_file", "c.f32")
        c = np.fromfile(scratch / c_file, dtype="<f4").reshape(*self.grid, order="F")
        self.c_full = c
        self.c_f = c[::self.mod, ::self.mod, ::self.mod]

    def t_s(self, k):
        return k * self.dt_frame_s

    def plane(self, axis, index):
        """All frames of one plane, ``(nframes, A, B)``, cropped to the interior grid."""
        lo, nf = self.lo, self.nf
        j = lo + int(round(index))
        sl = [slice(None), slice(lo, lo + nf[0]), slice(lo, lo + nf[1]), slice(lo, lo + nf[2])]
        sl[axis + 1] = j
        return np.asarray(self.vol[tuple(sl)], dtype=np.float32)

    def mm_from_target(self, axis, n=None):
        n = self.nf[axis] if n is None else n
        return (np.arange(n) * self.mod - self.target_vox[axis]) * self.dx_mm

    def extent(self, a_axis, b_axis):
        a = self.mm_from_target(a_axis)
        b = self.mm_from_target(b_axis)
        h = 0.5 * self.dxf_mm
        return (a[0] - h, a[-1] + h, b[0] - h, b[-1] + h)


# ============================================================================ display
def _window_vox():
    """Full-res voxel of the chosen window, or None. Decorative, so failures are swallowed."""
    try:
        from run_saimiri_report import chosen_placement
        _b, _t, pl, *_ = chosen_placement(verbose=False)
        return np.asarray(pl.window_center_fullres_voxel, float)
    except Exception as exc:                       # noqa: BLE001 -- decoration only
        print(f"  (no window marker: {type(exc).__name__}: {exc})")
        return None


def _rgba(sl, vmax, cmap, gamma=GAMMA):
    """Signed pressure slice -> RGBA, compressed about zero and transparent where quiet."""
    u = np.clip(sl / vmax, -1.0, 1.0)
    s = np.sign(u) * np.abs(u) ** gamma
    rgba = cmap(0.5 * (s + 1.0))
    rgba[..., 3] = np.clip((np.abs(s) - ALPHA_LO) / (ALPHA_HI - ALPHA_LO), 0.0, 1.0) * ALPHA_MAX
    return rgba


PANELS = (
    ("sagittal", 0, (1, 2), "y  anterior + (mm)", "z  superior + (mm)"),
    ("coronal",  1, (0, 2), "x  right + (mm)",    "z  superior + (mm)"),
    ("axial",    2, (0, 1), "x  right + (mm)",    "y  anterior + (mm)"),
)
TITLE = "Squirrel monkey, 1 MHz -- outward wave from the brain center"


def _setup_axis(F, ax, name, fix_axis, ip, xl, yl, bg, win_vox):
    """Draw the static parts of one panel: skull, target cross, optional window marker."""
    ext = F.extent(ip[0], ip[1])
    ax.imshow(bg.T, cmap="gray", origin="lower", extent=ext, vmin=BG_VMIN, vmax=BG_VMAX,
              interpolation="bilinear", zorder=1)
    im = ax.imshow(np.zeros(bg.T.shape + (4,)), origin="lower", extent=ext,
                   interpolation="bilinear", zorder=2)
    ax.contour(np.linspace(ext[0], ext[1], bg.shape[0]), np.linspace(ext[2], ext[3], bg.shape[1]),
               bg.T, levels=[F.bone_threshold], colors="k", linewidths=0.45, alpha=0.65, zorder=3)
    ax.plot(0.0, 0.0, "+", color="#39ff5e", ms=11, mew=1.8, zorder=4)     # the brain center
    if win_vox is not None:
        off = abs((win_vox[fix_axis] - F.target_vox[fix_axis]) * F.dx_mm)
        if off < 4.0:
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

    vals = []
    for ax_i, (ip0, ip1) in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
        a = F.mm_from_target(ip0)[:, None]
        b = F.mm_from_target(ip1)[None, :]
        far = (a ** 2 + b ** 2) > NEAR_SOURCE_MM ** 2
        vals.append(np.abs(planes[ax_i][:, far]).ravel())
    vmax = float(np.percentile(np.concatenate(vals), PCTL))
    pmax = float(max(np.abs(planes[i]).max() for i in planes))
    print(f"  |p| on the displayed planes: max {pmax:.3g} Pa, p{PCTL} (>{NEAR_SOURCE_MM} mm "
          f"from source) {vmax:.3g} Pa  ->  dynamic range {pmax / vmax:.0f}x, "
          f"compressed with gamma = {GAMMA}")

    win_vox = _window_vox()
    cmap = plt.get_cmap("RdBu_r")
    frames = list(range(0, F.nframes, tstride))

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
    figs.suptitle(TITLE + " (sagittal through the target)", fontsize=11)
    figs.tight_layout(rect=(0, 0, 1, 0.96))
    figs.savefig(SHEET, dpi=92, pil_kwargs={"optimize": True})
    plt.close(figs)
    print(f"  contact sheet -> {SHEET}  ({SHEET.stat().st_size / 1e6:.2f} MB, frames {list(picks)})")
    if sheet_only:
        return {"sheet": SHEET}

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
        print(f"  (no mp4 encoder: {type(exc).__name__}: {exc}; sheet only)")
    gif_frames = []

    for k in frames:
        for i, (nm, fx, ip, xl, yl) in enumerate(PANELS):
            ims[i].set_data(_rgba(planes[fx][k].T, vmax, cmap))
        sup.set_text(f"{TITLE}      t = {F.t_s(k) * 1e6:6.2f} us")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        for _, w in writers:
            w.append_data(buf)
        if len(gif_frames) < 400:
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
    gm = SCRATCH / "outward" / "genout_mod.dat"
    if gm.exists() and not os.environ.get("KEEP_DUMP"):
        sz = gm.stat().st_size / 1e9
        gm.unlink()
        print(f"  removed {gm} ({sz:.2f} GB)")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "clean":
        shutil.rmtree(SCRATCH, ignore_errors=True)
        print(f"  removed {SCRATCH}")
        return 0
    if cmd in ("solve", "all"):
        solve()
    if cmd == "sheet":
        render(sheet_only=True)
        return 0
    if cmd in ("render", "all"):
        render()
    if cmd == "all":
        drop_dump()
    return 0


if __name__ == "__main__":
    sys.exit(main())
