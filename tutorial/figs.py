#!/usr/bin/env python3
"""Generate the schematic figures for the CT-import / grid tutorial.

Verified constants of the manuscript whole-skull simulation
(from the graded 6-ppw run's meta.json, cross-checked vs
halle_c_graded.f32 = 652*814*650*4 bytes):
  N=[652,814,650] voxels, dX=0.25 mm, F0=1 MHz, c0=1540 m/s, ppw=6.16, graded
  medium, R_world->grid = diag(-1,1,1) (axis-aligned: i=Left,j=Anterior,k=Superior),
  target dentate MNI(-12,-57,-34) <-> voxel(372.65,324.71,140.59).

Schematics produced:  fig1_pipeline.png (pipeline + the artifact each step writes),
fig_grid.png (the box in space + the rigid map), fig5_medium.png (the medium).
The animation, targets-in-skull and positioning-tool figures are reused from the
manuscript (../manuscript/figs/). Run:  python figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

# ---- shared style -----------------------------------------------------------
INK = "#1d1d1f"
BLUE = "#2c6fbf"      # in-package (skull_transparency)
ORANGE = "#d98030"    # upstream (you supply)
GREEN = "#2e8b57"     # full-wave GPU solver
RED = "#c0392b"       # target
WATER = "#bcd9ef"
BONE = "#e0cfa6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": "#444", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 200,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# ---- verified grid constants (whole-skull 6.16-ppw graded domain) -----------
N = np.array([652, 814, 650])
DX_MM = 0.25
EXTENT = N * DX_MM                       # [163.0, 203.5, 162.5] mm
TGT_MNI = np.array([-12.0, -57.0, -34.0])
TGT_VOX = np.array([372.65, 324.71, 140.59])
R = np.diag([-1.0, 1.0, 1.0])            # world -> grid (axis-aligned, det = -1)


def v2m(v):                              # voxel -> world mm (R is its own inverse here)
    v = np.atleast_2d(np.asarray(v, float))
    return (TGT_MNI + DX_MM * (R @ (v - TGT_VOX).T).T).reshape(np.shape(v))


def rounded(ax, xy, w, h, fc, ec, text, tcolor=None, fs=10, lw=1.6, bold=False):
    ax.add_patch(FancyBboxPatch((xy[0], xy[1]), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tcolor or INK, zorder=3,
            fontweight="bold" if bold else "normal")


def arrow(ax, p0, p1, color=INK, lw=2.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=15,
                                 lw=lw, color=color, zorder=1, shrinkA=2, shrinkB=2))


# =============================================================================
# FIG (pipeline) -- data flow + the artifact each stage writes (old 1 + 6)
# =============================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.2, 7.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.0); ax.axis("off")
    bx, bw, bh = 0.25, 4.25, 0.62
    stages = [
        ("Clinical CT  (DICOM / NIfTI)", ORANGE, "#fbeede", "Hounsfield-unit volume"),
        ("HU → c (ρ, α) map   ·   your model", ORANGE, "#fbeede",
         "c.nii.gz  (+ voxel-to-world affine)"),
        ("prepare:  pose · resample · register", BLUE, "#e7f0fb",
         "c.f32 · meta.json · registration.json"),
        ("outward full-wave solve  (Fullwave2, GPU)", GREEN, "#e6f2ea",
         "surface field · decimated volume · element signals"),
        ("extract → Field Bundle", BLUE, "#e7f0fb", "bundle/  (arrays memory-mapped)"),
        ("compute_transparency_map", BLUE, "#e7f0fb", "surface_map.npz  (|p|, I, Δt per patch)"),
        ("place_bowl / place_cap_optimal", BLUE, "#e7f0fb", "placement.json · score.json"),
        ("inward time-reversal solve  (re-emit → refocus)", GREEN, "#e6f2ea",
         "focal field: peak pressure, focusing gain"),
        ("scale to insonication", BLUE, "#e7f0fb",
         "Isppa, Ispta, MI  (drive amplitude · duty cycle)"),
    ]
    n = len(stages); ytop = 8.2; gap = 0.95
    ys = [ytop - i * gap for i in range(n)]
    for i, (title, ec, fc, art) in enumerate(stages):
        y = ys[i]
        rounded(ax, (bx, y), bw, bh, fc, ec, title, fs=9.0, bold=True)
        if i < n - 1:
            arrow(ax, (bx + bw / 2, y), (bx + bw / 2, ys[i + 1] + bh), color="#888", lw=1.5)
        arrow(ax, (bx + bw + 0.05, y + bh / 2), (bx + bw + 0.42, y + bh / 2), color="#bbb",
              lw=1.0, style="->")
        ax.text(bx + bw + 0.52, y + bh / 2, art, ha="left", va="center", fontsize=7.7,
                family="DejaVu Sans Mono", color="#444")
    ydiv = (ys[1] + ys[2] + bh) / 2
    ax.plot([0.15, 9.85], [ydiv, ydiv], ls=(0, (6, 4)), color=RED, lw=1.4)
    ax.text(0.2, ydiv + 0.09, "your model (upstream, per-subject):  CT → sound-speed map",
            fontsize=8, color=ORANGE, va="bottom", fontweight="bold")
    ax.text(0.2, ydiv - 0.09, "skull_transparency:  prepare → solve → extract → place",
            fontsize=8, color=BLUE, va="top", fontweight="bold")
    leg = [Line2D([0], [0], marker="s", color="w", markerfacecolor="#fbeede",
                  markeredgecolor=ORANGE, markersize=13, label="upstream (CT → sound-speed map)"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="#e6f2ea",
                  markeredgecolor=GREEN, markersize=13, label="full-wave GPU solver"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="#e7f0fb",
                  markeredgecolor=BLUE, markersize=13, label="pure-Python package")]
    ax.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=3,
              frameon=False, fontsize=8, handletextpad=0.5, columnspacing=1.3)
    ax.set_title("From clinical CT to insonication: the pipeline and what each step writes",
                 fontsize=12.5, fontweight="bold", pad=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIG (grid) -- the box in space + the rigid map that places it (old 2 + 3)
# =============================================================================
def fig_grid():
    fig = plt.figure(figsize=(12.2, 5.7))
    # ---- Panel A: 3D axis-aligned box in world (MNI) coordinates ----
    axA = fig.add_subplot(1, 2, 1, projection="3d")
    corners = v2m(np.array([[i, j, k] for i in (0, N[0]) for j in (0, N[1])
                            for k in (0, N[2])], float))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
             (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        axA.plot(*zip(corners[a], corners[b]), color=BLUE, lw=1.3, alpha=0.85)
    tgt = v2m(TGT_VOX)[0]
    for vec, lab in [((1, 0, 0), "+x Right"), ((0, 1, 0), "+y Ant"), ((0, 0, 1), "+z Sup")]:
        v = np.array(vec, float) * 68
        axA.quiver(*tgt, *v, color="#444", lw=2.0, arrow_length_ratio=0.12)
        axA.text(*(tgt + v * 1.05), lab, color="#444", fontsize=8.5, fontweight="bold")
    d = np.array([-1, 0, 0], float) * 50          # grid i runs Left = -x (the flip)
    axA.quiver(*tgt, *d, color="#1f6f1f", lw=2.0, arrow_length_ratio=0.16, alpha=0.7)
    axA.text(*(tgt + d * 1.12), "grid i (Left)", color="#1f6f1f", fontsize=8.5, fontweight="bold")
    axA.scatter(*tgt, color=RED, s=70, depthshade=False)
    axA.text(*(tgt + np.array([10, 4, -34])), "target\nMNI (-12,-57,-34)\nvox (372.7,324.7,140.6)",
             color=RED, fontsize=7.6)
    axA.set_xlabel("x (mm)"); axA.set_ylabel("y (mm)"); axA.set_zlabel("z (mm)")
    axA.set_title("652 × 814 × 650 box at 0.25 mm,\naxis-aligned in world space", fontsize=10)
    axA.view_init(elev=16, azim=-62)
    try:
        axA.set_box_aspect(EXTENT)
    except Exception:
        pass
    # ---- Panel B: the rigid map (= the affine) ----
    axB = fig.add_subplot(1, 2, 2)
    axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
    rounded(axB, (0.3, 7.4), 3.2, 1.4, "#eef3fb", BLUE, "world (MNI) mm\n$p=(x,y,z)$", fs=10, bold=True)
    rounded(axB, (6.5, 7.4), 3.2, 1.4, "#fef1e7", ORANGE, "grid voxel\n$v=(i,j,k)$", fs=10, bold=True)
    arrow(axB, (3.55, 8.35), (6.45, 8.35), color="#444", lw=1.8)
    axB.text(5.0, 8.64, r"$v = v_* + \frac{1}{\delta}\,R\,(p-p_*)$", ha="center", fontsize=10.5)
    arrow(axB, (6.45, 7.55), (3.55, 7.55), color="#444", lw=1.8)
    axB.text(5.0, 7.0, r"$p = p_* + \delta\,R^{\!\top}(v-v_*)$", ha="center", fontsize=10.5)
    axB.text(0.3, 5.95,
             "anchor (one matched point pins the grid):\n"
             r"   $p_*=$ MNI(-12,-57,-34) mm $\Leftrightarrow$ $v_*=$ (372.7, 324.7, 140.6) vox"
             "\n   isotropic pitch  $\\delta = 0.25$ mm",
             fontsize=8.8, va="top")
    axB.text(0.3, 4.05,
             "$R$ (world $\\to$ grid) $=$ diag(-1, 1, 1):\n"
             "   $i$ = Left ($-x$),  $j$ = Anterior,  $k$ = Superior\n"
             "   det $=-1$  (an L/R handedness flip, not a rotation)",
             fontsize=8.8, va="top")
    axB.add_patch(FancyBboxPatch((0.2, 0.4), 9.5, 1.75,
                  boxstyle="round,pad=0.02,rounding_size=0.06", fc="#f6f6f6", ec="#aaa", lw=1.2))
    axB.text(0.45, 1.92, "Same information as the 4×4 input affine:",
             fontsize=8.8, va="top", fontweight="bold")
    axB.text(0.45, 1.45,
             r"   affine$[:3,:3]=\delta R^{\!\top}$,   affine$[:3,3]$ pins the anchor."
             "\n   A NIfTI sform IS an $(R,\\delta,$ anchor$)$ — prepare reads orientation off its signs.",
             fontsize=8.6, va="top")
    axB.set_title("The rigid map = one $R$ + one anchored point (= the affine)", fontsize=10)
    fig.suptitle("The simulation grid in space, and the rigid map that places it",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "fig_grid.png"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIG (medium) -- three user-defined maps + scalar beta; rho fallback
# =============================================================================
def fig_medium():
    fig = plt.figure(figsize=(11.0, 5.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.18)
    axA = fig.add_subplot(gs[0, 0]); axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
    axA.text(2.3, 9.5, "Default: user-defined maps + scalar", ha="center",
             fontsize=11, fontweight="bold", color=INK)
    maps = [(r"sound speed  $c$", "m/s", WATER, 7.2),
            (r"density  $\rho$", "kg/m$^3$", "#cfe0c2", 5.1),
            (r"absorption  $\alpha$", "dB/MHz/cm", "#e8cfd6", 3.0)]
    for name, unit, col, y in maps:
        axA.add_patch(FancyBboxPatch((0.5, y), 3.5, 1.6,
                      boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec="#777", lw=1.4))
        axA.text(2.25, y + 0.98, name, ha="center", fontsize=11, fontweight="bold")
        axA.text(2.25, y + 0.46, unit, ha="center", fontsize=8.5, color="#555")
        arrow(axA, (4.05, y + 0.8), (6.1, 5.0), color="#999", lw=1.5)
    axA.add_patch(FancyBboxPatch((0.7, 1.15), 3.1, 0.95,
                  boxstyle="round,pad=0.02,rounding_size=0.18", fc="#ededf0", ec="#999", lw=1.2))
    axA.text(2.25, 1.78, r"$\beta$  nonlinearity (scalar)", ha="center", fontsize=9.5, fontweight="bold")
    axA.text(2.25, 1.4, "default 5.5", ha="center", fontsize=8.3, color="#555")
    arrow(axA, (3.85, 1.62), (6.1, 4.6), color="#999", lw=1.3)
    axA.add_patch(FancyBboxPatch((6.2, 3.9), 3.3, 2.4,
                  boxstyle="round,pad=0.02,rounding_size=0.08", fc="#e6f2ea", ec=GREEN, lw=2))
    axA.text(7.85, 5.5, "Fullwave2", ha="center", fontsize=11.5, fontweight="bold", color=GREEN)
    axA.text(7.85, 4.8, r"forms $K=c^2\rho$", ha="center", fontsize=10)
    axA.text(5.0, 0.4, r"supply $c,\rho,\alpha$ maps and the scalar $\beta$ from your own HU$\to$property models",
             ha="center", fontsize=8.7, color="#555", style="italic")
    axB = fig.add_subplot(gs[0, 1])
    c = np.linspace(1450, 3050, 400)
    rho = 1000 + np.clip((c - 1540) / 1360, 0, 1) * 1200.0
    axB.fill_between(c, 0, rho, where=(c <= 1540), color=WATER, alpha=0.4)
    axB.fill_between(c, 0, rho, where=(c >= 2900), color=BONE, alpha=0.5)
    axB.plot(c, rho, color=BLUE, lw=2.4)
    axB.scatter([1540, 2900], [1000, 2200], color=RED, s=35, zorder=5)
    axB.set_xlabel("sound speed  c  (m/s)"); axB.set_ylabel(r"density  $\rho$  (kg/m$^3$)")
    axB.set_ylim(950, 2350); axB.grid(alpha=0.25)
    axB.set_title(r"Fallback: synthesize $\rho$ from $c$", fontsize=10.5)
    axB.text(0.5, 0.04, r"$\rho=1000+\mathrm{clip}(\frac{c-1540}{1360},0,1)\cdot1200$",
             transform=axB.transAxes, fontsize=8.4, ha="center")
    fig.suptitle("The medium, three user-defined maps plus a scalar nonlinearity",
                 fontsize=11.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(os.path.join(OUT, "fig5_medium.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline(); print("pipeline ok")
    fig_grid(); print("grid ok")
    fig_medium(); print("medium ok")
    print("figures ->", OUT)
