#!/usr/bin/env python3
"""Sparse-array element selection for the dentate + a pure-Python placement preview.

Selects N elements by transparency (incidence-weighted, spacing-constrained) and reports
the FAIR, power-normalized placement gain that the pure-Python layer can compute from the
surface map alone: under time reversal the on-target peak scales as sqrt(sum_i E_i), so the
honest placement metric is sqrt(sum coupling) at matched N versus a transparency-blind
uniform-tiling baseline -- NOT the old "38x vs random scatter", which compared intensities
without power normalization against a diffraction-confounded baseline.

The authoritative, data-driven validation (recorded per-element traces; TR vs geometric
focusing; the manuscript Figure 5 written to manuscript/figs/array_placement.png) lives in
  hemisphere_tr/analysis/placement_validation/{fair_compare.py, make_figure.py}
because it needs the heavy per-element sim traces, not just the small surface map.
This example writes only a local preview. Run with PYTHONPATH=../../src."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import skull_transparency as st
from skull_transparency import paths

DATA = str(paths.bundle_dir())   # override via $SKULL_TR_DATA_ROOT
FIG = Path(__file__).resolve().parent / "array_placement_preview.png"   # NOT the manuscript figure
N, SP, TH, RWIN = 64, 5.0, 35.0, 45.0   # elements, spacing(mm), incid cap(deg), window radius(mm)


def main():
    rng = np.random.default_rng(0)
    tm = st.compute_transparency_map(st.load_bundle(DATA))
    # 1. transparency picks the WINDOW (occiput); 2. populate it with a conformal sparse array
    win = st.place_bowl(tm, st.BowlConstraints(theta_max_deg=TH)).window_center_mni_mm
    al = st.place_array(tm, st.ArrayConstraints(n_elements=N, min_spacing_mm=SP, theta_max_deg=TH,
                                                region_center_mni_mm=win, region_radius_mm=RWIN))
    print(f"placed {al.n_placed}/{al.n_requested}  aperture {al.aperture_extent_mm:.1f} mm  "
          f"mean incid {al.incidence_deg.mean():.1f}+/-{al.incidence_deg.std():.1f} deg  "
          f"sum-coupling {al.coupling.sum():.2f}  (n_legal {al.extras['n_legal']})")

    # FAIR baseline: the same N spread UNIFORMLY (farthest-point) over the same legal window,
    # transparency-blind. Placement gain under TR = sqrt(sum coupling) ratio (power-normalized).
    sm = tm.surf_mni_mm()
    cos = np.clip(np.sum(tm.rhat * tm.true_normal, axis=1), -1, 1)
    legal = np.where((cos >= np.cos(np.deg2rad(TH))) &
                     (np.linalg.norm(sm - win, axis=1) <= RWIN))[0]
    fps = [int(legal[np.argmin(np.linalg.norm(sm[legal] - al.target_mni_mm, axis=1))])]
    dmin = np.linalg.norm(sm[legal] - sm[fps[0]], axis=1)
    while len(fps) < al.n_placed:
        j = int(np.argmax(dmin)); fps.append(int(legal[j]))
        dmin = np.minimum(dmin, np.linalg.norm(sm[legal] - sm[legal[j]], axis=1))
    sumE_T, sumE_U = al.coupling.sum(), tm.Ipk_Wcm2[fps].sum()
    peak_gain = float(np.sqrt(sumE_T / sumE_U))
    print(f"map-only proxy vs uniform-tiling baseline: energy {sumE_T/sumE_U:.2f}x -> peak {peak_gain:.2f}x")
    print("  NOTE: coarse surface-map proxy (per-patch peak intensity over a permissive legal\n"
          "  region). The authoritative power-matched gain from the recorded element traces is\n"
          "  1.2x (vs uniform tiling of the same window) to 1.7x (vs the whole-array spread);\n"
          "  see hemisphere_tr/analysis/placement_validation/fair_compare.py.")

    el, t = al.element_mni_mm, al.target_mni_mm
    fig = plt.figure(figsize=(11, 4.6), facecolor="white")
    ax = fig.add_subplot(1, 2, 1)
    sub = rng.choice(len(sm), min(30000, len(sm)), replace=False)
    ax.scatter(sm[sub, 0], sm[sub, 2], s=1, c="0.82", rasterized=True)
    ax.scatter(sm[fps, 0], sm[fps, 2], s=30, facecolors="none", edgecolors="#d95f02",
               linewidths=1.0, label="uniform baseline")
    sc = ax.scatter(el[:, 0], el[:, 2], c=al.coupling, s=26, cmap="viridis", edgecolor="k", linewidth=0.3)
    ax.scatter([t[0]], [t[2]], marker="*", s=180, c="red", edgecolor="k", label="target", zorder=5)
    ax.set_aspect("equal"); ax.set_xlabel("x  L-R (mm)"); ax.set_ylabel("z  I-S (mm)")
    ax.set_title(f"{al.n_placed} transparency-selected elements\n(posterior view)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(sc, ax=ax, label="per-element coupling (W/cm$^2$)")

    ax = fig.add_subplot(1, 2, 2)
    ax.bar([0, 1], [np.sqrt(sumE_T), np.sqrt(sumE_U)],
           color=["#2c7fb8", "#bdbdbd"], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["transparency\n-selected", "uniform\ntiling"])
    ax.set_ylabel(r"TR focal-peak proxy  $\sqrt{\sum_i E_i}$")
    ax.set_title(f"map-only proxy (coarse); authoritative gain 1.2-1.7x\n"
                 f"(N={al.n_placed}, spacing {SP:.0f} mm; see placement_validation/)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor="white")
    print(f"WROTE {FIG}  (manuscript Fig.5 comes from hemisphere_tr/analysis/placement_validation)")


if __name__ == "__main__":
    main()
