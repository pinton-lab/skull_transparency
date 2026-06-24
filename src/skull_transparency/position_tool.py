"""Transducer-positioning preview / viewer — the optional "positioning tool" output.

Two entry points, both on the package's own abstractions (a :class:`TransparencyMap`
+ a ``BowlPlacement``), so they work for any subject — not just the Halle CTX-500:

* :func:`preview_placement` — a STATIC figure (the transparency map on the skull
  surface + the chosen window / beam / target), headless-safe (matplotlib ``Agg``),
  so it renders without a display (and is testable in CI). Needs the ``[viz]`` extra
  (matplotlib).
* :func:`view_napari` — the interactive 3-D viewer (the ``[viz]`` extra; lazily imports
  napari). Runs on the user's desktop (Linux/macOS/Windows); not for headless use.

This generalises the Halle-specific ``ctx500_position_tool.py`` (manuscript Appendix A)
onto the library's frame-agnostic objects.
"""
from __future__ import annotations

import numpy as np


def _surface_points(tmap):
    """Surface patch coords in MNI mm if a registration is attached, else full-res voxels."""
    if getattr(tmap, "registration", None) is not None:
        return np.asarray(tmap.surf_mni_mm(), float), "MNI mm"
    return np.asarray(tmap.surf_vox, float), "voxel"


def preview_placement(tmap, placement, out_png=None, *, show=False, title="placement"):
    """Render the transparency map (surface patches coloured by coupling) with the chosen
    window, beam, and target marked, from two viewpoints. Saves to ``out_png`` if given;
    returns the path. Headless-safe unless ``show=True``."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    P, frame = _surface_points(tmap)
    coup = np.asarray(tmap.Ipk_Wcm2, float)
    win = np.asarray(placement.window_center_mni_mm, float)
    tgt = np.asarray(placement.target_mni_mm, float)
    apex = np.asarray(placement.apex_mni_mm, float)

    fig = plt.figure(figsize=(12, 5.5))
    sc = None
    for k, (az, el) in enumerate([(30, 20), (210, 20)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=coup, s=4, cmap="viridis")
        ax.scatter(*win, c="red", s=140, marker="*", label="window", depthshade=False)
        ax.scatter(*tgt, c="k", s=90, marker="x", label="target", depthshade=False)
        ax.plot([apex[0], tgt[0]], [apex[1], tgt[1]], [apex[2], tgt[2]], "r--", lw=1.2, label="beam")
        ax.view_init(elev=el, azim=az)
        ax.set_title(f"view {k + 1}  ({frame})", fontsize=9)
        if k == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=fig.axes, shrink=0.55, pad=0.02, label="coupling Ipk (W/cm^2)")
    sco = float(getattr(placement, "transparency_score", float("nan")))
    inc = float(getattr(placement, "incidence_deg", float("nan")))
    fig.suptitle(f"{title}  |  score {sco:.3f}  |  incidence {inc:.1f} deg", fontsize=11)
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def view_napari(tmap, placement):
    """Open an interactive napari window: the surface coupling as a Points layer, plus the
    window / target markers. Desktop only (needs a display + the ``[viz]`` extra)."""
    try:
        import napari
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise ImportError("the interactive viewer needs napari: pip install -e '.[viz]'") from e

    P, _ = _surface_points(tmap)
    coup = np.asarray(tmap.Ipk_Wcm2, float)
    win = np.asarray(placement.window_center_mni_mm, float)[None, :]
    tgt = np.asarray(placement.target_mni_mm, float)[None, :]
    v = napari.Viewer(title="skull-transparency placement")
    v.add_points(P, features={"coupling": coup}, face_color="coupling", face_colormap="viridis",
                 size=2.0, name="transparency")
    v.add_points(win, face_color="red", size=6.0, name="window")
    v.add_points(tgt, face_color="white", size=5.0, symbol="x", name="target")
    napari.run()
    return v
