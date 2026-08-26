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
    """Render the transparency map (surface patches coloured by amplitude in dB) with the
    chosen window, beam, and target marked, from two viewpoints — one looking down the
    beam axis, one from the opposite side. Saves to ``out_png`` if given; returns the
    path. Headless-safe unless ``show=True``."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    P, frame = _surface_points(tmap)
    # amplitude in dB re the 98th-percentile patch: a linear intensity scale collapses the
    # map's dynamic range into a single dark tone
    amp = np.sqrt(np.maximum(np.asarray(tmap.value, float), 0.0))
    ref = np.percentile(amp, 98.0) or 1.0
    db = 20.0 * np.log10(np.maximum(amp / ref, 1e-2))
    vlo, vhi = np.percentile(db, 25.0), np.percentile(db, 99.0)
    rhat = np.asarray(tmap.rhat, float)
    win = np.asarray(placement.window_center_mni_mm, float)
    tgt = np.asarray(placement.target_mni_mm, float)
    apex = np.asarray(placement.apex_mni_mm, float)

    # camera 1 looks down the beam axis (window face-on); camera 2 from the far side
    v = apex - tgt
    v = v / (np.linalg.norm(v) or 1.0)
    az0 = float(np.degrees(np.arctan2(v[1], v[0])))
    el0 = float(np.clip(np.degrees(np.arcsin(np.clip(v[2], -1, 1))), -60, 60))
    views = [(el0, az0), (el0, az0 + 180.0)]

    fig = plt.figure(figsize=(12.5, 5.5))
    sc = None
    for k, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.computed_zorder = False           # keep the markers/beam painted over the dense cloud
        e, a = np.radians(el), np.radians(az)
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        m = rhat @ cam > 0.0                 # camera-facing hemisphere only (no occlusion in mpl 3-D)
        sc = ax.scatter(*P[m].T, c=db[m], s=4, cmap="inferno", vmin=vlo, vmax=vhi,
                        linewidths=0, zorder=2)
        ax.scatter(*win, c="red", s=200, marker="*", edgecolors="k", linewidths=0.6,
                   label="window", depthshade=False, zorder=6)
        ax.scatter(*tgt, c="#2ca02c", s=110, marker="X", edgecolors="k", linewidths=0.6,
                   label="target", depthshade=False, zorder=6)
        ax.plot([apex[0], tgt[0]], [apex[1], tgt[1]], [apex[2], tgt[2]], color="#2ca02c",
                ls="--", lw=1.6, label="beam", zorder=5)
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect((1, 1, 1))
        lo, hi = P.min(0), P.max(0)
        mid, sp = (hi + lo) / 2, (hi - lo).max() / 2 or 1.0
        for kk, mm in zip("xyz", mid):
            getattr(ax, f"set_{kk}lim")(mm - sp, mm + sp)
        ax.set_title(("beam-axis view" if k == 0 else "far side") + f"  ({frame})", fontsize=9)
        if k == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=fig.axes, shrink=0.55, pad=0.02,
                 label="transparency amplitude (dB re 98th pct)")
    sco = float(getattr(placement, "transparency_score", float("nan")))
    inc = float(getattr(placement, "incidence_deg", float("nan")))
    fig.suptitle(f"{title}  |  score {sco:.3f}  |  incidence {inc:.1f} deg", fontsize=11)
    if out_png:
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def transparency_layers(tmap, placement=None, name="transparency"):
    """napari LayerData tuples for a transparency map (and optionally its placement):
    the surface coupling as a Points layer, plus window / target markers. Shared by
    :func:`view_napari`, the napari plugin reader, and the gallery widget."""
    P, _ = _surface_points(tmap)
    coup = np.asarray(tmap.Ipk_Wcm2, float)
    layers = [(P, dict(name=name, features={"coupling": coup}, face_color="coupling",
                       face_colormap="viridis", size=2.0), "points")]
    if placement is not None:
        win = np.asarray(placement.window_center_mni_mm, float)[None, :]
        tgt = np.asarray(placement.target_mni_mm, float)[None, :]
        layers.append((win, dict(name="window", face_color="red", size=6.0), "points"))
        layers.append((tgt, dict(name="target", face_color="white", size=5.0, symbol="x"),
                       "points"))
    return layers


def view_napari(tmap, placement):
    """Open an interactive napari window with the transparency map and the chosen
    placement. Desktop only (needs a display + the ``[viz]`` extra)."""
    try:
        import napari
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise ImportError("the interactive viewer needs napari: pip install -e '.[viz]'") from e

    v = napari.Viewer(title="skull-transparency placement")
    for data, kwargs, _kind in transparency_layers(tmap, placement):
        v.add_points(data, **kwargs)
    v.dims.ndisplay = 3          # every layer here is a 3-D point cloud; 2-D opens on one slice
    napari.run()
    return v
