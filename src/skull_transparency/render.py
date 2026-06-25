"""Render a whole-skull transparency map as a static figure.

For a brain-center run the natural picture is the outer calvaria coloured by the
distance-corrected transparency, shown in three anatomical projections (bright =
acoustically transparent bone). The colour is the **amplitude in decibels** -- the
manuscript's whole-skull convention is ``log peak |p|``, NOT linear intensity: linear
``p^2`` has far too wide a dynamic range and crushes most of the skull to black. So we
plot ``20*log10(|p|/ref)`` of the 1/r^2-corrected field (``TransparencyMap.value`` is the
distance-corrected *intensity*; its square root is the amplitude). This is a
*visualisation* of bone transmission, not a placement objective."""
from __future__ import annotations

import numpy as np

#: anatomical projection planes: (name, horizontal axis, vertical axis) on MNI/voxel xyz
_VIEWS = (("sagittal (y-z)", 1, 2), ("coronal (x-z)", 0, 2), ("axial (x-y)", 0, 1))
_LABELS = ("x (L-R)", "y (P-A)", "z (I-S)")


def render_transparency_surface(tmap, out_png, *, title=None, cmap="inferno",
                                clip=(2.0, 98.0), point_size=2.0, use_mni=True, dpi=170,
                                db_floor=-40.0):
    """Write a 3-panel PNG of the skull surface coloured by transparency amplitude in dB.

    The colour is ``20*log10(amplitude / ref)`` with ``ref`` the 98th-percentile patch, so
    0 dB is the most transparent windows and the scale runs down to ``clip[0]``-percentile
    (or ``db_floor``, whichever is higher). ``clip`` sets the lower colour bound; if the map
    carries a registration and ``use_mni`` is set, axes are MNI millimetres, else full-res
    voxels. Each panel sorts points by depth so the near surface draws on top. Returns
    ``out_png``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf_vox = np.asarray(tmap.surf_vox, float)
    val = np.maximum(np.asarray(tmap.value, float), 0.0)     # distance-corrected intensity
    # recover the actual source (the radial reference) from the map's own geometry, so the
    # marker is the brain-center source -- NOT the surface centroid (which sits ~skull base).
    dx_mm = tmap.meta.get("dx_mm") or (tmap.registration.dx_mm if tmap.registration is not None else 1.0)
    src_vox = np.median(surf_vox - np.asarray(tmap.rhat, float)
                        * (np.asarray(tmap.rad_mm, float) / dx_mm)[:, None], axis=0)
    if tmap.registration is not None and use_mni:
        pts = np.asarray(tmap.registration.fullres_to_mni(surf_vox), float)
        center = np.asarray(tmap.registration.fullres_to_mni(src_vox), float)
        # label with the registration's TRUE world frame (a non-MNI subject's own
        # world-mm frame is not MNI, so do not mislabel its axes)
        wf = getattr(tmap.registration, "world_frame", None) or "mni_ras_mm"
        units = "MNI mm" if "mni" in str(wf).lower() else f"world mm ({wf})"
    else:
        pts, center, units = surf_vox, src_vox, "voxel"

    amp = np.sqrt(val)                                        # amplitude = sqrt(intensity)
    ref = np.percentile(amp, 98.0) or (amp.max() or 1.0)
    disp = 20.0 * np.log10(np.maximum(amp / ref, 10.0 ** (db_floor / 20.0)))   # dB re 98th pct
    vlo = max(float(np.percentile(disp, clip[0])), db_floor)
    vhi = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    sc = None
    for ax, (name, h, v) in zip(axes, _VIEWS):
        depth = sorted(set(range(3)) - {h, v})[0]            # the out-of-plane axis
        order = np.argsort(pts[:, depth])                    # far -> near (near drawn last/on top)
        sc = ax.scatter(pts[order, h], pts[order, v], c=disp[order], s=point_size,
                        cmap=cmap, vmin=vlo, vmax=vhi, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel(_LABELS[h]); ax.set_ylabel(_LABELS[v])
        ax.set_title(name, fontsize=10)
        ax.scatter([center[h]], [center[v]], marker="+", c="cyan", s=80,
                   linewidths=1.5, zorder=5)             # brain-center source
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("transparency amplitude  (dB re 98th pct, 1/r$^2$-corrected)")
    fig.suptitle(title or "Whole-skull transparency (brain-center source, + = center)",
                 fontsize=12)
    fig.text(0.01, 0.01, f"{len(pts)} surface patches  |  axes in {units}", fontsize=8,
             color="0.4")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_png
