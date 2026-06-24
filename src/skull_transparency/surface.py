"""External calvarial surface extraction from a speed-of-sound map.

The external surface = bone/tissue boundary voxels whose OUTWARD (away-from-target)
neighbour is tissue and inward neighbour is bone.  We also provide the TRUE local
surface normal (from the smoothed bone-occupancy gradient) -- distinct from the
radial direction (target -> patch), which over-credits obliquely-hit patches."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, map_coordinates


def extract_external_surface(c, target_fullres, bone_threshold=2200.0, probe_vox=2.5):
    """Return (surf_vox (M,3) f8, rhat (M,3) f8) for external-surface voxels.

    ``rhat`` is the RADIAL unit vector (target -> patch); for the true geometric
    normal use :func:`true_normals`.  Matches the legacy extraction in
    ``skull_external_intensity_ppw55.py`` exactly (same erosion / probe / order)."""
    c = np.asarray(c)
    target = np.asarray(target_fullres, float)
    B = c > bone_threshold
    surf = B & ~binary_erosion(B)
    si = np.argwhere(surf).astype(np.float64)
    rhat = si - target
    rhat /= np.linalg.norm(rhat, axis=1, keepdims=True)
    co = map_coordinates(c, (si + probe_vox * rhat).T, order=1)   # speed just OUTWARD
    ci = map_coordinates(c, (si - probe_vox * rhat).T, order=1)   # speed just INWARD
    outer = (co < bone_threshold) & (ci > bone_threshold)
    return si[outer], rhat[outer]


def true_normals(c, pts_fullres, bone_threshold=2200.0, smooth=1.0, h=1.0):
    """Outward unit surface normals at ``pts_fullres`` from the bone-occupancy gradient.

    Central differences of a lightly-smoothed occupancy field, sampled at the
    (sub-voxel) surface points; outward = direction of *decreasing* occupancy."""
    occ = gaussian_filter((np.asarray(c) > bone_threshold).astype(np.float32), smooth)
    pts = np.asarray(pts_fullres, float)

    def s(off):
        return map_coordinates(occ, (pts + np.asarray(off, float)).T, order=1)

    g = np.stack([s([h, 0, 0]) - s([-h, 0, 0]),
                  s([0, h, 0]) - s([0, -h, 0]),
                  s([0, 0, h]) - s([0, 0, -h])], axis=1)
    nrm = np.linalg.norm(g, axis=1, keepdims=True)
    return -g / np.maximum(nrm, 1e-12)
