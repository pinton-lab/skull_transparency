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


def bone_occupancy(c, bone_threshold=2200.0, c_water=1540.0):
    """Soft bone occupancy: 0 at ``c_water``, 1 from ``2*bone_threshold - c_water`` up, linear
    between, so ``bone_threshold`` sits at 0.5.

    A hard ``c > bone_threshold`` cut puts every boundary on a whole voxel and the normals then
    follow the staircase; the ramp keeps a partial-volume voxel's fraction, and saturating above
    the threshold keeps the diploe's internal speed variation out of the surface gradient."""
    c = np.asarray(c, np.float32)
    if bone_threshold <= c_water:                       # no room for a ramp: fall back to the cut
        return (c > bone_threshold).astype(np.float32)
    return np.clip((c - c_water) / (2.0 * (bone_threshold - c_water)), 0.0, 1.0)


def true_normals(c, pts_fullres, bone_threshold=2200.0, smooth=1.0, h=1.0, c_water=1540.0):
    """Outward unit surface normals at ``pts_fullres`` from the bone-occupancy gradient.

    Central differences of the lightly-smoothed :func:`bone_occupancy`, sampled at the
    (sub-voxel) surface points; outward = direction of *decreasing* occupancy. On the ITRUSST
    skull with partial-volume edges this halves the error against the mesh normals compared
    with the hard-threshold occupancy (median 3.4 -> 1.8 deg)."""
    occ = gaussian_filter(bone_occupancy(c, bone_threshold, c_water), smooth)
    pts = np.asarray(pts_fullres, float)

    def s(off):
        return map_coordinates(occ, (pts + np.asarray(off, float)).T, order=1)

    g = np.stack([s([h, 0, 0]) - s([-h, 0, 0]),
                  s([0, h, 0]) - s([0, -h, 0]),
                  s([0, 0, h]) - s([0, 0, -h])], axis=1)
    nrm = np.linalg.norm(g, axis=1, keepdims=True)
    return -g / np.maximum(nrm, 1e-12)


def smooth_on_surface(surf_vox, values, sigma_vox):
    """Gaussian-smooth per-patch ``values`` along the surface, ``sigma_vox`` in grid voxels.

    The patches are voxel positions, so the smooth is a normalised splat: scatter the values
    and their counts onto the grid, blur both, divide, and read back at the patches. Only the
    surface's own patches contribute, so empty space does not dilute the edge of the map."""
    v = np.asarray(values, float)
    if not sigma_vox:
        return v.copy()
    iv = np.rint(np.asarray(surf_vox, float)).astype(np.int64)
    pad = int(np.ceil(3 * sigma_vox))
    iv = iv - iv.min(0) + pad
    shape = tuple(iv.max(0) + pad + 1)
    num = np.zeros(shape, np.float32)
    den = np.zeros(shape, np.float32)
    np.add.at(num, tuple(iv.T), v)
    np.add.at(den, tuple(iv.T), 1.0)
    num = gaussian_filter(num, sigma_vox, mode="constant")
    den = gaussian_filter(den, sigma_vox, mode="constant")
    return (num[tuple(iv.T)] / np.maximum(den[tuple(iv.T)], 1e-12)).astype(float)
