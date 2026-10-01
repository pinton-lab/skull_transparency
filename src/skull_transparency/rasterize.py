"""Closed surface meshes -> partial-volume voxel maps, on the solver's own grid.

A skull supplied as surface meshes (the ITRUSST benchmark STLs, a segmentation export) has to
become a sound-speed volume before it can be simulated. Filling each mesh into a binary mask
(``trimesh``'s ``voxelized().fill()``, the benchmark's own ``surf2vol``) puts every bone edge
on a whole voxel, and the solver then sees the boundary as a staircase. Every stair edge
scatters a little, and two millimetres out the overlapping scatter becomes a speckle in the
transparency map that is about half voxel-staircase artefact: rotate the skull against the
grid before filling, and roughly half of the fine (< 3 mm) texture changes.

Here each voxel straddling the surface gets the fraction of it that lies inside, from its
signed distance to the plane of the nearest mesh face (exact for a boundary that is flat
across one voxel), so the boundary sits where the mesh says rather than on the lattice.
Rasterise at the solver pitch itself: a volume rasterised at one pitch and resampled onto a
slightly different solver grid (0.500 mm vs 0.513 mm, say) blurs its edges unevenly, in a
beat pattern tens of millimetres long.

Only numpy and scipy are needed; the meshes come in as plain ``(vertices, faces)`` arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


def _face_samples(V, F, spacing):
    """Points on every face, no farther than ``spacing`` apart, each tagged with its face."""
    T = V[F]                                                       # (nF, 3, 3)
    edge = np.linalg.norm(T - np.roll(T, 1, axis=1), axis=2).max(1)
    n = np.maximum(1, np.ceil(edge / spacing).astype(int))         # subdivisions per edge
    pts, owner = [], []
    for k in np.unique(n):                                         # barycentric grid, per level
        f = np.flatnonzero(n == k)
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        a, b = i[keep] / k, j[keep] / k
        w = np.stack([1.0 - a - b, a, b], 1)                       # (m, 3)
        pts.append(np.einsum("mk,fkd->fmd", w, T[f]).reshape(-1, 3))
        owner.append(np.repeat(f, len(w)))
    return np.concatenate(pts), np.concatenate(owner)


def _inside_centres(T, lo, h, shape):
    """Exact inside/outside of every voxel centre by crossing parity along z.

    Each triangle is intersected with the vertical lines through the voxel-centre columns it
    covers; a centre is inside when an odd number of crossings lie below it. Parity, unlike a
    hole fill, keeps cavities nested inside a body (a sinus shell inside the skull surface)
    empty. The columns are nudged by an irrational sub-nanometre offset so no line passes
    exactly through a shared edge or vertex and gets counted twice."""
    eps = h * 1e-7 * np.array([np.sqrt(2.0), np.sqrt(3.0)])
    xy = T[:, :, :2]
    i0 = np.ceil((xy.min(1) - lo[:2] - eps) / h - 0.5).astype(np.int64)    # first column centre
    i1 = np.floor((xy.max(1) - lo[:2] - eps) / h - 0.5).astype(np.int64)   # last column centre
    nx, ny = i1[:, 0] - i0[:, 0] + 1, i1[:, 1] - i0[:, 1] + 1
    has = (nx > 0) & (ny > 0)
    tri = np.flatnonzero(has)
    cnt = (nx * ny)[tri]
    t = np.repeat(tri, cnt)                                            # (triangle, column) pairs
    k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    cx = i0[t, 0] + k % nx[t]
    cy = i0[t, 1] + k // nx[t]
    p = lo[:2] + (np.c_[cx, cy] + 0.5) * h + eps
    a, b, c = T[t, 0], T[t, 1], T[t, 2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    ok = np.abs(det) > 1e-18                                           # edge-on triangles: no crossing
    w1 = ((p[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1]))
    w2 = ((b[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1]) - (p[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    safe = np.where(ok, det, 1.0)
    w1, w2 = np.where(ok, w1 / safe, -1.0), np.where(ok, w2 / safe, -1.0)
    hit = ok & (w1 >= 0) & (w2 >= 0) & (w1 + w2 <= 1)
    z = (1 - w1 - w2) * a[:, 2] + w1 * b[:, 2] + w2 * c[:, 2]
    cx, cy, z = cx[hit], cy[hit], z[hit]
    kz = np.ceil((z - lo[2]) / h - 0.5).astype(np.int64)               # first centre ABOVE the crossing
    keep = (cx >= 0) & (cx < shape[0]) & (cy >= 0) & (cy < shape[1])
    cx, cy, kz = cx[keep], cy[keep], np.clip(kz[keep], 0, shape[2])
    flips = np.zeros((shape[0], shape[1], shape[2] + 1), np.int32)
    np.add.at(flips, (cx, cy, kz), 1)
    return (np.cumsum(flips[:, :, :-1], axis=2) & 1).astype(bool)


def _drop_cavities(V, F):
    """Faces of the outward-wound bodies only: a body whose signed volume is negative is a
    cavity shell (a sinus or air cell inside the skull surface) and is removed."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    _, lab = connected_components(g, directed=False)
    body = lab[F[:, 0]]
    T = V[F]
    vol6 = np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2]))   # 6 x signed volume per face
    keep_body = np.bincount(body, weights=vol6) > 0
    return F[keep_body[body]]


def mesh_fraction(vertices, faces, pitch_mm, lo_mm, shape, *, fill_cavities=False):
    """Inside fraction of each voxel of a closed (watertight) triangle mesh.

    The grid is axis-aligned in the mesh frame: voxel ``i`` spans
    ``lo_mm + pitch_mm * [i, i + 1)`` and its centre is ``lo_mm + pitch_mm * (i + 0.5)``.
    Returns float32 ``shape``: 1 deep inside, 0 outside, the partial volume on the surface.
    Which side a voxel centre is on comes from crossing parity (exact, cavity-safe); how far
    inside, for voxels the surface passes near, from the plane of the nearest face.

    ``fill_cavities`` drops inward-wound shells nested inside the body (sinuses, mastoid air
    cells) so they count as inside, which is what a hole-filling rasteriser
    (``trimesh``'s ``voxelized().fill()``, iso2mesh ``surf2vol(..., 'fill')``) does.
    """
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    if fill_cavities:
        F = _drop_cavities(V, F)
    lo = np.asarray(lo_mm, float)
    shape = tuple(int(s) for s in shape)
    h = float(pitch_mm)

    T = V[F]
    inside = _inside_centres(T, lo, h, shape)
    nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-30)

    S, owner = _face_samples(V, F, 0.5 * h)
    seed = np.zeros(shape, bool)
    iv = np.floor((S - lo) / h).astype(np.int64)
    ok = np.all((iv >= 0) & (iv < shape), axis=1)
    seed[tuple(iv[ok].T)] = True
    band = binary_dilation(seed, iterations=1)                     # voxels the surface can cut

    bi = np.argwhere(band)
    ctr = lo + (bi + 0.5) * h
    _, j = cKDTree(S).query(ctr)
    f = owner[j]
    # |distance to that face's plane|, signed by parity -- the nearest face's own winding can
    # point the wrong way at a fold, parity cannot.
    d = np.abs(np.einsum("ij,ij->i", ctr - T[f, 0], nrm[f]))
    sgn = np.where(inside[tuple(bi.T)], 1.0, -1.0)
    out = inside.astype(np.float32)
    out[tuple(bi.T)] = np.clip(0.5 + sgn * d / h, 0.0, 1.0)       # plane cut through the voxel
    return out


def shell_speed_map(outer, inner, pitch_mm, *, c_tissue=1500.0, c_bone=2800.0, pad_mm=None,
                    fill_cavities=True):
    """Sound-speed volume of a bone shell between two closed meshes, with partial-volume edges.

    ``outer`` / ``inner`` are ``(vertices, faces)`` of the outer and inner bone surfaces (e.g.
    the ITRUSST ``skull_outer.stl`` / ``skull_inner.stl``). Each voxel gets
    ``c_tissue + f * (c_bone - c_tissue)`` with ``f`` its bone fraction. Pass the SOLVER pitch
    (``TransducerSpec.dx_mm``) so ``prepare`` needs no resampling.

    ``fill_cavities=True`` (default) treats air cavities enclosed in the outer mesh -- the
    ITRUSST outer STL carries six, ~48 cm^3, in the sinus and mastoid regions -- as bone, as the benchmark's own ``surf2vol(..., 'fill')`` does.
    ``False`` leaves them as tissue (``c_tissue``); air itself is not modelled either way.

    Returns ``(c, affine)``: float32 ``c`` and the 4x4 voxel-index -> mm affine of voxel
    centres, ready for ``skull-transparency prepare --c-map ... --affine ...``.
    """
    h = float(pitch_mm)
    pad = 2.0 * h if pad_mm is None else float(pad_mm)
    Vo = np.asarray(outer[0], float)
    lo = Vo.min(0) - pad
    shape = tuple(np.ceil((Vo.max(0) + pad - lo) / h).astype(int))
    fo = mesh_fraction(outer[0], outer[1], h, lo, shape, fill_cavities=fill_cavities)
    fi = mesh_fraction(inner[0], inner[1], h, lo, shape, fill_cavities=fill_cavities)
    frac = np.clip(fo - fi, 0.0, 1.0)
    c = (c_tissue + frac * (c_bone - c_tissue)).astype(np.float32)
    affine = np.diag([h, h, h, 1.0])
    affine[:3, 3] = lo + 0.5 * h                                   # index -> voxel CENTRE
    return c, affine
