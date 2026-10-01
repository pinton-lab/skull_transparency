"""Draw a per-patch surface map as a solid, coloured skull surface.

The surface patches sit on the voxel lattice. Scattered as dots they leave gaps that line up
along the lattice terraces, and the white background showing through them reads as rings of
staircasing on the skull, whatever the values are. Drawn as a mesh, the surface is closed.

The mesh is built from the patches alone, no sound-speed volume needed: the patch voxels are
blurred into a thin slab, marching cubes wraps it, and the side of the slab facing away from
the target (the scalp side) is kept. Each vertex takes a gaussian-weighted mean of the nearby
patch values. Needs scikit-image (``marching_cubes``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


@dataclass
class PatchSurface:
    verts: np.ndarray        # (V,3) f8  vertex positions, full-res voxel coordinates
    faces: np.ndarray        # (F,3) i8  triangles (outer side of the patch slab only)
    vert_values: np.ndarray  # (V,)  f8  the patch values, splatted onto the vertices
    face_rhat: np.ndarray    # (F,3) f8  radial direction (target -> patch) under each face

    def face_values(self) -> np.ndarray:
        return self.vert_values[self.faces].mean(1)


def _rhat_in_frame(rhat, to_world):
    """Directions carried into the plotting frame. ``to_world`` is affine, so its linear part
    maps directions; without this a mirrored registration (voxel i running Left) would cull the
    side of the skull facing the camera and show the far side's inside."""
    if to_world is None:
        return rhat
    o = np.asarray(to_world(np.zeros((1, 3))), float)
    L = np.asarray(to_world(np.eye(3)), float) - o                 # row k = image of voxel axis k
    d = rhat @ L
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def patch_surface(surf_vox, rhat, values, *, sigma_vox=1.0, bin_vox=None, max_faces=300_000,
                  k=8):
    """Mesh the outer side of the patch sheet and carry ``values`` onto it.

    ``bin_vox`` coarsens the mesh: patches are binned into cubes of that many voxels BEFORE
    the blur, so the slab stays a few cells thick at any coarseness (striding marching cubes
    over a fine slab instead tears it into rings). By default the finest bin whose mesh stays
    under about ``max_faces`` triangles -- matplotlib's 3-D renderer sorts and paints every
    face, so a full-resolution whole-skull mesh would take minutes per view."""
    from skimage.measure import marching_cubes

    P = np.asarray(surf_vox, float)
    R = np.asarray(rhat, float)
    v = np.asarray(values, float)
    if bin_vox is None:                                      # ~2 faces per occupied bin per side
        bin_vox = max(1, int(np.ceil(np.sqrt(4.0 * len(P) / max_faces))))
    b = float(bin_vox)
    iv = np.floor(P / b + 0.5).astype(np.int64)
    pad = int(np.ceil(3 * sigma_vox)) + 2
    off = iv.min(0) - pad
    iv = iv - off
    occ = np.zeros(tuple(iv.max(0) + pad + 1), np.float32)
    occ[tuple(iv.T)] = 1.0
    occ = gaussian_filter(occ, sigma_vox)
    level = 0.5 / (np.sqrt(2.0 * np.pi) * sigma_vox)        # half the peak of a blurred flat sheet
    V, F, _, _ = marching_cubes(occ, level=level)
    V = (V + off) * b                                        # back to full-res voxels

    s = sigma_vox * b                                        # the value splat, in full-res voxels
    tree = cKDTree(P)
    d, j = tree.query(V, k=min(k, len(P)))
    d, j = np.atleast_2d(d.T).T, np.atleast_2d(j.T).T
    w = np.exp(-0.5 * (d / s) ** 2) + 1e-12
    vv = (w * v[j]).sum(1) / w.sum(1)

    C = V[F].mean(1)
    _, jf = tree.query(C)
    outward = np.einsum("ij,ij->i", C - P[jf], R[jf]) > 0.0   # scalp side of the slab
    return PatchSurface(verts=V, faces=F[outward], vert_values=vv, face_rhat=R[jf][outward])


def draw_patch_surface(ax, surf: PatchSurface, *, cam, cmap="inferno", vmin=None, vmax=None,
                       to_world=None, face_values=None):
    """Add the camera-facing part of ``surf`` to a 3-D axes; returns a ScalarMappable.

    ``cam`` is the unit vector from the scene toward the camera; faces whose patch looks away
    from it are dropped (matplotlib has no occlusion culling, so the far side would otherwise
    paint through). ``to_world`` maps (N,3) voxel coordinates to plotting coordinates."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fv = surf.face_values() if face_values is None else np.asarray(face_values, float)
    vmin = np.nanmin(fv) if vmin is None else vmin
    vmax = np.nanmax(fv) if vmax is None else vmax
    keep = _rhat_in_frame(surf.face_rhat, to_world) @ np.asarray(cam, float) > 0.0
    V = surf.verts if to_world is None else np.asarray(to_world(surf.verts), float)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap)(norm(fv[keep]))
    # edges painted in the face colour, unantialiased: otherwise every triangle gets a hairline
    # seam and the mesh reads as a lattice of rings, the artefact this module exists to remove
    pc = Poly3DCollection(V[surf.faces[keep]], facecolors=rgba, edgecolors=rgba, linewidths=0.3,
                          antialiased=False)
    ax.add_collection3d(pc)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm


def draw_patch_surface_2d(ax, surf: PatchSurface, h, v, *, cmap="inferno", vmin=None, vmax=None,
                          to_world=None, face_values=None):
    """Orthographic projection of ``surf`` onto world axes ``(h, v)`` of a 2-D axes, seen from
    the + side of the remaining axis. Every face is painted, far to near, so the near surface
    covers the far one and the silhouette is complete (culling by the radial direction would
    trim grazing faces at the outline). Returns a ScalarMappable for the colour bar."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    depth = ({0, 1, 2} - {h, v}).pop()
    fv = surf.face_values() if face_values is None else np.asarray(face_values, float)
    vmin = np.nanmin(fv) if vmin is None else vmin
    vmax = np.nanmax(fv) if vmax is None else vmax
    V = surf.verts if to_world is None else np.asarray(to_world(surf.verts), float)
    keep = np.arange(len(surf.faces))
    T = V[surf.faces[keep]]
    order = np.argsort(T[:, :, depth].mean(1))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap)(norm(fv[keep][order]))
    pc = PolyCollection(T[order][:, :, [h, v]], facecolors=rgba, edgecolors=rgba, linewidths=0.3,
                        antialiased=False, rasterized=True)
    ax.add_collection(pc)
    ax.autoscale_view()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return sm
