import numpy as np
import pytest

import skull_transparency as st
from skull_transparency.surface import bone_occupancy, smooth_on_surface, true_normals


def test_bone_occupancy_ramp():
    c = np.array([1500.0, 1540.0, 1870.0, 2200.0, 2860.0, 3200.0])
    occ = bone_occupancy(c, bone_threshold=2200.0, c_water=1540.0)
    assert np.allclose(occ, [0.0, 0.0, 0.25, 0.5, 1.0, 1.0])
    # a threshold at or below water cannot ramp: the hard cut
    assert np.array_equal(bone_occupancy(c, 1500.0, 1540.0), (c > 1500.0).astype(np.float32))


def _pv_shell(N=64, r0=20.0, r1=26.0):
    """Partial-volume spherical shell (analytic 1-voxel ramp on both surfaces)."""
    g = np.arange(N) - (N - 1) / 2.0
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    f = np.clip(r1 + 0.5 - r, 0, 1) - np.clip(r0 + 0.5 - r, 0, 1)
    return (1540.0 + f * 1360.0).astype(np.float32), (N - 1) / 2.0


def test_normals_on_partial_volume_sphere_are_radial():
    c, ctr = _pv_shell()
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = ctr + 26.0 * d                                     # on the outer surface
    ang = lambda n: np.degrees(np.arccos(np.clip((n * d).sum(1), -1, 1)))
    soft = ang(true_normals(c, pts, 2200.0, 1.0))
    # the old hard-threshold occupancy, for comparison
    from scipy.ndimage import gaussian_filter, map_coordinates
    occ = gaussian_filter((c > 2200.0).astype(np.float32), 1.0)
    g = np.stack([map_coordinates(occ, (pts + e).T, order=1) - map_coordinates(occ, (pts - e).T, order=1)
                  for e in np.eye(3)], 1)
    hard = ang(-g / np.linalg.norm(g, axis=1, keepdims=True))
    assert np.median(soft) < 2.0 and np.percentile(soft, 90) < 3.0
    assert np.median(soft) < 0.75 * np.median(hard)


def test_smooth_on_surface():
    rng = np.random.default_rng(1)
    pts = np.argwhere(np.ones((20, 20, 1), bool)).astype(float)   # a flat 20x20 sheet
    assert np.allclose(smooth_on_surface(pts, np.full(len(pts), 3.0), 2.0), 3.0)
    v = rng.normal(size=len(pts))
    assert np.array_equal(smooth_on_surface(pts, v, 0), v)
    assert smooth_on_surface(pts, v, 2.0).std() < 0.4 * v.std()


def test_transparency_smooth_mm(synthetic_bundle):
    raw = st.compute_transparency_map(synthetic_bundle, st.TransparencyOptions())
    sm = st.compute_transparency_map(synthetic_bundle, st.TransparencyOptions(smooth_mm=10.0))
    assert raw.meta["smooth_mm"] == 0.0 and sm.meta["smooth_mm"] == 10.0
    assert np.array_equal(raw.surf_vox, sm.surf_vox)
    assert not np.allclose(raw.Pmax, sm.Pmax)
    cv = lambda x: np.std(x) / np.mean(x)
    assert cv(sm.Pmax) < cv(raw.Pmax)


def test_patch_surface_mesh(synthetic_bundle):
    pytest.importorskip("skimage")
    from skull_transparency.surface_render import patch_surface
    t = st.compute_transparency_map(synthetic_bundle, st.TransparencyOptions())
    s = patch_surface(t.surf_vox, t.rhat, t.value)
    assert len(s.faces) > 100
    assert s.faces.max() < len(s.verts) and len(s.vert_values) == len(s.verts)
    # kept faces are the scalp side: farther from the target than the patch layer
    tgt = np.asarray(st.load_bundle(synthetic_bundle).target["fullres_voxel"], float)
    rc = np.linalg.norm(s.verts[s.faces].mean(1) - tgt, axis=1)
    rp = np.linalg.norm(t.surf_vox - tgt, axis=1)
    assert np.median(rc) > np.median(rp)
