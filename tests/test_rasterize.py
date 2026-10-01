import numpy as np
import pytest

from skull_transparency.rasterize import mesh_fraction, shell_speed_map


def _sphere(r, n_lat=48, n_lon=96, center=(0.0, 0.0, 0.0), inward=False):
    """Closed UV sphere, outward winding (or inward, for a cavity shell)."""
    th = np.linspace(0, np.pi, n_lat + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    V = np.c_[np.sin(T).ravel() * np.cos(P).ravel(), np.sin(T).ravel() * np.sin(P).ravel(),
              np.cos(T).ravel()]
    V = np.r_[[[0, 0, 1.0]], V, [[0, 0, -1.0]]] * r + np.asarray(center)
    idx = lambda i, j: 1 + i * n_lon + (j % n_lon)
    F = []
    for j in range(n_lon):
        F.append([0, idx(0, j), idx(0, j + 1)])
        F.append([len(V) - 1, idx(n_lat - 2, j + 1), idx(n_lat - 2, j)])
        for i in range(n_lat - 2):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            F += [[a, b, c], [a, c, d]]
    F = np.asarray(F)
    return V, (F[:, ::-1] if inward else F)


def _mesh_volume(V, F):
    T = V[F]
    return np.einsum("ij,ij->i", T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6.0


def test_partial_volume_sphere_matches_mesh_volume():
    V, F = _sphere(10.0)
    h = 0.7                                        # deliberately not a divisor of anything
    lo = np.full(3, -12.0)
    f = mesh_fraction(V, F, h, lo, (35, 35, 35))
    assert f.min() >= 0.0 and f.max() <= 1.0
    assert abs(f.sum() * h ** 3 / _mesh_volume(V, F) - 1.0) < 0.005     # binary fill: ~5% off
    partial = (f > 0.02) & (f < 0.98)
    assert partial.sum() > 1000                    # the boundary is partial-volume, not stepped
    c = np.round((np.zeros(3) - lo) / h - 0.5).astype(int)
    assert f[tuple(c)] == 1.0 and f[0, 0, 0] == 0.0


def test_boundary_sits_on_the_true_surface():
    """Along any line from the centre, the 0.5 crossing of the fraction is at r within ~h/10."""
    from scipy.ndimage import map_coordinates
    V, F = _sphere(10.0, 64, 128)
    h, lo = 0.5, np.full(3, -12.0)
    f = mesh_fraction(V, F, h, lo, (48, 48, 48))
    rng = np.random.default_rng(0)
    for d in rng.normal(size=(20, 3)):
        d /= np.linalg.norm(d)
        r = np.linspace(8.0, 12.0, 801)
        prof = map_coordinates(f, ((r[:, None] * d - lo) / h - 0.5).T, order=1)
        r50 = r[np.argmin(np.abs(prof - 0.5))]
        assert abs(r50 - 10.0) < 0.06              # vs h/2 = 0.25 for a binary mask


def test_cavity_shell_parity_and_fill():
    Vo, Fo = _sphere(10.0)
    Vc, Fc = _sphere(3.0, center=(2.0, 0.0, 0.0), inward=True)  # an air cell inside the bone
    V = np.r_[Vo, Vc]
    F = np.r_[Fo, Fc + len(Vo)]
    h, lo, shape = 0.5, np.full(3, -11.0), (44, 44, 44)
    empty = mesh_fraction(V, F, h, lo, shape)
    filled = mesh_fraction(V, F, h, lo, shape, fill_cavities=True)
    ic = np.round((np.array([2.0, 0, 0]) - lo) / h - 0.5).astype(int)
    assert empty[tuple(ic)] == 0.0 and filled[tuple(ic)] == 1.0
    cav = 4.0 / 3.0 * np.pi * 27.0
    assert abs((filled.sum() - empty.sum()) * h ** 3 / cav - 1.0) < 0.03


def test_shell_speed_map_affine_is_voxel_centres():
    outer, inner = _sphere(10.0), _sphere(8.0)
    h = 0.5
    c, A = shell_speed_map(outer, inner, h, c_tissue=1500.0, c_bone=2800.0)
    assert c.dtype == np.float32 and A.shape == (4, 4)
    frac = (c - 1500.0) / 1300.0
    vol = 4.0 / 3.0 * np.pi * (10.0 ** 3 - 8.0 ** 3)
    assert abs(frac.sum() * h ** 3 / vol - 1.0) < 0.01
    # the bone-weighted centroid through the affine lands on the sphere centre (no half-voxel shift)
    idx = np.argwhere(frac > 0)
    w = frac[tuple(idx.T)]
    cen = (A[:3, :3] @ (idx * w[:, None]).sum(0) / w.sum()) + A[:3, 3]
    assert np.all(np.abs(cen) < 0.02)


def test_trimesh_cross_check():
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.creation.icosphere(subdivisions=4, radius=9.0)
    h, lo = 0.6, np.full(3, -10.5)
    f = mesh_fraction(m.vertices, m.faces, h, lo, (35, 35, 35))
    assert abs(f.sum() * h ** 3 / m.volume - 1.0) < 0.005
