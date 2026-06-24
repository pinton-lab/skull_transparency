"""Golden tests for prepare._resample_to_grid — purely synthetic, no /celerina data,
GPU, or tuba. Exploits the fact that trilinear interpolation reproduces a field that
is linear in voxel index *exactly*, so the resampled grid can be predicted in closed
form even with a rotated pose, anisotropic voxels, and a translated frame."""
import numpy as np

from skull_transparency.sim.prepare import Pose, _resample_to_grid


def _world(vox, affine):
    return vox @ affine[:3, :3].T + affine[:3, 3]


def test_linear_field_is_reproduced_exactly():
    # anisotropic, translated input frame (voxel -> world mm)
    affine = np.array([[0.8, 0.0, 0.0, -10.0],
                       [0.0, 1.2, 0.0,   5.0],
                       [0.0, 0.0, 0.6,   2.0],
                       [0.0, 0.0, 0.0,   1.0]])
    nx, ny, nz = 50, 52, 54
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    vox = np.stack([ii, jj, kk], axis=-1).astype(float)

    # a field that is linear in WORLD position: f(p) = q.p + s
    q = np.array([0.3, -0.2, 0.5])
    s = 7.0
    vol = (_world(vox, affine) @ q + s).astype(np.float32)

    # rotated pose, sub-mm pitch, anchored at the world position of the vol centre
    th = np.deg2rad(20.0)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th),  np.cos(th), 0.0],
                  [0.0,         0.0,        1.0]])   # world-mm -> grid-mm
    N = 10
    dx_m = 0.4e-3                                     # dx_mm = 0.4
    tvox = np.array([N / 2.0] * 3)
    tmm = _world(np.array([nx / 2.0, ny / 2.0, nz / 2.0]), affine)   # interior anchor
    pose = Pose(R_phys_to_grid=R, target_grid_vox=tvox, N=N, target_phys_mm=tmm)

    out = _resample_to_grid(vol, affine, pose, dx_m)

    # closed-form expectation: the world-linear field at each grid point's world position
    gi, gj, gk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    g = np.stack([gi, gj, gk], axis=-1).astype(float)
    dx_mm = dx_m * 1e3
    world_g = tmm + dx_mm * (g - tvox) @ R           # (g-tvox)@R == R^T@(g-tvox) per point
    expected = world_g @ q + s

    assert out.shape == (N, N, N)
    assert out.dtype == np.float32
    assert np.max(np.abs(out - expected)) < 1e-2     # exact up to float32 rounding


def test_target_voxel_samples_the_target_world_point():
    # the outward source sits at target_grid_vox; it must sample vol at target_phys_mm
    affine = np.diag([0.5, 0.5, 0.5, 1.0]); affine[:3, 3] = [3.0, -4.0, 1.0]
    n = 60
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    vol = (np.stack([ii, jj, kk], -1).astype(float) @ np.array([1.0, 2.0, 3.0])).astype(np.float32)
    tvox = np.array([7.0, 7.0, 7.0])
    tmm = _world(np.array([30.0, 28.0, 26.0]), affine)
    pose = Pose(R_phys_to_grid=np.eye(3), target_grid_vox=tvox, N=15, target_phys_mm=tmm)

    out = _resample_to_grid(vol, affine, pose, dx_m=0.5e-3)
    expected_at_target = np.array([30.0, 28.0, 26.0]) @ np.array([1.0, 2.0, 3.0])
    i = tvox.astype(int)
    assert np.isclose(out[i[0], i[1], i[2]], expected_at_target, atol=1e-2)


def test_outside_volume_uses_background_fill():
    # identity frame, identity pose, unit pitch -> grid voxel g maps to input voxel g;
    # with N > vol size the far half falls outside and must take the background value.
    vol = np.full((20, 20, 20), 1500.0, np.float32)
    pose = Pose(R_phys_to_grid=np.eye(3), target_grid_vox=np.zeros(3), N=30,
                target_phys_mm=np.zeros(3))
    out = _resample_to_grid(vol, np.eye(4), pose, dx_m=1e-3, background=777.0)
    assert np.isclose(out[5, 5, 5], 1500.0)          # inside the volume
    assert np.isclose(out[25, 0, 0], 777.0)          # outside -> background
    # a constant medium is preserved everywhere it overlaps the volume
    assert np.allclose(out[:20, :20, :20], 1500.0)


def test_missing_anchor_raises():
    import pytest
    pose = Pose(R_phys_to_grid=np.eye(3), target_grid_vox=np.zeros(3), N=4)  # no target_phys_mm
    with pytest.raises(ValueError):
        _resample_to_grid(np.zeros((4, 4, 4), np.float32), np.eye(4), pose, dx_m=1e-3)
