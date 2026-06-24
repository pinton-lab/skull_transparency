"""Tests for prepare._choose_pose (explicit-aim path) + _rotation_aligning. Pure
geometry — no /celerina data, GPU, or tuba. The pose must be self-consistent with the
same world<->grid map used by the resampler and the descriptor (Registration)."""
import numpy as np
import pytest

from skull_transparency import TransducerSpec, Registration
from skull_transparency.sim.prepare import _choose_pose, _rotation_aligning, _APPROACH_AXIS


def test_rotation_aligning_is_a_proper_rotation_mapping_a_to_b():
    b = np.array([0.0, 0.0, 1.0])
    for a in [np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0]),   # aligned
              np.array([0.0, 0.0, -1.0]),                              # antiparallel
              np.array([1.0, 0.0, 0.0]), np.array([-0.4, 0.7, -0.2])]:
        R = _rotation_aligning(a, b)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)              # orthonormal
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)           # proper (no reflection)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b, atol=1e-9)  # a -> b


def test_approach_maps_to_plus_z():
    spec = TransducerSpec.ctx500(f0_hz=500e3, ppw=6.0)
    approach = np.array([0.3, 0.8, -0.5])
    pose = _choose_pose(None, np.eye(4), np.array([10.0, -5.0, 3.0]), approach, spec, 20.0)
    au = approach / np.linalg.norm(approach)
    assert np.allclose(pose.R_phys_to_grid @ au, _APPROACH_AXIS, atol=1e-9)
    assert np.allclose(pose.R_phys_to_grid @ pose.R_phys_to_grid.T, np.eye(3), atol=1e-9)


def test_pose_is_self_consistent_with_registration():
    # the pose must encode the SAME rigid map Registration uses downstream
    spec = TransducerSpec.ctx500(f0_hz=500e3, ppw=6.0)
    approach = np.array([0.3, 0.8, -0.5])
    target = np.array([10.0, -5.0, 3.0])
    pose = _choose_pose(None, np.eye(4), target, approach, spec, 20.0)
    reg = Registration(pose.R_phys_to_grid, spec.dx_mm, pose.target_phys_mm, pose.target_grid_vox)
    # the target lands on the seated voxel
    assert np.allclose(reg.mni_to_fullres(target), pose.target_grid_vox, atol=1e-9)
    # a point L mm out along the approach aim moves +L/dx purely along grid +Z
    L = 40.0
    p = target + L * approach / np.linalg.norm(approach)
    expect = pose.target_grid_vox + np.array([0.0, 0.0, L / spec.dx_mm])
    assert np.allclose(reg.mni_to_fullres(p), expect, atol=1e-6)


def test_grid_contains_target_bowl_reach_and_aperture():
    spec = TransducerSpec.ctx500(f0_hz=1e6, ppw=5.5)        # tightest pitch -> largest N
    pose = _choose_pose(None, np.eye(4), np.zeros(3), np.array([0.0, 0.0, 1.0]), spec, 20.0)
    N, t, dx = pose.N, pose.target_grid_vox, spec.dx_mm
    assert (t >= 0).all() and (t < N).all()                # target inside the grid
    assert t[2] + spec.roc_mm / dx <= N                    # bowl apex (focus=target) fits in +Z
    assert t[0] - (spec.aperture_mm / 2) / dx >= 0         # aperture fits laterally
    assert t[0] + (spec.aperture_mm / 2) / dx <= N


def test_grid_size_scales_down_with_frequency():
    # lower frequency -> coarser dx -> fewer voxels for the same physical reach
    def N_at(f):
        return _choose_pose(None, np.eye(4), np.zeros(3), np.array([0.0, 0.0, 1.0]),
                            TransducerSpec.ctx500(f0_hz=f, ppw=6.0), 20.0).N
    assert N_at(250e3) < N_at(500e3) < N_at(1e6)


def test_auto_and_bad_approach_raise():
    spec = TransducerSpec.ctx500()
    with pytest.raises(NotImplementedError):
        _choose_pose(None, np.eye(4), np.zeros(3), "auto", spec, 20.0)
    with pytest.raises(NotImplementedError):
        _choose_pose(None, np.eye(4), np.zeros(3), None, spec, 20.0)
    with pytest.raises(ValueError):
        _choose_pose(None, np.eye(4), np.zeros(3), np.zeros(3), spec, 20.0)   # zero aim
    with pytest.raises(ValueError):
        _choose_pose(None, np.eye(4), np.zeros(3), np.array([1.0, 0.0]), spec, 20.0)  # wrong shape
