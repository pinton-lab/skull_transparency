"""Tests for the brain-center capability: the intracranial/atlas center, the centered
omnidirectional pose, and the end-to-end brain-center sim-tree builder. Pure synthetic --
no /celerina data, GPU, or tuba."""
import json

import numpy as np
import pytest

from skull_transparency import (TransducerSpec, Registration, MNI_BRAIN_COM_MM,
                                 intracranial_centroid, cavity_mask_centroid,
                                 brain_center_phys_mm, brain_center_from_registration)
from skull_transparency.sim.prepare import _choose_pose_centered, build_brain_center_run


def _shell(N=90, center=(45, 40, 50), r_in=30, r_out=36):
    """A closed spherical bone shell (c=2900) in water (c=1540)."""
    ii, jj, kk = np.meshgrid(*[np.arange(N)] * 3, indexing="ij")
    c0 = np.asarray(center, float)
    r = np.sqrt((ii - c0[0])**2 + (jj - c0[1])**2 + (kk - c0[2])**2)
    return np.where((r >= r_in) & (r <= r_out), 2900.0, 1540.0).astype(np.float32)


def test_intracranial_centroid_on_closed_shell():
    c = _shell(center=(45, 40, 50))
    cen = intracranial_centroid(c)
    assert np.linalg.norm(cen - np.array([45, 40, 50])) < 2.0      # 3D fill nails it


def test_intracranial_centroid_no_bone_raises():
    with pytest.raises(ValueError):
        intracranial_centroid(np.full((20, 20, 20), 1540.0, np.float32))


def test_brain_center_mni_frame_returns_atlas_com_verbatim():
    c = _shell()
    # any affine: the MNI branch ignores the image and returns the atlas CoM
    com = brain_center_phys_mm(c, np.eye(4), world_frame="mni_ras_mm")
    assert np.allclose(com, MNI_BRAIN_COM_MM)
    assert np.allclose(brain_center_phys_mm(c, np.eye(4), world_frame="MNI152"), MNI_BRAIN_COM_MM)


def test_brain_center_nonmni_maps_intracranial_centroid_through_affine():
    c = _shell(center=(45, 40, 50))
    affine = np.diag([0.5, 0.5, 0.5, 1.0]); affine[:3, 3] = [10.0, -3.0, 7.0]
    com = brain_center_phys_mm(c, affine, world_frame="subject_ras_mm")
    expect = (affine @ np.array([45, 40, 50, 1.0]))[:3]
    assert np.linalg.norm(com - expect) < 1.5                     # within ~1.5 mm of the true center


def test_cavity_mask_centroid_voxel_and_world():
    # a curated cavity blob: its centroid is exact in voxel space and maps through an affine
    m = np.zeros((40, 40, 40), bool)
    m[10:21, 14:25, 18:29] = True            # centroid (15, 19, 23)
    v = cavity_mask_centroid(m)
    assert np.allclose(v, [15.0, 19.0, 23.0])
    affine = np.diag([0.2, 0.2, 0.2, 1.0]); affine[:3, 3] = [1.0, -2.0, 3.0]
    w = cavity_mask_centroid(m, affine)
    assert np.allclose(w, (affine @ np.array([15.0, 19.0, 23.0, 1.0]))[:3])


def test_cavity_mask_centroid_takes_largest_component():
    # a stray speck must not bias the center toward it
    m = np.zeros((40, 40, 40), bool)
    m[10:21, 14:25, 18:29] = True            # main blob, centroid (15,19,23)
    m[0, 0, 0] = True                        # speck
    assert np.allclose(cavity_mask_centroid(m), [15.0, 19.0, 23.0])


def test_cavity_mask_centroid_empty_raises():
    with pytest.raises(ValueError):
        cavity_mask_centroid(np.zeros((8, 8, 8), bool))


def test_build_brain_center_run_threads_bone_threshold(tmp_path):
    # a sub-2200 bone shell (c=2000) must still resolve its recording surface when the
    # run's bone_threshold is lowered -- guards the _recording_surface threading fix
    # (the default 2200 would find no surface in this medium and raise).
    spec = TransducerSpec.ctx500(f0_hz=1e6, ppw=6.0)
    c = np.where(_shell(N=90, center=(45, 40, 50), r_in=30, r_out=36) > 1540.0,
                 2000.0, 1540.0).astype(np.float32)            # thin/low-speed bone
    out = build_brain_center_run(c, np.diag([0.5, 0.5, 0.5, 1.0]), spec, tmp_path / "bc",
                                 input_frame="subject_ras_mm", surround_mm=8.0,
                                 bone_threshold=1800.0)
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_array"] > 0                                 # surface found at the lowered cutoff
    with pytest.raises(ValueError):                            # default cutoff sees no bone here
        build_brain_center_run(c, np.diag([0.5, 0.5, 0.5, 1.0]), spec, tmp_path / "bc2",
                               input_frame="subject_ras_mm", surround_mm=8.0)


def test_brain_center_from_registration_maps_mni_com():
    reg = Registration(R_mni_to_sim=np.eye(3), dx_mm=0.5,
                       target_mni_mm=np.zeros(3), target_fullres_voxel=np.array([100.0, 100, 100]))
    v = brain_center_from_registration(reg)
    assert np.allclose(v, reg.mni_to_fullres(np.asarray(MNI_BRAIN_COM_MM, float)))


def test_centered_pose_source_at_cube_center_identity_rotation():
    spec = TransducerSpec.ctx500(f0_hz=1e6, ppw=6.0)
    c = _shell(N=90, center=(45, 40, 50))
    center_world = (np.eye(4) @ np.array([45, 40, 50, 1.0]))[:3]
    pose = _choose_pose_centered(c, np.eye(4), center_world, spec, surround_mm=10.0)
    assert np.allclose(pose.R_phys_to_grid, np.eye(3))            # omnidirectional: no rotation
    assert np.allclose(pose.target_grid_vox, pose.N / 2.0)        # source at the cube center
    assert pose.N % 2 == 0


def test_centered_pose_contains_whole_head_and_is_registration_consistent():
    spec = TransducerSpec.ctx500(f0_hz=1e6, ppw=6.0)
    c = _shell(N=90, center=(45, 40, 50), r_in=30, r_out=36)
    affine = np.diag([0.6, 0.6, 0.6, 1.0])
    center_world = (affine @ np.array([45, 40, 50, 1.0]))[:3]
    pose = _choose_pose_centered(c, affine, center_world, spec, surround_mm=12.0)
    reg = Registration(pose.R_phys_to_grid, spec.dx_mm, pose.target_phys_mm, pose.target_grid_vox)
    # the center lands on the cube center
    assert np.allclose(reg.mni_to_fullres(center_world), pose.target_grid_vox, atol=1e-6)
    # the farthest bone (radius 36 vox * 0.6 mm = 21.6 mm) + 12 mm surround must fit in the cube
    half_mm = pose.N / 2.0 * spec.dx_mm
    assert half_mm >= 36 * 0.6 + 12 - 1e-6
    # a point L mm out along +x maps to +L/dx along grid x (identity rotation)
    L = 15.0
    p = center_world + np.array([L, 0, 0])
    assert np.allclose(reg.mni_to_fullres(p), pose.target_grid_vox + [L / spec.dx_mm, 0, 0], atol=1e-6)


def test_build_brain_center_run_writes_centered_sim_tree(tmp_path):
    spec = TransducerSpec.ctx500(f0_hz=1e6, ppw=6.0)
    c = _shell(N=90, center=(45, 40, 50))
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    out = build_brain_center_run(c, affine, spec, tmp_path / "bc", input_frame="subject_ras_mm",
                                 surround_mm=10.0)
    files = {p.name for p in out.iterdir()}
    assert {"c.f32", "array_coords.i32", "meta.json", "registration.json"} <= files
    meta = json.loads((out / "meta.json").read_text())
    N = meta["N"]
    assert np.allclose(meta["dent_grid"], [N / 2.0] * 3)         # source at the cube center
    assert meta["n_array"] > 0                                   # whole-skull recording shell
    # registration anchors at the source (the brain center), non-MNI world frame preserved
    reg = Registration.from_json(out / "registration.json")
    assert np.allclose(reg.target_fullres_voxel, [N / 2.0] * 3)
    assert reg.world_frame == "subject_ras_mm"
