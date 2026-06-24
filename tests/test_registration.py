import numpy as np

from skull_transparency import Registration


def test_roundtrip_and_anchor(transform_npz):
    reg = Registration.from_ppw55_npz(transform_npz, target_fullres_voxel=[360.0, 190.0, 217.0])
    # anchor: target MNI maps exactly to the target voxel
    assert np.allclose(reg.mni_to_fullres(reg.target_mni_mm), [360.0, 190.0, 217.0])
    # rigid roundtrip (single point and batch)
    pts = np.array([[-12.0, -57.0, -34.0], [5.0, 10.0, 20.0], [-40.0, -30.0, 50.0]])
    assert np.allclose(reg.fullres_to_mni(reg.mni_to_fullres(pts)), pts, atol=1e-9)
    assert np.allclose(reg.fullres_to_mni(reg.mni_to_fullres(pts[0])), pts[0], atol=1e-9)
    # orthonormal; |det| == 1 (det may be -1: MNI RAS <-> LAS-stored sim grid is a reflection)
    R = reg.R_mni_to_sim
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(abs(np.linalg.det(R)), 1.0, atol=1e-6)


def test_reproduces_clean_Rtot(transform_npz):
    t = np.load(transform_npz)
    U, _, Vt = np.linalg.svd(np.asarray(t["Amn"], float))
    Rtot = (np.asarray(t["M"], float).T @ np.asarray(t["Arot"], float).T) @ (U @ Vt)
    reg = Registration.from_ppw55_npz(transform_npz, target_fullres_voxel=[360.0, 190.0, 217.0])
    assert np.allclose(reg.R_mni_to_sim, Rtot, atol=1e-12)
