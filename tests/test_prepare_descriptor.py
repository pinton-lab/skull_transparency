"""write_run_descriptor is the complete part of the producer front-end; it must
round-trip without the (still-scaffolded) resample/pose/recording steps and with no
/celerina data, GPU, or tuba."""
import json

import numpy as np
import pytest

from skull_transparency import TransducerSpec, Registration
from skull_transparency.sim.prepare import Pose, write_run_descriptor


def _pose(N=648, target_vox=(360.0, 190.0, 217.0)):
    # an explicit, hand-built pose (the contract _choose_pose will produce): here a
    # pure 45-deg roll about x, like the Halle M matrix, as a stand-in rotation.
    th = np.deg2rad(45.0)
    R = np.array([[1, 0, 0],
                  [0, np.cos(th), -np.sin(th)],
                  [0, np.sin(th),  np.cos(th)]])
    return Pose(R_phys_to_grid=R, target_grid_vox=np.asarray(target_vox, float), N=N)


def test_meta_has_grid_physics_and_files(tmp_path):
    spec = TransducerSpec.ctx500(f0_hz=500e3, ppw=6.0)
    meta = write_run_descriptor(tmp_path, spec, _pose(), target_phys_mm=[-12.0, -57.0, -34.0],
                                input_frame="ras_mm", n_array=1599)
    on_disk = json.loads((tmp_path / "meta.json").read_text())
    assert on_disk == meta
    # grid + physics from the spec
    assert on_disk["N"] == 648
    assert np.isclose(on_disk["dX_m"], spec.dx_m)
    assert on_disk["C0"] == 1540.0 and on_disk["F0"] == 500e3 and on_disk["ppw"] == 6.0
    # producer-owned fields the generalised launcher reads
    assert on_disk["c_file"] == "c.f32"
    assert on_disk["rho_file"] is None and on_disk["alpha_file"] is None
    assert on_disk["dent_grid"] == [360.0, 190.0, 217.0]
    assert on_disk["n_array"] == 1599
    assert on_disk["array_center"] == [324.0, 324.0, 324.0]
    assert on_disk["input_frame"] == "ras_mm"


def test_supplying_alpha_auto_enables_attenuation(tmp_path):
    spec = TransducerSpec.ctx500()
    meta = write_run_descriptor(tmp_path, spec, _pose(), target_phys_mm=[0, 0, 0],
                                input_frame="lps_mm", n_array=1000,
                                rho_file="rho.f32", alpha_file="alpha.f32")
    assert meta["attenuation"] is True          # alpha present -> attenuation on
    assert meta["alpha_units"] == "db_mhz_cm"
    assert meta["rho_file"] == "rho.f32" and meta["alpha_file"] == "alpha.f32"


def test_no_alpha_leaves_attenuation_off(tmp_path):
    # preserves the current default (Halle bit-identity): no alpha map -> attenuation off
    meta = write_run_descriptor(tmp_path, TransducerSpec.ctx500(), _pose(),
                                target_phys_mm=[0, 0, 0], input_frame="ras_mm", n_array=10)
    assert meta["attenuation"] is False


def test_registration_round_trips_target(tmp_path):
    spec = TransducerSpec.ctx500(f0_hz=500e3, ppw=6.0)
    pose = _pose()
    tgt_mm = np.array([-12.0, -57.0, -34.0])
    write_run_descriptor(tmp_path, spec, pose, target_phys_mm=tgt_mm,
                         input_frame="ras_mm", n_array=1599)
    reg = Registration.from_json(tmp_path / "registration.json")
    # the world target maps to the grid voxel where the outward source sits
    assert np.allclose(reg.mni_to_fullres(tgt_mm), pose.target_grid_vox, atol=1e-6)
    # and back again
    assert np.allclose(reg.fullres_to_mni(pose.target_grid_vox), tgt_mm, atol=1e-6)
    # the clean map is rigid: voxel pitch is isotropic dx_mm and R is orthonormal
    assert np.isclose(reg.dx_mm, spec.dx_mm)
    assert np.allclose(reg.R_mni_to_sim @ reg.R_mni_to_sim.T, np.eye(3), atol=1e-9)


def test_clean_registration_has_no_legacy_affine(tmp_path):
    # the new producer emits the clean rigid map directly; no Amn/bmn/dds/scale detour
    write_run_descriptor(tmp_path, TransducerSpec.ctx500(), _pose(),
                         target_phys_mm=[0, 0, 0], input_frame="ras_mm", n_array=10)
    reg = json.loads((tmp_path / "registration.json").read_text())
    assert reg["rigid"] is True
    assert "deprecated_affine" not in reg


def test_auto_approach_still_scaffolded():
    # every producer step is implemented except the approach='auto' outward-normal aim
    from skull_transparency.sim.prepare import _choose_pose
    with pytest.raises(NotImplementedError):
        _choose_pose(None, np.eye(4), np.zeros(3), "auto", TransducerSpec.ctx500(), 20.0)
