"""Frame-label correctness for generic (non-MNI) subjects (audit finding C1).

A subject prepared via ``sim.prepare`` lives in its OWN world frame (``meta['input_frame']``),
not MNI. These pure tests pin that:
  * ``Registration`` carries ``world_frame`` through to/from JSON (default MNI for legacy files);
  * ``sim.prepare`` stamps ``world_frame = input_frame`` into ``registration.json``;
  * ``neuromod.to_placement_dict`` applies tuba's MNI->Halle->NRRD map ONLY for an MNI subject,
    and returns a non-MNI subject's coordinates verbatim (no silent corruption).
"""
from types import SimpleNamespace

import numpy as np

from skull_transparency import Registration, to_placement_dict


def test_registration_world_frame_roundtrip(tmp_path):
    reg = Registration(R_mni_to_sim=np.eye(3), dx_mm=0.5,
                       target_mni_mm=np.array([1.0, 2.0, 3.0]),
                       target_fullres_voxel=np.array([10.0, 20.0, 30.0]),
                       world_frame="ras_mm")
    d = reg.to_dict()
    assert d["frame_a"] == "ras_mm"
    assert "tuba" not in d["source"]                       # honest provenance for a non-MNI subject
    reg.to_json(tmp_path / "r.json")
    assert Registration.from_json(tmp_path / "r.json").world_frame == "ras_mm"


def test_registration_defaults_mni_when_absent():
    # a legacy registration.json without frame_a loads as MNI (Halle behaviour unchanged)
    reg = Registration.from_dict({"R_mni_to_sim": np.eye(3).tolist(), "dx_mm": 0.28,
                                  "target_mni_mm": [0, 0, 0], "target_fullres_voxel": [1, 2, 3]})
    assert reg.world_frame == "mni_ras_mm"


class _FakeHuman:
    """tuba species stand-in that visibly MOVES coordinates, so a wrong application shows up."""
    called = False

    @staticmethod
    def mni_ras_to_halle_ras(p):
        _FakeHuman.called = True
        return np.asarray(p, float) + 1000.0

    @staticmethod
    def halle_ras_to_nrrd_voxel_mm(p):
        return np.asarray(p, float)


def _placement(win, targ, focal=60.0):
    return SimpleNamespace(window_center_mni_mm=np.asarray(win, float),
                           target_mni_mm=np.asarray(targ, float),
                           focal_length_mm=focal, transparency_score=1.0, incidence_deg=10.0)


def test_non_mni_subject_skips_tuba_and_keeps_frame():
    _FakeHuman.called = False
    targ = [0.0, 0.0, 0.0]
    d = to_placement_dict(_placement([5.0, -3.0, 2.0], targ), target_name="t",
                          species_human=_FakeHuman, world_frame="ras_mm")
    assert d["frame"] == "ras_mm"
    assert _FakeHuman.called is False                      # tuba NOT applied to a non-MNI subject
    assert np.allclose(d["target_lps"], targ)              # coordinates returned verbatim


def test_mni_subject_still_maps_through_tuba():
    _FakeHuman.called = False
    d = to_placement_dict(_placement([5.0, -3.0, 2.0], [0.0, 0.0, 0.0]), target_name="t",
                          species_human=_FakeHuman, world_frame="mni_ras_mm")
    assert d["frame"] == "nrrd_voxel_mm"
    assert _FakeHuman.called is True                       # an MNI subject DOES go through tuba
    assert np.allclose(d["target_lps"], np.array([0.0, 0.0, 0.0]) + 1000.0)


def test_prepare_stamps_world_frame(tmp_path):
    from skull_transparency import TransducerSpec
    from skull_transparency.sim.prepare import Pose, write_run_descriptor
    spec = TransducerSpec(f0_hz=250e3, geometry="bowl", roc_mm=10.0, aperture_mm=8.0, ppw=2.0)
    pose = Pose(R_phys_to_grid=np.eye(3), target_grid_vox=np.array([5.0, 5.0, 5.0]), N=20,
                target_phys_mm=np.array([1.0, 2.0, 3.0]))
    write_run_descriptor(tmp_path, spec, pose, np.array([1.0, 2.0, 3.0]),
                         input_frame="lps_mm", n_array=4)
    reg = Registration.from_json(tmp_path / "registration.json")
    assert reg.world_frame == "lps_mm"
