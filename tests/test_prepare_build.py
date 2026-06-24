"""End-to-end test for build_run_from_medium: with every producer step implemented it
must turn a synthetic (c, affine, target, TransducerSpec) into a complete sim tree
(c.f32 + array_coords.i32 + meta.json + registration.json). Tiny grid + pytest tmp
(not /celerina); no GPU/MATLAB/tuba — it only *writes* the solver inputs."""
import json

import numpy as np

from skull_transparency import TransducerSpec, Registration
from skull_transparency.sim.prepare import build_run_from_medium
from skull_transparency.sim._common import array_coords_from_i32


def test_build_run_from_medium_writes_full_sim_tree(tmp_path):
    # synthetic skull: a solid bone ball (recording target) in water; identity world frame
    n_in = 60
    target_phys = np.array([30.0, 30.0, 30.0])
    ii, jj, kk = np.meshgrid(np.arange(n_in), np.arange(n_in), np.arange(n_in), indexing="ij")
    r = np.sqrt((ii - 30) ** 2 + (jj - 30) ** 2 + (kk - 30) ** 2)
    c_in = np.where(r < 16.0, 2900.0, 1540.0).astype(np.float32)   # mm == voxel (eye affine)

    spec = TransducerSpec(f0_hz=250e3, geometry="bowl", roc_mm=10.0, aperture_mm=8.0, ppw=2.0)
    out = tmp_path / "run"
    ret = build_run_from_medium(c_in, np.eye(4), target_phys, spec, out,
                                approach=np.array([0.0, 0.0, 1.0]), standoff_mm=5.0, surround_mm=20.0)
    assert ret == out

    for fname in ("c.f32", "array_coords.i32", "meta.json", "registration.json"):
        assert (out / fname).exists(), f"missing {fname}"

    meta = json.loads((out / "meta.json").read_text())
    N = meta["N"]
    assert (out / "c.f32").stat().st_size == N ** 3 * 4         # N^3 float32
    assert meta["n_array"] > 0
    assert meta["c_file"] == "c.f32" and meta["rho_file"] is None and meta["alpha_file"] is None
    assert meta["attenuation"] is False                        # no alpha map supplied

    arr, nA = array_coords_from_i32(out / "array_coords.i32")   # round-trips through the reader
    assert nA == meta["n_array"]
    assert (arr >= 0).all() and (arr < N).all()                # recorders inside the grid
    assert (arr[:, 2] > meta["dent_grid"][2]).all()            # on the +Z (transducer) side

    reg = Registration.from_json(out / "registration.json")
    assert np.allclose(reg.mni_to_fullres(target_phys), meta["dent_grid"], atol=1e-6)


def test_supplied_alpha_writes_maps_and_enables_attenuation(tmp_path):
    n_in = 60
    tgt = np.array([30.0, 30.0, 30.0])
    ii, jj, kk = np.meshgrid(np.arange(n_in), np.arange(n_in), np.arange(n_in), indexing="ij")
    r = np.sqrt((ii - 30) ** 2 + (jj - 30) ** 2 + (kk - 30) ** 2)
    c_in = np.where(r < 16.0, 2900.0, 1540.0).astype(np.float32)
    rho_in = np.where(r < 16.0, 1900.0, 1000.0).astype(np.float32)
    alpha_in = np.where(r < 16.0, 8.0, 0.2).astype(np.float32)        # dB/cm/MHz

    spec = TransducerSpec(f0_hz=250e3, geometry="bowl", roc_mm=10.0, aperture_mm=8.0, ppw=2.0)
    out = tmp_path / "run"
    build_run_from_medium(c_in, np.eye(4), tgt, spec, out, rho_map=rho_in, alpha_map=alpha_in,
                          approach=np.array([0.0, 0.0, 1.0]), standoff_mm=5.0, surround_mm=20.0)

    meta = json.loads((out / "meta.json").read_text())
    assert (out / "rho.f32").exists() and (out / "alpha.f32").exists()
    assert meta["rho_file"] == "rho.f32" and meta["alpha_file"] == "alpha.f32"
    assert meta["attenuation"] is True                               # alpha present -> on
    assert meta["alpha_units"] == "db_mhz_cm"
