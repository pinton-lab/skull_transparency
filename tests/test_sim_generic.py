"""Generic-subject launcher path: the guarded edits that let launch_outward consume a
producer-built sim tree (meta c_file + supplied rho/alpha) and that keep the c-porosity
absorption byte-identical when no alpha is supplied. No /celerina, GPU, or MATLAB; the
end-to-end case writes only a tiny (~N=17) map set to pytest tmp."""
import json
import os

import numpy as np

from skull_transparency import TransducerSpec
from skull_transparency.sim.launch_core import _porosity_aexp
from skull_transparency.sim.launchers import launch_outward
from skull_transparency.sim.prepare import build_run_from_medium

C0, OMEGA0, DT = 1540.0, 2.0 * np.pi * 1e6, 1e-8


def _ref_aexp(c_ext):
    # the ORIGINAL c-porosity formula (the spec the refactor must preserve byte-for-byte)
    phi = np.clip(1.0 - (c_ext - 1540.0) / (2900.0 - 1540.0), 0.0, 1.0)
    alpha = np.where(c_ext > 1540.0, (2.0 + 78.0 * np.sqrt(phi)) * (12.0 / 40.0), 0.4)
    texp = (alpha / 2.0) * (OMEGA0 / (2.0 * np.pi) / 1e6) * C0 / 1e-2 * (np.log(10.0) / 20.0)
    return np.exp(-DT * texp)


def test_porosity_aexp_default_is_byte_identical():
    rng = np.random.default_rng(3)
    c = rng.uniform(1450.0, 2950.0, (5, 6, 7))
    got = _porosity_aexp(c, C0, OMEGA0, DT)            # alpha_dbmhzcm defaults to None
    assert got.tobytes() == _ref_aexp(c).tobytes()    # exact bytes -> Halle path unchanged


def test_supplied_alpha_overrides_porosity():
    c = np.full((4, 4, 4), 2900.0)                     # bone everywhere
    derived = _porosity_aexp(c, C0, OMEGA0, DT)        # high (porous-bone) attenuation
    supplied = _porosity_aexp(c, C0, OMEGA0, DT, alpha_dbmhzcm=np.full((4, 4, 4), 0.4))
    assert (supplied > derived).all()                  # 0.4 dB/MHz/cm attenuates far less
    water = _porosity_aexp(np.full((4, 4, 4), 1540.0), C0, OMEGA0, DT)
    assert np.allclose(supplied, water)                # matches the 0.4 water value


def test_launch_outward_consumes_a_generic_producer_tree(tmp_path):
    # producer builds the sim tree (c_file + supplied rho/alpha); launcher writes the
    # solver INPUTS (no solver, no GPU) -> proves the meta c_file + rho/alpha overrides work
    n_in = 60
    tgt = np.array([30.0, 30.0, 30.0])
    ii, jj, kk = np.meshgrid(np.arange(n_in), np.arange(n_in), np.arange(n_in), indexing="ij")
    r = np.sqrt((ii - 30) ** 2 + (jj - 30) ** 2 + (kk - 30) ** 2)
    c_in = np.where(r < 16.0, 2900.0, 1540.0).astype(np.float32)
    rho_in = np.where(r < 16.0, 1900.0, 1000.0).astype(np.float32)     # bone 1900 (not rho_from_c=2200)
    alpha_in = np.where(r < 16.0, 8.0, 0.2).astype(np.float32)
    spec = TransducerSpec(f0_hz=250e3, geometry="bowl", roc_mm=10.0, aperture_mm=8.0, ppw=2.0)

    sim_dir = tmp_path / "sim"
    build_run_from_medium(c_in, np.eye(4), tgt, spec, sim_dir, rho_map=rho_in, alpha_map=alpha_in,
                          approach=np.array([0.0, 0.0, 1.0]), standoff_mm=5.0, surround_mm=20.0)

    outdir = launch_outward(str(sim_dir), str(tmp_path / "run"), write_maps=True)

    N = json.loads((sim_dir / "meta.json").read_text())["N"]
    nXe = N + 96                                                       # 2*PAD, PAD = nbdy(40)+M(8)
    for f in ("c.dat", "K.dat", "rho.dat", "Aexp.dat", "dcmap.dat", "beta.dat", "icc.dat", "outc.dat"):
        assert os.path.exists(os.path.join(outdir, f)), f
    assert os.path.getsize(os.path.join(outdir, "c.dat")) == nXe ** 3 * 4

    rho_dat = np.fromfile(os.path.join(outdir, "rho.dat"), dtype="<f4")
    assert 1850.0 < rho_dat.max() < 1950.0            # supplied density used (not rho_from_c -> 2200)
