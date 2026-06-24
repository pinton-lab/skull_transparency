"""Tests for the pure-Python fullwave2 time-reversal launchers.

The fast tests exercise the byte-format helpers and the deterministic numerics
against the committed small reference files. The full bit-identity regeneration
(which writes multi-GB maps) is opt-in via ``FULLWAVE2_VERIFY=1``.
"""
import os
import numpy as np
import pytest

from skull_transparency import paths
from skull_transparency.sim import fwio, forcoef, mlcompat, _common as C

SIM_DIR = str(paths.sim_dir())
HAVE_SIM = os.path.isdir(SIM_DIR) and os.path.exists(os.path.join(SIM_DIR, "meta.json"))
needs_sim = pytest.mark.skipif(not HAVE_SIM, reason="legacy sim tree not present")


# ---- byte-format helpers ---------------------------------------------------

def test_extendmap3d_is_edge_pad():
    rng = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    assert np.array_equal(fwio.extendMap3d(rng, 2), np.pad(rng, 2, mode="edge"))


def test_writeMapXYZ_is_c_order(tmp_path):
    m = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    p = tmp_path / "m.dat"
    fwio.writeMapXYZ(str(p), m)
    got = np.fromfile(p, dtype="<f4")
    assert np.array_equal(got, m.ravel(order="C"))


def test_writeCoords_is_fortran_order(tmp_path):
    coords = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float64)
    p = tmp_path / "c.dat"
    fwio.writeCoords(str(p), coords)
    got = np.fromfile(p, dtype="<i4")
    assert np.array_equal(got, coords.ravel(order="F").astype(np.int32))


def test_matlab_round_half_away():
    x = np.array([-2.5, -0.5, 0.5, 1.5, 2.5])
    assert np.array_equal(mlcompat.matlab_round(x), [-3, -1, 1, 2, 3])


def test_tukeywin_endpoints_and_symmetry():
    w = mlcompat.tukeywin(101, 0.4)
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] == pytest.approx(0.0, abs=1e-12)
    assert w[50] == pytest.approx(1.0)
    assert np.allclose(w, w[::-1])


# ---- deterministic numerics vs committed references ------------------------

@needs_sim
def test_pulse_d_dmap_match_committed():
    import json
    meta = json.load(open(os.path.join(SIM_DIR, "meta.json")))
    dX = meta["dX_m"]; c0 = meta["C0"]; f0 = meta["F0"]
    omega0 = 2 * np.pi * f0; cfl = 0.2; p0 = 1e6
    lam = c0 / f0; ppw = lam / dX
    arr = np.fromfile(os.path.join(SIM_DIR, "array_coords.i32"), "<i4").reshape(-1, 3).astype(float)
    dent = np.asarray(meta["dent_grid"])
    dmax = np.sqrt(((arr - dent) ** 2).sum(1)).max() * dX
    duration = 1.84 * dmax / c0
    nT = int(mlcompat.matlab_round(duration * c0 / lam * ppw / cfl))
    _, nTic = mlcompat.transmit_pulse(nT, duration, omega0, p0)
    assert nT == 4410 and nTic == 106

    d = np.ascontiguousarray(forcoef.build_d_matrix(cfl)).astype("<f4").ravel()
    ref_d = np.fromfile(os.path.join(SIM_DIR, "outward", "d.dat"), "<f4")
    assert np.array_equal(d, ref_d)

    dX_l = c0 / omega0 * 2 * np.pi / ppw
    dmap = forcoef.build_dmap(1540.0, 2900.0, dX_l / c0 * cfl, dX_l)
    ref_dm = np.fromfile(os.path.join(SIM_DIR, "outward", "dmap.dat"), "<f4")
    assert np.array_equal(np.ascontiguousarray(dmap).astype("<f4").ravel(), ref_dm)


@needs_sim
def test_source_sphere_matches_committed_icc():
    import json
    meta = json.load(open(os.path.join(SIM_DIR, "meta.json")))
    dent = np.asarray(meta["dent_grid"])
    inc = C.source_sphere(dent, 3)
    icc = np.fromfile(os.path.join(SIM_DIR, "outward", "icc.dat"), "<i4").reshape(5, -1).T
    # stored coords = xyz + 48 - 1 = xyz + 47 ; col4,col5 -> 0
    exp = inc.copy()
    exp[:, 0:3] += 47
    exp[:, 3:5] -= 1
    assert np.array_equal(exp.astype(np.int32), icc)


# ---- optimization equivalences (must preserve exact bytes) -----------------

def test_matlab_single_sum_matches_naive_loop():
    rng = np.random.default_rng(0)
    for n in (100, 1000, 12345, 50000):
        v = (rng.standard_normal(n).astype(np.float32) * np.float32(1e3))
        # naive 4-lane sequential reference (the spec the vectorized form replaces)
        x = v.ravel(order="F"); k = 4; m = (x.size + k - 1) // k
        pad = np.zeros(m * k, np.float32); pad[:x.size] = x
        acc = np.zeros(k, np.float32)
        for r in range(m):
            acc = (acc + pad.reshape(m, k)[r]).astype(np.float32)
        s = np.float32(0.0)
        for j in range(k):
            s = np.float32(s + acc[j])
        assert mlcompat.matlab_single_sum(v) == s


def test_rho_from_c_matches_naive():
    rng = np.random.default_rng(1)
    c = (rng.uniform(1400, 3000, size=(8, 9, 10))).astype(np.float64)
    naive = np.maximum(np.minimum((c - 1540.0) / 1360.0, 1.0), 0.0)
    naive = (1000.0 + naive * (2200.0 - 1000.0)).astype(np.float32)
    assert np.array_equal(C.rho_from_c(c).view(np.uint32), naive.view(np.uint32))


def test_inplace_K_and_dcmap_bytes_match():
    rng = np.random.default_rng(2)
    ce = rng.uniform(1500, 2900, size=(6, 7, 8)).astype(np.float64)
    re = rng.uniform(1000, 2200, size=(6, 7, 8)).astype(np.float64)
    K_ref = (ce ** 2 * re).astype("<f4")
    t = ce.copy(); t *= t; t *= re
    assert np.array_equal(t.astype("<f4").view(np.uint32), K_ref.view(np.uint32))
    minc = float(ce.min())
    dc_ref = ((np.sign(ce) * np.floor(np.abs(ce) + 0.5)) - minc)
    dc_new = ce + 0.5; np.floor(dc_new, out=dc_new); dc_new -= minc
    assert np.array_equal(fwio._to_int32(dc_new), fwio._to_int32(dc_ref))


# ---- rebuild_medium: user-defined c/rho/alpha maps + back-compatible fallback ----

def test_rebuild_medium_fallback_and_supplied_maps(tmp_path):
    rng = np.random.default_rng(3)
    nx, ny, nz = 5, 6, 7                                   # anisotropic on purpose
    c = rng.uniform(1500, 2900, size=(nx, ny, nz)).astype("<f4")
    c.ravel(order="F").tofile(tmp_path / "c.f32")

    # fallback: no rho_file/alpha_file -> rho synthesized from c, alpha None,
    # and c is byte-identical to the original two-tuple behaviour.
    cc, rho, alpha = C.rebuild_medium(str(tmp_path),
                                      dict(kind="maps", file="c.f32", N=[nx, ny, nz]))
    assert cc.shape == (nx, ny, nz) and alpha is None
    assert np.array_equal(cc, c.reshape((nx, ny, nz), order="F").astype(np.float64))
    assert np.array_equal(rho.view(np.uint32), C.rho_from_c(cc).view(np.uint32))

    # supplied density + absorption maps are loaded verbatim (Fortran order)
    rho_in = rng.uniform(1000, 2200, size=(nx, ny, nz)).astype("<f4")
    al_in = rng.uniform(0.0, 9.0, size=(nx, ny, nz)).astype("<f4")
    rho_in.ravel(order="F").tofile(tmp_path / "rho.f32")
    al_in.ravel(order="F").tofile(tmp_path / "a.f32")
    _, rho2, alpha2 = C.rebuild_medium(str(tmp_path),
        dict(kind="maps", file="c.f32", N=[nx, ny, nz], rho_file="rho.f32", alpha_file="a.f32"))
    assert np.array_equal(rho2.view(np.uint32),
                          rho_in.reshape((nx, ny, nz), order="F").view(np.uint32))
    assert np.array_equal(alpha2, al_in.reshape((nx, ny, nz), order="F").astype(np.float64))

    # cubic N as a plain int (legacy halle_c descriptor) still works
    n = 4
    cube = rng.uniform(1500, 2900, size=(n, n, n)).astype("<f4")
    cube.ravel(order="F").tofile(tmp_path / "cube.f32")
    c3, _, a3 = C.rebuild_medium(str(tmp_path), dict(kind="halle_c", file="cube.f32", N=n))
    assert c3.shape == (n, n, n) and a3 is None


# ---- full bit-identity (opt-in; writes multi-GB scratch) -------------------

@needs_sim
@pytest.mark.skipif(os.environ.get("FULLWAVE2_VERIFY") != "1",
                    reason="set FULLWAVE2_VERIFY=1 to run the full regeneration")
def test_full_bit_identity(tmp_path):
    from skull_transparency.sim.verify import verify_dirs
    dirs = (os.environ.get("FULLWAVE2_VERIFY_DIRS") or "outward").split()
    assert verify_dirs(SIM_DIR, str(tmp_path), dirs) == 0
