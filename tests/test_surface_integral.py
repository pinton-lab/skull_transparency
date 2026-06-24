"""Surface-integral placement method: projection (Pinton 2012), TR/phase-only drives, and the
angular-spectrum PSF. Pure-synthetic unit tests + a couple of bundle-gated integration checks."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import skull_transparency as st
from skull_transparency import paths
from skull_transparency.complex_field import ComplexField, ballistic_window_global


def _synthetic_cf(M=64, R=85.0, k0=4.08, seed=0):
    """A ComplexField of M elements on a sphere of radius R about the origin, with a radial
    carrier phase + random per-element aberration and amplitude."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((M, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pos = R * dirs                                   # mm; target at origin
    rad = np.full(M, R)
    amp = 1.0 + 0.5 * rng.random(M)
    ab = 0.0 * rng.standard_normal(M)                # aberration-free by default
    G = amp * np.exp(1j * (-k0 * rad + ab))
    return ComplexField(pos_fullres=pos, G=G, G_win=G.copy(), E_win=amp ** 2, radius_mm=rad,
                        source_T_fullres=np.zeros(3), f0=1e6, k0=k0, dx_mm=1.0, c0=1540.0)


# ---- projection (Pinton 2012) ----
def test_projection_roundtrip_identity():
    cf = _synthetic_cf()
    rad = cf.radius_mm * (0.6 + 0.4 * np.linspace(0, 1, len(cf.radius_mm)))  # varied radii
    G = cf.G_win
    GR = st.project_to_sphere(G, rad, 90.0, cf.k0)
    Gback = st.project_to_sphere(GR, np.full(len(G), 90.0), rad, cf.k0)
    assert np.max(np.abs(Gback - G)) / np.max(np.abs(G)) < 1e-12


def test_projection_synthetic_point_source_exact():
    # element_field emits the TRACE convention G = e^{-ikr}/r (numpy e^{-iωt} DFT of a causal
    # arrival), which must project to e^{-ikR}/R. Using the spatial e^{+ikr} convention here would
    # mask a projection sign error, so test the convention the package actually produces.
    k = 4.08; r1 = np.linspace(50, 130, 40); R = 95.0
    G = np.exp(-1j * k * r1) / r1
    GR = st.project_to_sphere(G, r1, R, k)
    expect = np.exp(-1j * k * R) / R
    assert np.max(np.abs(GR - expect)) / abs(expect) < 1e-10


def test_projection_cap_energy_invariant():
    cf = _synthetic_cf()
    rad = cf.radius_mm * np.linspace(0.7, 1.0, len(cf.radius_mm))
    GR = st.project_to_sphere(cf.G_win, rad, 100.0, cf.k0)
    e1 = st.energy_on_unit_sphere(cf.G_win, rad)
    e2 = st.energy_on_unit_sphere(GR, np.full(len(GR), 100.0))
    assert np.allclose(e1, e2, rtol=1e-10)


# ---- drives (Cauchy-Schwarz optimality) ----
def test_drive_optimal_is_cauchy_schwarz_max():
    cf = _synthetic_cf(M=48, seed=1)
    idx = np.arange(cf.G_win.size)
    u, pmax = st.drive_optimal(cf, idx)
    assert abs((np.abs(u) ** 2).sum() - 1.0) < 1e-9
    g = cf.G_win[idx]
    assert abs(pmax - np.sqrt((np.abs(g) ** 2).sum())) < 1e-9 * pmax
    rng = np.random.default_rng(3)
    for _ in range(500):
        v = rng.standard_normal(len(g)) + 1j * rng.standard_normal(len(g)); v /= np.linalg.norm(v)
        assert abs((v * g).sum()) <= pmax * (1 + 1e-9)


def test_phase_only_ratio_is_apodization_loss():
    cf = _synthetic_cf(M=80, seed=2)
    idx = np.arange(cf.G_win.size)
    _, p_opt = st.drive_optimal(cf, idx)
    _, p_po = st.drive_phase_only(cf, idx)
    g = np.abs(cf.G_win[idx]); cv = g.std() / g.mean()
    assert abs(p_po / p_opt - 1 / np.sqrt(1 + cv ** 2)) < 1e-9
    assert p_po <= p_opt + 1e-12          # phase-only never beats amplitude-apodized


# ---- angular-spectrum PSF ----
def test_focal_psf_peaks_at_target():
    cf = _synthetic_cf(M=120, seed=4)
    idx = np.arange(cf.G_win.size)
    off = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [5, 5, 5]], float)
    p = st.focal_psf(cf, idx, off)
    assert np.argmax(p) == 0                              # brightest exactly on target
    assert p[0] == pytest.approx((np.abs(cf.G_win) ** 2).sum())
    fw = st.focal_fwhm(cf, idx)
    assert all(fw[k] > 0 for k in fw)                    # finite focal widths


def test_projection_sign_matches_trace_convention():
    # element_field emits e^{-ikr}; project_to_sphere must re-sphere it (the +ikr operator would fail)
    cf = _synthetic_cf(M=40, R=80.0, seed=6)          # G = amp·e^{-ik0·rad}
    GR = st.project_to_sphere(cf.G_win, cf.radius_mm, 110.0, cf.k0)
    # phase at the new radius must be -k0·110 (mod 2π), amplitude scaled by rad/R
    expect_phase = np.angle(np.exp(-1j * cf.k0 * 110.0))
    assert np.allclose(np.angle(GR / np.abs(GR)), expect_phase, atol=1e-9)


def test_drive_rejects_surface_index_space():
    cf = _synthetic_cf(M=32, seed=7)
    bad = np.array([0, 5, cf.G_win.size + 100])       # an out-of-range (e.g. dense-surface) index
    for fn in (st.drive_optimal, st.drive_phase_only):
        with pytest.raises(ValueError):
            fn(cf, bad)
    with pytest.raises(ValueError):
        st.focal_fwhm(cf, bad)


def test_drives_handle_empty_selection():
    cf = _synthetic_cf(M=16, seed=8)
    for fn in (st.drive_optimal, st.drive_phase_only):
        u, p = fn(cf, np.array([], int))
        assert u.shape == (0,) and p == 0.0


def test_focal_fwhm_degenerate_is_nan():
    cf = _synthetic_cf(M=16, seed=9)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fw = st.focal_fwhm(cf, np.array([3]))         # single element -> undefined
    assert all(np.isnan(v) for v in fw.values())


def test_drive_regime_ordering_single_le_phase_le_tr():
    # single-element (geometric, aberration uncorrected) <= phase-only array <= full-TR array
    cf = _synthetic_cf(M=64, seed=11)
    # inject aberration so the single-element coherent sum is genuinely penalized
    rng = np.random.default_rng(11)
    cf.G_win = cf.G_win * np.exp(1j * 1.5 * rng.standard_normal(cf.G_win.size))
    idx = np.arange(cf.G_win.size)
    _, p_se = st.drive_single_element(cf, idx)
    _, p_po = st.drive_phase_only(cf, idx)
    _, p_tr = st.drive_optimal(cf, idx)
    assert p_se <= p_po + 1e-9 <= p_tr + 1e-9
    g = st.coherence_factor(cf, idx)
    assert 0.0 <= g <= 1.0
    assert p_se == pytest.approx(np.sqrt(1.0 / len(idx)) * abs((cf.G_win * np.exp(1j * cf.k0 * cf.radius_mm)).sum()))


def test_coherence_factor_unity_when_phase_flat():
    # if the aberration phase Δφ = argG + k0 r is constant, γ = 1 (single element loses nothing)
    cf = _synthetic_cf(M=40, seed=12)
    cf.G_win = np.abs(cf.G_win) * np.exp(-1j * cf.k0 * cf.radius_mm)  # Δφ ≡ 0
    assert st.coherence_factor(cf, np.arange(cf.G_win.size)) == pytest.approx(1.0)


def test_ballistic_window_shape():
    rng = np.random.default_rng(5)
    nT, M = 400, 30
    R = 1e-3 * rng.standard_normal((nT, M))
    R[180:210, :] += np.hanning(30)[:, None] * np.sin(np.linspace(0, 12, 30))[:, None]  # arrival burst
    w, a0, b0 = ballistic_window_global(R)
    assert 0 <= a0 < b0 <= nT and w.shape == (nT,)
    assert w[a0:b0].max() > 0 and w[:a0].sum() == 0 and w[b0:].sum() == 0


# ---- integration (needs the Halle bundle + array_traces.npz) ----
def _val(name):
    p = paths.analysis_dir() / "placement_validation" / name
    if not p.exists():
        pytest.skip(f"{name} not available")
    return p


def test_element_field_reproduces_fair_compare_energy(bundle):
    if not (bundle.dir / bundle.files.get("array_traces", "array_traces.npz")).exists():
        pytest.skip("array_traces.npz not available in bundle dir")
    FC = json.load(open(_val("fair_compare.json")))
    cf = bundle.element_complex_field()
    sel_T = np.array(FC["sel_transparency"])
    assert cf.E_win[sel_T].sum() == pytest.approx(FC["sumE_transparency"], rel=1e-6)


def test_place_bowl_optimal_picks_inferior_posterior_window(bundle):
    tm = st.compute_transparency_map(bundle)
    bp = st.place_bowl_optimal(tm, st.BowlConstraints(theta_max_deg=35.0, bowl_radius_mm=20.0))
    tgt = np.asarray(tm.meta["target_mni_mm"])
    assert bp.window_center_mni_mm[2] < tgt[2]           # inferior (suboccipital approach)
    assert bp.footprint_surf_idx is not None and bp.extras["p_max_proxy"] > 0
