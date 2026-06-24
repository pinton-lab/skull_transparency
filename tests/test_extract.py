"""genout_mod -> Field Bundle extraction. Validated with a SYNTHETIC genout_mod (no
GPU/solver): the PML-pad crop is checked against an interior marker, and the extracted
bundle is fed through the real transparency -> placement -> score chain. Format logic
only — absolute-number parity vs the real solver is a separate (GPU) confirmation."""
import json

import numpy as np

import skull_transparency as st
from skull_transparency import TransducerSpec
from skull_transparency.sim.prepare import build_run_from_medium
from skull_transparency.sim.extract import extract_bundle, MOD, PAD


def _producer_sim(tmp):
    """A tiny producer sim tree (c.f32 + meta.json + registration.json), N ~ 17."""
    n = 60
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt((ii - 30) ** 2 + (jj - 30) ** 2 + (kk - 30) ** 2)
    c = np.where(r < 16.0, 2900.0, 1540.0).astype(np.float32)
    spec = TransducerSpec(f0_hz=250e3, geometry="bowl", roc_mm=10.0, aperture_mm=8.0, ppw=2.0)
    sim = tmp / "sim"
    build_run_from_medium(c, np.eye(4), np.array([30.0, 30.0, 30.0]), spec, sim,
                          approach=np.array([0.0, 0.0, 1.0]), standoff_mm=5.0, surround_mm=20.0)
    return sim, int(json.loads((sim / "meta.json").read_text())["N"])


def _dims(N):
    nXe = N + 2 * PAD
    return (nXe + MOD - 1) // MOD, len(range(0, N, MOD)), PAD // MOD     # (n2, nf, lo)


def test_extract_crops_the_pml_pad(tmp_path):
    sim, N = _producer_sim(tmp_path)
    n2, nf, lo = _dims(N)
    field = np.full((n2, n2, n2), 1.0, np.float32)          # pad value
    field[lo:lo + nf, lo:lo + nf, lo:lo + nf] = 5.0          # interior marker
    run = tmp_path / "run"
    run.mkdir()
    np.repeat(field[None], 4, axis=0).tofile(run / "genout_mod.dat")   # 4 identical frames

    out = extract_bundle(run, tmp_path / "bundle", sim)
    Pmax = np.load(out / "outward_Pmax.npy")
    Iint = np.load(out / "outward_Iint.npy")
    assert Pmax.shape == (nf, nf, nf)
    assert np.allclose(Pmax, 5.0)                            # interior selected, NOT the pad (1.0)
    assert np.allclose(Iint, 4 * 25.0)                       # 4 frames * 5^2


def test_extracted_bundle_feeds_the_chain(tmp_path):
    sim, N = _producer_sim(tmp_path)
    n2, nf, lo = _dims(N)
    ii, jj, kk = np.meshgrid(np.arange(n2), np.arange(n2), np.arange(n2), indexing="ij")
    base = (1000.0 + 5.0 * kk).astype(np.float32)            # positive, brighter toward +k
    gm = np.stack([base * (1.0 + 0.1 * t) for t in range(4)], axis=0)
    run = tmp_path / "run"
    run.mkdir()
    gm.tofile(run / "genout_mod.dat")

    out = extract_bundle(run, tmp_path / "bundle", sim)
    for f in ("bundle.json", "outward_Iint.npy", "outward_Pmax.npy", "skull_fullres_c.npy",
              "registration.json", "phase_info.json"):
        assert (out / f).exists(), f

    bundle = st.load_bundle(out)
    tmap = st.compute_transparency_map(bundle)
    assert len(tmap.surf_vox) > 0 and np.isfinite(tmap.Ipk_Wcm2).all()
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=20.0, bowl_radius_mm=6.0,
                                                theta_max_deg=35.0))
    score = st.PositioningScore.from_placement(pl)
    assert 0.0 <= score.normalized <= 1.0


def test_missing_genout_mod_raises(tmp_path):
    import pytest
    sim, _ = _producer_sim(tmp_path)
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="genout_mod"):
        extract_bundle(tmp_path / "empty", tmp_path / "bundle", sim)
