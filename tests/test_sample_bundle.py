"""The synthetic sample bundle exercises the whole back-half chain (transparency ->
placement -> score) AND the `skull-transparency place` CLI with zero /celerina data,
no GPU, and no tuba required."""
import json

import numpy as np

import skull_transparency as st
from skull_transparency import cli


def test_chain_runs_on_synthetic_bundle(tmp_path):
    bundle_dir = st.make_synthetic_bundle(tmp_path / "bundle")
    bundle = st.load_bundle(bundle_dir)

    tmap = st.compute_transparency_map(bundle)
    assert len(tmap.surf_vox) > 0
    assert np.isfinite(tmap.Ipk_Wcm2).all() and (tmap.Ipk_Wcm2 > 0).any()

    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60.0, bowl_radius_mm=15.0,
                                                theta_max_deg=35.0))
    assert 0.0 <= pl.transparency_score <= 1.0
    assert pl.incidence_deg <= 35.0 + 1e-6

    score = st.PositioningScore.from_placement(pl, target_name="synthetic_target")
    assert 0.0 <= score.normalized <= 1.0
    assert np.isfinite(score.focal_pressure_proxy)

    # placement dict works without tuba (mni fallback)
    d = st.to_placement_dict(pl, target_name="synthetic_target", species_human=None)
    assert d["frame"] == "mni_ras_mm"
    assert np.isclose(np.linalg.norm(d["beam_dir_3d"]), 1.0)


def test_place_cli_on_synthetic_bundle(tmp_path):
    bundle_dir = st.make_synthetic_bundle(tmp_path / "bundle")
    out = tmp_path / "result"
    rc = cli.main(["place", "--bundle", str(bundle_dir), "--out", str(out),
                   "--target-name", "synthetic_target"])
    assert rc == 0
    for f in ("surface_map.npz", "score.json", "placement.json"):
        assert (out / f).exists(), f
    score = json.loads((out / "score.json").read_text())
    assert score["schema"] == "skull_transparency.positioning_score/1"
    assert 0.0 <= score["normalized"] <= 1.0
    # the transparency map round-trips
    tm = st.TransparencyMap.from_npz(out / "surface_map.npz")
    assert len(tm.surf_vox) > 0
