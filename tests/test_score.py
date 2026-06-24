"""PositioningScore packaging — pure, no data/GPU. Uses lightweight stand-ins with the
attributes BowlPlacement / CapPlacement expose."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from skull_transparency import PositioningScore


def _bowl(score=0.82, inc=12.3, patches=140, pmax=3.4, csmax=5.1, obj="peak"):
    return SimpleNamespace(transparency_score=score, incidence_deg=inc, n_footprint_patches=patches,
                           extras={"p_max_proxy": pmax, "objective": obj, "candidate_scores_max": csmax})


def test_from_bowl_placement():
    s = PositioningScore.from_placement(_bowl(), target_name="dentate_left")
    assert s.normalized == 0.82
    assert s.focal_pressure_proxy == 3.4
    assert s.incidence_deg == 12.3
    assert s.objective == "peak"
    assert s.target_name == "dentate_left"
    assert s.extras["n_footprint_patches"] == 140
    assert s.extras["candidate_scores_max"] == 5.1


def test_from_cap_placement():
    cap = SimpleNamespace(J_cap=9.0, n_kept=80, n_cap=100, window_incidence_deg=20.0,
                          extras={"score_norm": 0.7})
    s = PositioningScore.from_placement(cap)
    assert s.objective == "cap"
    assert s.normalized == 0.7
    assert np.isclose(s.focal_pressure_proxy, 3.0)        # sqrt(J_cap)
    assert s.incidence_deg == 20.0
    assert s.extras["J_cap"] == 9.0 and s.extras["n_kept"] == 80


def test_to_dict_and_json_round_trip(tmp_path):
    s = PositioningScore.from_placement(_bowl(), target_name="thalamus")
    d = s.to_dict()
    for k in ("schema", "normalized", "focal_pressure_proxy", "incidence_deg", "objective",
              "target_name", "definition", "n_footprint_patches", "candidate_scores_max"):
        assert k in d
    assert d["schema"] == "skull_transparency.positioning_score/1"
    assert "OVERSTATES" in d["definition"]                # the focusing-gain caveat is carried
    p = tmp_path / "score.json"
    s.to_json(p)
    assert json.loads(p.read_text()) == d


def test_unknown_placement_raises():
    with pytest.raises(TypeError):
        PositioningScore.from_placement(SimpleNamespace())
