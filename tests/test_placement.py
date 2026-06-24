import numpy as np

from scipy.spatial import cKDTree

from skull_transparency import (compute_transparency_map, place_bowl, BowlConstraints,
                                place_array, ArrayConstraints, to_placement_dict)


def test_dentate_window_is_occipital(bundle):
    """For the (left posterior-fossa) dentate, the delivered-energy-optimal bowl
    window must be posterior + inferior (the occipital/suboccipital approach)."""
    tm = compute_transparency_map(bundle)
    pl = place_bowl(tm, BowlConstraints(focal_length_mm=60.0, bowl_radius_mm=15.0, theta_max_deg=35.0))
    w, t = pl.window_center_mni_mm, pl.target_mni_mm
    assert w[1] < t[1], "window should be POSTERIOR of the dentate"
    assert w[2] < t[2], "window should be INFERIOR of the dentate"
    assert pl.incidence_deg <= 35.0
    # apex sits one focal length from the target along the beam
    assert np.isclose(np.linalg.norm(pl.apex_mni_mm - pl.target_mni_mm), 60.0, atol=1e-6)
    assert np.isclose(np.linalg.norm(pl.beam_dir_mni), 1.0)


def test_placement_dict_mni_fallback(bundle):
    tm = compute_transparency_map(bundle)
    pl = place_bowl(tm, BowlConstraints(focal_length_mm=60.0))
    d = to_placement_dict(pl, target_name="dentate_left", species_human=None)
    for k in ("xdc_center_lps", "beam_dir_3d", "target_lps", "normal_3d", "tangent_3d",
              "bitangent_3d", "transparency_score", "target_name"):
        assert k in d
    assert d["frame"] == "mni_ras_mm"            # no tuba injected
    assert np.isclose(np.linalg.norm(d["beam_dir_3d"]), 1.0)


def test_array_selection_respects_constraints(bundle):
    tm = compute_transparency_map(bundle)
    al = place_array(tm, ArrayConstraints(n_elements=40, min_spacing_mm=5.0, theta_max_deg=35.0,
                                          region_center_mni_mm=[-12.0, -57.0, -34.0], region_radius_mm=60.0))
    assert al.n_placed <= 40
    assert (al.incidence_deg <= 35.0 + 1e-6).all()           # incidence cap honoured
    d = cKDTree(al.element_mni_mm).query(al.element_mni_mm, k=2)[0][:, 1]
    assert d.min() >= 5.0 - 1e-6                              # min spacing honoured
    assert al.aggregate_coupling > 0
    assert al.orientation_mni.shape == (al.n_placed, 3)


def test_raw_vs_distance_corrected_differ(bundle):
    """Sanity: the distance-corrected map favours a different (thin-bone) window than
    the raw delivered-energy map — confirming the placement default matters."""
    tm = compute_transparency_map(bundle)
    raw = place_bowl(tm, BowlConstraints(use_distance_corrected=False))
    dc = place_bowl(tm, BowlConstraints(use_distance_corrected=True))
    assert not np.allclose(raw.window_center_mni_mm, dc.window_center_mni_mm)
