"""Exclusion zones (exclusion.py) and their use in placement.

The property that matters: a zone keeps the WHOLE transducer off the excluded area, not just its
centre. Synthetic data only (no /celerina data, GPU or tuba)."""
import numpy as np
import pytest

from skull_transparency import (Exclusion, Zone, EAR_ZONES, zone_mask, image_mask,
                                compute_transparency_map, place_bowl, BowlConstraints,
                                place_array, ArrayConstraints, CapPose, score_cap_pose,
                                place_cap_optimal)


def _plane(n=81, step=1.0):
    g = (np.arange(n) - n // 2) * step
    X, Y = np.meshgrid(g, g, indexing="ij")
    return np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]


def test_zone_mask_labels_first_zone():
    p = np.array([[0, 0, 0], [5, 0, 0], [30, 0, 0], [100, 0, 0.]])
    z = [Zone("a", (0, 0, 0), 10.0), Zone("b", (4, 0, 0), 30.0)]
    assert zone_mask(p, z).tolist() == [1, 1, 2, 0]


def test_legal_centers_buffer_by_footprint():
    p = _plane()
    ex = Exclusion(p, zone_mask(p, [Zone("ear", (0, 0, 0), 10.0)]), ["ear"])
    legal = ex.legal_centers(15.0)
    d = np.linalg.norm(p, axis=1)
    # the nearest excluded patch to a centre is ~ (d - 10); legal iff that exceeds the footprint
    assert not legal[d <= 25.0 - 1.0].any()
    assert legal[d >= 25.0 + 1.5].all()
    assert Exclusion(p, np.zeros(len(p), int), []).legal_centers(15.0).all()


def test_covered_reports_zone_names():
    p = _plane()
    ex = Exclusion(p, zone_mask(p, [Zone("left ear", (-20, 0, 0), 5.0), Zone("scar", (20, 0, 0), 5.0)]),
                   ["left ear", "scar"])
    above = lambda x, y: np.array([[x, y, 8.0]])            # an element 8 mm above the skull
    assert ex.covered(above(-20, 0)) == ["left ear"]
    assert ex.covered(np.r_[above(-20, 0), above(20, 1)]) == ["left ear", "scar"]
    assert ex.covered(above(0, 0)) == []
    assert ex.covered(above(-12, 0), margin_mm=4.0) == ["left ear"]   # within the margin


def test_image_mask_samples_nifti_in_world_mm():
    nib = pytest.importorskip("nibabel")
    data = np.zeros((40, 40, 40), np.uint8)
    data[10:20, 10:20, 10:20] = 1
    aff = np.diag([2.0, 2.0, 2.0, 1.0]); aff[:3, 3] = -40.0  # world = 2*ijk - 40
    img = nib.Nifti1Image(data, aff)
    pts = np.array([[-40 + 2 * 15, -40 + 2 * 15, -40 + 2 * 15],   # inside the box
                    [0.0, 0.0, 0.0],                              # ijk 20 -> just outside
                    [500.0, 0.0, 0.0]])                           # outside the image
    assert image_mask(pts, img).tolist() == [True, False, False]


def test_place_bowl_keeps_whole_footprint_clear(synthetic_bundle):
    tm = compute_transparency_map(synthetic_bundle)
    R = 15.0
    c = dict(focal_length_mm=60.0, bowl_radius_mm=R, theta_max_deg=35.0)
    free = place_bowl(tm, BowlConstraints(**c))
    surf = tm.surf_mni_mm()
    zone = Zone("no-go", tuple(free.window_center_mni_mm), 8.0)   # forbid the unconstrained window
    ex = Exclusion.from_map(tm, [zone])
    pl = place_bowl(tm, BowlConstraints(**c, exclude_mask=ex.excluded))
    w = np.asarray(pl.window_center_mni_mm)
    d = np.linalg.norm(surf[ex.excluded] - w, axis=1)
    assert d.min() > R                                           # no excluded patch under the bowl
    assert np.linalg.norm(w - np.asarray(zone.center_mm)) > zone.radius_mm + R - 1e-6


def test_place_array_skips_excluded_patches(synthetic_bundle):
    tm = compute_transparency_map(synthetic_bundle)
    free = place_array(tm, ArrayConstraints(n_elements=30, min_spacing_mm=4.0))
    ex = Exclusion.from_map(tm, [Zone("no-go", tuple(free.element_mni_mm[0]), 12.0)])
    al = place_array(tm, ArrayConstraints(n_elements=30, min_spacing_mm=4.0, exclude_mask=ex.excluded))
    assert al.n_placed > 0
    assert not ex.covered(al.element_mni_mm)                     # no element on the zone


def test_cap_optimum_avoids_zone():
    from test_cap_placement import _synthetic_field, _keep_all, ROC, APER, G_AZ, DX, TARGET
    field, f_hat, g_hat = _synthetic_field()
    surf_mm = (field.surf_vox - TARGET) * DX
    kw = dict(roc_mm=ROC, aperture_mm=APER, radius_mm=ROC, search_density=0.5, final_density=0.5,
              refine_pose=False, n_az=19, n_el=9)
    free = place_cap_optimal(field, _keep_all, seed_az_deg=G_AZ, seed_el_deg=0.0,
                             az_halfspan_deg=90.0, el_halfspan_deg=40.0, **kw)
    assert free.extras["covered_zones"] == []
    # forbid the brightest window: the optimum must move and stay clear of it
    ex = Exclusion(surf_mm, zone_mask(surf_mm, [Zone("ear", tuple(free.window_mni_mm), 10.0)]), ["ear"])
    s = score_cap_pose(field, free.pose, _keep_all, roc_mm=ROC, aperture_mm=APER, density=0.5,
                       exclusion=ex)
    assert s["covered"] == ["ear"]
    pl = place_cap_optimal(field, _keep_all, seed_az_deg=G_AZ, seed_el_deg=0.0,
                           az_halfspan_deg=90.0, el_halfspan_deg=40.0, exclusion=ex, **kw)
    assert pl.extras["covered_zones"] == []
    assert np.linalg.norm(np.asarray(pl.window_mni_mm) - np.asarray(free.window_mni_mm)) > 10.0


def test_ear_zones_are_lateral_and_symmetric():
    l, r = (np.asarray(z.center_mm) for z in EAR_ZONES)
    assert l[0] < -60 and r[0] > 60 and abs(l[0] + r[0]) < 2 and abs(l[2] - r[2]) < 2
