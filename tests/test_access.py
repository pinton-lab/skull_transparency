"""Accessibility criteria: what a transducer may actually be placed on.

Built on the synthetic bundle (a closed spherical bone shell), which is then deliberately
damaged -- a hole punched through it, a second shell fragment placed in front of it -- so
each criterion can be checked against a known ground truth.
"""
import numpy as np
import pytest

import skull_transparency as st


DIRS = {"+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0]),
        "+x": np.array([1.0, 0.0, 0.0])}


def _bundle(tmp_path, hole_dir=None, hole_deg=20.0, blocker_dir=None, blocker_deg=15.0):
    """Synthetic bundle, optionally with a cone-shaped hole in the shell and/or a detached
    'mandible' fragment sitting outside it along ``blocker_dir``."""
    d = st.make_synthetic_bundle(tmp_path / "b")
    c = np.load(d / "skull_fullres_c.npy")
    reg = st.Registration.from_json(d / "registration.json")
    tgt = np.asarray(reg.target_fullres_voxel, float)
    idx = np.indices(c.shape, dtype=float)
    v = idx - tgt[:, None, None, None]
    r = np.linalg.norm(v, axis=0)
    u = v / np.maximum(r, 1e-9)
    if hole_dir is not None:
        cone = (np.tensordot(np.asarray(hole_dir, float), u, axes=(0, 0))
                >= np.cos(np.deg2rad(hole_deg)))
        c = np.where(cone, 1540.0, c).astype(np.float32)     # punch through the shell
    if blocker_dir is not None:
        cone = (np.tensordot(np.asarray(blocker_dir, float), u, axes=(0, 0))
                >= np.cos(np.deg2rad(blocker_deg)))
        N = c.shape[0]
        shell2 = (r >= N * 0.40) & (r <= N * 0.44)           # a second layer further out
        c = np.where(cone & shell2, 2900.0, c).astype(np.float32)
    np.save(d / "skull_fullres_c.npy", c)
    return d


def _tmap(d):
    b = st.load_bundle(d)
    return st.compute_transparency_map(b), b


# --------------------------------------------------------------------- escape directions
def test_closed_shell_has_no_openings(tmp_path):
    """A skull with no holes escapes nowhere -- the foramen criteria must find nothing."""
    _, b = _tmap(_bundle(tmp_path))
    _dirs, escaped, openings = st.escape_directions(b, reach_mm=200.0)
    assert not escaped.any()
    assert openings == []


@pytest.mark.parametrize("name", ["+z", "-z", "+x"])
def test_hole_is_found_with_the_right_axis_and_size(tmp_path, name):
    """A punched cone shows up as ONE opening, on-axis and of the expected solid angle."""
    axis = DIRS[name]
    _, b = _tmap(_bundle(tmp_path, hole_dir=axis, hole_deg=20.0))
    _dirs, escaped, openings = st.escape_directions(b, reach_mm=200.0, n_dirs=4000)
    assert len(openings) == 1
    o = openings[0]
    assert float(o.axis_grid @ axis) > 0.995                     # points down the hole
    # a 20 deg cone subtends 2*pi*(1-cos20) sr = 3.0% of the sphere. Voxel rounding can only
    # ERODE the measured cone (edge rays clip bone), never widen it.
    geom = 0.5 * (1 - np.cos(np.deg2rad(20.0)))
    assert 0.6 * geom <= o.fraction <= 1.15 * geom
    assert 15.0 <= o.half_angle_deg <= 24.0


def test_two_holes_cluster_separately(tmp_path):
    """Opposite holes must not be merged into one opening by the angular linkage."""
    d = _bundle(tmp_path, hole_dir=DIRS["+z"], hole_deg=15.0)
    c = np.load(d / "skull_fullres_c.npy")
    reg = st.Registration.from_json(d / "registration.json")
    tgt = np.asarray(reg.target_fullres_voxel, float)
    v = np.indices(c.shape, dtype=float) - tgt[:, None, None, None]
    u = v / np.maximum(np.linalg.norm(v, axis=0), 1e-9)
    cone = np.tensordot(DIRS["-z"], u, axes=(0, 0)) >= np.cos(np.deg2rad(10.0))
    np.save(d / "skull_fullres_c.npy", np.where(cone, 1540.0, c).astype(np.float32))
    _dirs, _esc, openings = st.escape_directions(st.load_bundle(d), reach_mm=200.0)
    assert len(openings) == 2
    assert openings[0].fraction > openings[1].fraction           # sorted largest first
    assert float(openings[0].axis_grid @ openings[1].axis_grid) < -0.9   # antipodal


# --------------------------------------------------------------------- the mask criteria
def test_open_pad_drops_the_rim_and_neck_cone_drops_more(tmp_path):
    """Rim patches next to a hole are dropped, and the neck cone drops a wider region."""
    axis = DIRS["+z"]
    d = _bundle(tmp_path, hole_dir=axis, hole_deg=20.0)
    tmap, b = _tmap(d)
    base, _ = st.access_mask(tmap, b, standoff_mm=60.0, open_pad_deg=None,
                             neck_cone_deg=None, min_bone_mm=None, max_layers=None)
    pad, ipad = st.access_mask(tmap, b, standoff_mm=60.0, open_pad_deg=10.0,
                               neck_cone_deg=None, min_bone_mm=None, max_layers=None)
    neck, ineck = st.access_mask(tmap, b, standoff_mm=60.0, open_pad_deg=10.0,
                                 neck_cone_deg=45.0, min_bone_mm=None, max_layers=None)
    assert base.all()
    assert ipad.dropped_open > 0 and pad.sum() < base.sum()
    assert ineck.dropped_neck > 0 and neck.sum() < pad.sum()
    # everything dropped sits on the hole side; nothing on the far side is touched
    cos = np.asarray(tmap.rhat, float) @ axis
    assert cos[~neck].min() > np.cos(np.deg2rad(60.0))
    assert neck[cos < 0].all()


def test_max_layers_drops_patches_behind_a_blocker(tmp_path):
    """A second shell in front of a window means the beam crosses it -- drop those patches."""
    axis = DIRS["+x"]
    tmap, b = _tmap(_bundle(tmp_path, blocker_dir=axis, blocker_deg=15.0))
    keep, info = st.access_mask(tmap, b, standoff_mm=90.0, max_layers=1, min_bone_mm=None,
                                open_pad_deg=None, neck_cone_deg=None)
    assert info.dropped_layers > 0
    cos = np.asarray(tmap.rhat, float) @ axis
    assert not keep[cos > 0.99].any()                 # straight down the blocker: dropped
    assert keep[cos < 0.0].all()                      # opposite side: untouched
    assert info.n_layers[cos > 0.99].min() >= 2


def test_min_bone_drops_a_thin_lip(tmp_path):
    """Patches whose radial crosses almost no bone are the classic foramen leak."""
    tmap, b = _tmap(_bundle(tmp_path, hole_dir=DIRS["+z"], hole_deg=20.0))
    _keep, info = st.access_mask(tmap, b, standoff_mm=60.0, min_bone_mm=1e6,
                                 max_layers=None, open_pad_deg=None, neck_cone_deg=None)
    assert info.dropped_thin == info.n_total          # an absurd floor drops everything
    _keep2, info2 = st.access_mask(tmap, b, standoff_mm=60.0, min_bone_mm=0.0,
                                   max_layers=None, open_pad_deg=None, neck_cone_deg=None)
    assert info2.dropped_thin == 0


# --------------------------------------------------------------------- cap clearance
def test_cap_clearance_shrinks_with_a_bigger_dish(tmp_path):
    """The binding constraint on a small skull: a wide dish cannot be seated anywhere."""
    tmap, b = _tmap(_bundle(tmp_path))
    # standoff != roc, so a wider aperture swings the rim back toward the shell (an f/0.25
    # dish at 50 mm reaches radii down to ~35 mm, inside the 34-43 mm shell)
    kw = dict(standoff_mm=50.0, max_layers=None, min_bone_mm=None,
              open_pad_deg=None, neck_cone_deg=None)
    small, i_small = st.access_mask(tmap, b, cap_roc_mm=25.0, cap_aperture_mm=10.0, **kw)
    big, i_big = st.access_mask(tmap, b, cap_roc_mm=25.0, cap_aperture_mm=49.0, **kw)
    assert i_small.dropped_cap < i_big.dropped_cap
    assert big.sum() < small.sum()


def test_off_grid_is_free_space_not_obstruction(tmp_path):
    """A dish overhanging the domain is in air, not in bone -- rejecting it would reject
    windows for the shape of the simulation box. Opt in with allow_off_grid=False."""
    tmap, b = _tmap(_bundle(tmp_path))
    kw = dict(standoff_mm=110.0, roc_mm=110.0, aperture_mm=60.0)   # well outside the 120 mm box
    ok_free, n_bone, n_oob = st.cap_clearance(tmap, b, **kw)
    assert n_oob.max() > 0 and n_bone.max() == 0
    assert ok_free.all()
    ok_strict, _b, _o = st.cap_clearance(tmap, b, allow_off_grid=False, **kw)
    assert not ok_strict.all()


def test_cap_clearance_flags_a_dish_that_would_hit_the_blocker(tmp_path):
    """A dish seated over the blocker collides with it; the far side stays clear."""
    axis = DIRS["+x"]
    tmap, b = _tmap(_bundle(tmp_path, blocker_dir=axis, blocker_deg=25.0))
    # seat the dish AT the blocker's radius (48-52.8 mm): with standoff == roc every cap
    # point lies on the sphere of that radius about the target, so it intersects the blocker.
    ok, n_bone, _oob = st.cap_clearance(tmap, b, standoff_mm=50.0, roc_mm=50.0, aperture_mm=30.0)
    cos = np.asarray(tmap.rhat, float) @ axis
    assert n_bone[cos > 0.99].max() > 0 and not ok[cos > 0.99].any()
    assert n_bone[cos < -0.5].max() == 0


# --------------------------------------------------------------------- plumbing
def test_mask_feeds_place_bowl_and_confines_the_window(tmp_path):
    """The mask is accepted by BowlConstraints and the chosen window respects it."""
    tmap, b = _tmap(_bundle(tmp_path, hole_dir=DIRS["+z"], hole_deg=25.0))
    mask, info = st.access_mask(tmap, b, standoff_mm=60.0, neck_cone_deg=50.0)
    assert 0 < info.n_legal < info.n_total
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60.0, bowl_radius_mm=10.0,
                                                legal_mask=mask))
    P = np.asarray(tmap.surf_mni_mm(), float)
    j = int(np.argmin(np.linalg.norm(P - np.asarray(pl.window_center_mni_mm, float), axis=1)))
    assert mask[j]
    assert "accessible" in info.summary()


def test_criteria_are_individually_switchable(tmp_path):
    """All criteria off == everything legal, so each drop is attributable."""
    tmap, b = _tmap(_bundle(tmp_path, hole_dir=DIRS["+z"]))
    keep, info = st.access_mask(tmap, b, standoff_mm=60.0, max_layers=None, min_bone_mm=None,
                                open_pad_deg=None, neck_cone_deg=None)
    assert keep.all() and info.n_legal == info.n_total
    assert (info.dropped_layers, info.dropped_thin, info.dropped_open,
            info.dropped_neck, info.dropped_cap) == (0, 0, 0, 0, 0)


def test_neck_cone_guards_every_significant_opening_not_just_the_largest(tmp_path):
    """Two openings of different size must BOTH be guarded.

    Guarding only the largest is the bug this test exists for: a real skull has the foramen
    magnum caudally and a basicranial gap ventrally, and a window that scores well through
    the second one points straight at the animal's throat."""
    d = _bundle(tmp_path, hole_dir=DIRS["+z"], hole_deg=22.0)      # the "foramen": larger
    c = np.load(d / "skull_fullres_c.npy")
    reg = st.Registration.from_json(d / "registration.json")
    tgt = np.asarray(reg.target_fullres_voxel, float)
    v = np.indices(c.shape, dtype=float) - tgt[:, None, None, None]
    u = v / np.maximum(np.linalg.norm(v, axis=0), 1e-9)
    cone = np.tensordot(DIRS["+x"], u, axes=(0, 0)) >= np.cos(np.deg2rad(15.0))
    np.save(d / "skull_fullres_c.npy", np.where(cone, 1540.0, c).astype(np.float32))

    tmap, b = _tmap(d)
    _dirs, _esc, openings = st.escape_directions(b, reach_mm=200.0)
    assert len(openings) == 2 and openings[0].fraction > openings[1].fraction

    keep, info = st.access_mask(tmap, b, standoff_mm=60.0, neck_cone_deg=40.0,
                                max_layers=None, min_bone_mm=None, open_pad_deg=None)
    assert len(info.neck_openings) == 2
    rhat = np.asarray(tmap.rhat, float)
    for axis in (DIRS["+z"], DIRS["+x"]):                 # BOTH axes must be cleared out
        assert not keep[rhat @ axis > np.cos(np.deg2rad(20.0))].any()
    assert keep[rhat @ DIRS["-z"] > 0.9].all()            # an unopened direction survives

    # the size floor keeps a tiny opening from sterilising the skull
    _k2, info2 = st.access_mask(tmap, b, standoff_mm=60.0, neck_cone_deg=40.0,
                                neck_min_fraction=0.5, max_layers=None, min_bone_mm=None,
                                open_pad_deg=None)
    assert info2.neck_openings == [] and info2.dropped_neck == 0
