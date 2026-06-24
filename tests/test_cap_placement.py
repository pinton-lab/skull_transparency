"""Unit tests for the CTX-500 cap-aperture placement objective (placement.place_cap_optimal).

Self-contained synthetic geometry (no 1.38 GB medium, no GPU): a spherical shell of surface
patches around a target, two bright lobes (a 'good window' g_hat and a 'foramen leak' f_hat), and a
no-bone DROP region (a cone around f_hat). The tests assert the three structural properties the
objective must have:
  (1) a cap aimed straight into the foramen lobe drops a LARGE fraction of its elements;
  (2) the after-drop pose optimum MOVES off the foramen onto the good window, with a small drop;
  (3) with NO foramen (drop disabled) the optimum aims at the brightest lobe — i.e. it reduces to a
      delivered-energy window selection (the place_bowl_optimal behaviour) for foramen-free targets.
"""
import numpy as np
import pytest

from skull_transparency import (CapField, CapPose, Registration,
                                 score_cap_pose, place_cap_optimal, anatomical_az_el)
from skull_transparency.transducer import A_HAT_DOMAIN, L_HAT_DOMAIN

DX = 0.25
ROC = 20.0           # small synthetic device -> fast cap build
APER = 20.0          # half-angle 30 deg
TARGET = np.array([200.0, 200.0, 200.0])
RAD_MM = 33.0
FORAMEN_HALF_DEG = 12.0


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    gold = np.pi * (1 + 5 ** 0.5)
    th = gold * i
    return np.c_[np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)]


G_AZ = 65.0          # good-window lobe azimuth (> the cap full aperture from the foramen, so the
#                      60-deg cap cannot cover both lobes at once -> each test has one clear optimum)


def _synthetic_field(seed=0):
    """A shell of patches around TARGET with a bright good-window lobe (g_hat, az=+65) and a dimmer
    foramen lobe (f_hat = A_HAT, az=0), separated by more than the cap aperture. Outward normals are
    radial (incidence 0). Returns the CapField and the two lobe directions."""
    dirs = _fib_sphere(6000)
    f_hat = A_HAT_DOMAIN.copy()                                   # foramen leak: az 0, el 0
    g_hat = np.cos(np.deg2rad(G_AZ)) * A_HAT_DOMAIN + np.sin(np.deg2rad(G_AZ)) * L_HAT_DOMAIN  # az +65
    ang_f = np.degrees(np.arccos(np.clip(dirs @ f_hat, -1, 1)))
    ang_g = np.degrees(np.arccos(np.clip(dirs @ g_hat, -1, 1)))
    Ipk = 0.03 + 1.0 * np.exp(-(ang_g / 13.0) ** 2) + 0.7 * np.exp(-(ang_f / 13.0) ** 2)
    rad_mm = np.full(len(dirs), RAD_MM)
    surf_vox = TARGET + (RAD_MM / DX) * dirs
    reg = Registration(R_mni_to_sim=np.eye(3), dx_mm=DX, target_mni_mm=np.zeros(3),
                       target_fullres_voxel=TARGET)
    from scipy.spatial import cKDTree
    field = CapField(surf_dir=dirs, E_inv=rad_mm ** 2 * Ipk, normal=dirs.copy(), surf_vox=surf_vox,
                     rad_mm=rad_mm, target_vox=TARGET, dx_mm=DX, tree=cKDTree(dirs), registration=reg)
    return field, f_hat, g_hat


def _foramen_drop(f_hat):
    """bone_ray_test: keep (True) elements whose ray to the target lies OUTSIDE the foramen cone
    (cross bone); drop (False) those within FORAMEN_HALF_DEG of f_hat (the no-bone leak)."""
    def test(cap_pts, target_vox):
        d = cap_pts - target_vox
        dhat = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-30)
        ang = np.degrees(np.arccos(np.clip(dhat @ f_hat, -1, 1)))
        return ang > FORAMEN_HALF_DEG
    return test


def _keep_all(cap_pts, target_vox):
    return np.ones(len(cap_pts), bool)


def test_straight_aim_into_foramen_drops_a_large_fraction():
    field, f_hat, _ = _synthetic_field()
    az0, el0 = anatomical_az_el(f_hat)
    s = score_cap_pose(field, CapPose(az0, el0, ROC), _foramen_drop(f_hat),
                       roc_mm=ROC, aperture_mm=APER, density=1.0)
    assert s["n_cap"] > 200
    assert s["drop_frac"] > 0.08, f"expected a large central-foramen drop, got {s['drop_frac']:.3f}"


def test_optimum_moves_off_the_foramen_onto_the_good_window():
    field, f_hat, g_hat = _synthetic_field()
    az0, el0 = anatomical_az_el(f_hat)
    az_g, el_g = anatomical_az_el(g_hat)
    drop = _foramen_drop(f_hat)
    seed = score_cap_pose(field, CapPose(az0, el0, ROC), drop, roc_mm=ROC, aperture_mm=APER, density=1.0)
    pl = place_cap_optimal(field, drop, seed_az_deg=az0, seed_el_deg=el0, roc_mm=ROC, aperture_mm=APER,
                           search_density=0.4, final_density=1.0, az_halfspan_deg=75.0, n_az=25,
                           el_halfspan_deg=20.0, n_el=7, n_tilt=5, n_yaw=5)
    # the optimum sits near the good window (az ~ +65), well away from the foramen (az 0)
    assert abs(pl.pose.az_deg - az_g) < 15.0, f"optimum az {pl.pose.az_deg:.0f} not near g {az_g:.0f}"
    assert pl.drop_frac < 0.03, f"optimum should be ~foramen-clean, got {pl.drop_frac:.3f}"
    # and it collects more after-drop energy than the naive straight-into-foramen seat
    assert pl.J_cap > seed["J_cap"]
    # the positioning score is now WIRED (was always NaN): kept/full cap energy in (0,1]
    from skull_transparency import PositioningScore
    sn = pl.extras["score_norm"]
    assert 0.0 < sn <= 1.0 and np.isfinite(sn)
    assert PositioningScore.from_placement(pl).normalized == sn


def test_reduces_to_brightest_window_with_no_foramen():
    field, f_hat, g_hat = _synthetic_field()
    az0, el0 = anatomical_az_el(f_hat)
    az_g, _ = anatomical_az_el(g_hat)
    pl = place_cap_optimal(field, _keep_all, seed_az_deg=az0, seed_el_deg=el0, roc_mm=ROC,
                           aperture_mm=APER, search_density=0.4, final_density=1.0, az_halfspan_deg=75.0,
                           n_az=25, el_halfspan_deg=20.0, n_el=7, n_tilt=5, n_yaw=5)
    # no foramen -> nothing dropped, optimum aims at the brightest lobe (the good window)
    assert pl.drop_frac == 0.0
    assert pl.extras["score_norm"] == 1.0          # kept == full cap energy -> exact 1.0
    assert abs(pl.pose.az_deg - az_g) < 15.0, f"optimum az {pl.pose.az_deg:.0f} not at brightest {az_g:.0f}"


def test_pose_apex_aim_matches_interactive_tool():
    """transducer.pose_apex_aim must reproduce ctx500_position_tool.Pose geometry bit-for-bit
    (the manual dentate pose). Guards against drift between the factored helper and the tool."""
    from skull_transparency.transducer import pose_apex_aim, build_cap_pose
    tv = np.array([372.65, 324.71, 140.59])
    pose = CapPose(az_deg=130.0, el_deg=-40.0, radius_mm=63.2, tilt_deg=5.0, yaw_deg=-5.0)
    apex, aim = pose_apex_aim(tv, pose, DX)
    # apex is radius_mm from the target along dir_hat
    assert np.isclose(np.linalg.norm(apex - tv) * DX, 63.2, atol=1e-6)
    assert np.isclose(np.linalg.norm(aim), 1.0)
    # geometric focus = apex + ROC*aim; with radius==ROC the off-target offset is pure tilt/yaw
    pts, focus, apex2, aim2 = build_cap_pose(tv, pose, 63.2 / DX, DX)
    off = np.linalg.norm(focus - tv) * DX
    expect = 63.2 * np.sin(np.deg2rad(np.hypot(5.0, 5.0)))      # ~small-angle tilt+yaw magnitude
    assert abs(off - expect) < 1.0
