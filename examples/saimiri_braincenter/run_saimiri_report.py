#!/usr/bin/env python3
"""Placement report for the Saimiri brain-center bundle, in the style of the human and
mouse reports (``skull_transparency.report.write_report``).

Three things are wired in that a bare ``skull-transparency report`` would get wrong on this
subject, all for the same reason -- a squirrel monkey is small and this is a bare dry skull:

1. **Footprint from the cone, not the aperture.** ``BowlConstraints`` defaults the moving
   window to the aperture radius, which is right for a human (the skull sits about a focal
   length from the target, so the cone crosses it at roughly the aperture radius) and wrong
   here by ~2x: the skull is 18.6 mm from the brain center, so a 25.4 deg half-angle cone
   crosses it at 8.0 mm, not 15 mm. Passing 15 mm puts a fifth of the whole skull inside
   every candidate footprint and flattens the window search. Same correction the mouse TIPS
   case needed (:func:`footprint_radius_mm`).

2. **Accessibility exclusions** (:mod:`skull_transparency.access`). Unconstrained placement
   maximises coupling over bone a transducer can never be seated against. On this skull
   that means a ventral basicranial window -- best-coupled, and aimed through the animal's
   throat. The mask rules out multi-layer beam paths (mandible, zygomatic arch, tympanic
   bulla), thin-bone foramen lips, the rims of the open apertures, cones about every
   significant opening (the animal's neck and pharynx lie beyond them), and windows where
   the dish itself would collide with bone.

3. **Brain from the atlas.** VALiDATe29 warped into this skull's frame supplies the
   anatomy section, so the target is drawn inside the real brain rather than as a sphere.

    python run_saimiri_report.py            # -> report_saimiri_braincenter.pdf (+ .html)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import skull_transparency as st                                       # noqa: E402
from skull_transparency.report import write_report                    # noqa: E402

BUNDLE = HERE / "bundle"
REG = Path("/celerina/gfp/mfs/tuba/saimiri/registration")
BRAINMASK_NII = REG / "validate29_brainmask_in_saimiri.nii.gz"        # VALiDATe29 brain, 24.9 mL
OUT = HERE / "report_saimiri_braincenter.pdf"
FWD = HERE / "forward"                                        # run_saimiri_forward.py (GPU)
MOVIE = HERE / "propagation_saimiri_braincenter.mp4"     # make_saimiri_propagation_movie.py (GPU)

# the small-NHP 1 MHz bowl this study positions (build_saimiri_braincenter.SPEC)
ROC_MM, APERTURE_MM = 35.0, 30.0
THETA_MAX_DEG = 35.0
NECK_CONE_DEG = 45.0

HALF_ANGLE_DEG = float(np.degrees(np.arcsin((APERTURE_MM / 2.0) / ROC_MM)))


def footprint_radius_mm(tmap, half_angle_deg=HALF_ANGLE_DEG, pctile=30.0):
    """Radius on the SKULL that the bowl's cone actually covers, ``r*sin(half_angle)``.

    ``r`` is the target-to-surface distance over the near patches (the part of the skull the
    beam can reach). Equals the aperture radius only when the skull sits at the focal
    length -- true for a human head, false by ~2x here."""
    r = float(np.percentile(np.asarray(tmap.rad_mm, float), pctile))
    return r * float(np.sin(np.radians(half_angle_deg))), r


def chosen_placement(verbose=True):
    """The placement every other script in this case must agree with.

    Returns ``(bundle, tmap, pl, mask, acc, foot, r30)``. Factored out so the forward
    comparison and the propagation movie drive the SAME window the report quotes -- the
    cone-derived footprint and the accessibility mask are not defaults, so a script that
    rebuilt the placement by hand would silently use a different one."""
    bundle = st.load_bundle(BUNDLE)
    thr = float(bundle.physics["bone_threshold"])
    tmap = st.compute_transparency_map(
        bundle, options=st.TransparencyOptions(bone_threshold=thr))

    foot, r30 = footprint_radius_mm(tmap)
    mask, acc = st.access_mask(tmap, bundle, standoff_mm=ROC_MM, neck_cone_deg=NECK_CONE_DEG,
                               cap_roc_mm=ROC_MM, cap_aperture_mm=APERTURE_MM)
    pl = st.place_bowl(tmap, st.BowlConstraints(
        focal_length_mm=ROC_MM, bowl_radius_mm=foot, theta_max_deg=THETA_MAX_DEG,
        legal_mask=mask))
    if verbose:
        print(f"  skull at r~{r30:.1f} mm from the target; half-angle {HALF_ANGLE_DEG:.1f} deg "
              f"-> footprint radius {foot:.1f} mm (aperture radius {APERTURE_MM/2:.0f} mm is "
              f"{(APERTURE_MM/2)/foot:.1f}x too big here)")
        print("  " + acc.summary())
        for i, o in enumerate(acc.openings[:4]):
            print(f"    opening {i}: {o.fraction*100:5.2f}% of the sphere, half-angle "
                  f"{o.half_angle_deg:4.1f} deg, world axis {np.round(o.axis_world, 2)}")
        print(f"  window {np.round(pl.window_center_mni_mm, 2)} mm, incidence "
              f"{pl.incidence_deg:.1f} deg, {pl.n_footprint_patches} patches in footprint")
    return bundle, tmap, pl, mask, acc, foot, r30


def main():
    bundle, tmap, pl, mask, acc, foot, r30 = chosen_placement()
    thr = float(bundle.physics["bone_threshold"])
    o0 = acc.openings[0] if acc.openings else None
    params = [
        ("Drive frequency", "1.0 MHz", "Convention",
         "whole-skull survey; the neuromod band for this species is 2-4 MHz"),
        ("Grid pitch", f"{bundle.grid['dx_m']*1e3:.3f} mm (6 PPW)", "Derived",
         "c0/f0/ppw; the ~1.1 mm calvarium is 4-6 voxels through bone"),
        ("Sound speed map", "6000 -> 1540 m/s, 35000 -> 2900 m/s", "Assumption",
         "tuba SLAB_RAMP placeholder ramp; the museum microCT is NOT HU-calibrated, so "
         "absolute speeds are nominal cortical-bone endpoints, not measured"),
        ("Intensity knees", "6000 / 35000 counts", "Measured",
         "calibrated from this scan's own histogram (tuba.species.saimiri)"),
        ("Bone threshold", f"{thr:.0f} m/s", "Derived",
         "partial-volume shoulder between the mount/soft-tissue mode (<=5000 counts) and "
         "the cortical plateau (>=10000); NOT the human 2200 default"),
        ("Brain center", f"{np.round(pl.target_mni_mm, 2).tolist()} mm", "Measured",
         "centroid of the curated 25.4 mL endocranial cavity mask"),
        ("Calvarial thickness", "1.1 mm (median bone path from the center)", "Measured",
         "matches the 1.0-1.5 mm comparative literature for Saimiri"),
        ("Bowl", f"ROC {ROC_MM:.0f} mm, aperture {APERTURE_MM:.0f} mm "
                 f"(half-angle {HALF_ANGLE_DEG:.1f} deg)", "Assumption",
         "a plausible small-NHP 1 MHz bowl; no specific device is committed to"),
        ("Footprint radius", f"{foot:.1f} mm", "Derived",
         f"r*sin(half-angle) at r={r30:.1f} mm, NOT the {APERTURE_MM/2:.0f} mm aperture radius"),
        ("Accessible patches", f"{acc.n_legal} of {acc.n_total} "
                               f"({100*acc.n_legal/acc.n_total:.1f}%)", "Derived",
         f"dropped {acc.dropped_layers} multi-layer beam paths, {acc.dropped_thin} thin-bone, "
         f"{acc.dropped_open} open-aperture rim, {acc.dropped_neck} neck/pharynx cone, "
         f"{acc.dropped_cap} no dish clearance"),
        ("Open solid angle", f"{acc.escape_fraction*100:.2f}% of the sphere", "Measured",
         "rays leaving the brain center without meeting bone: the foramen magnum "
         + (f"({o0.fraction*100:.1f}%, caudal) " if o0 else "")
         + "and the ventral basicranial gap; both are guarded by cones"),
        ("Forward focal box", "+/-12 mm half-width", "Derived",
         "the box the peak and FWHM are measured in; it must exceed the AXIAL -6 dB lobe "
         "(~7*lambda*f#^2 ~ 15 mm at f/1.17, 1 MHz), not the lateral one -- a +/-4 mm box "
         "truncates it and the axial FWHM reads as infinite"),
        ("Propagation movie", "volume recorder, modT 8", "Derived",
         "the SAME outward solve as the transparency map, re-run with the volume recorder "
         "so a full field exists to animate (the map itself needs only the surface)"),
        ("Specimen", "Saimiri sp., NMNH USNM 194346", "Measured",
         "MorphoSource media 000116521 (UTCT). GENUS confirmed independently by size: skull "
         "57 x 38 x 48 mm and a 25.4 mL endocranial cavity, against 20-26 mL for Saimiri and "
         "80-110 mL for a macaque. SPECIES is not settled: the MorphoSource taxonomy field "
         "says S. boliviensis, its own description and the DigiMorph page say S. sciureus; "
         "no result here depends on which"),
        ("Specimen state", "bare dry skull", "Measured",
         "no soft tissue, so scalp, pharynx and orbital contents are not modelled and final "
         "accessibility remains a protocol judgement"),
    ]

    rep = write_report(
        tmap, pl, OUT, target_name="brain center",
        bowl_radius_mm=foot, aperture_mm=APERTURE_MM,   # the DISH; foot is its footprint
        theta_max_deg=THETA_MAX_DEG,
        title="Squirrel monkey (Saimiri) — brain-center whole-skull transparency at 1 MHz",
        bundle=bundle, atlas=str(BRAINMASK_NII), atlas_ids=(1,),
        atlas_label="brain (VALiDATe29)", parameters=params, frame_is_mni=False,
        forward=(FWD if FWD.exists() else None),
        movie=(MOVIE if MOVIE.exists() else None),
        movie_caption=(
            "Outward time-reversal wave leaving the brain center and sweeping through the "
            "Saimiri skull (sagittal, coronal and axial through the target; time in "
            "microseconds). Plays in Acrobat and compatible PDF viewers; other readers show "
            "one frame. Regenerate with make_saimiri_propagation_movie.py."))
    print(f"REPORT -> {rep}")
    return rep


if __name__ == "__main__":
    main()
