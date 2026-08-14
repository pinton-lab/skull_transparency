#!/usr/bin/env python3
"""MOUSE case 1: the TIPS bowl at 1 MHz on the middle of the cerebellum.

One omnidirectional time-reversal source at the cerebellum centroid of the Maga (UW 4K)
dry-skull microCT radiates outward through the mouse skull; the single outward solve gives
the whole-skull transparency map for that target, and the placement step seats the TIPS
bowl on it.

Inputs (staged by the TUBA mouse pipeline, ``tuba.species.mouse``):
  * ``maga_skull_aligned_native_200um.nii.gz`` -- whole-skull intensity, aligned-native RAS,
    ramped to sound speed with the same Section-8 map the rest of the mouse work uses
    (I_LOW 5000 -> water 1540 m/s, I_HIGH 25000 -> cortical bone 2900 m/s);
  * ``allen_annotation_in_maga.nii.gz`` -- Allen CCFv3 annotation warped into that same
    frame, from which the cerebellum centroid is computed (below).

Two knobs differ from the human cases, both because a mouse is small:

  1. PPW is set by BONE THICKNESS, not by the wavelength. At 1 MHz the water wavelength is
     1.54 mm and 6 ppw would give dx = 0.257 mm -- but the mouse calvaria is only
     ~0.2-0.5 mm thick, i.e. one voxel. PPW_ = 12 (dx = 0.128 mm) resolves it in 2-4
     voxels and still costs almost nothing on this domain. (This is the opposite of the
     human runs, where 6 ppw is already fine because the skull is ~7 mm thick.)

  2. The placement FOOTPRINT comes from the bowl's cone half-angle, not from its aperture
     radius. ``TransducerSpec.to_bowl_constraints`` sets bowl_radius_mm = aperture/2, which
     is right for a human (the skull sits about one focal length from the target, so the
     cone crosses it at roughly the aperture radius) and badly wrong here: the TIPS
     aperture radius is 46 mm while the mouse skull sits ~5 mm from the cerebellum, so the
     cone crosses it at r*sin(half_angle) ~ 2.7 mm. Passing 46 mm would put the ENTIRE
     25 mm head inside every candidate footprint and the window search would go flat. The
     footprint radius is therefore derived here (:func:`footprint_radius_mm`).

Run (a few minutes on one GPU):

    GPU=0 python run_mouse_tips_cerebellum.py            # solve + extract + place
    GPU=0 python run_mouse_tips_cerebellum.py solve      # phases separately
    OUT=... SCRATCH=... python run_mouse_tips_cerebellum.py extract
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import skull_transparency as st
from skull_transparency.transducer_spec import TransducerSpec
from skull_transparency.sim.prepare import build_brain_center_run

# --------------------------------------------------------------------------- inputs
REG = Path(os.environ.get("TUBA_MOUSE_REG_DIR", "/celerina/gfp/mfs/mouse_therapy/registration"))
SKULL_NII = REG / "maga_skull_aligned_native_200um.nii.gz"    # whole-skull intensity, RAS
ANNOT_NII = REG / "allen_annotation_in_maga.nii.gz"           # Allen CCFv3 in the same frame

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("SCRATCH", "/dev/shm/mouse_tips_cb"))   # transient solver deck
OUT = Path(os.environ.get("OUT", HERE / "bundle"))                    # the (small) Field Bundle

# maga Section-8 intensity -> sound-speed ramp (shared with the rest of the mouse work)
I_LOW, I_HIGH = 5000.0, 25000.0
C_WATER, C_BONE_MAX = 1540.0, 2900.0
BONE_THRESHOLD = 1800.0             # m/s; calvarial-surface cutoff for this c-map
INPUT_FRAME = "maga_aligned_native_ras_mm"

# --------------------------------------------------------------------------- target
#: Allen CCFv3 structure IDs of the cerebellum: every descendant of CB (id 512) in the
#: Allen structure graph (api.brain-map.org structure_graph_download/1.json), embedded so
#: the run needs no network. 20 of these appear in the warped annotation.
CEREBELLUM_IDS = (
    91, 512, 519, 528, 645, 846, 912, 920, 928, 936, 944, 951,
    957, 968, 976, 984, 989, 992, 1001, 1007, 1017, 1025, 1033, 1041,
    1049, 1056, 1064, 1073, 1091, 1143, 1144, 1145, 10672, 10673, 10674, 10675,
    10676, 10677, 10678, 10679, 10680, 10681, 10682, 10683, 10684, 10685, 10686, 10687,
    10688, 10689, 10690, 10691, 10692, 10705, 10706, 10707, 10708, 10709, 10710, 10711,
    10712, 10713, 10714, 10715, 10716, 10717, 10718, 10719, 10720, 10721, 10722, 10723,
    10724, 10725, 10726, 10727, 10728, 10729, 10730, 10731, 10732, 10733, 10734, 10735,
    10736, 10737, 589508455,
)

#: Middle of the cerebellum in the Maga aligned-native RAS frame (mm): the centroid of the
#: warped Allen cerebellum annotation (40.8 mm^3 recovered). Midline, 4.5 mm caudal of the
#: brain center used by the whole-skull baseline. Recomputed at run time and asserted.
CEREBELLUM_CENTER_RAS_MM = (-0.06, -11.60, 1.41)

# --------------------------------------------------------------------------- device
#: TIPS (Therapeutic Imaging Probe and Sonication, Philips): spherical bowl, 80 mm radius
#: of curvature, annular aperture 20.5-46 mm radius in 8 concentric rings, driven here at
#: 1 MHz. The central hole is not modelled by TransducerSpec; it does not enter the window
#: search (which uses the cone half-angle), only the drive.
TIPS = TransducerSpec(
    f0_hz=1e6, geometry="annular", roc_mm=80.0, aperture_mm=92.0, n_rings=8,
    c0_ms=C_WATER, ppw=12.0, acceptance_angle_deg=35.0,
)
SURROUND_MM = float(os.environ.get("SURROUND_MM", 6.0))


def cerebellum_center_ras_mm(verbose=True):
    """Centroid of the Allen cerebellum warped into the Maga skull frame (world RAS mm)."""
    ann = nib.load(str(ANNOT_NII))
    A = np.asarray(ann.affine, float)
    lab = np.asarray(ann.dataobj).astype(np.int64)
    mask = np.isin(lab, np.asarray(CEREBELLUM_IDS))
    if not mask.any():
        raise SystemExit(f"no cerebellum labels found in {ANNOT_NII}")
    idx = np.array(np.nonzero(mask), float)
    ctr = A[:3, :3] @ idx.mean(1) + A[:3, 3]
    if verbose:
        vol = mask.sum() * abs(np.linalg.det(A[:3, :3]))
        print(f"  cerebellum: {int(mask.sum())} voxels = {vol:.1f} mm^3, "
              f"centroid RAS {np.round(ctr, 2)} mm")
    off = float(np.linalg.norm(ctr - np.asarray(CEREBELLUM_CENTER_RAS_MM, float)))
    assert off < 0.3, f"cerebellum centroid moved {off:.2f} mm from the recorded constant"
    return ctr


def footprint_radius_mm(spec, tmap, pctile=30.0):
    """Radius on the SKULL that the bowl's cone actually covers, ``r*sin(half_angle)``.

    ``r`` is the target-to-surface distance over the near patches (``pctile``-th percentile
    of the radius distribution — the part of the skull the beam can reach). Equals the
    aperture radius only when the skull sits at the focal length, which is true for a human
    head and false by ~17x for a mouse under the TIPS."""
    r = float(np.percentile(np.asarray(tmap.rad_mm, float), pctile))
    return r * float(np.sin(np.radians(spec.half_angle_deg))), r


def _c_map():
    img = nib.load(str(SKULL_NII))
    v = np.asarray(img.dataobj, dtype=np.float32)
    t = np.clip((v - I_LOW) / (I_HIGH - I_LOW), 0.0, 1.0).astype(np.float32)
    return (C_WATER + t * (C_BONE_MAX - C_WATER)).astype(np.float32), np.asarray(img.affine, float)


def solve():
    c_map, affine = _c_map()
    center = cerebellum_center_ras_mm()
    print(f"  c-map {c_map.shape} @ {abs(affine[0,0])*1e3:.0f} um -> grid dx {TIPS.dx_mm:.3f} mm "
          f"({TIPS.ppw:.0f} ppw at {TIPS.f0_hz/1e6:.0f} MHz)")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(c_map, affine, TIPS, SCRATCH, center_phys_mm=center,
                                 bone_threshold=BONE_THRESHOLD, surround_mm=SURROUND_MM,
                                 input_frame=INPUT_FRAME)
    meta = json.loads((Path(sim) / "meta.json").read_text())
    gs = meta.get("grid_shape", [meta["N"]] * 3)
    print(f"  sim tree {sim}  grid {gs[0]}x{gs[1]}x{gs[2]}  nT={meta.get('nT')}")

    os.environ.setdefault("FULLWAVE2_BIN", "/celerina/gfp/mfs/fullwave2-ultra/bin/bench_3d_opt")
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
    from skull_transparency.sim.launchers import launch_outward
    outdir = launch_outward(str(sim), str(SCRATCH), run_solver=True, recorder="shell")
    if not (Path(outdir) / "SUCCESS").exists():
        raise SystemExit("solver did not write SUCCESS; check the run log")
    print(f"SOLVE DONE -> {outdir}")
    return outdir


def extract():
    from skull_transparency.sim.extract import extract_bundle
    OUT.mkdir(parents=True, exist_ok=True)
    out = extract_bundle(SCRATCH / "outward", OUT, SCRATCH, bone_threshold=BONE_THRESHOLD)
    print(f"EXTRACT DONE -> {out}")
    g = SCRATCH / "outward" / "genout.dat"
    if g.exists():
        sz = g.stat().st_size / 1e9
        g.unlink()
        print(f"  removed scratch genout ({sz:.1f} GB)")
    return out


def place():
    """Seat the TIPS on the map and write the report, with the cone-derived footprint."""
    tmap = st.compute_transparency_map(st.load_bundle(OUT))
    foot_mm, r_mm = footprint_radius_mm(TIPS, tmap)
    print(f"  skull at r~{r_mm:.1f} mm from the target; TIPS half-angle "
          f"{TIPS.half_angle_deg:.1f} deg -> footprint radius {foot_mm:.1f} mm "
          f"(aperture radius is {TIPS.aperture_mm/2:.0f} mm — {TIPS.aperture_mm/2/foot_mm:.0f}x too big here)")
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=TIPS.roc_mm,
                                                bowl_radius_mm=foot_mm,
                                                theta_max_deg=TIPS.acceptance_angle_deg))
    print(f"  window RAS {np.round(pl.window_center_mni_mm, 2)} mm, "
          f"incidence {pl.incidence_deg:.1f} deg, {pl.n_footprint_patches} patches in footprint")
    from skull_transparency.report import write_report
    rep = write_report(tmap, pl, HERE / "report_mouse_tips_cerebellum.pdf",
                       target_name="cerebellum (middle)", bowl_radius_mm=foot_mm,
                       theta_max_deg=TIPS.acceptance_angle_deg,
                       title="Mouse — TIPS 1 MHz on the middle of the cerebellum")
    print(f"REPORT -> {rep}")
    return pl


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("solve", "all"):
        solve()
    if cmd in ("extract", "all"):
        extract()
    if cmd in ("place", "all"):
        place()


if __name__ == "__main__":
    main()
