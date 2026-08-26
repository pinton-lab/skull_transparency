#!/usr/bin/env python3
"""Build the SQUIRREL MONKEY (Saimiri) brain-center whole-skull transparency bundle.

The saimiri analogue of ``mouse_therapy/transparency/build_mouse_braincenter.py`` and of
the human Halle brain-center baseline: one omnidirectional time-reversal source at the
center of the cranial cavity -> a single outward solve illuminates the whole calvaria ->
a neutral, 1/r^2-corrected whole-skull map of where this skull transmits.

Inputs (tuba saimiri pillar, ``../tuba/saimiri/registration``; museum microCT of
*Saimiri sp.* NMNH USNM 194346, MorphoSource media 000116521, UTCT ACTIS; the record's
taxonomy field says *S. boliviensis* while its own description and the DigiMorph page say
*S. sciureus* -- genus is certain, species is not, and nothing here depends on it):
  * sound speed ``c`` -- the ~98 um aligned-native RAS intensity volume ramped through the
    tuba ``SLAB_RAMP`` endpoints (``tuba.species.saimiri``: i_low = BONE_LOW 6000 counts ->
    water, i_high = BONE_RAMP_HIGH 35000 counts -> cortical bone 2900 m/s). Those intensity
    knees are calibrated from this scan's own histogram.
  * brain center -- the centroid of the curated endocranial-cavity mask
    (``saimiri_cranial_cavity.nii.gz``, 26.2 mL), saimiri aligned-native RAS
    (0.21, -7.70, 7.55) mm. The specimen is a DRY museum skull (open at the foramina), so
    the curated mask is used rather than the image-only hole-fill, which would leak (the
    human Halle / mouse lesson).
  * bone cutoff 1700 m/s -- the calvarial-surface threshold for this ramp. NOT the human
    2200 default: on this ramp the histogram bone knee (6000 counts) sits exactly at water,
    so bone runs SLOW; 1700 m/s == 9400 counts, inside the partial-volume shoulder between
    the mount/soft-tissue mode (<= ~5000 counts) and the cortical plateau (>= ~10000).

CAVEAT (inherited from tuba): the museum scan is NOT HU-calibrated, so the intensity->c
ramp is tuba's explicitly-flagged *placeholder* ramp (``hu_acoustics.placeholder_ramp``,
``.placeholder = True``). The GEOMETRY -- shell shape, thickness, aperture, window
ranking -- is real; the absolute sound speeds are nominal cortical-bone endpoints, not
measured. Swap in an HU-calibrated colony CT to make the acoustics quantitative.

Drive: f0 = 1 MHz, ppw 6 -> dx ~= 0.257 mm; the calvarium is ~1.0-1.5 mm thick, i.e. 4-6
grid voxels through bone.

Two phases (mirrors the human/mouse solve/extract split so the big genout is reclaimable):

    SCRATCH=/dev/shm/saim_bc GPU=0 python build_saimiri_braincenter.py solve
    SCRATCH=/dev/shm/saim_bc                python build_saimiri_braincenter.py extract
    # or simply:  GPU=0 python build_saimiri_braincenter.py all
"""
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, "/celerina/gfp/mfs/skull_transparency/src")
from skull_transparency import cavity_mask_centroid          # generic helper (the pkg is species-free)
from skull_transparency.transducer_spec import TransducerSpec
from skull_transparency.sim.prepare import build_brain_center_run

REG = Path("/celerina/gfp/mfs/tuba/saimiri/registration")
SKULL_NII = REG / "saimiri_skull_aligned_98um.nii.gz"        # whole-skull intensity, RAS
CAVITY_NII = REG / "saimiri_cranial_cavity.nii.gz"           # curated endocranial cavity

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("SCRATCH", "/dev/shm/saim_bc"))    # big genout -> fast RAM scratch
OUT = Path(os.environ.get("OUT", HERE / "bundle"))               # the (small) Field Bundle

# tuba.species.saimiri SLAB_RAMP endpoints (uncalibrated museum microCT; geometry only)
I_LOW, I_HIGH = 6000.0, 35000.0        # BONE_LOW, BONE_RAMP_HIGH (16-bit counts)
C_WATER, C_BONE_MAX = 1540.0, 2900.0   # c_water pinned to the solver background C0, not 1500
BONE_THRESHOLD = 1700.0                # m/s; calvarial-surface cutoff for this ramp (see docstring)
INPUT_FRAME = "saimiri_aligned_native_ras_mm"

#: Saimiri brain center in that skull's aligned-native RAS frame (mm): the centroid of the
#: curated endocranial-cavity mask (``saimiri_cranial_cavity``, 26.2 mL -- inside tuba's
#: 20-30 mL QC bracket for a ~22 mL Saimiri brain plus CSF). Recomputed from the mask at run
#: time and asserted to match. Near the midline (x ~ 0) as the aligned frame requires.
#:
#: CHANGED 2026-08-26, from (0.23, -8.07, 7.91) at 25.35 mL. tuba abf948a found the museum
#: scan's field of view ends flush against the specimen along the stack axis (~0.3 mm of
#: margin), leaving the cavity extractor's morphology no headroom, and pads it by
#: AP_PAD_SLICES=30 (~3.6 mm) at each end. That restores BOTH AP poles -- the cavity grew
#: 1.9 mm posteriorly and 1.7 mm anteriorly -- so the centroid moved 0.51 mm, which is
#: lambda/3 at 1 MHz and therefore not ignorable for a time-reversal source planted on it.
#: The pillar had to be regenerated for this: the fix is in tuba's source, but the
#: registration products it feeds are gitignored build artifacts and were three days stale.
SAIMIRI_BRAIN_CENTER_RAS_MM = (0.21, -7.70, 7.55)

PPW = 6.0
F0_HZ = float(os.environ.get("F0_HZ", 1e6))          # 1 MHz whole-skull survey
SURROUND_MM = float(os.environ.get("SURROUND_MM", 6.0))

#: Survey shell, not a device: the brain-center run records the WHOLE calvaria, so only
#: dx (= c0/f0/ppw) and the ppw/2 recorder spacing come from the spec. ROC/aperture are a
#: plausible small-NHP 1 MHz bowl so the same spec can drive placement/forward later.
SPEC = TransducerSpec(f0_hz=F0_HZ, geometry="bowl", roc_mm=35.0, aperture_mm=30.0, ppw=PPW)


def _ramp_c(counts):
    t = np.clip((counts - I_LOW) / (I_HIGH - I_LOW), 0.0, 1.0).astype(np.float32)
    return (C_WATER + t * (C_BONE_MAX - C_WATER)).astype(np.float32)


def _load_inputs():
    img = nib.load(str(SKULL_NII))
    affine = np.asarray(img.affine, float)
    c_map = _ramp_c(np.asarray(img.dataobj, dtype=np.float32))
    cav = nib.load(str(CAVITY_NII))
    center = cavity_mask_centroid(np.asarray(cav.dataobj), np.asarray(cav.affine, float))
    off = float(np.linalg.norm(center - np.asarray(SAIMIRI_BRAIN_CENTER_RAS_MM, float)))
    assert off < 0.25, f"cavity centroid {center} drifted {off:.2f} mm from the recorded constant"
    print(f"  c-map {c_map.shape} dx~{abs(affine[0,0])*1e3:.1f} um  c[{c_map.min():.0f},{c_map.max():.0f}]")
    print(f"  bone fraction (c > {BONE_THRESHOLD:.0f}) {float((c_map > BONE_THRESHOLD).mean()):.4f}")
    print(f"  brain center (cavity centroid) RAS {np.round(center,2)} mm  "
          f"(|delta| vs constant {off:.2f} mm)")
    return c_map, affine, center


def solve():
    c_map, affine, center = _load_inputs()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(
        c_map, affine, SPEC, SCRATCH,
        center_phys_mm=center, bone_threshold=BONE_THRESHOLD,
        surround_mm=SURROUND_MM, input_frame=INPUT_FRAME)
    import json
    meta = json.loads((Path(sim) / "meta.json").read_text())
    print(f"  sim tree {sim}  (grid {meta.get('shape') or meta['N']}, dx {meta['dX_m']*1e3:.3f} mm, "
          f"source @ grid {np.round(meta['dent_grid'],1)}, n_array={meta['n_array']})")

    os.environ["FULLWAVE2_BIN"] = "/celerina/gfp/mfs/fullwave2-ultra/bin/bench_3d_opt"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
    from skull_transparency.sim.launchers import launch_outward
    outdir = launch_outward(str(sim), str(SCRATCH), run_solver=True)
    ok = (Path(outdir) / "SUCCESS").exists()
    print(f"SOLVE {'DONE' if ok else 'FAILED'} -> {outdir}")
    if not ok:
        raise SystemExit("solver did not write SUCCESS; check the run log")
    return outdir


def extract():
    from skull_transparency.sim.extract import extract_bundle
    OUT.mkdir(parents=True, exist_ok=True)
    out = extract_bundle(SCRATCH / "outward", OUT, SCRATCH, bone_threshold=BONE_THRESHOLD)
    print(f"EXTRACT DONE -> {out}")
    for g in ("genout_mod.dat", "genout.dat"):
        gp = SCRATCH / "outward" / g
        if gp.exists():
            sz = gp.stat().st_size / 1e9
            gp.unlink()
            print(f"  removed scratch {gp} ({sz:.1f} GB)")
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("solve", "all"):
        solve()
    if cmd in ("extract", "all"):
        extract()


if __name__ == "__main__":
    main()
