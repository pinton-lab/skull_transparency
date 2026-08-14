#!/usr/bin/env python3
"""Transcranial vs free-field target pressure for the mouse TIPS/cerebellum placement.

The transparency map says which window couples best; this says what actually arrives. Two
forward solves on the same grid with the same source and drive, differing only in the
medium:

  * transcranial -- the mouse skull, with water between the transducer and the head;
  * free field   -- the identical run with the skull replaced by water everywhere.

Their peak-pressure ratio at the target is the insertion loss of that placement.

**Why the bowl here is not the literal TIPS.** The TIPS focal length is 80 mm and the mouse
head is 25 mm across, so a full-size TIPS deck would need a domain ~110 mm on a side; at the
0.128 mm pitch this case needs to resolve the ~0.3 mm calvaria that is ~640 M voxels, and
coarsening the grid to fit would put the mouse skull back under one voxel per wall -- which
is exactly the error the transmission number is supposed to measure. The bowl driven here is
therefore GEOMETRICALLY SIMILAR: the same 35.1 deg half-angle and the same f/0.87 as the
TIPS, at a radius of curvature that fits the domain. It crosses the same skull at the same
angles with the same aperture solid angle; only the wavefront curvature at the bone differs,
which is second-order for bone a tenth of a wavelength thick.

Run (a couple of minutes on one GPU):

    GPU=1 python run_forward_comparison.py
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import skull_transparency as st
from skull_transparency.forward import run_forward_pair
from skull_transparency.sim.prepare import build_brain_center_run
from skull_transparency.transducer_spec import TransducerSpec

from run_mouse_tips_cerebellum import (  # the case definition, so nothing is duplicated
    BONE_THRESHOLD, C_WATER, INPUT_FRAME, TIPS, _c_map, cerebellum_center_ras_mm)

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("SCRATCH", "/dev/shm/mouse_tips_fwd"))
OUT = Path(os.environ.get("OUT", HERE / "forward"))

#: Radius of curvature that fits a mouse-scale domain, with the aperture chosen to keep the
#: TIPS half-angle (and therefore its f-number) exactly.
ROC_MM = 18.0
APERTURE_MM = 2.0 * ROC_MM * float(np.sin(np.radians(TIPS.half_angle_deg)))
#: Water margin around the head: enough to hold the whole bowl, which lies within ROC of
#: the target, plus a few mm of coupling water beyond it.
SURROUND_MM = float(os.environ.get("SURROUND_MM", 22.0))
BOX_HALF_MM = 3.0                      # focal box around the target (mouse-scale)


def main():
    c_map, affine = _c_map()
    target = cerebellum_center_ras_mm()

    # the window the transparency map chose for this target, and the bowl axis through it
    tmap = st.compute_transparency_map(st.load_bundle(HERE / "bundle"))
    foot = float(np.percentile(np.asarray(tmap.rad_mm, float), 30.0)) * \
        float(np.sin(np.radians(TIPS.half_angle_deg)))
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=ROC_MM, bowl_radius_mm=foot,
                                                theta_max_deg=TIPS.acceptance_angle_deg))
    win = np.asarray(pl.window_center_mni_mm, float)
    axis = win - target
    axis /= np.linalg.norm(axis)
    apex = target + ROC_MM * axis
    print(f"  window {np.round(win, 2)} mm | bowl ROC {ROC_MM:.0f} mm, aperture "
          f"{APERTURE_MM:.1f} mm (f/{ROC_MM / APERTURE_MM:.2f}, half-angle "
          f"{TIPS.half_angle_deg:.1f} deg) | apex {np.round(apex, 1)} mm")

    # a domain big enough to hold the bowl as well as the head
    spec = TransducerSpec(f0_hz=TIPS.f0_hz, geometry="bowl", roc_mm=ROC_MM,
                          aperture_mm=APERTURE_MM, c0_ms=C_WATER, ppw=TIPS.ppw,
                          acceptance_angle_deg=TIPS.acceptance_angle_deg)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(c_map, affine, spec, SCRATCH, center_phys_mm=target,
                                 bone_threshold=BONE_THRESHOLD, surround_mm=SURROUND_MM,
                                 input_frame=INPUT_FRAME)

    cmp_ = run_forward_pair(sim, out_dir=OUT, gpu=int(os.environ.get("GPU", 1)),
                            apex_mm=apex, target_mm=target, roc_mm=ROC_MM,
                            aperture_mm=APERTURE_MM, box_half_mm=BOX_HALF_MM, log=print)
    print("\n" + cmp_.summary())
    return cmp_


if __name__ == "__main__":
    main()
