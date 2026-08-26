#!/usr/bin/env python3
"""Transcranial vs free-field target pressure for the mouse TIPS/cerebellum placement.

The transparency map says which window couples best; this says what actually arrives. Two
forward solves on the same grid with the same source and drive, differing only in the
medium:

  * transcranial -- the mouse skull, with water between the transducer and the head;
  * free field   -- the identical run with the skull replaced by water everywhere.

Their peak-pressure ratio at the target is the insertion loss of that placement.

**The bowl is the real TIPS.** It did not used to be. The TIPS focal length is 80 mm against
a 25 mm mouse head, and ``surround_mm`` pads all six faces of the domain, so buying that
stand-off with it cost ``(head + 2*80)^3`` -- 6.6 G cells and 147 GiB of medium maps at the
0.128 mm pitch this case needs to resolve the ~0.3 mm calvaria. That was not runnable, and
coarsening the grid to fit would have put the mouse skull back under one voxel per wall,
which is exactly the error the transmission number is supposed to measure. So the bowl used
to be GEOMETRICALLY SIMILAR instead: the same 35.1 deg half-angle and f/0.87 at an 18 mm
radius of curvature that fit.

That compromise is gone. ``build_brain_center_run(include_points_mm=...)`` sizes the domain
to the union of the head and the transducer rather than padding every face, which puts the
real 80 mm bowl in 0.755 G cells (16.9 GiB of maps) -- 8.7x smaller, and runnable. Set
``SUBSTITUTE_BOWL=1`` to reproduce the old 18 mm numbers.

Still not modelled: the TIPS central hole (annular, 20.5-46 mm radius). The bowl here is
filled, so this is the real focal length and the real OUTER aperture, not the real annulus;
the hole affects the drive and side-lobe structure, not the window search.

Run (~50 min on one A6000: ~12 min for the transcranial solve, most of the rest for the
free-field one -- the solver takes a different, slower path on a uniform-water medium).
``--free-field rs`` in the CLI replaces that second solve with ~3 s of exact integration
and agrees with it to 0.4 % here; see ``skull_transparency.forward.free_field_rs``.

    GPU=1 python run_forward_comparison.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import skull_transparency as st
from skull_transparency.forward import run_forward_pair
from skull_transparency.sim.prepare import build_brain_center_run
from skull_transparency.transducer import build_cap
from skull_transparency.transducer_spec import TransducerSpec

from run_mouse_tips_cerebellum import (  # the case definition, so nothing is duplicated
    BONE_THRESHOLD, C_WATER, INPUT_FRAME, TIPS, _c_map, cerebellum_center_ras_mm)

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("SCRATCH", "/dev/shm/mouse_tips_fwd"))
OUT = Path(os.environ.get("OUT", HERE / "forward"))

#: ``SUBSTITUTE_BOWL=1`` restores the pre-2026-08 geometrically-similar 18 mm bowl (same
#: half-angle, same f-number) that the old isotropic-surround domain forced.
SUBSTITUTE_BOWL = bool(int(os.environ.get("SUBSTITUTE_BOWL", "0")))
ROC_MM = 18.0 if SUBSTITUTE_BOWL else TIPS.roc_mm
APERTURE_MM = (2.0 * ROC_MM * float(np.sin(np.radians(TIPS.half_angle_deg)))
               if SUBSTITUTE_BOWL else TIPS.aperture_mm)
#: Coupling water kept around the head AND around the declared transducer points. With the
#: bowl declared via ``include_points_mm`` this no longer has to reach the transducer, so it
#: is just a margin -- it does not buy the stand-off any more.
SURROUND_MM = float(os.environ.get("SURROUND_MM", 6.0))
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
          f"{TIPS.half_angle_deg:.1f} deg) | apex {np.round(apex, 1)} mm"
          + ("  [SUBSTITUTE bowl]" if SUBSTITUTE_BOWL else "  [real TIPS]"))

    # Size the domain to the union of the head and the BOWL ITSELF, rather than padding
    # every face far enough to reach the bowl -- see the module docstring.
    cap_mm, _ = build_cap(apex, target - apex, ROC_MM,
                          half_angle_deg=TIPS.half_angle_deg, density=0.3)
    spec = TransducerSpec(f0_hz=TIPS.f0_hz, geometry="bowl", roc_mm=ROC_MM,
                          aperture_mm=APERTURE_MM, c0_ms=C_WATER, ppw=TIPS.ppw,
                          acceptance_angle_deg=TIPS.acceptance_angle_deg)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(c_map, affine, spec, SCRATCH, center_phys_mm=target,
                                 bone_threshold=BONE_THRESHOLD, surround_mm=SURROUND_MM,
                                 input_frame=INPUT_FRAME, include_points_mm=cap_mm)
    meta = json.loads((Path(sim) / "meta.json").read_text())
    gs = meta.get("grid_shape") or [meta["N"]] * 3
    print(f"  domain {tuple(gs)} at {spec.dx_mm:.4f} mm "
          f"({np.prod(np.asarray(gs) + 96) / 1e9:.3f} G cells with the absorbing pad)")

    cmp_ = run_forward_pair(sim, out_dir=OUT, gpu=int(os.environ.get("GPU", 1)),
                            apex_mm=apex, target_mm=target, roc_mm=ROC_MM,
                            aperture_mm=APERTURE_MM, box_half_mm=BOX_HALF_MM, log=print)
    print("\n" + cmp_.summary())
    return cmp_


if __name__ == "__main__":
    main()
