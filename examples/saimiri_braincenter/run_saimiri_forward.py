#!/usr/bin/env python3
"""Transcranial vs free-field target pressure for the Saimiri brain-center placement.

The transparency map says which window couples best; this says what actually arrives. Two
forward solves on the same grid with the same source and drive, differing only in the
medium:

  * transcranial -- the Saimiri skull, with water between the transducer and the head;
  * free field   -- the identical run with the skull replaced by water everywhere.

Their peak-pressure ratio at the target is the insertion loss of that placement.

Unlike the mouse TIPS case, no geometrically-similar substitute bowl is needed here: the
35 mm ROC of this study's bowl is comparable to the head (the skull sits ~19 mm from the
brain center), so the real device geometry fits a domain that still resolves the ~1.1 mm
calvarium at 6 PPW. The window, footprint and accessibility mask come from
``run_saimiri_report.chosen_placement``, so this drives exactly the placement the report
quotes rather than a re-derived one.

CAVEAT inherited from the medium: the museum microCT is not HU-calibrated, so the absolute
insertion loss carries the placeholder ramp's uncertainty. The RATIO between windows on the
same skull is far more robust than the absolute number.

Run (a few minutes on one GPU):

    GPU=0 python run_saimiri_forward.py
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from skull_transparency.forward import run_forward_pair                  # noqa: E402
from skull_transparency.sim.prepare import build_brain_center_run        # noqa: E402
from skull_transparency.transducer_spec import TransducerSpec            # noqa: E402

import build_saimiri_braincenter as CASE                                 # noqa: E402
from run_saimiri_report import ROC_MM, APERTURE_MM, THETA_MAX_DEG, chosen_placement  # noqa: E402

SCRATCH = Path(os.environ.get("SCRATCH", "/dev/shm/saim_fwd"))
OUT = Path(os.environ.get("OUT", HERE / "forward"))

#: Water margin around the head. The bowl apex sits one ROC (35 mm) from the target while
#: the vault is only ~16 mm away, so the domain needs ~20 mm more than the head to hold the
#: dish plus a few mm of coupling water beyond it.
SURROUND_MM = float(os.environ.get("SURROUND_MM", 26.0))
#: Half-width of the focal box the peak/FWHM are measured in. Must exceed the AXIAL lobe,
#: not the lateral one: at f/1.17 and 1 MHz the -6 dB axial extent is ~7*lambda*f#^2 ~ 15 mm,
#: so a squirrel-monkey-scale +-4 mm box truncates it and the axial FWHM comes back "inf".
BOX_HALF_MM = float(os.environ.get("BOX_HALF_MM", 12.0))


def main():
    _bundle, _tmap, pl, _mask, _acc, _foot, _r30 = chosen_placement()
    target = np.asarray(pl.target_mni_mm, float)
    win = np.asarray(pl.window_center_mni_mm, float)
    axis = win - target
    axis /= np.linalg.norm(axis)
    apex = target + ROC_MM * axis
    print(f"  bowl ROC {ROC_MM:.0f} mm, aperture {APERTURE_MM:.0f} mm "
          f"(f/{ROC_MM/APERTURE_MM:.2f}) | apex {np.round(apex, 1)} mm")

    c_map, affine, center = CASE._load_inputs()
    off = float(np.linalg.norm(center - target))
    assert off < 1e-6, f"bundle target and case center disagree by {off:.3f} mm"

    spec = TransducerSpec(f0_hz=CASE.F0_HZ, geometry="bowl", roc_mm=ROC_MM,
                          aperture_mm=APERTURE_MM, c0_ms=CASE.C_WATER, ppw=CASE.PPW,
                          acceptance_angle_deg=THETA_MAX_DEG)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(c_map, affine, spec, SCRATCH, center_phys_mm=center,
                                 bone_threshold=CASE.BONE_THRESHOLD,
                                 surround_mm=SURROUND_MM, input_frame=CASE.INPUT_FRAME)
    import json
    meta = json.loads((Path(sim) / "meta.json").read_text())
    gs = meta.get("grid_shape", [meta["N"]] * 3)
    print(f"  forward grid {gs[0]}x{gs[1]}x{gs[2]} at dx {meta['dX_m']*1e3:.3f} mm "
          f"({np.prod(gs)/1e6:.0f} M voxels)")

    os.environ.setdefault("FULLWAVE2_BIN", "/celerina/gfp/mfs/fullwave2-ultra/bin/bench_3d_opt")
    cmp_ = run_forward_pair(sim, out_dir=OUT, gpu=int(os.environ.get("GPU", 0)),
                            apex_mm=apex, target_mm=target, roc_mm=ROC_MM,
                            aperture_mm=APERTURE_MM, box_half_mm=BOX_HALF_MM, log=print)
    print("\n" + cmp_.summary())
    return cmp_


if __name__ == "__main__":
    main()
