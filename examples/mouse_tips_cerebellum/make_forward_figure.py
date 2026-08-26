#!/usr/bin/env python3
"""Field figure for the mouse / real-TIPS forward case, from ONE solve instead of two.

``run_forward_comparison.py`` solves the pair. This solves only the TRANSCRANIAL half and
takes the all-water twin from :func:`skull_transparency.forward.free_field_rs`, which
integrates it exactly (the free-field half is a bowl radiating into uniform water, with
nothing in the way). On this deck the two twins agree to 0.4 % and give the same insertion
loss to 0.03 dB, so the picture is the same one -- at roughly a third of the cost, because
the full-wave free-field solve is the slow half.

Writes ``forward/focal_peaks.npz`` (the small distillate: peak |p| per box point, so the
multi-GB traces can be deleted and the figure still regenerates in a second) and
``forward_fields.png`` -- the PNG at the example root rather than under ``forward/``,
which ``.gitignore`` excludes, so the README's image survives a clone.

    GPU=0 python make_forward_figure.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import skull_transparency as st                                        # noqa: E402
from skull_transparency.forward import write_forward_pair              # noqa: E402
from skull_transparency.report import _forward_figure, forward_peak_volumes  # noqa: E402
from skull_transparency.sim.prepare import build_brain_center_run      # noqa: E402
from skull_transparency.transducer import build_cap                    # noqa: E402
from skull_transparency.transducer_spec import TransducerSpec          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_forward_comparison import (APERTURE_MM, BOX_HALF_MM,          # noqa: E402
                                    ROC_MM, SURROUND_MM)
from run_mouse_tips_cerebellum import (BONE_THRESHOLD, C_WATER,        # noqa: E402
                                       INPUT_FRAME, TIPS, _c_map,
                                       cerebellum_center_ras_mm)

#: The sim tree and the solved deck are ~19 GB of medium maps; keep them on tmpfs and copy
#: only the small distillate back into the example.
SCRATCH = Path(os.environ.get("SCRATCH", "/dev/shm/mouse_tips_fig"))
WORK = Path(os.environ.get("WORK", "/dev/shm/mouse_tips_fig_out"))
OUT = Path(os.environ.get("OUT", HERE / "forward"))


def main():
    c_map, affine = _c_map()
    target = cerebellum_center_ras_mm()

    tmap = st.compute_transparency_map(st.load_bundle(HERE / "bundle"))
    foot = float(np.percentile(np.asarray(tmap.rad_mm, float), 30.0)) * \
        float(np.sin(np.radians(TIPS.half_angle_deg)))
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=ROC_MM, bowl_radius_mm=foot,
                                                theta_max_deg=TIPS.acceptance_angle_deg))
    axis = np.asarray(pl.window_center_mni_mm, float) - target
    axis /= np.linalg.norm(axis)
    apex = target + ROC_MM * axis

    cap_mm, _ = build_cap(apex, target - apex, ROC_MM,
                          half_angle_deg=TIPS.half_angle_deg, density=0.3)
    spec = TransducerSpec(f0_hz=TIPS.f0_hz, geometry="bowl", roc_mm=ROC_MM,
                          aperture_mm=APERTURE_MM, c0_ms=C_WATER, ppw=TIPS.ppw,
                          acceptance_angle_deg=TIPS.acceptance_angle_deg)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    sim = build_brain_center_run(c_map, affine, spec, SCRATCH, center_phys_mm=target,
                                 bone_threshold=BONE_THRESHOLD, surround_mm=SURROUND_MM,
                                 input_frame=INPUT_FRAME, include_points_mm=cap_mm)
    gs = json.loads((Path(sim) / "meta.json").read_text()).get("grid_shape")
    print(f"  domain {tuple(gs)} at {spec.dx_mm:.4f} mm", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    write_forward_pair(sim, out_dir=WORK, free_field="rs", apex_mm=apex, target_mm=target,
                       roc_mm=ROC_MM, aperture_mm=APERTURE_MM, box_half_mm=BOX_HALF_MM,
                       run_solver=True, gpu=int(os.environ.get("GPU", 0)), log=print)

    peaks = forward_peak_volumes(WORK, OUT / "focal_peaks.npz")  # free field -> analytic
    print(f"  peak volumes {peaks['transcranial'].shape}, free field via "
          f"{peaks['free_field_method']}", flush=True)

    fwd_json = json.loads((OUT / "forward.json").read_text()) if (OUT / "forward.json").exists() else None
    fig = _forward_figure(peaks, pl, tmap.registration, comparison=fwd_json,
                          c_map=c_map, c0=C_WATER, bone_threshold=BONE_THRESHOLD)
    png = HERE / "forward_fields.png"      # tracked; forward/ is gitignored
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  wrote {png}", flush=True)


if __name__ == "__main__":
    main()
