#!/usr/bin/env python3
"""Zero-data example: generate a synthetic Field Bundle and run the placement chain end
to end — transparency map + positioning score + placement.json. Needs no /celerina data,
GPU, MATLAB, or tuba; runs anywhere the package is installed.

    python run_synthetic.py [out_dir]
"""
import json
import sys
from pathlib import Path

import skull_transparency as st


def main(out="synthetic_run"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    bundle = st.load_bundle(st.make_synthetic_bundle(out / "bundle"))   # the tiny fixture
    tmap = st.compute_transparency_map(bundle)
    tmap.to_npz(out / "surface_map.npz")

    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60.0, bowl_radius_mm=15.0,
                                                theta_max_deg=35.0))
    score = st.PositioningScore.from_placement(pl, target_name="synthetic_target")
    score.to_json(out / "score.json")
    (out / "placement.json").write_text(
        json.dumps(st.to_placement_dict(pl, target_name="synthetic_target", species_human=None), indent=1))

    print(f"score {score.normalized:.3f} (focal-pressure proxy {score.focal_pressure_proxy:.3g}), "
          f"incidence {pl.incidence_deg:.1f} deg")
    print(f"wrote {out}/surface_map.npz, score.json, placement.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic_run")
