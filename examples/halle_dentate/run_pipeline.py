#!/usr/bin/env python3
"""End-to-end example = the original hemisphere_tr behaviour in a few library calls:
build a Field Bundle from the existing run, compute the skull transparency map, and
place a focused bowl for the target. Writes surface_map.npz + placement.json.

    python run_pipeline.py [config.json]
"""
import json
import sys
from pathlib import Path

import skull_transparency as st


def main(cfg_path="config.json"):
    cfg = json.loads(Path(cfg_path).read_text())
    b = cfg["bundle"]
    data_dir = Path(b["data_dir"])

    # 1. Field Bundle (idempotent: writes bundle.json + registration.json if absent)
    if not (data_dir / "bundle.json").exists():
        st.build_field_bundle(data_dir, b["meta"], b.get("transform"), target_name=b.get("target_name"))
    bundle = st.load_bundle(data_dir)

    # 2. transparency map (-> small distributable surface_map.npz)
    tmap = st.compute_transparency_map(bundle, log=print)
    tmap.to_npz(data_dir / "surface_map.npz")
    print(f"transparency: {len(tmap.surf_vox)} surface patches; "
          f"peak |p| {tmap.Pmax.max()/1e3:.1f} kPa; peak I {tmap.Ipk_Wcm2.max():.3f} W/cm^2")

    # 3. place a focused bowl (delivered-energy-optimal window)
    bw = cfg["bowl"]
    pl = st.place_bowl(tmap, st.BowlConstraints(
        focal_length_mm=bw["focal_length_mm"], bowl_radius_mm=bw["bowl_radius_mm"],
        theta_max_deg=bw["theta_max_deg"]))
    placement = st.to_placement_dict(pl, target_name=b.get("target_name"))
    (data_dir / "placement.json").write_text(json.dumps(placement, indent=1))
    print(f"window MNI {pl.window_center_mni_mm.round(1)}  incidence {pl.incidence_deg:.1f} deg  "
          f"apex {pl.apex_mni_mm.round(1)}  (frame={placement['frame']})")
    print(f"wrote {data_dir/'surface_map.npz'} and {data_dir/'placement.json'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("config.json"))
