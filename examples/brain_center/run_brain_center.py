#!/usr/bin/env python3
"""Brain-center whole-skull transparency, end to end on the consumer side.

A brain-center run seats one omnidirectional source at the center of the brain so a
single outward solve illuminates the whole calvaria; the 1/r^2-corrected transparency map
is then a neutral picture of where the skull transmits (Section 5 of the tutorial). This
script computes and renders that map from a Field Bundle.

It prefers the real brain-center bundle (``halle_braincenter``); if that lab data is not
present it falls back to the zero-data synthetic fixture, so it runs anywhere the package
is installed (no GPU, MATLAB, or tuba).

    python run_brain_center.py [out_dir]

To produce the bundle for your OWN subject (needs a GPU):
    skull-transparency prepare --c-map c.nii.gz --center --transducer ctx500.json --out run/
    #   ... outward solve + `skull-transparency extract` (see the tutorial) ...
    skull-transparency transparency --bundle run/bundle --out transparency.png
"""
import sys
from pathlib import Path

import skull_transparency as st
from skull_transparency import paths

# Real brain-center bundle (override the root with $SKULL_TR_DATA_ROOT); falls back
# to the zero-data synthetic fixture below if this isn't present.
REAL = paths.bundle_dir("halle_braincenter")


def main(out="brain_center_run"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if (REAL / "bundle.json").exists():
        bundle = st.load_bundle(REAL)
        title = "Brain-center whole-skull transparency (Halle, atlas CoM)"
        src = "real brain-center solve (halle_braincenter)"
    else:
        bundle = st.load_bundle(st.make_synthetic_bundle(out / "bundle"))
        title = "Whole-skull transparency (synthetic fixture)"
        src = "synthetic fixture (no lab data found)"

    tmap = st.compute_transparency_map(bundle)              # 1/r^2 correction on by default
    tmap.to_npz(out / "transparency_map.npz")
    png = st.render_transparency_surface(tmap, out / "transparency.png", title=title)

    print(f"source: {src}")
    print(f"transparency: {len(tmap.surf_vox)} surface patches, "
          f"radius {tmap.rad_mm.min():.0f}-{tmap.rad_mm.max():.0f} mm from the brain center, "
          f"peak {tmap.Ipk_Wcm2.max():.3g} W/cm^2")
    print(f"wrote {out/'transparency_map.npz'} and {png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "brain_center_run")
