#!/usr/bin/env python3
"""Open the brain-center whole-skull transparency (tutorial Figure 5) in napari, 3-D.

Shows the external skull surface patches coloured by the SAME dB log-|p| scale as the
static figure (bright = transparent bone), the brain-center source marker, and the skull
sound-speed volume (off by default). Desktop only (needs a display + the napari [viz] extra).

    python view_braincenter_napari.py [bundle_dir]
"""
import sys
from pathlib import Path

import numpy as np

import skull_transparency as st
from skull_transparency import paths

# Bundle dir: CLI arg, else the halle_braincenter Field Bundle under
# $SKULL_TR_DATA_ROOT (see skull_transparency.paths).
BUNDLE = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.bundle_dir("halle_braincenter")


def main():
    import napari
    b = st.load_bundle(BUNDLE)
    thr = float(b.physics.get("bone_threshold", 2200.0))
    tmap = st.compute_transparency_map(b, options=st.TransparencyOptions(bone_threshold=thr))

    # transparency = distance-corrected AMPLITUDE (sqrt of the 1/r^2-corrected intensity).
    val = np.maximum(np.asarray(tmap.value, float), 0.0)
    amp = np.sqrt(val)
    # LINEAR-scale view: clip to the 2-98 percentile so a few hot patches don't wash it out
    lo, hi = np.percentile(amp, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    amp_lin = np.clip(amp, lo, hi)
    # dB (log-amplitude) view, exactly as render_transparency_surface (re 98th-pct, -40 dB floor)
    ref = np.percentile(amp, 98.0) or (amp.max() or 1.0)
    disp_db = 20.0 * np.log10(np.maximum(amp / ref, 10.0 ** (-40.0 / 20.0)))

    surf = np.asarray(tmap.surf_vox, float)                 # voxel coords (match the volume)
    c = np.asarray(b.skull_c())                             # (N,N,N) sound speed
    src = np.asarray(b.target["fullres_voxel"], float)[None, :]

    v = napari.Viewer(title=f"brain-center whole-skull transparency (Fig 5) - {BUNDLE.name}")
    v.add_image(c, name="skull c (m/s)", colormap="gray", rendering="attenuated_mip",
                contrast_limits=[1540.0, 2900.0], opacity=0.4, visible=True)
    # LINEAR amplitude (primary, visible)
    v.add_points(surf, features={"amp_linear": amp_lin}, face_color="amp_linear",
                 face_colormap="inferno", size=2.5, name="transparency (linear amp)", border_width=0)
    # dB log-amplitude (hidden; toggle in the layer list to compare)
    v.add_points(surf, features={"transparency_dB": disp_db}, face_color="transparency_dB",
                 face_colormap="inferno", size=2.5, name="transparency (dB)", border_width=0,
                 visible=False)
    v.add_points(src, face_color="cyan", size=10.0, symbol="cross", name="brain center")
    v.dims.ndisplay = 3
    print(f"napari: {len(surf)} patches, bone_threshold {thr:.0f} m/s, "
          f"linear amp clip [{lo:.2g}, {hi:.2g}] (dB [{disp_db.min():.1f}, {disp_db.max():.1f}]); "
          f"source voxel {src[0].astype(int)}", flush=True)
    napari.run()


if __name__ == "__main__":
    main()
