#!/usr/bin/env python3
"""Interactive element-positioning view for the Saimiri brain-center bundle (napari, 3-D).

The squirrel-monkey counterpart of ``examples/brain_center/view_braincenter_napari.py``,
extended with the *element* layer the positioning task needs: every recorder patch on the
outer calvarial surface is, by reciprocity, a candidate transducer-element site, and its
recorded outward coupling is what an element there would deliver to the brain center.

Layers (toggle in the layer list):
  * ``skull c (m/s)``            -- the resampled sound-speed volume (attenuated MIP)
  * ``transparency (linear amp)``-- the dense surface map, linear amplitude (primary)
  * ``transparency (dB)``        -- the same map on the dB log-amplitude scale (hidden)
  * ``candidate elements``       -- the 17k recorder sites, coloured by delivered coupling;
                                    THIS is the element-positioning layer
  * ``bowl footprint``           -- the patches inside the chosen bowl's aperture
  * ``window`` / ``beam axis`` / ``apex`` / ``brain center``

Everything is drawn in GRID VOXEL coordinates so the point clouds sit inside the volume
layer (napari has one shared canvas frame); the printout gives the world-mm equivalents.

    DISPLAY=:0 python view_saimiri_position_napari.py [bundle_dir] [--focal-length 35]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/celerina/gfp/mfs/skull_transparency/src")
import skull_transparency as st

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default=str(HERE / "bundle"))
    ap.add_argument("--focal-length", type=float, default=35.0, help="bowl focal length (mm)")
    ap.add_argument("--bowl-radius", type=float, default=15.0, help="bowl aperture radius (mm)")
    args = ap.parse_args()

    import napari

    b = st.load_bundle(args.bundle)
    thr = float(b.physics.get("bone_threshold", 2200.0))
    tmap = st.compute_transparency_map(b, options=st.TransparencyOptions(bone_threshold=thr))
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=args.focal_length,
                                                bowl_radius_mm=args.bowl_radius))

    reg = b.registration
    to_vox = reg.mni_to_fullres          # world mm -> grid voxel (the canvas frame)
    surf = np.asarray(tmap.surf_vox, float)

    # transparency = distance-corrected AMPLITUDE (sqrt of the 1/r^2-corrected intensity)
    amp = np.sqrt(np.maximum(np.asarray(tmap.value, float), 0.0))
    lo, hi = np.percentile(amp, [2.0, 98.0])
    amp_lin = np.clip(amp, lo, max(hi, lo + 1e-30))
    ref = np.percentile(amp, 98.0) or (amp.max() or 1.0)
    disp_db = 20.0 * np.log10(np.maximum(amp / ref, 1e-2))

    # candidate ELEMENT sites: the recorder shell written by prepare (array_coords.i32),
    # coloured by the delivered coupling of the nearest dense-surface patch.
    coords_path = Path(args.bundle) / "array_coords.i32"
    elem = None
    if coords_path.exists():
        from skull_transparency.sim._common import array_coords_from_i32
        elem = np.asarray(array_coords_from_i32(str(coords_path))[0], float)
        from scipy.spatial import cKDTree
        coup = np.asarray(tmap.Ipk_Wcm2, float)
        _, nn = cKDTree(surf).query(elem)
        elem_amp = np.clip(np.sqrt(np.maximum(coup[nn], 0.0)), lo, max(hi, lo + 1e-30))

    src = to_vox(np.asarray(pl.target_mni_mm, float)[None, :])
    win = to_vox(np.asarray(pl.window_center_mni_mm, float)[None, :])
    apex = to_vox(np.asarray(pl.apex_mni_mm, float)[None, :])
    beam = np.vstack([apex[0], src[0]])[None, :, :]        # a single 2-point path

    c = np.asarray(b.skull_c())
    v = napari.Viewer(title=f"Saimiri element positioning - {Path(args.bundle).name}")
    v.add_image(c, name="skull c (m/s)", colormap="gray", rendering="attenuated_mip",
                contrast_limits=[1540.0, 2900.0], opacity=0.4)
    v.add_points(surf, features={"amp_linear": amp_lin}, face_color="amp_linear",
                 face_colormap="inferno", size=2.0, name="transparency (linear amp)",
                 border_width=0)
    v.add_points(surf, features={"transparency_dB": disp_db}, face_color="transparency_dB",
                 face_colormap="inferno", size=2.0, name="transparency (dB)",
                 border_width=0, visible=False)
    if elem is not None:
        v.add_points(elem, features={"coupling_amp": elem_amp}, face_color="coupling_amp",
                     face_colormap="viridis", size=3.0, name="candidate elements",
                     border_width=0)
    if pl.footprint_surf_idx is not None and len(pl.footprint_surf_idx):
        v.add_points(surf[np.asarray(pl.footprint_surf_idx, int)], face_color="deepskyblue",
                     size=2.6, name="bowl footprint", border_width=0, opacity=0.7)
    v.add_shapes(beam, shape_type="path", edge_color="lime", edge_width=1.2,
                 name="beam axis")
    v.add_points(apex, face_color="lime", size=8.0, symbol="disc", name="apex")
    v.add_points(win, face_color="red", size=7.0, symbol="star", name="window")
    v.add_points(src, face_color="cyan", size=9.0, symbol="cross", name="brain center")
    v.dims.ndisplay = 3

    frame = getattr(reg, "world_frame", "mni_ras_mm")
    print(f"napari: {len(surf)} surface patches"
          + (f", {len(elem)} candidate elements" if elem is not None else "")
          + f", bone_threshold {thr:.0f} m/s")
    print(f"  brain center  {np.round(pl.target_mni_mm, 2)} mm  ({frame})")
    print(f"  window centre {np.round(pl.window_center_mni_mm, 2)} mm, "
          f"apex {np.round(pl.apex_mni_mm, 2)} mm, incidence {pl.incidence_deg:.1f} deg, "
          f"{pl.n_footprint_patches} footprint patches", flush=True)
    napari.run()


if __name__ == "__main__":
    main()
