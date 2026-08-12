#!/usr/bin/env python
"""Build the Sparse1024-3 array TransducerSpec and prepare a 1.5 MHz sim tree aimed
at the MNI thalamus target. Reports the grid size so we can judge solve feasibility."""
import os, sys, time, numpy as np, nibabel as nib
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "src"))
from _paths import halle_cache, asset
from skull_transparency.transducer_spec import TransducerSpec
from skull_transparency.sim.prepare import build_run_from_medium

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--ppw", type=float, default=4.0)
ap.add_argument("--surround-mm", type=float, default=90.0)
ap.add_argument("--standoff-mm", type=float, default=20.0)
ap.add_argument("--target", default="-82.86,97.73,97.6")     # thalamus_left Halle RAS
ap.add_argument("--approach", default="0,0,1")               # vertex
ap.add_argument("--c-map", default=halle_cache("halle_c.nii.gz"))
ap.add_argument("--rho-map", default=halle_cache("halle_rho.nii.gz"))
ap.add_argument("--out", default="thalamus_array_run")
a = ap.parse_args()

cmap_path = a.c_map
rho_path  = a.rho_map
elems = np.load(asset("sparse1024_elements_mm.npy"))         # (1024,3) mm, centroid-centered

spec = TransducerSpec(f0_hz=1.5e6, geometry="array",
                      element_positions_mm=elems, n_elements=1024,
                      aperture_mm=65.0, ppw=a.ppw, acceptance_angle_deg=35.0)
print(f"array: {spec.n_elements} elems, f0={spec.f0_hz/1e6} MHz, dx={spec.dx_mm:.3f} mm, "
      f"lambda={spec.wavelength_mm:.3f} mm")

img = nib.load(cmap_path); c = np.asarray(img.dataobj, np.float32); aff = np.asarray(img.affine)
rho = np.asarray(nib.load(rho_path).dataobj, np.float32)
tgt = np.array([float(x) for x in a.target.split(",")])
appr = np.array([float(x) for x in a.approach.split(",")])
print(f"loading c-map {c.shape}; target {tgt} mm; approach {appr}; ppw {a.ppw}, surround {a.surround_mm}")

t0 = time.time()
out = build_run_from_medium(c, aff, tgt, spec, a.out, rho_map=rho,
                            approach=appr, standoff_mm=a.standoff_mm, surround_mm=a.surround_mm,
                            input_frame="halle_ras_mm")
import json
meta = json.load(open(f"{a.out}/meta.json"))
N = meta["N"]
print(f"\nprepared {out} in {time.time()-t0:.0f}s")
print(f"  N={N}  -> {N**3/1e6:.0f}M voxels, c.f32={N**3*4/1e6:.0f} MB/field, n_array={meta['n_array']}")
print(f"  est genout ~ {3.59*(N/280)**4:.0f} GB (scales ~N^4 vs the N=280 CTX run's 3.59 GB)")
print(f"  target grid voxel = {meta['dent_grid']}")
