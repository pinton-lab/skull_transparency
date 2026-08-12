#!/usr/bin/env python
"""Halle skull microCT (HU NRRD) -> sound-speed c-map (m/s) NIfTI for
`skull-transparency prepare`. Ramp matches tuba.species.human SLAB_RAMP.

Usage:
  python build_halle_cmap.py [in.nrrd] [out_c.nii.gz] [out_rho.nii.gz]
Defaults read the tuba cache and write next to it.
"""
import os, sys
import numpy as np
import nibabel as nib

# tuba's NRRD reader (resolved as a sibling repo or installed package; portable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import add_tuba, halle_cache
add_tuba()
from tuba.io.nrrd import read_nrrd

# --- tuba.species.human SLAB_RAMP (HU -> acoustic) ---
I_LOW, I_HIGH = 700.0, 1973.0           # HU: cancellous lower / cortical upper
C_WATER, C_BONE = 1540.0, 2900.0        # m/s
RHO_WATER, RHO_BONE = 1000.0, 2200.0    # kg/m^3
VOXEL_MM = 0.125                        # Halle native pitch

src = sys.argv[1] if len(sys.argv) > 1 else halle_cache("halle_skull.nrrd")
out_c = sys.argv[2] if len(sys.argv) > 2 else halle_cache("halle_c.nii.gz")
out_rho = sys.argv[3] if len(sys.argv) > 3 else halle_cache("halle_rho.nii.gz")

print(f"reading {src}")
data, voxel_mm, _ = read_nrrd(src, verbose=True)        # int16 ~HU, LAS voxel order
hu = data.astype(np.float32)
t = np.clip((hu - I_LOW) / (I_HIGH - I_LOW), 0.0, 1.0)  # piecewise-linear ramp
c   = (C_WATER   + (C_BONE   - C_WATER)   * t).astype(np.float32)
rho = (RHO_WATER + (RHO_BONE - RHO_WATER) * t).astype(np.float32)

# voxel-index -> world-mm RAS: Halle NRRD is LAS storage, so diag(-vox,+vox,+vox)
vox = float(voxel_mm if np.ndim(voxel_mm) == 0 else voxel_mm[0])
affine = np.diag([-vox, vox, vox, 1.0])

nib.save(nib.Nifti1Image(c,   affine), out_c)
nib.save(nib.Nifti1Image(rho, affine), out_rho)
print(f"wrote {out_c}  shape={c.shape}  c in [{c.min():.0f},{c.max():.0f}] m/s")
print(f"wrote {out_rho}")
print("NIfTI carries the affine, so `prepare` needs no separate --affine.")
