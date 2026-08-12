#!/usr/bin/env python
"""Locate a thalamus target in the Halle CT's own RAS-mm frame, without ANTs/MNI.

The thalamus is deep and central; we approximate it by the centroid of the
intracranial cavity (the largest air/brain region enclosed by the skull), which
sits within ~1 cm of the thalamus. Prints a target (mm) + a superior (vertex)
approach vector and a ready-to-run `skull-transparency prepare` command.

Frame: Halle NRRD is LAS voxel storage; affine = diag(-vox,+vox,+vox) -> RAS
(+x=R, +y=A, +z=S), matching tuba.species.human and the c-map NIfTI we feed prepare.

Usage:  python find_thalamus_target.py [halle_skull.nrrd]
"""
import os, sys
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import add_tuba, halle_cache
add_tuba()
from tuba.io.nrrd import read_nrrd

BONE_HU = 700.0
DS = 8                      # downsample for the centroid search (0.125 -> 1.0 mm); fast + robust
CLOSE_MM = 22.0             # close radius to bridge foramen magnum (~30 mm) + orbits (cf. tuba human.py)

src = sys.argv[1] if len(sys.argv) > 1 else halle_cache("halle_skull.nrrd")

data, voxel_mm, _ = read_nrrd(src, verbose=True)
vox = float(voxel_mm if np.ndim(voxel_mm) == 0 else voxel_mm[0])

# block-max downsample (preserves thin bone) for a fast, robust centroid
sh = data.shape
n = [s // DS for s in sh]
trim = data[:n[0]*DS, :n[1]*DS, :n[2]*DS]
hu = trim.reshape(n[0], DS, n[1], DS, n[2], DS).max(axis=(1, 3, 5))
dvox = vox * DS

bone = hu >= BONE_HU
print(f"  downsampled grid {hu.shape} @ {dvox:.2f} mm | bone {bone.mean()*100:.1f}%")
# Robust cranial-cavity detection: a voxel is "intracranial" only if it is an
# enclosed (filled) hole of the bone mask in 2D slices along ALL THREE axes.
# The calvarium forms a closed ring in axial/coronal/sagittal slices through the
# vault, so 3-way agreement isolates the vault and rejects basal openings
# (foramen magnum, nasopharynx) that are open in at least one slice direction.
def fill_along(mask, axis):
    out = np.empty_like(mask)
    m = np.moveaxis(mask, axis, 0)
    o = np.moveaxis(out, axis, 0)
    for s in range(m.shape[0]):
        o[s] = ndi.binary_fill_holes(m[s])
    return out
interior = (fill_along(bone, 0) & fill_along(bone, 1) & fill_along(bone, 2)) & ~bone
lbl, nlbl = ndi.label(interior)
if nlbl == 0:
    raise SystemExit("no intracranial cavity found; check bone threshold / CT")
sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, nlbl + 1))
brain = lbl == (1 + int(np.argmax(sizes)))
com_ds = np.array(ndi.center_of_mass(brain))         # voxel index in downsampled grid
com_full = com_ds * DS                                # back to native-voxel index

# native-voxel index -> RAS mm:  affine = diag(-vox, +vox, +vox)
affine = np.diag([-vox, vox, vox, 1.0])
target_mm = (affine @ np.array([*com_full, 1.0]))[:3]

# superior (vertex) approach: target -> skin along +S (+z in RAS)
approach = np.array([0.0, 0.0, 1.0])

print("\n=== thalamus target (intracranial-cavity centroid proxy) ===")
print(f"  brain-cavity voxels (ds): {int(brain.sum())}  ({brain.mean()*100:.1f}% of grid)")
print(f"  centroid voxel (native):  {np.round(com_full,1)}")
print(f"  TARGET (RAS mm):          {np.round(target_mm,2)}")
print(f"  APPROACH (target->skin):  {approach}  (superior / vertex window)")
print("\n=== prepare command ===")
print(f'  --target "{target_mm[0]:.2f},{target_mm[1]:.2f},{target_mm[2]:.2f}" '
      f'--approach "0,0,1"')
