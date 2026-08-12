#!/usr/bin/env python
"""Stage MNI152, register MNI->Halle (ANTs SyN via antspyx), emit thalamus target
in the Halle RAS frame (== the c-map NIfTI frame fed to `prepare`)."""
import os, gzip, shutil, time
os.environ.setdefault("TUBA_HUMAN_NRRD_PATH",
                      os.path.expanduser("~/.cache/tuba/human/source/halle_skull.nrrd"))
import numpy as np
from nilearn.datasets import fetch_icbm152_2009
from tuba.species import human
from tuba.atlases.mni152 import ANATOMICAL_TARGETS_MNI

t0 = time.time()
# 1. stage MNI152 ICBM 2009a T1 + brain mask into tuba's atlas_dir (gunzip to .nii)
print("fetching MNI152 ICBM 2009a via nilearn ...", flush=True)
d = fetch_icbm152_2009()
adir = os.path.dirname(human.ATLAS.t1_path)
os.makedirs(adir, exist_ok=True)
def gunzip_to(src, dst):
    if os.path.exists(dst):
        print("  cached", dst); return
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print("  staged", dst)
gunzip_to(d["t1"], human.ATLAS.t1_path)
gunzip_to(d["mask"], human.ATLAS.brain_mask_path)

# 2. ANTs SyN: MNI T1 (moving) -> Halle CT 1mm (fixed). Saves transforms to REG_DIR.
print(f"\n[{time.time()-t0:.0f}s] running ANTs SyN MNI->Halle (this is the slow step) ...", flush=True)
human.register_to_atlas(verbose=True)

# 3. thalamus targets in Halle RAS (same frame as the c-map; pass to prepare --target)
print(f"\n[{time.time()-t0:.0f}s] thalamus targets in Halle RAS mm:", flush=True)
out = {}
for name in ("thalamus_left", "thalamus_right", "thalamus_central", "VIM_thalamus_left"):
    t = human.mni_ras_to_halle_ras(np.array(ANATOMICAL_TARGETS_MNI[name], float))
    out[name] = [round(float(v), 2) for v in t]
    print(f"  {name:18s} MNI {ANATOMICAL_TARGETS_MNI[name]} -> Halle RAS {out[name]} mm")
np.save(os.path.join(human.REG_DIR, "thalamus_targets_halle_ras.npy"), out, allow_pickle=True)
print(f"\nDONE in {time.time()-t0:.0f}s. Saved thalamus_targets_halle_ras.npy")
