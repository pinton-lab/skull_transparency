#!/usr/bin/env python
"""Robust MNI thalamus -> Halle RAS target. Re-runs the (fast) ANTs SyN in-process so
the inverse transforms are valid in memory (tuba's saved invwarp path is broken), and
maps the atlas point with tuba's exact LPS convention. Prints target + native voxel."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import add_tuba, halle_cache
add_tuba()
os.environ.setdefault("TUBA_HUMAN_NRRD_PATH", halle_cache("halle_skull.nrrd"))
import gzip, shutil
import numpy as np, pandas as pd, ants
from tuba.species import human
from tuba.atlases.mni152 import ANATOMICAL_TARGETS_MNI

# --- self-stage prerequisites (idempotent, so this runs from scratch) ---
adir = os.path.dirname(human.ATLAS.t1_path); os.makedirs(adir, exist_ok=True)
if not os.path.exists(human.ATLAS.t1_path):     # stage MNI152 ICBM 2009a via nilearn
    from nilearn.datasets import fetch_icbm152_2009
    print("fetching MNI152 ICBM 2009a ...", flush=True)
    d = fetch_icbm152_2009()
    for src, dst in ((d["t1"], human.ATLAS.t1_path), (d["mask"], human.ATLAS.brain_mask_path)):
        with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
human.downsample_halle_to_1mm()                  # build halle_ct_1mm.nii.gz if missing

fixed = ants.image_read(human.HALLE_CT_1MM)     # Halle CT @1mm
moving = ants.image_read(human.ATLAS.t1_path)   # MNI152 T1
print("registering MNI T1 -> Halle CT (SyN, mattes MI cross-modality) ...", flush=True)
reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN",
                        aff_metric="mattes", syn_metric="mattes")
inv = reg["invtransforms"]                      # [affine, invwarp] : maps atlas->subject for POINTS

VOX = 0.125
def mni_ras_to_halle(mni_ras):
    x, y, z = mni_ras
    df = pd.DataFrame([[-x, -y, z]], columns=["x", "y", "z"])     # RAS -> LPS (ANTs)
    o = ants.apply_transforms_to_points(3, df, transformlist=inv).iloc[0]
    ras = np.array([-o["x"], -o["y"], o["z"]])                   # subject LPS -> Halle RAS
    vox = np.array([-ras[0] / VOX, ras[1] / VOX, ras[2] / VOX])  # RAS -> native 0.125mm voxel
    return ras, vox

out = {}
for name in ("thalamus_left", "thalamus_right", "thalamus_central", "VIM_thalamus_left"):
    ras, vox = mni_ras_to_halle(ANATOMICAL_TARGETS_MNI[name])
    out[name] = {"ras_mm": [round(float(v), 2) for v in ras],
                 "native_voxel": [int(round(v)) for v in vox]}
    print(f"  {name:18s} MNI {ANATOMICAL_TARGETS_MNI[name]} -> Halle RAS "
          f"{out[name]['ras_mm']} mm | voxel {out[name]['native_voxel']}")
np.save(os.path.join(human.REG_DIR, "thalamus_targets_halle_ras.npy"), out, allow_pickle=True)
print("saved thalamus_targets_halle_ras.npy")
