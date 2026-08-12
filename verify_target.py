#!/usr/bin/env python
"""Render 3 orthogonal CT slices through a target voxel to sanity-check placement.
Usage: python verify_target.py <i> <j> <k> [out.png]   (native NRRD voxel indices)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import add_tuba, halle_cache
add_tuba()
from tuba.io.nrrd import read_nrrd

i, j, k = (int(round(float(x))) for x in sys.argv[1:4])
out = sys.argv[4] if len(sys.argv) > 4 else "target_check.png"
src = halle_cache("halle_skull.nrrd")

data, vox, _ = read_nrrd(src, verbose=True)        # (L, A, S) storage, HU
vox = float(vox if np.ndim(vox) == 0 else vox[0])
clim = (-500, 2000)

fig, ax = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="black")
panels = [
    ("axial  (S=%d)" % k, data[:, :, k].T,        (i, j)),   # (L,A) shown, origin lower
    ("coronal (A=%d)" % j, data[:, j, :].T,        (i, k)),   # (L,S)
    ("sagittal (L=%d)" % i, data[i, :, :].T,       (j, k)),   # (A,S)
]
for a, (title, img, (px, py)) in zip(ax, panels):
    a.imshow(img, cmap="gray", vmin=clim[0], vmax=clim[1], origin="lower", aspect="equal")
    a.plot(px, py, "+", color="red", ms=18, mew=2.5)
    a.set_title(title, color="white", fontsize=11)
    a.set_facecolor("black"); a.tick_params(colors="white")
fig.suptitle(f"thalamus target proxy @ native voxel ({i},{j},{k})  [{vox} mm]",
             color="white", fontsize=13)
fig.tight_layout()
fig.savefig(out, dpi=110, facecolor="black", bbox_inches="tight")
print(f"wrote {out}")
