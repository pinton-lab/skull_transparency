#!/usr/bin/env python
"""Compute several candidate 'brain-center' targets in the Halle CT and overlay
them on 3 orthogonal slices so we can pick the right one (dry skull: vault air
leaks at the base, so naive enclosed-cavity detection is unreliable)."""
import os, sys
import numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import add_tuba, halle_cache, REPO
add_tuba()
from tuba.io.nrrd import read_nrrd

DS = 8; BONE_HU = 700.0
src = halle_cache("halle_skull.nrrd")
data, vox, _ = read_nrrd(src, verbose=True)
vox = float(vox if np.ndim(vox) == 0 else vox[0])
sh = data.shape; n = [s // DS for s in sh]
hu = data[:n[0]*DS, :n[1]*DS, :n[2]*DS].reshape(n[0],DS,n[1],DS,n[2],DS).max(axis=(1,3,5))
bone = hu >= BONE_HU

def fill_along(mask, axis):
    out = np.empty_like(mask); m = np.moveaxis(mask, axis, 0); o = np.moveaxis(out, axis, 0)
    for s in range(m.shape[0]): o[s] = ndi.binary_fill_holes(m[s])
    return out
def largest_cc(mask):
    lbl, n = ndi.label(mask)
    if n == 0: return mask
    sz = ndi.sum(np.ones_like(lbl), lbl, np.arange(1, n+1))
    return lbl == (1 + int(np.argmax(sz)))

cands = {}
# C1: bone centroid (whole skull)
cands["C1 bone-centroid"] = np.array(ndi.center_of_mass(bone))
# C2: bone bounding-box center
idx = np.argwhere(bone); cands["C2 bone-bbox-mid"] = (idx.min(0)+idx.max(0))/2.0
# C3: enclosed air after sealing base (8 mm close), border-flood exterior
rad = max(1, int(round(8.0/(vox*DS))))
closed = ndi.binary_closing(bone, iterations=rad)
ext = largest_cc(~closed)                      # exterior air (touches border, via foramina)
interior = (~closed) & (~ext)
cands["C3 enclosed-air"] = np.array(ndi.center_of_mass(largest_cc(interior))) if interior.any() else cands["C1 bone-centroid"]
# C4: axial per-slice fill cavity (largest CC)
ax = fill_along(bone, 2) & ~bone
cands["C4 axial-fill"] = np.array(ndi.center_of_mass(largest_cc(ax)))

print("\n=== candidates (native voxel) + RAS mm ===")
aff = np.diag([-vox, vox, vox, 1.0])
native = {}
for k, c_ds in cands.items():
    cf = np.asarray(c_ds) * DS
    mm = (aff @ np.array([*cf, 1.0]))[:3]
    native[k] = cf
    print(f"  {k:18s} vox={np.round(cf,0)}  RAS={np.round(mm,1)} mm")

# overlay all candidates on 3 views through C4 (or change here)
ref = native["C4 axial-fill"]; ci,cj,ck = (int(round(v)) for v in ref)
clim=(-500,2000); cols={"C1 bone-centroid":"red","C2 bone-bbox-mid":"yellow","C3 enclosed-air":"cyan","C4 axial-fill":"magenta"}
fig,axx=plt.subplots(1,3,figsize=(15,5.2),facecolor="black")
views=[("axial S=%d"%ck,data[:,:,ck].T,(0,1)),("coronal A=%d"%cj,data[:,cj,:].T,(0,2)),("sagittal L=%d"%ci,data[ci,:,:].T,(1,2))]
for a,(t,img,(ax0,ax1)) in zip(axx,views):
    a.imshow(img,cmap="gray",vmin=clim[0],vmax=clim[1],origin="lower",aspect="equal")
    for k,cf in native.items(): a.plot(cf[ax0],cf[ax1],"+",color=cols[k],ms=14,mew=2.2)
    a.set_title(t,color="white",fontsize=11); a.tick_params(colors="white")
handles=[plt.Line2D([],[],color=c,marker="+",ls="",ms=10,label=k) for k,c in cols.items()]
fig.legend(handles=handles,loc="lower center",ncol=4,facecolor="black",labelcolor="white",fontsize=9)
fig.suptitle("brain-center target candidates (views through C4)",color="white",fontsize=13)
fig.tight_layout(rect=[0,0.05,1,1])
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "target_candidates.png")
fig.savefig(out,dpi=90,facecolor="black",bbox_inches="tight"); print("wrote",out)
