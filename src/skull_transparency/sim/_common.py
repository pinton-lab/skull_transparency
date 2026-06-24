"""Shared geometry / medium / workspace helpers for the launchers.

These reproduce the column-major (MATLAB ``(:)`` / ``ndgrid``) ordering that the
``fullwave2_launch_*`` scripts rely on, the binary-skull medium recipe, and a
compact Python workspace sidecar (``workspace.npz``) that replaces the MATLAB
``workspace.mat`` consumed by the inward launchers.
"""
from __future__ import annotations

import json
import os
import numpy as np

# ---- medium recipe (identical across all launchers) ------------------------

def rho_from_c(c):
    """``rho = single(1000 + max(min((c-1540)/1360,1),0)*(2200-1000))``.

    Computed in float64 then cast to float32 (MATLAB ``single(...)``). Evaluated
    in place (min-then-max, matching MATLAB's clamp order) to avoid several
    full-grid temporaries; bit-identical to the naive expression.
    """
    f = np.asarray(c, dtype=np.float64) - 1540.0
    f /= 1360.0
    np.minimum(f, 1.0, out=f)
    np.maximum(f, 0.0, out=f)
    f *= 1200.0           # 2200 - 1000
    f += 1000.0
    return f.astype(np.float32)


def load_meta(sim_dir):
    with open(os.path.join(sim_dir, "meta.json")) as f:
        return json.load(f)


# ---- source / recorder geometry (column-major like MATLAB) -----------------

def source_sphere(dent, rsrc=3):
    """``[xg,yg,zg]=ndgrid(-rsrc:rsrc,...); m=(xg^2+yg^2+zg^2)<=rsrc^2;
    incoords=[dent+xg(m), ...]; col4=1; col5=1`` — column-major masked order."""
    dent = np.asarray(dent, dtype=np.int64).ravel()
    xs = np.arange(-rsrc, rsrc + 1)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    m = (X ** 2 + Y ** 2 + Z ** 2) <= rsrc ** 2
    # column-major (Fortran) ravel == MATLAB linear index order (dim1 fastest)
    sel = m.ravel(order="F")
    xv = X.ravel(order="F")[sel]
    yv = Y.ravel(order="F")[sel]
    zv = Z.ravel(order="F")[sel]
    n = xv.size
    inc = np.empty((n, 5), dtype=np.float64)
    inc[:, 0] = dent[0] + xv
    inc[:, 1] = dent[1] + yv
    inc[:, 2] = dent[2] + zv
    inc[:, 3] = 1
    inc[:, 4] = 1
    return inc


def volume_recorders(N, modX, modY, modZ):
    """``[jx,jy,jz]=ndgrid(0:modX:N-1,...); vol=[jx(:) jy(:) jz(:)]; col4=-1; col5=2``."""
    N = np.atleast_1d(np.asarray(N, dtype=np.int64))
    if N.size == 1:
        Nx = Ny = Nz = int(N[0])
    else:
        Nx, Ny, Nz = int(N[0]), int(N[1]), int(N[2])
    jxs = np.arange(0, Nx, modX)
    jys = np.arange(0, Ny, modY)
    jzs = np.arange(0, Nz, modZ)
    JX, JY, JZ = np.meshgrid(jxs, jys, jzs, indexing="ij")
    vol = np.empty((JX.size, 5), dtype=np.float64)
    vol[:, 0] = JX.ravel(order="F")
    vol[:, 1] = JY.ravel(order="F")
    vol[:, 2] = JZ.ravel(order="F")
    vol[:, 3] = -1
    vol[:, 4] = 2
    return vol


def focal_box(dent, N, fb=36):
    """``[bx,by,bz]=ndgrid(dent-fb:dent+fb,...); box=[..]; col4=-1; col5=2``
    then keep rows with ``0<=xyz<N`` (per-axis)."""
    dent = np.asarray(dent, dtype=np.int64).ravel()
    N = np.atleast_1d(np.asarray(N, dtype=np.int64))
    if N.size == 1:
        Nx = Ny = Nz = int(N[0])
    else:
        Nx, Ny, Nz = int(N[0]), int(N[1]), int(N[2])
    bx = np.arange(dent[0] - fb, dent[0] + fb + 1)
    by = np.arange(dent[1] - fb, dent[1] + fb + 1)
    bz = np.arange(dent[2] - fb, dent[2] + fb + 1)
    BX, BY, BZ = np.meshgrid(bx, by, bz, indexing="ij")
    box = np.empty((BX.size, 5), dtype=np.float64)
    box[:, 0] = BX.ravel(order="F")
    box[:, 1] = BY.ravel(order="F")
    box[:, 2] = BZ.ravel(order="F")
    box[:, 3] = -1
    box[:, 4] = 2
    keep = ((box[:, 0] >= 0) & (box[:, 0] < Nx) &
            (box[:, 1] >= 0) & (box[:, 1] < Ny) &
            (box[:, 2] >= 0) & (box[:, 2] < Nz))
    return box[keep]


def array_coords_from_i32(path):
    """Read ``*_array_coords.i32``: int32, reshape ``[3, nA]`` then transpose."""
    a = np.fromfile(path, dtype="<i4")
    nA = a.size // 3
    return a.reshape(nA, 3).astype(np.float64), nA


def array_outcoords(arr):
    """``outcoordsA=[arr,(1:nA)',ones(nA,1)]`` — col4=elem id (1..nA), col5=1."""
    nA = arr.shape[0]
    oc = np.empty((nA, 5), dtype=np.float64)
    oc[:, 0:3] = arr
    oc[:, 3] = np.arange(1, nA + 1)
    oc[:, 4] = 1
    return oc


# ---- workspace sidecar -----------------------------------------------------

def save_workspace(path, scalars: dict, incoords, oc_array, vol_params,
                   medium: dict, extra: dict | None = None):
    """Write ``workspace.npz``. ``oc_array`` is the col5==1 block (or None);
    ``vol_params`` is ``(N, modX, modY, modZ)`` if a decimated volume is part of
    outcoords (or None); ``medium`` describes how to rebuild ``cmap``."""
    payload = {f"s_{k}": np.asarray(v) for k, v in scalars.items()}
    payload["incoords"] = np.asarray(incoords)
    payload["has_array"] = np.asarray(oc_array is not None)
    if oc_array is not None:
        payload["oc_array"] = np.asarray(oc_array)
    payload["has_vol"] = np.asarray(vol_params is not None)
    if vol_params is not None:
        Nv, mx, my, mz = vol_params
        payload["vol_N"] = np.atleast_1d(np.asarray(Nv, dtype=np.int64))
        payload["vol_mod"] = np.asarray([mx, my, mz], dtype=np.int64)
    payload["medium_json"] = np.asarray(json.dumps(medium))
    if extra:
        for k, v in extra.items():
            payload[f"x_{k}"] = np.asarray(v)
    np.savez(path, **payload)


def load_workspace(path):
    z = np.load(path, allow_pickle=False)
    ws = {"scalars": {}, "extra": {}}
    for k in z.files:
        if k.startswith("s_"):
            ws["scalars"][k[2:]] = z[k]
        elif k.startswith("x_"):
            ws["extra"][k[2:]] = z[k]
        else:
            ws[k] = z[k]
    ws["medium"] = json.loads(str(ws["medium_json"]))
    return ws


def rebuild_outcoords(ws):
    """Reconstruct the full ``outcoords`` matrix [array; vol] from a workspace."""
    blocks = []
    if bool(ws["has_array"]):
        blocks.append(ws["oc_array"])
    if bool(ws["has_vol"]):
        N = ws["vol_N"]
        mx, my, mz = [int(x) for x in ws["vol_mod"]]
        blocks.append(volume_recorders(N, mx, my, mz))
    return np.concatenate(blocks, axis=0)


def _grid_shape(medium):
    """``(nx, ny, nz)`` from a descriptor; ``N`` may be an int (cube) or ``[nx,ny,nz]``."""
    N = np.atleast_1d(np.asarray(medium["N"], dtype=np.int64))
    if N.size == 1:
        N = np.repeat(N, 3)
    elif N.size != 3:
        raise ValueError(f"medium['N'] must be a scalar or 3 values, got {N.size}")
    return int(N[0]), int(N[1]), int(N[2])


def _load_f32_map(sim_dir, fname, shape, out_dtype):
    """Read a raw little-endian float32 volume and Fortran-reshape it to ``shape``."""
    a = np.fromfile(os.path.join(sim_dir, fname), dtype="<f4")
    return a.reshape(shape, order="F").astype(out_dtype)


def rebuild_medium(sim_dir, medium):
    """Rebuild ``(cmap_float64, rho_float32, alpha_or_None)`` from a medium descriptor.

    The sound speed ``c`` is loaded per ``kind``. Density is loaded from
    ``medium['rho_file']`` when present (a user-supplied map) and otherwise
    synthesized from ``c`` via :func:`rho_from_c` (the fallback). An absorption map
    (dB/MHz/cm) is loaded from ``medium['alpha_file']`` when present, else ``None``
    (the c-porosity model is then used downstream). Grids may be cubic (``N`` an int)
    or anisotropic (``N`` = ``[nx, ny, nz]``).

    Back-compatible: a descriptor without ``rho_file``/``alpha_file`` returns exactly
    ``(c, rho_from_c(c), None)`` -- byte-identical to the original two-tuple plus None.
    """
    kind = medium["kind"]
    if kind in ("halle_c", "maps"):
        c = _load_f32_map(sim_dir, medium["file"], _grid_shape(medium), np.float64)
    elif kind == "crop":
        NB = int(medium["NB"])
        clo = np.asarray(medium["clo"], dtype=np.int64)
        chi = np.asarray(medium["chi"], dtype=np.int64)
        buf = np.fromfile(os.path.join(sim_dir, medium["file"]), dtype="<f4")
        buf = buf.reshape(NB, NB, NB, order="F")
        c = buf[clo[0]:chi[0] + 1, clo[1]:chi[1] + 1, clo[2]:chi[2] + 1].astype(np.float64)
    else:
        raise ValueError(f"unknown medium kind {kind!r}")
    rho_file = medium.get("rho_file")
    rho = (_load_f32_map(sim_dir, rho_file, c.shape, np.float32) if rho_file
           else rho_from_c(c))
    alpha_file = medium.get("alpha_file")
    alpha = (_load_f32_map(sim_dir, alpha_file, c.shape, np.float64) if alpha_file
             else None)
    return c, rho, alpha
