"""Low-level fullwave2 binary I/O — bit-identical Python ports of the MATLAB
``fullwave2_3d`` helpers (``writeVabs``, ``writeMapXYZ``, ``writeCoords``,
``extendMap3d``, ``maskBdy``, ``readGenoutSlice``, ``sizeOfFile``).

Byte-format notes (verified against the committed ``sim/*/*.dat`` reference files):

* All ``.dat`` are raw little-endian (x86 native, matching MATLAB ``fwrite``'s
  default machine format). ``'float'`` → ``float32``, ``'int'`` → ``int32``.
* ``writeMapXYZ`` writes, for each i-slice, ``permute(cmap(i,:,:),[3 2 1])`` in
  column-major order — i.e. k fastest, then j, then i. That is exactly the
  C-order ravel of an ``(nX, nY, nZ)`` array indexed ``[i, j, k]``.
* ``writeCoords`` writes each column of the ``(ncoords, ncols)`` matrix in turn
  as int32 — i.e. the Fortran-order (column-major) ravel.
* ``extendMap3d(map, nbdy)`` pads by ``nbdy`` on every side, replicating the
  nearest face/edge/corner value — identical to ``numpy.pad(..., mode='edge')``.
"""
from __future__ import annotations

import os
import numpy as np

__all__ = [
    "writeVabs", "writeMapXYZ", "writeCoords", "extendMap3d", "maskBdy",
    "readGenoutSlice", "sizeOfFile", "write_raw",
]


def _to_int32(arr: np.ndarray) -> np.ndarray:
    """Convert to int32 the way MATLAB ``fwrite(...,'int')`` does: round half
    away from zero, then store as int32 (inputs here are exact integers, so the
    rounding rule only matters for defensiveness)."""
    a = np.asarray(arr, dtype=np.float64)
    r = np.sign(a) * np.floor(np.abs(a) + 0.5)
    return r.astype("<i4")


def write_raw(fname: str, arr: np.ndarray, dtype: str) -> None:
    """Write ``arr`` as a flat little-endian binary of the given numpy dtype."""
    np.ascontiguousarray(arr, dtype=dtype).tofile(fname)


def writeVabs(typ: str, *args) -> None:
    """Port of ``writeVabs(typ, v1,'name1', v2,'name2', ...)``.

    Writes one ``<name>.dat`` per (value, name) pair, each a single scalar of
    type ``typ`` (``'float'`` → float32, ``'int'`` → int32). ``fwrite`` rounds
    when converting to an integer type, so we round-to-nearest for ``'int'``.
    """
    np_dtype = {"float": "<f4", "int": "<i4", "double": "<f8"}[typ]
    if len(args) % 2:
        raise ValueError("writeVabs: optargin not even")
    for k in range(0, len(args), 2):
        val, name = args[k], args[k + 1]
        if typ == "int":
            v = float(val)
            val = int(np.sign(v) * np.floor(np.abs(v) + 0.5))
        np.array([val], dtype=np_dtype).tofile(f"{name}.dat")


def writeMapXYZ(fname: str, cmap: np.ndarray, typ: str = "float") -> None:
    """Port of ``writeMapXYZ`` — C-order ravel of ``cmap[i,j,k]`` as ``typ``."""
    arr = np.asarray(cmap)
    if typ == "int":
        _to_int32(np.ascontiguousarray(arr)).tofile(fname)
    else:
        np.ascontiguousarray(arr).astype("<f4").tofile(fname)


def writeCoords(fname: str, coords: np.ndarray) -> None:
    """Port of ``writeCoords`` — each column written in turn as int32
    (= Fortran-order ravel of the ``(ncoords, ncols)`` matrix)."""
    coords = np.asarray(coords)
    out = _to_int32(coords).ravel(order="F")
    out.tofile(fname)


def extendMap3d(map3: np.ndarray, nbdy: int) -> np.ndarray:
    """Port of ``extendMap3d`` — replicate-pad by ``nbdy`` on every side.

    MATLAB ``extendMap3d`` allocates a *double* output (``zeros(...)``) and fills
    the centre with the map, faces/edges/corners with the nearest boundary value.
    That is exactly ``numpy.pad(..., mode='edge')``; the result is float64 to
    match MATLAB's promotion of the (possibly single) input to double.
    """
    return np.pad(np.asarray(map3, dtype=np.float64), nbdy, mode="edge")


def maskBdy(nX: int, nY: int, nZ: int, nbdy: int) -> np.ndarray:
    """Port of ``maskBdy`` (the vectorized single-precision version).

    ``r`` along each axis is 0 in the interior and rises into the nbdy-thick
    boundary layers; ``mask = sqrt(ri^2+rj^2+rk^2)`` normalised by its max.
    Computed in float32, exactly as the MATLAB source.
    """
    i = np.arange(1, nX + 1)
    ri = np.zeros(nX, dtype=np.float32)
    ri[i <= nbdy] = (nbdy - i + 1)[i <= nbdy]
    ri[i > nX - nbdy] = (i - (nX - nbdy))[i > nX - nbdy]
    j = np.arange(1, nY + 1)
    rj = np.zeros(nY, dtype=np.float32)
    rj[j <= nbdy] = (nbdy - j + 1)[j <= nbdy]
    rj[j > nY - nbdy] = (j - (nY - nbdy))[j > nY - nbdy]
    k = np.arange(1, nZ + 1)
    rk = np.zeros(nZ, dtype=np.float32)
    rk[k <= nbdy] = (nbdy - k + 1)[k <= nbdy]
    rk[k > nZ - nbdy] = (k - (nZ - nbdy))[k > nZ - nbdy]
    ri = ri.reshape(nX, 1, 1)
    rj = rj.reshape(1, nY, 1)
    rk = rk.reshape(1, 1, nZ)
    mask = np.sqrt(ri ** 2 + rj ** 2 + rk ** 2)   # broadcast -> nX,nY,nZ, float32
    mask = mask / mask.max()
    return mask


def sizeOfFile(fname: str) -> int:
    """Port of ``sizeOfFile`` — byte length, or 0 if the file is missing."""
    try:
        return os.path.getsize(fname)
    except OSError:
        return 0


def readGenoutSlice(fname: str, nTvec, ncoordsout: int, idc=None) -> np.ndarray:
    """Port of ``readGenoutSlice`` — read frames ``nTvec`` (0-based) of a
    ``genout.dat`` whose frame width is ``ncoordsout`` float32 values.

    Returns a ``(len(nTvec), ncoordsout)`` array, or ``(len(nTvec), len(idc))``
    if a 0-based channel index list ``idc`` is given. Byte-for-byte equivalent
    to the MATLAB ``fseek``/``fread`` loop.
    """
    nTvec = np.asarray(nTvec, dtype=np.int64).ravel()
    if idc is None:
        out = np.empty((nTvec.size, ncoordsout), dtype=np.float32)
        with open(fname, "rb") as fid:
            for i, t in enumerate(nTvec):
                fid.seek(int(t) * ncoordsout * 4, 0)
                out[i, :] = np.fromfile(fid, dtype="<f4", count=ncoordsout)
        return out

    idc = np.asarray(idc, dtype=np.int64).ravel()
    out = np.empty((nTvec.size, idc.size), dtype=np.float32)
    # Read only up to the highest requested channel per frame (a disk prefix),
    # then index — bit-identical to reading the whole frame and indexing.
    nread = int(idc.max()) + 1
    with open(fname, "rb") as fid:
        for i, t in enumerate(nTvec):
            fid.seek(int(t) * ncoordsout * 4, 0)
            frame = np.fromfile(fid, dtype="<f4", count=nread)
            out[i, :] = frame[idc]
    return out
