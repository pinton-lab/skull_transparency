"""The ITRUSST benchmark skull as a ready-to-simulate medium.

Downloads the benchmark's inner/outer surface meshes (Aubry et al., JASA 2022; ~13 MB,
openly hosted) and rasterizes them to homogeneous cortical bone on an isotropic grid at
the requested points-per-wavelength — the same construction the Colab notebook uses. The
skull is authored in MNI-aligned RAS, so MNI coordinates are world-mm targets on it.

Results are cached under ``$SKULL_TRANSPARENCY_CACHE``/``~/.cache/skull_transparency``;
rasterization needs ``trimesh`` (``pip install trimesh``).
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

C0 = 1500.0          # water/soft tissue (m/s)
C_BONE = 2800.0      # homogeneous cortical bone, as in the benchmark
F0 = 500e3           # the benchmark's frequency (Hz)
STL_BASE = ("https://raw.githubusercontent.com/ucl-bug/transcranial-ultrasound-benchmarks"
            "/master/intercomparison/skull-stl")
STL_FILES = ("skull_inner.stl", "skull_outer.stl")


def cache_dir() -> Path:
    d = os.environ.get("SKULL_TRANSPARENCY_CACHE")
    d = Path(d) if d else Path.home() / ".cache" / "skull_transparency"
    d = d / "itrusst"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pitch_mm(ppw: int = 6, f0: float = F0, c0: float = C0) -> float:
    """Grid pitch for a points-per-wavelength choice (matches the sim grid)."""
    return round(c0 / (f0 * ppw) * 1e3, 3)


def fetch_stls(verbose: bool = True) -> tuple[Path, Path]:
    """Download (once) and return the inner/outer STL paths."""
    d = cache_dir()
    out = []
    for f in STL_FILES:
        p = d / f
        if not p.exists():
            if verbose:
                print(f"[itrusst] downloading {f} ...")
            urllib.request.urlretrieve(f"{STL_BASE}/{f}", p)
        out.append(p)
    return tuple(out)


def build_medium(ppw: int = 6, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Rasterized sound-speed map + world affine at ``ppw`` (cached per pitch).

    Returns ``(c, affine)`` with ``c`` float32 (water ``C0``, bone ``C_BONE``) and
    ``affine`` the 4x4 voxel->world(MNI RAS mm) map."""
    d = cache_dir()
    pitch = pitch_mm(ppw)
    cpath = d / f"c_{pitch:g}mm.npy"
    apath = d / f"affine_{pitch:g}mm.npy"
    if cpath.exists() and apath.exists():
        return np.load(cpath), np.load(apath)
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("rasterizing the ITRUSST skull needs trimesh "
                          "(pip install trimesh)") from e
    inner_p, outer_p = fetch_stls(verbose=verbose)
    outer = trimesh.load(outer_p)
    inner = trimesh.load(inner_p)
    lo = outer.bounds[0] - 2 * pitch
    dims = np.ceil((outer.bounds[1] + 2 * pitch - lo) / pitch).astype(int)

    def _raster(mesh):
        vg = mesh.voxelized(pitch=pitch).fill()
        idx = np.floor((vg.points - lo) / pitch).astype(int)
        g = np.zeros(dims, bool)
        ok = np.all((idx >= 0) & (idx < dims), axis=1)
        idx = idx[ok]
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return g

    if verbose:
        print(f"[itrusst] rasterizing at {pitch} mm ({ppw} PPW) ...")
    bone = _raster(outer) & ~_raster(inner)
    c = np.where(bone, np.float32(C_BONE), np.float32(C0))
    affine = np.diag([pitch, pitch, pitch, 1.0])
    affine[:3, 3] = lo
    np.save(cpath, c)
    np.save(apath, affine)
    if verbose:
        print(f"[itrusst] {tuple(int(x) for x in dims)} voxels at {pitch} mm -> cached")
    return c, affine
