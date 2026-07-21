"""Python port of ``launch_fullwave2_3d_Aexp_noicmat`` — writes the simulation
input ``.dat`` files (everything except ``icmat.dat``, which each launcher
writes itself, and ``modT/modX/modY/modZ/nTic/ncoords_add/nAo``, written by the
launcher via :func:`writeVabs`).

Constants baked in (every ``fullwave2_launch_*`` caller passes exactly these):
``Az = 0`` (the attenuation map ``A`` is extended but never written),
``Aex = 1`` everywhere, ``beta = 5.5`` everywhere. The boundary geometry is the
solver default ``M = 8``, ``nbdy = 40`` → 48-voxel pad per side (an ``nbdy``-thick
absorbing layer plus the ``M`` stencil halo), so ``nXe = N + 96``.

Boundaries are handled by the per-cell ``Aexp`` absorption map (``Aexp <= 1``,
applied as ``field *= Aexp`` each step over the ``nbdy`` layer) -- the current
solver kernels are the Aexp kernels, NOT PML. The legacy MATLAB ``apml*/bpml*``
arrays are true-PML coefficients that these kernels ignore, so they are not
reproduced. ("PML pad" elsewhere is a misnomer for this absorbing-layer pad.)
"""
from __future__ import annotations

import os
import numpy as np

from . import fwio
from .forcoef import build_d_matrix, build_dmap
from .mlcompat import matlab_round

M = 8
NBDY = 40
PAD = NBDY + M     # 48

# the six medium maps that depend only on (cmap, grid size) — identical across
# any run sharing the same medium, so they can be reused via hardlink.
_MEDIUM_MAPS = ("c.dat", "K.dat", "rho.dat", "beta.dat", "Aexp.dat", "dcmap.dat")


def _reuse_medium_maps(src, expect_bytes):
    """Hardlink (fall back to copy) the six medium maps from ``src`` into the cwd.
    Returns True only if all six are present AND match the current grid
    (``c.dat`` size == ``expect_bytes``); the maps depend solely on the medium +
    grid size, so for the inward time-reversal runs they are byte-identical to
    the source (outward) run — reusing them avoids rewriting ~10 GB of identical
    data per run. The size guard prevents silently linking a mismatched grid."""
    import shutil
    paths = [os.path.join(src, m) for m in _MEDIUM_MAPS]
    if not all(os.path.exists(p) for p in paths):
        return False
    if os.path.getsize(os.path.join(src, "c.dat")) != expect_bytes:
        return False
    for m, p in zip(_MEDIUM_MAPS, paths):
        if os.path.exists(m):
            os.remove(m)
        try:
            os.link(p, m)               # hardlink: no extra bytes written
        except OSError:
            shutil.copyfile(p, m)       # cross-device fallback
    return True


def _porosity_aexp(c_ext, c0, omega0, dT, alpha_dbmhzcm=None):
    """Per-voxel, per-timestep amplitude-decay factor for the lab CT-porosity absorption model.
    Port of fullwave_launcher_3d_Aexp_sparse_skull_beamforming2.m (L181-188) + dbmhzcm2aexp.m:
      tissue/water (c == c0)  -> alpha = 0.4 dB/MHz/cm
      bone (c > c0)           -> alpha = (2 + 78*sqrt(phi)) * 12/40 dB/MHz/cm,
                                 phi = 1 - (c-1540)/(2900-1540)         (porosity, 0 dense .. 1 porous)
    alpha is HALVED (lab convention) before the exponential conversion; the per-step factor is
    evaluated at the reference speed c0 (as in the MATLAB call dbmhzcm2aexp(A/2,c0,omega0,dT)).
    Returns exp(-dT*texp) <= 1 (== 1 only where alpha == 0).  c_ext = extended sound-speed map (m/s).

    ``alpha_dbmhzcm`` (an extended per-voxel dB/MHz/cm map) overrides the c-derived porosity
    model; when None the original c-porosity path runs byte-for-byte unchanged."""
    if alpha_dbmhzcm is None:
        phi = np.clip(1.0 - (c_ext - 1540.0) / (2900.0 - 1540.0), 0.0, 1.0)
        alpha = np.where(c_ext > 1540.0, (2.0 + 78.0 * np.sqrt(phi)) * (12.0 / 40.0), 0.4)  # dB/MHz/cm
    else:
        alpha = np.asarray(alpha_dbmhzcm, dtype=np.float64)  # supplied per-voxel attenuation
    np_per_db = np.log(10.0) / 20.0                          # Nepers per dB  (= 1/8.6859)
    F0_MHz = omega0 / (2.0 * np.pi) / 1e6
    texp = (alpha / 2.0) * F0_MHz * c0 / 1e-2 * np_per_db    # Nepers/second ; A/2 lab convention
    return np.exp(-dT * texp)


def launch_core(c0, omega0, wX, wY, wZ, duration, p0, ppw, cfl,
                cmap, rhomap, incoords, outcoords, nTic, betaval=5.5,
                write_maps=True, reuse_maps_from=None, attenuation=False, alpha_map=None):
    """Write the medium/coord/scalar/coefficient ``.dat`` files into the current
    working directory (the caller ``chdir``-s into the output subdir, mirroring
    the MATLAB ``cd(outdir)`` structure).

    Parameters mirror the MATLAB call
    ``launch_fullwave2_3d_Aexp_noicmat(c0,omega0,wX,wY,wZ,duration,p0,ppw,cfl,
    c,rho,Az,Aex,bet, incoords,outcoords, nTic)``. ``cmap`` is float64 (MATLAB
    ``fread('single')`` returns double); ``rhomap`` is the single-precision
    density map; ``incoords``/``outcoords`` are 0-based ``(n,5)`` coordinate
    matrices. Returns a dict of the derived scalars.
    """
    cmap = np.asarray(cmap, dtype=np.float64)
    rhomap = np.asarray(rhomap, dtype=np.float32)

    lam = c0 / omega0 * 2 * np.pi
    nX, nY, nZ = cmap.shape
    nXe = nX + 2 * PAD
    nYe = nY + 2 * PAD
    nZe = nZ + 2 * PAD
    nT = int(matlab_round(duration * c0 / lam * ppw / cfl))

    dX = c0 / omega0 * 2 * np.pi / ppw
    dY = dX
    dZ = dX
    dT = dY / c0 * cfl

    ncoords = incoords.shape[0]
    ncoordsout = outcoords.shape[0]

    # min/max of the extended c equal those of cmap (edge replication preserves
    # the value range), so dmap/dcmap can use cmap's range directly.
    minc = float(cmap.min())
    maxc = float(cmap.max())

    # coordinate matrices: +48 on x,y,z then -1 on all columns (writeCoords)
    incoords = np.array(incoords, dtype=np.float64, copy=True)
    outcoords = np.array(outcoords, dtype=np.float64, copy=True)
    incoords[:, 0:3] += PAD
    outcoords[:, 0:3] += PAD

    d = build_d_matrix(cfl)
    dmap = build_dmap(minc, maxc, dT, dX)
    ndmap = dmap.shape[2]

    reused = (bool(reuse_maps_from) and write_maps
              and _reuse_medium_maps(reuse_maps_from, nXe * nYe * nZe * 4))
    if write_maps and not reused:
        # extended maps (replicate-pad by 48; float64 like MATLAB's double output).
        # Built and freed one at a time, and K is formed in place, to keep the
        # peak working set near two extended grids rather than four.
        c = fwio.extendMap3d(cmap, PAD)             # f64
        fwio.writeMapXYZ("c.dat", c)
        # physical absorption factor, captured before c is overwritten into K below; a supplied
        # alpha_map (extended like c) overrides the c-porosity model when attenuation is on.
        aexp_phys = None
        if attenuation:
            alpha_ext = fwio.extendMap3d(alpha_map, PAD) if alpha_map is not None else None
            aexp_phys = _porosity_aexp(c, c0, omega0, dT, alpha_dbmhzcm=alpha_ext).astype("<f4")
        # dcmap = round(c) - min(c); sound speeds are >= 0 so round == floor(c+0.5)
        dcmap = c + 0.5
        np.floor(dcmap, out=dcmap)
        dcmap -= minc
        fwio.writeMapXYZ("dcmap.dat", dcmap, "int")
        del dcmap
        rho = fwio.extendMap3d(rhomap, PAD)          # f64
        fwio.writeMapXYZ("rho.dat", rho)
        c *= c          # c^2 in place (== c**2 for float)
        c *= rho        # K = c^2 * rho in place
        fwio.writeMapXYZ("K.dat", c)
        del c, rho
        # beta = 5.5 everywhere (extended) -> float32
        np.full(nXe * nYe * nZe, betaval, dtype="<f4").tofile("beta.dat")
        # Aexp = (1 - maskBdy) boundary taper, optionally x physical per-step absorption (one map,
        # one field*=Aexp per step in the solver, so both effects must be pre-multiplied together)
        amask = np.float32(1.0) - fwio.maskBdy(nXe, nYe, nZe, PAD)
        if aexp_phys is not None:
            amask *= aexp_phys
        np.ascontiguousarray(amask).astype("<f4").tofile("Aexp.dat")

    fwio.writeCoords("icc.dat", incoords - 1)
    fwio.writeCoords("outc.dat", outcoords - 1)

    fwio.writeVabs("float", dX, "dX", dY, "dY", dZ, "dZ", dT, "dT", c0, "c0")
    fwio.writeVabs("int", nXe, "nX", nYe, "nY", nZe, "nZ", nT, "nT",
                   ncoords, "ncoords", ncoordsout, "ncoordsout", nTic, "nTic")

    # d.dat : fwrite(d','float') == row-major (C-order) float32
    np.ascontiguousarray(d).astype("<f4").tofile("d.dat")
    # dmap.dat : nested i,j,k loop == C-order ravel, float32
    np.ascontiguousarray(dmap).astype("<f4").tofile("dmap.dat")
    fwio.writeVabs("int", ndmap, "ndmap")

    return dict(nXe=nXe, nYe=nYe, nZe=nZe, nT=nT, dX=dX, dY=dY, dZ=dZ, dT=dT,
                ncoords=ncoords, ncoordsout=ncoordsout, ndmap=ndmap,
                minc=minc, maxc=maxc, maps_reused=reused)
