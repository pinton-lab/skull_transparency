#!/usr/bin/env python3
"""build_grid -- pick a frequency, get a self-consistent simulation grid.

The grid pitch is set by the *acoustics*, not by the CT. With c0 fixed you choose
the frequency f0 (physical) and a points-per-wavelength target ppw (numerical, ~6):

    delta = c0 / (f0 * ppw)          # grid pitch (mm)
    N_axis = ceil(extent_axis / delta)

The grid *size* N is therefore derived from the physical extent of the head, never
typed in. Your CT is resampled to delta as a separate step -- downsample if its
native pitch is finer than delta (e.g. the 0.125 mm micro-CT, factor 2), interpolate
if it is coarser (the usual clinical CT). The CT's resolution does not set delta.
Pass --ct-pitch to see that resample factor for your scanner.

    python build_grid.py --f0 1e6                  # one frequency
    python build_grid.py --f0 5e5 --ppw 6
    python build_grid.py --f0 1e6 --ct-pitch 0.5   # a 0.5 mm clinical CT
    python build_grid.py --table                   # sweep of frequencies
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

C0 = 1540.0          # m/s, reference sound speed (water/soft tissue)
BYTES_PER_VOX = 56   # Fullwave2 keeps 14 single-precision grids
BND_VOX = 96         # attenuating absorbing boundary, voxels per axis
PPW_DEFAULT = 6.16   # the validated reference value (~6 is the general FDTD target)
PPW_FLOOR = 5.0      # below this the FDTD solve is under-resolved (dispersion)
# trimmed skull-centred extent of the reference subject (mm)
EXTENT_MM = (163.0, 203.5, 162.5)


@dataclass
class Grid:
    f0: float
    ppw: float
    delta_mm: float
    N: tuple
    n_vox: int
    mem_gb: float
    warn: str

    def report(self, ct_pitch=None) -> str:
        nx, ny, nz = self.N
        lines = [
            f"frequency           {self.f0/1e6:g} MHz",
            f"points/wavelength   {self.ppw:g}",
            f"grid pitch delta    {self.delta_mm:.4g} mm   = c0 / (f0 * ppw)",
            f"grid size N         {nx} x {ny} x {nz}  voxels   ({self.n_vox/1e6:.0f} M)",
            f"GPU memory (+bnd)   ~{self.mem_gb:.1f} GB   ({BYTES_PER_VOX} B/voxel)",
        ]
        if ct_pitch:
            r = self.delta_mm / ct_pitch          # grid pitch / CT pitch
            how = (f"downsample {r:.3g}x (block-max)" if r > 1.0 else
                   f"interpolate up {1/r:.3g}x" if r < 1.0 else "1:1, no resample")
            lines.append(f"your CT ({ct_pitch:g} mm)   resample to delta: {how}")
        if self.warn:
            lines.append(f"WARNING             {self.warn}")
        return "\n".join(lines)


def design_grid(f0, ppw=PPW_DEFAULT, extent_mm=EXTENT_MM, c0=C0,
                bnd_vox=BND_VOX, bytes_per_vox=BYTES_PER_VOX) -> Grid:
    delta = c0 / (f0 * ppw) * 1e3                  # mm  (delta is continuous)
    N = tuple(int(math.ceil(e / delta)) for e in extent_mm)
    n_vox = N[0] * N[1] * N[2]
    n_bnd = (N[0] + bnd_vox) * (N[1] + bnd_vox) * (N[2] + bnd_vox)
    mem_gb = n_bnd * bytes_per_vox / 1e9
    warn = f"ppw {ppw:g} < {PPW_FLOOR:g}: under-resolved (dispersion)" if ppw < PPW_FLOOR else ""
    return Grid(f0, ppw, delta, N, n_vox, mem_gb, warn)


def table(freqs_mhz, ppw=PPW_DEFAULT):
    print(f"{'f0':>8}  {'delta':>8}  {'ppw':>5}  {'N (x,y,z)':>20}  {'voxels':>8}  {'GPU GB':>7}")
    print("-" * 66)
    for fm in freqs_mhz:
        g = design_grid(fm * 1e6, ppw)
        print(f"{fm:>6g}MHz  {g.delta_mm:>7.3g}mm  {g.ppw:>5.3g}  "
              f"{f'{g.N[0]}x{g.N[1]}x{g.N[2]}':>20}  {g.n_vox/1e6:>6.0f}M  {g.mem_gb:>6.1f}")


def main():
    ap = argparse.ArgumentParser(description="derive a self-consistent sim grid from frequency")
    ap.add_argument("--f0", type=float, default=1e6, help="center frequency in Hz (default 1e6)")
    ap.add_argument("--ppw", type=float, default=PPW_DEFAULT,
                    help=f"points per wavelength (default {PPW_DEFAULT}; ~6 is the FDTD target)")
    ap.add_argument("--ct-pitch", type=float, default=None,
                    help="your CT's native pitch (mm); reports the resample factor to delta")
    ap.add_argument("--extent", type=str, default=None,
                    help="physical extent mm as x,y,z (default the reference skull)")
    ap.add_argument("--table", action="store_true", help="print a frequency-sweep comparison")
    a = ap.parse_args()
    if a.table:
        table([0.25, 0.5, 1.0, 2.0], a.ppw)
        return
    extent = tuple(float(x) for x in a.extent.split(",")) if a.extent else EXTENT_MM
    print(design_grid(a.f0, a.ppw, extent_mm=extent).report(ct_pitch=a.ct_pitch))


if __name__ == "__main__":
    main()
