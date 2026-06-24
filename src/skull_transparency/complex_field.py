"""Complex (amplitude+phase) transcranial field at the array elements, from the
time-reversal outward recording — the data layer for phase-aware placement, drives and
focal-spot prediction.

DESIGN PIVOT (verified): the phase comes ONLY from the array-element TIME traces, which are
clean (dt2=290.9 ns, Nyquist 1.72 MHz > f0). The sparse outward *volume* map is broadband and
impulsive, so a single-frequency phasor sampled from it is corrupted (corr 0.16 with the array
phasor) — it is used only for the alias-safe energy/transparency surface map (``metrics``).

Convention (verified): with the numpy ``e^{-iωt}`` DFT kernel, the recorded phasor of a causal
outgoing arrival at delay r/c is ``G ∝ e^{-ikr}/r`` (``arg G ≈ -k0 r``) — the time-trace
convention. k0 = ω/c0. The optimal (time-reversal) drive is the phase conjugate ``u ∝ G*``; the
radial projection that re-references G to another sphere is ``e^{-ik0(R-r1)}`` (see ``projection``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass
class ComplexField:
    """The complex element field G_i = G(T, x_i; f0) of a point source at the target T,
    recorded at element positions x_i (reciprocity: equals what element i delivers to T)."""
    pos_fullres: np.ndarray     # (M,3) f8  element positions, full-res voxel
    G: np.ndarray               # (M,)  c16  full-record exact-ω phasor (matches stored probe)
    G_win: np.ndarray           # (M,)  c16  ballistic-windowed phasor (clean direct arrival) -> drives/PSF
    E_win: np.ndarray           # (M,)  f8   ballistic-windowed broadband energy -> placement objective
    radius_mm: np.ndarray       # (M,)  f8   |x_i - T|
    source_T_fullres: np.ndarray
    f0: float
    k0: float                   # rad/mm
    dx_mm: float
    c0: float

    @property
    def radius_m(self) -> np.ndarray:
        return self.radius_mm * 1e-3

    def direction(self) -> np.ndarray:
        """Unit vectors target -> element (the ray directions). Elements must not coincide with the
        target; a zero-radius element (unreachable for an external transducer) yields a zeroed row
        with a warning rather than NaNs poisoning the PSF."""
        d = self.pos_fullres - self.source_T_fullres
        n = np.linalg.norm(d, axis=1, keepdims=True)
        if np.any(n == 0):
            import warnings
            warnings.warn("element coincident with target (radius 0); direction zeroed")
            n = np.where(n == 0, 1.0, n)
        return d / n

    def to_npz(self, path) -> None:
        np.savez(path, pos_fullres=self.pos_fullres, G_re=self.G.real, G_im=self.G.imag,
                 Gw_re=self.G_win.real, Gw_im=self.G_win.imag, E_win=self.E_win,
                 radius_mm=self.radius_mm, source_T_fullres=self.source_T_fullres,
                 f0=self.f0, k0=self.k0, dx_mm=self.dx_mm, c0=self.c0)

    @classmethod
    def from_npz(cls, path) -> "ComplexField":
        d = np.load(path)
        return cls(pos_fullres=d["pos_fullres"], G=d["G_re"] + 1j * d["G_im"],
                   G_win=d["Gw_re"] + 1j * d["Gw_im"], E_win=d["E_win"], radius_mm=d["radius_mm"],
                   source_T_fullres=d["source_T_fullres"], f0=float(d["f0"]), k0=float(d["k0"]),
                   dx_mm=float(d["dx_mm"]), c0=float(d["c0"]))


def ballistic_window_global(R: np.ndarray):
    """Global direct-arrival Hann window over all elements — EXACT replica of the window in
    ``placement_validation/fair_compare.py`` (so ΣE_win reproduces its sumE to machine
    precision). Returns (w (nT,), a0, b0)."""
    nT = R.shape[0]
    env = np.abs(hilbert(R, axis=0))
    agg = env.sum(1); agg[:4] = 0.0
    fpk = int(np.argmax(agg))
    lo = fpk
    while lo > 0 and agg[lo] > 0.25 * agg[fpk]:
        lo -= 1
    hi = fpk
    while hi < nT - 1 and agg[hi] > 0.25 * agg[fpk]:
        hi += 1
    a0, b0 = max(0, lo - 6), min(nT, hi + 6)
    w = np.zeros(nT)
    w[a0:b0] = np.hanning(b0 - a0)
    return w, a0, b0


def _per_element_window(R: np.ndarray, dt: float, f0: float, n_periods: float = 1.5) -> np.ndarray:
    """Per-element Gaussian window about each element's own envelope-peak arrival (σ≈n_periods
    of f0) — a clean, multipath-free phasor for the projection/PSF/drive phase."""
    nT, nA = R.shape
    env = np.abs(hilbert(R, axis=0))
    arrv = np.argmax(env, axis=0)
    sig = n_periods * (1.0 / f0) / dt
    tt = np.arange(nT)
    return np.exp(-0.5 * ((tt[:, None] - arrv[None, :]) / sig) ** 2)


def element_field(R: np.ndarray, positions_fullres: np.ndarray, target_fullres: np.ndarray,
                  f0: float, dt: float, dx_mm: float, c0: float = 1540.0) -> ComplexField:
    """Build the :class:`ComplexField` from the recorded array traces.

    ``R`` (nT, M) element pressure traces; ``positions_fullres`` (M,3); ``target_fullres`` (3,);
    ``dt`` the trace sample interval (s). The phasor uses an EXACT-ω Goertzel (not a rounded FFT
    bin). Placement energy ``E_win`` uses the global ballistic window (energy is broadband-robust);
    the phasor ``G_win`` uses per-element direct-arrival windows (clean phase)."""
    R = np.asarray(R, float)
    nT, M = R.shape
    k0 = 2.0 * np.pi * f0 / c0 * 1e-3                     # rad/mm
    t = np.arange(nT) * dt
    phasor = np.exp(-1j * 2.0 * np.pi * f0 * t)           # exact-ω kernel
    G = (R * phasor[:, None]).sum(0)                      # full-record phasor (matches probe)
    wg, a0, b0 = ballistic_window_global(R)
    Rwg = R * wg[:, None]
    E_win = (Rwg ** 2).sum(0)                             # global-window broadband energy
    wpe = _per_element_window(R, dt, f0)
    G_win = ((R * wpe) * phasor[:, None]).sum(0)          # per-element windowed phasor
    radius_mm = np.linalg.norm(positions_fullres - target_fullres, axis=1) * dx_mm
    return ComplexField(pos_fullres=np.asarray(positions_fullres, float), G=G, G_win=G_win,
                        E_win=E_win, radius_mm=radius_mm,
                        source_T_fullres=np.asarray(target_fullres, float),
                        f0=float(f0), k0=float(k0), dx_mm=float(dx_mm), c0=float(c0))
