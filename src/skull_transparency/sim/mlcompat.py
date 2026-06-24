"""MATLAB-compatible numerics used by the launchers.

Every routine here is a faithful port of the MATLAB built-in / toolbox function
as actually used by the ``fullwave2_launch_*`` scripts:

* :func:`matlab_round`     — round half away from zero (MATLAB ``round``).
* :func:`transmit_pulse`   — the outward virtual-source pulse + active-sample truncation.
* :func:`unit_pulse`       — the unit reference pulse for energy matching.
* :func:`tukeywin`         — MATLAB ``tukeywin(N, r)``.
* :func:`hilbert_envelope` — ``abs(hilbert(x))`` along columns.
* :func:`interp1_linear`   — ``interp1(x, V, xq, 'linear')`` (column-wise).
* :func:`ballistic_window` — the shared global-ballistic-window selection.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert as _scipy_hilbert

__all__ = [
    "matlab_round", "transmit_pulse", "unit_pulse", "tukeywin",
    "hilbert_envelope", "interp1_linear", "ballistic_window",
    "matlab_single_sum",
]


def matlab_single_sum(a, order="F"):
    """``sum`` of a single (float32) array, reproducing MATLAB's reduction
    bit-for-bit (verified against MATLAB on multiple vectors up to n=50000).

    MATLAB's single-threaded single-precision ``sum`` uses a 128-bit SIMD
    accumulator — 4 interleaved float32 lanes (element ``i`` → lane ``i mod 4``)
    accumulated sequentially, then the 4 lane totals summed in order. The vector
    is taken in MATLAB's column-major order (``A(:)``), hence ``order='F'``.

    NOTE: above MATLAB's multithreading threshold (~5e4 elements) MATLAB splits
    the array across threads and this single-threaded model no longer matches;
    that regime only shifts an overall energy-normalisation *scalar* by ~1e-6
    and never the drive waveform. The TR subset ``icmat`` arrays are well below
    that threshold.
    """
    x = np.asarray(a, dtype=np.float32).ravel(order=order)
    n = x.size
    k = 4
    m = (n + k - 1) // k
    pad = np.zeros(m * k, dtype=np.float32)
    pad[:n] = x
    M = pad.reshape(m, k)
    # 4 lane totals = sequential float32 sum down the rows of each column. That is
    # exactly np.add.accumulate(..., dtype=float32) (cumulative single-precision
    # add), vectorised over the lanes — bit-identical to the per-row loop.
    lane = np.add.accumulate(M, axis=0, dtype=np.float32)[-1]
    s = np.float32(0.0)
    for j in range(k):
        s = np.float32(s + lane[j])
    return s


def matlab_round(x):
    """MATLAB ``round`` — round half away from zero."""
    x = np.asarray(x, dtype=np.float64)
    out = np.sign(x) * np.floor(np.abs(x) + 0.5)
    return out if out.ndim else out.item()


def transmit_pulse(nT: int, duration: float, omega0: float, p0: float,
                   ncycles: int = 2, dur: int = 2):
    """Outward virtual-source pulse.

    Reproduces::

        t = (0:nT-1)/nT*duration - ncycles/omega0*2*pi;
        icvec = exp(-(1.05*t*omega0/(ncycles*pi)).^(2*dur)).*sin(t*omega0)*p0;
        nz = find(abs(icvec) > 1e-6*max(abs(icvec)));  nTic = max(nz);
        icvec = icvec(1:nTic);

    Returns ``(icvec_float64, nTic)``.
    """
    t = np.arange(nT, dtype=np.float64) / nT * duration - ncycles / omega0 * 2 * np.pi
    icvec = (np.exp(-(1.05 * t * omega0 / (ncycles * np.pi)) ** (2 * dur))
             * np.sin(t * omega0) * p0)
    thr = 1e-6 * np.max(np.abs(icvec))
    nz = np.nonzero(np.abs(icvec) > thr)[0]
    nTic = int(nz.max()) + 1          # MATLAB max(nz) is 1-based -> length
    return icvec[:nTic], nTic


def unit_pulse(dT: float, omega0: float, ncycles: int = 2, dur: int = 2):
    """Unit reference pulse used for energy matching.

    Reproduces::

        np = round(4e-6/dT);  tp = (0:np-1)*dT - 2e-6;
        p_unit = exp(-(1.05*tp*omega0/(ncycles*pi)).^(2*dur)).*sin(tp*omega0);

    Returns ``(p_unit_float64, np)``.
    """
    n = int(matlab_round(4e-6 / dT))
    tp = np.arange(n, dtype=np.float64) * dT - 2e-6
    p_unit = (np.exp(-(1.05 * tp * omega0 / (ncycles * np.pi)) ** (2 * dur))
              * np.sin(tp * omega0))
    return p_unit, n


def tukeywin(N: int, r: float) -> np.ndarray:
    """MATLAB ``tukeywin(N, r)`` (tapered cosine window), float64, shape (N,)."""
    if r <= 0:
        return np.ones(N, dtype=np.float64)
    if r >= 1:
        # hann(N) (symmetric)
        n = np.arange(N, dtype=np.float64)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))
    t = np.linspace(0.0, 1.0, N)
    per = r / 2.0
    tl = int(np.floor(per * (N - 1))) + 1      # 1-based count
    th = N - tl + 1                            # 1-based index
    w = np.ones(N, dtype=np.float64)
    # taper 1: indices 1..tl  -> python 0..tl-1
    w[:tl] = 0.5 * (1 + np.cos(np.pi / per * (t[:tl] - per)))
    # taper 2: indices th..N  -> python th-1..N-1
    w[th - 1:] = 0.5 * (1 + np.cos(np.pi / per * (t[th - 1:] - 1 + per)))
    return w


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """``abs(hilbert(x))`` along columns (axis 0), matching MATLAB ``hilbert``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return np.abs(_scipy_hilbert(x))
    return np.abs(_scipy_hilbert(x, axis=0))


def interp1_linear(x: np.ndarray, V: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """``interp1(x, V, xq, 'linear')`` for monotone-increasing ``x``, column-wise.

    Reproduces MATLAB's arithmetic exactly (verified against committed
    references): the breakpoints ``x``/``xq`` are double, the fraction
    ``t = (xq-x0)/(x1-x0)`` is formed in double then cast to the value class, and
    each sample is the lerp ``v0*(1-t) + v1*t`` evaluated **in the class of**
    ``V`` (single for the single-precision genout traces). So passing a single
    ``V`` returns single, matching ``interp1`` on single data bit-for-bit.

    ``V`` may be a vector or ``(len(x), ncols)`` matrix.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    xq = np.asarray(xq, dtype=np.float64).ravel()
    V = np.asarray(V)
    vdt = V.dtype if V.dtype.kind == "f" else np.dtype(np.float64)
    L = x.size
    k = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, L - 2)
    x0 = x[k]; x1 = x[k + 1]
    t = ((xq - x0) / (x1 - x0)).astype(vdt)        # fraction, in V's class
    if V.ndim == 1:
        v0 = V[k]; v1 = V[k + 1]
        return (v0 * (1 - t) + v1 * t).astype(vdt)
    v0 = V[k, :]; v1 = V[k + 1, :]
    tt = t.reshape(-1, 1)
    return (v0 * (1 - tt) + v1 * tt).astype(vdt)


def ballistic_window(g, nRun: int, dt2: float, lambda_t: float):
    """Shared global ballistic-window selection (the ``inward_windowed`` /
    ``subset`` recipe). ``g`` is ``(nRun, nS)`` single. Returns 1-based
    ``(a, b, L, fpk, lo, hi)``.

    Reproduces::

        env=abs(hilbert(g)); agg=sum(env,2); agg(1:4)=0; [apk,fpk]=max(agg);
        lo=fpk; while lo>1    && agg(lo)>0.25*apk, lo=lo-1; end
        hi=fpk; while hi<nRun && agg(hi)>0.25*apk, hi=hi+1; end
        pad=round(2*lambda_t/dt2); a=max(1,lo-pad); b=min(nRun,hi+pad); L=b-a+1;
    """
    env = hilbert_envelope(g)
    agg = env.sum(axis=1)
    agg[:4] = 0.0
    fpk = int(np.argmax(agg)) + 1          # 1-based
    apk = agg[fpk - 1]
    thr = 0.25 * apk
    lo = fpk
    while lo > 1 and agg[lo - 1] > thr:
        lo -= 1
    hi = fpk
    while hi < nRun and agg[hi - 1] > thr:
        hi += 1
    pad = int(matlab_round(2 * lambda_t / dt2))
    a = max(1, lo - pad)
    b = min(nRun, hi + pad)
    L = b - a + 1
    return a, b, L, fpk, lo, hi
