"""Field-reduction metrics for the time-reversal outward phase.

The peak (max_t|p|) is dominated by the *direct* arrival, so it is clean of the
reverberation coda; the time-integral (sum_t p^2) is the deposited energy density.
The distance correction cancels the point-source 1/r^2 spreading so the surface
map reflects transcranial *coupling*, not mere proximity to the target."""
from __future__ import annotations

import numpy as np

RHO0 = 1000.0   # kg/m^3
C0 = 1540.0     # m/s


def integrate_outward(propmap, n_out, log=None):
    """One streaming pass over outward frames -> (Iint, Pmax), both (nf,nf,nf) f8.

    ``propmap`` is the (n_total, nf, nf, nf) field (memmap ok); frames [0, n_out)
    are the outward (target -> skull) phase.  Iint = sum p^2, Pmax = max |p|."""
    nf = propmap.shape[1]
    Iint = np.zeros((nf, nf, nf), np.float64)
    Pmax = np.zeros((nf, nf, nf), np.float64)
    for t in range(int(n_out)):
        f = np.asarray(propmap[t], np.float64)
        Iint += f * f
        np.maximum(Pmax, np.abs(f), out=Pmax)
        if log and t % 60 == 0:
            log(f"integrating outward frame {t}/{n_out}")
    return Iint, Pmax


def peak_intensity(pmax_pa, rho=RHO0, c=C0):
    """Spatial-peak instantaneous intensity I = p^2 / (2 rho c)  [W/m^2]."""
    return np.asarray(pmax_pa, float) ** 2 / (2.0 * rho * c)


def distance_correct(value, rad_mm):
    """Cancel point-source 1/r^2 spreading on an *intensity*-like quantity:
    multiply by (r / median_r)^2 so all radii are referenced to a common distance."""
    rad = np.asarray(rad_mm, float)
    return np.asarray(value, float) * (rad / np.median(rad)) ** 2
