"""Pinton-Aubry-Tanter (2012) radial-phase projection of the transcranial field.

A spherical wave from the target T, aberrated by the skull, retains predominantly RADIAL phase
progression beyond the bone (∇arg G ≈ k0 r̂ + frozen skull excess). Propagation between two
concentric spheres about T is then a DIAGONAL, per-ray operator — not a full Kirchhoff integral —
so the complex field recorded near the skull can be re-referenced to a candidate transducer
sphere of any radius R with one O(M) pass, no new wave solve.

CONVENTION (verified against the data, not assumed). With numpy's ``e^{-iωt}`` DFT, the *recorded*
phasor of a causal outgoing wave arriving at delay r/c is ``G ∝ e^{-ikr}/r`` (i.e. ``arg G ≈
-k_0 r``) — the time-trace convention the whole package uses, NOT the spatial Green's-function
``e^{+ikr}/r``. The outward radial projection therefore MULTIPLIES by ``e^{-i k0 (R-r1)}`` (the
field at the larger radius arrives later → more negative phase). The amplitude factor ``(r1/R)``
conserves ``r²|p|²`` along the ray (cap-energy invariant). This per-ray ``(r1/R)`` RE-TARGETING
factor must NEVER be combined with ``metrics.distance_correct``'s ``(r/median_r)²`` *visualisation*
factor (they answer different questions; stacking them double-counts spreading).
"""
from __future__ import annotations

import warnings

import numpy as np


def project_to_sphere(G: np.ndarray, rad_mm: np.ndarray, R_mm: float, k0: float,
                      directions: np.ndarray | None = None) -> np.ndarray:
    """Project the complex field ``G`` (recorded at per-ray radii ``rad_mm`` from T) radially to a
    common sphere of radius ``R_mm``.  ``k0`` in rad/mm. ``G`` is in the time-trace ``e^{-ikr}``
    convention (see module docstring), so:

        G_proj = G · (rad/R) · exp(-i k0 (R - rad)).

    Exact inverse with R→rad (round-trip ~1e-16). The operator is applied unconditionally (it is
    O(M) and reduces to the identity when R≡rad); there is deliberately no near-identity fast path,
    which would silently drop the *differential* radial phase across rays of differing ``rad``."""
    rad = np.asarray(rad_mm, float)
    if directions is not None:                     # optional caustic guard on the (radius-free) ray
        d = np.asarray(directions, float)          # directions, since projection preserves direction
        if d.shape[0] > 4:
            from scipy.spatial import cKDTree
            dh = d / np.linalg.norm(d, axis=1, keepdims=True)
            nn = cKDTree(dh).query(dh, k=2)[0][:, 1]
            if np.min(nn) < 1e-6:
                warnings.warn("two projected rays share a direction; resampling may fold a caustic")
    return np.asarray(G, complex) * (rad / R_mm) * np.exp(-1j * k0 * (R_mm - rad))


def energy_on_unit_sphere(G: np.ndarray, rad_mm: np.ndarray) -> np.ndarray:
    """Projection-invariant per-ray energy density  E(r̂) = rad²·|G|²  (independent of the sphere
    radius). This is the right quantity to integrate over a cap. Do NOT additionally apply
    ``metrics.distance_correct`` — that would double-count the spreading."""
    return (np.asarray(rad_mm, float) ** 2) * (np.abs(G) ** 2)
