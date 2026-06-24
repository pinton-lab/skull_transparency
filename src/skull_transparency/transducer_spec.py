"""TransducerSpec — the single user-facing description of the device + drive frequency.

One object is the source of truth for everything downstream:

  (a) the simulation grid pitch          ``dx = c0 / (f0 * ppw)``     -> ``meta['dX_m']``
  (b) the recording-surface geometry      (element coords, half-angle)
  (c) the placement / window constraints  (``BowlConstraints``, acceptance angle)

The existing device geometry (:class:`skull_transparency.transducer.CTX500`,
``ROC_MM``, ``APERTURE_MM``) and the placement-side constraint objects
(:class:`~skull_transparency.placement.BowlConstraints`,
:class:`~skull_transparency.placement.ArrayConstraints`) are unchanged — this type
*derives* them, so adopting it is purely additive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Geometry = Literal["bowl", "annular", "array", "hemisphere_probe"]


@dataclass(frozen=True)
class TransducerSpec:
    """Device + drive description. Lengths in mm, frequency in Hz, speed in m/s.

    ``geometry``:
      * ``"bowl"``     — continuous focused bowl (needs ``roc_mm`` + ``aperture_mm``)
      * ``"annular"``  — concentric-ring bowl, e.g. the CTX-500 (``n_rings=4``)
      * ``"array"``    — discrete elements at ``element_positions_mm`` (device frame)
      * ``"hemisphere_probe"`` — large recording shell (transparency-map survey)
    """
    f0_hz: float                                    # drive frequency        -> meta['F0']
    geometry: Geometry = "bowl"
    # --- focused bowl / annular ---
    roc_mm: Optional[float] = None                  # radius of curvature
    aperture_mm: Optional[float] = None             # full aperture diameter
    n_rings: int = 0                                # annular only (CTX-500 -> 4)
    # --- discrete array ---
    element_positions_mm: Optional[np.ndarray] = None   # (M,3) in the device frame
    n_elements: int = 0
    # --- physics / grid ---
    c0_ms: float = 1540.0                           # reference speed        -> meta['C0']
    ppw: float = 6.0                                # points per wavelength  -> sets dx
    acceptance_angle_deg: float = 35.0              # incidence cap for window scoring

    def __post_init__(self):
        if self.f0_hz <= 0:
            raise ValueError(f"f0_hz must be > 0, got {self.f0_hz}")
        if self.ppw <= 0:
            raise ValueError(f"ppw must be > 0, got {self.ppw}")
        if self.c0_ms <= 0:
            raise ValueError(f"c0_ms must be > 0, got {self.c0_ms}")
        if self.geometry in ("bowl", "annular") and not (self.roc_mm and self.aperture_mm):
            raise ValueError(f"geometry={self.geometry!r} requires roc_mm and aperture_mm")
        if self.roc_mm and self.aperture_mm and self.aperture_mm > 2.0 * self.roc_mm:
            raise ValueError("aperture_mm cannot exceed 2*roc_mm (cap would exceed a hemisphere)")

    # ---- derived geometry / grid ----
    @property
    def wavelength_mm(self) -> float:
        """Acoustic wavelength at the reference speed, ``c0 / f0`` (mm)."""
        return 1e3 * self.c0_ms / self.f0_hz

    @property
    def dx_m(self) -> float:
        """Grid pitch (m) that realises ``ppw`` at the reference speed; == ``meta['dX_m']``.

        ``ppw = wavelength / dx`` so ``dx = c0 / (f0 * ppw)`` — e.g. 1 MHz / 5.5 ppw
        / 1540 m/s gives 0.28 mm, matching the Halle ppw55 grid."""
        return self.c0_ms / (self.f0_hz * self.ppw)

    @property
    def dx_mm(self) -> float:
        return 1e3 * self.dx_m

    @property
    def half_angle_deg(self) -> float:
        """Bowl half-angle from ``arcsin((aperture/2)/roc)``; falls back to the
        acceptance angle for geometries without a defined aperture."""
        if self.roc_mm and self.aperture_mm:
            return float(np.degrees(np.arcsin((self.aperture_mm / 2.0) / self.roc_mm)))
        return self.acceptance_angle_deg

    # ---- adapters to the existing API (no behavioural change) ----
    def to_bowl_constraints(self, focal_length_mm: Optional[float] = None):
        """Derive a :class:`~skull_transparency.placement.BowlConstraints`. The focal
        length defaults to the geometric focus (``roc_mm``)."""
        from .placement import BowlConstraints
        return BowlConstraints(
            focal_length_mm=focal_length_mm if focal_length_mm is not None else self.roc_mm,
            bowl_radius_mm=(self.aperture_mm / 2.0) if self.aperture_mm else 15.0,
            theta_max_deg=self.acceptance_angle_deg,
        )

    def to_array_constraints(self, region_center_mni_mm, region_radius_mm,
                             n_elements: Optional[int] = None,
                             min_spacing_mm: Optional[float] = None):
        """Derive an :class:`~skull_transparency.placement.ArrayConstraints`. Element
        count defaults to ``n_elements``; spacing defaults to a half-wavelength."""
        from .placement import ArrayConstraints
        return ArrayConstraints(
            n_elements=n_elements if n_elements is not None else self.n_elements,
            min_spacing_mm=min_spacing_mm if min_spacing_mm is not None else self.wavelength_mm / 2.0,
            theta_max_deg=self.acceptance_angle_deg,
            region_center_mni_mm=list(region_center_mni_mm),
            region_radius_mm=float(region_radius_mm),
        )

    def to_meta_fields(self) -> dict:
        """The ``meta.json`` fragment this spec owns (grid + physics + transducer block).
        The producer (:mod:`skull_transparency.sim.prepare`) merges this with the
        pose-derived geometry (``N``, ``dent_grid``, ``n_array``, ...)."""
        return {
            "dX_m": float(self.dx_m),
            "C0": float(self.c0_ms),
            "F0": float(self.f0_hz),
            "ppw": float(self.ppw),
            "transducer": {
                "geometry": self.geometry,
                "roc_mm": self.roc_mm,
                "aperture_mm": self.aperture_mm,
                "n_rings": int(self.n_rings),
                "half_angle_deg": self.half_angle_deg,
                "acceptance_angle_deg": float(self.acceptance_angle_deg),
            },
        }

    # ---- named constructors ----
    @classmethod
    def ctx500(cls, f0_hz: float = 500e3, ppw: float = 6.0,
               acceptance_angle_deg: float = 35.0) -> "TransducerSpec":
        """The CTX-500 / NeuroFUS 4-annular bowl (ROC 63.2 mm, aperture 64 mm)."""
        from .transducer import ROC_MM, APERTURE_MM
        return cls(f0_hz=f0_hz, geometry="annular", roc_mm=ROC_MM, aperture_mm=APERTURE_MM,
                   n_rings=4, ppw=ppw, acceptance_angle_deg=acceptance_angle_deg)
