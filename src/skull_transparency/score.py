"""PositioningScore — the single, documented "how good is this window/pose" output.

This introduces **no new physics**: it packages the quantities the placement step
already computes (and that are validated by the transparency tests) into one
self-describing artifact (``score.json``), with the important caveat baked in.

Definition:
  * ``normalized``  — 0..1, the delivered coupling at the chosen placement divided by
    the loss-free ideal for that device. For ``place_bowl`` / ``place_bowl_optimal``
    the optimiser returns the argmax window, so this is **1.0 by construction** (it
    drops below 1 only when a non-optimal window is scored). For the CTX-500 cap
    (``place_cap_optimal``) it is the fraction of cap energy that survives the no-bone
    (foramen) drop, i.e. ``J_cap / J_full`` — genuinely ``< 1`` when a leak is unavoidable.
  * ``focal_pressure_proxy`` — ``sqrt(integral_S |G|^2 dS)`` over the chosen aperture:
    the relative achievable focal peak under the optimal (time-reversal) drive
    (arbitrary units). It is the right proxy for PLACEMENT (delivered energy) but
    **overstates** the broadband time-reversal *focusing* gain and the axial depth
    of field — quote those from a broadband / inward re-simulation (see the README
    "Notes on correctness").
  * ``incidence_deg`` — the window incidence angle (0 == normal incidence).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "skull_transparency.positioning_score/1"

_DEFINITION = (
    "normalized = delivered coupling at the chosen placement / the loss-free ideal for that "
    "device (1.0 = no loss). For bowl/peak placement the optimiser returns the optimal window, "
    "so it is 1.0 by construction; for the CTX-500 cap it is the fraction of cap energy "
    "surviving the foramen/incidence drop (J_cap/J_full), <1 when a leak is unavoidable. "
    "focal_pressure_proxy = sqrt(integral_S |G|^2 dS) = the relative achievable focal peak "
    "under the optimal time-reversal drive (arb. units) — a PLACEMENT/energy proxy that "
    "OVERSTATES the broadband focusing gain and axial DOF; quote those from a "
    "broadband/inward re-simulation. incidence_deg = window incidence (0 = normal)."
)


@dataclass(frozen=True)
class PositioningScore:
    normalized: float            # 0..1: chosen window/pose vs the best achievable
    focal_pressure_proxy: float  # sqrt(integral_S |G|^2 dS): relative focal peak (arb. units)
    incidence_deg: float         # window incidence angle (deg; 0 = normal)
    objective: str               # "peak" | "energy" | "cap" — the per-patch coupling proxy used
    target_name: str = "target"
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_placement(cls, placement, target_name: str = "target") -> "PositioningScore":
        """Extract the score from a :class:`~skull_transparency.placement.BowlPlacement`
        (``place_bowl`` / ``place_bowl_optimal``) or a ``CapPlacement`` (``place_cap_optimal``)."""
        ex = dict(getattr(placement, "extras", {}) or {})
        if hasattr(placement, "transparency_score"):                 # BowlPlacement
            return cls(
                normalized=float(placement.transparency_score),
                focal_pressure_proxy=float(ex.get("p_max_proxy", math.nan)),
                incidence_deg=float(getattr(placement, "incidence_deg", math.nan)),
                objective=str(ex.get("objective", "peak")),
                target_name=target_name,
                extras={"n_footprint_patches": int(getattr(placement, "n_footprint_patches", 0)),
                        "candidate_scores_max": float(ex.get("candidate_scores_max", math.nan))},
            )
        if hasattr(placement, "J_cap"):                              # CapPlacement
            return cls(
                normalized=float(ex.get("score_norm", math.nan)),
                focal_pressure_proxy=math.sqrt(max(float(placement.J_cap), 0.0)),
                incidence_deg=float(getattr(placement, "window_incidence_deg",
                                            ex.get("window_incidence_deg", math.nan))),
                objective="cap", target_name=target_name,
                extras={"J_cap": float(placement.J_cap),
                        "n_kept": int(getattr(placement, "n_kept", 0)),
                        "n_cap": int(getattr(placement, "n_cap", 0))},
            )
        raise TypeError(f"don't know how to score a {type(placement).__name__}")

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "normalized": self.normalized,
            "focal_pressure_proxy": self.focal_pressure_proxy,
            "incidence_deg": self.incidence_deg,
            "objective": self.objective,
            "target_name": self.target_name,
            "definition": _DEFINITION,
            **self.extras,
        }

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))
