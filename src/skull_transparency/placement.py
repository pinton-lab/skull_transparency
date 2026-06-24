"""Transducer placement from a skull transparency map.

First deliverable: ``place_bowl`` -- choose the best acoustic window + bowl apex +
beam direction for a single focused bowl, by maximising incidence-weighted
transparency aggregated over the bowl footprint, subject to incidence / region /
legality constraints.  By reciprocity, the high-transparency window is where the
bowl couples the most energy into the target.

The single most important correctness point vs the raw map: incidence uses the
TRUE local surface normal (``true_normal``), so obliquely-hit patches are not
over-credited."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .transparency import TransparencyMap


@dataclass
class BowlConstraints:
    focal_length_mm: float = 30.0
    bowl_radius_mm: float = 15.0          # aperture radius -> surface footprint radius
    theta_max_deg: float = 35.0           # hard incidence cut (true-normal vs beam)
    objective: str = "peak"               # placement energy proxy. "peak": Ipk_Wcm2 (direct-
    #   arrival peak intensity) — coda-robust, the TR-optimal objective (footprint sum=∫_S E dS,
    #   p_max=√J). "energy": full-record Iint — reverberation-contaminated on the dense surface
    #   (mis-selects parietal); kept for completeness, NOT for placement.
    min_value_pctile: float = 0.0         # ignore patches below this percentile of the coupling metric
    max_candidates: int = 4000            # score only the top-K legal patches by transparency
    use_distance_corrected: bool = False  # placement uses RAW delivered intensity (reciprocity):
    #   raw outward intensity at a patch == what a transducer there delivers to the target
    #   (spreading included), so it ranks delivered energy correctly. The distance-corrected
    #   map removes the genuine proximity advantage and is for *visualising* bone transmission.
    region_center_mni_mm: np.ndarray | None = None   # optional spherical restriction
    region_radius_mm: float | None = None
    legal_mask: np.ndarray | None = None  # (M,) bool of allowed patches (e.g. tuba vault/no-go)


@dataclass
class BowlPlacement:
    apex_mni_mm: np.ndarray
    beam_dir_mni: np.ndarray              # unit, apex -> target
    target_mni_mm: np.ndarray
    window_center_mni_mm: np.ndarray
    transparency_score: float            # chosen-window footprint score / best legal candidate.
    #   ==1.0 BY CONSTRUCTION here: place_bowl returns the argmax window, so it equals its own max.
    #   It drops below 1 only when a NON-optimal window is scored. Use incidence_deg / the footprint
    #   extras (not this) to discriminate the optimal placement's quality.
    apex_fullres_voxel: np.ndarray
    window_center_fullres_voxel: np.ndarray
    incidence_deg: float
    focal_length_mm: float
    bowl_radius_mm: float
    n_footprint_patches: int
    footprint_surf_idx: np.ndarray | None = None   # indices into the DENSE surface map (surf_vox)
    #   inside the winning window footprint — NOT array-element indices. Do not pass to the
    #   ComplexField drive/PSF functions (which index the 1599 array elements); select array
    #   elements for that aperture separately (by coupling/proximity, as in the example).
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "apex_mni_mm": np.asarray(self.apex_mni_mm).tolist(),
            "beam_dir_mni": np.asarray(self.beam_dir_mni).tolist(),
            "target_mni_mm": np.asarray(self.target_mni_mm).tolist(),
            "window_center_mni_mm": np.asarray(self.window_center_mni_mm).tolist(),
            "transparency_score": float(self.transparency_score),
            "incidence_deg": float(self.incidence_deg),
            "focal_length_mm": float(self.focal_length_mm),
            "bowl_radius_mm": float(self.bowl_radius_mm),
            "n_footprint_patches": int(self.n_footprint_patches),
        }


def _incidence_cos(tmap: TransparencyMap) -> np.ndarray:
    """cos(incidence) = <rhat, true_normal>; rhat is (patch - target)/|.|, so a
    well-aimed bowl whose axis is along the local normal has cos ~ 1."""
    n = tmap.true_normal
    if not np.any(n):                                  # no normals -> fall back to radial
        return np.ones(len(tmap.surf_vox))
    return np.clip(np.sum(tmap.rhat * n, axis=1), -1.0, 1.0)


def place_bowl(tmap: TransparencyMap, constraints: BowlConstraints = BowlConstraints()) -> BowlPlacement:
    if tmap.registration is None:
        raise ValueError("TransparencyMap needs a Registration for MNI output")
    reg = tmap.registration
    c = constraints
    sv = tmap.surf_vox
    surf_mni = reg.fullres_to_mni(sv)
    target_mni = np.asarray(reg.target_mni_mm, float)
    dx_mm = reg.dx_mm

    # coupling proxy: RAW delivered intensity (reciprocity) by default; the distance-
    # corrected map only for "equal-distance bone-transparency" comparisons. With
    # objective="energy" use the broadband windowed energy Iint (the TR-optimal placement
    # objective J=∫_S E dS), whose footprint sum gives p_max=√J.
    if c.objective == "energy":
        coupling = np.asarray(tmap.Iint, float)
    else:
        coupling = np.asarray(tmap.value if c.use_distance_corrected else tmap.Ipk_Wcm2, float)
    cos_inc = _incidence_cos(tmap)
    w_inc = np.clip(cos_inc, 0.0, 1.0) ** 2
    Tw = coupling * w_inc                                # incidence-weighted coupling

    # ---- legality mask ----
    legal = cos_inc >= np.cos(np.deg2rad(c.theta_max_deg))
    if c.min_value_pctile > 0:
        legal &= coupling >= np.percentile(coupling, c.min_value_pctile)
    if c.region_center_mni_mm is not None and c.region_radius_mm is not None:
        legal &= np.linalg.norm(surf_mni - np.asarray(c.region_center_mni_mm, float), axis=1) <= c.region_radius_mm
    if c.legal_mask is not None:
        legal &= np.asarray(c.legal_mask, bool)
    if not legal.any():
        raise ValueError("no legal patches under the given constraints")

    # ---- aggregate transparency over the bowl footprint (sum of Tw within bowl_radius) ----
    # Only the top-K legal patches by transparency can host the optimal window, so cap the
    # (expensive) ball-sum to those — orders of magnitude faster, same winner.
    tree = cKDTree(surf_mni)
    cand = np.where(legal)[0]
    if len(cand) > c.max_candidates:
        cand = cand[np.argsort(-Tw[cand])[:c.max_candidates]]
    score = np.zeros(len(cand))
    nfoot = np.zeros(len(cand), int)
    for a, i in enumerate(cand):
        nbr = tree.query_ball_point(surf_mni[i], c.bowl_radius_mm)
        score[a] = Tw[nbr].sum()
        nfoot[a] = len(nbr)
    if not np.isfinite(score).any():
        raise ValueError("all window scores are non-finite (NaN/inf in the transparency map?)")
    bi = int(np.nanargmax(score))                        # nanargmax: a stray NaN never wins
    best = cand[bi]
    score_norm = float(score[bi] / (np.nanmax(score) + 1e-30))  # 1.0 for the winner

    # ---- geometry: window contact -> beam -> apex at focal_length from the target ----
    win_mni = surf_mni[best]
    beam = target_mni - win_mni
    beam /= np.linalg.norm(beam)                         # apex -> target direction (into head)
    apex_mni = target_mni - c.focal_length_mm * beam     # focus assumed at target
    apex_vox = reg.mni_to_fullres(apex_mni)
    footprint_idx = np.asarray(tree.query_ball_point(surf_mni[best], c.bowl_radius_mm), int)
    p_max = float(np.sqrt(max(score[bi], 0.0)))          # √(∫_S E dS) when objective="energy"

    return BowlPlacement(
        apex_mni_mm=apex_mni, beam_dir_mni=beam, target_mni_mm=target_mni,
        window_center_mni_mm=win_mni,
        transparency_score=score_norm,
        apex_fullres_voxel=apex_vox, window_center_fullres_voxel=sv[best],
        incidence_deg=float(np.degrees(np.arccos(cos_inc[best]))),
        focal_length_mm=c.focal_length_mm, bowl_radius_mm=c.bowl_radius_mm,
        n_footprint_patches=int(nfoot[bi]), footprint_surf_idx=footprint_idx,
        extras={"window_value": float(tmap.value[best]),
                "apex_to_target_mm": float(np.linalg.norm(apex_mni - target_mni)),
                "candidate_scores_max": float(score.max()),
                "objective": c.objective, "p_max_proxy": p_max},
    )


@dataclass
class ArrayConstraints:
    """Sparse-array element selection."""
    n_elements: int = 128
    min_spacing_mm: float = 4.0           # element center-to-center (>= element diameter)
    theta_max_deg: float = 35.0           # hard incidence cut (true-normal vs beam)
    use_distance_corrected: bool = False  # RAW delivered coupling by default (reciprocity)
    min_value_pctile: float = 0.0
    max_candidates: int = 60000           # cap the greedy walk (sorted by coupling)
    region_center_mni_mm: np.ndarray | None = None
    region_radius_mm: float | None = None
    legal_mask: np.ndarray | None = None


@dataclass
class ArrayLayout:
    element_mni_mm: np.ndarray            # (n,3) chosen element centres
    element_fullres_voxel: np.ndarray     # (n,3)
    orientation_mni: np.ndarray           # (n,3) element axis -> target (unit)
    incidence_deg: np.ndarray             # (n,)
    coupling: np.ndarray                  # (n,) per-element coupling metric
    target_mni_mm: np.ndarray
    aggregate_coupling: float             # sum of per-element coupling (delivered-energy proxy)
    n_requested: int
    aperture_extent_mm: float             # max pairwise element separation
    extras: dict = field(default_factory=dict)

    @property
    def n_placed(self) -> int:
        return len(self.element_mni_mm)


def place_array(tmap: TransparencyMap, constraints: ArrayConstraints = ArrayConstraints()) -> ArrayLayout:
    """Greedy sparse-array selection: take the highest incidence-weighted-coupling patches
    that respect a minimum center-to-center spacing, incidence cap and legality masks.
    By reciprocity this maximises the available delivered energy for an N-element aperture;
    inter-element *coherence* still needs the inward re-simulation (see docs)."""
    if tmap.registration is None:
        raise ValueError("TransparencyMap needs a Registration for MNI output")
    reg = tmap.registration
    c = constraints
    sv = tmap.surf_vox
    surf_mni = reg.fullres_to_mni(sv)
    target_mni = np.asarray(reg.target_mni_mm, float)

    coupling = np.asarray(tmap.value if c.use_distance_corrected else tmap.Ipk_Wcm2, float)
    cos_inc = _incidence_cos(tmap)
    w_inc = np.clip(cos_inc, 0.0, 1.0) ** 2
    Tw = coupling * w_inc

    legal = cos_inc >= np.cos(np.deg2rad(c.theta_max_deg))
    if c.min_value_pctile > 0:
        legal &= coupling >= np.percentile(coupling, c.min_value_pctile)
    if c.region_center_mni_mm is not None and c.region_radius_mm is not None:
        legal &= np.linalg.norm(surf_mni - np.asarray(c.region_center_mni_mm, float), axis=1) <= c.region_radius_mm
    if c.legal_mask is not None:
        legal &= np.asarray(c.legal_mask, bool)
    if not legal.any():
        raise ValueError("no legal patches under the given constraints")

    # greedy: walk candidates by descending coupling, accept if >= min_spacing from all accepted.
    cand = np.where(legal)[0]
    cand = cand[np.argsort(-Tw[cand])]
    if len(cand) > c.max_candidates:
        cand = cand[:c.max_candidates]
    s2 = c.min_spacing_mm ** 2
    chosen: list[int] = []
    acc = np.empty((0, 3))
    for i in cand:
        p = surf_mni[i]
        if acc.shape[0] == 0 or np.min(np.sum((acc - p) ** 2, axis=1)) >= s2:
            chosen.append(int(i))
            acc = np.vstack([acc, p])
            if len(chosen) == c.n_elements:
                break
    chosen = np.asarray(chosen, int)

    el_mni = surf_mni[chosen]
    orient = target_mni - el_mni
    orient /= np.linalg.norm(orient, axis=1, keepdims=True)
    inc = np.degrees(np.arccos(np.clip(cos_inc[chosen], -1, 1)))
    extent = 0.0
    if len(chosen) > 1:
        d = cKDTree(el_mni)
        extent = float(d.query(el_mni, k=len(el_mni))[0].max())
    return ArrayLayout(
        element_mni_mm=el_mni, element_fullres_voxel=sv[chosen], orientation_mni=orient,
        incidence_deg=inc, coupling=coupling[chosen], target_mni_mm=target_mni,
        aggregate_coupling=float(coupling[chosen].sum()), n_requested=c.n_elements,
        aperture_extent_mm=extent,
        extras={"min_spacing_mm": c.min_spacing_mm, "theta_max_deg": c.theta_max_deg,
                "n_legal": int(legal.sum())},
    )


# ---------------------------------------------------------------------------
# Phase-aware placement, drives and focal-spot prediction (surface-integral method).
# These consume a ComplexField (phase from the array traces); the placement OBJECTIVE itself
# (below, place_bowl_optimal) only needs the broadband energy and stays on the dense map.
# ---------------------------------------------------------------------------

def place_bowl_optimal(tmap: TransparencyMap, constraints: BowlConstraints = BowlConstraints()) -> BowlPlacement:
    """The TR-optimal window placement: maximise the surface integral J(S)=∫_S E dS of the
    delivered (direct-arrival) energy over the bowl footprint; the achievable focal peak under
    the optimal (time-reversal) drive is ∝ √J (in ``extras['p_max_proxy']``). One precomputed
    map ⇒ every candidate window is a footprint sum (a moving-cap correlation), no per-candidate
    wave solve (cf. SCOUT).

    Uses the **peak intensity** ``Ipk_Wcm2`` as the per-patch energy proxy — it is dominated by
    the direct (ballistic) arrival and so is coda-robust, matching the ballistic-windowed array
    energy ``E_win``. (The full-record time-integral ``Iint`` — ``objective='energy'`` — is
    reverberation-contaminated on the dense surface and mis-selects a parietal window; do not use
    it for placement. This is the surface analogue of the ballistic-windowing requirement on the
    array traces.)"""
    from dataclasses import replace
    return place_bowl(tmap, replace(constraints, objective="peak"))


# ---------------------------------------------------------------------------
# CTX-500 cap-aperture placement objective (foramen-dropped, pose-optimized).
#
# ``place_bowl_optimal`` integrates the delivered-energy density over a fixed 15 mm f/0.6
# FOOTPRINT (``bowl_radius_mm``) with the beam aimed straight at the target (it chooses only a
# window CENTRE). For a realistic CTX-500 cap (ROC 63.2 mm, 64 mm aperture) that objective has
# three structural blind spots, which matter for the dentate (deep posterior fossa):
#   (a) the foramen-magnum leak is a RIM effect of the 64 mm aperture, entirely outside the
#       15 mm footprint, so the footprint score never sees it;
#   (b) it has no term for how much energy the per-element no-bone DROP removes — at the
#       sqrt(J) optimum a full cap aims ~14% of (high-coupling) elements through the foramen,
#       so "argmax sqrt(J) then drop" picks a wasteful window;
#   (c) it cannot express the aim (tilt/yaw) and standoff that collapse that drop.
#
# The functions below score the ACTUAL cap AFTER the per-element no-bone drop (the foramen
# exclusion: drop elements whose target->element ray crosses ZERO bone), over the full pose:
#
#     J_cap(pose) = sum_{kept i} |G(x_i)|^2 * clip(cos theta_i, 0, 1)^2 ,
#
# KEPT = cap elements whose ray to the target crosses bone; |G(x_i)|^2 is the recorded outward
# surface energy radially PROJECTED to the cap-element radius R_i by the Pinton-Aubry-Tanter
# operator (|G(x_i)|^2 = E(r_hat_i)/R_i^2 with E = r^2|G|^2 the projection invariant, see
# projection.py); theta_i is the incidence of the element ray on the local surface. This stays
# single-solve post-processing on the one recorded outward field — NO new wave solve. For a
# foramen-free target (thalamus, dACC) the drop is ~0 and the optimum coincides with
# place_bowl_optimal's window, so the objective is target-agnostic.
# ---------------------------------------------------------------------------

@dataclass
class CapField:
    """Precomputed surface quantities for the cap objective, from the single outward solve.

    Build with :meth:`from_transparency_map`. All geometry is in the DOMAIN-VOXEL frame (the
    frame the cap is posed in); ``E_inv`` is the projection-invariant per-ray energy r^2|G|^2."""
    surf_dir: np.ndarray          # (M,3) unit target->patch directions
    E_inv: np.ndarray             # (M,)  invariant ray energy = rad_mm^2 * Ipk  (proportional to E)
    normal: np.ndarray            # (M,3) outward surface normals
    surf_vox: np.ndarray          # (M,3) patch voxels (window reporting / MNI)
    rad_mm: np.ndarray            # (M,)  patch radius from target (mm)
    target_vox: np.ndarray        # (3,)
    dx_mm: float
    tree: "cKDTree"               # cKDTree on surf_dir (nearest ray direction)
    registration: object = None

    @classmethod
    def from_transparency_map(cls, tmap: TransparencyMap) -> "CapField":
        if tmap.registration is None:
            raise ValueError("TransparencyMap needs a Registration (target voxel + dx)")
        reg = tmap.registration
        surf_dir = np.asarray(tmap.rhat, float)
        E_inv = np.asarray(tmap.rad_mm, float) ** 2 * np.asarray(tmap.Ipk_Wcm2, float)
        # Orient the surface normal to FACE the target's source (n . rhat >= 0, rhat = target->patch),
        # i.e. the patch face the bowl couples to. A bare bone-occupancy gradient normal is sometimes
        # flipped on the oblique base/occiput (it can point back toward the target), which would
        # spuriously zero-clip those rim patches in the incidence weight and corrupt the cap integral.
        # This matches the outward-orientation the interactive optimizer field applies.
        nrm = np.asarray(tmap.true_normal, float)
        sgn = np.sign(np.sum(nrm * surf_dir, axis=1))
        sgn[sgn == 0] = 1.0
        nrm = nrm * sgn[:, None]
        return cls(surf_dir=surf_dir, E_inv=E_inv, normal=nrm,
                   surf_vox=np.asarray(tmap.surf_vox, float), rad_mm=np.asarray(tmap.rad_mm, float),
                   target_vox=np.asarray(reg.target_fullres_voxel, float), dx_mm=float(reg.dx_mm),
                   tree=cKDTree(surf_dir), registration=reg)


@dataclass
class CapPlacement:
    pose: "object"                       # transducer.CapPose
    J_cap: float                         # after-drop cap objective (arb. units; argmax is meaningful)
    n_cap: int
    n_kept: int
    n_dropped: int
    drop_frac: float                     # fraction of cap elements dropped (no-bone foramen leak)
    window_vox: np.ndarray
    window_mni_mm: np.ndarray | None
    window_dist_mm: float
    window_incidence_deg: float
    apex_vox: np.ndarray
    aim_vox: np.ndarray
    focus_vox: np.ndarray
    focus_to_target_mm: float
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        p = self.pose
        return {
            "pose": {"az_deg": p.az_deg, "el_deg": p.el_deg, "radius_mm": p.radius_mm,
                     "tilt_deg": p.tilt_deg, "yaw_deg": p.yaw_deg},
            "J_cap": float(self.J_cap), "n_cap": int(self.n_cap), "n_kept": int(self.n_kept),
            "n_dropped": int(self.n_dropped), "drop_frac": float(self.drop_frac),
            "window_mni_mm": (None if self.window_mni_mm is None
                              else np.asarray(self.window_mni_mm).tolist()),
            "window_dist_mm": float(self.window_dist_mm),
            "window_incidence_deg": float(self.window_incidence_deg),
            "focus_to_target_mm": float(self.focus_to_target_mm),
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.extras.items()},
        }


def score_cap_pose(field: CapField, pose, bone_ray_test, *, roc_mm: float = 63.2,
                   aperture_mm: float = 64.0, density: float = 1.0) -> dict:
    """Score ONE pose: build the CTX-500 cap, drop no-bone (foramen) elements, project the
    recorded surface energy to each kept element and incidence-weight it, and sum.

    ``bone_ray_test(cap_pts, target_vox) -> kept_mask`` is injected (True == the element's ray to
    the target crosses bone == keep; False == no-bone foramen leak == drop). Keeping it a callable
    keeps this module free of the 1.38 GB medium and unit-testable. Returns a dict with ``J_cap``,
    drop counts, and the window patch (beam-axis entry)."""
    from .transducer import build_cap_pose
    roc_vox = roc_mm / field.dx_mm
    half = float(np.degrees(np.arcsin((aperture_mm / 2.0) / roc_mm)))
    pts, focus, apex, aim = build_cap_pose(field.target_vox, pose, roc_vox, field.dx_mm,
                                           half_angle_deg=half, density=density)
    pts = np.unique(np.round(pts).astype(int), axis=0).astype(float)   # match export/solver dedup
    n_cap = len(pts)
    keep = np.asarray(bone_ray_test(pts, field.target_vox), bool)      # True = crosses bone = keep
    n_kept = int(keep.sum())
    # Project + incidence-weight EVERY cap element to get the loss-free reference energy J_full;
    # the kept subset is the after-drop delivered energy J_cap. Boolean masks preserve order, so
    # ``contrib[keep]`` sums the kept elements in the SAME order as the old kept-only path —
    # J_cap is bit-identical; J_full / score_norm are purely additive.
    if n_cap:
        d = pts - field.target_vox
        L = np.linalg.norm(d, axis=1)
        R_mm = L * field.dx_mm
        dhat = d / (L[:, None] + 1e-30)
        j = field.tree.query(dhat, workers=-1)[1]
        G2 = field.E_inv[j] / (R_mm ** 2 + 1e-30)                      # Pinton radial projection to R_i
        cos_i = np.clip(np.sum(dhat * field.normal[j], axis=1), 0.0, 1.0)
        contrib = G2 * cos_i ** 2
        J_full = float(contrib.sum())
        J_cap = float(contrib[keep].sum())
    else:
        J_full = J_cap = 0.0
    # positioning efficiency in [0,1]: fraction of the cap's incidence-weighted energy that
    # survives the no-bone (foramen) drop (1.0 = nothing dropped; <1 = an unavoidable leak).
    score_norm = (J_cap / J_full) if J_full > 0.0 else 0.0
    # window = surface patch along the beam axis (target -> apex direction)
    da = apex - field.target_vox
    da = da / (np.linalg.norm(da) + 1e-30)
    wj = int(field.tree.query(da, workers=-1)[1])
    cos_w = float(np.clip(field.surf_dir[wj] @ field.normal[wj], -1.0, 1.0))
    return dict(J_cap=J_cap, J_full=J_full, score_norm=score_norm,
                n_cap=n_cap, n_kept=n_kept, n_dropped=n_cap - n_kept,
                drop_frac=(n_cap - n_kept) / max(n_cap, 1), window_idx=wj,
                window_vox=field.surf_vox[wj], window_dist_mm=float(field.rad_mm[wj]),
                window_incidence_deg=float(np.degrees(np.arccos(cos_w))),
                pose=pose, apex=apex, aim=aim, focus=focus)


def _cap_placement_from_score(field: CapField, s: dict) -> CapPlacement:
    reg = field.registration
    foc_to_tgt = float(np.linalg.norm(s["focus"] - field.target_vox) * field.dx_mm)
    win_mni = (reg.fullres_to_mni(s["window_vox"]) if reg is not None else None)
    return CapPlacement(
        pose=s["pose"], J_cap=s["J_cap"], n_cap=s["n_cap"], n_kept=s["n_kept"],
        n_dropped=s["n_dropped"], drop_frac=s["drop_frac"], window_vox=s["window_vox"],
        window_mni_mm=win_mni, window_dist_mm=s["window_dist_mm"],
        window_incidence_deg=s["window_incidence_deg"], apex_vox=s["apex"], aim_vox=s["aim"],
        focus_vox=s["focus"], focus_to_target_mm=foc_to_tgt,
        extras={"window_idx": s["window_idx"], "score_norm": s["score_norm"],
                "J_full": s["J_full"]})


def place_cap_optimal(field: CapField, bone_ray_test, *, seed_az_deg: float, seed_el_deg: float,
                      roc_mm: float = 63.2, aperture_mm: float = 64.0, theta_max_deg: float = 35.0,
                      radius_mm: float | None = None, search_density: float = 0.2,
                      final_density: float = 1.0, az_halfspan_deg: float = 55.0,
                      el_halfspan_deg: float = 50.0, n_az: int = 23, n_el: int = 21,
                      refine_pose: bool = True, tilt_span_deg: float = 12.0,
                      yaw_span_deg: float = 12.0, n_tilt: int = 7, n_yaw: int = 7,
                      log=None) -> CapPlacement:
    """Maximise the after-drop cap objective J_cap over the CTX-500 pose, by a coarse (az, el)
    sweep around ``(seed_az_deg, seed_el_deg)`` (straight aim, standoff == ROC), then a local
    tilt/yaw/az/el refinement, with a final full-density re-score of the winner. The window is
    constrained incidence-legal (beam-axis patch incidence <= ``theta_max_deg``). Pure
    post-processing — every score is a moving-cap summation on the one recorded field, no wave
    solve. Returns a :class:`CapPlacement`."""
    from .transducer import CapPose
    radius_mm = roc_mm if radius_mm is None else radius_mm
    _log = (lambda *_: None) if log is None else log

    def score(az, el, tilt=0.0, yaw=0.0, density=search_density):
        return score_cap_pose(field, CapPose(az, el, radius_mm, tilt, yaw), bone_ray_test,
                              roc_mm=roc_mm, aperture_mm=aperture_mm, density=density)

    def legal(s):
        return s["window_incidence_deg"] <= theta_max_deg and s["n_kept"] > 0

    # ---- 1) coarse (az, el) sweep, straight aim ----
    azs = seed_az_deg + np.linspace(-az_halfspan_deg, az_halfspan_deg, n_az)
    els = np.clip(seed_el_deg + np.linspace(-el_halfspan_deg, el_halfspan_deg, n_el), -89.0, 89.0)
    best = None
    for az in azs:
        for el in els:
            s = score(az, el)
            if legal(s) and (best is None or s["J_cap"] > best["J_cap"]):
                best = s
    if best is None:
        raise ValueError("no incidence-legal pose found in the coarse sweep")
    _log(f"[cap] coarse best az{best['pose'].az_deg:+.0f} el{best['pose'].el_deg:+.0f} "
         f"J={best['J_cap']:.3e} drop={100*best['drop_frac']:.2f}% inc={best['window_incidence_deg']:.0f}")

    # ---- 2) local az/el refinement (half the coarse step) ----
    if refine_pose:
        d_az = (azs[1] - azs[0]) if n_az > 1 else 5.0
        d_el = (els[1] - els[0]) if n_el > 1 else 5.0
        for az in best["pose"].az_deg + np.linspace(-d_az, d_az, 5):
            for el in np.clip(best["pose"].el_deg + np.linspace(-d_el, d_el, 5), -89.0, 89.0):
                s = score(az, el)
                if legal(s) and s["J_cap"] > best["J_cap"]:
                    best = s
        # ---- 3) tilt / yaw refinement at the basin ----
        b = best["pose"]
        for tilt in np.linspace(-tilt_span_deg, tilt_span_deg, n_tilt):
            for yaw in np.linspace(-yaw_span_deg, yaw_span_deg, n_yaw):
                s = score(b.az_deg, b.el_deg, tilt, yaw)
                if legal(s) and s["J_cap"] > best["J_cap"]:
                    best = s
        _log(f"[cap] refined best az{best['pose'].az_deg:+.0f} el{best['pose'].el_deg:+.0f} "
             f"tilt{best['pose'].tilt_deg:+.0f} yaw{best['pose'].yaw_deg:+.0f} "
             f"J={best['J_cap']:.3e} drop={100*best['drop_frac']:.2f}%")

    # ---- 4) final full-density re-score of the winning pose ----
    final = score_cap_pose(field, best["pose"], bone_ray_test, roc_mm=roc_mm,
                           aperture_mm=aperture_mm, density=final_density)
    return _cap_placement_from_score(field, final)


def _elem_idx(cf, idx):
    """Validate that ``idx`` indexes the ComplexField's array elements (not dense-surface patches —
    a BowlPlacement.footprint_surf_idx would be out of range here)."""
    idx = np.asarray(idx, int)
    M = np.asarray(cf.G_win).size
    if idx.size and (idx.min() < 0 or idx.max() >= M):
        raise ValueError(
            f"element index out of range for the {M}-element ComplexField (got "
            f"[{idx.min()},{idx.max()}]). These functions take ARRAY-ELEMENT indices; do not pass "
            "BowlPlacement.footprint_surf_idx (dense-surface indices). Select array elements for the "
            "aperture by coupling/proximity instead.")
    return idx


def drive_optimal(cf, idx, P: float = 1.0):
    """Time-reversal (phase-conjugate) drive for the chosen elements, power-normalised to
    Σ|u|²=P. Returns (u complex, p_max). p_max=√(P·Σ|G|²) is the Cauchy-Schwarz optimum — no
    unit-power drive exceeds it. Uses the ballistic-windowed phasor G_win."""
    g = np.asarray(cf.G_win)[_elem_idx(cf, idx)]
    if g.size == 0:
        return np.zeros(0, complex), 0.0
    nrm = float(np.sqrt((np.abs(g) ** 2).sum()))
    u = np.sqrt(P) * np.conj(g) / (nrm + 1e-300)
    return u, float(np.sqrt(P) * nrm)


def drive_phase_only(cf, idx, P: float = 1.0):
    """Equal-amplitude phase-only (phase conjugate) drive, Σ|u|²=P. Returns (u, p_max) with
    p_max=√(P/N)·Σ|G|. Ratio to drive_optimal = 1/√(1+CV²(|G|)) (apodisation loss)."""
    g = np.asarray(cf.G_win)[_elem_idx(cf, idx)]
    N = len(g)
    if N == 0:
        return np.zeros(0, complex), 0.0
    u = np.sqrt(P / N) * np.exp(-1j * np.angle(g))
    return u, float(np.sqrt(P / N) * np.abs(g).sum())


def drive_single_element(cf, idx, P: float = 1.0):
    """SINGLE-ELEMENT (no per-element phase control) drive: one waveform across the whole face with
    the element's fixed geometric-focusing curvature φ_geo,i = k0·r_i (so a homogeneous medium would
    focus at T). The residual skull aberration is NOT corrected, so the on-target peak is the
    COHERENT sum p=√(P/N)·|Σ_i G_i e^{i k0 r_i}| = √(P/N)·|Σ_i |G_i| e^{iΔφ_i}|, Δφ the aberration
    phase. This is the single-element limit (cf. Park et al. 2022's score |Σ A e^{iΔφ}|); contrast
    drive_optimal (phased array: phase corrected away → energy √(P·Σ|G|²)). Phase coherence across
    the face MATTERS here, so the single-element placement optimum differs from the array's."""
    ii = _elem_idx(cf, idx)
    g = np.asarray(cf.G_win)[ii]
    N = len(g)
    if N == 0:
        return np.zeros(0, complex), 0.0
    phi_geo = cf.k0 * np.asarray(cf.radius_mm)[ii]
    u = np.sqrt(P / N) * np.exp(1j * phi_geo)
    return u, float(np.sqrt(P / N) * np.abs((g * np.exp(1j * phi_geo)).sum()))


def coherence_factor(cf, idx) -> float:
    """Spatial phase-coherence γ(S)=|Σ_i G_i e^{i k0 r_i}| / Σ_i|G_i| ∈ [0,1] of a candidate
    aperture: the fraction of the available amplitude a SINGLE fixed-phase element captures (γ=1 ⇔
    flat aberration across the face; γ→0 ⇔ aberration scrambles the coherent sum). A phased array
    effectively restores γ→1 by per-element delays; the single-element penalty is exactly 1−γ."""
    ii = _elem_idx(cf, idx)
    g = np.asarray(cf.G_win)[ii]
    if len(g) == 0:
        return 0.0
    phi_geo = cf.k0 * np.asarray(cf.radius_mm)[ii]
    return float(np.abs((g * np.exp(1j * phi_geo)).sum()) / (np.abs(g).sum() + 1e-300))


def focal_psf(cf, idx, offsets_mm: np.ndarray) -> np.ndarray:
    """Angular-spectrum focal point-spread under the TR drive, evaluated at displacements
    ``offsets_mm`` (Np,3) from the target T (mm).  p(T') ≈ Σ_i |G_i|² exp(-i k0 r̂_i·ΔT'),
    r̂_i = unit(x_i−T) (the ray direction). Returns |p| (Np,). ORDER-of-magnitude shape predictor
    only (transverse FWHM within ~±35% of a full solve; single-frequency overestimates the axial
    depth of field)."""
    idx = _elem_idx(cf, idx)
    nhat = cf.direction()[idx]                       # (n,3) target->element
    W = np.abs(np.asarray(cf.G_win)[idx]) ** 2       # |G|^2 apodisation
    off = np.atleast_2d(np.asarray(offsets_mm, float))
    phase = -cf.k0 * (off @ nhat.T)                  # (Np, n)
    return np.abs((W[None, :] * np.exp(1j * phase)).sum(1))


def focal_fwhm(cf, idx, axes: np.ndarray | None = None, span_mm: float = 60.0, n: int = 2401):
    """-6 dB (half-max-intensity) FWHM of the angular-spectrum PSF along three beam-frame axes
    (axial = mean ray direction; lateral, elevation complete it). Returns dict with
    fwhm_axial/lateral/elevation_mm (``inf`` if the half-max is not reached within ``span_mm``).
    Order-correct (see :func:`focal_psf`); needs ≥2 non-collinear elements."""
    idx = _elem_idx(cf, idx)
    nhat = cf.direction()[idx]
    axial = nhat.mean(0); na = np.linalg.norm(axial)
    if len(idx) < 2 or na < 1e-6:                    # single element or symmetric/degenerate aperture
        nan = float("nan")
        import warnings
        warnings.warn("degenerate aperture (too few or near-symmetric elements); FWHM undefined")
        return {"fwhm_axial_mm": nan, "fwhm_lateral_mm": nan, "fwhm_elevation_mm": nan}
    axial = axial / na
    ref = np.array([1., 0, 0]) if abs(axial[2]) < 0.9 else np.array([0, 1., 0])
    lat = np.cross(axial, ref); lat /= np.linalg.norm(lat); elev = np.cross(axial, lat)
    s = np.linspace(-span_mm, span_mm, n)

    def width(d):
        I = focal_psf(cf, idx, s[:, None] * d[None, :]) ** 2
        I = I / I.max(); half = 0.5; im = int(np.argmax(I)); l = im; r = im
        while l > 0 and I[l] > half:
            l -= 1
        while r < n - 1 and I[r] > half:
            r += 1
        if (l == 0 and I[l] > half) or (r == n - 1 and I[r] > half):
            return float("inf")                      # half-max not crossed within the scanned span
        cr = lambda a, b: s[a] + (half - I[a]) / (I[b] - I[a] + 1e-30) * (s[b] - s[a])
        return float(cr(r, r - 1) - cr(l, l + 1))
    return {"fwhm_axial_mm": width(axial), "fwhm_lateral_mm": width(lat),
            "fwhm_elevation_mm": width(elev)}
