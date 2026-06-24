"""Adapter: a :class:`BowlPlacement` (MNI / sim frame) -> the ``placement.json``
schema that ``neuromod_parameters`` consumes (NRRD-voxel-mm via tuba).

Frame chain:  fullres voxel --Registration--> MNI RAS mm
              --tuba.species.human.mni_ras_to_halle_ras--> subject RAS mm
              --halle_ras_to_nrrd_voxel_mm--> NRRD-voxel-mm  (neuromod's frame)

Requires the ``registration`` extra (tuba).  Without tuba the MNI-frame dict is
returned with ``frame='mni_ras_mm'`` so callers can still inspect the geometry."""
from __future__ import annotations

import numpy as np

from .placement import BowlPlacement


def _orthonormal_frame(beam):
    """Build (normal, tangent, bitangent) with normal == beam."""
    n = beam / np.linalg.norm(beam)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t = np.cross(n, seed); t /= np.linalg.norm(t)
    b = np.cross(n, t)
    return n, t, b


def to_placement_dict(placement: BowlPlacement, target_name: str | None = None,
                      species_human=None, world_frame: str = "mni_ras_mm") -> dict:
    """Return the neuromod ``placement.json`` dict.

    ``world_frame`` is the frame ``placement``'s coordinates actually live in (read it
    off the bundle's ``Registration.world_frame``). The tuba MNI->subject->NRRD mapping
    is applied ONLY when ``world_frame == 'mni_ras_mm'`` (the Halle study); for any other
    frame (a generic ``sim.prepare`` subject) the coordinates are returned verbatim --
    pushing a non-MNI subject through tuba's MNI chain would silently corrupt them.

    ``species_human`` is a tuba species module (injected for testability); if None we
    import ``tuba.species.human`` lazily. The OUTPUT FRAME:
      * MNI subject + tuba  -> coordinates in NRRD-voxel-mm  (``frame='nrrd_voxel_mm'``)
      * MNI subject, no tuba -> coordinates in MNI RAS mm     (``frame='mni_ras_mm'``)
      * non-MNI subject     -> coordinates verbatim           (``frame=world_frame``)
    The ``_lps``-suffixed keys (``xdc_center_lps``, ``target_lps``, ``focus_lps``) keep
    neuromod's legacy schema names but hold coordinates in whichever ``frame`` reports —
    they are NOT necessarily LPS. Always read the ``frame`` key."""
    win_mni = np.asarray(placement.window_center_mni_mm, float)
    targ_mni = np.asarray(placement.target_mni_mm, float)
    focal = float(placement.focal_length_mm)

    is_mni = (world_frame == "mni_ras_mm")
    if is_mni and species_human is None:
        try:
            from tuba.species import human as species_human  # type: ignore
        except Exception:
            species_human = None

    if is_mni and species_human is not None:
        to_subj = species_human.mni_ras_to_halle_ras
        to_nrrd = species_human.halle_ras_to_nrrd_voxel_mm
        win = np.asarray(to_nrrd(to_subj(win_mni)), float)
        targ = np.asarray(to_nrrd(to_subj(targ_mni)), float)
        frame = "nrrd_voxel_mm"
    else:
        win, targ, frame = win_mni, targ_mni, world_frame

    # Build the bowl geometry in the OUTPUT frame: beam through the chosen window, apex one
    # focal length from the target so the focus lands on the target. (tuba's MNI->subject map
    # is a non-isometric SyN warp, so the focal length must be applied AFTER mapping.)
    beam = targ - win
    beam /= np.linalg.norm(beam)
    apex = targ - focal * beam
    n, t, bt = _orthonormal_frame(beam)
    return {
        "xdc_center_lps": apex.tolist(),
        "beam_dir_3d": beam.tolist(),
        "beam_3d": beam.tolist(),
        "normal_3d": n.tolist(), "tangent_3d": t.tolist(), "bitangent_3d": bt.tolist(),
        # legacy 2-D schema fields: in-plane projection of the 3-D frame (NOT a 2-D
        # orthonormal basis); the *_3d vectors above are authoritative.
        "normal": n[:2].tolist(), "tangent": t[:2].tolist(),
        "target_lps": targ.tolist(),
        "focus_lps": targ.tolist(),
        "z_mm": float(apex[2]),
        "apex_to_target_mm": float(np.linalg.norm(apex - targ)),
        "focus_to_target_mm": 0.0,
        "target_name": target_name or "target",
        "transparency_score": float(placement.transparency_score),
        "window_used": "transparency",
        "incidence_deg": float(placement.incidence_deg),
        "frame": frame,
    }
