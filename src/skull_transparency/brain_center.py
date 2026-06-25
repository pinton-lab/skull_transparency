"""Seat the omnidirectional time-reversal source at the **center of the brain** for a
neutral whole-skull transparency run.

A brain-center run places one virtual point source at the brain centroid and lets it
radiate in *every* direction, so a single outward solve illuminates the whole calvaria
roughly uniformly (near-normal incidence everywhere, a small spread of source-to-bone
distances). The :func:`~skull_transparency.metrics.distance_correct` (1/r^2) step then
cancels the residual geometric spreading, leaving a map of bone *transmission* that is
not biased toward any one window or target -- the deprecated ``skullonly`` "center"
launcher, generalised.

Two ways to locate the center, best first:

* **Atlas (preferred).** If the subject is in (or rigidly registered to) MNI, map the
  canonical MNI brain center of mass :data:`MNI_BRAIN_COM_MM` into the subject -- exact
  and robust (:func:`brain_center_from_registration`, or the MNI branch of
  :func:`brain_center_phys_mm`). The default is the ICBM152 2009a brain-mask centroid,
  MNI ``(0, -22, 9.5)`` mm.
* **Image-only fallback.** With no atlas, take the intracranial-cavity centroid straight
  from the speed map (:func:`intracranial_centroid`). This is accurate for an intact
  CT (closed vault, brain present) but only ~2 cm-accurate on a *dry* specimen skull
  whose vault is open at the foramen magnum/orbits (hole-filling leaks); prefer the
  atlas path whenever a registration exists.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, label, generate_binary_structure

#: ICBM152 2009a brain-mask center of mass in MNI RAS millimetres (midline, slightly
#: posterior and superior -- near the thalamus/midbrain). Override per atlas if needed.
MNI_BRAIN_COM_MM = (0.0, -22.0, 9.5)


def _is_mni(world_frame: str | None) -> bool:
    return bool(world_frame) and "mni" in str(world_frame).lower()


def _largest_cc_centroid(mask) -> np.ndarray:
    """Voxel centroid of the largest 6-connected component of a boolean ``mask``."""
    lab, n = label(mask, generate_binary_structure(3, 1))
    if n == 0:
        raise ValueError("empty cavity mask")
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.array(np.argwhere(lab == int(sizes.argmax())).mean(0), float)


def _fill_axis(mask, axis) -> np.ndarray:
    """``binary_fill_holes`` on each 2-D slice stacked along ``axis`` (a cavity open in
    one direction -- e.g. the foramen magnum, axially -- is still closed in the other two)."""
    out = np.zeros_like(mask)
    mt, ot = np.moveaxis(mask, axis, 0), np.moveaxis(out, axis, 0)
    for i in range(mt.shape[0]):
        ot[i] = binary_fill_holes(mt[i])
    return out


def intracranial_centroid(c, bone_threshold: float = 2200.0,
                          min_cavity_frac: float = 0.01) -> np.ndarray:
    """Image-only brain centroid (full-res voxel index) from a speed map ``c``.

    Layered for robustness: (1) a 3-D hole fill encloses the brain in an intact vault;
    (2) if that leaks (a dry/open skull), a 2.5-D fill keeps voxels walled off in at
    least two of the three stacking directions; (3) failing both, the bone mass centroid.
    Returns the centroid of the largest enclosed cavity (or the bone centroid)."""
    bone = np.asarray(c) > bone_threshold
    if not bone.any():
        raise ValueError(f"no bone voxels (c > {bone_threshold}); check the speed-map units/threshold.")
    grid = bone.size

    cav = binary_fill_holes(bone) & ~bone                      # (1) closed vault
    if cav.sum() >= min_cavity_frac * grid:
        return _largest_cc_centroid(cav)

    votes = sum((_fill_axis(bone, a) & ~bone) for a in range(3))
    cav = votes >= 2                                           # (2) open/dry vault
    if cav.sum() >= min_cavity_frac * grid:
        return _largest_cc_centroid(cav)

    return np.argwhere(bone).mean(0)                          # (3) last resort


def cavity_mask_centroid(mask, affine=None) -> np.ndarray:
    """Centroid of a *curated* intracranial-cavity mask, as a full-res voxel index, or --
    if ``affine`` (voxel-index -> world-mm) is given -- in world millimetres.

    Use this when a trustworthy endocranial-cavity mask already exists: it is more robust
    than the image-only :func:`intracranial_centroid` hole-fill, which leaks on a *dry*
    specimen open at the foramina. Takes the largest connected component so stray mask
    specks do not bias the center."""
    m = np.asarray(mask) > 0.5
    if not m.any():
        raise ValueError("empty cavity mask")
    v = _largest_cc_centroid(m)
    if affine is None:
        return v
    return (np.asarray(affine, float) @ np.array([*v, 1.0], float))[:3]


def brain_center_from_registration(registration, atlas_com_mm=MNI_BRAIN_COM_MM) -> np.ndarray:
    """Atlas brain center as a full-res voxel index, via a :class:`Registration`
    (``mni_to_fullres`` of the MNI brain CoM). Exact for an MNI-registered subject."""
    return np.asarray(registration.mni_to_fullres(np.asarray(atlas_com_mm, float)), float)


def brain_center_phys_mm(c, affine, *, world_frame: str = "ras_mm",
                         bone_threshold: float = 2200.0,
                         atlas_com_mm=MNI_BRAIN_COM_MM) -> np.ndarray:
    """Brain center in the ``affine`` world frame (mm). If the world frame is MNI the
    atlas CoM is returned verbatim; otherwise the image-only intracranial centroid is
    mapped through ``affine`` (voxel-index -> world-mm)."""
    if _is_mni(world_frame):
        return np.asarray(atlas_com_mm, float)
    v = intracranial_centroid(c, bone_threshold)
    return (np.asarray(affine, float) @ np.array([*v, 1.0], float))[:3]
