"""Exclusion zones: parts of the head the transducer must not touch.

The transparency map ranks every patch of the skull, but some of those patches are not
available: the ears, a scar or a burr hole, an EEG cap's electrodes, a fiducial, the eyes.
An :class:`Exclusion` marks them and answers the two questions placement needs:

* **which window centres are legal** -- a window is illegal if ANY patch of the transducer's
  footprint (every patch within ``footprint_radius_mm`` of the centre) is excluded. Testing
  the centre alone is not enough: a 64 mm bowl centred just outside a 20 mm ear zone still
  sits on the ear. :meth:`Exclusion.legal_centers` buffers by the footprint itself, and
  :class:`~skull_transparency.placement.BowlConstraints` ``exclude_mask`` applies it for you.
* **whether a given transducer pose is over an excluded area** -- :meth:`Exclusion.covered`
  projects transducer element positions onto the skull (nearest patch) and reports which
  zones they land on; the interactive placement tool uses it to flag and skip such poses.

Zones are defined in the **world frame of the map's registration** (``TransparencyMap.surf_mni_mm``):
MNI RAS millimetres for an MNI-registered skull such as the ITRUSST benchmark, otherwise the
subject's own world-mm frame. Two ways to define them, freely combined:

* :class:`Zone` -- a sphere (centre + radius, mm) around a landmark; :data:`EAR_ZONES` gives both
  ears at ear level.
* :func:`image_mask` -- any 3-D mask image (NIfTI) in the same world frame, e.g. drawn on the
  subject's MRI: patches where the image is non-zero are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["Zone", "EAR_ZONES", "zone_mask", "image_mask", "Exclusion"]


@dataclass(frozen=True)
class Zone:
    """A spherical no-go region: every skull patch within ``radius_mm`` of ``center_mm`` (world mm,
    i.e. MNI for an MNI-registered skull) is excluded."""
    name: str
    center_mm: tuple
    radius_mm: float


#: Both ears, centred at ear level (the EEG T3/T4 sites, MNI mm; about the height of the ear canal),
#: 20 mm no-go radius. A starting point: replace the centres with the subject's own preauricular or
#: ear-canal landmarks when you have them.
EAR_ZONES = (Zone("left ear", (-72.0, -22.6, -32.4), 20.0),
             Zone("right ear", (72.4, -23.6, -33.5), 20.0))


def zone_mask(points_mm, zones) -> np.ndarray:
    """(M,) int: index+1 of the first zone containing each point, 0 where none does."""
    p = np.asarray(points_mm, float)
    lab = np.zeros(len(p), np.int32)
    for k, z in enumerate(zones, start=1):
        inside = np.linalg.norm(p - np.asarray(z.center_mm, float), axis=1) <= float(z.radius_mm)
        lab[(lab == 0) & inside] = k
    return lab


def image_mask(points_mm, image, threshold: float = 0.5) -> np.ndarray:
    """(M,) bool: True where a 3-D mask image is above ``threshold`` at each point (nearest voxel).

    ``image`` is a NIfTI path or any object with ``get_fdata()`` and ``affine`` (nibabel). Its
    world frame must be the registration's world frame (MNI for MNI-registered skulls). Points
    outside the image are not excluded."""
    import nibabel as nib                            # optional dependency (the `registration` extra)
    img = nib.load(str(image)) if not hasattr(image, "get_fdata") else image
    data = np.asarray(img.get_fdata(), float)
    if data.ndim == 4:
        data = data[..., 0]
    inv = np.linalg.inv(np.asarray(img.affine, float))
    p = np.asarray(points_mm, float)
    ijk = np.rint(p @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(data.shape)), axis=1)
    out = np.zeros(len(p), bool)
    out[ok] = data[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]] > threshold
    return out


@dataclass
class Exclusion:
    """Excluded skull patches of one transparency map, with the zone each belongs to.

    ``surf_mm`` are the map's patch positions in world mm (``tmap.surf_mni_mm()``); ``label`` is 0
    for an allowed patch, k for zone ``names[k-1]``."""
    surf_mm: np.ndarray
    label: np.ndarray
    names: list = field(default_factory=list)

    def __post_init__(self):
        self.surf_mm = np.asarray(self.surf_mm, float)
        self.label = np.asarray(self.label, np.int32)
        self._all = cKDTree(self.surf_mm)
        ex = np.nonzero(self.label)[0]
        self._ex_idx = ex
        self._ex = cKDTree(self.surf_mm[ex]) if len(ex) else None

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_map(cls, tmap, zones=(), image=None, threshold: float = 0.5,
                 image_name: str = "mask image") -> "Exclusion":
        """Build from a :class:`~skull_transparency.transparency.TransparencyMap` (needs its
        registration) and any combination of :class:`Zone` spheres and a mask image."""
        pts = np.asarray(tmap.surf_mni_mm(), float)
        zones = list(zones)
        lab = zone_mask(pts, zones)
        names = [z.name for z in zones]
        if image is not None:
            hit = image_mask(pts, image, threshold) & (lab == 0)
            names.append(image_name)
            lab[hit] = len(names)
        return cls(pts, lab, names)

    # ---------------------------------------------------------------- queries
    @property
    def excluded(self) -> np.ndarray:
        """(M,) bool: patches the transducer may not touch."""
        return self.label > 0

    def legal_centers(self, footprint_radius_mm: float) -> np.ndarray:
        """(M,) bool: patches that can host a window centre -- no excluded patch lies within
        ``footprint_radius_mm`` of them, so the whole footprint stays clear."""
        if self._ex is None:
            return np.ones(len(self.surf_mm), bool)
        d = self._ex.query(self.surf_mm, workers=-1)[0]
        return d > float(footprint_radius_mm)

    def covered(self, points_mm, margin_mm: float = 0.0) -> list:
        """Names of the zones a transducer covers. Each element position (world mm) is projected
        onto the skull (its nearest patch); a zone is covered if any projected patch lies in it,
        or within ``margin_mm`` of it."""
        if self._ex is None:
            return []
        p = np.asarray(points_mm, float)
        foot = self._all.query(p, workers=-1)[1]
        hit = set(self.label[foot][self.label[foot] > 0].tolist())
        if margin_mm > 0:
            d, j = self._ex.query(self.surf_mm[foot], workers=-1)
            hit |= set(self.label[self._ex_idx[j[d <= margin_mm]]].tolist())
        return [self.names[k - 1] for k in sorted(hit)]

    def summary(self) -> str:
        n = len(self.label)
        lines = [f"{int(self.excluded.sum())} of {n} skull patches excluded "
                 f"({100.0 * self.excluded.mean():.1f}%)"]
        for k, nm in enumerate(self.names, start=1):
            lines.append(f"  {nm}: {int((self.label == k).sum())} patches")
        return "\n".join(lines)
