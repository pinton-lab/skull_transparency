"""Compute a skull acoustic-transparency map from a time-reversal Field Bundle.

Generalises ``hemisphere_tr/analysis/skull_external_intensity_ppw55.py``: integrate
the OUTWARD phase, extract the external calvarial surface, sample the field just
outside it, and report per-patch peak pressure / integrated intensity / peak
intensity, the distance-corrected transparency value, and the true surface normal."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .metrics import peak_intensity, distance_correct
from .registration import Registration
from .surface import extract_external_surface, true_normals

_LEGACY_KEYS = ("surf_vox", "rhat", "Iint", "Pmax", "Ipk_Wcm2", "rad_mm")


@dataclass
class TransparencyOptions:
    bone_threshold: float = 2200.0
    surface_probe_vox: float = 2.5      # K: outward/inward classify probe distance
    standoff_pad_vox: float = 4.0       # PAD: tissue standoff where the field is sampled
    metric: str = "peak_intensity"      # {"peak_intensity", "time_integrated"}
    distance_correct: bool = True       # multiply the value by (r/median_r)^2
    rho0: float = 1000.0
    c0: float = 1540.0
    normal_smooth: float = 1.0          # gaussian sigma for the gradient normal


@dataclass
class TransparencyMap:
    surf_vox: np.ndarray       # (M,3) f8  external-surface points, full-res voxel
    rhat: np.ndarray           # (M,3) f8  radial unit vector (target -> patch)
    true_normal: np.ndarray    # (M,3) f8  true outward surface normal (gradient)
    Iint: np.ndarray           # (M,)  f8  time-integrated intensity at the patch (Pa^2 . samples)
    Pmax: np.ndarray           # (M,)  f8  peak |p| at the patch (Pa)
    Ipk_Wcm2: np.ndarray       # (M,)  f8  peak intensity p^2/2rc (W/cm^2)
    rad_mm: np.ndarray         # (M,)  f8  distance from target (mm)
    value: np.ndarray          # (M,)  f8  transparency metric (distance-corrected if requested)
    registration: Registration | None = None
    meta: dict = field(default_factory=dict)

    def surf_mni_mm(self) -> np.ndarray:
        if self.registration is None:
            raise ValueError("no registration attached")
        return self.registration.fullres_to_mni(self.surf_vox)

    def to_npz(self, path) -> None:
        np.savez(path, surf_vox=self.surf_vox, rhat=self.rhat, true_normal=self.true_normal,
                 Iint=self.Iint, Pmax=self.Pmax, Ipk_Wcm2=self.Ipk_Wcm2, rad_mm=self.rad_mm,
                 value=self.value)

    @classmethod
    def from_npz(cls, path, registration: Registration | None = None) -> "TransparencyMap":
        d = np.load(path)
        return cls(
            surf_vox=d["surf_vox"], rhat=d["rhat"],
            true_normal=d["true_normal"] if "true_normal" in d.files else np.zeros_like(d["rhat"]),
            Iint=d["Iint"], Pmax=d["Pmax"], Ipk_Wcm2=d["Ipk_Wcm2"], rad_mm=d["rad_mm"],
            value=d["value"] if "value" in d.files else d["Ipk_Wcm2"],
            registration=registration,
        )


def compute_transparency_map(source, options: TransparencyOptions = TransparencyOptions(),
                             log=None) -> TransparencyMap:
    """Pure-Python. ``source`` is a FieldBundle (or a path to one). Reads the outward
    field + speed map, extracts the external surface, samples the field at the standoff,
    and returns a :class:`TransparencyMap`."""
    from .bundle import FieldBundle, load_bundle
    bundle = source if isinstance(source, FieldBundle) else load_bundle(source)

    c = bundle.skull_c()                                   # (N,N,N) speed map
    dx_mm = bundle.grid["dx_m"] * 1e3
    target_fullres = np.asarray(bundle.target["fullres_voxel"], float)
    o = options

    sf = bundle.surface_field() if hasattr(bundle, "surface_field") else None
    if sf is not None:                                     # shell recorder: field recorded on the surface directly
        surf_vox, rhat = sf["surf_vox"], sf["rhat"]
        Ival, Pval = sf["Iint"], sf["Pmax"]
    else:                                                  # volume recorder: sample the decimated field at the standoff
        Iint_vol, Pmax_vol = bundle.outward_iint_pmax(log=log)  # (nf,nf,nf) each
        mod = bundle.grid["field_mod"]
        surf_vox, rhat = extract_external_surface(c, target_fullres, o.bone_threshold, o.surface_probe_vox)
        Pout = surf_vox + o.standoff_pad_vox * rhat
        Ival = map_coordinates(Iint_vol, (Pout / mod).T, order=1)
        Pval = map_coordinates(Pmax_vol, (Pout / mod).T, order=1)
    rad_mm = np.linalg.norm(surf_vox - target_fullres, axis=1) * dx_mm
    Ipk_Wcm2 = peak_intensity(Pval, o.rho0, o.c0) / 1e4
    nrm = true_normals(c, surf_vox, o.bone_threshold, o.normal_smooth)

    base = Ipk_Wcm2 if o.metric == "peak_intensity" else Ival
    value = distance_correct(base, rad_mm) if o.distance_correct else np.asarray(base, float)

    return TransparencyMap(
        surf_vox=surf_vox, rhat=rhat, true_normal=nrm,
        Iint=Ival, Pmax=Pval, Ipk_Wcm2=Ipk_Wcm2, rad_mm=rad_mm, value=value,
        registration=bundle.registration,
        meta={"metric": o.metric, "distance_correct": o.distance_correct,
              "target_mni_mm": list(bundle.target.get("mni_ras_mm", [])),
              "dx_mm": dx_mm, "units": "Ipk W/cm^2; Iint Pa^2.samples; Pmax Pa"},
    )
