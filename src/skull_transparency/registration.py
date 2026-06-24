"""Rigid registration between the posed full-resolution sim grid (voxel indices on
the N^3 grid of ``skull_fullres_c``) and MNI RAS millimetres.

The sim grid is genuinely isotropic (``dx_mm``) and sim<->MNI is a rigid motion, so
we represent it as a single orthonormal rotation + uniform scale, anchored at the
target (which was independently validated to be the focus == the MNI target).

This promotes the *clean* path from ``focal_overlay_intensity_ppw55.py`` and
deliberately drops the internally-inconsistent ``Amn/bmn/dds/scale`` affine stored
in ``ppw55_transform.npz`` (anisotropic, mni2sim(target) != dc by ~8 mm)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Registration:
    R_mni_to_sim: np.ndarray        # (3,3) orthonormal: world mm displacement -> sim mm displacement
    dx_mm: float                    # isotropic sim voxel pitch (mm)
    target_mni_mm: np.ndarray       # (3,) anchor in the world frame (MNI RAS mm for Halle)
    target_fullres_voxel: np.ndarray  # (3,) same anchor as a full-res sim voxel index
    world_frame: str = "mni_ras_mm"   # label of the world frame the *_mni fields actually live in.
    #   "mni_ras_mm" for the Halle study; a generic subject (sim.prepare) sets it to its own
    #   meta['input_frame'] so the placement output does NOT get mis-mapped through tuba's MNI chain.

    # ---- coordinate maps (accept (3,) or (N,3); return matching shape) ----
    def mni_to_fullres(self, pts_mni_mm) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_mni_mm, float))
        out = self.target_fullres_voxel + (1.0 / self.dx_mm) * (self.R_mni_to_sim @ (p - self.target_mni_mm).T).T
        return out.reshape(np.shape(pts_mni_mm))

    def fullres_to_mni(self, pts_fullres) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts_fullres, float))
        out = self.target_mni_mm + self.dx_mm * (self.R_mni_to_sim.T @ (p - self.target_fullres_voxel).T).T
        return out.reshape(np.shape(pts_fullres))

    # ---- (de)serialisation ----
    def to_dict(self, deprecated_affine: dict | None = None) -> dict:
        d = {
            "schema": "skull_transparency.registration/1",
            "frame_a": self.world_frame,
            "frame_b": "fullres_voxel",
            "rigid": True,
            "dx_mm": float(self.dx_mm),
            "R_mni_to_sim": self.R_mni_to_sim.tolist(),
            "target_mni_mm": np.asarray(self.target_mni_mm, float).tolist(),
            "target_fullres_voxel": np.asarray(self.target_fullres_voxel, float).tolist(),
            "source": ("tuba mni_ras_to_subject_ras + pose (Arot,M) with svd-cleaned Amn rotation"
                       if self.world_frame == "mni_ras_mm"
                       else f"pose from sim.prepare (world frame {self.world_frame!r})"),
        }
        if deprecated_affine is not None:
            d["deprecated_affine"] = deprecated_affine
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Registration":
        return cls(
            R_mni_to_sim=np.asarray(d["R_mni_to_sim"], float),
            dx_mm=float(d["dx_mm"]),
            target_mni_mm=np.asarray(d["target_mni_mm"], float),
            target_fullres_voxel=np.asarray(d["target_fullres_voxel"], float),
            world_frame=d.get("frame_a", "mni_ras_mm"),
        )

    def to_json(self, path, deprecated_affine: dict | None = None) -> None:
        Path(path).write_text(json.dumps(self.to_dict(deprecated_affine), indent=1))

    @classmethod
    def from_json(cls, path) -> "Registration":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_ppw55_npz(cls, npz_path, target_fullres_voxel=None) -> "Registration":
        """Build the clean rigid map from a legacy ``ppw55_transform.npz``.

        ``Rtot = (M^T @ Arot^T) @ polar(Amn)`` strips Amn's bad anisotropic scale via
        its polar rotation (U @ Vt of the SVD).  Anchor at the target: by default the
        rounded ``dent_grid`` used by the legacy overlay; pass an explicit
        ``target_fullres_voxel`` (e.g. meta['dent_grid']) to match a specific run.
        """
        t = np.load(npz_path)
        Arot, M, Amn = np.asarray(t["Arot"], float), np.asarray(t["M"], float), np.asarray(t["Amn"], float)
        U, _, Vt = np.linalg.svd(Amn)
        Rtot = (M.T @ Arot.T) @ (U @ Vt)
        dent_mni = np.asarray(t["dent_mni"], float)
        dx_mm = float(t["dXmm"])
        if target_fullres_voxel is None:
            target_fullres_voxel = np.asarray(t["dc"], float)
        return cls(Rtot, dx_mm, dent_mni, np.asarray(target_fullres_voxel, float))

    def deprecated_affine_from_npz(self, npz_path) -> dict:
        """Carry the legacy affine forward as provenance only (do not use it)."""
        t = np.load(npz_path)
        return {
            "Amn": np.asarray(t["Amn"], float).tolist(),
            "bmn": np.asarray(t["bmn"], float).tolist(),
            "dds": np.asarray(t["dds"], float).tolist(),
            "scale": float(t["scale"]),
            "note": "anisotropic, internally inconsistent; superseded by R_mni_to_sim",
        }
