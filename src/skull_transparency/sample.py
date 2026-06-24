"""A tiny SYNTHETIC Field Bundle so the transparency -> placement -> score chain (and
the ``skull-transparency place`` CLI) runs with **zero** external data, GPU, or tuba.

It is a geometric fixture, NOT anatomy: a spherical bone shell around the target and a
smooth synthetic outward field (peak pressure decaying with range, brighter on one
side so placement is non-degenerate). It writes exactly what
:func:`~skull_transparency.transparency.compute_transparency_map` consumes — the
cached ``outward_Iint``/``outward_Pmax`` volumes (so no ``propagation_map`` /
``array_traces`` are needed) plus a clean ``registration.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .registration import Registration

SCHEMA = "skull_transparency.field_bundle/1"


def make_synthetic_bundle(out_dir, *, N: int = 48, mod: int = 2, dx_m: float = 2.5e-3,
                          target_mni_mm=(-12.0, -57.0, -34.0)) -> Path:
    """Write a minimal valid Field Bundle into ``out_dir`` and return it. ``N`` = cubic
    grid size, ``mod`` = volume-recorder decimation (field volumes are ``(N/mod)^3``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nf = N // mod
    target = np.array([N / 2.0, N / 2.0, N * 0.42])           # a bit below centre

    # --- bone shell around the target (gives extract_external_surface an outer surface) ---
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((ii - target[0]) ** 2 + (jj - target[1]) ** 2 + (kk - target[2]) ** 2)
    c = np.where((r >= N * 0.28) & (r <= N * 0.36), 2900.0, 1540.0).astype(np.float32)
    np.save(out / "skull_fullres_c.npy", c)

    # --- synthetic outward field volumes (nf^3): range decay + a brighter +Z side ---
    fi, fj, fk = np.meshgrid(np.arange(nf), np.arange(nf), np.arange(nf), indexing="ij")
    tf = target / mod
    rf = np.sqrt((fi - tf[0]) ** 2 + (fj - tf[1]) ** 2 + (fk - tf[2]) ** 2) + 1.0
    ang = np.clip(1.0 + 0.6 * (fk - tf[2]) / nf, 0.2, None)   # window preference along +Z
    Pmax = (1.0e5 / rf * ang).astype(np.float32)
    Iint = (Pmax.astype(np.float64) ** 2).astype(np.float32)  # arb. Pa^2.samples
    np.save(out / "outward_Pmax.npy", Pmax)
    np.save(out / "outward_Iint.npy", Iint)

    # --- clean rigid registration (identity rotation, anchored at the target) ---
    Registration(R_mni_to_sim=np.eye(3), dx_mm=dx_m * 1e3,
                 target_mni_mm=np.asarray(target_mni_mm, float),
                 target_fullres_voxel=target).to_json(out / "registration.json")

    spec = {
        "schema": SCHEMA,
        "subject_id": "synthetic",
        "grid": {"N": N, "dx_m": dx_m, "order": "C", "field_mod": mod, "n_field": nf},
        "physics": {"c0": 1540.0, "f0": 1e6, "rho0": 1000.0, "c_bone": 2900.0,
                    "bone_threshold": 2200.0, "ppw": 5.5},
        "target": {"name": "synthetic_target", "mni_ras_mm": list(map(float, target_mni_mm)),
                   "fullres_voxel": target.tolist(), "field_voxel": (target / mod).tolist()},
        "array": {"n_elements": 0, "center_fullres_voxel": [N / 2.0] * 3,
                  "radius_vox": N * 0.33, "coords_file": "array_coords.npy", "geometry": "synthetic"},
        "phases": {"n_out": 1, "n_total": 1, "n_total_file": 1},
        "files": {"skull_fullres_c": "skull_fullres_c.npy", "outward_Iint": "outward_Iint.npy",
                  "outward_Pmax": "outward_Pmax.npy", "propagation_map": "propagation_map.npy",
                  "array_traces": "array_traces.npz"},
        "registration": "registration.json",
        "provenance": {"synthetic": True, "note": "geometric fixture, not anatomy"},
    }
    (out / "bundle.json").write_text(json.dumps(spec, indent=1))
    return out
