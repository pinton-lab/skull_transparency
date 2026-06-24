"""Field Bundle: the self-describing on-disk interchange artifact between the heavy
TR-simulation producer and the pure-Python transparency/placement consumer.

A bundle is a directory with ``bundle.json`` (grid/physics/target/array/phases/files)
+ ``registration.json`` + the field arrays.  :func:`build_field_bundle` synthesises
these from a legacy ``hemisphere_tr`` data dir (+ its ``sim/meta.json`` and
``ppw55_transform.npz``) so existing runs become bundles without re-simulating."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .registration import Registration

SCHEMA = "skull_transparency.field_bundle/1"


@dataclass
class FieldBundle:
    dir: Path
    grid: dict
    physics: dict
    target: dict
    array: dict
    phases: dict
    files: dict
    registration: Registration | None = None

    # ---- field accessors (memmapped) ----
    def _p(self, key):
        return self.dir / self.files[key]

    def propagation_map(self):
        return np.load(self._p("propagation_map"), mmap_mode="r")

    def skull_c(self):
        return np.asarray(np.load(self._p("skull_fullres_c"), mmap_mode="r"))

    def outward_iint_pmax(self, log=None):
        """Cached (outward_Iint/Pmax.npy) if present, else integrate the outward phase."""
        fI = self.dir / self.files.get("outward_Iint", "outward_Iint.npy")
        fP = self.dir / self.files.get("outward_Pmax", "outward_Pmax.npy")
        if fI.exists() and fP.exists():
            return np.load(fI).astype(np.float64), np.load(fP).astype(np.float64)
        from .metrics import integrate_outward
        Iint, Pmax = integrate_outward(self.propagation_map(), self.phases["n_out"], log=log)
        np.save(fI, Iint.astype(np.float32))
        np.save(fP, Pmax.astype(np.float32))
        return Iint, Pmax

    # ---- complex (phase) path: from the array-element TIME traces only ----
    def array_traces(self):
        """Load the recorded array-element time traces (the clean source of phase). Returns
        (R (nT,M), positions_fullres (M,3), target_fullres (3,), dt_s, c0). Produced by the heavy
        layer (first ``n_array`` channels of the outward genout) into ``array_traces.npz``."""
        f = self.dir / self.files.get("array_traces", "array_traces.npz")
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found — the complex/phase path needs the array traces "
                "(first n_array channels of sim/outward/genout.dat) saved as array_traces.npz "
                "with keys R, positions(or arr), target(or dent), dt(or dt2), c0.")
        d = np.load(f)
        R = d["R"]
        pos = d["positions"] if "positions" in d.files else d["arr"]
        tgt = d["target"] if "target" in d.files else d["dent"]
        dt = float(d["dt"]) if "dt" in d.files else float(d["dt2"])
        c0 = float(d["c0"]) if "c0" in d.files else float(self.physics.get("c0", 1540.0))
        return np.asarray(R, float), np.asarray(pos, float), np.asarray(tgt, float), dt, c0

    def element_complex_field(self, f0: float | None = None):
        """Build the :class:`~skull_transparency.complex_field.ComplexField` (phase from the
        array traces) for phase-only drives, radial projection and focal-spot prediction."""
        from .complex_field import element_field
        R, pos, tgt, dt, c0 = self.array_traces()
        f0 = float(self.physics.get("f0", 1e6)) if f0 is None else f0
        dx_mm = self.grid["dx_m"] * 1e3
        return element_field(R, pos, tgt, f0, dt, dx_mm, c0)


def load_bundle(path) -> FieldBundle:
    d = Path(path)
    spec = json.loads((d / "bundle.json").read_text())
    reg = None
    regf = d / spec.get("registration", "registration.json")
    if regf.exists():
        reg = Registration.from_json(regf)
    return FieldBundle(dir=d, grid=spec["grid"], physics=spec["physics"], target=spec["target"],
                       array=spec["array"], phases=spec["phases"], files=spec["files"], registration=reg)


def build_field_bundle(data_dir, meta_path, transform_path=None,
                       target_name=None, registration_path=None) -> FieldBundle:
    """Write ``bundle.json`` + ``registration.json`` into ``data_dir`` from a run:
    ``meta_path`` = sim/meta.json. Provide EITHER ``registration_path`` (a clean
    ``registration.json``, e.g. emitted by ``sim.prepare`` for a generic subject) OR
    ``transform_path`` (a legacy ``ppw55_transform.npz``, repaired via
    :meth:`Registration.from_ppw55_npz`). Derives phases from the on-disk
    phase_info.json and field shape (not literals)."""
    data_dir = Path(data_dir)
    meta = json.loads(Path(meta_path).read_text())
    N = int(meta["N"])
    phase_info = json.loads((data_dir / "phase_info.json").read_text())
    pm = np.load(data_dir / "propagation_map.npy", mmap_mode="r")
    n_total_file, nf = pm.shape[0], pm.shape[1]
    if N <= 0:
        raise ValueError(f"invalid grid size N={N} in {meta_path}")
    if nf <= 0:
        raise ValueError(f"invalid propagation_map shape {pm.shape} in {data_dir}")
    field_mod = int(round(N / nf))
    n_total = int(phase_info.get("n_total", n_total_file))
    n_out = int(phase_info["n_out"])
    # frame_stride: outward genout frames per saved frame (best-effort, for provenance)
    dent_full = np.asarray(meta["dent_grid"], float)

    reg = None
    if registration_path is not None and Path(registration_path).exists():
        reg = Registration.from_json(registration_path)          # clean rigid map (generic subject)
    elif transform_path is not None and Path(transform_path).exists():
        reg = Registration.from_ppw55_npz(transform_path, target_fullres_voxel=dent_full)

    spec = {
        "schema": SCHEMA,
        "subject_id": meta.get("subject_id", "halle"),
        "grid": {"N": N, "dx_m": float(meta["dX_m"]), "order": "C",
                 "field_mod": field_mod, "n_field": int(nf)},
        "physics": {"c0": float(meta.get("C0", 1540.0)), "f0": float(meta.get("F0", 1e6)),
                    "rho0": 1000.0, "c_bone": 2900.0, "bone_threshold": 2200.0,
                    "ppw": meta.get("ppw")},
        "target": {"name": target_name or meta.get("target_name", "target"),
                   "mni_ras_mm": list(map(float, reg.target_mni_mm)) if reg is not None else None,
                   "fullres_voxel": dent_full.tolist(),
                   "field_voxel": (dent_full / field_mod).tolist()},
        "array": {"n_elements": int(meta.get("n_array", 0)),
                  "center_fullres_voxel": meta.get("array_center"),
                  "radius_vox": meta.get("arr_R_vox"),
                  "coords_file": "array_coords.npy",
                  "geometry": meta.get("array", "")},
        "phases": {"n_out": n_out, "n_total": n_total, "n_total_file": int(n_total_file)},
        "files": {"propagation_map": "propagation_map.npy",
                  "propagation_map_lo": "propagation_map_lo.npy",
                  "skull_fullres_c": "skull_fullres_c.npy",
                  "outward_Iint": "outward_Iint.npy",
                  "outward_Pmax": "outward_Pmax.npy",
                  "array_traces": "array_traces.npz"},
        "registration": "registration.json",
        "provenance": {"built_from": str(data_dir), "meta": str(meta_path),
                       "transform": str(transform_path) if transform_path else None},
    }
    (data_dir / "bundle.json").write_text(json.dumps(spec, indent=1))
    if reg is not None and registration_path is not None:
        reg.to_json(data_dir / "registration.json")              # already clean; no legacy affine
    elif reg is not None:
        dep = reg.deprecated_affine_from_npz(transform_path)
        reg.to_json(data_dir / "registration.json", deprecated_affine=dep)
    return load_bundle(data_dir)
