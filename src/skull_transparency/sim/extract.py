"""Extract a Field Bundle from a SOLVED outward run — the glue between
``prepare``+solve and ``place``.

The fullwave2-ultra solver streams the ``(modX,modY,modZ)``-decimated FULL (extended)
field to ``genout_mod.dat`` as ``(nframes, nXe2, nYe2, nZe2)`` (see the package's
``docs/io_contract.md`` / ``io_dat.read_genout_mod``). We:

  1. read it and crop the 48-voxel PML/boundary pad to the interior ``N`` grid, giving
     the ``(n_total, nf, nf, nf)`` propagation field;
  2. time-integrate it (:func:`skull_transparency.metrics.integrate_outward`) to the
     cached ``outward_Iint``/``outward_Pmax`` ``(nf,nf,nf)`` volumes that
     :func:`~skull_transparency.transparency.compute_transparency_map` reads;
  3. convert the posed sound-speed map ``c.f32`` -> ``skull_fullres_c.npy``;
  4. write ``bundle.json`` + copy ``registration.json``.

No giant ``propagation_map.npy`` is written (the integral is taken streaming).

CAVEAT: the ``genout_mod`` *volume* layout has not yet been parity-checked against the
retired aperturegrowth_opt on a real solve (the existing byte-parity covered the
coord/array recording). Confirm on a real run before trusting absolute numbers. The
format logic here is validated by a synthetic ``genout_mod`` round-trip (test_extract.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import _common as C
from .launch_core import PAD              # 48 = nbdy(40) + M(8) per side
from ..metrics import integrate_outward

MOD = 2   # launch_outward's fixed modX = modY = modZ (the volume-recorder decimation)
SCHEMA = "skull_transparency.field_bundle/1"


def _read_genout_mod(path, nXe, mod):
    """``(nframes, nXe2, nYe2, nZe2)`` float32 from a (cubic) genout_mod dump. Uses the
    fullwave2_ultra reader when importable, else the identical local reshape."""
    try:
        from fullwave2_ultra.io_dat import read_genout_mod
        return read_genout_mod(str(path), nXe, nXe, nXe, mod, mod, mod)
    except Exception:
        n2 = (nXe + mod - 1) // mod
        return np.fromfile(str(path), dtype="<f4").reshape(-1, n2, n2, n2)


def extract_bundle(run_dir, out_dir, sim_dir, *, n_out=None, mod: int = MOD,
                   bone_threshold: float = 2200.0, c_bone: float = 2900.0) -> Path:
    """Build a Field Bundle in ``out_dir`` from the solved outward ``run_dir``
    (which holds ``genout_mod.dat``) and the producer ``sim_dir`` (``meta.json`` +
    ``c.f32`` + ``registration.json``). ``n_out`` defaults to all recorded frames.

    ``bone_threshold`` / ``c_bone`` (m/s) are recorded in ``bundle.json`` ``physics`` so
    the calvarial-surface cutoff travels with the bundle: the human default is 2200/2900,
    but a medium whose bone is slower (thin bone well below 2200 m/s) must pass its own
    value so ``transparency`` finds the right surface. The producer's own ``bone_threshold``
    is the natural value to pass here."""
    run_dir, out_dir, sim_dir = Path(run_dir), Path(out_dir), Path(sim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = C.load_meta(sim_dir)
    N = int(meta["N"])
    nf = len(range(0, N, mod))                         # == volume_recorders' per-axis count
    nXe = N + 2 * PAD

    gm = run_dir / "genout_mod.dat"
    if not gm.exists():
        raise FileNotFoundError(
            f"{gm} not found — extract needs the fullwave2-ultra solver's decimated full-field "
            "dump (run the solve with run_solver=True on a modX/Y/Z run).")
    vol = _read_genout_mod(gm, nXe, mod)               # (nframes, nXe2, nXe2, nXe2)
    lo = PAD // mod
    propmap = vol[:, lo:lo + nf, lo:lo + nf, lo:lo + nf]   # crop pad -> interior (nframes,nf,nf,nf)
    if propmap.shape[1:] != (nf, nf, nf):
        raise ValueError(f"genout_mod interior crop is {propmap.shape[1:]}, expected {(nf, nf, nf)} "
                         f"(grid N={N}, mod={mod}); check the solver's mod settings / grid size.")
    nframes = propmap.shape[0]
    n_out = int(n_out) if n_out is not None else nframes

    Iint, Pmax = integrate_outward(propmap, n_out)
    np.save(out_dir / "outward_Iint.npy", Iint.astype(np.float32))
    np.save(out_dir / "outward_Pmax.npy", Pmax.astype(np.float32))

    # posed sound speed: c.f32 was written F-order by the producer (build_run_from_medium)
    c_file = meta.get("c_file", "halle_c.f32")
    c = np.fromfile(sim_dir / c_file, dtype="<f4").reshape(N, N, N, order="F")
    np.save(out_dir / "skull_fullres_c.npy", c)

    target_fullres = np.asarray(meta["dent_grid"], float)
    reg_src = sim_dir / "registration.json"
    target_mni = None
    if reg_src.exists():
        from ..registration import Registration
        reg = Registration.from_json(reg_src)
        reg.to_json(out_dir / "registration.json")
        target_mni = list(map(float, reg.target_mni_mm))

    (out_dir / "phase_info.json").write_text(json.dumps({"n_out": n_out, "n_total": nframes}))
    spec = {
        "schema": SCHEMA,
        "subject_id": meta.get("subject_id", sim_dir.name),
        "grid": {"N": N, "dx_m": float(meta["dX_m"]), "order": "C",
                 "field_mod": mod, "n_field": nf},
        "physics": {"c0": float(meta.get("C0", 1540.0)), "f0": float(meta.get("F0", 1e6)),
                    "rho0": 1000.0, "c_bone": float(c_bone), "bone_threshold": float(bone_threshold),
                    "ppw": meta.get("ppw")},
        "target": {"name": meta.get("subject_id", "target") + "_target",
                   "mni_ras_mm": target_mni, "fullres_voxel": target_fullres.tolist(),
                   "field_voxel": (target_fullres / mod).tolist()},
        "array": {"n_elements": int(meta.get("n_array", 0)),
                  "center_fullres_voxel": meta.get("array_center"),
                  "coords_file": "array_coords.i32", "geometry": meta.get("transducer", {})},
        "phases": {"n_out": n_out, "n_total": nframes, "n_total_file": nframes},
        "files": {"skull_fullres_c": "skull_fullres_c.npy", "outward_Iint": "outward_Iint.npy",
                  "outward_Pmax": "outward_Pmax.npy", "propagation_map": "propagation_map.npy",
                  "array_traces": "array_traces.npz"},
        "registration": "registration.json",
        "provenance": {"extracted_from": str(run_dir), "sim_dir": str(sim_dir),
                       "genout_mod": True},
    }
    (out_dir / "bundle.json").write_text(json.dumps(spec, indent=1))
    return out_dir
