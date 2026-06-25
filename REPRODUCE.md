# REPRODUCE — manuscript PDF and results

Pinned entry points for regenerating the paper and its numbers. The repo is now
git-tracked (main); treat the files listed under "Results-critical — do not modify" as
the frozen baseline; the new front-end work (see bottom) is additive and inert.

## 1. The manuscript

Canonical source is **`manuscript/manuscript_pmb.tex`** (PMB / IOP; uses the local
`manuscript/iopjournal.cls`). It is the freshest tex and the one that produced the
submitted `manuscript/manuscript_pmb.pdf`.

> `manuscript/manuscript.tex` and `manuscript/manuscript_v2.tex` (and the `*.bak*`
> files) are **historical / stale** — do not build from them. `manuscript.pdf` is an
> older render of the stale `manuscript.tex`.

Rebuild the PDF (no bibtex — bibliography is an inline `thebibliography`):

```bash
cd manuscript
pdflatex manuscript_pmb.tex     # run 2-3x to settle cross-references
pdflatex manuscript_pmb.tex
```

Requires the 76 figures in `manuscript/figs/` (committed; regenerate only if a number
changes — see §2/§3).

## 2. The pure-Python results pipeline (transparency map + placement)

Reproduces the per-target transparency maps, placement windows, and focus numbers
from the precomputed Halle Field Bundle. No GPU/MATLAB needed for this layer.

```bash
pip install -e .                              # or: export PYTHONPATH=src
python examples/halle_dentate/run_pipeline.py examples/halle_dentate/config.json
# -> writes surface_map.npz + placement.json into the bundle data_dir
```

`config.json` points at the external, read-only inputs via the
`${SKULL_TR_DATA_ROOT}` placeholder (default `/celerina/gfp/mfs/hemisphere_tr`).
To relocate, `export SKULL_TR_DATA_ROOT=/your/path` or edit `config.json`:

| input | path |
|---|---|
| Field Bundle data | `$SKULL_TR_DATA_ROOT/data/halle_hemis_ppw55` |
| grid/physics meta | `$SKULL_TR_DATA_ROOT/sim/meta.json` |
| sim↔MNI transform | `$SKULL_TR_DATA_ROOT/sim/ppw55_transform.npz` |

The finalized figures/numbers live under **`runs/rebuild_6ppw_graded/`** (the graded
6-ppw rebuild; `runs/rebuild_6ppw_20260616/` is the prior pass). The headline
focus-gain figures are stated in `manuscript_pmb.tex` — treat the manuscript as the
source of truth for the exact values.

## 3. The simulation layer (only if re-solving the wavefields)

The outward/inward time-reversal solver **inputs** are regenerated bit-for-bit by the
pure-Python launchers; the solve itself uses the external CUDA binary.

```bash
python -m skull_transparency.sim outward          --out /scratch/run   # --sim defaults to $SKULL_TR_DATA_ROOT/sim
python -m skull_transparency.sim inward_windowed  --out /scratch/run   # (or pass --sim / set $FULLWAVE2_SIM_DIR)
# add --run to invoke the solver (needs a GPU + the binary below)
```

External CUDA solver: the fullwave2-ultra `bench_3d_opt` (public repo, the sibling
`fullwave2-ultra/bin/bench_3d_opt` checkout, PolyForm Noncommercial). `bench_3d_opt` is
deterministic and reproduced the retired `fullwave2_3d_Aexp_genout_cuda_aperturegrowth_opt`
bit-for-bit on the dentmanual *targeting* run (prior work); a generic small-grid re-check (2026-06-22)
diverged 17-43% in PML-dominated regimes, so real-grid parity for the *transparency/volume* path is
UNCONFIRMED (needs a representative N>=~400 run). For an identical legacy re-run set
`FULLWAVE2_BIN` to the old binary.
Resolved via `fullwave2_ultra.solver.resolve_binary` (`pip install -e .[solver]`); override with
`FULLWAVE2_BIN=/path/to/bench_3d_opt`.

**Older clusters / glibc.** The prebuilt `bench_3d_opt` is built against **GLIBC ≥ 2.34**
(Ubuntu 22.04+). On an older host (e.g. Ubuntu 20.04 / glibc 2.31) it aborts with
`GLIBC_2.3x not found` (the Python launcher still writes all `.dat` inputs first; only the binary
load fails). Its only dynamic deps are `libc`/`libm` (CUDA is statically linked, the driver arrives
via `--nv`), so running it inside a stock Ubuntu-22.04 container fixes it — point `FULLWAVE2_BIN` at
a one-line wrapper:
```bash
apptainer pull ubuntu2204.sif docker://ubuntu:22.04          # (or singularity / docker)
printf '#!/bin/bash\nexec apptainer exec --nv %s/ubuntu2204.sif /path/to/bench_3d_opt "$@"\n' "$PWD" > bench_wrap.sh
chmod +x bench_wrap.sh && export FULLWAVE2_BIN=$PWD/bench_wrap.sh
```
Keep the run's `--out` on **node-local** storage (genout is tens of GB), not a network mount.

## 4. Verification (proves the baseline is intact)

```bash
PYTHONPATH=src python -m pytest tests/test_transparency_golden.py -q   # pure-Python golden vs legacy npz
PYTHONPATH=src python -m pytest tests/test_sim_launchers.py -q         # LIGHT: medium-recipe byte equivalences + committed d/dmap/icc.dat refs
# full multi-GB outward regeneration (opt-in; writes ~10 GB scratch):
FULLWAVE2_VERIFY=1 PYTHONPATH=src python -m pytest tests/test_sim_launchers.py::test_full_bit_identity -q
```

The default `test_sim_launchers.py` run is light (small committed reference files +
pure-numeric equivalences); the GB-writing regeneration is gated behind
`FULLWAVE2_VERIFY=1`. (The package is not pip-installed on every interpreter here;
`PYTHONPATH=src` runs against the source tree without an install.)

## Results-critical — keep the legacy path bit-identical

Compute layer: `src/skull_transparency/{transparency,placement,metrics,bundle,registration,surface,complex_field,projection,neuromod,transducer}.py`
Solver inputs: `src/skull_transparency/sim/{launch_core,launchers,_common,fwio,forcoef,mlcompat}.py`
Data/artifacts: `runs/`, `manuscript/`, and the external `hemisphere_tr` tree above.

## Generic-subject front-end (added; guarded, Halle-identical)

The ingestion front-end `src/skull_transparency/transducer_spec.py` +
`src/skull_transparency/sim/prepare.py` is **inert** for the legacy pipeline (nothing
above imports it). It needed small **guarded** edits to three results-path files:

* `sim/launchers.py` `launch_outward` — read `meta["c_file"]` (default `"halle_c.f32"`)
  and optional `meta["rho_file"]`/`meta["alpha_file"]`;
* `sim/launch_core.py` — `_porosity_aexp`/`launch_core` accept an optional supplied
  `alpha_map` (the `None` path is byte-for-byte the old c-porosity model);
* `bundle.py` `build_field_bundle` — accept a ready `registration_path` as an
  alternative to `transform_path`.

Each guard keys off a field **absent** from Halle's `meta.json` / call sites, so the
legacy run takes the identical code path. Verified: §4 light run passes (committed
`.dat` refs + `_porosity_aexp` `tobytes()` equality). Re-run §4 (and the opt-in full
regen) after any further change here before relying on a result.
