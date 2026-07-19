# REPRODUCE — results

Pinned entry points for regenerating the transparency maps, placement windows, and focus
numbers. Treat the files listed under "Results-critical — do not modify" as the frozen
baseline; the new front-end work (see bottom) is additive and inert.

## 1. The manuscript

The manuscript is maintained **outside this repository** and is **not included in the
public tree** (`manuscript/` is gitignored). The headline focus-gain numbers it reports
are reproduced by the pure-Python pipeline in §2 — treat those regenerated outputs as the
source of truth here.

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

The precomputed solver outputs live under **`runs/`** (e.g. `runs/rebuild_6ppw_graded/`,
the graded 6-ppw rebuild), which is **not committed** (~275 GB; gitignored). The pipeline
above regenerates the maps, placement, and focus numbers from the Halle Field Bundle —
that is the reproducible path in this repo.

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
Data/artifacts (NOT committed to this public repo): `runs/`, `manuscript/`, and the
external `hemisphere_tr` tree above.

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

## Brain-center whole-skull transparency baseline (added)

A neutral whole-skull transparency variant: one omnidirectional source at the **brain
center** → `1/r²`-corrected map of where the skull transmits. Code is additive —
`src/skull_transparency/brain_center.py` (atlas/cavity/image-only centers),
`sim/prepare.build_brain_center_run` + `_choose_pose_centered`, `render.py`, the
`prepare --center [--center-mm] [--bone-threshold]` / `extract --bone-threshold` /
`transparency [--bone-threshold]` CLI subcommands (`tests/test_brain_center.py`; tutorial §5).

**Generic (portable, in-repo) path — any subject, no lab data:**

```bash
skull-transparency prepare --c-map c.nii.gz --center --transducer ctx500.json --out run/
python -m skull_transparency.sim outward --sim run --out run --run        # GPU solve
skull-transparency extract     --run run/outward --sim run --out run/bundle
skull-transparency transparency --bundle run/bundle --out transparency.png
```

**Exact Halle reference figure** — uses the atlas brain CoM (MNI152 brain-mask centroid
`(0,-22,9.5)` → crop voxel `[191,304,372]`) on the same `[409,539,529]` 0.28 mm whole-skull
grid as `data/halle_skullonly`, via the lab build script (sibling of `bridge_skullonly.py`,
under `$SKULL_TR_DATA_ROOT/analysis/`, default `/celerina/gfp/mfs/hemisphere_tr/analysis`):

```bash
# solve (needs a free GPU) + bridge to $SKULL_TR_DATA_ROOT/data/halle_braincenter:
SCRATCH=/tmp/braincenter_run GPU=0 python "$SKULL_TR_DATA_ROOT/analysis/build_braincenter.py" solve   # ~4 min, ~35 GB scratch
SCRATCH=/tmp/braincenter_run            python "$SKULL_TR_DATA_ROOT/analysis/build_braincenter.py" bridge
# render the tutorial figure from the bundle (regenerates tutorial/figs/fig_braincenter.png):
cd tutorial && python -c "import figs; figs.fig_braincenter()"
```

> Note (corrects the deprecated `launch_skullonly` "center"): its `CEN=[360,360,360]`
> buffer-center seated the source at the **dentate**, 57 mm off the brain center. The
> baseline above uses the atlas CoM. The whole-skull figure plots transmitted **amplitude
> in dB** (`log |p|`), not linear intensity.
>
> For a **non-human** subject the same generic `prepare --center` path applies; set
> `--bone-threshold` to that skull's bone sound speed and, if the image-only centroid is
> off, `--center-mm x,y,z`.
