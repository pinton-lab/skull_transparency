# Tutorial — from a CT-derived skull map to a transducer placement

You have a **sound-speed map** from a head CT, a **target** in the brain, and a
**transducer**. This walks the full pipeline that turns those into a *transparency map*,
a *positioning score*, and a *placement.json*.

The idea (see the top-level `README.md`): run **one** outward time-reversal solve — a
point source at the target radiates through the skull — and record the field on the
external skull surface. By acoustic reciprocity the field reaching each surface patch is
what a transducer *there* would deliver to the target, so a single solve scores **every**
window at once.

For the exact input formats (map units, the affine, the `TransducerSpec` JSON), see
[`input_format.md`](input_format.md).

## The pipeline

```
prepare  ──►  (GPU solve)  ──►  extract  ──►  place
 .dat          genout_mod.dat     Field        surface_map.npz
 inputs        (CUDA)             Bundle       + score.json
                                               + placement.json
```

`prepare` needs **no GPU** — it only writes solver inputs. The solve is GPU-bound. `extract`
and `place` are pure Python. If you have no data or GPU, skip to
[Zero-data smoke test](#zero-data-smoke-test) first to see the back half run.

### 1. `prepare` — skull map → sim tree

```bash
skull-transparency prepare --c-map c.npy --affine A.npy --target -12,-57,-34 \
    --transducer ctx500.json --approach 0,0,1 --out run/
```

This resamples your medium onto a posed cubic grid, plants the outward point source at the
target, synthesises the recording surface for the device, and writes a self-contained sim
tree (`c.f32` [+ optional `rho.f32`/`alpha.f32`], `meta.json`, `array_coords.i32`,
`registration.json`).

Flags:

| flag | meaning |
|---|---|
| `--c-map` | sound-speed map, `.npy` or `.nii`/`.nii.gz` (m/s) — **required** |
| `--rho-map` | optional density map, `.npy`/`.nii` (kg/m³) |
| `--alpha-map` | optional attenuation map, `.npy`/`.nii` (dB/cm/MHz) — supplying it **auto-enables attenuation** |
| `--affine` | 4×4 voxel→world-mm `.npy` (else taken from a NIfTI c-map) |
| `--target` | target in world mm, `'x,y,z'` — **required** |
| `--transducer` | `TransducerSpec` JSON file or inline string — **required** |
| `--approach` | aim unit vector target→skin, `'x,y,z'` (required until `approach='auto'` lands) |
| `--input-frame` | provenance label for the world frame (default `ras_mm`) |
| `--standoff-mm` | acoustic-path headroom around the head when sizing the grid (default `20.0`) |
| `--surround-mm` | medium kept around the target when sizing the grid (default `90.0`) |
| `--out` | output sim-tree directory — **required** |

> `--target` and `--approach` are comma-or-space separated; the `--affine` and `--target`
> must be in the **same world frame** (see [`input_format.md`](input_format.md)).

### 2. The GPU solve

Run the CUDA solve on the sim tree:

```bash
python -m skull_transparency.sim outward --sim run --out run --run
```

`--run` invokes the solver (without it, the launcher only re-writes/verifies the `.dat`
inputs). The solver is the external **fullwave2-ultra `bench_3d_opt`** binary (a GPU CUDA
solver, public repo, PolyForm Noncommercial). It is resolved via
`fullwave2_ultra.solver.resolve_binary` (`pip install -e '.[solver]'`), or you can point at
it directly:

```bash
export FULLWAVE2_BIN=/path/to/bench_3d_opt
```

The solve streams a decimated full field to **`genout_mod.dat`** inside the run directory.
(`bench_3d_opt` is deterministic and matched the retired
`fullwave2_3d_Aexp_genout_cuda_aperturegrowth_opt` bit-for-bit on the dentmanual *targeting* run,
but representative-grid parity for the *transparency/volume* path is **not yet confirmed** — see the
Limitations note and `REPRODUCE.md` §3.)

### 3. `extract` — solved run → Field Bundle

```bash
skull-transparency extract --run run/outward --sim run --out run/bundle
```

This reads `genout_mod.dat`, crops the 48-voxel boundary pad, time-integrates the field to
the cached `outward_Iint`/`outward_Pmax` volumes, converts the posed `c.f32` to
`skull_fullres_c.npy`, and writes `bundle.json` + `registration.json`.

| flag | meaning |
|---|---|
| `--run` | solved outward run dir (holds `genout_mod.dat`) — **required** |
| `--sim` | producer sim tree (`meta.json` + `c.f32` + `registration.json`) — **required** |
| `--n-out` | outward frame count (default: all recorded) |
| `--out` | output Field Bundle directory — **required** |

### 4. `place` — Field Bundle → the three outputs

```bash
skull-transparency place --bundle run/bundle --out result/
```

| flag | meaning |
|---|---|
| `--bundle` | Field Bundle directory (post-solve) — **required** |
| `--transducer` | optional `TransducerSpec` JSON for the window constraints |
| `--focal-length` | bowl focal length mm (default: ROC, else 60) |
| `--target-name` | name carried into the outputs |
| `--verbose` | log the transparency computation |
| `--out` | output directory — **required** |

This writes the three outputs into `result/`:

* **`surface_map.npz`** — the transparency map (per-patch coupling on the external surface),
  the small (~26 MB) distributable bundle.
* **`score.json`** — the positioning score (see below).
* **`placement.json`** — the transducer placement (a `neuromod_parameters`-style dict; read
  its `frame` key — `nrrd_voxel_mm` with tuba present, else `mni_ras_mm`).

### Or: `run` (prepare + the chain to run next)

```bash
skull-transparency run --c-map c.npy --affine A.npy --target -12,-57,-34 \
    --transducer ctx500.json --approach 0,0,1 --out run/
```

`run` takes the same flags as `prepare`; it writes the sim tree and then **prints** the
solve/extract/place commands to run next (it does not invoke the GPU solve itself).

## Zero-data smoke test

No data, GPU, MATLAB, or tuba? Run the whole back half (transparency → place → score) on a
synthetic geometric fixture:

```bash
python examples/synthetic/run_synthetic.py            # writes ./synthetic_run/
```

or in code:

```python
import skull_transparency as st

bundle = st.load_bundle(st.make_synthetic_bundle("synthetic_run/bundle"))
tmap   = st.compute_transparency_map(bundle)
tmap.to_npz("synthetic_run/surface_map.npz")
pl     = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60.0, bowl_radius_mm=15.0,
                                                theta_max_deg=35.0))
score  = st.PositioningScore.from_placement(pl, target_name="synthetic_target")
score.to_json("synthetic_run/score.json")
```

The fixture is a spherical bone shell around the target plus a smooth synthetic outward
field (range-decaying, brighter on one side so placement is non-degenerate). It is **not
anatomy** — it only exercises the chain and the `surface_map.npz` / `score.json` /
`placement.json` outputs.

## Reading `score.json`

`score.json` is one self-describing artifact (schema
`skull_transparency.positioning_score/1`). The fields:

| field | meaning |
|---|---|
| `normalized` | 0..1 — chosen window coupling ÷ best achievable over all candidate windows (`1.0` = the optimal window) |
| `focal_pressure_proxy` | `sqrt(integral_S \|G\|² dS)` over the chosen aperture — the relative achievable focal peak under the optimal (time-reversal) drive (arbitrary units) |
| `incidence_deg` | window incidence angle in degrees (`0` = normal incidence) |
| `objective` | the per-patch coupling proxy used: `"peak"`, `"energy"`, or `"cap"` |
| `target_name` | the name you passed |
| `definition` | a verbatim prose statement of all of the above, baked into the file |

The `place` CLI also prints a one-line summary, e.g.:

```
score 0.873 (focal-pressure proxy 1.2e+05), incidence 12.4 deg
```

## Limitations — how to trust your result

These come straight from the code (`score.py`, `extract.py`, `sim/prepare.py`) and the
README's "Notes on correctness". Read them before relying on a number.

* **The single-frequency phasor score is a PLACEMENT / energy proxy, not a focusing-gain
  number.** `focal_pressure_proxy` (= `sqrt(integral_S |G|² dS)`) is the right quantity for
  *where to put the transducer* (delivered energy), but it **overstates the broadband
  time-reversal focusing gain and the axial depth of field**. Quote the focusing gain from a
  **broadband / inward re-simulation**, not from `score.json`. (`score.py` `_DEFINITION`
  bakes this caveat into the file itself.)

* **Intracranial air is mapped to water.** The medium ingestion maps sinus / mastoid air
  toward water, so **frontal and other air-crossing windows are untrustworthy**. Prefer
  windows over solid bone; treat any window whose path crosses an air cavity with suspicion.

* **Foramen-leak windows are excluded by a no-bone-ray test.** A candidate window is rejected
  if the ray target→element crosses **zero bone** (a true foramen leak), rather than by a
  fixed bone-thickness threshold (which would over-aggressively cut thin real bone and
  sutures). Keep this in mind if a high-scoring window looks anatomically implausible.

* **`genout_mod` volume parity is not yet real-solve-verified.** `extract.py` notes that the
  `genout_mod` *volume* layout has **not** been parity-checked against the retired
  `aperturegrowth_opt` solver on a real solve (the existing byte-parity covered the
  coordinate/array recording). The format logic is validated by a synthetic `genout_mod`
  round-trip, but **confirm on a real run before trusting absolute numbers.**

> Rule of thumb (from the README): use the **direct-arrival** energy for placement
> (`Ipk_Wcm2` / the ballistic-windowed `E_win`), not the full-record time-integral `Iint`,
> which is reverberation-contaminated and mis-selects a window.
