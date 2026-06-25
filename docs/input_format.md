# Input format & units

What `skull-transparency prepare` (and `build_run_from_medium`) expect: a medium, an affine,
a target, an approach vector, and a `TransducerSpec`. For the end-to-end walkthrough see
[`tutorial.md`](tutorial.md).

> **Brain-center variant.** `prepare --center` (`build_brain_center_run`) needs only the
> medium, affine, and `TransducerSpec` — **no target and no approach**: it seats one
> omnidirectional source at the brain center and sizes a cube around the whole head, for a
> neutral whole-skull transparency map. The target/approach sections below then don't apply.
> The center is the atlas CoM (MNI frame), else the image-only intracranial centroid, else
> an explicit `--center-mm x,y,z` (e.g. a curated cavity centroid). For a medium whose bone
> is slower than the human `2200` m/s default, set `--bone-threshold` (a value above
> water/soft tissue but below that medium's bone) so the calvarial surface, recording
> shell, and figure all use the right cutoff.

## The medium maps

| map | flag | units | required | accepted formats |
|---|---|---|---|---|
| sound speed `c` | `--c-map` | m/s | **yes** | `.npy` or NIfTI (`.nii` / `.nii.gz`) |
| density `rho` | `--rho-map` | kg/m³ | no | `.npy` or NIfTI |
| attenuation `alpha` | `--alpha-map` | dB/cm/MHz | no | `.npy` or NIfTI |

All maps are 3-D arrays in voxel order `(nx, ny, nz)` and must share the same grid (the same
affine).

* **Sound speed `c`** is the only required map. Outside the head it is treated as water
  (`c0`, from the transducer spec); voxels outside the resampled volume are filled with `c0`.
* **Density `rho`** is optional. If absent it is derived from `c` (`rho_from_c`); outside the
  head it is water (1000 kg/m³).
* **Attenuation `alpha`** is optional and in **dB/cm/MHz** (`alpha_units` defaults to
  `'db_mhz_cm'`). **Supplying `--alpha-map` auto-enables attenuation** and overrides the
  c-derived porosity model; outside the head it is 0. If you omit it, attenuation comes from
  the c-porosity model (the legacy behaviour).

> NIfTI inputs carry their own affine (the `sform`); see below. A `.npy` map has no affine,
> so you must pass `--affine`.

## The affine

A **4×4 matrix mapping voxel index → world-mm**:

```
[ x_world ]       [ i ]
[ y_world ] = A · [ j ]      A is 4x4, world coordinates in mm
[ z_world ]       [ k ]
[   1     ]       [ 1 ]
```

Supply it either as:

* a **NIfTI** c-map — its `sform` affine is used automatically (no `--affine` needed), or
* a **`.npy`** file via `--affine A.npy`.

The pipeline is **frame-agnostic**: the affine may be **RAS** (a NIfTI `sform`) or **LPS**
(e.g. built from NRRD `space directions` / `origin`). The physics does not care which; the
frame is carried through as a provenance label via `--input-frame` (default `ras_mm`), echoed
into `meta.json` so the result round-trips back to the coordinates you supplied. Anatomical
orientation is read off the sign + permutation of `affine[:3,:3]` — no tuba / MNI needed on
the input path.

> If you pass a `.npy` c-map with **no** `--affine`, `prepare` errors:
> *"no affine: pass --affine A.npy, or give a NIfTI c-map (which carries one)."*

## The target

`--target x,y,z` — the brain target in **world mm**, in the **same frame as the affine**.
Comma- or space-separated. (In code, `build_run_from_medium` also accepts a `target_voxel`
convenience, in which case `target_phys_mm = affine @ [*target_voxel, 1]`.) Omitted for a
brain-center run (`--center` computes the center for you).

## The approach vector

`--approach x,y,z` — a **unit vector from the target out to the skin / transducer**, in the
**world frame** (same frame as the affine). Comma- or space-separated.

It is rotated onto the grid `+Z` axis so the transducer sits at high Z and the target is
seated in from the low-Z face; the grid is sized so the focused-bowl reach
(`roc + aperture/2 + standoff`) fits along `+Z` and `surround_mm` of medium surrounds the
target elsewhere.

**Required** until `approach='auto'` (the outward skull-normal heuristic) lands — passing it
as `auto`, or omitting it, currently raises `NotImplementedError`. (A `--center` run needs
**no** approach: its source radiates in every direction, so it sidesteps this entirely.)

## The `TransducerSpec` JSON

One object is the source of truth for the grid pitch, the recording geometry, and the
placement/window constraints. Pass it to `--transducer` as a **JSON file path or an inline
JSON string**.

### Fields

| field | type | default | meaning |
|---|---|---|---|
| `f0_hz` | float | — (**required**) | drive frequency (Hz) → `meta['F0']` |
| `geometry` | string | `"bowl"` | `"bowl"`, `"annular"`, `"array"`, or `"hemisphere_probe"` |
| `roc_mm` | float | `null` | radius of curvature (mm) — required for `bowl`/`annular` |
| `aperture_mm` | float | `null` | full aperture diameter (mm) — required for `bowl`/`annular` |
| `n_rings` | int | `0` | annular only (CTX-500 → `4`) |
| `element_positions_mm` | (M,3) array | `null` | element coords in the device frame (`array` geometry) |
| `n_elements` | int | `0` | discrete-array element count |
| `c0_ms` | float | `1540.0` | reference sound speed (m/s) → `meta['C0']` |
| `ppw` | float | `6.0` | points per wavelength → sets `dx` |
| `acceptance_angle_deg` | float | `35.0` | incidence cap for window scoring |

Validation (`TransducerSpec.__post_init__`):

* `f0_hz`, `ppw`, `c0_ms` must all be `> 0`;
* `geometry` in `{"bowl", "annular"}` requires both `roc_mm` and `aperture_mm`;
* `aperture_mm` cannot exceed `2 · roc_mm` (the cap would exceed a hemisphere).

### Grid pitch

The spec sets the simulation grid pitch:

```
dx = c0 / (f0 · ppw)
```

e.g. 1 MHz / 5.5 ppw / 1540 m/s → 0.28 mm (the Halle `ppw55` grid). This becomes
`meta['dX_m']`. The acoustic wavelength is `c0 / f0`.

### The `preset` form

A spec may instead name a preset:

```json
{ "preset": "ctx500", "f0_hz": 500000, "ppw": 6.0, "acceptance_angle_deg": 35.0 }
```

`{"preset": "ctx500", ...}` dispatches to `TransducerSpec.ctx500(**rest)` — the
CTX-500 / NeuroFUS 4-annular bowl (ROC 63.2 mm, aperture 64 mm, `n_rings=4`). The remaining
keys (`f0_hz`, `ppw`, `acceptance_angle_deg`) are optional overrides; `ctx500` defaults to
`f0_hz=500000`, `ppw=6.0`, `acceptance_angle_deg=35.0`. Any other `preset` value errors.
Without a `preset` key the whole dict is passed straight to `TransducerSpec(**dict)`.

### Worked example — `ctx500.json`

A plain (non-preset) spec for the CTX-500 at 500 kHz:

```json
{
  "f0_hz": 500000,
  "geometry": "annular",
  "roc_mm": 63.2,
  "aperture_mm": 64.0,
  "n_rings": 4,
  "c0_ms": 1540.0,
  "ppw": 6.0,
  "acceptance_angle_deg": 35.0
}
```

The equivalent preset form (recommended — it pulls `roc_mm`/`aperture_mm` from the device
definition so you can't mistype them):

```json
{ "preset": "ctx500", "f0_hz": 500000, "ppw": 6.0 }
```

Either is a valid `--transducer ctx500.json`. With the above, `dx = 1540 / (500000 · 6.0)`
≈ 0.513 mm.
