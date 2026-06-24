# skull_transparency

Skull acoustic **transparency maps** (via time-reversal reciprocity) and **transducer placement**
for transcranial ultrasound. Extracted from the `hemisphere_tr` study into a reusable library that
outside repos (e.g. `neuromod_parameters`) can consume.

## The idea

Run **one** outward time-reversal simulation: a point source at the brain **target** radiates through
the skull CT, and we record the field that emerges on the **external skull surface**. By acoustic
**reciprocity**, the field reaching each surface patch equals what a transducer *at that patch* would
deliver to the target. So a single solve yields the transmit coupling of **every** surface patch — a
*skull transparency map* — and the best place to put a transducer is where that coupling is highest.

## Two-layer design (consumers never need MATLAB/GPU)

```
LAYER A  sim/   (optional)  MATLAB fullwave2 CUDA outward/inward TR  ──► Field Bundle
LAYER C  core   (pure py)   compute_transparency_map ─► place_bowl ─► placement.json
```

A consumer installs only the pure-Python layer and works from a precomputed **Field Bundle** (or the
small **`surface_map.npz`** transparency bundle, ~26 MB) — no MATLAB, CUDA, or napari.

```bash
pip install -e .                       # core: numpy, scipy
pip install -e '.[registration]'       # + tuba, nibabel  (MNI<->subject, NRRD frame)
pip install -e '.[viz]'                # + napari, matplotlib  (surface render)
```

## Quick start

```python
import skull_transparency as st

bundle = st.load_bundle("/path/to/field_bundle")          # or build_field_bundle(...) from a legacy run
tmap   = st.compute_transparency_map(bundle)              # TransparencyMap (per-patch coupling)
tmap.to_npz("surface_map.npz")                            # small distributable

pl   = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60, bowl_radius_mm=15))
pdct = st.to_placement_dict(pl, target_name="dentate_left")   # neuromod placement.json (NRRD-voxel-mm via tuba)
```

> **Output frame:** `to_placement_dict()` maps coordinates into the subject's NRRD-voxel-mm
> frame via a tuba species module — `tuba.species.human` (Halle) by default. With tuba present
> the dict's `frame` is `nrrd_voxel_mm`; without it (or for a species you haven't wired) it falls
> back to `mni_ras_mm`. Always read the `frame` key. For non-Halle subjects pass the matching
> species to the neuromod adapter's `place_with_transparency(..., species=...)`.

See `examples/halle_dentate/` for the full pipeline reproducing the original dentate result.

## Command line

The `skull-transparency` console script wraps the chain:

```bash
# 1. skull map (+ target + transducer) -> a sim tree of .dat solver inputs (no GPU)
skull-transparency prepare --c-map c.npy --affine A.npy --target -12,-57,-34 \
    --transducer ctx500.json --approach 0,0,1 --out run/

# 2. run the CUDA solve (GPU), then extract a Field Bundle from genout_mod.dat:
python -m skull_transparency.sim outward --sim run --out run --run
skull-transparency extract --run run/outward --sim run --out run/bundle

# 3. bundle -> the three outputs:
skull-transparency place --bundle run/bundle --out result/
#   -> surface_map.npz (transparency map) + score.json (positioning score) + placement.json

# optional positioning tool: a placement-preview PNG (or --interactive napari, needs [viz] + a display):
skull-transparency position --bundle run/bundle --out preview.png
```

`score.json` reports `normalized` (0..1, chosen window vs best achievable) and
`focal_pressure_proxy` (√∫|G|² over the aperture) — a placement/energy proxy that
**overstates** the broadband focusing gain (quote that from the inward re-simulation).

No data, GPU, or tuba? Smoke-test the whole back half on a synthetic fixture:
`python examples/synthetic/run_synthetic.py` (or `st.make_synthetic_bundle(dir)` in code).

## Key API

| call | purpose |
|---|---|
| `build_field_bundle(data_dir, meta, transform)` | turn a legacy `hemisphere_tr` run into a self-describing bundle |
| `compute_transparency_map(bundle, options)` | external-surface coupling map (`TransparencyMap`) |
| `place_bowl(tmap, constraints)` | delivered-energy-optimal bowl window + apex + beam (`BowlPlacement`) |
| `place_bowl_optimal(tmap, constraints)` | TR-optimal window = argmax of the surface integral J(S)=∫_S\|G\|² dS; focal peak ∝ √J |
| `place_array(tmap, constraints)` | sparse multi-element selection (greedy, spacing/incidence/region constrained) (`ArrayLayout`) |
| `bundle.element_complex_field()` | complex (amplitude+phase) element field from the array traces (`ComplexField`) |
| `drive_optimal / drive_phase_only(cf, idx)` | TR (phase-conjugate) and equal-amplitude phase-only drives + on-target peak |
| `focal_psf / focal_fwhm(cf, idx, …)` | angular-spectrum focal-spot predictor (order-correct; transverse FWHM ±35%) |
| `project_to_sphere(G, rad, R, k0)` | Pinton-2012 radial projection of the complex field to a transducer sphere of radius R |
| `to_placement_dict(placement)` | emit `neuromod_parameters` `placement.json` (frame chain via `tuba`) |
| `Registration` | rigid full-res-voxel ↔ MNI-RAS-mm map (clean rotation + 0.28 mm) |

## Surface-integral placement (single-solve optimization)

By reciprocity the field a point source at the **target** radiates to every surface point is, by
acoustic reciprocity, what a transducer *there* delivers back. So for **any** aperture S the
achievable focal peak under the optimal (time-reversal) drive is the power-constrained surface
integral `p_max ∝ √(∫_S |G|² dS)` — a single precomputed map. Optimizing placement / aperture /
orientation is then a **moving-window correlation over one map**, not a wave solve per candidate
(contrast filter-then-optimize methods like SCOUT). The phase (from the array traces) also gives
the Pinton-2012 radial projection to any transducer radius, the phase-only drive, and an
angular-spectrum focal-spot predictor.

```python
cf  = bundle.element_complex_field()                 # complex element field (phase from traces)
bp  = st.place_bowl_optimal(tm, st.BowlConstraints(bowl_radius_mm=20))   # √J over all windows
# select the ARRAY elements over that window (drive/PSF take array-element indices, NOT the
# dense-surface bp.footprint_surf_idx); e.g. the high-coupling cluster near the window:
import numpy as np
pos_mm = cf.pos_fullres * cf.dx_mm
idx = np.where(np.linalg.norm(pos_mm - pos_mm[cf.E_win.argmax()], axis=1) <= 50)[0]
u, pmax = st.drive_optimal(cf, idx)                  # TR (phase-conjugate) drive, optimal peak
fwhm    = st.focal_fwhm(cf, idx)                     # predicted focal-spot FWHM, no extra solve
```

> **Use the direct-arrival energy for placement** (`Ipk_Wcm2` / ballistic-windowed `E_win`), not
> the full-record time-integral `Iint` — the latter is reverberation-contaminated and mis-selects
> a window. The single-frequency phasor is correct for *placement* (energy) but **overstates** the
> TR-vs-geometric *focusing* gain (broadband) and the axial depth of field; quote those from the
> broadband/inward re-simulation. See `examples/halle_dentate/surface_integral_placement.py`.

## Notes on correctness

- **Placement uses RAW delivered intensity, not the distance-corrected map.** By reciprocity the raw
  outward intensity at a patch already equals delivered energy (spreading included); the
  `(r/r̄)²` distance correction (`TransparencyOptions.distance_correct`, `BowlConstraints.use_distance_corrected`)
  is for *visualising* bone transmission and over-rewards far thin-bone windows if used for placement.
- **Incidence uses the true local surface normal** (speed-map gradient), not the radial direction, so
  obliquely-hit patches are not over-credited.
- The clean rigid `Registration` deliberately supersedes the internally-inconsistent `Amn/bmn/dds/scale`
  affine in legacy `ppw55_transform.npz` (kept only as `deprecated_affine` provenance).

## Status

v1: the pure-Python layer (`bundle`, `registration`, `surface`, `transparency`, `metrics`, `placement`,
`neuromod`) + tests + the dentate example. The MATLAB sim layer (`sim/`) is reused from `hemisphere_tr`
as-is; generalising its launchers/bridge into `sim/` runners is the next step.
