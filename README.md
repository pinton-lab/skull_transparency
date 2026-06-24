# skull_transparency

Skull acoustic **transparency maps** (via time-reversal reciprocity) and **transducer placement**
for transcranial focused ultrasound.

> **📖 Full guide: [`tutorial/tutorial.pdf`](tutorial/tutorial.pdf)** — what the tool does, how the
> grid, target and medium are defined, how to bring your own CT, the solver, and the API. This
> README is just the 60-second version.

## The idea

Run **one** outward time-reversal solve: a point source at the brain **target** radiates through the
skull CT, and we record the field that emerges on the **external skull surface**. By acoustic
**reciprocity** the field reaching each patch equals what a transducer *there* would deliver to the
target — so one solve yields the transmit coupling of every surface patch (a *skull transparency
map*), and the best window is where that coupling is highest.

## Two layers (consumers never need a GPU)

- **Consumer** (pure Python) — work from a precomputed **Field Bundle**:
  `compute_transparency_map → place_bowl → placement.json`. No GPU, no external solver.
- **Producer** (GPU) — build a bundle from your own CT: `prepare → outward solve → extract`.

```bash
pip install -e .                   # core: numpy, scipy  (consumer layer)
pip install -e '.[registration]'   # + tuba, nibabel       (MNI <-> subject frames)
pip install -e '.[viz]'            # + napari, matplotlib  (surface render / positioning tool)
pip install -e '.[solver]'         # + fullwave2-ultra     (producer layer, needs a CUDA GPU)
```

## Quick start (consumer)

```python
import skull_transparency as st

bundle = st.load_bundle("/path/to/field_bundle")
tmap   = st.compute_transparency_map(bundle)                    # per-patch coupling
pl     = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=63.2))
pdct   = st.to_placement_dict(pl, target_name="dentate_left")   # placement.json
```

Always read the output dict's `frame` key (`nrrd_voxel_mm` with tuba, else `mni_ras_mm`). See
`examples/halle_dentate/` for the full pipeline and `examples/synthetic/run_synthetic.py` for a
no-GPU/no-data smoke test. The CLI (`skull-transparency prepare | extract | place | position`), the
full API, and the method are all in **[`tutorial/tutorial.pdf`](tutorial/tutorial.pdf)**.

## One correctness note

Placement uses the **raw** delivered intensity, not the distance-corrected map: by reciprocity the
raw outward intensity at a patch already equals delivered energy (spreading included). The `(r/r̄)²`
distance correction is for *visualising* bone transmission only — using it for placement
over-rewards far, thin-bone windows. The single-frequency `√∫|G|²` score is correct for *choosing*
a window but **overstates** the broadband focusing gain and depth of field; quote those from the
inward re-simulation.
