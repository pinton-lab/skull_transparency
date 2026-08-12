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
`examples/halle_dentate/` for the full pipeline, `examples/brain_center/run_brain_center.py` for the
neutral whole-skull baseline, and `examples/synthetic/run_synthetic.py` for a no-GPU/no-data smoke
test. The CLI (`skull-transparency prepare | extract | place | position | transparency | explore |
report`), the full API, and the method are all in **[`tutorial/tutorial.pdf`](tutorial/tutorial.pdf)**.

## On your own computer, without a GPU

The GPU is only needed to *mint* a map; exploring one is laptop-fast. Precomputed maps for the
ITRUSST benchmark skull (500 kHz, one per named MNI target) live in the **gallery** — fetch one by
name and place a transducer on it interactively:

```bash
pip install 'skull-transparency[viz]'
skull-transparency explore --list-targets           # what's precomputed
skull-transparency explore --target dACC_left       # fetch + open in napari (3-D)
skull-transparency report  --target dACC_left --out dacc.html   # self-contained HTML report
```

`explore`/`report` also take `--bundle DIR` for a map you computed yourself. On a Linux/NVIDIA
machine (or WSL2), **one command mints a new map** — it fetches the solver on first use behind its
license prompt: `skull-transparency compute --target dACC_left` (any MNI point via
`--target-mm x,y,z`). No GPU anywhere? The [Colab
notebook](notebooks/skull_transparency_colab_500kHz.ipynb) computes the same bundle on a free T4 —
download its `run/bundle`. The package is also a **napari plugin**: bundle directories and `*.skullbundle.zip`
files open straight from *File ▸ Open*, and the *Transparency explorer* widget picks gallery
targets from a menu. Platform notes (Linux / Windows-WSL2 / macOS) are in
[`docs/desktop.md`](docs/desktop.md).

For a **neutral whole-skull** view independent of any one target, run a brain-center baseline:
`skull-transparency prepare --center …` seats one omnidirectional source at the brain center, and
`skull-transparency transparency --bundle … --out map.png` renders the 1/r²-corrected map
(tutorial §5). It is not human-specific — for a non-human skull set `--bone-threshold` to its bone
sound speed (thin bone is slower than the human `2200` default) and, if the image-only centroid is
off, `--center-mm x,y,z`.

## One correctness note

Placement uses the **raw** delivered intensity, not the distance-corrected map: by reciprocity the
raw outward intensity at a patch already equals delivered energy (spreading included). The `(r/r̄)²`
distance correction is for *visualising* bone transmission only — using it for placement
over-rewards far, thin-bone windows. The single-frequency `√∫|G|²` score is correct for *choosing*
a window but **overstates** the broadband focusing gain and depth of field; quote those from the
inward re-simulation.
