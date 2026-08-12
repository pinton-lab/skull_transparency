# Desktop use — platforms, the gallery, and the napari plugin

The pipeline splits into two products with the **Field Bundle** as the interchange format:

- **Map maker** (`prepare → sim outward → extract`): runs the CUDA solver. Needs an
  NVIDIA GPU (~6 GB for the 500 kHz benchmark skull at 6 PPW).
- **Map explorer** (`explore`, `report`, `place`, the napari plugin): pure post-processing
  on a bundle. Runs anywhere Python runs, in seconds, no GPU.

Most day-to-day use — choosing a window, posing a transducer, exporting a report — is the
explorer side.

## Platform matrix

| Platform | Explore / report / napari | Compute new maps |
|---|---|---|
| Linux + NVIDIA GPU | yes | yes (local solver binary) |
| Windows + NVIDIA GPU | yes (native Python) | yes, **via WSL2** (the solver is a Linux binary; CUDA on WSL2 is supported by NVIDIA) |
| macOS (Intel or Apple silicon) | yes | no local solves — use the gallery, the Colab notebook, or a lab Linux box |

## Install

```bash
pip install 'skull-transparency[viz]'    # explorer + napari + matplotlib
```

The base package (`pip install skull-transparency`) is numpy/scipy only and covers the
scripting API and `report`. The CUDA solver binary is **not** bundled: computing new maps
fetches it from the `fullwave2-ultra` distribution (its noncommercial license is shown at
that point), or you point `FULLWAVE2_BIN` at a copy.

## The precomputed gallery (no GPU at all)

One transparency map exists per named MNI target on the ITRUSST benchmark skull at
500 kHz. `skull-transparency explore --list-targets` shows them;
`--target NAME` fetches (checksum-verified, cached under `~/.cache/skull_transparency`)
and opens the map. Sources, in order:

1. the local cache;
2. a local directory of `*.skullbundle.zip` files named by `$SKULL_TRANSPARENCY_GALLERY`
   (labs can mirror the gallery on a shared drive);
3. the published download URL in the packaged registry.

Need a target that is not in the gallery, or a different frequency/skull? Mint the bundle
yourself — locally on a CUDA GPU (`skull-transparency run …`) or with the
[Colab notebook](../notebooks/skull_transparency_colab_500kHz.ipynb) (free T4; download
`run/bundle` and pass `--bundle`). **Subject CTs are different**: do not upload them to
Colab — compute locally or on an institutional machine.

## napari plugin

Installing `[viz]` registers the plugin automatically:

- *File ▸ Open* accepts a Field Bundle directory, its `bundle.json`, or a
  `*.skullbundle.zip` — the map loads as a coloured surface Points layer with the default
  placement marked.
- *Plugins ▸ Skull Transparency ▸ Transparency explorer* is the gallery picker: choose a
  target and focal length, and the fetched map + placement appear in the viewer.
- `skull-transparency explore --target NAME` does the same from the terminal, and falls
  back to a static preview PNG when napari (or a display) is missing.

## Reports

`skull-transparency report --target NAME --out report.html` writes a single
self-contained HTML file: the transparency map (four views), the chosen placement, and a
summary table — target and window in MNI mm, window-to-target distance, incidence,
transparency score, and the nearest EEG 10-20 site (so the window can be located on a
head the way an experimenter would). Attach it to a protocol or share it as-is; there are
no external assets.
