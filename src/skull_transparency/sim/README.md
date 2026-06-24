# `skull_transparency.sim` — pure-Python fullwave2 time-reversal launchers

A NumPy/SciPy rewrite of the MATLAB time-reversal **launcher** layer that used
to live in `hemisphere_tr/launchers/*.m` and the `fullwave2_3d` helper
functions. It generates the simulation **input** `.dat` files that feed the
external CUDA solver — **bit-for-bit identical** to what the MATLAB scripts
produce. No MATLAB, Octave, or MATLAB Runtime is needed.

> The CUDA solver binary is the fullwave2-ultra `bench_3d_opt` (byte-identical genout
> to the retired `fullwave2_3d_Aexp_genout_cuda_aperturegrowth_opt`), external and
> resolved via `$FULLWAVE2_BIN` or the `fullwave2_ultra` package. This package only
> replaces the MATLAB that
> *prepares* a run and *post-processes the recorded field into the next run's
> drive* (the time-reversal step). Pass `run_solver=True` to a launcher to
> invoke the solver exactly as the MATLAB `system(...)` calls did.

## What was ported

| MATLAB launcher | Python entry point |
|---|---|
| `fullwave2_launch_halle_hemis_tr_outward.m` | `launch_outward` |
| `fullwave2_launch_halle_hemis_tr_inward.m` | `launch_inward` |
| `fullwave2_launch_halle_hemis_tr_inward_windowed.m` | `launch_inward_windowed` |
| `fullwave2_launch_halle_hemis_tr_inward_focalbox.m` | `launch_inward_focalbox` |
| `fullwave2_launch_skullonly.m` | `launch_skullonly` |
| `fullwave2_launch_skullonly_array.m` | `launch_skullonly_array` |
| `fullwave2_launch_skullonly_target.m` | `launch_skullonly_target` |
| `fullwave2_launch_skullonly_target_array.m` | `launch_skullonly_target_array` |
| `fullwave2_launch_subset_focalbox.m` | `launch_subset_focalbox` |
| `fullwave2_launch_skullonly_subset_focalbox.m` | `launch_skullonly_subset_focalbox` |
| `fullwave2_launch_skullonly_target_focalbox.m` | `launch_skullonly_target_focalbox` |

The shared `fullwave2_3d` helpers were ported into:

* `fwio.py` — `writeVabs`, `writeMapXYZ`, `writeCoords`, `extendMap3d`,
  `maskBdy`, `readGenoutSlice`, `sizeOfFile`.
* `forcoef.py` — the FOR³D finite-difference delay coefficients (`d`, `dmap`).
* `launch_core.py` — `launch_fullwave2_3d_Aexp_noicmat`.
* `mlcompat.py` — the MATLAB built-ins the launchers rely on: the transmit
  pulse, `round` (half away from zero), `tukeywin`, `hilbert`-envelope,
  `interp1` (linear), and the global ballistic-window selection.

## Byte-format facts (the reverse-engineering)

* `.dat` files are raw little-endian; `'float'`→`float32`, `'int'`→`int32`.
* `writeMapXYZ` writes a map in **C-order** (`[i,j,k]`, k fastest) — the
  per-slice `permute(...,[3 2 1])` column-major write equals a C-order ravel.
* `writeCoords` writes each column in turn as int32 — a **Fortran-order** ravel
  of the `(ncoords, 5)` matrix, after `+48` on the x/y/z columns and `-1` on all.
* `extendMap3d` ≡ `numpy.pad(map, 48, mode='edge')` (48 = `nbdy(40)+M(8)`).
* The medium maps are computed in float64 (`K = c²·rho`) and cast to float32
  only at write time — matching MATLAB's double maps + `fwrite('float')`.
* The PML arrays the MATLAB function builds are only `plot`ted, never written,
  so they are intentionally not reproduced (the solver builds its own PML).

## CLI

```bash
# write the regenerated tree somewhere of your choosing (never the source tree)
python -m skull_transparency.sim outward          --sim /path/hemisphere_tr/sim --out /scratch/run
python -m skull_transparency.sim inward_windowed  --sim /path/hemisphere_tr/sim --out /scratch/run
python -m skull_transparency.sim skullonly_target --target-vox 300,200,210 --outsub thalamus
python -m skull_transparency.sim subset_focalbox  --mode tr --selfile sel_transparency.i32

# add --run to invoke the CUDA solver (off by default); --gpuid N to pick a device
```

## Verifying bit-identity

```bash
python -m skull_transparency.sim verify --out /scratch/run \
    --dirs outward skullonly skullonly_array skullonly_dACC skullonly_thalamus \
           inward_win inward_focalbox inward_sub_tr inward_sub_geo
```

This regenerates each subdirectory into `/scratch/run/_verify_scratch/` and
md5-compares every `.dat` against the committed reference produced by the
original MATLAB launcher. `genout.dat` (the solver output, which this layer does
not reproduce) is symlinked from the source tree so the inward launchers can
read the recorded field.

## Performance

Generating a run is **disk-write-bound** — the six 744³ medium maps are ~1.6 GB
each and the RAID writes at ~90 MB/s, so ~70 s of a ~62 s–88 s outward run is the
mandatory `.dat` I/O (the exact bytes are fixed by bit-identity). All compute
(replicate-pad, `K=c²·rho`, `rho_from_c`, `maskBdy`, the FOR³D coefficients) is
vectorised NumPy and only ~16 s. Optimisations applied:

* **`launch_core`** builds the extended maps one at a time and forms `K` in place
  (`c·=c; c·=rho`), keeping the peak working set near two extended grids instead
  of four; `rho_from_c` is evaluated in place. (All bit-identical — proven by
  `tests/test_sim_launchers.py`.)
* **Medium-map reuse (the big one).** The six maps depend only on the medium and
  grid size, so they are identical between an outward run and every inward
  time-reversal run that re-emits through the same skull. Each inward launcher
  therefore **hardlinks** them from the source run (`reuse_maps=True`, default)
  instead of recomputing and rewriting ~10 GB — instant, no extra disk, and
  byte-identical (a hardlink is the same bytes). MATLAB rewrote them every time.
  Pass `reuse_maps=False` to force independent copies.
* **`matlab_single_sum`** uses a vectorised `np.add.accumulate` (the 4-lane SIMD
  reduction) rather than a Python row loop.
* **`readGenoutSlice`** reads only the channel prefix actually requested per
  frame (the array channels are the leading rows on disk), so the inward
  launchers touch a few MB of the 75 GB `genout.dat` rather than all of it.

## `workspace.npz` (replaces `workspace.mat`)

The inward launchers need the outward run's parameters, source/recorder
coordinates and medium. MATLAB stored these in a `workspace.mat`
(v7.3 HDF5, embedding the full ~GB sound-speed map). The Python launchers write
a compact `workspace.npz` that stores the scalars + a *descriptor* of the medium
and the decimated-volume recorder grid, and reconstructs the large arrays on
demand (`_common.rebuild_medium` / `rebuild_outcoords`). The generated solver
inputs are unaffected (and verified bit-identical); only the intermediate
hand-off format changed. `box_info.mat` is still written via `scipy.io.savemat`
so the existing analysis code can read it.
