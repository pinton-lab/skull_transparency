"""Command-line front-end for the time-reversal launchers.

Mirrors the per-script MATLAB env vars. Examples::

  python -m skull_transparency.sim outward          --sim DIR --out DIR
  python -m skull_transparency.sim inward_windowed   --sim DIR --out DIR
  python -m skull_transparency.sim skullonly         --margin 16
  python -m skull_transparency.sim skullonly_target  --target-vox 300,200,210 --outsub thalamus
  python -m skull_transparency.sim subset_focalbox   --mode tr --selfile sel_transparency.i32
  python -m skull_transparency.sim verify            --dirs outward inward_win   # bit-identity check

``--sim`` defaults to the legacy ``hemisphere_tr/sim`` tree; ``--out`` defaults
to ``$PWD`` (write the regenerated tree wherever you like — never the source).
Add ``--run`` to invoke the CUDA solver (off by default).
"""
from __future__ import annotations

import argparse
import os
import sys

from .. import paths

DEFAULT_SIM = str(paths.sim_dir())   # FULLWAVE2_SIM_DIR, else SKULL_TR_DATA_ROOT/sim


def _parse_vox(s):
    return [int(x) for x in s.replace(",", " ").split()]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="skull_transparency.sim",
                                description="Pure-Python fullwave2 TR launchers")
    p.add_argument("which", help="launcher name (e.g. outward, inward_windowed, "
                   "skullonly, subset_focalbox, verify)")
    p.add_argument("--sim", default=DEFAULT_SIM,
                   help="source sim tree (read-only); defaults via $FULLWAVE2_SIM_DIR / $SKULL_TR_DATA_ROOT")
    p.add_argument("--out", default=os.getcwd(), help="output root")
    p.add_argument("--margin", type=int, default=16)
    p.add_argument("--mode", default="tr", choices=["tr", "geo", "flat"],
                   help="inward drive: tr (time-reversal), geo (geometric delays), flat (zero-phase)")
    p.add_argument("--selfile", default=None)
    p.add_argument("--outsub", default=None)
    p.add_argument("--srcdir", default=None)
    p.add_argument("--arrfile", default=None)
    p.add_argument("--target-vox", default=None)
    p.add_argument("--gpuid", default="0")
    p.add_argument("--run", action="store_true", help="invoke the CUDA solver")
    p.add_argument("--recorder", default="volume", choices=["volume", "shell"],
                   help="(outward) field recorder: 'volume' = genout_mod full-field dump (default); "
                        "'shell' = calvarial-surface only (~150-1000x smaller; enough for the "
                        "transparency map, so a 6-PPW whole-skull run fits a laptop / free Colab)")
    p.add_argument("--dirs", nargs="*", default=None, help="(verify) subdirs to check")
    p.add_argument("--full", action="store_true",
                   help="(verify) also regenerate+compare the multi-GB medium maps "
                        "for inward dirs (default: light — icmat/coords/scalars)")
    a = p.parse_args(argv)

    from . import launchers as L
    sim, out, run = a.sim, a.out, a.run

    if a.which == "verify":
        from .verify import verify_dirs
        rc = verify_dirs(a.sim, a.out, a.dirs, full=a.full)
        sys.exit(rc)

    tv = _parse_vox(a.target_vox) if a.target_vox else None
    dispatch = {
        "outward": lambda: L.launch_outward(sim, out, run_solver=run, recorder=a.recorder),
        "inward": lambda: L.launch_inward(sim, out, run_solver=run),
        "inward_windowed": lambda: L.launch_inward_windowed(sim, out, run_solver=run),
        "inward_focalbox": lambda: L.launch_inward_focalbox(sim, out, run_solver=run),
        "skullonly": lambda: L.launch_skullonly(sim, out, MARGIN=a.margin, run_solver=run, gpuid=a.gpuid),
        "skullonly_array": lambda: L.launch_skullonly_array(sim, out, MARGIN=a.margin, run_solver=run, gpuid=a.gpuid),
        "skullonly_target": lambda: L.launch_skullonly_target(sim, out, tv, a.outsub, MARGIN=a.margin, run_solver=run, gpuid=a.gpuid),
        "skullonly_target_array": lambda: L.launch_skullonly_target_array(sim, out, tv, a.arrfile, a.outsub, MARGIN=a.margin, run_solver=run, gpuid=a.gpuid),
        "subset_focalbox": lambda: L.launch_subset_focalbox(sim, out, mode=a.mode, selfile=a.selfile, outsub=a.outsub, run_solver=run, gpuid=a.gpuid),
        "skullonly_subset_focalbox": lambda: L.launch_skullonly_subset_focalbox(sim, out, mode=a.mode, selfile=a.selfile, outsub=a.outsub, run_solver=run, gpuid=a.gpuid),
        "skullonly_target_focalbox": lambda: L.launch_skullonly_target_focalbox(sim, out, a.srcdir, a.selfile, mode=a.mode, outsub=a.outsub, run_solver=run, gpuid=a.gpuid),
    }
    if a.which not in dispatch:
        p.error(f"unknown launcher {a.which!r}; choose from {sorted(dispatch)}")
    outdir = dispatch[a.which]()
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
