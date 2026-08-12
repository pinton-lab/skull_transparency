#!/usr/bin/env python
"""Drive a TRUE fullwave FORWARD focusing solve at a chosen focus point, end to end.

This is the backend the GUI's focus-depth chooser calls. Given a Field Bundle and a focus point
(MNI mm, default = the run's target), it:

  1. maps the focus MNI -> fullres voxel (the run's registration),
  2. ssh's the GPU host, runs ``skull_transparency.sim forward_focus`` (writes the drive/coords,
     reusing the outward medium maps),
  3. runs ``bench_3d_opt`` inside the Ubuntu-22.04 Apptainer (orta glibc 2.31 < 2.34 — see the log),
  4. extracts the focal field (``skull_transparency.sim.extract_focal``),
  5. rsyncs the small focal artifacts back beside the local bundle as ``focus_<id>/``.

It prints (and writes ``<focus_dir>/focal_gain.json``) the focal peak + gain. Progress is appended to
``<focus_dir>/driver.log`` so the GUI (or a human) can tail it — ScheduleWakeup is unreliable here.

Example::

  python run_focus_solve.py --bundle thalamus_vim_left_run_v2/bundle --focus-at-target --id target
  python run_focus_solve.py --bundle thalamus_vim_left_run_v2/bundle --focus-mni -84,100,112 --id deep
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import skull_transparency as st
from skull_transparency.registration import Registration

# --- GPU-host defaults (orta). All overridable via flags / env. ---
GPU_HOST = os.environ.get("FOCUS_GPU_HOST", "orta.bme.unc.edu")
SCRATCH = os.environ.get("FOCUS_SCRATCH", "/home/hatim/skull_transparency_runs")
REPO_ON_HOST = os.environ.get("FOCUS_REPO", "/home/hatim/mount/Github/pinton-lab")
HOST_PY = os.environ.get("FOCUS_HOST_PY", "/home/hatim/.venvs/diffbf/bin/python")
SIF = os.environ.get("FOCUS_SIF", "ubuntu2204.sif")            # under $SCRATCH
BENCH = os.environ.get("FOCUS_BENCH", "bin/bench_3d_opt")      # under $SCRATCH


def _run(cmd, logf):
    logf.write(f"\n$ {cmd}\n"); logf.flush()
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    logf.write(p.stdout); logf.flush()
    return p.returncode, p.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, help="local Field Bundle dir")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--focus-at-target", action="store_true", help="focus at the run target (default)")
    g.add_argument("--focus-mni", default=None, help="focus point in MNI-RAS mm 'X,Y,Z'")
    ap.add_argument("--mode", default="geo", choices=["geo", "tr", "flat"],
                    help="drive: geo (steerable, skull-uncorrected; default), tr (skull-corrected, target only), flat")
    ap.add_argument("--id", default=None, help="short name for this focus run (-> focus_<id>/)")
    ap.add_argument("--gpu", default="3", help="GPU id on the host")
    ap.add_argument("--orta-run", default=None, help="run-tree name under $SCRATCH (default: bundle's grandparent name)")
    ap.add_argument("--host", default=GPU_HOST)
    a = ap.parse_args(argv)

    bundle_dir = Path(a.bundle).resolve()
    bundle = st.load_bundle(bundle_dir)
    reg = bundle.registration
    if reg is None:
        sys.exit("bundle has no registration — cannot map MNI focus to voxel")
    target_mni = np.asarray(bundle.target["mni_ras_mm"], float)

    if a.focus_mni:
        focus_mni = np.array([float(x) for x in a.focus_mni.replace(",", " ").split()])
    else:
        focus_mni = target_mni                                    # default / --focus-at-target
    focus_vox = np.round(reg.mni_to_fullres(focus_mni)).astype(int)

    fid = a.id or ("target" if np.allclose(focus_mni, target_mni) else
                   "f_" + "_".join(f"{v:.0f}" for v in focus_mni))
    outsub = f"focus_{a.mode}_{fid}"
    local_focus = bundle_dir.parent / outsub
    local_focus.mkdir(parents=True, exist_ok=True)

    run_name = a.orta_run or bundle_dir.parent.name                # e.g. thalamus_vim_left_run_v2
    RUN = f"{SCRATCH}/{run_name}"
    D = f"{RUN}/{outsub}"
    host = a.host
    PYH = f"PYTHONPATH={REPO_ON_HOST}/skull_transparency/src {HOST_PY}"

    logf = open(local_focus / "driver.log", "w")
    def status(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    status(f"focus MNI {np.round(focus_mni,2)} mm -> fullres voxel {focus_vox.tolist()} "
           f"(target MNI {np.round(target_mni,2)})")
    status(f"mode={a.mode}  host={host} GPU{a.gpu}  run={run_name}  outsub={outsub}")

    # 1) write inputs on the host (reuse outward maps)
    fvox = ",".join(map(str, focus_vox.tolist()))
    cmd_prep = (f"ssh {host} '{PYH} -m skull_transparency.sim forward_focus "
                f"--sim {RUN} --out {RUN} --src {RUN}/outward --mode {a.mode} "
                f"--focus-vox {fvox} --outsub {outsub}'")
    status("STEP 1/3 write drive+coords (reuse maps) ...")
    rc, _ = _run(cmd_prep, logf)
    if rc != 0:
        status("FAILED at step 1 (prepare)"); sys.exit(rc)

    # 2) GPU solve in the Apptainer container (glibc fix)
    cmd_solve = (f"ssh {host} 'cd {D} && apptainer exec --nv "
                 f"--env CUDA_VISIBLE_DEVICES={a.gpu} {SCRATCH}/{SIF} {SCRATCH}/{BENCH} "
                 f"> {D}/solve.log 2>&1; tail -c 40 {D}/solve.log'")
    status("STEP 2/3 GPU forward solve (bench_3d_opt in Apptainer) ...")
    t0 = time.time()
    rc, out = _run(cmd_solve, logf)
    status(f"solve returned in {time.time()-t0:.0f}s (tail: {out.strip()[-40:]!r})")

    # 3) extract focal field + pull artifacts back
    cmd_extract = (f"ssh {host} '{PYH} -m skull_transparency.sim.extract_focal "
                   f"--run {D} --registration {RUN}/registration.json'")
    status("STEP 3/3 extract focal field ...")
    rc, _ = _run(cmd_extract, logf)
    if rc != 0:
        status("FAILED at step 3 (extract)"); sys.exit(rc)

    # pull the small focal artifacts (NOT the multi-GB genout). One rsync per file: the driver's
    # shell is /bin/sh (dash) which does not brace-expand, and rsync itself never brace-expands.
    for fn in ("focal_Pmax.npy", "focal_coords_vox.npy", "focal_coords_mni.npy",
               "focal_gain.json", "box_info.mat"):
        _run(f"rsync -a {host}:{D}/{fn} {local_focus}/", logf)

    # clean up the GPU host: the per-run genout/genout_mod (and rebuilt medium maps if not reused)
    # are large and no longer needed once the focal artifacts are extracted + pulled. Keep only the
    # small focal_*/box_info so a re-pull is possible; this stops the run dir from filling orta.
    status("cleaning up GPU-host genout (focal artifacts already pulled) ...")
    _run(f"ssh {host} 'rm -f {D}/genout.dat {D}/genout_mod.dat'", logf)

    info = json.loads((local_focus / "focal_gain.json").read_text())
    status(f"DONE  focal_peak={info['focal_peak']:.3g}  gain={info['gain']:.3g}  "
           f"peak_mni={np.round(info['peak_loc_mni'],2).tolist()}")
    status(f"artifacts -> {local_focus}")
    logf.close()
    # final machine-readable line for the GUI
    print("FOCUS_RESULT " + json.dumps({"focus_dir": str(local_focus), **info}))


if __name__ == "__main__":
    main()
