"""Extract the FOCAL pressure field from a solved forward-focus run.

The forward-focus launcher (:func:`launchers.launch_forward_focus`) records, in addition to
the aperture channels, a fine focal box ``C.focal_box(focus, N, fb)`` whose voxels are the LAST
``len(box)`` columns of ``genout.dat``. We take the peak |p| over time at each box voxel to get
the focal **peak-pressure** field, the focal **gain** (focal peak / per-element drive ``p0``), and
the box voxel coords (+ MNI mm via the run's registration) so the GUI can overlay the focal spot
in the same frame as the transparency surface.

Outputs (small — brought back to the workstation; the multi-GB ``genout.dat`` stays on the GPU host):
  * ``focal_Pmax.npy``      (Nbox,)   f32  peak pressure per box voxel
  * ``focal_coords_vox.npy``(Nbox,3)  f8   fullres voxel coords of the box
  * ``focal_coords_mni.npy``(Nbox,3)  f8   MNI-RAS mm (only if a registration is found)
  * ``focal_gain.json``               focus/dent voxel, p0, focal_peak, gain, peak location, mode
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from . import fwio


def extract_focal(run_dir, out_dir=None, registration_path=None, frame_chunk=256):
    """Build the focal-field artifacts from a solved forward-focus ``run_dir`` (holds
    ``box_info.mat`` + ``genout.dat``). ``out_dir`` defaults to ``run_dir``. Streams the
    genout in ``frame_chunk`` frame blocks (running max of |p|) to bound memory."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bi = sio.loadmat(run_dir / "box_info.mat")
    box = np.asarray(bi["box"], dtype=np.float64)          # (Nbox,3) fullres voxel coords
    focus = np.asarray(bi["focus"], float).ravel()
    dent = np.asarray(bi["dent"], float).ravel()
    p0 = float(np.asarray(bi["p0"]).ravel()[0]) if "p0" in bi else 1.0
    mode = str(np.asarray(bi["MODE"]).ravel()[0]) if "MODE" in bi else "?"
    dX = float(np.asarray(bi["dX"]).ravel()[0])
    Nbox = box.shape[0]

    gfile = run_dir / "genout.dat"
    nbytes = fwio.sizeOfFile(str(gfile))
    # ncoordsout = aperture channels + box; box is the LAST Nbox columns.
    # Recover ncoordsout from the recorded outcoords count (outc.dat) when present, else infer.
    outc_f = run_dir / "outc.dat"
    if outc_f.exists():
        ncoordsout = int(np.fromfile(outc_f, dtype="<i4").reshape(-1, 5).shape[0])
    else:
        ncoordsout = int(np.fromfile(run_dir / "ncoordsout.dat", dtype="<i4").ravel()[0])
    nframes = nbytes // 4 // ncoordsout
    box_cols = np.arange(ncoordsout - Nbox, ncoordsout, dtype=np.int64)

    # streaming running-max of |p| over the box columns
    pk = np.zeros(Nbox, dtype=np.float64)
    for f0 in range(0, nframes, frame_chunk):
        frames = np.arange(f0, min(f0 + frame_chunk, nframes))
        chunk = fwio.readGenoutSlice(str(gfile), frames, ncoordsout, box_cols)  # (nf, Nbox)
        np.maximum(pk, np.abs(chunk).max(axis=0), out=pk)

    focal_peak = float(pk.max())
    ipk = int(np.argmax(pk))
    gain = focal_peak / p0 if p0 else float("nan")

    np.save(out_dir / "focal_Pmax.npy", pk.astype(np.float32))
    np.save(out_dir / "focal_coords_vox.npy", box)

    coords_mni = None
    reg_path = registration_path
    if reg_path is None:
        for cand in (run_dir / "registration.json", run_dir.parent / "registration.json"):
            if cand.exists():
                reg_path = cand
                break
    if reg_path is not None and Path(reg_path).exists():
        from ..registration import Registration
        reg = Registration.from_json(reg_path)
        coords_mni = reg.fullres_to_mni(box)
        np.save(out_dir / "focal_coords_mni.npy", coords_mni)

    info = {
        "mode": mode,
        "focus_vox": focus.tolist(),
        "dent_vox": dent.tolist(),
        "p0": p0,
        "focal_peak": focal_peak,
        "gain": gain,
        "peak_loc_vox": box[ipk].tolist(),
        "peak_loc_mni": (coords_mni[ipk].tolist() if coords_mni is not None else None),
        "n_box": int(Nbox),
        "dx_mm": dX,
        "ncoordsout": int(ncoordsout),
        "nframes": int(nframes),
    }
    (out_dir / "focal_gain.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="extract focal field from a forward-focus run")
    ap.add_argument("--run", required=True, help="forward-focus run dir (box_info.mat + genout.dat)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--registration", default=None)
    a = ap.parse_args()
    info = extract_focal(a.run, a.out, a.registration)
    print(json.dumps(info, indent=1))
