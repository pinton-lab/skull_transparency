"""Bit-identity verification harness.

Regenerates a sim subdir's ``.dat`` files into a scratch tree and byte-compares
(md5) each against the committed reference produced by the original MATLAB
launcher. The source tree (``sim_dir``) is read-only; ``genout.dat`` (the CUDA
solver output, which we don't reproduce) is symlinked into the scratch tree so
the inward launchers can read it.
"""
from __future__ import annotations

import hashlib
import json
import os
import numpy as np

from .. import paths

from . import launchers as L

# committed dirs we can verify, with how to (re)generate them.
# 'src' lists prerequisite source dirs that must exist (workspace.npz + genout).
REGISTRY = {
    "outward":        dict(kind="outward"),
    "skullonly":      dict(kind="skullonly"),
    "skullonly_array": dict(kind="skullonly_array"),
    "skullonly_dACC": dict(kind="skullonly_target", outsub="dACC"),
    "skullonly_thalamus": dict(kind="skullonly_target", outsub="thalamus"),
    "inward_win":     dict(kind="inward_windowed", src="outward"),
    "inward_focalbox": dict(kind="inward_focalbox", src="outward"),
    "inward_sub_tr":  dict(kind="subset_focalbox", mode="tr", src="outward"),
    "inward_sub_geo": dict(kind="subset_focalbox", mode="geo", src="outward"),
}

_SKIP = {"genout.dat"}   # solver output, not produced by the launcher


def _md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _link_genout(sim_dir, scratch, subdir):
    ref = os.path.join(sim_dir, subdir, "genout.dat")
    dst_dir = os.path.join(scratch, subdir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "genout.dat")
    if os.path.exists(ref) and not os.path.exists(dst):
        os.symlink(ref, dst)


def _ensure_source(sim_dir, scratch, src):
    """Generate the source workspace (so inward can read it) + link its genout.
    Uses light mode (no multi-GB maps) — only workspace.npz is needed here."""
    if os.path.exists(os.path.join(scratch, src, "workspace.npz")):
        _link_genout(sim_dir, scratch, src)
        return
    if src == "outward":
        L.launch_outward(sim_dir, scratch, write_maps=False)
    elif src == "skullonly_array":
        L.launch_skullonly_array(sim_dir, scratch, write_maps=False)
    else:
        raise ValueError(f"don't know how to build source {src!r}")
    _link_genout(sim_dir, scratch, src)


def _generate(sim_dir, scratch, subdir, spec, light):
    """``light`` skips the multi-GB medium maps (c/K/rho/beta/Aexp/dcmap), which
    are already proven bit-identical by the full outward/skullonly runs (same
    ``launch_core``, same medium). Inward dirs default to light."""
    kind = spec["kind"]
    if "src" in spec:
        _ensure_source(sim_dir, scratch, spec["src"])
    wm = not light
    if kind == "outward":
        return L.launch_outward(sim_dir, scratch, write_maps=wm)
    if kind == "skullonly":
        return L.launch_skullonly(sim_dir, scratch)
    if kind == "skullonly_array":
        return L.launch_skullonly_array(sim_dir, scratch, write_maps=wm)
    if kind == "skullonly_target":
        # target voxel = ref meta.json dent_grid (crop-frame voxel)
        with open(os.path.join(sim_dir, subdir, "meta.json")) as f:
            meta = json.load(f)
        tv = [int(round(x)) for x in meta["dent_grid"]]
        return L.launch_skullonly_target(sim_dir, scratch, tv, spec["outsub"])
    if kind == "inward_windowed":
        return L.launch_inward_windowed(sim_dir, scratch, write_maps=wm)
    if kind == "inward_focalbox":
        return L.launch_inward_focalbox(sim_dir, scratch, write_maps=wm)
    if kind == "subset_focalbox":
        return L.launch_subset_focalbox(sim_dir, scratch, mode=spec["mode"], write_maps=wm)
    raise ValueError(kind)


def verify_one(sim_dir, scratch, subdir, light=False):
    spec = REGISTRY[subdir]
    gen = _generate(sim_dir, scratch, subdir, spec, light)
    ref_dir = os.path.join(sim_dir, subdir)
    gen_files = {f for f in os.listdir(gen) if f.endswith(".dat") and f not in _SKIP}
    results = []
    for f in sorted(gen_files):
        ref = os.path.join(ref_dir, f)
        cur = os.path.join(gen, f)
        if not os.path.exists(ref):
            results.append((f, "no-ref", None, None))
            continue
        rs, cs = os.path.getsize(ref), os.path.getsize(cur)
        if rs != cs:
            results.append((f, "SIZE", rs, cs))
            continue
        if _md5(ref) == _md5(cur):
            results.append((f, "ok", rs, cs))
            continue
        # md5 differs — distinguish a numerically-identical signed-zero (-0.0 vs
        # +0.0) difference (benign, value-equal) from a real value difference.
        status = _classify_diff(ref, cur)
        results.append((f, status, rs, cs))
    return results


def _classify_diff(ref, cur):
    """Return ``'~0'`` if two float32 files differ only in signed zeros
    (numerically identical), else ``'DIFF'``."""
    a = np.fromfile(ref, dtype="<f4")
    b = np.fromfile(cur, dtype="<f4")
    if a.size != b.size:
        return "DIFF"
    ne = a != b               # NaNs aside, -0.0 == 0.0 here -> not flagged
    if not ne.any():
        # values equal but bytes differ -> signed zeros only
        return "~0"
    return "DIFF"


_LIGHT_KINDS = {"inward_windowed", "inward_focalbox", "subset_focalbox"}


def verify_dirs(sim_dir, out_root, dirs=None, full=False):
    scratch = os.path.join(out_root, "_verify_scratch")
    os.makedirs(scratch, exist_ok=True)
    dirs = dirs or list(REGISTRY)
    rc = 0
    for subdir in dirs:
        if subdir not in REGISTRY:
            print(f"[skip] {subdir}: not in registry")
            continue
        # inward dirs verify in light mode by default (maps proven via outward);
        # pass full=True to also regenerate + compare the multi-GB maps.
        light = (not full) and REGISTRY[subdir]["kind"] in _LIGHT_KINDS
        print(f"\n=== {subdir} ==={'  [light: icmat/coords/scalars]' if light else ''}")
        try:
            results = verify_one(sim_dir, scratch, subdir, light=light)
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            print(f"[ERROR] {subdir}: {e}")
            rc = 1
            continue
        nok = sum(1 for _, s, *_ in results if s == "ok")
        nz = sum(1 for _, s, *_ in results if s == "~0")
        for f, status, rs, cs in results:
            if status == "ok":
                print(f"  ok   {f}  ({rs} bytes)")
            elif status == "~0":
                print(f"  ~0   {f}  ({rs} bytes; numerically identical, signed-zero bytes only)")
            elif status == "no-ref":
                print(f"  ---  {f}  (no reference)")
            else:
                print(f"  {status:4s} {f}  ref={rs} cur={cs}")
                rc = 1
        ndiff = sum(1 for _, s, *_ in results if s in ("DIFF", "SIZE"))
        extra = f", {nz} value-identical (signed-zero only)" if nz else ""
        print(f"  -> {nok} byte-identical{extra}, {ndiff} differing")
    return rc


if __name__ == "__main__":
    import sys
    sim = str(paths.sim_dir())
    out = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    dirs = sys.argv[2:] or None
    raise SystemExit(verify_dirs(sim, out, dirs))
