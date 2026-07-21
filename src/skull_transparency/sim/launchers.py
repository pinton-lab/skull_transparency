"""Pure-Python ports of the 11 ``fullwave2_launch_*.m`` time-reversal launchers.

Each function generates the same simulation input ``.dat`` files (bit-identical
to the MATLAB scripts) into ``<out_root>/<subdir>``, reading the medium / array
/ recorded fields from ``sim_dir`` (the legacy ``hemisphere_tr/sim`` tree, used
read-only). The CUDA solver itself is external; pass ``run_solver=True`` to
mirror the MATLAB ``system(...)`` call when the binary is present.

The MATLAB ``workspace.mat`` consumed by the inward launchers is replaced by a
compact ``workspace.npz`` sidecar (see :mod:`._common`); ``box_info.mat`` is
written via :mod:`scipy.io` so the existing analysis code can still read it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
import numpy as np

from . import fwio, _common as C
from .launch_core import launch_core
from .mlcompat import (matlab_round, transmit_pulse, unit_pulse, tukeywin,
                       interp1_linear, ballistic_window, matlab_single_sum)

BINARY = "bench_3d_opt"   # fullwave2-ultra CUDA solver (PUBLIC repo build); genout byte-parity with
#                           the retired fullwave2_3d_Aexp_genout_cuda_aperturegrowth_opt was verified
#                           on dentmanual outward (the 2026-06-19 public build) — re-confirm if rebuilt.
# Last-resort solver bin location: the sibling fullwave2-ultra checkout
# (see pyproject [tool.uv.sources]). Override with $FULLWAVE2_BIN_DIR (or
# $FULLWAVE2_BIN for an explicit binary); preferred path is the package resolver.
_DEFAULT_BIN_DIR = str(Path(__file__).resolve().parents[3].parent / "fullwave2-ultra" / "bin")


# ---- small shared utilities -------------------------------------------------

def _chdir(outdir):
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _write_icmat(icmat):
    """``for ii: fwrite(icmat(ii,:),'float')`` == C-order float32."""
    np.ascontiguousarray(icmat).astype("<f4").tofile("icmat.dat")


def _resolve_solver_binary(name=BINARY):
    """Locate the CUDA solver. Priority:
      1. ``$FULLWAVE2_BIN`` — an explicit binary path (wins everything; e.g. the exact
         legacy binary for an identical re-run);
      2. ``fullwave2_ultra.solver.resolve_binary(name)`` — the PUBLIC distribution channel
         (its ``FW2U_BIN_DIR`` / ``config.BIN_DIR`` = the public ``fullwave2-ultra/bin`` /
         per-tag cache / opt-in download);
      3. ``$FULLWAVE2_BIN_DIR``/``name`` — defaults to the public ``fullwave2-ultra/bin``
         when the package isn't importable.
    """
    explicit = os.environ.get("FULLWAVE2_BIN")
    if explicit:
        return explicit
    try:
        from fullwave2_ultra import solver as _u
        return str(_u.resolve_binary(name))
    except Exception:
        pass
    return os.path.join(os.environ.get("FULLWAVE2_BIN_DIR", _DEFAULT_BIN_DIR), name)


def _maybe_run(outdir, sim_dir, run_solver, gpuid=None):
    """Optionally invoke the external CUDA solver in ``outdir`` (which already holds the
    ``.dat`` inputs). Off by default. The binary is the fullwave2-ultra ``bench_3d_opt``
    (see :func:`_resolve_solver_binary`); when the ``fullwave2_ultra`` package is
    importable its GPU preflight runs first (actionable arch/driver errors). The solver
    runs in place with ``cwd=outdir`` — no copy, and bench_3d_opt needs no special
    ``LD_LIBRARY_PATH`` (it links only libc/libm; CUDA is static)."""
    if not run_solver:
        return None
    binary = _resolve_solver_binary()
    if not os.path.exists(binary):
        raise FileNotFoundError(
            f"solver binary not found: {binary}. Set FULLWAVE2_BIN to a bench_3d_opt path, or "
            "install the solver extra (pip install -e .[solver]) to resolve/fetch it.")
    try:
        from fullwave2_ultra import solver as _u
        _u.preflight(binary, device=int(gpuid) if gpuid is not None else 0)
    except ImportError:
        pass                                  # preflight optional; run without it
    env = dict(os.environ)
    if gpuid is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    status = subprocess.call([os.path.abspath(binary)], cwd=outdir, env=env)
    if status != 0:
        raise RuntimeError(
            f"solver exited {status} (binary {binary}, cwd {outdir}) -- no output was written. "
            "The solver's own error is printed above; common causes: GPU runtime off / not visible, "
            "or CUDA out-of-memory (the whole skull at 6 PPW needs ~14 GB -- use an L4/A100 runtime).")
    with open(os.path.join(outdir, "SUCCESS"), "w") as f:
        f.write("ok\n")
    return status


def _bone_bbox_crop(sim_dir, NB, MARGIN):
    """Whole-skull crop from the 720^3 buffer (the skullonly recipe)."""
    buf = np.fromfile(os.path.join(sim_dir, "halle_c_BUFFER.f32"), dtype="<f4")
    buf = buf.reshape(NB, NB, NB, order="F")
    bone = buf > 1600
    v1 = bone.any(axis=(1, 2))
    v2 = bone.any(axis=(0, 2))
    v3 = bone.any(axis=(0, 1))
    lo0 = np.array([np.argmax(v1), np.argmax(v2), np.argmax(v3)], dtype=np.int64)
    hi0 = np.array([len(v1) - 1 - np.argmax(v1[::-1]),
                    len(v2) - 1 - np.argmax(v2[::-1]),
                    len(v3) - 1 - np.argmax(v3[::-1])], dtype=np.int64)
    clo = np.maximum(0, lo0 - MARGIN)
    chi = np.minimum(NB - 1, hi0 + MARGIN)
    N = chi - clo + 1
    c = buf[clo[0]:chi[0] + 1, clo[1]:chi[1] + 1, clo[2]:chi[2] + 1].astype(np.float64)
    return c, lo0, hi0, clo, chi, N


# ============================================================================
# OUTWARD launchers
# ============================================================================

def launch_outward(sim_dir, out_root, run_solver=False, write_maps=True, attenuation=False, p0=1.0,
                   recorder="volume", surf_bone_threshold=2200.0, surf_probe_vox=2.5,
                   surf_standoff_vox=4.0):
    """fullwave2_launch_halle_hemis_tr_outward.m  ->  <out_root>/outward

    ``recorder`` selects what the outward run records:
      * ``"volume"`` (default) -- the (modX,modY,modZ)-decimated FULL field -> ``genout_mod.dat``
        (bit-for-bit the legacy behaviour; extract reads the whole padded volume).
      * ``"shell"`` -- ONLY the calvarial-surface standoff points (via the ``genout`` coordinate
        recorder). No ``genout_mod`` is written; ``genout.dat`` gains M extra channels holding the
        surface time-series. This is ~150-1000x less data (the transparency map samples only that
        shell), so a 6-PPW whole-skull run fits a laptop / free Colab. ``surf_*`` mirror
        ``TransparencyOptions`` (bone_threshold / surface_probe_vox / standoff_pad_vox); they must
        match the values ``compute_transparency_map`` uses. The launch-time surface is persisted in
        ``workspace.npz`` so ``extract`` aligns the channels exactly."""
    outdir = _chdir(os.path.join(out_root, "outward"))
    meta = C.load_meta(sim_dir)
    N = int(meta["N"]); dX = meta["dX_m"]; c0 = meta["C0"]; f0 = meta["F0"]
    omega0 = 2 * np.pi * f0; cfl = 0.2
    lam = c0 / f0; ppw = lam / dX
    c_file = meta.get("c_file", "halle_c.f32")   # generic producer writes c_file; Halle = halle_c.f32
    c = np.fromfile(os.path.join(sim_dir, c_file), dtype="<f4").reshape(N, N, N, order="F").astype(np.float64)
    rho_file, alpha_file = meta.get("rho_file"), meta.get("alpha_file")
    rho = (np.fromfile(os.path.join(sim_dir, rho_file), dtype="<f4").reshape(N, N, N, order="F").astype(np.float32)
           if rho_file else C.rho_from_c(c))                       # supplied density else rho_from_c
    alpha_map = (np.fromfile(os.path.join(sim_dir, alpha_file), dtype="<f4").reshape(N, N, N, order="F").astype(np.float64)
                 if alpha_file else None)                          # supplied dB/MHz/cm else c-porosity
    use_atten = bool(attenuation or alpha_file or meta.get("attenuation"))
    betaval = float(meta.get("beta", 5.5))                          # user-set nonlinearity
    dent = np.asarray(meta["dent_grid"], dtype=np.float64)
    incoords = C.source_sphere(dent, 3)
    nA = int(meta["n_array"])
    arr, _ = C.array_coords_from_i32(os.path.join(sim_dir, "array_coords.i32"))
    outcoordsA = C.array_outcoords(arr)
    modT, modX, modY, modZ = 8, 2, 2, 2
    shell_extra = None
    if recorder == "shell":
        from ..surface import extract_external_surface
        surf_vox, rhat = extract_external_surface(c, dent, surf_bone_threshold, surf_probe_vox)
        Pout = np.clip(np.rint(surf_vox + surf_standoff_vox * rhat), 0, N - 1)   # standoff -> nearest voxel
        outcoords = np.concatenate([outcoordsA, C.surface_recorders(Pout)], axis=0)
        shell_extra = dict(recorder="shell", n_array=nA, surf_vox=surf_vox, rhat=rhat,
                           surf_bone_threshold=surf_bone_threshold, surf_probe_vox=surf_probe_vox,
                           surf_standoff_vox=surf_standoff_vox)
    else:
        vol = C.volume_recorders(N, modX, modY, modZ)
        outcoords = np.concatenate([outcoordsA, vol], axis=0)
    dmax = np.sqrt(((arr - dent) ** 2).sum(axis=1)).max() * dX
    duration = 1.84 * dmax / c0
    nT = int(matlab_round(duration * c0 / lam * ppw / cfl))
    icvec, nTic = transmit_pulse(nT, duration, omega0, p0)

    cwd = os.getcwd(); os.chdir(outdir)
    try:
        fwio.writeVabs("int", modT, "modT")
        if recorder != "shell":            # modX/Y/Z absent -> solver skips the genout_mod volume dump
            fwio.writeVabs("int", modX, "modX", modY, "modY", modZ, "modZ")
        launch_core(c0, omega0, N * dX, N * dX, N * dX, duration, p0, ppw, cfl,
                    c, rho, incoords, outcoords, nTic, write_maps=write_maps,
                    attenuation=use_atten, alpha_map=alpha_map, betaval=betaval)
        fwio.writeVabs("int", nTic, "nTic")
        fwio.writeVabs("int", 0, "ncoords_add")
        icmat = np.tile(icvec.astype(np.float32), (incoords.shape[0], 1))
        _write_icmat(icmat)
        C.save_workspace("workspace.npz",
                         scalars=dict(c0=c0, omega0=omega0, wX=N * dX, wY=N * dX, wZ=N * dX,
                                      duration=duration, p0=p0, ppw=ppw, cfl=cfl,
                                      dX=dX, dY=dX, dZ=dX, N=N, dent=dent, attenuation=int(use_atten),
                                      beta=betaval,
                                      modT=modT, modX=modX, modY=modY, modZ=modZ, nTic=nTic),
                         incoords=incoords, oc_array=outcoordsA,
                         vol_params=(None if recorder == "shell" else (N, modX, modY, modZ)),
                         medium=dict(kind="halle_c", file=c_file, N=N,
                                     rho_file=rho_file, alpha_file=alpha_file),
                         extra=shell_extra)
    finally:
        os.chdir(cwd)
    _maybe_run(outdir, sim_dir, run_solver)
    return outdir


def _skullonly_common(sim_dir, out_root, subdir, dent_mode, target_vox=None,
                      arrfile=None, MARGIN=16, run_solver=False, gpuid="0",
                      write_meta_extra=None, write_maps=True, attenuation=False, p0=1.0):
    """Shared body for the whole-skull outward launchers (skullonly /
    skullonly_array / skullonly_target / skullonly_target_array)."""
    outdir = _chdir(os.path.join(out_root, subdir))
    meta = C.load_meta(sim_dir)
    dX = meta["dX_m"]; c0 = meta["C0"]; f0 = meta["F0"]
    omega0 = 2 * np.pi * f0; cfl = 0.2
    lam = c0 / f0; ppw = lam / dX
    NB = 720; CEN = np.array([360, 360, 360], dtype=np.int64)
    c, lo0, hi0, clo, chi, N = _bone_bbox_crop(sim_dir, NB, MARGIN)
    rho = C.rho_from_c(c)
    betaval = float(meta.get("beta", 5.5))                         # user-set nonlinearity
    wX, wY, wZ = N[0] * dX, N[1] * dX, N[2] * dX

    if dent_mode == "center":
        dent = (CEN - clo).astype(np.float64)
    elif dent_mode == "target_int":          # skullonly_target: sscanf %d
        dent = np.asarray(target_vox, dtype=np.int64).astype(np.float64)
    elif dent_mode == "target_round":        # skullonly_target_array: round(tv)
        dent = matlab_round(np.asarray(target_vox, dtype=np.float64))
    incoords = C.source_sphere(dent, 3)

    use_array = arrfile is not None or subdir.endswith("array") or dent_mode == "target_round"
    modT, modX, modY, modZ = 8, 2, 2, 2
    if arrfile is not None:
        arr, nA = C.array_coords_from_i32(arrfile)
        outcoords = C.array_outcoords(arr)
        dmax = np.sqrt(((arr - dent) ** 2).sum(axis=1)).max() * dX
        vol_params = None; oc_array = outcoords
    else:
        vol = C.volume_recorders(N, modX, modY, modZ)
        outcoords = vol
        # farthest corner distance
        if dent_mode == "center":
            cx = np.array([lo0[0], hi0[0]]) - clo[0]
            cy = np.array([lo0[1], hi0[1]]) - clo[1]
            cz = np.array([lo0[2], hi0[2]]) - clo[2]
        else:
            cx = np.array([0, N[0] - 1]); cy = np.array([0, N[1] - 1]); cz = np.array([0, N[2] - 1])
        CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
        corners = np.stack([CX.ravel(order="F"), CY.ravel(order="F"), CZ.ravel(order="F")], axis=1).astype(np.float64)
        dmax = np.sqrt(((corners - dent) ** 2).sum(axis=1)).max() * dX
        vol_params = (N, modX, modY, modZ); oc_array = None
    duration = 1.84 * dmax / c0
    nT = int(matlab_round(duration * c0 / lam * ppw / cfl))
    icvec, nTic = transmit_pulse(nT, duration, omega0, p0)

    cwd = os.getcwd(); os.chdir(outdir)
    try:
        fwio.writeVabs("int", modT, "modT", modX, "modX", modY, "modY", modZ, "modZ")
        launch_core(c0, omega0, wX, wY, wZ, duration, p0, ppw, cfl,
                    c, rho, incoords, outcoords, nTic, write_maps=write_maps, attenuation=attenuation,
                    betaval=betaval)
        fwio.writeVabs("int", nTic, "nTic")
        fwio.writeVabs("int", 0, "ncoords_add")
        icmat = np.tile(icvec.astype(np.float32), (incoords.shape[0], 1))
        _write_icmat(icmat)
        C.save_workspace("workspace.npz",
                         scalars=dict(c0=c0, omega0=omega0, wX=wX, wY=wY, wZ=wZ,
                                      duration=duration, p0=p0, ppw=ppw, cfl=cfl,
                                      dX=dX, dY=dX, dZ=dX, N=N, dent=dent, attenuation=int(attenuation),
                                      beta=betaval, clo=clo, chi=chi, MARGIN=MARGIN,
                                      modT=modT, modX=modX, modY=modY, modZ=modZ, nTic=nTic),
                         incoords=incoords, oc_array=oc_array, vol_params=vol_params,
                         medium=dict(kind="crop", file="halle_c_BUFFER.f32",
                                     NB=NB, clo=clo.tolist(), chi=chi.tolist()))
        if write_meta_extra is not None:
            import json
            meta_out = dict(N=N.tolist(), dent_grid=dent.tolist(),
                            crop_lo=clo.tolist(), crop_hi=chi.tolist(),
                            dX_m=dX, C0=c0, F0=f0, ppw=ppw, cfl=cfl)
            meta_out.update(write_meta_extra)
            with open("meta.json", "w") as f:
                json.dump(meta_out, f)
    finally:
        os.chdir(cwd)
    _maybe_run(outdir, sim_dir, run_solver, gpuid)
    return outdir


def launch_skullonly(sim_dir, out_root, MARGIN=16, run_solver=False, gpuid="0",
                     attenuation=False, p0=1.0):
    """fullwave2_launch_skullonly.m  ->  <out_root>/skullonly"""
    return _skullonly_common(sim_dir, out_root, "skullonly", "center",
                             MARGIN=MARGIN, run_solver=run_solver, gpuid=gpuid,
                             attenuation=attenuation, p0=p0,
                             write_meta_extra=dict(buffer_N=720, source="dentate_point",
                                                   recorders="sparse_volume_mod2"))


def launch_skullonly_array(sim_dir, out_root, MARGIN=16, run_solver=False, gpuid="0",
                           write_maps=True, attenuation=False, p0=1.0):
    """fullwave2_launch_skullonly_array.m  ->  <out_root>/skullonly_array"""
    arrfile = os.path.join(sim_dir, "skullonly_array_coords.i32")
    return _skullonly_common(sim_dir, out_root, "skullonly_array", "center",
                             arrfile=arrfile, MARGIN=MARGIN, run_solver=run_solver,
                             gpuid=gpuid, write_maps=write_maps,
                             attenuation=attenuation, p0=p0,
                             write_meta_extra=dict(source="dentate_point",
                                                   recorders="surface_conformal_array"))


def launch_skullonly_target(sim_dir, out_root, target_vox, outsub, MARGIN=16,
                            run_solver=False, gpuid="0", attenuation=False, p0=1.0):
    """fullwave2_launch_skullonly_target.m  ->  <out_root>/skullonly_<outsub>"""
    return _skullonly_common(sim_dir, out_root, "skullonly_" + outsub, "target_int",
                             target_vox=target_vox, MARGIN=MARGIN,
                             run_solver=run_solver, gpuid=gpuid,
                             attenuation=attenuation, p0=p0,
                             write_meta_extra=dict(target=outsub,
                                                   recorders="sparse_volume_mod2"))


def launch_skullonly_target_array(sim_dir, out_root, target_vox, arrfile, outsub,
                                  MARGIN=16, run_solver=False, gpuid="0",
                                  attenuation=False, p0=1.0):
    """fullwave2_launch_skullonly_target_array.m  ->  <out_root>/<outsub>"""
    return _skullonly_common(sim_dir, out_root, outsub, "target_round",
                             target_vox=target_vox, arrfile=arrfile, MARGIN=MARGIN,
                             run_solver=run_solver, gpuid=gpuid,
                             attenuation=attenuation, p0=p0,
                             write_meta_extra=dict(source="target_point",
                                                   recorders="surface_conformal_array"))


# ============================================================================
# INWARD launchers (time-reversal re-emission)
# ============================================================================

def _load_src(src_dir, sim_dir):
    """Load a source workspace + reconstruct its medium (from ``sim_dir``) and
    full outcoords."""
    ws = C.load_workspace(os.path.join(src_dir, "workspace.npz"))
    s = ws["scalars"]
    cmap, rho, alpha = C.rebuild_medium(sim_dir, ws["medium"])
    oc = C.rebuild_outcoords(ws)
    return ws, s, cmap, rho, alpha, oc


def _src_atten(s):
    """Attenuation flag inherited from a source (outward) workspace; off for legacy runs.
    Ensures a differently-cropped inward run that writes FRESH maps still gets the medium's
    absorption (reused maps already carry it via hardlink)."""
    try:
        return bool(int(s["attenuation"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _src_beta(s):
    """Nonlinearity beta inherited from a source (outward) workspace; 5.5 for legacy runs
    (the historical baked-in default), so fresh-maps inward runs stay consistent with outward."""
    try:
        return float(s["beta"])
    except (KeyError, TypeError, ValueError, IndexError):
        return 5.5


def launch_inward(sim_dir, out_root, run_solver=False, write_maps=True,
                  reuse_maps=True):
    """fullwave2_launch_halle_hemis_tr_inward.m  ->  <out_root>/inward
    (full, un-windowed time reversal of the outward array recording)."""
    src_dir = os.path.abspath(os.path.join(out_root, "outward"))
    ws, s, cmap, rho, alpha, oc = _load_src(src_dir, sim_dir)
    c0 = float(s["c0"]); omega0 = float(s["omega0"]); p0 = float(s["p0"])
    ppw = float(s["ppw"]); cfl = float(s["cfl"]); modT = int(s["modT"])
    wX = float(s["wX"]); wY = float(s["wY"]); wZ = float(s["wZ"])
    modX = int(s["modX"]); modY = int(s["modY"]); modZ = int(s["modZ"])
    duration = float(s["duration"])
    ncoordsout = oc.shape[0]
    idc_ap = np.where(oc[:, 4] == 1)[0]
    gfile = os.path.join(src_dir, "genout.dat")
    nRun = fwio.sizeOfFile(gfile) // 4 // ncoordsout
    g_ap = fwio.readGenoutSlice(gfile, np.arange(nRun), ncoordsout, idc_ap)

    g_tr = np.flipud(g_ap)
    nUp = int(matlab_round(nRun * modT))
    icmat_full = interp1_linear(np.linspace(0, 1, nRun), g_tr, np.linspace(0, 1, nUp))
    pk = np.abs(icmat_full).max()
    if pk > 0:
        icmat_full = icmat_full / pk * np.float32(p0)   # single chain
    tr_icmat = icmat_full.T          # (nA, nUp)
    nTic = tr_icmat.shape[1]
    duration_tr = duration * 1.4
    tr_incoords = oc[idc_ap, :]
    tr_outcoords = oc

    outdir = _chdir(os.path.join(out_root, "inward"))
    cwd = os.getcwd(); os.chdir(outdir)
    try:
        fwio.writeVabs("int", modT, "modT", modX, "modX", modY, "modY", modZ, "modZ")
        launch_core(c0, omega0, wX, wY, wZ, duration_tr, p0, ppw, cfl,
                    cmap, rho, tr_incoords, tr_outcoords, nTic, write_maps=write_maps,
                    reuse_maps_from=src_dir if reuse_maps else None, attenuation=_src_atten(s),
                    alpha_map=alpha, betaval=_src_beta(s))
        fwio.writeVabs("int", nTic, "nTic")
        _write_icmat(tr_icmat)
    finally:
        os.chdir(cwd)
    _maybe_run(outdir, sim_dir, run_solver)
    return outdir


def _inward_windowed(sim_dir, out_root, subdir, focalbox, run_solver=False,
                     write_maps=True, reuse_maps=True):
    """Shared body for the windowed inward launchers (inward_win /
    inward_focalbox). ``focalbox`` selects modX=modY=modZ=1 + focal-box recorder
    and the box_info sidecar."""
    src_dir = os.path.abspath(os.path.join(out_root, "outward"))
    ws, s, cmap, rho, alpha, oc = _load_src(src_dir, sim_dir)
    c0 = float(s["c0"]); omega0 = float(s["omega0"]); p0 = float(s["p0"])
    ppw = float(s["ppw"]); cfl = float(s["cfl"]); modT = int(s["modT"])
    wX = float(s["wX"]); wY = float(s["wY"]); wZ = float(s["wZ"])
    modX = int(s["modX"]); modY = int(s["modY"]); modZ = int(s["modZ"])
    dX = float(s["dX"]); N = int(np.atleast_1d(s["N"])[0])
    dent = np.asarray(s["dent"], dtype=np.float64).ravel()
    dT = dX / c0 * cfl; dt2 = modT * dT; lambda_t = 1.0 / (omega0 / 2 / np.pi)
    ncoordsout = oc.shape[0]
    idc_ap = np.where(oc[:, 4] == 1)[0]; nA = idc_ap.size
    gfile = os.path.join(src_dir, "genout.dat")
    nRun = fwio.sizeOfFile(gfile) // 4 // ncoordsout
    g_ap = fwio.readGenoutSlice(gfile, np.arange(nRun), ncoordsout, idc_ap)

    a, b, L, *_ = ballistic_window(g_ap, nRun, dt2, lambda_t)
    tw = tukeywin(L, 0.4).astype(np.float32).reshape(-1, 1)
    g_crop = (g_ap[a - 1:b, :] * tw)
    g_tr = np.flipud(g_crop)
    nUp = int(matlab_round(L * modT))
    icmat_full = interp1_linear(np.linspace(0, 1, L), g_tr, np.linspace(0, 1, nUp))
    pk = np.abs(icmat_full).max()
    if pk > 0:
        icmat_full = icmat_full / pk * np.float32(p0)   # single chain
    tr_icmat = icmat_full.T
    nTic = tr_icmat.shape[1]
    duration_tr = 1.4 * b * dt2

    tr_incoords = oc[idc_ap, :]
    if focalbox:
        box = C.focal_box(dent, N, 36)
        tr_outcoords = np.concatenate([oc[idc_ap, :], box], axis=0)
        smodX = smodY = smodZ = 1
    else:
        box = None
        tr_outcoords = oc
        smodX, smodY, smodZ = modX, modY, modZ

    outdir = _chdir(os.path.join(out_root, subdir))
    if focalbox:
        from scipy.io import savemat
        savemat(os.path.join(outdir, "box_info.mat"),
                dict(box=box[:, :3], dent=dent.reshape(1, -1), fb=36, dt2=dt2,
                     dX=dX, modT=modT, nA=nA, duration_tr=duration_tr))
    cwd = os.getcwd(); os.chdir(outdir)
    try:
        fwio.writeVabs("int", modT, "modT", smodX, "modX", smodY, "modY", smodZ, "modZ")
        launch_core(c0, omega0, wX, wY, wZ, duration_tr, p0, ppw, cfl,
                    cmap, rho, tr_incoords, tr_outcoords, nTic, write_maps=write_maps,
                    reuse_maps_from=src_dir if reuse_maps else None, attenuation=_src_atten(s),
                    alpha_map=alpha, betaval=_src_beta(s))
        fwio.writeVabs("int", nTic, "nTic")
        fwio.writeVabs("int", 0, "ncoords_add")
        _write_icmat(tr_icmat)
    finally:
        os.chdir(cwd)
    _maybe_run(outdir, sim_dir, run_solver)
    return outdir


def launch_inward_windowed(sim_dir, out_root, run_solver=False, write_maps=True,
                           reuse_maps=True):
    """fullwave2_launch_halle_hemis_tr_inward_windowed.m -> <out_root>/inward_win"""
    return _inward_windowed(sim_dir, out_root, "inward_win", focalbox=False,
                            run_solver=run_solver, write_maps=write_maps,
                            reuse_maps=reuse_maps)


def launch_inward_focalbox(sim_dir, out_root, run_solver=False, write_maps=True,
                           reuse_maps=True):
    """fullwave2_launch_halle_hemis_tr_inward_focalbox.m -> <out_root>/inward_focalbox"""
    return _inward_windowed(sim_dir, out_root, "inward_focalbox", focalbox=True,
                            run_solver=run_solver, write_maps=write_maps,
                            reuse_maps=reuse_maps)


# ============================================================================
# SUBSET focal-box inward launchers (array-vs-single crossover)
# ============================================================================

def _subset_focalbox(sim_dir, out_root, src_dir, tr_dir, selfile, mode, write_nAo,
                     run_solver=False, gpuid="0", write_maps=True, reuse_maps=True):
    """Shared body for the three subset focal-box launchers. ``src_dir`` holds
    the source workspace+genout; results go to ``tr_dir``."""
    src_dir = os.path.abspath(src_dir)
    ws, s, cmap, rho, alpha, oc = _load_src(src_dir, sim_dir)
    c0 = float(s["c0"]); omega0 = float(s["omega0"]); p0 = float(s["p0"])
    ppw = float(s["ppw"]); cfl = float(s["cfl"]); modT = int(s["modT"])
    wX = float(s["wX"]); wY = float(s["wY"]); wZ = float(s["wZ"])
    dX = float(s["dX"]); N = np.atleast_1d(np.asarray(s["N"], dtype=np.int64))
    dent = np.asarray(s["dent"], dtype=np.float64).ravel()
    dT = dX / c0 * cfl; dt2 = modT * dT; lambda_t = 1.0 / (omega0 / 2 / np.pi)
    ncoordsout = oc.shape[0]
    idc_ap = np.where(oc[:, 4] == 1)[0]; nA = idc_ap.size

    sel = np.fromfile(selfile, dtype="<i4").ravel()          # 0-based
    nS = sel.size
    rows = idc_ap[sel]

    p_unit, npn = unit_pulse(dT, omega0)
    E_target = p0 ** 2 * nS * np.sum(p_unit ** 2)

    if mode == "tr":
        gfile = os.path.join(src_dir, "genout.dat")
        nRun = fwio.sizeOfFile(gfile) // 4 // ncoordsout
        g_sel = fwio.readGenoutSlice(gfile, np.arange(nRun), ncoordsout, rows)
        a, b, L, *_ = ballistic_window(g_sel, nRun, dt2, lambda_t)
        tw = tukeywin(L, 0.4).astype(np.float32).reshape(-1, 1)
        g_crop = g_sel[a - 1:b, :] * tw
        g_tr = np.flipud(g_crop)
        nUp = int(matlab_round(L * modT))
        icmat = interp1_linear(np.linspace(0, 1, L), g_tr, np.linspace(0, 1, nUp)).T
        duration = 1.4 * b * dt2
    elif mode == "flat":                       # zero-phase: every element fires in phase (no delay)
        arr_sel = oc[rows, 0:3]
        d_i = np.sqrt(((arr_sel - dent) ** 2).sum(axis=1)) * dX
        icmat = np.tile(p_unit, (nS, 1)) * p0   # (nS, npn), all rows == the unit pulse, no shift
        duration = 1.4 * d_i.max() / c0
    else:  # geo
        arr_sel = oc[rows, 0:3]
        d_i = np.sqrt(((arr_sel - dent) ** 2).sum(axis=1)) * dX
        tau_fire = (d_i.max() - d_i) / c0
        shift = matlab_round(tau_fire / dT).astype(np.int64)
        nTic_g = int(shift.max()) + npn
        icmat = np.zeros((nS, nTic_g), dtype=np.float64)
        for ii in range(nS):
            icmat[ii, shift[ii]:shift[ii] + npn] = p_unit
        icmat = icmat * p0
        duration = 1.4 * d_i.max() / c0

    # Energy match. MATLAB ``sum(icmat(:).^2)`` is single for the TR drive
    # (icmat came from single interp1) and double for the geo drive (built from
    # the double unit pulse); the scale is applied in single (scalar cast).
    if mode == "tr":
        Ecur = matlab_single_sum(icmat ** 2)                # sum(icmat(:).^2), single
        sc = np.float32(np.sqrt(E_target / np.float64(Ecur)))
        icmat = (icmat * sc).astype(np.float32)
    else:
        Ecur = np.sum(icmat ** 2)                           # double
        icmat = (icmat * np.sqrt(E_target / Ecur)).astype(np.float32)
    nTic = icmat.shape[1]

    box = C.focal_box(dent, N, 36)
    tr_incoords = oc[rows, :]
    tr_outcoords = np.concatenate([oc[rows, :], box], axis=0)

    outdir = _chdir(tr_dir)
    from scipy.io import savemat
    savemat(os.path.join(outdir, "box_info.mat"),
            dict(box=box[:, :3], dent=dent.reshape(1, -1), fb=36, dt2=dt2, dX=dX,
                 modT=modT, duration=duration, sel=(sel + 1).reshape(1, -1),
                 MODE=mode, N=N.reshape(1, -1)))
    cwd = os.getcwd(); os.chdir(outdir)
    try:
        fwio.writeVabs("int", modT, "modT", 1, "modX", 1, "modY", 1, "modZ")
        launch_core(c0, omega0, wX, wY, wZ, duration, p0, ppw, cfl,
                    cmap, rho, tr_incoords, tr_outcoords, nTic, write_maps=write_maps,
                    reuse_maps_from=src_dir if reuse_maps else None, attenuation=_src_atten(s),
                    alpha_map=alpha, betaval=_src_beta(s))
        fwio.writeVabs("int", nTic, "nTic")
        fwio.writeVabs("int", 0, "ncoords_add")
        if write_nAo:
            fwio.writeVabs("int", nS, "nAo")
        _write_icmat(icmat)
    finally:
        os.chdir(cwd)
    _maybe_run(outdir, None, run_solver, gpuid)
    return outdir


def launch_subset_focalbox(sim_dir, out_root, mode="tr", selfile=None, outsub=None,
                           run_solver=False, gpuid="0", write_maps=True):
    """fullwave2_launch_subset_focalbox.m -> <out_root>/inward_sub_<mode>
    (whole-head outward source; writes nAo)."""
    if selfile is None:
        selfile = os.path.join(sim_dir, "sel_transparency.i32")
    if outsub is None:
        outsub = "inward_sub_" + mode
    return _subset_focalbox(sim_dir, out_root, os.path.join(out_root, "outward"),
                            os.path.join(out_root, outsub), selfile, mode,
                            write_nAo=True, run_solver=run_solver, gpuid=gpuid,
                            write_maps=write_maps)


def launch_skullonly_subset_focalbox(sim_dir, out_root, mode="tr", selfile=None,
                                     outsub=None, run_solver=False, gpuid="0",
                                     write_maps=True):
    """fullwave2_launch_skullonly_subset_focalbox.m
    -> <out_root>/skullonly_array/inward_<mode> (no nAo)."""
    if outsub is None:
        outsub = "inward_" + mode
    src_dir = os.path.join(out_root, "skullonly_array")
    return _subset_focalbox(sim_dir, out_root, src_dir, os.path.join(src_dir, outsub),
                            selfile, mode, write_nAo=False, run_solver=run_solver,
                            gpuid=gpuid, write_maps=write_maps)


def launch_skullonly_target_focalbox(sim_dir, out_root, srcdir, selfile, mode="tr",
                                     outsub=None, run_solver=False, gpuid="0",
                                     write_maps=True):
    """fullwave2_launch_skullonly_target_focalbox.m
    -> <out_root>/<srcdir>/inward_<mode> (no nAo)."""
    if outsub is None:
        outsub = "inward_" + mode
    src_dir = os.path.join(out_root, srcdir)
    return _subset_focalbox(sim_dir, out_root, src_dir, os.path.join(src_dir, outsub),
                            selfile, mode, write_nAo=False, run_solver=run_solver,
                            gpuid=gpuid, write_maps=write_maps)
