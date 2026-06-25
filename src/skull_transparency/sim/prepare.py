"""Producer front-end: ``(c[,rho,alpha] volume + affine + target + TransducerSpec)``
-> a self-contained sim tree that :func:`skull_transparency.sim.launchers.launch_outward`
consumes verbatim (``c.f32`` [+ ``rho.f32``/``alpha.f32``], ``meta.json``,
``array_coords.i32``, ``registration.json``).

Two entry points: :func:`build_run_from_medium` (a *targeted* run -- needs ``target`` +
``approach``) and :func:`build_brain_center_run` (a *brain-center whole-skull* run -- one
omnidirectional source at the brain center, no target/approach; see
:mod:`skull_transparency.brain_center`).

This is the missing ingestion layer: today the launchers read an *already-posed*
Halle grid (``halle_c.f32``) plus a hand-built ``meta.json`` / ``ppw55_transform.npz``.
This module rebuilds those artifacts from a generic medium so a new subject can be
run from their own CT-derived maps.

Status
------
* :func:`write_run_descriptor` is **complete** — it emits ``meta.json`` and a *clean*
  rigid ``registration.json`` straight from the pose (no legacy anisotropic
  ``Amn/bmn/dds/scale`` detour; cf. :meth:`Registration.from_ppw55_npz`).
* The heavy steps — :func:`_choose_pose`, :func:`_resample_to_grid`,
  :func:`_recording_surface` (and the small :func:`_write_array_coords_i32`) — are
  **implemented**, so :func:`build_run_from_medium` runs end to end: a generic
  ``(c[,rho,alpha], affine, target, spec)`` yields a full sim tree. The only
  unimplemented path is ``approach='auto'`` (the outward skull-normal aim) in
  :func:`_choose_pose`, which raises ``NotImplementedError``; pass an explicit
  ``approach`` unit vector (target -> skin) instead.
* :func:`build_brain_center_run` (with :func:`_choose_pose_centered`) is **complete**: it
  seats one omnidirectional source at the brain center in a cube sized to the whole head,
  so it needs no ``target``/``approach`` and sidesteps ``approach='auto'`` entirely.

Frame convention (decided): ``affine`` is a 4x4 voxel-index -> world-mm map (the
medium's own frame; NIfTI ``sform`` -> RAS, or a 4x4 built from NRRD
``space directions``/``origin`` -> usually LPS). ``target_phys_mm`` is in that same
world frame. The physics is frame-agnostic; ``input_frame`` is a provenance label
echoed into ``meta.json`` (and, later, the placement output's ``frame`` key) so the
result round-trips back to the coordinates the user supplied. Anatomical
orientation (for auto-pose / az-el) is read off the sign+permutation of
``affine[:3,:3]`` — no tuba / MNI required on the input path.

Attenuation (decided): supplying ``alpha_map`` auto-enables attenuation and
overrides the c-derived porosity model. ``alpha_units`` defaults to ``'db_mhz_cm'``
(the units the verified ``launch_core._porosity_aexp`` conversion already uses).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from ..brain_center import MNI_BRAIN_COM_MM, brain_center_phys_mm
from ..registration import Registration
from ..transducer_spec import TransducerSpec


@dataclass
class Pose:
    """Rigid seat of the subject anatomy into the cubic ``N^3`` sim grid.

    ``R_phys_to_grid`` is the ORTHONORMAL rotation taking a world-mm displacement to
    a grid-mm displacement (voxel scaling is applied separately via ``dx``). It is
    fed verbatim to :attr:`Registration.R_mni_to_sim`, whose ``/dx`` converts the
    rotated displacement to voxels — so it must be a pure rotation, not mm->voxel.
    """
    R_phys_to_grid: np.ndarray      # (3,3) orthonormal, world-mm -> grid-mm
    target_grid_vox: np.ndarray     # (3,) target voxel index (== meta['dent_grid'])
    N: int                          # cubic grid size (the 48-voxel PAD is added later
    #                                 by launch_core, not here)
    target_phys_mm: np.ndarray | None = None   # (3,) world-frame anchor (== Registration
    #   target_mni_mm). With R + target_grid_vox + dx this fully defines the world<->grid
    #   map; required by `_resample_to_grid`. `_choose_pose` must set it.


def build_run_from_medium(c_map, affine, target_phys_mm, spec: TransducerSpec, out_sim_dir, *,
                          rho_map=None, alpha_map=None,
                          input_frame: str = "ras_mm", target_voxel=None,
                          approach=None, standoff_mm: float = 20.0,
                          surround_mm: float = 90.0, array_n_elements=None,
                          attenuation: bool = False, alpha_units: str = "db_mhz_cm"):
    """Resample ``c_map`` (and optional ``rho_map``/``alpha_map``) onto a posed cubic
    grid, plant the outward point source at ``target_phys_mm``, synthesise the
    recording surface for ``spec``, and write the sim tree.

    Parameters
    ----------
    c_map : (nx,ny,nz) array — sound speed (m/s) in voxel order.
    affine : (4,4) — voxel-index -> world-mm (see module docstring).
    target_phys_mm : (3,) — target in the same world frame as ``affine``.
    spec : TransducerSpec — sets the grid pitch (``spec.dx_m``) and recording geometry.
    rho_map, alpha_map : optional independent density / attenuation maps. If absent,
        density is derived via ``rho_from_c`` and attenuation via the c-porosity model
        (the current behaviour). Supplying ``alpha_map`` auto-enables attenuation.
    target_voxel : optional (3,) — convenience; if given, ``target_phys_mm`` is taken
        as ``affine @ [*target_voxel, 1]``.
    approach : optional unit vector (target -> skin, world frame) or ``"auto"`` —
        orients the recording shell toward the transducer's reachable windows.
    standoff_mm : acoustic-path headroom kept around the head when sizing the grid.

    Returns the ``out_sim_dir`` path (a tree ready for ``launch_outward``).
    """
    out_sim_dir = Path(out_sim_dir)
    out_sim_dir.mkdir(parents=True, exist_ok=True)
    affine = np.asarray(affine, float)

    if target_voxel is not None:
        target_phys_mm = (affine @ np.array([*target_voxel, 1.0], float))[:3]
    target_phys_mm = np.asarray(target_phys_mm, float)

    use_attenuation = bool(alpha_map is not None or attenuation)

    # 1. grid geometry + pose from the transducer (dx) and the anatomy
    pose = _choose_pose(c_map, affine, target_phys_mm, approach, spec, standoff_mm,
                        surround_mm=surround_mm)

    # 2. resample the medium into the posed grid (Fortran-order f32, like halle_c.f32)
    def _emit(vol, name, background):
        g = _resample_to_grid(vol, affine, pose, spec.dx_m, background=background)
        g.ravel(order="F").tofile(str(out_sim_dir / name))   # already float32 from resample

    _emit(c_map, "c.f32", spec.c0_ms)              # water sound speed outside the head
    if rho_map is not None:
        _emit(rho_map, "rho.f32", 1000.0)          # water density
    if alpha_map is not None:
        _emit(alpha_map, "alpha.f32", 0.0)         # no attenuation outside

    # 3. recording surface (element coords on the skull-facing shell for this device)
    c_grid = np.fromfile(str(out_sim_dir / "c.f32"), dtype="<f4").reshape(
        pose.N, pose.N, pose.N, order="F")
    arr = _recording_surface(c_grid, pose.target_grid_vox, spec, n=array_n_elements)
    _write_array_coords_i32(out_sim_dir / "array_coords.i32", arr)

    # 4. descriptor: meta.json + clean registration.json
    write_run_descriptor(
        out_sim_dir, spec, pose, target_phys_mm, input_frame=input_frame,
        n_array=len(arr), c_file="c.f32",
        rho_file="rho.f32" if rho_map is not None else None,
        alpha_file="alpha.f32" if alpha_map is not None else None,
        attenuation=use_attenuation, alpha_units=alpha_units)
    return out_sim_dir


def build_brain_center_run(c_map, affine, spec: TransducerSpec, out_sim_dir, *,
                           rho_map=None, alpha_map=None, input_frame: str = "ras_mm",
                           bone_threshold: float = 2200.0, atlas_com_mm=MNI_BRAIN_COM_MM,
                           center_phys_mm=None, surround_mm: float = 25.0,
                           array_n_elements=None, attenuation: bool = False,
                           alpha_units: str = "db_mhz_cm"):
    """Brain-center variant of :func:`build_run_from_medium`: seat ONE omnidirectional
    point source at the brain center and size a cube that holds the WHOLE head, so a
    single outward solve illuminates the entire calvaria for a neutral whole-skull
    transparency map (the generalised, deprecated ``skullonly`` "center" launcher).

    The center is the atlas brain CoM when the world frame is MNI, else the image-only
    intracranial centroid (see :mod:`skull_transparency.brain_center`); pass
    ``center_phys_mm`` to override. No ``--approach`` is needed (the source radiates in
    every direction), so this also sidesteps the unimplemented ``approach='auto'`` path.
    The recording shell spans the whole skull (no window cap), so the outward record
    length covers the farthest bone. Returns the ``out_sim_dir`` path."""
    out_sim_dir = Path(out_sim_dir)
    out_sim_dir.mkdir(parents=True, exist_ok=True)
    affine = np.asarray(affine, float)
    if center_phys_mm is None:
        center_phys_mm = brain_center_phys_mm(c_map, affine, world_frame=input_frame,
                                              bone_threshold=bone_threshold, atlas_com_mm=atlas_com_mm)
    center_phys_mm = np.asarray(center_phys_mm, float)
    use_attenuation = bool(alpha_map is not None or attenuation)

    pose = _choose_pose_centered(c_map, affine, center_phys_mm, spec,
                                 surround_mm=surround_mm, bone_threshold=bone_threshold)

    def _emit(vol, name, background):
        g = _resample_to_grid(vol, affine, pose, spec.dx_m, background=background)
        g.ravel(order="F").tofile(str(out_sim_dir / name))

    _emit(c_map, "c.f32", spec.c0_ms)
    if rho_map is not None:
        _emit(rho_map, "rho.f32", 1000.0)
    if alpha_map is not None:
        _emit(alpha_map, "alpha.f32", 0.0)

    c_grid = np.fromfile(str(out_sim_dir / "c.f32"), dtype="<f4").reshape(
        pose.N, pose.N, pose.N, order="F")
    # whole-skull recording shell (no window cap): max_angle_deg=180 keeps every patch,
    # so launch_outward's record length spans the farthest bone. The bone_threshold flows
    # through so a c-map whose bone is slower than 2200 m/s still resolves its calvarial
    # surface here exactly as the pose sizing and the transparency map do.
    arr = _recording_surface(c_grid, pose.target_grid_vox, spec, n=array_n_elements,
                             max_angle_deg=180.0, bone_threshold=bone_threshold)
    _write_array_coords_i32(out_sim_dir / "array_coords.i32", arr)

    write_run_descriptor(
        out_sim_dir, spec, pose, center_phys_mm, input_frame=input_frame,
        n_array=len(arr), c_file="c.f32",
        rho_file="rho.f32" if rho_map is not None else None,
        alpha_file="alpha.f32" if alpha_map is not None else None,
        attenuation=use_attenuation, alpha_units=alpha_units, subject_id=out_sim_dir.name)
    return out_sim_dir


def write_run_descriptor(sim_dir, spec: TransducerSpec, pose: Pose, target_phys_mm, *,
                         input_frame: str, n_array: int, c_file: str = "c.f32",
                         rho_file=None, alpha_file=None,
                         attenuation: bool = False, alpha_units: str = "db_mhz_cm",
                         array_center=None, subject_id=None) -> dict:
    """Write ``meta.json`` + ``registration.json`` for a posed run. Complete and
    self-contained (no dependency on the scaffolded resample/pose steps), so it is
    independently testable. Returns the ``meta`` dict.

    The registration is the *clean* rigid map built directly from the pose — it
    deliberately skips the legacy anisotropic ``Amn/bmn`` affine that
    :meth:`Registration.from_ppw55_npz` had to repair for the Halle bundle.
    """
    sim_dir = Path(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)
    N = int(pose.N)
    tgt_vox = np.asarray(pose.target_grid_vox, float)

    meta = {
        "N": N,
        # grid + physics + transducer block come straight from the spec
        **spec.to_meta_fields(),
        "dent_grid": [float(x) for x in tgt_vox],          # outward source @ target
        "n_array": int(n_array),
        "array_center": list(array_center) if array_center is not None else [N / 2.0] * 3,
        "c_file": c_file,                                  # launchers read this, not "halle_c.f32"
        "rho_file": rho_file,                              # None -> rho_from_c(c)
        "alpha_file": alpha_file,                          # None -> c-porosity model
        "attenuation": bool(attenuation or alpha_file is not None),
        "alpha_units": alpha_units,
        "input_frame": input_frame,                        # provenance; output round-trips to it
        "subject_id": subject_id or sim_dir.name,
    }
    (sim_dir / "meta.json").write_text(json.dumps(meta, indent=1))

    reg = Registration(
        R_mni_to_sim=np.asarray(pose.R_phys_to_grid, float),   # world-mm -> grid-mm (orthonormal)
        dx_mm=float(spec.dx_mm),
        target_mni_mm=np.asarray(target_phys_mm, float),
        target_fullres_voxel=tgt_vox,
        world_frame=input_frame,                               # the TRUE world frame (not MNI)
    )
    # world_frame carries meta['input_frame'] through registration.json, so the placement
    # output (neuromod.to_placement_dict) reports coordinates in the user's frame and does NOT
    # mis-map a non-MNI subject through tuba's MNI->Halle chain.
    reg.to_json(sim_dir / "registration.json")
    return meta


# ---------------------------------------------------------------------------
# Heavy steps. Implemented for the explicit-approach path; only `approach='auto'`
# (in `_choose_pose`) still raises NotImplementedError. Algorithm in each docstring.
# ---------------------------------------------------------------------------

#: the grid axis the approach aim is mapped to (the transducer fires from +Z toward the
#: target, which is seated near the -Z face); `_recording_surface` honours the same axis.
_APPROACH_AXIS = np.array([0.0, 0.0, 1.0])


def _rotation_aligning(a, b) -> np.ndarray:
    """Minimal proper rotation ``R`` with ``R @ a == b`` (Rodrigues), for unit-ish ``a,b``."""
    a = np.asarray(a, float); a = a / np.linalg.norm(a)
    b = np.asarray(b, float); b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:                       # already aligned
        return np.eye(3)
    if c < -1.0 + 1e-12:                      # antiparallel: 180 deg about any axis _|_ a
        t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0.0, 1, 0])
        axis = np.cross(a, t); axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _choose_pose(c_map, affine, target_phys_mm, approach, spec, standoff_mm, *,
                 surround_mm: float = 90.0, margin_mm: float = 10.0) -> Pose:
    """Choose the rigid seat of the anatomy in the cubic ``N^3`` grid.

    This implements the **explicit-aim** path: ``approach`` is the unit vector from the
    target out to the skin / transducer, in the ``affine`` world-mm frame. It is rotated
    onto the grid ``+Z`` axis (:data:`_APPROACH_AXIS`), so the transducer sits at high Z
    and the target is seated ``surround_mm`` in from the low-Z face. The grid is sized so
    that, along ``+Z``, the focused-bowl reach (``roc + aperture/2 + standoff``) fits, and
    in every other direction ``surround_mm`` of medium around the target is included.

    ``surround_mm`` should exceed the distance from the target to the farthest relevant
    skull window; enlarge it for shallow/peripheral targets. (Tightening ``N`` from the
    actual head bounding box, and the ``approach="auto"`` outward-normal heuristic, are
    deferred — they need the skull surface, i.e. ``_recording_surface``.)

    Returns a fully-populated :class:`Pose` (including ``target_phys_mm``, which
    :func:`_resample_to_grid` requires).
    """
    if approach is None or (isinstance(approach, str) and approach == "auto"):
        raise NotImplementedError(
            "approach='auto' (outward skull-normal aim) not yet implemented; pass an "
            "explicit `approach` unit vector (target -> skin, in the affine world frame).")
    a = np.asarray(approach, float)
    if a.shape != (3,) or not np.isfinite(a).all() or np.linalg.norm(a) == 0:
        raise ValueError("approach must be a finite, nonzero 3-vector (target -> skin, world mm).")

    R = _rotation_aligning(a, _APPROACH_AXIS)             # world-mm -> grid-mm; approach -> +Z
    dx_mm = float(spec.dx_m) * 1e3
    reach_mm = (spec.roc_mm or 0.0) + (spec.aperture_mm or 0.0) / 2.0 + standoff_mm + margin_mm
    surround_vox = int(np.ceil(surround_mm / dx_mm))
    approach_vox = int(np.ceil(reach_mm / dx_mm))
    N = int(max(approach_vox + surround_vox, 2 * surround_vox))
    # centred laterally; seated `surround_vox` from the low-Z face so +Z holds the bowl reach
    target_grid_vox = np.array([N / 2.0, N / 2.0, float(surround_vox)])
    return Pose(R_phys_to_grid=R, target_grid_vox=target_grid_vox, N=N,
                target_phys_mm=np.asarray(target_phys_mm, float))


def _choose_pose_centered(c_map, affine, center_phys_mm, spec, *,
                          surround_mm: float = 25.0, bone_threshold: float = 2200.0) -> Pose:
    """Centred, OMNIDIRECTIONAL seat for a brain-center run: the source sits at the cube
    center and the cube is sized to hold the whole head plus ``surround_mm`` of coupling
    medium on every side. No approach axis (the wave radiates in all directions), so the
    rotation is identity (grid axes = world axes); the resample handles the affine.

    The half-extent is read off the bone bounding-box corners mapped to world mm, so an
    anisotropic head still gets a cube large enough on its longest axis."""
    c = np.asarray(c_map)
    affine = np.asarray(affine, float)
    center = np.asarray(center_phys_mm, float)
    dx_mm = float(spec.dx_m) * 1e3
    bone = c > bone_threshold
    if not bone.any():
        raise ValueError(f"no bone voxels (c > {bone_threshold}); check the speed-map units/threshold.")
    idx = np.argwhere(bone)
    lo, hi = idx.min(0), idx.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])], float)
    cw = (affine @ np.c_[corners, np.ones(len(corners))].T).T[:, :3]   # bbox corners -> world mm
    half = np.abs(cw - center).max(0)                                  # per-axis half-extent from center
    reach_vox = int(np.ceil((half.max() + surround_mm) / dx_mm))
    N = int(2 * reach_vox)                                             # even cube, source at the center
    target_grid_vox = np.array([N / 2.0, N / 2.0, N / 2.0])
    return Pose(R_phys_to_grid=np.eye(3), target_grid_vox=target_grid_vox, N=N,
                target_phys_mm=center)


def _resample_to_grid(vol, affine, pose: Pose, dx_m, *, background=0.0, order=1) -> np.ndarray:
    """Trilinearly resample ``vol`` (in the ``affine`` frame) onto the posed ``N^3`` grid.

    Per output voxel ``g`` the world-mm point is
    ``p = target_phys_mm + dx_mm * R^T @ (g - target_grid_vox)`` and the input voxel is
    ``inv(affine) @ [p, 1]``. Both maps are affine, so their composition is one affine
    ``in_vox = matrix @ g + offset`` — evaluated by ``scipy.ndimage.affine_transform``,
    which never materialises the (potentially ~10^8-point) coordinate arrays. Points
    outside ``vol`` are filled with ``background`` (c0 for c, 1000 for rho, 0 for alpha).
    Returns an ``(N,N,N)`` ``float32`` array (the caller writes it Fortran-order).
    """
    if pose.target_phys_mm is None:
        raise ValueError("pose.target_phys_mm must be set to resample "
                         "(the world-frame anchor; _choose_pose sets it).")
    vol = np.asarray(vol, np.float32)
    affine = np.asarray(affine, float)
    N = int(pose.N)
    dx_mm = float(dx_m) * 1e3
    Rt = np.asarray(pose.R_phys_to_grid, float).T          # grid-mm -> world-mm
    tvox = np.asarray(pose.target_grid_vox, float)
    tmm = np.asarray(pose.target_phys_mm, float)
    inv3 = np.linalg.inv(affine[:3, :3])                   # world-mm -> input-voxel (linear)
    origin = affine[:3, 3]

    # in_vox = inv3 @ (tmm + dx_mm*Rt@(g - tvox) - origin) = matrix @ g + offset
    matrix = inv3 @ (dx_mm * Rt)
    offset = inv3 @ (tmm - dx_mm * (Rt @ tvox) - origin)
    out = affine_transform(vol, matrix, offset=offset, output_shape=(N, N, N),
                           order=order, mode="constant", cval=float(background), prefilter=False)
    return out.astype(np.float32)


def _grid_subsample(pts, spacing) -> np.ndarray:
    """Keep one point per ``spacing``-sized voxel bin (first in original order)."""
    if spacing <= 1.0 or len(pts) == 0:
        return pts
    keys = np.floor(np.asarray(pts) / spacing).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return pts[np.sort(first)]


def _recording_surface(c_grid, target_grid_vox, spec: TransducerSpec, n=None, *,
                       max_angle_deg: float = 60.0, recorder_offset_vox: float = 1.5,
                       bone_threshold: float = 2200.0) -> np.ndarray:
    """Recorder-element voxel coords on the approach-facing outer skull shell.

    By reciprocity these are the candidate transducer-element positions whose recorded
    outward field yields the per-element coupling (the phase path; the dense transparency
    map itself comes from the volume recorders). Steps:

      1. extract the outer skull surface from ``c_grid`` — the *same* definition the
         transparency map uses (:func:`skull_transparency.surface.extract_external_surface`);
      2. keep the patches facing the transducer: radial direction (target -> patch) within
         ``max_angle_deg`` of the grid +Z approach axis (the candidate-window cap);
      3. push them ``recorder_offset_vox`` outward into coupling medium (just outside bone);
      4. subsample to ~half-wavelength spacing (``spec.ppw / 2`` voxels), or to ``n`` points.

    Returns an ``(M, 3)`` int voxel-index array in ``[0, N)``. Widen ``max_angle_deg`` to
    let placement explore more windows (at extra recorder/solve cost).
    """
    from ..surface import extract_external_surface
    c_grid = np.asarray(c_grid)
    N = c_grid.shape[0]
    tvox = np.asarray(target_grid_vox, float)

    surf, rhat = extract_external_surface(c_grid, tvox, bone_threshold=bone_threshold)
    if len(surf) == 0:
        raise ValueError(f"no outer skull surface found (no voxels with c > {bone_threshold}); "
                         "check the medium units/threshold or that the head is inside the grid.")
    face = rhat @ _APPROACH_AXIS >= np.cos(np.deg2rad(max_angle_deg))
    surf, rhat = surf[face], rhat[face]
    if len(surf) == 0:
        raise ValueError(f"no skull surface within {max_angle_deg} deg of the approach axis; "
                         "check `approach` points target->skin, or widen max_angle_deg.")

    pts = np.rint(surf + recorder_offset_vox * rhat).astype(np.int64)   # just outside bone
    np.clip(pts, 0, N - 1, out=pts)
    pts = np.unique(pts, axis=0)
    pts = _grid_subsample(pts, max(1.0, float(spec.ppw) / 2.0))
    if n is not None and len(pts) > n:
        keep = np.unique(np.linspace(0, len(pts) - 1, n).round().astype(int))
        pts = pts[keep]
    return pts


def _write_array_coords_i32(path, arr) -> None:
    """Write element voxel coords as the ``array_coords.i32`` layout that
    :func:`skull_transparency.sim._common.array_coords_from_i32` reads back
    (int32, C-order ravel of the ``(M, 3)`` matrix -> interleaved x,y,z)."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"array coords must be (M,3), got shape {a.shape}")
    np.ascontiguousarray(a.astype("<i4")).tofile(str(path))
