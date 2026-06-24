"""CTX-500 focused-bowl transducer geometry (reusable).

Factored out of ``runs/rebuild_6ppw_graded/scripts/build_ctx500_caps_6ppw_graded.py``
(``cap_dirs``) and ``render_position_optimizer_3cases.py`` (``cap_surface``) so the
spherical-cap point cloud + dish triangulation can be reused by the interactive
positioning tool and any future placement script.

Geometry (physical CTX-500 / NeuroFUS, the device built here):
  * radius of curvature  ROC = 63.2 mm
  * aperture diameter         = 64 mm  -> half-angle = arcsin(32/63.2) = 30.4 deg

A placement is fully specified by (apex, aim):
  * ``apex`` = cap face-centre point (on the acoustic axis, the closest point of the
    bowl to nothing -- it is the pole of the cap).
  * ``aim``  = unit vector from the apex INTO the head (the acoustic axis, apex->focus).
The geometric focus (centre of curvature) is then ``C = apex + ROC*aim`` and the cap
surface is ``P = C + ROC * dirs`` for unit directions ``dirs`` within the half-angle of
the OUTWARD axis ``-aim`` (matches the build-script convention exactly).

All lengths here are in the caller's units; pass ``roc`` already converted to that unit
(e.g. ROC_MM/dx_mm for voxel coordinates, or ROC_MM for millimetres)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROC_MM = 63.2
APERTURE_MM = 64.0
HALF_ANGLE_DEG = float(np.degrees(np.arcsin((APERTURE_MM / 2.0) / ROC_MM)))  # 30.4 deg


@dataclass(frozen=True)
class CTX500:
    """CTX-500 bowl parameters (mm)."""
    roc_mm: float = ROC_MM
    aperture_mm: float = APERTURE_MM

    @property
    def half_angle_deg(self) -> float:
        return float(np.degrees(np.arcsin((self.aperture_mm / 2.0) / self.roc_mm)))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-300)


def cap_directions(axis: np.ndarray, roc: float, half_angle_deg: float = HALF_ANGLE_DEG,
                   density: float = 1.0) -> np.ndarray:
    """Unit directions within ``half_angle_deg`` of ``axis`` (the OUTWARD bowl axis), sampled
    on the cap as concentric rings.  ``density=1.0`` reproduces the build-script's
    continuous-surface sampling (~grid density at ``roc`` given in voxels); smaller values
    thin it out for fast interactive display.  Returns (M,3) unit vectors (first row = axis).

    Identical algorithm to build_ctx500_caps_6ppw_graded.cap_dirs, with the ring/azimuth
    counts scaled by ``density`` (and floored so the cap never degenerates)."""
    axis = _unit(axis)
    t0 = np.array([1., 0, 0])
    if abs(axis @ t0) > 0.9:
        t0 = np.array([0., 1, 0])
    e1 = _unit(t0 - (t0 @ axis) * axis)
    e2 = np.cross(axis, e1)
    half = np.deg2rad(half_angle_deg)
    n_theta = max(2, int(np.degrees(half) * roc * np.pi / 180.0 * density))
    dirs = [axis.copy()]
    for th in np.linspace(0.0, half, n_theta):
        if th == 0.0:
            continue
        n_phi = max(6, int(2 * np.pi * roc * np.sin(th) * density))
        for ph in np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False):
            dirs.append(np.cos(th) * axis + np.sin(th) * (np.cos(ph) * e1 + np.sin(ph) * e2))
    return np.asarray(dirs)


def build_cap(apex: np.ndarray, aim: np.ndarray, roc: float,
              half_angle_deg: float = HALF_ANGLE_DEG, density: float = 1.0):
    """Build a CTX-500 cap point cloud from (apex, aim).

    ``apex`` = cap face-centre, ``aim`` = unit axis apex->focus (INTO the head).
    Returns ``(points, focus)``: points (M,3) on the bowl surface, focus = centre of
    curvature = ``apex + roc*aim``.  The cap pole (``dirs[0]``) lands exactly on ``apex``."""
    apex = np.asarray(apex, float)
    aim = _unit(aim)
    focus = apex + roc * aim
    dirs = cap_directions(-aim, roc, half_angle_deg, density)   # cap axis points OUTWARD = -aim
    pts = focus + roc * dirs
    return pts, focus


def triangulate_cap(cap_pts: np.ndarray, prune: float = 2.5):
    """Triangulate a bowl point cloud into a SMOOTH curved surface (for a translucent dish).

    PCA the cap to its best-fit tangent plane, project to 2-D, Delaunay there, then keep those
    faces with the ORIGINAL curved 3-D vertices (preserves bowl curvature).  Prune triangles
    whose longest edge exceeds ``prune``x the median edge (clean rim).  Returns (verts, faces).

    Identical to render_position_optimizer_3cases.cap_surface."""
    from scipy.spatial import Delaunay
    X = cap_pts - cap_pts.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ vt[:2].T
    faces = Delaunay(proj).simplices
    e0 = np.linalg.norm(cap_pts[faces[:, 0]] - cap_pts[faces[:, 1]], axis=1)
    e1 = np.linalg.norm(cap_pts[faces[:, 1]] - cap_pts[faces[:, 2]], axis=1)
    e2 = np.linalg.norm(cap_pts[faces[:, 2]] - cap_pts[faces[:, 0]], axis=1)
    emax = np.maximum.reduce([e0, e1, e2])
    med = np.median(np.concatenate([e0, e1, e2]))
    faces = faces[emax < prune * med]
    return cap_pts, faces


def rodrigues(v: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotate ``v`` about ``axis`` by ``deg`` degrees (Rodrigues). Mirrors render_*.rot."""
    a = np.deg2rad(deg)
    k = _unit(axis)
    v = np.asarray(v, float)
    return v * np.cos(a) + np.cross(k, v) * np.sin(a) + k * (k @ v) * (1 - np.cos(a))


# ---------------------------------------------------------------------------
# Anatomical pose -> cap geometry (apex, aim) around a fixed target.
#
# Factored from ``ctx500_position_tool.Pose`` so the placement objective
# (placement.place_cap_optimal) can build a CTX-500 cap from a 5-DOF pose without the
# interactive napari tool. The transducer face-centre (apex) rides on a sphere of radius
# ``radius_mm`` about the target; by default the bowl aims straight at the target (geometric
# focus on target when radius == ROC), and tilt/yaw rotate the aim off-target. All geometry
# is in the DOMAIN-VOXEL frame, whose anatomical unit axes (from the meta axis_map
# R=-axis0, A=+axis1, S=+axis2) are the module defaults below.
# ---------------------------------------------------------------------------

S_HAT_DOMAIN = np.array([0.0, 0.0, 1.0])   # superior
A_HAT_DOMAIN = np.array([0.0, 1.0, 0.0])   # anterior
L_HAT_DOMAIN = np.array([1.0, 0.0, 0.0])   # anatomical left


@dataclass(frozen=True)
class CapPose:
    """CTX-500 pose around a fixed target (mirrors ctx500_position_tool.Pose).

    ``az_deg``  azimuth of the apex direction (0 = anterior, +90 = anatomical left).
    ``el_deg``  elevation of the apex direction (+ toward superior).
    ``radius_mm`` apex (face-centre) distance from the target (== ROC -> focus on target).
    ``tilt_deg``/``yaw_deg`` rotate the aim off-target (about the beam right / up axes)."""
    az_deg: float
    el_deg: float
    radius_mm: float
    tilt_deg: float = 0.0
    yaw_deg: float = 0.0


def anatomical_az_el(direction_vox: np.ndarray, *, s_hat=S_HAT_DOMAIN, a_hat=A_HAT_DOMAIN,
                     l_hat=L_HAT_DOMAIN) -> tuple[float, float]:
    """(az, el) degrees of a domain-voxel direction, the inverse of :func:`CapPose.dir_hat`.
    Use it to seed a pose-search from a known window direction (target -> window)."""
    d = _unit(direction_vox)
    el = float(np.degrees(np.arcsin(np.clip(d @ s_hat, -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(d @ l_hat, d @ a_hat)))
    return az, el


def pose_apex_aim(target_vox: np.ndarray, pose: CapPose, dx_mm: float, *,
                  s_hat=S_HAT_DOMAIN, a_hat=A_HAT_DOMAIN, l_hat=L_HAT_DOMAIN):
    """Apex (cap face-centre) and aim (unit acoustic axis, apex->target by default; tilt/yaw
    rotate it off-target), in the domain-voxel frame.  Identical math to
    ``ctx500_position_tool.Pose.{dir_hat, apex_vox, aim_hat}``."""
    az, el = np.deg2rad(pose.az_deg), np.deg2rad(pose.el_deg)
    dir_hat = (np.cos(el) * (np.cos(az) * a_hat + np.sin(az) * l_hat) + np.sin(el) * s_hat)
    apex = np.asarray(target_vox, float) + (pose.radius_mm / dx_mm) * dir_hat
    u0 = -dir_hat                                   # apex -> target
    right = np.cross(u0, s_hat)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(u0, a_hat)
    right = right / np.linalg.norm(right)
    up = np.cross(right, u0); up /= np.linalg.norm(up)
    u = rodrigues(u0, right, pose.tilt_deg)
    aim = rodrigues(u, up, pose.yaw_deg)
    return apex, _unit(aim)


def build_cap_pose(target_vox: np.ndarray, pose: CapPose, roc_vox: float, dx_mm: float,
                   half_angle_deg: float = HALF_ANGLE_DEG, density: float = 1.0,
                   *, s_hat=S_HAT_DOMAIN, a_hat=A_HAT_DOMAIN, l_hat=L_HAT_DOMAIN):
    """CTX-500 cap point cloud (domain voxel) + geometric focus for an anatomical pose.

    ``roc_vox`` is the bowl radius of curvature in voxels (ROC_MM/dx_mm); ``pose.radius_mm``
    is the apex standoff (so the geometric focus sits on the target iff radius_mm == ROC and
    tilt == yaw == 0).  Returns ``(points, focus, apex, aim)`` exactly as
    ``ctx500_position_tool.cap_for_pose``."""
    apex, aim = pose_apex_aim(target_vox, pose, dx_mm, s_hat=s_hat, a_hat=a_hat, l_hat=l_hat)
    pts, focus = build_cap(apex, aim, roc_vox, half_angle_deg, density=density)
    return pts, focus, apex, aim
