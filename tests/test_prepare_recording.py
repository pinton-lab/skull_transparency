"""Tests for prepare._recording_surface and _write_array_coords_i32 — controlled
synthetic skull (a solid bone ball in water), no /celerina data, GPU, or tuba."""
import numpy as np
import pytest

from skull_transparency import TransducerSpec
from skull_transparency.sim.prepare import _recording_surface, _write_array_coords_i32
from skull_transparency.sim._common import array_coords_from_i32

SPEC = TransducerSpec.ctx500(f0_hz=500e3, ppw=6.0)


def _bone_ball(N, center, radius, c_bone=2900.0, c_bg=1540.0):
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    r = np.sqrt((ii - center[0]) ** 2 + (jj - center[1]) ** 2 + (kk - center[2]) ** 2)
    return np.where(r < radius, c_bone, c_bg).astype(np.float32)


def test_recorders_form_a_plus_z_cap_on_the_outer_shell():
    N, R = 40, 10.0
    tvox = np.array([20.0, 20.0, 14.0])
    pts = _recording_surface(_bone_ball(N, tvox, R), tvox, SPEC)

    assert pts.ndim == 2 and pts.shape[1] == 3
    assert np.issubdtype(pts.dtype, np.integer)
    assert len(pts) > 0
    assert (pts >= 0).all() and (pts < N).all()                # inside the grid

    d = pts - tvox
    u = d / np.linalg.norm(d, axis=1, keepdims=True)
    assert (u[:, 2] >= np.cos(np.deg2rad(65.0))).all()         # within the +Z cap (60 deg + slack)
    rad = np.linalg.norm(d, axis=1)
    assert (rad >= R - 1.0).all() and (rad <= R + 3.0).all()   # on the outer shell, pushed out


def test_wider_cone_keeps_more_recorders():
    N, R = 40, 10.0
    tvox = np.array([20.0, 20.0, 14.0])
    c = _bone_ball(N, tvox, R)
    narrow = _recording_surface(c, tvox, SPEC, max_angle_deg=30.0)
    wide = _recording_surface(c, tvox, SPEC, max_angle_deg=80.0)
    assert len(wide) >= len(narrow) > 0


def test_target_count_is_capped_by_n():
    N, R = 40, 10.0
    tvox = np.array([20.0, 20.0, 14.0])
    pts = _recording_surface(_bone_ball(N, tvox, R), tvox, SPEC, n=8)
    assert 0 < len(pts) <= 8


def test_no_bone_raises():
    c = np.full((30, 30, 30), 1540.0, np.float32)
    with pytest.raises(ValueError, match="no outer skull surface"):
        _recording_surface(c, np.array([15.0, 15.0, 15.0]), SPEC)


def test_write_array_coords_i32_round_trips():
    import tempfile
    arr = np.array([[1, 2, 3], [10, 20, 30], [7, 8, 9]], dtype=np.int64)
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/array_coords.i32"
        _write_array_coords_i32(p, arr)
        back, nA = array_coords_from_i32(p)
    assert nA == 3
    assert np.array_equal(back.astype(np.int64), arr)


def test_write_array_coords_rejects_bad_shape():
    with pytest.raises(ValueError):
        _write_array_coords_i32("/tmp/unused.i32", np.zeros((4, 2), dtype=np.int64))
