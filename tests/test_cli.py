"""CLI glue. `prepare` runs end-to-end on a synthetic skull (no GPU, pytest tmp); the
parser/loaders are exercised directly. `place` needs a post-solve bundle and is covered
by the library tests."""
import json

import numpy as np
import pytest

from skull_transparency import cli, TransducerSpec


def _bone_ball_npy(path, n=60, c0=30.0, r=16.0):
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    d = np.sqrt((ii - c0) ** 2 + (jj - c0) ** 2 + (kk - c0) ** 2)
    np.save(path, np.where(d < r, 2900.0, 1540.0).astype(np.float32))


def test_parse_vec():
    assert np.allclose(cli._parse_vec("1,2,3"), [1, 2, 3])
    assert np.allclose(cli._parse_vec("0 0 1"), [0, 0, 1])


def test_load_transducer_preset_and_plain(tmp_path):
    s = cli._load_transducer('{"preset": "ctx500", "f0_hz": 500000, "ppw": 6}')
    assert isinstance(s, TransducerSpec) and s.geometry == "annular"
    p = tmp_path / "t.json"
    p.write_text('{"f0_hz": 1000000, "geometry": "bowl", "roc_mm": 60, "aperture_mm": 30, "ppw": 5.5}')
    s2 = cli._load_transducer(str(p))
    assert s2.geometry == "bowl" and s2.roc_mm == 60


def test_prepare_writes_sim_tree(tmp_path):
    cmap = tmp_path / "c.npy"
    _bone_ball_npy(cmap)
    affine = tmp_path / "A.npy"
    np.save(affine, np.eye(4))
    spec = tmp_path / "t.json"
    spec.write_text('{"f0_hz": 250000, "geometry": "bowl", "roc_mm": 10, "aperture_mm": 8, "ppw": 2}')
    out = tmp_path / "run"

    rc = cli.main(["prepare", "--c-map", str(cmap), "--affine", str(affine),
                   "--target", "30,30,30", "--transducer", str(spec), "--approach", "0,0,1",
                   "--standoff-mm", "5", "--surround-mm", "20", "--out", str(out)])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert (out / "c.f32").exists() and (out / "array_coords.i32").exists()
    assert (out / "registration.json").exists()
    assert meta["n_array"] > 0 and meta["F0"] == 250000


def test_prepare_requires_args():
    with pytest.raises(SystemExit):                       # argparse: missing required options
        cli.main(["prepare"])
