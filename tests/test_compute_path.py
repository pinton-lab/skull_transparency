"""Phase-2 units: solver resolution/license gate, ITRUSST medium cache, compute CLI.
The full GPU compute path is exercised out-of-band (needs solver + GPU + network)."""
import numpy as np
import pytest

from skull_transparency import cli, itrusst, solver_fetch


def test_pitch_math():
    assert itrusst.pitch_mm(6) == 0.5            # 1500 / (500 kHz * 6) = 0.5 mm
    assert itrusst.pitch_mm(3) == 1.0


def test_build_medium_uses_cache(tmp_path, monkeypatch):
    # a pre-seeded cache is returned as-is: no trimesh, no network
    monkeypatch.setenv("SKULL_TRANSPARENCY_CACHE", str(tmp_path))
    d = tmp_path / "itrusst"
    d.mkdir()
    c = np.full((4, 4, 4), 1500.0, np.float32)
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    np.save(d / "c_0.5mm.npy", c)
    np.save(d / "affine_0.5mm.npy", aff)
    c2, a2 = itrusst.build_medium(ppw=6, verbose=False)
    assert np.array_equal(c2, c) and np.array_equal(a2, aff)


def test_ensure_solver_prefers_configured(tmp_path, monkeypatch):
    fake = tmp_path / "bench_3d_opt"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("FULLWAVE2_BIN", str(fake))
    assert solver_fetch.ensure_solver(interactive=False) == str(fake)


def test_license_gate_refuses_noninteractively(tmp_path, monkeypatch):
    lic = tmp_path / "LICENSE-binaries.txt"
    lic.write_text("Some Noncommercial License v1\nterms...\n")
    monkeypatch.delenv(solver_fetch.ACCEPT_ENV, raising=False)
    assert solver_fetch._license_gate(lic, accept_license=False, interactive=False) is False
    assert solver_fetch._license_gate(lic, accept_license=True, interactive=False) is True
    monkeypatch.setenv(solver_fetch.ACCEPT_ENV, "1")
    assert solver_fetch._license_gate(lic, accept_license=False, interactive=False) is True


def test_compute_cli_parsing():
    p = cli.build_parser()
    a = p.parse_args(cli._merge_coord_args(
        ["compute", "--target-mm", "-4,24,28", "--name", "myspot", "--accept-license"]))
    assert a.target_mm == "-4,24,28" and a.name == "myspot" and a.accept_license
    with pytest.raises(SystemExit):              # --target and --target-mm are exclusive
        p.parse_args(["compute", "--target", "V1", "--target-mm", "0,0,0"])
    with pytest.raises(SystemExit):              # one of them is required
        p.parse_args(["compute"])
