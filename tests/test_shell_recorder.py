"""Shell recorder (``recorder='shell'``): record the field only on the calvarial surface via
the ``genout`` coordinate recorder, instead of the whole decimated volume (``genout_mod``).

CPU-only, with a SYNTHETIC ``genout.dat`` (no GPU/solver): checks that a shell run writes no
``modX/Y/Z`` (so no ``genout_mod``), persists the launch-time surface, and that ``extract`` +
``compute_transparency_map`` consume the surface time-series correctly. The absolute-number
parity vs the volume recorder was confirmed on a GPU run -- agreement tightens with resolution
(3 PPW Pmax r~0.71 -> 6 PPW r~0.88, ratio ~0.97; the shell is full-res, the volume path is
mod=2-decimated), so this gate covers the format/logic, not GPU numbers."""
import json
from pathlib import Path

import numpy as np

import skull_transparency as st
from skull_transparency import TransducerSpec
from skull_transparency.sim.prepare import build_run_from_medium
from skull_transparency.sim import launchers as L
from skull_transparency.sim.extract import extract_bundle
from skull_transparency.sim._common import surface_recorders


def _producer_sim(tmp):
    """Tiny producer sim tree (bone sphere in water)."""
    n = 60
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    r = np.sqrt((ii - 30) ** 2 + (jj - 30) ** 2 + (kk - 30) ** 2)
    c = np.where(r < 16.0, 2900.0, 1540.0).astype(np.float32)
    spec = TransducerSpec(f0_hz=250e3, geometry="bowl", roc_mm=10.0, aperture_mm=8.0, ppw=2.0)
    sim = tmp / "sim"
    build_run_from_medium(c, np.eye(4), np.array([30.0, 30.0, 30.0]), spec, sim,
                          approach=np.array([0.0, 0.0, 1.0]), standoff_mm=5.0, surround_mm=20.0)
    return sim


def test_surface_recorders_layout():
    pts = np.array([[1, 2, 3], [4, 5, 6]], float)
    oc = surface_recorders(pts)
    assert oc.shape == (2, 5)
    assert np.array_equal(oc[:, :3], pts)
    assert np.all(oc[:, 3] == -2) and np.all(oc[:, 4] == 3)      # shell marker / type


def test_shell_launch_skips_modXYZ_and_persists_surface(tmp_path):
    sim = _producer_sim(tmp_path)
    out = Path(L.launch_outward(sim, tmp_path / "run", run_solver=False, recorder="shell"))
    assert (out / "modT.dat").exists()                          # temporal decimation still set
    assert not (out / "modX.dat").exists()                      # -> solver writes NO genout_mod
    ws = np.load(out / "workspace.npz", allow_pickle=True)
    assert str(ws["x_recorder"]) == "shell"
    surf = ws["x_surf_vox"]
    assert surf.ndim == 2 and surf.shape[1] == 3 and surf.shape[0] > 0
    # outc.dat carries n_array + M recorder rows
    n_array = int(ws["x_n_array"])
    meta = json.loads((sim / "meta.json").read_text())
    assert n_array == int(meta["n_array"])


def test_shell_extract_and_transparency(tmp_path):
    sim = _producer_sim(tmp_path)
    run = Path(L.launch_outward(sim, tmp_path / "run", run_solver=False, recorder="shell"))
    ws = np.load(run / "workspace.npz", allow_pickle=True)
    n_array = int(ws["x_n_array"]); M = int(ws["x_surf_vox"].shape[0])
    ncoordsout = n_array + M

    # synthetic solved genout: 4 frames; shell channels carry a known per-patch, per-frame ramp
    nframes = 4
    shell_vals = (100.0 + np.arange(M)[None, :]) * (1.0 + 0.1 * np.arange(nframes)[:, None])
    G = np.zeros((nframes, ncoordsout), np.float32)
    G[:, n_array:n_array + M] = shell_vals
    G.astype("<f4").tofile(run / "genout.dat")

    out = extract_bundle(run, tmp_path / "bundle", sim)
    assert not (out / "outward_Pmax.npy").exists()              # no volume products for a shell bundle
    d = np.load(out / "surface_field.npz")
    assert np.allclose(d["Iint"], (shell_vals.astype(np.float64) ** 2).sum(0), rtol=1e-4)   # sum p^2
    assert np.allclose(d["Pmax"], np.abs(shell_vals).max(0), rtol=1e-4)                     # max |p|

    b = st.load_bundle(out)
    assert b.surface_field() is not None                        # shell bundle recognised
    tmap = st.compute_transparency_map(b)
    assert len(tmap.surf_vox) == M
    assert np.array_equal(tmap.surf_vox, ws["x_surf_vox"])      # uses the persisted launch-time surface
    assert np.all(np.isfinite(tmap.value)) and tmap.value.shape == (M,)
