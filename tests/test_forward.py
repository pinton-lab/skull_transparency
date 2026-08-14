"""Tests for the forward (transducer -> target) transcranial vs free-field comparison.

Everything here runs without a GPU: the free-field medium construction, the bowl-source
geometry, the focal-box math (peaks, ratio/dB, focal shift, FWHM) on synthetic boxes, the
dataclass serialisation, and the CLI parsing. The tests that write real solver decks do so
with ``write_maps=False``, so no multi-GB medium map is ever produced.

The end-to-end pair (two real solves of a tiny water-plus-bone-slab grid) needs the CUDA
solver AND ``SKULL_TR_FORWARD_GPU=1`` -- it takes a GPU and a few hundred MB of scratch, so
it is opt-in rather than part of every run::

    SKULL_TR_FORWARD_GPU=1 pytest tests/test_forward.py -k end_to_end
"""
import json
import os
import shutil

import numpy as np
import pytest

from skull_transparency import cli
from skull_transparency import forward as F


# ---- free-field medium -----------------------------------------------------

def test_water_medium_like_is_uniform_water():
    c = np.linspace(1400.0, 2900.0, 2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
    cw, rw, aw = F.water_medium_like(c, 1540.0)
    assert cw.shape == c.shape and rw.shape == c.shape
    assert cw.dtype == np.float64 and rw.dtype == np.float32   # rebuild_medium's contract
    assert np.all(cw == 1540.0)                                # c0 preserved everywhere
    assert np.all(rw == 1000.0)                                # water density (rho_from_c(c0))
    assert aw is None                                          # -> c-porosity model = water's 0.4


def test_water_medium_like_honours_c0_and_explicit_maps():
    c = np.full((3, 3, 3), 2900.0)
    cw, rw, _ = F.water_medium_like(c, 1480.0)
    assert np.all(cw == 1480.0) and np.all(rw == 1000.0)

    # with a supplied absorption map the water twin is uniform at the coupling water's own
    # value (the median over the non-bone voxels), not at the bone's.
    c2 = np.array([[[1540.0, 1540.0, 2900.0, 2900.0]]])
    alpha = np.array([[[0.4, 0.4, 8.0, 9.0]]])
    _, _, aw = F.water_medium_like(c2, 1540.0, alpha_map=alpha)
    assert aw.shape == c2.shape and np.allclose(aw, 0.4)
    _, _, aw2 = F.water_medium_like(c2, 1540.0, alpha_map=alpha, alpha_water=0.25)
    assert np.allclose(aw2, 0.25)


def test_water_medium_like_rho_matches_prepare_background_with_rho_map():
    # a run that carries its own density map has water filled at 1000 kg/m^3 by prepare,
    # so the free-field twin must use that same value.
    c = np.full((2, 2, 2), 2900.0)
    _, rw, _ = F.water_medium_like(c, 1540.0, rho_map=np.full((2, 2, 2), 2200.0))
    assert np.all(rw == 1000.0)


# ---- bowl source geometry --------------------------------------------------

def test_bowl_source_geometry_and_bounds():
    target = np.array([40.0, 40.0, 10.0])
    dx_mm = 1.0
    apex = target + np.array([0.0, 0.0, 30.0])         # one ROC out along +Z
    s = F.bowl_source(target, apex, dx_mm, roc_mm=30.0, aperture_mm=24.0,
                      grid_shape=(80, 80, 80), density=0.5)
    assert s.n_points > 10
    assert np.allclose(s.aim, [0, 0, -1])              # apex -> target
    assert s.standoff_mm == pytest.approx(30.0)
    assert s.focus_to_target_mm == pytest.approx(0.0, abs=1e-9)   # standoff == ROC
    # every face voxel sits one ROC from the geometric focus, within the rounding to voxels
    r = np.linalg.norm(s.points_vox - s.focus_vox, axis=1)
    assert np.allclose(r, 30.0 / dx_mm, atol=1.0)
    assert json.dumps(s.to_dict())                     # serialisable

    # a bowl that pokes out of the grid is refused, not silently clipped into a dented dish
    with pytest.raises(ValueError, match="outside the interior grid"):
        F.bowl_source(target, apex, dx_mm, roc_mm=30.0, aperture_mm=24.0,
                      grid_shape=(20, 20, 20))


def test_bowl_source_from_placement_defaults_to_approach_axis(tmp_path):
    _write_sim_tree(tmp_path, n=64, dx_m=1e-3, target=(32, 32, 12))
    s = F.bowl_source_from_placement(tmp_path, roc_mm=20.0, aperture_mm=16.0, density=0.5)
    # default seat: one ROC out along grid +Z (the axis `prepare` rotates --approach onto)
    assert np.allclose(s.target_vox, [32, 32, 12])
    assert np.allclose(s.apex_vox, [32, 32, 32])
    assert s.roc_mm == 20.0 and s.aperture_mm == 16.0


def test_bowl_source_from_placement_reads_placement_json(tmp_path):
    _write_sim_tree(tmp_path, n=64, dx_m=1e-3, target=(32, 32, 12))
    # world frame == voxel frame here (identity registration anchored at the target)
    pl = {"xdc_center_lps": [32.0, 32.0, 32.0], "target_lps": [32.0, 32.0, 12.0],
          "frame": "ras_mm", "apex_to_target_mm": 20.0}
    p = tmp_path / "placement.json"
    p.write_text(json.dumps(pl))
    s = F.bowl_source_from_placement(tmp_path, p, aperture_mm=16.0, density=0.5)
    assert np.allclose(s.apex_vox, [32, 32, 32]) and s.roc_mm == pytest.approx(20.0)

    # a placement from another world frame is refused rather than silently mis-mapped
    pl["frame"] = "nrrd_voxel_mm"
    p.write_text(json.dumps(pl))
    with pytest.raises(ValueError, match="frame"):
        F.bowl_source_from_placement(tmp_path, p, aperture_mm=16.0, density=0.5)


# ---- focal-box math on synthetic boxes -------------------------------------

def _box(target, fb=6):
    """The focal-box recorder coordinates around ``target`` (same lattice the deck uses)."""
    from skull_transparency.sim import _common as C
    return C.focal_box(np.asarray(target), (64, 64, 64), fb).astype(int)[:, :3]


def _sinusoid_box(box, target, amp_vox, *, n_frames=64, period=8.0, sigma=3.0, peak=1.0):
    """A synthetic recording: a Gaussian-in-space, sinusoidal-in-time focal spot centred on
    ``amp_vox`` with peak amplitude ``peak`` (Pa) and amplitude e-folding ``sigma`` voxels."""
    d2 = ((box - np.asarray(amp_vox, float)) ** 2).sum(1)
    env = peak * np.exp(-d2 / (2.0 * sigma ** 2))
    t = np.arange(n_frames)[:, None]
    return (env[None, :] * np.sin(2 * np.pi * t / period)).astype(np.float32)


def test_peak_pressure_recovers_amplitude_between_samples():
    # a crest that falls between two recorded samples: the parabolic refinement recovers it
    t = np.arange(40) + 0.37
    traces = (2.5 * np.sin(2 * np.pi * t / 7.0)).astype(np.float32)[:, None]
    raw = np.abs(traces).max()
    pk = F.peak_pressure(traces)[0]
    assert raw < 2.5                       # the naive sample max under-reads
    assert pk == pytest.approx(2.5, rel=0.02)
    assert pk >= raw


def test_focal_metrics_peak_shift_and_fwhm():
    target = np.array([32.0, 32.0, 32.0])
    box = _box(target, fb=8)
    sigma = 3.0
    peak_at = target + np.array([2.0, 0.0, 0.0])          # focus pushed 2 voxels off target
    traces = _sinusoid_box(box, target, peak_at, sigma=sigma, peak=4.0)

    m = F.focal_metrics(traces, box, target, dx_mm=0.5, label="tc")
    assert m.peak_pa == pytest.approx(4.0, rel=0.02)
    assert np.allclose(m.peak_vox, peak_at)
    assert m.focal_shift_mm == pytest.approx(2.0 * 0.5)    # 2 voxels x 0.5 mm
    assert not m.on_box_edge                               # the peak is interior to the box
    # the recorder ON the target sees the Gaussian 2 voxels off its crest
    assert m.peak_at_target_pa == pytest.approx(4.0 * np.exp(-4.0 / (2 * sigma ** 2)), rel=0.03)
    # -6 dB (half-pressure) width of a Gaussian amplitude profile = 2*sqrt(2 ln2)*sigma
    expect_mm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma * 0.5
    for w in m.fwhm_mm:
        assert w == pytest.approx(expect_mm, rel=0.05)
    assert m.to_dict()["label"] == "tc"


def test_focal_metrics_flags_a_peak_on_the_box_edge():
    # a peak on the box face means the true maximum may be outside the recorded box
    target = np.array([32.0, 32.0, 32.0])
    box = _box(target, fb=6)
    traces = _sinusoid_box(box, target, target + np.array([6.0, 0.0, 0.0]), sigma=2.0)
    with pytest.warns(UserWarning, match="boundary"):
        m = F.focal_metrics(traces, box, target, dx_mm=1.0, label="transcranial")
    assert m.on_box_edge


def test_fwhm_is_inf_when_the_box_is_too_small():
    target = np.array([32.0, 32.0, 32.0])
    box = _box(target, fb=3)                              # box narrower than the spot
    traces = _sinusoid_box(box, target, target, sigma=20.0)
    m = F.focal_metrics(traces, box, target, dx_mm=0.5)
    assert all(np.isinf(w) for w in m.fwhm_mm)


def test_compare_focal_boxes_ratio_and_db():
    target = np.array([32.0, 32.0, 32.0])
    box = _box(target, fb=8)
    ff = _sinusoid_box(box, target, target, peak=2.0)                   # free field: on target
    tc = _sinusoid_box(box, target, target + np.array([0, 0, 3.0]), peak=1.0)   # halved + shifted

    cmp = F.compare_focal_boxes(tc, ff, box, target, dx_mm=1.0)
    assert cmp.p_transcranial_pa == pytest.approx(1.0, rel=0.02)
    assert cmp.p_freefield_pa == pytest.approx(2.0, rel=0.02)
    assert cmp.transmission == pytest.approx(0.5, rel=0.03)
    assert cmp.transmission_db == pytest.approx(-6.02, abs=0.3)         # half pressure = -6 dB
    assert cmp.focal_shift_mm == pytest.approx(3.0)                     # transcranial case
    assert cmp.freefield_focal_shift_mm == pytest.approx(0.0)
    assert cmp.free_field is not None and cmp.freefield_fwhm_mm is not None
    # at the target itself the shifted transcranial focus delivers less than half
    assert cmp.transmission_at_target < cmp.transmission
    assert cmp.transmission_at_target_db < cmp.transmission_db
    assert "insertion loss" in cmp.summary()


def test_compare_focal_boxes_without_free_field():
    target = np.array([32.0, 32.0, 32.0])
    box = _box(target, fb=6)
    cmp = F.compare_focal_boxes(_sinusoid_box(box, target, target, peak=3.0), None,
                                box, target, dx_mm=1.0)
    assert cmp.p_freefield_pa is None and cmp.transmission is None
    assert cmp.transmission_db is None and cmp.freefield_fwhm_mm is None
    assert cmp.transmission_at_target is None
    assert cmp.p_transcranial_pa == pytest.approx(3.0, rel=0.02)


def test_forward_comparison_to_dict_is_json_round_trippable():
    target = np.array([32.0, 32.0, 32.0])
    box = _box(target, fb=6)
    cmp = F.compare_focal_boxes(_sinusoid_box(box, target, target, peak=1.0),
                                _sinusoid_box(box, target, target, peak=4.0),
                                box, target, dx_mm=0.5,
                                run_dirs={"transcranial": "a", "free_field": "b"},
                                source={"n_points": 7})
    d = json.loads(json.dumps(cmp.to_dict()))              # must be plain JSON types
    assert d["transmission"] == pytest.approx(0.25, rel=0.03)
    assert d["transmission_db"] == pytest.approx(-12.04, abs=0.3)
    assert d["dx_mm"] == 0.5 and d["source"]["n_points"] == 7
    assert d["run_dirs"]["free_field"] == "b"
    assert d["transcranial"]["peak_pa"] == pytest.approx(1.0, rel=0.02)
    assert d["free_field"]["peak_pa"] == pytest.approx(4.0, rel=0.02)
    # both foci are on target here, so the two ratios agree
    assert d["transmission_at_target"] == pytest.approx(d["transmission"], rel=1e-6)
    assert d["transcranial"]["on_box_edge"] is False


# ---- deck writing (no medium maps, no solver) ------------------------------

def _write_sim_tree(sim_dir, *, n=64, dx_m=1e-3, target=(32, 32, 12), bone=False,
                    bone_z=(24, 28)):
    """A minimal sim tree (meta.json + c.f32 + registration.json) for the deck tests."""
    from skull_transparency.registration import Registration
    sim_dir = str(sim_dir)
    os.makedirs(sim_dir, exist_ok=True)
    c = np.full((n, n, n), 1540.0, dtype="<f4")
    if bone:
        c[:, :, bone_z[0]:bone_z[1]] = 2900.0         # a slab between transducer and target
    c.ravel(order="F").tofile(os.path.join(sim_dir, "c.f32"))
    meta = {"N": n, "grid_shape": [n, n, n], "dX_m": dx_m, "C0": 1540.0, "F0": 5e5,
            "ppw": 1540.0 / 5e5 / dx_m, "dent_grid": list(map(float, target)),
            "n_array": 0, "c_file": "c.f32", "rho_file": None, "alpha_file": None,
            "attenuation": False, "input_frame": "ras_mm", "subject_id": "test",
            "transducer": {"geometry": "bowl", "roc_mm": 20.0, "aperture_mm": 16.0}}
    with open(os.path.join(sim_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    Registration(R_mni_to_sim=np.eye(3), dx_mm=dx_m * 1e3,
                 target_mni_mm=np.asarray(target, float),
                 target_fullres_voxel=np.asarray(target, float),
                 world_frame="ras_mm").to_json(os.path.join(sim_dir, "registration.json"))
    return meta


def test_write_forward_pair_writes_two_identical_decks(tmp_path):
    """The pair must differ ONLY in the medium: same source cells, same drive, same box."""
    sim = tmp_path / "sim"
    meta = _write_sim_tree(sim, n=64, dx_m=1e-3, target=(32, 32, 12), bone=True)
    out = tmp_path / "fwd"
    decks = F.write_forward_pair(sim, out_dir=out, roc_mm=20.0, aperture_mm=16.0,
                                 density=0.4, box_half_mm=5.0, modT=2,
                                 write_maps=False, run_solver=False)
    tc, ff = decks["transcranial"][0], decks["free_field"][0]
    for name in ("icc.dat", "outc.dat", "icmat.dat", "nTic.dat", "modT.dat", "box_vox.npy"):
        a = (tc / name).read_bytes()
        b = (ff / name).read_bytes()
        assert a == b, f"{name} differs between the two forward decks"
    assert not (tc / "modX.dat").exists()             # no genout_mod whole-volume dump
    info = decks["transcranial"][1]
    assert info["n_box"] == decks["free_field"][1]["n_box"] > 0
    assert info["fb_vox"] == 5                        # 5 mm at 1 mm/voxel
    assert info["p0_pa"] == 1.0 and info["modT"] == 2
    assert info["n_frames_expected"] > 0
    # the drive is one delayed pulse per bowl-face voxel, at the face amplitude p0
    icmat = np.fromfile(tc / "icmat.dat", dtype="<f4").reshape(decks["source"].n_points, -1)
    assert icmat.shape[0] == decks["source"].n_points
    assert np.abs(icmat).max() == pytest.approx(1.0, rel=0.02)   # p0 (sampled pulse crest)
    assert (np.abs(icmat).max(axis=1) > 0).all()      # every face cell is driven


def test_write_forward_pair_warns_when_the_bowl_sits_in_bone(tmp_path):
    # a transducer buried in the skull would report a huge "insertion loss" that is really
    # a placement error -- it must be flagged at launch, not at read-out
    sim = tmp_path / "sim"
    _write_sim_tree(sim, n=64, dx_m=1e-3, target=(32, 32, 12), bone=True, bone_z=(30, 40))
    with pytest.warns(UserWarning, match="in bone"):
        F.write_forward_pair(sim, out_dir=tmp_path / "fwd", free_field=False, roc_mm=20.0,
                             aperture_mm=16.0, density=0.4, box_half_mm=5.0,
                             write_maps=False, run_solver=False)


def test_write_forward_deck_can_skip_the_free_field(tmp_path):
    sim = tmp_path / "sim"
    _write_sim_tree(sim, n=64, dx_m=1e-3, target=(32, 32, 12))
    decks = F.write_forward_pair(sim, out_dir=tmp_path / "fwd", free_field=False,
                                 roc_mm=20.0, aperture_mm=16.0, density=0.4,
                                 box_half_mm=5.0, write_maps=False, run_solver=False)
    assert decks["free_field"] is None
    assert (tmp_path / "fwd" / "transcranial" / "forward_deck.json").exists()


def test_load_focal_box_round_trip(tmp_path):
    """A solved run reads back as (traces, box, info) in the solver's frame-major layout."""
    sim = tmp_path / "sim"
    _write_sim_tree(sim, n=64, dx_m=1e-3, target=(32, 32, 12))
    out = tmp_path / "fwd"
    decks = F.write_forward_pair(sim, out_dir=out, free_field=False, roc_mm=20.0,
                                 aperture_mm=16.0, density=0.4, box_half_mm=4.0,
                                 write_maps=False, run_solver=False)
    d, info = decks["transcranial"]
    with pytest.raises(FileNotFoundError):            # not solved yet
        F.load_focal_box(d)
    n_box, n_frames = info["n_box"], 5
    fake = np.arange(n_frames * n_box, dtype="<f4")
    fake.tofile(d / "genout.dat")
    traces, box, got = F.load_focal_box(d)
    assert traces.shape == (n_frames, n_box) and len(box) == n_box
    assert got["fb_vox"] == info["fb_vox"]


# ---- CLI -------------------------------------------------------------------

def test_cli_forward_parses():
    p = cli.build_parser()
    a = p.parse_args(["forward", "--sim", "run", "--placement", "result/placement.json",
                      "--out", "run/forward", "--gpu", "1", "--no-free-field", "--box-mm", "8"])
    assert a.func is cli._cmd_forward
    assert a.sim == "run" and a.out == "run/forward" and a.gpu == 1
    assert a.no_free_field and a.box_mm == 8.0
    assert a.modt == 2 and a.p0 == 1.0 and not a.no_run    # documented defaults

    with pytest.raises(SystemExit):                        # --sim and --out are required
        p.parse_args(["forward", "--sim", "run"])


def test_cli_forward_accepts_negative_world_coordinates():
    argv = cli._merge_coord_args(["forward", "--sim", "run", "--out", "f",
                                  "--apex-mm", "-4,24,28", "--target-mm", "-12,-57,-34"])
    a = cli.build_parser().parse_args(argv)
    assert a.apex_mm == "-4,24,28" and a.target_mm == "-12,-57,-34"
    assert np.allclose(cli._parse_vec(a.target_mm), [-12, -57, -34])


def test_cli_forward_no_run_writes_decks(tmp_path):
    sim = tmp_path / "sim"
    _write_sim_tree(sim, n=48, dx_m=2e-3, target=(24, 24, 10))
    out = tmp_path / "fwd"
    rc = cli.main(["forward", "--sim", str(sim), "--out", str(out), "--no-run",
                   "--roc-mm", "20", "--aperture-mm", "16", "--density", "0.3",
                   "--box-mm", "8"])
    assert rc == 0
    for case in ("transcranial", "free_field"):
        assert (out / case / "forward_deck.json").exists()
        assert (out / case / "icmat.dat").exists()


# ---- the real pair (needs the CUDA solver + a GPU) -------------------------

def _solver_available():
    from skull_transparency.sim.launchers import _resolve_solver_binary
    return os.path.exists(_resolve_solver_binary()) and shutil.which("nvidia-smi") is not None


@pytest.mark.skipif(not _solver_available(),
                    reason="needs the fullwave2-ultra CUDA solver and a GPU")
@pytest.mark.skipif(os.environ.get("SKULL_TR_FORWARD_GPU") != "1",
                    reason="set SKULL_TR_FORWARD_GPU=1 to solve the pair for real "
                           "(two GPU solves, ~260 MB of scratch medium maps)")
def test_forward_pair_end_to_end(tmp_path):
    """Solve the pair on a tiny water-plus-bone-slab grid: a 4 mm plate between transducer
    and target can only take pressure away, so the transmission must be <= 1 (<= 0 dB).

    The recording box is kept clear of the plate (bone sits at z = 24..28, the box spans
    z = 9..23): a recorder inside bone reads the elevated in-bone pressure, not the focus.
    """
    sim = tmp_path / "sim"
    _write_sim_tree(sim, n=80, dx_m=1e-3, target=(40, 40, 16), bone=True, bone_z=(24, 28))
    cmp = F.run_forward_pair(sim, out_dir=tmp_path / "fwd", gpu=0, roc_mm=20.0,
                             aperture_mm=24.0, density=1.0, box_half_mm=7.0, modT=2)
    assert cmp.p_freefield_pa > 0 and cmp.p_transcranial_pa > 0
    assert not cmp.transcranial.on_box_edge and not cmp.free_field.on_box_edge
    assert cmp.transmission <= 1.0 and cmp.transmission_db <= 0.0
    assert cmp.transmission_at_target <= 1.0
    assert (tmp_path / "fwd" / "forward.json").exists()
