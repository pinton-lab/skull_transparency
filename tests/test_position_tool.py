"""Positioning preview (the [viz] tool). Headless matplotlib (Agg) on the synthetic
bundle — no display, GPU, or tuba. Skips cleanly if matplotlib isn't installed."""
import importlib.util

import pytest

import skull_transparency as st

pytest.importorskip("matplotlib")


def _place(tmp):
    bundle = st.load_bundle(st.make_synthetic_bundle(tmp / "bundle"))
    tmap = st.compute_transparency_map(bundle)
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60.0, bowl_radius_mm=15.0,
                                                theta_max_deg=35.0))
    return tmap, pl


def test_preview_placement_writes_png(tmp_path):
    from skull_transparency.position_tool import preview_placement
    tmap, pl = _place(tmp_path)
    out = tmp_path / "preview.png"
    p = preview_placement(tmap, pl, out_png=str(out), title="synthetic")
    assert p == str(out)
    assert out.exists() and out.stat().st_size > 1000          # a real PNG


def test_position_cli_writes_preview(tmp_path):
    from skull_transparency import cli
    bundle_dir = st.make_synthetic_bundle(tmp_path / "bundle")
    out = tmp_path / "pp.png"
    rc = cli.main(["position", "--bundle", str(bundle_dir), "--out", str(out)])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 1000


def test_view_napari_requires_napari(tmp_path):
    if importlib.util.find_spec("napari") is not None:
        pytest.skip("napari installed; the import-guard path isn't exercised")
    from skull_transparency.position_tool import view_napari
    tmap, pl = _place(tmp_path)
    with pytest.raises(ImportError, match="napari"):
        view_napari(tmap, pl)
