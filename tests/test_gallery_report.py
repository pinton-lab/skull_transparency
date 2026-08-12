"""Gallery fetch (checksum, cache, local-dir resolution) and HTML report generation."""
import hashlib
import json
import zipfile

import numpy as np
import pytest

import skull_transparency as st
from skull_transparency import gallery
from skull_transparency.atlas_targets import MNI_TARGETS, nearest_eeg_site, target_mni


def _zip_bundle(bundle_dir, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle_dir.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(bundle_dir.parent))
    return hashlib.sha256(zpath.read_bytes()).hexdigest()


@pytest.fixture()
def fake_gallery(synthetic_bundle, tmp_path):
    """A local gallery dir holding the synthetic bundle as a zip + a matching registry."""
    gdir = tmp_path / "gallery_src"
    gdir.mkdir()
    zpath = gdir / ("testtarget" + gallery.ZIP_SUFFIX)
    sha = _zip_bundle(synthetic_bundle, zpath)
    reg = {"schema": "skull_transparency.gallery/1", "skull": "synthetic", "base_url": None,
           "bundles": {"testtarget": {"file": zpath.name, "sha256": sha,
                                      "target_mni_mm": [0, 0, 0], "size_mb": 1}}}
    return gdir, reg


def test_gallery_fetch_local_dir(fake_gallery, tmp_path, monkeypatch):
    gdir, reg = fake_gallery
    cache = tmp_path / "cache"
    monkeypatch.setenv("SKULL_TRANSPARENCY_GALLERY", str(gdir))
    bdir = gallery.fetch("testtarget", reg=reg, cache=cache, verbose=False)
    assert (bdir / "bundle.json").exists()
    b = st.load_bundle(bdir)                       # the unzipped bundle round-trips
    assert b is not None
    # second fetch is served from the cache marker (delete the source zip to prove it)
    (gdir / ("testtarget" + gallery.ZIP_SUFFIX)).unlink()
    bdir2 = gallery.fetch("testtarget", reg=reg, cache=cache, verbose=False)
    assert bdir2 == bdir


def test_gallery_fetch_checksum_mismatch(fake_gallery, tmp_path, monkeypatch):
    gdir, reg = fake_gallery
    reg["bundles"]["testtarget"]["sha256"] = "0" * 64
    monkeypatch.setenv("SKULL_TRANSPARENCY_GALLERY", str(gdir))
    with pytest.raises(ValueError, match="checksum"):
        gallery.fetch("testtarget", reg=reg, cache=tmp_path / "cache2", verbose=False)


def test_gallery_unknown_and_unpublished(fake_gallery, tmp_path, monkeypatch):
    _, reg = fake_gallery
    with pytest.raises(KeyError, match="available"):
        gallery.fetch("nope", reg=reg, cache=tmp_path / "c3", verbose=False)
    monkeypatch.delenv("SKULL_TRANSPARENCY_GALLERY", raising=False)
    with pytest.raises(FileNotFoundError, match="not published"):
        gallery.fetch("testtarget", reg=reg, cache=tmp_path / "c4", verbose=False)


def test_atlas_targets():
    assert np.allclose(target_mni("dACC_left"), [-4, 24, 28])
    with pytest.raises(KeyError, match="available"):
        target_mni("nowhere")
    site, d = nearest_eeg_site(MNI_TARGETS["M1_left"])
    assert site in ("C3", "CP3") and d < 60.0      # M1 is under the C3 neighbourhood


def test_report_html(synthetic_bundle, tmp_path):
    tmap = st.compute_transparency_map(st.load_bundle(synthetic_bundle))
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=60.0))
    out = tmp_path / "report.html"
    from skull_transparency.report import write_report
    write_report(tmap, pl, out, target_name="testtarget")
    html = out.read_text()
    assert "data:image/png;base64," in html and html.count("data:image/png") == 2
    assert "Nearest EEG 10-20 site" in html and "Transparency score" in html
    assert not out.with_suffix(".placement.png").exists()   # temp figure cleaned up
