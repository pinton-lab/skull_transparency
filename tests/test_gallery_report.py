"""Gallery fetch (checksum, cache, local-dir resolution) and HTML report generation."""
import hashlib
import json
import shutil
import zipfile

from pathlib import Path

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
    assert html.count("data:image/png") == 7      # 3 unwraps, 3-D views, 2 scenes, placement
    assert "unwrapped" in html and "objective" in html and "@media print" in html
    assert "EEG 10-20 site" in html and "Transparency score" in html
    assert "Access (legal skull solid angle)" in html
    assert not out.with_suffix(".placement.png").exists()   # temp figure cleaned up


# ---------------------------------------------------------------------------
# The 3-D scene draws the DEVICE, not the beam's footprint on the skull
# ---------------------------------------------------------------------------

def test_scene3d_cap_is_sized_by_the_aperture_not_the_footprint(tmp_path):
    """`bowl_radius_mm` is where the cone crosses the SKULL; `aperture_mm` is the dish.

    They coincide only when the skull sits about one focal length from the target -- true
    of a human, false by an order of magnitude on a rodent, where the fallback would draw a
    2.4 deg cap for a device that subtends 35.1 deg.
    """
    import numpy as np
    from skull_transparency.report import _scene3d_figure

    bundle = st.load_bundle(st.make_synthetic_bundle(tmp_path / "b"))
    tmap = st.compute_transparency_map(bundle)
    pl = st.place_bowl(tmap, st.BowlConstraints())
    roc = float(np.linalg.norm(np.asarray(pl.apex_mni_mm, float)
                               - np.asarray(pl.target_mni_mm, float)))
    foot, aperture = 0.05 * roc, 1.15 * roc          # footprint tiny, dish wide (mouse-like)

    def cap_half_deg(**kw):
        import matplotlib.pyplot as plt
        fig = _scene3d_figure(tmap, pl, foot, **kw)
        # identify the cap by its defining property -- every point exactly one ROC from the
        # target -- rather than by size, which would pick up the (much larger) skull cloud
        cands = []
        for ax in fig.axes:
            for col in ax.collections:
                if not hasattr(col, "_offsets3d"):
                    continue
                q = np.column_stack(col._offsets3d)
                if len(q) < 100:
                    continue
                rr = np.linalg.norm(q - np.asarray(pl.target_mni_mm, float), axis=1)
                if np.allclose(rr, roc, rtol=1e-3):
                    cands.append(q)
        assert cands, "no bowl-cap scatter found in the scene"
        pts = cands[0]
        d = pts - np.asarray(pl.target_mni_mm, float)
        n = np.asarray(pl.apex_mni_mm, float) - np.asarray(pl.target_mni_mm, float)
        n = n / np.linalg.norm(n)
        cos = (d @ n) / np.linalg.norm(d, axis=1)
        plt.close(fig)
        return float(np.degrees(np.arccos(cos.min())))

    got_aperture = cap_half_deg(aperture_mm=aperture)
    got_fallback = cap_half_deg()
    assert got_aperture == pytest.approx(np.degrees(np.arcsin(aperture / 2 / roc)), abs=1.0)
    assert got_fallback == pytest.approx(np.degrees(np.arcsin(foot / roc)), abs=1.0)
    assert got_aperture > 5 * got_fallback           # the bug this guards is a ~15x error


def test_scene3d_device_frame_contains_the_whole_dish(tmp_path):
    """The head-framed view crops a dish that is large next to the head; the device frame
    must not. Checked on the axis limits, which is what actually does the cropping."""
    import numpy as np
    from skull_transparency.report import _scene3d_figure

    bundle = st.load_bundle(st.make_synthetic_bundle(tmp_path / "b"))
    tmap = st.compute_transparency_map(bundle)
    # A rodent-like regime: focal length far beyond the head, so the dish sits well outside
    # any head-framed view. At the packaged 63 mm focal length the synthetic head is itself
    # ~120 mm across and the head frame happens to contain the cap, which tests nothing.
    pl = st.place_bowl(tmap, st.BowlConstraints(focal_length_mm=300.0))
    roc = float(np.linalg.norm(np.asarray(pl.apex_mni_mm, float)
                               - np.asarray(pl.target_mni_mm, float)))
    foot, aperture = 0.02 * roc, 1.15 * roc          # mouse-like: tiny footprint, wide dish

    def cap_inside(frame):
        import matplotlib.pyplot as plt
        fig = _scene3d_figure(tmap, pl, foot, aperture_mm=aperture, frame=frame)
        ax = fig.axes[0]
        lims = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        cap = None
        for col in ax.collections:
            if not hasattr(col, "_offsets3d"):
                continue
            q = np.column_stack(col._offsets3d)
            if len(q) >= 100 and np.allclose(
                    np.linalg.norm(q - np.asarray(pl.target_mni_mm, float), axis=1),
                    roc, rtol=1e-3):
                cap = q
                break
        assert cap is not None
        plt.close(fig)
        return bool(np.all(cap >= lims[:, 0]) and np.all(cap <= lims[:, 1]))

    assert cap_inside("device"), "device frame must contain the whole dish"
    assert not cap_inside("head"), "this fixture should be one the head frame crops"


# ---------------------------------------------------------------------------
# Animation frames: the PDF page is embedded whole, so frame size matters
# ---------------------------------------------------------------------------

def test_movie_frames_hit_the_target_width_and_do_not_halve(tmp_path):
    """A 1200 px source against a 900 px target must land on 900, not 600.

    The old `im[::ceil(w/width)]` decimation halved anything between 1x and 2x the target,
    which is how the propagation movie ended up visibly soft in the report PDF.
    """
    import imageio.v3 as iio
    import numpy as np
    from skull_transparency.report import _movie_frames

    src = tmp_path / "frames"
    src.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        iio.imwrite(src / f"{i:03d}.png", rng.integers(0, 255, (400, 1200, 3), dtype=np.uint8))

    n, _ = _movie_frames(src, tmp_path / "out", width=900)
    assert n == 8
    im = iio.imread(tmp_path / "out" / "frame-0.png")
    assert im.shape[1] == 900 and im.shape[0] == 300      # exact width, aspect preserved


def test_movie_frames_do_not_upscale_a_small_source(tmp_path):
    import imageio.v3 as iio
    import numpy as np
    from skull_transparency.report import _movie_frames

    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        iio.imwrite(src / f"{i:03d}.png", np.zeros((100, 300, 3), np.uint8))
    _movie_frames(src, tmp_path / "out", width=1200)
    assert iio.imread(tmp_path / "out" / "frame-0.png").shape[:2] == (100, 300)


def test_movie_frames_subsample_in_time_and_keep_duration(tmp_path):
    """Time subsampling must preserve wall-clock duration, or the animation plays wrong."""
    import imageio.v3 as iio
    import numpy as np
    from skull_transparency.report import _movie_frames

    src = tmp_path / "frames"
    src.mkdir()
    for i in range(50):
        iio.imwrite(src / f"{i:03d}.png", np.zeros((20, 40, 3), np.uint8))
    n, fps = _movie_frames(src, tmp_path / "out", max_frames=10)
    assert n == 10
    assert n / fps == pytest.approx(50 / 12.0, rel=1e-6)   # 12 fps is the default source rate


@pytest.mark.parametrize("autoplay,want,unwanted", [(True, "autoplay", "controls"),
                                                    (False, "controls", "autoplay")])
def test_animated_page_option_string(tmp_path, monkeypatch, autoplay, want, unwanted):
    """The animation must start on its own by default rather than wait for a click.

    Asserted on the generated LaTeX -- `subprocess.run` is intercepted, so this neither
    needs a TeX installation nor spends one.
    """
    import subprocess

    import imageio.v3 as iio
    import numpy as np

    from skull_transparency import report as R

    src = tmp_path / "frames"
    src.mkdir()
    for i in range(3):
        iio.imwrite(src / f"{i:03d}.png", np.zeros((20, 40, 3), np.uint8))

    seen = {}
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd and cmd[0] == "pdflatex":
            seen["tex"] = (Path(kw["cwd"]) / "anim.tex").read_text()
            return subprocess.CompletedProcess(cmd, 1, "", "")     # -> warn, return early
        return real_run(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/pdflatex")
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    with pytest.warns(UserWarning):
        R.append_animated_page(pdf, src, autoplay=autoplay)
    assert want in seen["tex"] and unwanted not in seen["tex"]
    assert "loop" in seen["tex"]


def test_animated_page_merge_keeps_the_acroform(tmp_path):
    """The animation is AcroForm pushbutton fields driven by a /Screen page-open action.

    Merging page-by-page copies the annotations but drops the catalog's /AcroForm, which
    orphans every field: the page renders and nothing plays, in any viewer. This asserts
    the fields survive the merge into the report.
    """
    import shutil as _sh

    import imageio.v3 as iio
    import numpy as np
    import pypdf

    from skull_transparency.report import append_animated_page

    if not _sh.which("pdflatex"):
        pytest.skip("needs pdflatex")

    src = tmp_path / "frames"
    src.mkdir()
    for i in range(6):
        im = np.zeros((60, 120, 3), np.uint8)
        im[:, i * 20:(i + 1) * 20] = 255
        iio.imwrite(src / f"{i:03d}.png", im)

    host = tmp_path / "host.pdf"
    (tmp_path / "h.tex").write_text(
        "\\documentclass{article}\\begin{document}host\\end{document}\n")
    import subprocess
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "h.tex"],
                   cwd=tmp_path, capture_output=True, timeout=120)
    if not (tmp_path / "h.pdf").exists():
        pytest.skip("pdflatex could not build the host document")
    _sh.copy(tmp_path / "h.pdf", host)

    append_animated_page(host, src, caption="probe")
    root = pypdf.PdfReader(str(host)).trailer["/Root"]
    af = root.get("/AcroForm")
    af = af.get_object() if hasattr(af, "get_object") else af
    assert af, "the merge dropped /AcroForm; the animation cannot play"
    fields = af.get("/Fields")
    fields = fields.get_object() if hasattr(fields, "get_object") else fields
    assert len(fields) >= 6, f"expected one field per frame, got {len(fields)}"
