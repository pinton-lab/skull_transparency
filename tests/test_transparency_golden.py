import numpy as np

from skull_transparency import compute_transparency_map, TransparencyOptions


def test_matches_legacy_npz(bundle, data_dir):
    """compute_transparency_map reproduces the legacy skull_external_intensity.npz
    bit-for-bit (same surface extraction + sampling + constants)."""
    ref_path = data_dir / "skull_external_intensity.npz"
    if not ref_path.exists():
        import pytest
        pytest.skip("legacy skull_external_intensity.npz not present")
    ref = np.load(ref_path)
    tm = compute_transparency_map(bundle, TransparencyOptions())
    for key in ("surf_vox", "rhat", "Iint", "Pmax", "Ipk_Wcm2", "rad_mm"):
        a = getattr(tm, key)
        b = ref[key]
        assert a.shape == b.shape, f"{key} shape {a.shape} != {b.shape}"
        assert np.max(np.abs(a - b)) < 1e-6, f"{key} differs"


def test_true_normal_unit(bundle):
    tm = compute_transparency_map(bundle, TransparencyOptions())
    norms = np.linalg.norm(tm.true_normal, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
