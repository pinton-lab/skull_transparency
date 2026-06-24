import numpy as np
import pytest

from skull_transparency import TransducerSpec
from skull_transparency.transducer import ROC_MM, APERTURE_MM, HALF_ANGLE_DEG


def test_grid_pitch_matches_halle_ppw55():
    # 1 MHz, 5.5 ppw, 1540 m/s -> 0.28 mm, the Halle ppw55 grid (meta['dX_m'] = 0.00028)
    s = TransducerSpec(f0_hz=1e6, ppw=5.5, geometry="array")
    assert np.isclose(s.dx_m, 0.00028, atol=1e-9)
    assert np.isclose(s.dx_mm, 0.28, atol=1e-6)
    assert np.isclose(s.wavelength_mm, 1.54, atol=1e-6)
    # ppw is recoverable as wavelength / dx
    assert np.isclose(s.wavelength_mm / s.dx_mm, 5.5, atol=1e-9)


def test_frequency_sets_pitch():
    # halving the frequency doubles the wavelength and the pitch (the 500/250 kHz roadmap)
    hi = TransducerSpec.ctx500(f0_hz=500e3, ppw=6.0)
    lo = TransducerSpec.ctx500(f0_hz=250e3, ppw=6.0)
    assert np.isclose(lo.dx_m, 2.0 * hi.dx_m)


def test_ctx500_geometry():
    s = TransducerSpec.ctx500()
    assert s.geometry == "annular" and s.n_rings == 4
    assert s.roc_mm == ROC_MM and s.aperture_mm == APERTURE_MM
    assert np.isclose(s.half_angle_deg, HALF_ANGLE_DEG, atol=1e-9)   # ~30.4 deg


def test_half_angle_from_aperture():
    s = TransducerSpec(f0_hz=1e6, geometry="bowl", roc_mm=63.2, aperture_mm=64.0)
    assert np.isclose(s.half_angle_deg, np.degrees(np.arcsin(32.0 / 63.2)))


def test_to_bowl_constraints_defaults_focus_to_roc():
    s = TransducerSpec(f0_hz=1e6, geometry="bowl", roc_mm=60.0, aperture_mm=30.0,
                       acceptance_angle_deg=35.0)
    bc = s.to_bowl_constraints()
    assert bc.focal_length_mm == 60.0          # defaults to ROC
    assert bc.bowl_radius_mm == 15.0           # aperture / 2
    assert bc.theta_max_deg == 35.0
    assert s.to_bowl_constraints(focal_length_mm=72.0).focal_length_mm == 72.0


def test_to_array_constraints_spacing_defaults_to_half_wavelength():
    s = TransducerSpec(f0_hz=1e6, geometry="array", n_elements=64)
    ac = s.to_array_constraints(region_center_mni_mm=[-12.0, -57.0, -34.0],
                                region_radius_mm=60.0)
    assert ac.n_elements == 64
    assert np.isclose(ac.min_spacing_mm, s.wavelength_mm / 2.0)
    assert ac.region_radius_mm == 60.0
    assert list(ac.region_center_mni_mm) == [-12.0, -57.0, -34.0]


def test_to_meta_fields_shape():
    s = TransducerSpec.ctx500(f0_hz=500e3, ppw=6.0)
    m = s.to_meta_fields()
    assert set(m) >= {"dX_m", "C0", "F0", "ppw", "transducer"}
    assert m["F0"] == 500e3 and m["C0"] == 1540.0
    assert m["transducer"]["geometry"] == "annular" and m["transducer"]["n_rings"] == 4


@pytest.mark.parametrize("kwargs", [
    dict(f0_hz=0.0),                                          # non-positive frequency
    dict(f0_hz=1e6, ppw=0.0),                                 # non-positive ppw
    dict(f0_hz=1e6, geometry="bowl"),                         # bowl without roc/aperture
    dict(f0_hz=1e6, geometry="bowl", roc_mm=30.0, aperture_mm=70.0),  # aperture > 2*roc
])
def test_validation_rejects_bad_specs(kwargs):
    with pytest.raises(ValueError):
        TransducerSpec(**kwargs)
