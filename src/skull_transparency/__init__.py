"""skull_transparency — skull acoustic transparency maps (time-reversal reciprocity)
and transducer placement for transcranial ultrasound."""
from .registration import Registration
from .bundle import FieldBundle, load_bundle, build_field_bundle
from .surface import extract_external_surface, true_normals
from .metrics import integrate_outward, peak_intensity, distance_correct
from .transparency import compute_transparency_map, TransparencyMap, TransparencyOptions
from .placement import (place_bowl, BowlConstraints, BowlPlacement,
                        place_array, ArrayConstraints, ArrayLayout,
                        place_bowl_optimal, drive_optimal, drive_phase_only,
                        drive_single_element, coherence_factor, focal_psf, focal_fwhm,
                        CapField, CapPlacement, score_cap_pose, place_cap_optimal)
from .transducer import (CTX500, CapPose, build_cap, build_cap_pose, pose_apex_aim,
                         anatomical_az_el, ROC_MM, APERTURE_MM, HALF_ANGLE_DEG)
from .transducer_spec import TransducerSpec
from .complex_field import ComplexField, element_field, ballistic_window_global
from .projection import project_to_sphere, energy_on_unit_sphere
from .neuromod import to_placement_dict
from .score import PositioningScore
from .sample import make_synthetic_bundle
from .brain_center import (MNI_BRAIN_COM_MM, intracranial_centroid, cavity_mask_centroid,
                           brain_center_phys_mm, brain_center_from_registration)
from .render import render_transparency_surface

__all__ = [
    "Registration",
    "FieldBundle", "load_bundle", "build_field_bundle",
    "extract_external_surface", "true_normals",
    "integrate_outward", "peak_intensity", "distance_correct",
    "compute_transparency_map", "TransparencyMap", "TransparencyOptions",
    "place_bowl", "BowlConstraints", "BowlPlacement",
    "place_array", "ArrayConstraints", "ArrayLayout",
    "place_bowl_optimal", "drive_optimal", "drive_phase_only",
    "drive_single_element", "coherence_factor", "focal_psf", "focal_fwhm",
    "CapField", "CapPlacement", "score_cap_pose", "place_cap_optimal",
    "CTX500", "CapPose", "build_cap", "build_cap_pose", "pose_apex_aim",
    "anatomical_az_el", "ROC_MM", "APERTURE_MM", "HALF_ANGLE_DEG",
    "TransducerSpec",
    "ComplexField", "element_field", "ballistic_window_global",
    "project_to_sphere", "energy_on_unit_sphere",
    "to_placement_dict",
    "PositioningScore",
    "make_synthetic_bundle",
    "MNI_BRAIN_COM_MM",
    "intracranial_centroid", "cavity_mask_centroid",
    "brain_center_phys_mm", "brain_center_from_registration",
    "render_transparency_surface",
]
__version__ = "0.1.0"
