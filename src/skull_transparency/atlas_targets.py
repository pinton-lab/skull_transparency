"""Named MNI targets and EEG 10-20 scalp sites for atlas-space targeting.

The coordinates are standard atlas/literature centroids in MNI RAS mm (the same set the
TUBA atlas integration publishes for this skull family); the EEG 10-20 positions follow
Koessler et al. 2009 / Jurcak et al. 2007. For the ITRUSST benchmark skull — which is
authored in MNI-aligned RAS — these coordinates are usable directly as world-mm targets;
for a subject skull they need the atlas registration first.
"""
from __future__ import annotations

import numpy as np

# name -> (x, y, z) MNI RAS mm. Each entry lies inside the ITRUSST endocranial cavity,
# from shallow cortical (DLPFC, ~5 mm off the inner table) to deep (thalamus, ~43 mm).
MNI_TARGETS: dict[str, tuple[float, float, float]] = {
    "dACC_left":          (-4, 24, 28),     # dorsal anterior cingulate (deep cortical tFUS target)
    "M1_left":            (-37, -21, 58),   # hand-knob motor cortex
    "S1_left":            (-40, -25, 55),   # primary somatosensory
    "DLPFC_left":         (-40, 35, 35),    # dorsolateral prefrontal
    "pre_SMA":            (0, 10, 60),      # pre-supplementary motor
    "PCC":                (0, -50, 30),     # posterior cingulate
    "V1":                 (0, -90, 0),      # primary visual
    "thalamus_central":   (0, -16, 8),      # central thalamus
    "STN_left":           (-12, -14, -7),   # subthalamic nucleus
    "hippocampus_left":   (-25, -20, -15),
    "amygdala_left":      (-22, -6, -18),
    "cerebellum_central": (0, -60, -30),
}

# EEG 10-20 sites, MNI RAS mm (scalp). Used to describe a placement in the language an
# experimenter can find on a head ("window ~ C3, 12 mm posterior").
EEG_SITES_MNI: dict[str, tuple[float, float, float]] = {
    "Fp1": (-21.5, 70.2, -0.1), "Fp2": (28.4, 69.1, -0.4),
    "F3": (-39.7, 34.4, 53.9), "F4": (41.9, 37.6, 53.4),
    "F7": (-54.8, 33.9, -3.5), "F8": (56.6, 35.7, -3.7),
    "C3": (-52.2, -16.4, 57.8), "C4": (54.1, -18.0, 57.5),
    "Cz": (0.4, -21.3, 87.0),
    "CP3": (-55.9, -44.5, 56.8), "CP4": (56.6, -45.8, 56.6),
    "P3": (-52.4, -62.6, 46.4), "P4": (53.7, -64.7, 47.6),
    "O1": (-23.8, -94.0, -1.3), "O2": (25.4, -93.0, -1.6),
    "Oz": (0.0, -101.6, -2.7),
    "T3": (-72.0, -22.6, -32.4), "T4": (72.4, -23.6, -33.5),
    "T5": (-65.0, -68.0, -10.6), "T6": (65.0, -68.0, -10.6),
    "AF3": (-29.3, 60.6, 35.0), "AF4": (32.4, 59.6, 35.7),
}


def target_mni(name: str) -> np.ndarray:
    """MNI RAS-mm coordinate of a named target (KeyError lists the valid names)."""
    try:
        return np.asarray(MNI_TARGETS[name], float)
    except KeyError:
        raise KeyError(f"unknown target {name!r}; available: {', '.join(MNI_TARGETS)}") from None


def nearest_eeg_site(point_mni_mm) -> tuple[str, float]:
    """The EEG 10-20 site nearest a point (both MNI mm) -> (site name, distance mm)."""
    p = np.asarray(point_mni_mm, float)
    names = list(EEG_SITES_MNI)
    d = np.linalg.norm(np.array([EEG_SITES_MNI[n] for n in names], float) - p, axis=1)
    i = int(np.argmin(d))
    return names[i], float(d[i])
