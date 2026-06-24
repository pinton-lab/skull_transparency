"""Resolve the on-disk locations of the legacy ``hemisphere_tr`` study data.

Every location is overridable by an environment variable so a checkout on
another machine can point at its own copy of the data. The defaults reproduce
the original lab layout, so the in-place workflow is byte-for-byte unchanged
when nothing is set.

Environment variables (all optional):

* ``SKULL_TR_DATA_ROOT`` -- root directory holding ``data/``, ``sim/`` and
  ``analysis/``. Defaults to :data:`DEFAULT_DATA_ROOT`.
* ``FULLWAVE2_SIM_DIR`` -- the ``sim/`` tree (grid meta + transform); overrides
  the ``SKULL_TR_DATA_ROOT/sim`` default (kept for backward compatibility).

The CUDA solver binary is resolved separately in :mod:`skull_transparency.sim.launchers`
(``$FULLWAVE2_BIN`` / ``$FULLWAVE2_BIN_DIR`` / the sibling ``fullwave2-ultra`` checkout).
"""
from __future__ import annotations

import os
from pathlib import Path

#: Default root of the legacy hemisphere_tr study data (the original lab mount).
#: This is the single place the absolute default lives; override at runtime with
#: ``SKULL_TR_DATA_ROOT`` rather than editing it.
DEFAULT_DATA_ROOT = "/celerina/gfp/mfs/hemisphere_tr"


def data_root() -> Path:
    """Root directory containing ``data/``, ``sim/`` and ``analysis/``."""
    return Path(os.environ.get("SKULL_TR_DATA_ROOT", DEFAULT_DATA_ROOT))


def sim_dir() -> Path:
    """The ``sim/`` tree (grid ``meta.json`` + ``ppw55_transform.npz``).

    Honours ``FULLWAVE2_SIM_DIR`` first, else ``SKULL_TR_DATA_ROOT/sim``.
    """
    env = os.environ.get("FULLWAVE2_SIM_DIR")
    return Path(env) if env else data_root() / "sim"


def bundle_dir(name: str = "halle_hemis_ppw55") -> Path:
    """A Field Bundle directory under ``SKULL_TR_DATA_ROOT/data/``."""
    return data_root() / "data" / name


def analysis_dir() -> Path:
    """The ``analysis/`` tree (heavy per-element validation artifacts)."""
    return data_root() / "analysis"
