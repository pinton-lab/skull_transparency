import os
from pathlib import Path

import pytest

from skull_transparency import paths

# Integration tests run against the existing Halle/dentate bundle if present.
# Locations come from skull_transparency.paths (override via $SKULL_TR_DATA_ROOT);
# $SKULL_TR_TEST_BUNDLE still pins just the bundle dir.
DATA = Path(os.environ.get("SKULL_TR_TEST_BUNDLE", str(paths.bundle_dir())))
SIM = paths.sim_dir()


@pytest.fixture(scope="session")
def data_dir():
    if not (DATA / "propagation_map.npy").exists():
        pytest.skip(f"test bundle not available at {DATA}")
    return DATA


@pytest.fixture(scope="session")
def transform_npz():
    p = SIM / "ppw55_transform.npz"
    if not p.exists():
        pytest.skip("ppw55_transform.npz not available")
    return p


@pytest.fixture(scope="session")
def bundle(data_dir, transform_npz):
    import skull_transparency as st
    if not (data_dir / "bundle.json").exists():
        st.build_field_bundle(data_dir, SIM / "meta.json", transform_npz, target_name="dentate_left")
    return st.load_bundle(data_dir)


@pytest.fixture(scope="session")
def synthetic_bundle(tmp_path_factory):
    """A tiny synthetic Field Bundle — no /celerina data, GPU, or tuba. For tests of the
    transparency -> placement -> score back half that don't need the legacy golden data."""
    import skull_transparency as st
    return st.make_synthetic_bundle(tmp_path_factory.mktemp("synthetic_bundle"))
