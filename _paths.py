"""Portable path resolution — no machine-specific absolute paths.

- tuba: found as a sibling repo (../tuba/src) relative to THIS file, else assumed
  pip-installed (`import tuba` works). Override with $TUBA_SRC.
- Halle CT cache: $HALLE_DEST or the tuba default ~/.cache/tuba/human/source.
- repo assets (parula_256.npy, sparse1024_elements_mm.npy) live beside the scripts.

So the whole thing runs on any machine given: the sibling repo layout (or installed
packages) + the data fetched to the cache (tuba.data.fetch_human writes there).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))          # .../pinton-lab/skull_transparency
PINTON_LAB = os.path.dirname(REPO)                          # .../pinton-lab


def add_tuba():
    """Put tuba on sys.path from the sibling repo (or $TUBA_SRC); no-op if importable."""
    try:
        import tuba  # already installed / on path
        return os.path.dirname(os.path.dirname(tuba.__file__))
    except Exception:
        pass
    for p in (os.environ.get("TUBA_SRC"), os.path.join(PINTON_LAB, "tuba", "src")):
        if p and os.path.isdir(p):
            sys.path.insert(0, p)
            return p
    raise ImportError("tuba not found: install it (pip install -e ../tuba) or set $TUBA_SRC")


def halle_cache(name=""):
    """Path inside the Halle CT cache ($HALLE_DEST or tuba's ~/.cache default)."""
    base = os.environ.get("HALLE_DEST", os.path.expanduser("~/.cache/tuba/human/source"))
    return os.path.join(base, name) if name else base


def asset(name):
    """Path to a repo-vendored asset (parula_256.npy, sparse1024_elements_mm.npy, ...)."""
    return os.path.join(REPO, name)
