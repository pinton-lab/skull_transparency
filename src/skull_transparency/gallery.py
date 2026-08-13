"""Precomputed transparency-bundle gallery: the no-GPU path.

Computing a transparency map needs the CUDA solver; *using* one does not. The gallery is
a set of precomputed Field Bundles (one per named MNI target on the ITRUSST benchmark
skull at 500 kHz) that `explore`/`report` fetch on demand, so the map can be explored on
any computer — no GPU, no solver.

Resolution order for a bundle:
1. the local cache (``$SKULL_TRANSPARENCY_CACHE`` or ``~/.cache/skull_transparency``),
2. a local gallery directory of ``*.skullbundle.zip`` files (``$SKULL_TRANSPARENCY_GALLERY``),
3. the registry's download URL (``base_url`` + file), verified against its sha256.

The registry ships with the package (``gallery_registry.json``); entries whose checksum
is present but whose ``base_url`` is null exist but have not been published yet — point
``$SKULL_TRANSPARENCY_GALLERY`` at a directory holding the zips, or publish them and set
``base_url``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

REGISTRY_FILE = "gallery_registry.json"
ZIP_SUFFIX = ".skullbundle.zip"


def registry() -> dict:
    """The packaged gallery registry (schema, skull id, base_url, bundles{name: entry})."""
    p = Path(__file__).with_name(REGISTRY_FILE)
    return json.loads(p.read_text())


def cache_dir() -> Path:
    d = os.environ.get("SKULL_TRANSPARENCY_CACHE")
    d = Path(d) if d else Path.home() / ".cache" / "skull_transparency"
    d = d / "gallery"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_targets(reg: dict | None = None) -> dict:
    """name -> entry for every bundle in the registry."""
    return dict((reg or registry())["bundles"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unzip_bundle(zpath: Path, dest: Path) -> Path:
    """Extract a bundle zip into ``dest`` and return the directory holding bundle.json."""
    tmp = dest.with_suffix(".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp)
    hits = sorted(tmp.rglob("bundle.json"))
    if not hits:
        shutil.rmtree(tmp)
        raise ValueError(f"{zpath.name} holds no bundle.json — not a Field Bundle zip")
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest / hits[0].relative_to(tmp).parent


def fetch(name: str, reg: dict | None = None, cache: Path | None = None,
          verbose: bool = True) -> Path:
    """Return a local Field Bundle directory for a named gallery target, fetching and
    checksum-verifying it if it is not already cached."""
    reg = reg or registry()
    try:
        entry = reg["bundles"][name]
    except KeyError:
        raise KeyError(f"no gallery bundle {name!r}; available: "
                       f"{', '.join(reg['bundles'])}") from None
    cache = Path(cache) if cache else cache_dir()
    dest = cache / name
    marker = dest / ".sha256"
    if marker.exists() and marker.read_text().strip() == entry["sha256"]:
        hits = sorted(dest.rglob("bundle.json"))
        if hits:
            return hits[0].parent

    # locate the zip: local gallery dir first, then the registry URL
    zname = entry["file"]
    local = os.environ.get("SKULL_TRANSPARENCY_GALLERY")
    if local and (Path(local) / zname).exists():
        zpath = Path(local) / zname
        if verbose:
            print(f"[gallery] using local {zpath}")
    else:
        base = entry.get("url") or (reg.get("base_url") and reg["base_url"].rstrip("/") + "/" + zname)
        if not base:
            raise FileNotFoundError(
                f"gallery bundle {name!r} is not published yet and no local gallery holds "
                f"{zname}. Either set $SKULL_TRANSPARENCY_GALLERY to a directory with the "
                f"zips, or compute the map yourself (GPU or the Colab notebook) and pass "
                f"--bundle instead.")
        zpath = cache / zname
        if verbose:
            print(f"[gallery] downloading {base}")
        urllib.request.urlretrieve(base, zpath)

    got = _sha256(zpath)
    if got != entry["sha256"]:
        raise ValueError(f"checksum mismatch for {zname}: got {got[:12]}…, "
                         f"registry says {entry['sha256'][:12]}… — refusing to use it")
    bdir = _unzip_bundle(zpath, dest)
    marker.write_text(entry["sha256"] + "\n")
    if verbose:
        print(f"[gallery] {name} -> {bdir}")
    return bdir
