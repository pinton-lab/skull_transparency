"""Fetch the CUDA solver binary on first use, with its license shown up front.

Computing a new transparency map runs the ``fullwave2-ultra`` ``bench_3d_opt`` binary
(Linux x86-64, NVIDIA GPU; distributed under a noncommercial license, separate from this
package's Apache-2.0). It is never bundled: :func:`ensure_solver` resolves it in this
order and returns a path —

1. the ``sim`` package's own channels (``$FULLWAVE2_BIN``, the ``fullwave2_ultra``
   package resolver, ``$FULLWAVE2_BIN_DIR`` / a sibling checkout) — already-configured
   sources, used as-is;
2. the local cache (``~/.cache/skull_transparency/solver``), if its license was accepted;
3. a fresh fetch: shallow-clone the ``fullwave2-ultra`` repository, show the binary
   license, and require acceptance (``accept_license=True``, the
   ``$SKULL_TRANSPARENCY_ACCEPT_SOLVER_LICENSE=1`` environment variable, or an
   interactive yes) before caching the binary.
"""
from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

BINARY = "bench_3d_opt"
REPO_URL = "https://github.com/pinton-lab/fullwave2-ultra.git"
LICENSE_FILE = "LICENSE-binaries.txt"
ACCEPT_ENV = "SKULL_TRANSPARENCY_ACCEPT_SOLVER_LICENSE"


def cache_dir() -> Path:
    d = os.environ.get("SKULL_TRANSPARENCY_CACHE")
    d = Path(d) if d else Path.home() / ".cache" / "skull_transparency"
    d = d / "solver"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _platform_check():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError(
            f"the CUDA solver is a Linux x86-64 binary; this machine is "
            f"{platform.system()}/{platform.machine()}. Compute maps on a Linux box with "
            "an NVIDIA GPU (Windows: inside WSL2) or with the Colab notebook, then "
            "explore the bundle here.")


def _configured_binary() -> str | None:
    """A binary from the already-configured channels (env / package / sibling), if any."""
    from .sim.launchers import _resolve_solver_binary
    p = _resolve_solver_binary()
    return p if p and os.path.exists(p) else None


def _license_gate(license_path: Path, accept_license: bool, interactive: bool) -> bool:
    text = license_path.read_text(errors="replace") if license_path.exists() else ""
    head = "\n".join(text.splitlines()[:12])
    print("\nThe solver binary is distributed under its own (noncommercial) license,\n"
          "separate from this package:\n")
    print(head + ("\n  ..." if text.count("\n") > 12 else ""))
    print(f"\n(full text: {license_path})")
    if accept_license or os.environ.get(ACCEPT_ENV) == "1":
        print("license accepted (flag/environment).")
        return True
    if interactive and sys.stdin.isatty():
        return input("accept the solver license? [y/N] ").strip().lower() in ("y", "yes")
    print(f"not accepted -- re-run with --accept-license (or {ACCEPT_ENV}=1).")
    return False


def ensure_solver(accept_license: bool = False, interactive: bool = True,
                  verbose: bool = True) -> str:
    """Return a runnable ``bench_3d_opt`` path, fetching (behind the license gate) if no
    configured source has one. Raises with an actionable message otherwise."""
    p = _configured_binary()
    if p:
        return p
    _platform_check()

    cached = cache_dir() / BINARY
    marker = cache_dir() / ".license_accepted"
    if cached.exists() and marker.exists():
        return str(cached)

    if verbose:
        print(f"[solver] no configured solver found -- fetching {REPO_URL} (shallow) ...")
    with tempfile.TemporaryDirectory(prefix="fw2u_") as td:
        r = subprocess.run(["git", "clone", "--depth", "1", REPO_URL, td],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                "could not clone the solver repository (private repos need your GitHub "
                f"credentials):\n{r.stderr.strip()[-400:]}\n"
                "Alternatively set FULLWAVE2_BIN to an existing bench_3d_opt.")
        hits = sorted(Path(td).rglob(BINARY))
        if not hits:
            raise FileNotFoundError(f"{BINARY} not found in the cloned repository")
        lic = sorted(Path(td).rglob(LICENSE_FILE))
        lic_src = lic[0] if lic else Path(td) / "LICENSE"
        lic_dst = cache_dir() / LICENSE_FILE
        if lic_src.exists():
            shutil.copy2(lic_src, lic_dst)
        if not _license_gate(lic_dst, accept_license, interactive):
            raise PermissionError("solver license not accepted; nothing was installed")
        shutil.copy2(hits[0], cached)
        cached.chmod(cached.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        marker.write_text("accepted\n")
    if verbose:
        print(f"[solver] cached {cached}")
    return str(cached)
