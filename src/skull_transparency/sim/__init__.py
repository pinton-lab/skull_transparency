"""Pure-Python rewrite of the MATLAB ``fullwave2`` time-reversal launchers.

This package replaces the ``hemisphere_tr/launchers/fullwave2_launch_*.m``
scripts (and the ``fullwave2_3d`` helpers they call) with NumPy/SciPy code that
generates **bit-identical** simulation input ``.dat`` files. The external CUDA
solver is the fullwave2-ultra ``bench_3d_opt`` (byte-identical genout to the retired
``fullwave2_3d_Aexp_genout_cuda_aperturegrowth_opt``); pass ``run_solver=True`` to a
launcher to invoke it (resolved via ``$FULLWAVE2_BIN`` or the ``fullwave2_ultra`` package).

Entry points (see :mod:`.launchers` for full docstrings)::

    from skull_transparency.sim import launchers as L
    L.launch_outward(sim_dir, out_root)              # outward TR (virtual source)
    L.launch_inward_windowed(sim_dir, out_root)      # windowed inward refocus
    L.launch_inward_focalbox(sim_dir, out_root)      # + full-res focal box
    L.launch_skullonly(sim_dir, out_root)            # whole-skull, volume recorder
    L.launch_skullonly_array(sim_dir, out_root)      # whole-skull + array
    L.launch_subset_focalbox(sim_dir, out_root, mode='tr'|'geo')
    ...

A CLI mirrors the per-script env vars: ``python -m skull_transparency.sim ...``.
"""
from . import fwio, mlcompat, forcoef, launch_core, _common, launchers, prepare, extract

__all__ = ["fwio", "mlcompat", "forcoef", "launch_core", "_common", "launchers", "prepare", "extract"]
