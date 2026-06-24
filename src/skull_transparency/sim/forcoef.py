"""FOR^3D finite-difference delay coefficients.

Verbatim port of the polynomial set in
``fullwave2_3d/launch_fullwave2_3d_Aexp_noicmat.m`` (the active 3D branch). The
same 9 polynomials in the CFL-like ratio ``r`` produce both:

* ``d``    — evaluated once at ``r = cfl`` (written to ``d.dat``), and
* ``dmap`` — evaluated at ``r = (i-1+min(c))*dT/dX`` for each sound-speed bin
  (written to ``dmap.dat``).

Each polynomial is a degree-7 expression in ``r``; all arithmetic is float64,
matching MATLAB's double evaluation. Results are cast to float32 only when the
maps are written, so any sub-ULP double differences are washed out.
"""
from __future__ import annotations

import numpy as np

__all__ = ["for3d_rows", "build_d_matrix", "build_dmap"]


def for3d_rows(r):
    """Return the 9 FOR^3D coefficients at ratio ``r`` as a dict keyed by the
    1-based ``(row, col)`` index into MATLAB's ``d``/``dmap`` array.

    Keys: (2,1)..(9,1) and (2,2). Accepts a scalar or an ndarray ``r``.
    """
    return {
        (2, 1): (3.26627215252963e-3 * r**7 - 7.91679373564790e-4 * r**6 + 1.08663532410570e-3 * r**5 + 2.54974226454794e-2 * r**4 + 3.23083288193913e-5 * r**3 - 3.97704676886853e-1 * r**2 + 7.95584310128586e-8 * r + 1.25425295688331),
        (3, 1): (-2.83291379048757e-3 * r**7 + 8.52796449228369e-4 * r**6 - 9.45353822586534e-4 * r**5 - 8.82015372858580e-3 * r**4 - 2.81364895458027e-5 * r**3 + 6.73021045987599e-2 * r**2 - 6.93180036837075e-8 * r - 1.23448809066664e-1),
        (4, 1): (2.32775473203342e-3 * r**7 - 5.56793042789852e-4 * r**6 + 7.77649035879584e-4 * r**5 + 2.45547234243566e-3 * r**4 + 2.31537892801923e-5 * r**3 + 1.61900960524164e-2 * r**2 + 5.70523152308121e-8 * r + 3.46683979649506e-2),
        (5, 1): (-1.68883462553539e-3 * r**7 + 3.03535823592644e-4 * r**6 - 5.64777117315819e-4 * r**5 + 2.44582905523866e-4 * r**4 - 1.68215579314751e-5 * r**3 - 2.62344345204941e-2 * r**2 - 4.14559953526389e-8 * r - 1.19918511290930e-2),
        (6, 1): (1.08994931098070e-3 * r**7 - 1.41445142143525e-4 * r**6 + 3.64794490139160e-4 * r**5 - 8.86057426195227e-4 * r**4 + 1.08681882832738e-5 * r**3 + 2.07238558666603e-2 * r**2 + 2.67876079477806e-8 * r + 4.17058420250698e-3),
        (7, 1): (-6.39950124405340e-4 * r**7 + 6.06079815415080e-5 * r**6 - 2.14633466007892e-4 * r**5 + 6.84580412267934e-4 * r**4 - 6.39907927898092e-6 * r**3 - 1.29825288653404e-2 * r**2 - 1.57775422151124e-8 * r - 1.29998325971518e-3),
        (8, 1): (2.92716539609611e-4 * r**7 - 1.87446062803024e-5 * r**6 + 9.85389372183761e-5 * r**5 - 2.40360290348543e-4 * r**4 + 2.94166215515130e-6 * r**3 + 5.57066438452790e-3 * r**2 + 7.25741366376659e-9 * r + 3.18698432679400e-4),
        (9, 1): (-6.42183857909518e-5 * r**7 + 3.38552867751042e-6 * r**6 - 2.17377151411164e-5 * r**5 + 4.98269067389945e-5 * r**4 - 6.50197868987757e-7 * r**3 - 1.19096089679178e-3 * r**2 - 1.60559948991172e-9 * r - 4.57795411807702e-5),
        (2, 2): (-4.47723278782936e-5 * r**7 - 7.69502473399932e-5 * r**6 - 1.41765498250133e-5 * r**5 - 2.54672045901272e-3 * r**4 - 4.14343385915353e-7 * r**3 + 5.00210047924752e-2 * r**2 - 1.01220354410507e-9 * r - 8.07139347787336e-8),
    }


def build_d_matrix(cfl: float) -> np.ndarray:
    """Build MATLAB's ``d`` (9x2, float64) evaluated at ``r = cfl``.

    Unset entries are 0 (MATLAB auto-zero-fill). ``d.dat`` is then
    ``fwrite(d','float')`` = row-major (C-order) float32 = ``d.ravel('C')``.
    """
    d = np.zeros((9, 2), dtype=np.float64)
    rows = for3d_rows(float(cfl))
    for (ri, ci), val in rows.items():
        d[ri - 1, ci - 1] = val
    return d


def build_dmap(minc: float, maxc: float, dT: float, dX: float) -> np.ndarray:
    """Build MATLAB's ``dmap`` (9, 2, ndmap), float32.

    MATLAB preallocates the 3rd dim to ``round(maxc)-round(minc)`` but the loop
    runs ``i = 1 .. round(maxc-minc)+1``, auto-extending the array; the final
    ``ndmap = size(dmap,3)`` is therefore the larger of the two. ``r`` uses the
    *raw* (un-rounded) ``min(c)``.
    """
    from .mlcompat import matlab_round

    prealloc = int(matlab_round(maxc) - matlab_round(minc))
    loopmax = int(matlab_round(maxc - minc)) + 1
    ndmap = max(prealloc, loopmax)
    dmap = np.zeros((9, 2, ndmap), dtype=np.float32)
    i = np.arange(1, loopmax + 1)
    r = ((i - 1) + minc) * dT / dX            # float64
    rows = for3d_rows(r)
    for (ri, ci), val in rows.items():
        dmap[ri - 1, ci - 1, :loopmax] = np.asarray(val, dtype=np.float64)
    return dmap
