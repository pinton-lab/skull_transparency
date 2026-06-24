#!/usr/bin/env python3
"""Surface-integral placement, end to end on the Halle/dentate bundle.

One outward time-reversal solve → a precomputed field → optimal window (a moving-cap surface
integral, no per-candidate wave solve), the time-reversal and phase-only drives, the Pinton-2012
radial projection to a transducer sphere, and the angular-spectrum focal-spot prediction.
Run with PYTHONPATH=../../src."""
import numpy as np
import skull_transparency as st
from skull_transparency import paths

BUNDLE = str(paths.bundle_dir())   # override via $SKULL_TR_DATA_ROOT


def main():
    bundle = st.load_bundle(BUNDLE)
    tm = st.compute_transparency_map(bundle)
    cf = bundle.element_complex_field()                       # complex element field (phase)
    print(f"complex field: {len(cf.G_win)} elements, f0={cf.f0/1e6:.2f} MHz, k0={cf.k0:.3f} rad/mm")

    # 1. OPTIMAL WINDOW from one map — argmax of J(S)=∫_S |G|^2 dS over all candidate windows.
    bp = st.place_bowl_optimal(tm, st.BowlConstraints(theta_max_deg=35.0, bowl_radius_mm=20.0))
    print(f"\noptimal window  MNI {np.round(bp.window_center_mni_mm, 1).tolist()} mm "
          f"(incidence {bp.incidence_deg:.1f}°, {bp.n_footprint_patches} patches); "
          f"focal-peak proxy √J = {bp.extras['p_max_proxy']:.3g}")

    # 2. DRIVES for the elements over the productive window — the high-coupling occiput cluster
    #    (here: array elements within 50 mm of the best-coupling element; in practice the
    #    sparse-array selection or a physical aperture footprint).
    pos_mm = cf.pos_fullres * cf.dx_mm
    hub = int(np.argmax(cf.E_win))
    idx = np.where(np.linalg.norm(pos_mm - pos_mm[hub], axis=1) <= 50.0)[0]
    u_tr, p_tr = st.drive_optimal(cf, idx)
    u_po, p_po = st.drive_phase_only(cf, idx)
    print(f"\n{len(idx)} elements in window: TR-optimal peak {p_tr:.3g}; "
          f"phase-only {p_po:.3g} ({p_po/p_tr:.3f}× = apodization loss 1/√(1+CV²))")

    # 3. RADIAL PROJECTION to a candidate transducer sphere (e.g. 90 mm focal length).
    G90 = st.project_to_sphere(cf.G_win, cf.radius_mm, 90.0, cf.k0)
    print(f"\nprojected field to R=90 mm sphere (Pinton 2012): "
          f"cap-energy density preserved (max relerr "
          f"{np.max(np.abs(st.energy_on_unit_sphere(G90,90.0*np.ones(len(G90)))-st.energy_on_unit_sphere(cf.G_win,cf.radius_mm))/st.energy_on_unit_sphere(cf.G_win,cf.radius_mm).max()):.1e})")

    # 4. FOCAL-SPOT prediction from the same single solve (no further simulation).
    fw = st.focal_fwhm(cf, idx)
    print(f"\npredicted focal spot: transverse FWHM "
          f"{fw['fwhm_lateral_mm']:.1f} × {fw['fwhm_elevation_mm']:.1f} mm, "
          f"axial DOF {fw['fwhm_axial_mm']:.0f} mm")
    print("  (order-correct; single-frequency overestimates the axial DOF — see README/manuscript)")


if __name__ == "__main__":
    main()
