"""One-command placement report: a self-contained HTML page with the numbers and maps an
experimenter needs to take away. Everything is embedded (base64 PNGs) and the CSS carries
print rules, so the file mails/archives as-is and prints cleanly to PDF.

The presentation follows the manuscript's conventions:

* colour is transmitted **amplitude in dB** (log |p|, re the 98th-percentile patch) — the
  whole-skull convention; linear intensity crushes the map to black;
* the headline view is the **superior-pole, anterior-centred equirectangular unwrap** of
  the skull (no occlusion, prints well), amplitude-weighted with a coverage/confidence
  mask, with the EEG 10-20 sites overlaid so the window reads in head coordinates;
* a second unwrap shows the **placement objective** \\sqrt{J_w} — the incidence-weighted,
  legality-masked moving-footprint integral the placement search maximises. Its bright
  lobe is *why* the window won; the lobe's breadth is the placement margin;
* the table quotes the **access fraction**: the solid angle (as a fraction of 4\\pi about
  the target) subtended by good-transmission, incidence-legal skull.

Headless-safe (matplotlib Agg)."""
from __future__ import annotations

import base64
import datetime
import io
from pathlib import Path

import numpy as np

from .atlas_targets import EEG_SITES_MNI, nearest_eeg_site

_NLON, _NLAT = 360, 180
_SIG = 2.0                    # unwrap smoothing (bins)
_GOOD_FRAC = 0.03             # good-transmission floor: Pmax > 3% of max (render convention)


def _fig_png(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _db_amplitude(value):
    amp = np.sqrt(np.maximum(np.asarray(value, float), 0.0))
    ref = np.percentile(amp, 98.0) or 1.0
    return 20.0 * np.log10(np.maximum(amp / ref, 1e-2))


def _anat_axes_vox(reg):
    """Superior/anterior/left unit vectors in the full-res voxel frame."""
    o = np.asarray(reg.mni_to_fullres(np.zeros(3)), float)
    ax = []
    for v in ((0, 0, 1.0), (0, 1.0, 0), (-1.0, 0, 0)):        # S, A, L in world RAS
        d = np.asarray(reg.mni_to_fullres(np.asarray(v, float)), float) - o
        ax.append(d / (np.linalg.norm(d) or 1.0))
    return ax


def _lonlat(dirs, S, A, L):
    lat = np.degrees(np.arcsin(np.clip(dirs @ S, -1, 1)))
    lon = np.degrees(np.arctan2(dirs @ L, dirs @ A))
    return lon, lat


def _unwrap_field(lon, lat, weights, values):
    """Amplitude-weighted equirectangular field + confidence alpha (wrapped smoothing)."""
    from scipy.ndimage import gaussian_filter
    lon_e = np.linspace(-180, 180, _NLON + 1)
    lat_e = np.linspace(-90, 90, _NLAT + 1)
    num = np.histogram2d(lon, lat, bins=[lon_e, lat_e], weights=weights * values)[0]
    den = np.histogram2d(lon, lat, bins=[lon_e, lat_e], weights=weights)[0]
    field = (gaussian_filter(num, _SIG, mode="wrap")
             / np.maximum(gaussian_filter(den, _SIG, mode="wrap"), 1e-9))
    conf = gaussian_filter(den, _SIG, mode="wrap")
    # opaque wherever coverage is solid (>15% of peak), fading only at the sparse rim --
    # a global sqrt ramp dims well-covered regions and washes the whole map out
    alpha = np.clip(conf / (0.15 * (conf.max() or 1.0)), 0, 1) ** 0.7
    alpha[conf < 0.02 * (conf.max() or 1.0)] = 0.0
    return field, alpha


def _objective_field(tmap, radius_mm, theta_max_deg, n_candidates=4000):
    """Per-patch moving-footprint objective \\sqrt{J_w}, normalised to its max.

    J_w(window) = sum over the bowl footprint of cos^2(theta) * Ipk (RAW intensity — the
    placement quantity; the 1/r^2-corrected map is visualisation only), with patches past
    the incidence cutoff excluded. Evaluated on a candidate subsample and painted onto
    every patch by nearest candidate."""
    from scipy.spatial import cKDTree
    dx = tmap.registration.dx_mm if tmap.registration is not None else 1.0
    P = np.asarray(tmap.surf_vox, float) * dx
    cosi = np.clip(np.einsum("ij,ij->i", np.asarray(tmap.rhat, float),
                             np.asarray(tmap.true_normal, float)), 0.0, 1.0)
    legal = cosi >= np.cos(np.radians(theta_max_deg))
    w = np.where(legal, cosi ** 2 * np.maximum(np.asarray(tmap.Ipk_Wcm2, float), 0.0), 0.0)
    cand = np.arange(len(P))[:: max(1, len(P) // n_candidates)]
    tree = tree_all = cKDTree(P)
    J = np.array([w[i].sum() for i in tree_all.query_ball_point(P[cand], r=radius_mm,
                                                                workers=-1)])
    ctree = cKDTree(P[cand])
    obj = np.sqrt(np.maximum(J, 0.0))[ctree.query(P, workers=-1)[1]]
    return obj / (obj.max() or 1.0), legal


def _access_fraction(lon, lat, keep):
    """Solid angle (fraction of 4pi) covered by the kept patches, via occupied
    equirectangular bins weighted with cos(lat)."""
    lon_e = np.linspace(-180, 180, 91)
    lat_e = np.linspace(-90, 90, 46)
    H = np.histogram2d(lon[keep], lat[keep], bins=[lon_e, lat_e])[0] > 0
    latc = np.radians((lat_e[:-1] + lat_e[1:]) / 2)
    wbin = np.cos(latc)                                   # d(solid angle) per bin row
    return float((H * wbin[None, :]).sum() / (len(lon_e) - 1) / wbin.sum())


def _style_unwrap(ax):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels(["post.", "right", "anterior", "left", "post."], fontsize=8)
    ax.set_yticks([-60, 0, 60])
    ax.set_yticklabels(["inferior", "equator", "superior"], fontsize=8)


def _unwrap_figure(tmap, placement, radius_mm, theta_max_deg):
    """Two-row equirectangular figure: transparency (dB) + placement objective, with the
    EEG 10-20 sites and the chosen window overlaid. Returns (fig, access_fraction)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reg = tmap.registration
    S, A, L = _anat_axes_vox(reg)
    rhat = np.asarray(tmap.rhat, float)
    lon, lat = _lonlat(rhat, S, A, L)
    Pmax = np.asarray(tmap.Pmax, float)
    good = Pmax > _GOOD_FRAC * (Pmax.max() or 1.0)
    wamp = np.where(good, Pmax / (Pmax.max() or 1.0), 0.0)

    db = _db_amplitude(tmap.value)
    fld_t, alpha = _unwrap_field(lon, lat, wamp, db)
    obj, legal = _objective_field(tmap, radius_mm, theta_max_deg)
    fld_o, _ = _unwrap_field(lon, lat, wamp, obj)
    access = _access_fraction(lon, lat, good & legal)

    # markers: chosen window + EEG sites, in the same target-centred direction space
    tgt_vox = np.asarray(reg.target_fullres_voxel, float)
    win_vox = np.asarray(reg.mni_to_fullres(np.asarray(placement.window_center_mni_mm,
                                                       float)), float)
    dw = win_vox - tgt_vox
    wlon, wlat = _lonlat((dw / np.linalg.norm(dw))[None, :], S, A, L)
    sites, slon, slat = [], [], []
    for name, pos in EEG_SITES_MNI.items():
        d = np.asarray(reg.mni_to_fullres(np.asarray(pos, float)), float) - tgt_vox
        lo, la = _lonlat((d / np.linalg.norm(d))[None, :], S, A, L)
        sites.append(name)
        slon.append(lo[0])
        slat.append(la[0])

    vt = np.percentile(db, [25.0, 99.0])
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.6))
    for ax, fld, (vlo, vhi), cmap, ttl, clabel in (
            (axes[0], fld_t, vt, "inferno", "Skull transparency — unwrapped "
             "(superior-pole equirectangular, anterior-centred)",
             "transparency amplitude (dB re 98th pct)"),
            (axes[1], fld_o, (0.0, 1.0), "viridis",
             "Placement objective $\\sqrt{J_w}$ — what the window search maximises "
             "(bright lobe breadth = placement margin)", "objective (normalised)")):
        rgba = plt.get_cmap(cmap)(np.clip((fld - vlo) / (vhi - vlo or 1.0), 0, 1))
        rgba[..., 3] = alpha
        ax.set_facecolor("#d9dde3")
        im = ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower",
                       extent=[-180, 180, -90, 90], aspect="auto")
        ax.scatter(slon, slat, s=10, c="white", edgecolors="k", linewidths=0.4, zorder=5)
        for n, lo, la in zip(sites, slon, slat):
            ax.annotate(n, (lo, la), textcoords="offset points", xytext=(3, 3),
                        fontsize=6.5, color="k", zorder=6)
        ax.scatter(wlon, wlat, marker="*", s=260, c="red", edgecolors="k",
                   linewidths=0.7, zorder=7, label="chosen window")
        _style_unwrap(ax)
        ax.set_title(ttl, fontsize=10)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=vlo, vmax=vhi))
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01).set_label(clabel, fontsize=8)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig, access


def _map3d_figure(tmap):
    """The transparency map on the skull surface from four culled viewpoints."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    P = np.asarray(tmap.surf_mni_mm() if tmap.registration is not None else tmap.surf_vox,
                   float)
    db = _db_amplitude(tmap.value)
    vlo, vhi = np.percentile(db, 25.0), np.percentile(db, 99.0)
    rhat = np.asarray(tmap.rhat, float)
    fig = plt.figure(figsize=(16, 4.2))
    sc = None
    for k, (elev, azim) in enumerate([(18, -60), (18, 60), (18, 180), (78, -90)]):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        e, a = np.radians(elev), np.radians(azim)
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        m = rhat @ cam > 0.0                                  # camera-facing hemisphere
        sc = ax.scatter(*P[m].T, c=db[m], cmap="inferno", vmin=vlo, vmax=vhi, s=4,
                        linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        lo, hi = P.min(0), P.max(0)
        mid, sp = (hi + lo) / 2, (hi - lo).max() / 2
        for kk, mm in zip("xyz", mid):
            getattr(ax, f"set_{kk}lim")(mm - sp, mm + sp)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_ticklabels([])
        ax.grid(False)
    fig.colorbar(sc, ax=fig.axes, fraction=0.012, pad=0.01,
                 label="transparency amplitude (dB re 98th pct)")
    return fig


def write_report(tmap, placement, out_html, *, title="Skull transparency placement",
                 target_name=None, bowl_radius_mm=32.0, theta_max_deg=35.0) -> Path:
    """Write the self-contained HTML report; returns its path. ``bowl_radius_mm`` /
    ``theta_max_deg`` parameterise the objective panel and the access fraction (pass the
    values the placement used)."""
    from .position_tool import preview_placement
    import matplotlib
    matplotlib.use("Agg")

    out = Path(out_html)
    tgt = np.asarray(placement.target_mni_mm, float)
    win = np.asarray(placement.window_center_mni_mm, float)
    site, site_mm = nearest_eeg_site(win)

    unwrap_fig, access = _unwrap_figure(tmap, placement, bowl_radius_mm, theta_max_deg)
    unwrap_png = _fig_png(unwrap_fig)
    map_png = _fig_png(_map3d_figure(tmap))
    tmp = out.with_suffix(".placement.png")
    preview_placement(tmap, placement, out_png=str(tmp), title=title)
    place_png = base64.b64encode(tmp.read_bytes()).decode()
    tmp.unlink()

    fmt = lambda v: "(" + ", ".join(f"{x:.1f}" for x in np.asarray(v, float)) + ") mm"
    rows = [
        ("Target" + (f" ({target_name})" if target_name else ""), fmt(tgt) + " MNI"),
        ("Window centre", fmt(win) + " MNI"),
        ("Window ≈ EEG 10-20 site", f"{site}  ({site_mm:.0f} mm away)"),
        ("Window-to-target distance", f"{np.linalg.norm(win - tgt):.1f} mm"),
        ("Beam incidence at window", f"{float(placement.incidence_deg):.1f} deg "
                                     f"(legal limit {theta_max_deg:.0f} deg)"),
        ("Transparency score", f"{float(placement.transparency_score):.3f}"),
        ("Access (legal skull solid angle)", f"{100 * access:.0f}% of 4π about "
                                             "the target"),
        ("Surface patches", f"{len(np.asarray(tmap.surf_vox)):,}"),
        ("Median target depth", f"{float(np.median(tmap.rad_mm)):.0f} mm"),
    ]
    trs = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    stamp = datetime.date.today().isoformat()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto;
        color: #1d1d1f; }}
 h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
 table {{ border-collapse: collapse; }} td {{ border: 1px solid #ccd; padding: 5px 12px; }}
 td:first-child {{ background: #f4f6f9; font-weight: 600; }}
 img {{ max-width: 100%; }} .note {{ color: #445; font-size: 0.9em; background: #f4f6f9;
        border-left: 3px solid #2c6fbf; padding: 6px 10px; }}
 footer {{ color: #778; font-size: 0.85em; margin-top: 2em; }}
 @media print {{
   body {{ max-width: none; margin: 0.5cm; }}
   h2 {{ page-break-before: always; }} h2:first-of-type {{ page-break-before: avoid; }}
   img {{ page-break-inside: avoid; }}
 }}
</style></head><body>
<h1>{title}</h1>
<h2>Placement summary</h2>
<table>{trs}</table>
<h2>Where the skull transmits, and where to place — unwrapped</h2>
<img src="data:image/png;base64,{unwrap_png}" alt="unwrapped transparency + objective"/>
<p class="note">Top: transmitted amplitude (dB, 1/r&sup2;-corrected) — for <em>seeing</em>
bone transmission. Bottom: the placement objective &radic;J<sub>w</sub> (incidence-weighted
<em>raw</em> delivered intensity integrated over the bowl footprint, incidence-illegal
patches excluded) — the quantity the window search maximises; using the corrected map for
placement would over-reward far, thin-bone windows. Grey = no coherent coverage.</p>
<h2>Transparency map — 3-D views</h2>
<img src="data:image/png;base64,{map_png}" alt="transparency map 3-D"/>
<h2>Chosen placement</h2>
<img src="data:image/png;base64,{place_png}" alt="placement"/>
<footer>skull-transparency report — generated {stamp}. Coordinates are MNI RAS mm. The map
comes from one full-wave time-reversal solve (virtual source at the target); placement and
every figure here are post-processing on that recorded field. This file is self-contained
— print to PDF for archival.</footer>
</body></html>
"""
    out.write_text(html)
    return out
