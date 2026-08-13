"""One-command placement report: a self-contained HTML page with the numbers and maps an
experimenter needs to take away. Everything is embedded (base64 PNGs) and the CSS carries
print rules, so the file mails/archives as-is and prints cleanly to PDF.

The report follows the manuscript's logical structure and conventions —

1. placement summary (target, window, EEG site, access, tolerance);
2. surface coupling / bone transmission (manuscript Fig. 5): 3-D views + the
   superior-pole, anterior-centred equirectangular unwrap, amplitude in dB (log |p|);
3. beam incidence and the legality limit (the placement criteria);
4. the placement objective \\sqrt{J_w} (Eqs. 7 & 14; Fig. 8) and the chosen window;
5. ranked alternative windows (Table 1 criteria);
6. methods, with the manuscript figure/equation/table references.

Headless-safe (matplotlib Agg)."""
from __future__ import annotations

import base64
import datetime
import io
from pathlib import Path

import numpy as np

from .atlas_targets import EEG_SITES_MNI, nearest_eeg_site

_MANUSCRIPT = ('Pinton, "Whole-skull acoustic transparency from a single time-reversal '
               'solve for reciprocity-based transducer placement and aperture optimization '
               'in transcranial focused ultrasound" (2026)')

_NLON, _NLAT = 240, 120      # coarse enough that an ECCENTRIC target's far side
#                              (sparse angular sampling) still fills its bins
_SIG = 1.5                    # unwrap smoothing (bins)
_GOOD_FRAC = 0.03             # good-transmission floor on the distance-corrected amplitude


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
    """Amplitude-weighted equirectangular field + occupancy alpha (wrapped smoothing)."""
    from scipy.ndimage import gaussian_filter
    lon_e = np.linspace(-180, 180, _NLON + 1)
    lat_e = np.linspace(-90, 90, _NLAT + 1)
    num = np.histogram2d(lon, lat, bins=[lon_e, lat_e], weights=weights * values)[0]
    den = np.histogram2d(lon, lat, bins=[lon_e, lat_e], weights=weights)[0]
    field = (gaussian_filter(num, _SIG, mode="wrap")
             / np.maximum(gaussian_filter(den, _SIG, mode="wrap"), 1e-9))
    # ABSOLUTE occupancy: about one patch within the smoothing kernel counts as covered.
    # Any threshold relative to the max fails for eccentric targets, whose near side has
    # few patches per angular bin (each subtends a large angle) while the far side packs
    # hundreds -- density relative to the far side would mask the window's neighbourhood.
    cnt = np.histogram2d(lon, lat, bins=[lon_e, lat_e])[0]
    conf = gaussian_filter(cnt, _SIG, mode="wrap")
    alpha = np.clip(conf / 0.5, 0, 1) ** 0.7
    alpha[conf < 0.05] = 0.0
    return field, alpha


def _objective_field(tmap, radius_mm, theta_max_deg, n_candidates=4000):
    """Per-patch moving-footprint objective \\sqrt{J_w} (normalised), + legality, cos(i),
    and the candidate subsample it was evaluated on.

    J_w(window) = sum over the bowl-radius footprint of cos^2(theta) * Ipk (the RAW peak
    intensity — the reciprocity-correct placement quantity), incidence-illegal patches
    excluded; manuscript Eqs. (7) and (14)."""
    from scipy.spatial import cKDTree
    dx = tmap.registration.dx_mm if tmap.registration is not None else 1.0
    P = np.asarray(tmap.surf_vox, float) * dx
    cosi = np.clip(np.einsum("ij,ij->i", np.asarray(tmap.rhat, float),
                             np.asarray(tmap.true_normal, float)), 0.0, 1.0)
    legal = cosi >= np.cos(np.radians(theta_max_deg))
    w = np.where(legal, cosi ** 2 * np.maximum(np.asarray(tmap.Ipk_Wcm2, float), 0.0), 0.0)
    cand = np.arange(len(P))[:: max(1, len(P) // n_candidates)]
    tree_all = cKDTree(P)
    J = np.array([w[i].sum() for i in tree_all.query_ball_point(P[cand], r=radius_mm,
                                                                workers=-1)])
    ctree = cKDTree(P[cand])
    obj = np.sqrt(np.maximum(J, 0.0))[ctree.query(P, workers=-1)[1]]
    return obj / (obj.max() or 1.0), legal, cosi, cand


def _alternate_windows(P_mm, obj, cand, win_mm, n=4, sep_mm=30.0):
    """Ranked fallback windows: the top objective candidates, greedily separated from the
    chosen window and from each other by ``sep_mm`` (surface euclidean). Returns patch
    indices, best first."""
    order = cand[np.argsort(obj[cand])[::-1]]
    picked, kept = [np.asarray(win_mm, float)], []
    for i in order:
        if all(np.linalg.norm(P_mm[i] - q) >= sep_mm for q in picked):
            kept.append(int(i))
            picked.append(P_mm[i])
            if len(kept) >= n:
                break
    return kept


def _tolerance_mm(P_mm, obj, win_mm, thresh=0.9, link_mm=3.0):
    """Placement tolerance: how far the >= ``thresh``-of-max objective region extends from
    the chosen window, within its own connected lobe (BFS over ``link_mm`` adjacency, so a
    symmetric far-side lobe does not inflate the number)."""
    from scipy.spatial import cKDTree
    q = np.flatnonzero(obj >= thresh * (obj.max() or 1.0))
    if len(q) == 0:
        return 0.0
    Pq = P_mm[q]
    tq = cKDTree(Pq)
    seed = int(np.argmin(np.linalg.norm(Pq - np.asarray(win_mm, float), axis=1)))
    seen = {seed}
    frontier = [seed]
    while frontier:
        nxt = set()
        for j in tq.query_ball_point(Pq[frontier], r=link_mm, workers=-1):
            nxt.update(j)
        frontier = list(nxt - seen)
        seen.update(nxt)
    comp = Pq[sorted(seen)]
    return float(np.linalg.norm(comp - np.asarray(win_mm, float), axis=1).max())


def _access_fraction(lon, lat, keep):
    """Solid angle (fraction of 4pi) covered by the kept patches, via occupied
    equirectangular bins weighted with cos(lat) — the manuscript Table 1 access."""
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


class _UnwrapData:
    """Everything shared by the unwrap panels, computed once."""

    def __init__(self, tmap, placement, radius_mm, theta_max_deg):
        reg = tmap.registration
        self.S, self.A, self.L = _anat_axes_vox(reg)
        rhat = np.asarray(tmap.rhat, float)
        self.lon, self.lat = _lonlat(rhat, self.S, self.A, self.L)
        Pmax = np.asarray(tmap.Pmax, float)
        self.wamp = Pmax / (Pmax.max() or 1.0)
        # good transmission on the DISTANCE-CORRECTED amplitude (proximity-free): a raw
        # Pmax floor cuts the whole far side for a shallow eccentric target
        amp_c = np.sqrt(np.maximum(np.asarray(tmap.value, float), 0.0))
        good = amp_c > _GOOD_FRAC * (np.percentile(amp_c, 98.0) or 1.0)

        self.db = _db_amplitude(tmap.value)
        self.fld_t, self.alpha = _unwrap_field(self.lon, self.lat, self.wamp, self.db)
        self.obj, legal, cosi, cand = _objective_field(tmap, radius_mm, theta_max_deg)
        self.fld_o, _ = _unwrap_field(self.lon, self.lat, self.wamp, self.obj)
        self.inc_deg = np.degrees(np.arccos(np.clip(cosi, 0, 1)))
        self.fld_i, _ = _unwrap_field(self.lon, self.lat, self.wamp, self.inc_deg)
        self.access = _access_fraction(self.lon, self.lat, good & legal)

        dx = reg.dx_mm
        tgt_vox = np.asarray(reg.target_fullres_voxel, float)
        win_vox = np.asarray(reg.mni_to_fullres(
            np.asarray(placement.window_center_mni_mm, float)), float)
        dw = win_vox - tgt_vox
        self.wlon, self.wlat = _lonlat((dw / np.linalg.norm(dw))[None, :],
                                       self.S, self.A, self.L)
        P_mm = np.asarray(tmap.surf_vox, float) * dx
        self.alt_idx = _alternate_windows(P_mm, self.obj, cand, win_vox * dx)
        self.alon, self.alat = (_lonlat(rhat[self.alt_idx], self.S, self.A, self.L)
                                if self.alt_idx else (np.array([]), np.array([])))
        self.tolerance_mm = _tolerance_mm(P_mm, self.obj, win_vox * dx)
        self.sites, self.slon, self.slat = [], [], []
        for name, pos in EEG_SITES_MNI.items():
            d = np.asarray(reg.mni_to_fullres(np.asarray(pos, float)), float) - tgt_vox
            lo, la = _lonlat((d / np.linalg.norm(d))[None, :], self.S, self.A, self.L)
            self.sites.append(name)
            self.slon.append(lo[0])
            self.slat.append(la[0])


def _unwrap_panel(u, field, vlim, cmap, title, clabel, contour_at=None):
    """One equirectangular panel with the EEG grid, chosen window, and alternates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    vlo, vhi = vlim
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    rgba = plt.get_cmap(cmap)(np.clip((field - vlo) / ((vhi - vlo) or 1.0), 0, 1))
    rgba[..., 3] = u.alpha
    ax.set_facecolor("#d9dde3")
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower",
              extent=[-180, 180, -90, 90], aspect="auto")
    if contour_at is not None:
        lc = np.linspace(-180, 180, _NLON + 1)[:-1] + 180.0 / _NLON
        bc = np.linspace(-90, 90, _NLAT + 1)[:-1] + 90.0 / _NLAT
        ax.contour(lc, bc, np.where(u.alpha.T > 0, field.T, np.nan),
                   levels=[contour_at], colors="k", linestyles="--", linewidths=1.0)
    ax.scatter(u.slon, u.slat, s=10, c="white", edgecolors="k", linewidths=0.4, zorder=5)
    for n, lo, la in zip(u.sites, u.slon, u.slat):
        ax.annotate(n, (lo, la), textcoords="offset points", xytext=(3, 3), fontsize=6.5,
                    color="white", zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="black")])
    ax.scatter(u.wlon, u.wlat, marker="*", s=260, c="red", edgecolors="k",
               linewidths=0.7, zorder=7, label="chosen window")
    for r, (lo, la) in enumerate(zip(u.alon, u.alat), start=2):
        ax.scatter([lo], [la], marker="*", s=120, c="#ff9f1c", edgecolors="k",
                   linewidths=0.5, zorder=7)
        ax.annotate(str(r), (lo, la), textcoords="offset points", xytext=(5, -9),
                    fontsize=7.5, fontweight="bold", color="#7a4a00", zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
    _style_unwrap(ax)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vlo, vmax=vhi))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01).set_label(clabel, fontsize=8)
    fig.tight_layout()
    return fig


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
    ``theta_max_deg`` parameterise the objective, alternates, and access fraction (pass
    the values the placement used)."""
    from .position_tool import preview_placement
    import matplotlib
    matplotlib.use("Agg")

    out = Path(out_html)
    tgt = np.asarray(placement.target_mni_mm, float)
    win = np.asarray(placement.window_center_mni_mm, float)
    site, site_mm = nearest_eeg_site(win)

    u = _UnwrapData(tmap, placement, bowl_radius_mm, theta_max_deg)
    vt = tuple(np.percentile(u.db, [25.0, 99.0]))
    png_t = _fig_png(_unwrap_panel(
        u, u.fld_t, vt, "inferno",
        "Skull transparency — unwrapped (superior-pole equirectangular, "
        "anterior-centred; manuscript Fig. 5)",
        "transparency amplitude (dB re 98th pct)"))
    png_i = _fig_png(_unwrap_panel(
        u, u.fld_i, (0.0, 60.0), "coolwarm",
        f"Beam incidence on the bone — dashed contour = the {theta_max_deg:.0f}° "
        "legality limit", "incidence (deg)", contour_at=theta_max_deg))
    png_o = _fig_png(_unwrap_panel(
        u, u.fld_o, (0.0, 1.0), "viridis",
        "Placement objective $\\sqrt{J_w}$ — what the window search maximises "
        "(manuscript Fig. 8)", "objective (normalised)"))
    png_3d = _fig_png(_map3d_figure(tmap))
    tmp = out.with_suffix(".placement.png")
    preview_placement(tmap, placement, out_png=str(tmp), title=title)
    png_pl = base64.b64encode(tmp.read_bytes()).decode()
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
        ("Access (legal skull solid angle)", f"{100 * u.access:.0f}% of 4π about "
                                             "the target"),
        ("Aperture margin (vs f/0.6 demand)", f"{u.access / 0.25:.1f}× the ~25% solid "
                                              "angle an f/0.6 bowl subtends"),
        ("Placement tolerance", f"the ≥90%-objective lobe extends "
                                f"{u.tolerance_mm:.0f} mm from the chosen window"),
        ("Surface patches", f"{len(np.asarray(tmap.surf_vox)):,}"),
        ("Median target depth", f"{float(np.median(tmap.rad_mm)):.0f} mm"),
    ]
    trs = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)

    reg = tmap.registration
    fmtv = lambda v: "(" + ", ".join(f"{x:.0f}" for x in np.asarray(v, float)) + ")"
    arows = []
    for rank, i in enumerate(u.alt_idx, start=2):
        wm = np.asarray(reg.fullres_to_mni(np.asarray(tmap.surf_vox, float)[i]), float)
        asite, adist = nearest_eeg_site(wm)
        arows.append(f"<tr><td>{rank}</td><td>{fmtv(wm)} mm MNI</td>"
                     f"<td>{asite} ({adist:.0f} mm)</td>"
                     f"<td>{100 * u.obj[i]:.0f}%</td>"
                     f"<td>{u.inc_deg[i]:.0f}°</td>"
                     f"<td>{float(np.asarray(tmap.rad_mm, float)[i]):.0f} mm</td></tr>")
    alts_table = ("<table><tr><td>#</td><td>window (MNI)</td><td>≈ EEG site</td>"
                  "<td>objective vs best</td><td>incidence</td><td>target distance</td>"
                  "</tr>" + "".join(arows) + "</table>"
                  "<p class=\"note\">Ranked fallbacks (orange stars on the maps above), "
                  "each ≥30 mm from the chosen window and from each other — for when "
                  "hair, hardware, or a frame blocks the optimum. Columns follow the "
                  "manuscript Table 1 (window, incidence, distance).</p>"
                  ) if arows else "<p>(no separated alternatives found)</p>"
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

<h2>1 · Placement summary</h2>
<table>{trs}</table>

<h2>2 · Surface coupling and bone transmission</h2>
<img src="data:image/png;base64,{png_3d}" alt="transparency map 3-D"/>
<img src="data:image/png;base64,{png_t}" alt="transparency unwrapped"/>
<p class="note">Transmitted amplitude in dB (1/r&sup2;-corrected) — for <em>seeing</em>
where the bone transmits, in the manuscript Fig.&nbsp;5 presentation. Bright = transparent
bone; grey = no direct-arrival coverage. This corrected map is a visualisation — placement
uses the raw quantity below.</p>

<h2>3 · Beam incidence and the legality limit</h2>
<img src="data:image/png;base64,{png_i}" alt="incidence unwrapped"/>
<p class="note">Angle between the target→patch ray and the bone's outward normal. Beyond
the legality limit the fluid model's longitudinal transmission gives way to unmodelled
shear conversion, so those windows are excluded from placement (manuscript, Discussion:
the water–bone longitudinal critical angle).</p>

<h2>4 · The placement objective and the chosen window</h2>
<img src="data:image/png;base64,{png_o}" alt="objective unwrapped"/>
<img src="data:image/png;base64,{png_pl}" alt="placement"/>
<p class="note">&radic;J<sub>w</sub> — incidence-weighted <em>raw</em> delivered intensity
integrated over the bowl footprint, illegal patches excluded (manuscript Eqs.&nbsp;7
&amp;&nbsp;14; the field of Fig.&nbsp;8). The red star is the chosen window; the bright
lobe's breadth is the placement margin quoted in the summary.</p>

<h2>5 · Alternative windows (if the optimum is blocked)</h2>
{alts_table}

<h2>6 · Methods</h2>
<details><summary><b>How each number is computed</b></summary>
<ul style="font-size:0.9em">
<li><b>Transparency (dB)</b>: 20·log10 of the 1/r&sup2;-corrected transmitted amplitude
(&radic; of the corrected intensity), referenced to the 98th-percentile patch; colour range
= the map's 25th–99th percentiles. Presentation follows manuscript Fig.&nbsp;5.</li>
<li><b>Placement objective</b> &radic;J<sub>w</sub>: J<sub>w</sub>(window) =
&Sigma;<sub>footprint</sub> cos&sup2;&theta;&middot;I<sub>pk</sub> over the bowl-radius
footprint, with patches beyond the incidence limit excluded; I<sub>pk</sub> is the
<em>raw</em> direct-arrival peak intensity (spreading included), the reciprocity-correct
placement quantity — Eqs.&nbsp;(7) and (14) of the manuscript; the field shown is its
Fig.&nbsp;8 objective.</li>
<li><b>Incidence</b> &theta;: angle between the target→patch ray and the outward surface
normal; beyond the legality limit the fluid model's longitudinal transmission gives way to
unmodelled shear conversion, so those patches are excluded (manuscript, Discussion: the
water–bone longitudinal critical angle).</li>
<li><b>Access</b>: solid angle (fraction of 4&pi; about the target) of good-transmission
(&gt;3% of the 98th-percentile corrected amplitude), incidence-legal patches —
occupied-bin sum, cos-latitude weighted — the manuscript Table&nbsp;1 access criterion.
<b>Aperture margin</b> divides it by the ~25% an f/0.6 bowl subtends (manuscript
Table&nbsp;2 and §'Optimal aperture size').</li>
<li><b>Placement tolerance</b>: greatest distance from the chosen window within its own
connected &ge;90%-objective lobe (far-side lobes excluded by connectivity). Few-mm
placement accuracy requirements are set by the aberration coherence length (manuscript
Fig.&nbsp;7) — quote tolerance as the smaller of the two.</li>
<li><b>Alternative windows</b>: highest-objective candidates &ge;30 mm from the chosen
window and from each other.</li>
<li><b>Unwrap</b>: superior-pole, anterior-centred equirectangular projection of the
target→patch directions; amplitude-weighted wrapped histograms; grey = no coherent
direct-arrival coverage.</li>
</ul></details>
<footer>skull-transparency report — generated {stamp}. Coordinates are MNI RAS mm. The map
comes from one full-wave time-reversal solve (virtual source at the target); placement and
every figure here are post-processing on that recorded field. Method, conventions, and the
figure/equation/table numbers cited above: {_MANUSCRIPT}. This file is self-contained —
print to PDF for archival.</footer>
</body></html>
"""
    out.write_text(html)
    return out
