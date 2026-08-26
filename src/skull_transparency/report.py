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

def _frame_is_mni(reg) -> bool:
    """True when the bundle's world frame is MNI, i.e. the human atlas conventions
    (EEG 10-20 sites, "MNI mm" axis labels) actually apply. A subject or non-human
    frame -- e.g. a mouse skull's own aligned-native RAS -- gets neither."""
    return "mni" in str(getattr(reg, "world_frame", "mni_ras_mm") or "").lower()


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
        self.is_mni = _frame_is_mni(reg)
        self.sites, self.slon, self.slat = [], [], []
        for name, pos in (EEG_SITES_MNI.items() if self.is_mni else ()):
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
    if u.sites:
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


def _scene3d_figure(tmap, placement, bowl_radius_mm, aperture_mm=None, frame="head"):
    """The placement as a 3-D scene, manuscript Figs. 1 and 8 style: semi-transparent
    skull, the transducer bowl surface seated at its pose, the beam axis, and the target
    marked inside the head. Two views: down the beam axis, and side-on to it.

    ``frame`` sets the extent. ``"head"`` frames the skull, which is what you want to read
    the window against the anatomy -- but it crops the dish away whenever the device is
    large next to the head, and a bowl focused far beyond the skull is exactly that (the
    mouse TIPS is a 92 mm dish on a 25 mm head, entirely outside a head-framed view).
    ``"device"`` frames the skull AND the whole transducer, so the geometry is legible at
    the scale the hardware actually has. The report draws both.

    ``aperture_mm`` is the DEVICE's aperture diameter, and is what sizes the drawn dish.
    Without it the cap falls back to ``bowl_radius_mm``, which is the beam's FOOTPRINT on
    the skull -- the same radius the placement objective sweeps. Those two coincide only
    when the skull sits about one focal length from the target, which is true of a human
    and badly false of a rodent: the mouse TIPS (80 mm focal length, 92 mm aperture, skull
    6 mm from the target) has a 3.4 mm footprint, so the fallback would draw a 2.4 deg cap
    where the real dish subtends 35.1 deg. Pass the aperture whenever it is known."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    P = np.asarray(tmap.surf_mni_mm() if tmap.registration is not None else tmap.surf_vox,
                   float)
    if len(P) > 60000:                                       # translucency needs few, dim points
        P = P[np.random.default_rng(0).permutation(len(P))[:60000]]
    tgt = np.asarray(placement.target_mni_mm, float)
    apex = np.asarray(placement.apex_mni_mm, float)
    win = np.asarray(placement.window_center_mni_mm, float)

    # bowl cap surface: sphere of radius ROC about the FOCUS (= the target when the apex
    # sits one focal length out), within the aperture half-angle around the beam axis
    n = apex - tgt
    roc = float(np.linalg.norm(n)) or 1.0
    n = n / roc
    r_cap = (float(aperture_mm) / 2.0 if aperture_mm else float(bowl_radius_mm))
    half = float(np.arcsin(np.clip(r_cap / roc, 0.0, 1.0)))
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 0.2:
        e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    k = np.arange(2400)
    costh = 1.0 - (1.0 - np.cos(half)) * (k + 0.5) / len(k)
    sinth = np.sqrt(1.0 - costh ** 2)
    phi = k * 2.399963229728653                              # golden angle
    dirs = (costh[:, None] * n[None, :]
            + sinth[:, None] * (np.cos(phi)[:, None] * e1[None, :]
                                + np.sin(phi)[:, None] * e2[None, :]))
    cap = tgt + roc * dirs

    # target region: a small sphere marker (NOT a predicted focal volume)
    uu, vv = np.meshgrid(np.linspace(0, 2 * np.pi, 24), np.linspace(0, np.pi, 12))
    r_t = 5.0
    sx = tgt[0] + r_t * np.cos(uu) * np.sin(vv)
    sy = tgt[1] + r_t * np.sin(uu) * np.sin(vv)
    sz = tgt[2] + r_t * np.cos(vv)

    az0 = float(np.degrees(np.arctan2(n[1], n[0])))
    el0 = float(np.clip(np.degrees(np.arcsin(np.clip(n[2], -1, 1))), -60, 60))
    views = [("down the beam axis", el0, az0), ("side-on", 15.0, az0 + 90.0)]

    fig = plt.figure(figsize=(12.5, 5.8))
    for kview, (vname, el, az) in enumerate(views):
        ax = fig.add_subplot(1, 2, kview + 1, projection="3d")
        ax.computed_zorder = False
        wide = frame == "device"
        ax.scatter(*P.T, c="#9aa2ad", s=(4.0 if wide else 1.5),
                   alpha=(0.18 if wide else 0.05), linewidths=0, zorder=1)
        ax.plot_surface(sx, sy, sz, color="#2ca02c", alpha=0.45, linewidth=0, zorder=2)
        ext = tgt - 0.25 * roc * n
        ax.plot(*np.c_[apex, ext], color="#2ca02c", ls="--", lw=1.5, zorder=3)
        ax.scatter(*cap.T, c="#00b7c7", s=5, linewidths=0, alpha=0.9, zorder=4,
                   label="transducer")
        ax.scatter(*win, marker="*", s=180, c="red", edgecolors="k", linewidths=0.6,
                   zorder=5, label="window")
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect((1, 1, 1))
        # head frame: the skull alone. device frame: skull + dish + apex, so nothing is
        # cropped even when the transducer dwarfs the head.
        Q = np.vstack([P, cap, apex[None, :]]) if frame == "device" else P
        lo, hi = Q.min(0), Q.max(0)
        mid, sp = (hi + lo) / 2, (hi - lo).max() / 2 * 1.12   # margin so the bowl fits
        for kk, mm in zip("xyz", mid):
            getattr(ax, f"set_{kk}lim")(mm - sp, mm + sp)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_ticklabels([])
            pane.pane.set_facecolor((1, 1, 1, 0))
        ax.grid(False)
        ax.set_title(vname, fontsize=10)
        if kview == 0:
            ax.legend(fontsize=8, loc="upper left")
    if frame == "device":
        span = float((np.vstack([P, cap]).max(0) - np.vstack([P, cap]).min(0)).max())
        fig.suptitle(f"Placement scene at DEVICE scale — the whole {2 * r_cap:.0f} mm dish "
                     f"and its stand-off ({span:.0f} mm across; the head is the small cloud "
                     "at the focus)", fontsize=11)
    else:
        fig.suptitle("Placement scene — transducer on the skull, target in the head "
                     "(semi-transparent; green sphere = target marker, not a focal volume)",
                     fontsize=11)
    return fig


# --------------------------------------------------------------------------- anatomy
#: Anatomical views for the 3-D target scene, as (label, elevation, azimuth). Azimuth is
#: measured from +x in the xy plane, so in a RAS world frame (+x right, +y anterior,
#: +z superior) 180 looks from the left, -90 from behind, and a high elevation from above.
_ANAT_VIEWS = (("left lateral", 8.0, 180.0), ("posterior", 8.0, -90.0), ("dorsal", 72.0, -90.0))

#: (name, plane-normal axis, horizontal axis, vertical axis, x label, y label)
_ORTHO_PLANES = (("sagittal", 0, 1, 2, "y  P-A (mm)", "z  I-S (mm)"),
                 ("coronal", 1, 0, 2, "x  L-R (mm)", "z  I-S (mm)"),
                 ("axial", 2, 0, 1, "x  L-R (mm)", "y  P-A (mm)"))


def _plane_points(center_w, n_axis, a_axis, b_axis, half_mm, n=420):
    """``(n, n, 3)`` world-mm points on the anatomical plane through ``center_w``, plus the
    matplotlib ``extent``. Sampling in WORLD mm (not voxel indices) keeps the panels
    anatomically correct and mm-square whatever pose the simulation grid was built in."""
    a = np.linspace(-half_mm, half_mm, n) + center_w[a_axis]
    b = np.linspace(-half_mm, half_mm, n) + center_w[b_axis]
    A, B = np.meshgrid(a, b, indexing="xy")
    P = np.empty(A.shape + (3,), float)
    P[..., n_axis] = center_w[n_axis]
    P[..., a_axis] = A
    P[..., b_axis] = B
    return P, (a[0], a[-1], b[0], b[-1])


def _sample_volume(vol, reg, P, order=1, fill=np.nan):
    """Sample a bundle volume (indexed in full-resolution voxels) at world-mm points."""
    from scipy.ndimage import map_coordinates
    v = np.asarray(reg.mni_to_fullres(P.reshape(-1, 3)), float).T
    return map_coordinates(np.asarray(vol, float), v, order=order, mode="constant",
                           cval=fill).reshape(P.shape[:2])


def _atlas_mask_on_plane(atlas_img, ids, P):
    """Nearest-neighbour atlas membership on a plane, reading only the sub-block the plane
    touches — a 10 um atlas is far too large to hold in memory."""
    inv = np.linalg.inv(np.asarray(atlas_img.affine, float))
    v = inv[:3, :3] @ P.reshape(-1, 3).T + inv[:3, 3:4]
    vi = np.rint(v).astype(int)
    shape = np.array(atlas_img.shape[:3])
    ok = np.all((vi >= 0) & (vi < shape[:, None]), axis=0)
    out = np.zeros(vi.shape[1], bool)
    if ok.any():
        sub = vi[:, ok]
        lo, hi = sub.min(1), sub.max(1) + 1
        block = np.asarray(atlas_img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        out[ok] = np.isin(block[tuple(sub - lo[:, None])], np.asarray(ids))
    return out.reshape(P.shape[:2])


def _ortho_figure(tmap, placement, c_map, atlas_img=None, atlas_ids=None,
                  atlas_label="target structure", half_mm=None, c0=None):
    """Sagittal / coronal / axial cuts through the target: the subject's sound-speed volume
    as the greyscale background, the atlas structure (when supplied) overlaid, and the
    target, chosen window, and beam axis marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reg = tmap.registration
    tgt = np.asarray(placement.target_mni_mm, float)
    win = np.asarray(placement.window_center_mni_mm, float)
    # Window from the medium's REFERENCE speed, not the array minimum: a map can carry two
    # water values (e.g. a benchmark volume at 1500 m/s padded with the pipeline's 1540 m/s
    # coupling water), and stretching from the minimum would draw that seam as a box around
    # the head. Everything at or below water renders black; the ramp spans water -> bone.
    c0 = float(np.nanmin(c_map)) if c0 is None else float(c0)
    # Centre the panels on the HEAD but cut each plane through the TARGET: centring on an
    # eccentric target (a mouse cerebellum, say) wastes half of every panel on empty water.
    surf = np.asarray(tmap.surf_mni_mm() if tmap.registration is not None else tmap.surf_vox,
                      float)
    lo, hi = surf.min(0), surf.max(0)
    cen = (lo + hi) / 2.0
    if half_mm is None:
        half_mm = max(0.55 * float((hi - lo).max()),
                      1.08 * float(np.abs(tgt - cen).max()),
                      1.08 * float(np.abs(win - cen).max()))
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9))
    for ax, (name, n_ax, a_ax, b_ax, xl, yl) in zip(axes, _ORTHO_PLANES):
        centre = cen.copy()
        centre[n_ax] = tgt[n_ax]              # the plane itself passes through the target
        P, ext = _plane_points(centre, n_ax, a_ax, b_ax, half_mm)
        sl = _sample_volume(c_map, reg, P)
        # outside the simulation domain is the same coupling water the model assumes
        ax.imshow(np.where(np.isfinite(sl), sl, c0), origin="lower", extent=ext, cmap="bone",
                  vmin=c0, vmax=float(np.nanmax(c_map)), aspect="equal",
                  interpolation="bilinear")
        if atlas_img is not None and atlas_ids is not None:
            m = _atlas_mask_on_plane(atlas_img, atlas_ids, P)
            if m.any():
                rgba = np.zeros(m.shape + (4,))
                rgba[..., 1], rgba[..., 0], rgba[..., 2] = 0.78, 0.17, 0.35
                rgba[..., 3] = np.where(m, 0.42, 0.0)
                ax.imshow(rgba, origin="lower", extent=ext, aspect="equal",
                          interpolation="nearest")
                ax.contour(np.linspace(ext[0], ext[1], m.shape[1]),
                           np.linspace(ext[2], ext[3], m.shape[0]), m.astype(float),
                           levels=[0.5], colors="#2ca02c", linewidths=1.1)
        ax.plot([win[a_ax], tgt[a_ax]], [win[b_ax], tgt[b_ax]], color="#ff9f1c", ls="--", lw=1.4)
        ax.scatter([tgt[a_ax]], [tgt[b_ax]], marker="+", s=130, c="cyan", linewidths=1.8, zorder=6)
        ax.scatter([win[a_ax]], [win[b_ax]], marker="*", s=170, c="red", edgecolors="k",
                   linewidths=0.5, zorder=6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
    lbl = f", {atlas_label} in green" if (atlas_img is not None and atlas_ids is not None) else ""
    fig.suptitle(f"Anatomy through the target — sound speed{lbl} "
                 "(cyan + target, red ★ window, dashed beam axis)", fontsize=11)
    fig.tight_layout()
    return fig


def _target3d_figure(tmap, placement, c_map, bone_threshold, atlas_img=None, atlas_ids=None,
                     atlas_label="target structure", decimate=2):
    """The target inside a semi-transparent skull, in three anatomical views — the
    manuscript's Figure-1 scene. The skull is an isosurface of the sound-speed volume
    rendered at low opacity; the target is the atlas structure when one is supplied, else
    a small sphere at the target point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.ndimage import binary_closing, gaussian_filter
    try:                                       # scikit-image is not a dependency of this
        from skimage.measure import marching_cubes    # package: without it the scene falls
    except ImportError:                                # back to point clouds, which read the
        marching_cubes = None                          # same way but with rougher edges

    reg = tmap.registration
    dx = reg.dx_mm
    d = max(1, int(decimate))
    bone_b = np.asarray(c_map)[::d, ::d, ::d] > bone_threshold
    if not bone_b.any():
        return None                                  # nothing to draw
    if marching_cubes is not None:
        bone = gaussian_filter(bone_b.astype(np.float32), 0.7)
        if not (bone.max() > 0.5 > bone.min()):
            return None
        sv, sf, _, _ = marching_cubes(bone, level=0.5)
        Vs = sv * d * dx                             # decimated voxel -> grid mm
    else:
        sf = None
        Vs = np.argwhere(bone_b).astype(float) * d * dx

    tgt_g = np.asarray(reg.mni_to_fullres(np.asarray(placement.target_mni_mm, float)), float) * dx
    win_g = np.asarray(reg.mni_to_fullres(np.asarray(placement.window_center_mni_mm, float)), float) * dx
    Vt = ft = None
    if atlas_img is not None and atlas_ids is not None:
        A = np.asarray(atlas_img.affine, float)
        step = max(1, int(round(0.2 / abs(A[0, 0]))))      # sample the atlas at ~0.2 mm
        sub = np.asarray(atlas_img.dataobj[::step, ::step, ::step])
        mask = binary_closing(np.isin(sub, np.asarray(atlas_ids)), np.ones((2, 2, 2)))
        if mask.sum() > 8:
            if marching_cubes is not None:
                tv, ft, _, _ = marching_cubes(mask.astype(np.float32), level=0.5)
            else:
                tv, ft = np.argwhere(mask).astype(float), None
            world = (A[:3, :3] @ (tv * step).T + A[:3, 3:4]).T
            Vt = np.asarray(reg.mni_to_fullres(world), float) * dx

    fig = plt.figure(figsize=(13.5, 5.0))
    for k, (name, el, az) in enumerate(_ANAT_VIEWS):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.computed_zorder = False
        if sf is not None:
            ax.plot_trisurf(Vs[:, 0], Vs[:, 1], Vs[:, 2], triangles=sf, color="#c8ccd2",
                            alpha=0.16, linewidth=0, shade=True, zorder=1)
        else:
            ax.scatter(*Vs[::7].T, c="#c8ccd2", s=1.4, alpha=0.10, linewidths=0, zorder=1)
        if Vt is not None and ft is not None:
            ax.plot_trisurf(Vt[:, 0], Vt[:, 1], Vt[:, 2], triangles=ft, color="#2ca02c",
                            alpha=0.95, linewidth=0, shade=True, zorder=3)
        elif Vt is not None:
            ax.scatter(*Vt[::3].T, c="#2ca02c", s=4.0, alpha=0.85, linewidths=0, zorder=3)
        else:
            u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
            r = 0.05 * float(np.ptp(Vs, axis=0).max())
            ax.plot_surface(tgt_g[0] + r * np.cos(u) * np.sin(v),
                            tgt_g[1] + r * np.sin(u) * np.sin(v),
                            tgt_g[2] + r * np.cos(v), color="#2ca02c", alpha=0.85,
                            linewidth=0, zorder=3)
        ax.plot(*np.c_[win_g, tgt_g], color="#ff9f1c", ls="--", lw=1.6, zorder=4)
        ax.scatter(*win_g, c="red", marker="*", s=170, edgecolors="k", linewidths=0.5,
                   zorder=5, depthshade=False)
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect((1, 1, 1))
        lo, hi = Vs.min(0), Vs.max(0)
        mid, sp = (hi + lo) / 2, (hi - lo).max() / 2
        for kk, mm in zip("xyz", mid):
            getattr(ax, f"set_{kk}lim")(mm - sp, mm + sp)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_ticklabels([])
            pane.pane.set_facecolor((1, 1, 1, 0))
        ax.grid(False)
        ax.set_title(name, fontsize=10)
    what = atlas_label if Vt is not None else "target"
    fig.suptitle(f"The {what} (green) inside the semi-transparent skull "
                 "(red ★ window, dashed beam axis)", fontsize=11)
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


#: HTML->PDF backends, tried in order. WeasyPrint if it is importable, else a headless
#: Chrome/Chromium (which honours the report's @media print rules, so the PDF is the page
#: you see), else wkhtmltopdf.
_CHROME_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium",
                      "chromium-browser", "/opt/google/chrome/chrome")


def html_to_pdf(html_path, pdf_path=None, *, timeout_s: float = 180.0) -> Path:
    """Render a report HTML file to PDF, preserving its print layout.

    The report is self-contained (figures are embedded as data URIs), so the conversion
    needs no network and no asset paths. Returns the PDF path; raises RuntimeError with an
    actionable message when no backend is installed."""
    import os
    import shutil
    import subprocess
    import tempfile

    html = Path(html_path)
    pdf = Path(pdf_path) if pdf_path else html.with_suffix(".pdf")
    try:                                        # 1) WeasyPrint (pure-Python, no browser)
        from weasyprint import HTML as _WeasyHTML
        _WeasyHTML(filename=str(html)).write_pdf(str(pdf))
        if pdf.exists() and pdf.stat().st_size:
            return pdf
    except ImportError:
        pass
    for cand in _CHROME_CANDIDATES:             # 2) headless Chrome / Chromium
        exe = shutil.which(cand) or (cand if os.path.exists(cand) else None)
        if not exe:
            continue
        with tempfile.TemporaryDirectory(prefix="skull_transparency_pdf_") as profile:
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-sandbox",
                            f"--user-data-dir={profile}", "--no-pdf-header-footer",
                            "--virtual-time-budget=20000", f"--print-to-pdf={pdf}",
                            html.resolve().as_uri()],
                           capture_output=True, text=True, timeout=timeout_s)
        if pdf.exists() and pdf.stat().st_size:
            return pdf
    exe = shutil.which("wkhtmltopdf")           # 3) wkhtmltopdf
    if exe:
        subprocess.run([exe, "--enable-local-file-access", str(html), str(pdf)],
                       capture_output=True, text=True, timeout=timeout_s)
        if pdf.exists() and pdf.stat().st_size:
            return pdf
    raise RuntimeError(
        "no HTML-to-PDF backend found. Install one of: weasyprint (pip install weasyprint), "
        "Google Chrome / Chromium, or wkhtmltopdf. The HTML report at "
        f"{html} is complete and prints to PDF from any browser.")


def _movie_frames(movie, out_dir, *, max_frames: int = 60, width: int = 1200):
    """Extract evenly spaced PNG frames from a movie (mp4/gif) or a directory of frames.

    Returns ``(n_frames, fps)``; files are written as ``frame-0.png`` ... in ``out_dir``,
    the zero-based, unpadded naming ``\animategraphics`` expects. Frames are subsampled in
    TIME because the animation is embedded in the PDF whole and 105 full-size frames would
    add tens of megabytes for no visible gain.

    Spatial downscaling is area-averaged and only happens when the source is actually wider
    than ``width``. It used to be ``im[::step, ::step]`` with ``step = ceil(w / width)``,
    which is both aliased and systematically too small: any source between 1x and 2x the
    target was HALVED, so the common 1200 px propagation movie landed at 600 px against a
    900 px target and read as visibly soft on a full PDF page."""
    import imageio.v3 as iio
    src = Path(movie)
    fps = 12.0
    if src.is_dir():
        files = sorted(src.glob("*.png"))
        imgs = [iio.imread(f) for f in files]
    else:
        try:
            meta = iio.immeta(str(src))
            fps = float(meta.get("fps", fps))
        except Exception:
            pass
        imgs = list(iio.imiter(str(src)))
    if not imgs:
        return 0, fps
    keep = np.linspace(0, len(imgs) - 1, min(max_frames, len(imgs))).round().astype(int)
    fps = max(1.0, fps * len(keep) / len(imgs))          # keep the wall-clock duration
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, i in enumerate(keep):
        im = np.asarray(imgs[i])
        if im.ndim == 3 and im.shape[2] == 4:
            im = im[..., :3]
        if im.shape[1] > width:                          # area-averaged, exact target width
            h = int(round(im.shape[0] * width / im.shape[1]))
            try:
                from PIL import Image
                im = np.asarray(Image.fromarray(im).resize((width, h), Image.LANCZOS))
            except Exception:                            # no pillow: fall back to decimation
                step = int(np.ceil(im.shape[1] / width))
                im = im[::step, ::step]
        iio.imwrite(out_dir / f"frame-{j}.png", im)
    return len(keep), fps


def append_animated_page(pdf_path, movie, *, fps=None, caption="", autoplay: bool = True,
                         timeout_s: float = 300.0):
    """Append a page carrying a real ``\animategraphics`` animation to a report PDF.

    A browser-printed PDF can only ever show one still of a movie. LaTeX's ``animate``
    package embeds the frames and plays them in Acrobat and compatible viewers -- the same
    mechanism the manuscript's propagation figure uses -- so the movie is built as a
    one-page PDF here and concatenated onto the report. ``autoplay`` (the default) starts it
    as soon as the page is shown and loops; set it False for a click-to-play control bar.

    Either way this needs a viewer that runs the embedded JavaScript -- Acrobat and
    PDF-XChange do, okular and most browser viewers do not and show a single still. Returns the PDF path; when pdflatex,
    the animate package, or pypdf is missing the report is returned unchanged and a warning
    says why (the HTML still animates)."""
    import shutil
    import subprocess
    import tempfile
    import warnings

    pdf_path = Path(pdf_path)
    if not shutil.which("pdflatex"):
        warnings.warn("no pdflatex: the PDF gets no animation page (the HTML still animates)")
        return pdf_path
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter          # older name
        except ImportError:
            warnings.warn("no pypdf: cannot append the animation page to the PDF")
            return pdf_path
    with tempfile.TemporaryDirectory(prefix="skull_transparency_anim_") as td:
        td = Path(td)
        n, detected = _movie_frames(movie, td)
        if n < 2:
            warnings.warn(f"{movie}: fewer than two frames, no animation page appended")
            return pdf_path
        rate = float(fps or detected)
        # A caption is prose from the caller, so every LaTeX special has to be escaped --
        # a bare underscore in something like "make_propagation_movie.py" is enough to
        # abort the whole page.
        # autoplay+loop rather than a control bar: the page starts playing on its own when
        # it becomes visible, which is what a report reader wants. `controls` adds a button
        # strip (and ~16 extra annotations) that has to be clicked first.
        opts = "autoplay,loop" if autoplay else "controls,loop"
        cap = str(caption or "")
        for a, b in (("\\", " "), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                     ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}")):
            cap = cap.replace(a, b)
        (td / "anim.tex").write_text(
            "\\documentclass[a4paper]{article}\n"
            "\\usepackage[margin=1.6cm]{geometry}\\usepackage{graphicx}\\usepackage{animate}\n"
            "\\pagestyle{empty}\\setlength{\\parindent}{0pt}\n"
            "\\begin{document}\n"
            "{\\large\\bfseries Wave propagation}\\\\[4pt]\n"
            f"\\animategraphics[{opts},width=\\linewidth]{{{rate:g}}}{{frame-}}{{0}}{{{n - 1}}}\n"
            f"\\\\[6pt]\\small {cap}\n"
            "\\end{document}\n")
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "anim.tex"],
                               cwd=td, capture_output=True, text=True, timeout=timeout_s)
        anim = td / "anim.pdf"
        if r.returncode != 0 or not anim.exists():
            tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
            warnings.warn(f"pdflatex could not build the animation page; the PDF is "
                          f"otherwise complete. Last output:\n{tail}")
            return pdf_path
        # Merge with append(), NOT page-by-page add_page(). The animation is a set of
        # AcroForm pushbutton fields (one per frame) driven by a /Screen annotation's
        # page-open action; add_page() copies the annotations but leaves the document
        # catalog's /AcroForm behind, orphaning every field. The page then looks right and
        # does nothing -- which is exactly the "it won't play" symptom, in EVERY viewer,
        # rather than the viewer limitation it resembles. Measured on a 6-frame probe:
        # standalone pdflatex 7 fields, add_page 0, append 7.
        try:
            w = PdfWriter(clone_from=str(pdf_path))
            w.append(str(anim))
        except Exception:                       # older pypdf/PyPDF2 without clone/append
            w = PdfWriter()
            for page in PdfReader(str(pdf_path)).pages:
                w.add_page(page)
            for page in PdfReader(str(anim)).pages:
                w.add_page(page)
            warnings.warn("this pypdf cannot merge AcroForms; the animation page will not "
                          "play (upgrade pypdf, or open the animation PDF on its own)")
        with open(pdf_path, "wb") as fh:
            w.write(fh)
    return pdf_path


def forward_peak_volumes(forward_dir, out_npz=None, *, free_field: str = "auto",
                         rs_kwargs: dict | None = None):
    """Reduce a solved forward pair to the small artifact the report needs.

    Each solved run holds gigabytes of focal-box time traces; all the comparison figure
    wants is the peak |p| at every box point. This writes (or returns) those two volumes
    plus the box geometry, so the traces can be deleted afterwards and the figure still
    regenerates in a second. Returns a dict with ``transcranial``/``free_field`` peak
    volumes (Pa, dense box grid), ``origin_vox``, ``dx_mm``, and
    ``free_field_method`` -- ``'fullwave'`` or ``'rs'``.

    ``free_field`` selects where the all-water volume comes from:

    * ``'auto'`` (default) -- the solved ``free_field/`` deck when it is there, otherwise
      integrate it analytically from the transcranial deck's own bowl and box
      (:func:`skull_transparency.forward.free_field_rs`). A pair run with
      ``--free-field rs`` never writes that second deck, so without the fallback its
      figure could not be drawn at all;
    * ``'fullwave'`` -- require the solved deck, and omit the row if it is absent;
    * ``'rs'`` -- always integrate it, even when the solved deck exists (useful for
      putting the two side by side).

    Always read ``free_field_method`` back before quoting a ratio off these volumes: the
    analytic twin is the more accurate absolute field but does not cancel the solver's
    discretisation the way a full-wave pair does (see :func:`~.forward.free_field_rs`).
    """
    from .forward import load_focal_box
    if free_field not in ("auto", "fullwave", "rs"):
        raise ValueError(f"free_field must be 'auto', 'fullwave' or 'rs' (got {free_field!r})")
    forward_dir = Path(forward_dir)

    def _volume(traces, box, info, out):
        peak = np.abs(np.asarray(traces, float)).max(axis=0)      # peak |p| per point (Pa)
        box = np.asarray(box, np.int64)
        lo = box.min(0)
        shape = tuple((box.max(0) - lo + 1).tolist())
        vol = np.zeros(shape, float)
        vol[tuple((box - lo).T)] = peak
        out["origin_vox"] = lo.astype(float)
        out["dx_mm"] = float(info.get("dx_mm", 1.0))
        return vol

    out = {}
    for label in ("transcranial", "free_field"):
        run = forward_dir / label
        if not (run / "genout.dat").exists():
            continue
        if label == "free_field" and free_field == "rs":
            continue                                   # asked for the analytic one instead
        out[label] = _volume(*load_focal_box(run), out)
        if label == "free_field":
            out["free_field_method"] = "fullwave"

    if "free_field" not in out and free_field in ("auto", "rs") and "transcranial" in out:
        from .forward import free_field_rs
        traces, box, info = free_field_rs(forward_dir / "transcranial", **(rs_kwargs or {}))
        out["free_field"] = _volume(traces, box, info, out)
        out["free_field_method"] = "rs"

    if out_npz and out:
        np.savez_compressed(out_npz, **out)
    return out


def _forward_figure(peaks, placement, reg, comparison=None, c_map=None, c0=None,
                    bone_threshold=None):
    """Transcranial vs free-field focal fields on one colour scale.

    Rows are the two media, columns the three planes through the target. Same source, same
    grid, same drive -- only the medium differs -- so the panels are directly comparable:
    what the skull costs on target, and where it puts the energy instead.

    With ``c_map`` the skull is drawn underneath in grey and the field is composited over it
    with an amplitude-proportional alpha, so the focus can be read against the bone that
    made it. The recording box is expressed relative to the target, so placing it in the
    bundle's anatomy needs the two grids to share pitch and axis orientation -- true when
    both came from ``prepare`` on the same medium. That is checked, and the background is
    simply omitted rather than drawn wrong when it does not hold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "transcranial" not in peaks or "free_field" not in peaks:
        return None
    dx = float(peaks["dx_mm"])
    lo = np.asarray(peaks["origin_vox"], float)
    # The target must be taken in the FORWARD run's own grid: that pair may have been solved
    # in a different domain from the bundle (a bigger water margin, say, to fit the bowl),
    # so the bundle's voxel indexing does not apply to the box.
    tgt_v = None
    if comparison:
        tv = comparison.get("transcranial", {}).get("target_vox")
        if tv is not None:
            tgt_v = np.asarray(tv, float)
    if tgt_v is None:
        tgt_v = np.asarray(reg.mni_to_fullres(np.asarray(placement.target_mni_mm, float)), float)
    ctr = tgt_v - lo                                    # target in box-index coordinates
    if np.any(ctr < -1) or np.any(ctr > np.asarray(peaks["transcranial"]).shape) + 1:
        pass                                            # (checked below, per-plane clipping)
    vmax = max(float(np.asarray(peaks[k]).max()) for k in ("transcranial", "free_field"))
    # say which twin drew the bottom row: an analytic one is a different kind of number
    # from a second solve, and the ratio a reader takes off the panels depends on it
    ff_method = str(np.asarray(peaks.get("free_field_method", "fullwave")).item()
                    if "free_field_method" in peaks else "fullwave")
    ff_tag = "  (Rayleigh-Sommerfeld)" if ff_method == "rs" else ""

    # skull background: only when the forward grid and the bundle agree on pitch and the
    # bundle's world axes are the grid axes (both hold for runs prepared from one medium)
    show_bone = False
    if c_map is not None and reg is not None:
        R = np.asarray(getattr(reg, "R_mni_to_sim", np.eye(3)), float)
        show_bone = (abs(dx - float(reg.dx_mm)) < 1e-6
                     and np.allclose(R, np.eye(3), atol=1e-6))
    tgt_w = np.asarray(placement.target_mni_mm, float)
    c0 = float(np.nanmin(c_map)) if (c0 is None and c_map is not None) else c0
    planes = (("sagittal", 0, 1, 2, "y (mm)", "z (mm)"),
              ("coronal", 1, 0, 2, "x (mm)", "z (mm)"),
              ("axial", 2, 0, 1, "x (mm)", "y (mm)"))
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4))
    for r, label in enumerate(("transcranial", "free_field")):
        vol = np.asarray(peaks[label], float)
        for c, (name, n_ax, a_ax, b_ax, xl, yl) in enumerate(planes):
            ax = axes[r, c]
            idx = int(round(np.clip(ctr[n_ax], 0, vol.shape[n_ax] - 1)))
            sl = np.take(vol, idx, axis=n_ax)
            if a_ax > b_ax:
                sl = sl.T
            ext = [(-ctr[a_ax]) * dx, (vol.shape[a_ax] - ctr[a_ax]) * dx,
                   (-ctr[b_ax]) * dx, (vol.shape[b_ax] - ctr[b_ax]) * dx]
            if show_bone:
                ax.set_facecolor("black")     # same base as the transcranial row's water,
                #                               so the two rows are directly comparable
                # box offsets are relative to the target, so world = target + offset
                ga = np.linspace(ext[0], ext[1], sl.shape[0])
                gb = np.linspace(ext[2], ext[3], sl.shape[1])
                A2, B2 = np.meshgrid(ga, gb, indexing="ij")
                P = np.empty(A2.shape + (3,), float)
                P[..., n_ax] = tgt_w[n_ax]
                P[..., a_ax] = tgt_w[a_ax] + A2
                P[..., b_ax] = tgt_w[b_ax] + B2
                bg = np.where(np.isfinite(_sample_volume(c_map, reg, P)),
                              _sample_volume(c_map, reg, P), c0)
                if label == "transcranial":          # the bone is actually there
                    ax.imshow(bg.T, origin="lower", extent=ext, cmap="bone", vmin=c0,
                              vmax=float(np.nanmax(c_map)), aspect="equal",
                              interpolation="bilinear")
                    if bone_threshold:
                        ax.contour(ga, gb, bg.T, levels=[float(bone_threshold)],
                                   colors="#7fd4ff", linewidths=0.8, alpha=0.9)
                elif bone_threshold:                 # free field: OUTLINE ONLY -- this run
                    ax.contour(ga, gb, bg.T, levels=[float(bone_threshold)],  # has no skull
                               colors="#7fd4ff", linewidths=0.8, alpha=0.55,
                               linestyles="--")
            rgba = plt.get_cmap("inferno")(np.clip(sl / (vmax or 1.0), 0, 1))
            rgba[..., 3] = (np.clip(sl / (vmax or 1.0), 0, 1) ** 0.55) if show_bone else 1.0
            im = ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", extent=ext,
                           aspect="equal", interpolation="bilinear")
            im = ax.imshow(np.full((1, 1), np.nan), extent=ext, cmap="inferno",
                           vmin=0.0, vmax=vmax)          # carries the colourbar mapping
            ax.scatter([0], [0], marker="+", s=120, c="cyan", linewidths=1.6)
            pk = np.unravel_index(int(np.argmax(vol)), vol.shape)
            ax.scatter([(pk[a_ax] - ctr[a_ax]) * dx], [(pk[b_ax] - ctr[b_ax]) * dx],
                       marker="x", s=70, c="white", linewidths=1.4)
            if r == 0:
                ax.set_title(name, fontsize=10)
            ax.set_xlabel(xl, fontsize=8)
            row_lbl = ("through the skull" if r == 0
                       else f"free field{ff_tag}\n(no skull; dashed = where bone would be)")
            ax.set_ylabel(f"{row_lbl}\n{yl}" if c == 0 else yl, fontsize=8)
            ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, fraction=0.016, pad=0.01).set_label("peak |p| (Pa)", fontsize=9)
    sub = ""
    if comparison:
        sub = (f"  —  at the target {comparison['transcranial']['peak_at_target_pa']:.2f} Pa "
               f"vs {comparison['free_field']['peak_at_target_pa']:.2f} Pa "
               f"({comparison.get('transmission_at_target', float('nan')):.2f}x, "
               f"{comparison.get('transmission_at_target_db', float('nan')):+.1f} dB)")
    fig.suptitle("Transcranial vs free field: peak pressure around the target"
                 " (cyan + target, white x peak)" + sub, fontsize=11)
    return fig


#: Source classes for the parameter table, in the ultraneuron/neuromod convention: every
#: number a result depends on is labelled by where it came from, so a reader can see at a
#: glance which values are measured, which are taken from the literature, and which are
#: assumptions that a sensitivity check should bracket.
_SOURCE_CLASSES = ("Measured", "Literature", "Derived", "Convention", "Assumption")


def _provenance_rows(tmap, placement, bundle=None, *, bowl_radius_mm=None, aperture_mm=None,
                     theta_max_deg=None, extra=None):
    """Rows of ``(parameter, value, source, note)`` for the parameter table.

    Everything is read from the bundle the map came from, so the table describes the run
    that produced this report rather than the library's defaults. ``extra`` appends
    caller-supplied rows (a case can document its own choices)."""
    phys = dict(getattr(bundle, "physics", {}) or {})
    grid = dict(getattr(bundle, "grid", {}) or {})
    arr = dict(getattr(bundle, "array", {}) or {})
    geom = dict(arr.get("geometry", {}) or {})
    reg = tmap.registration
    dx_mm = 1e3 * float(grid.get("dx_m", (reg.dx_mm / 1e3 if reg else 0.0)))
    f0 = float(phys.get("f0", 0.0))
    c0 = float(phys.get("c0", 0.0))
    lam = (1e3 * c0 / f0) if f0 else float("nan")
    rows = [
        ("Target", f"{np.round(np.asarray(placement.target_mni_mm, float), 2).tolist()} mm",
         "Derived", "The point the virtual source is placed at; from an atlas structure or "
                    "given explicitly."),
        ("World frame", str(getattr(reg, "world_frame", "mni_ras_mm")), "Convention",
         "The frame every coordinate in this report is expressed in."),
        ("Drive frequency f0", f"{f0/1e3:.0f} kHz", "Measured",
         "The device's drive frequency; sets the wavelength and the grid."),
        ("Reference speed c0", f"{c0:.0f} m/s", "Literature",
         "Water / soft tissue; the coupling medium and the grid reference."),
        ("Bone speed (max)", f"{float(phys.get('c_bone', float('nan'))):.0f} m/s", "Literature",
         "Cortical-bone endpoint of the intensity-to-speed ramp."),
        ("Bone threshold", f"{float(phys.get('bone_threshold', float('nan'))):.0f} m/s",
         "Assumption", "Speed above which a voxel counts as bone, i.e. where the calvarial "
                       "surface is drawn. Between water and cortical bone; worth a "
                       "sensitivity check on a thin or a demineralised skull."),
        ("Points per wavelength", f"{float(phys.get('ppw', float('nan'))):.1f}", "Convention",
         f"Grid pitch dx = c0/(f0*ppw) = {dx_mm:.3f} mm against a {lam:.2f} mm wavelength. "
         "For a thin skull the bone thickness, not the wavelength, is the binding "
         "constraint."),
        ("Grid pitch dx", f"{dx_mm:.3f} mm", "Derived", "From f0, c0 and ppw."),
    ]
    if geom:
        rows += [
            ("Transducer geometry",
             f"{geom.get('geometry', '?')}, ROC {geom.get('roc_mm', float('nan')):.1f} mm, "
             f"aperture {geom.get('aperture_mm', float('nan')):.1f} mm"
             + (f", {geom['n_rings']} rings" if geom.get("n_rings") else ""),
             "Measured", "The device as built."),
            ("Cone half-angle", f"{float(geom.get('half_angle_deg', float('nan'))):.1f} deg",
             "Derived", "arcsin((aperture/2)/ROC); what the beam actually subtends."),
        ]
    if aperture_mm is not None:
        rows.append(("Transducer aperture", f"{float(aperture_mm):.0f} mm", "Device",
                     "Aperture DIAMETER of the dish; sizes the bowl drawn in the 3-D scene. "
                     "Distinct from the window footprint below, which is where the beam "
                     "cone actually crosses the skull."))
    if bowl_radius_mm is not None:
        rows.append(("Window footprint radius", f"{float(bowl_radius_mm):.1f} mm", "Derived",
                     "Radius on the skull the window score is aggregated over. Equals the "
                     "aperture radius only when the skull sits about one focal length from "
                     "the target; otherwise it is r_skull*sin(half-angle)."))
    if theta_max_deg is not None:
        rows.append(("Acceptance angle", f"{float(theta_max_deg):.0f} deg", "Literature",
                     "Incidence cap for a legal window, near the water-to-bone longitudinal "
                     "critical angle; past it the fluid model's longitudinal transmission "
                     "gives way to unmodelled shear conversion."))
    rows += [
        ("Surface patches", f"{len(np.asarray(tmap.surf_vox)):,}", "Derived",
         "Calvarial surface voxels the outward solve recorded."),
        ("Recorder", str(dict(getattr(bundle, "meta", {}) or {}).get("recorder",
                              getattr(bundle, "recorder", "shell"))), "Convention",
         "'shell' records only the calvarial surface, which is all the transparency map "
         "needs; 'volume' also records the decimated interior field."),
    ]
    return rows + list(extra or [])


def _provenance_html(rows):
    body = "\n".join(
        f"<tr><td>{p}</td><td>{v}</td><td><b>{src}</b></td><td>{note}</td></tr>"
        for p, v, src, note in rows)
    return ("<table><tr><th>Parameter</th><th>Value</th><th>Source</th><th>Notes</th></tr>"
            + body + "</table>")


def write_report(tmap, placement, out_html, *, title="Skull transparency placement",
                 target_name=None, bowl_radius_mm=32.0, theta_max_deg=35.0,
                 aperture_mm=None,
                 pdf=None, bundle=None, atlas=None, atlas_ids=None,
                 atlas_label="target structure", movie=None, movie_caption=None,
                 forward=None, parameters=None, frame_is_mni=None) -> Path:
    """Write the placement report; returns the path written.

    ``bowl_radius_mm`` / ``theta_max_deg`` parameterise the objective, alternates, and
    access fraction (pass the values the placement used). ``bowl_radius_mm`` is the beam's
    FOOTPRINT on the skull, not the device aperture -- pass ``aperture_mm`` as well so the
    3-D scene draws the real dish rather than the footprint (see :func:`_scene3d_figure`);
    they differ by more than an order of magnitude on a rodent. ``pdf=True`` also renders a PDF
    beside the HTML and returns *that* path; ``pdf=None`` (the default) infers it from an
    ``.pdf`` suffix on ``out_html``, so ``--out report.pdf`` does what it looks like.

    Pass ``bundle`` (the FieldBundle the map came from) to add the anatomy section:
    sagittal/coronal/axial cuts through the target and a 3-D view of it inside the
    semi-transparent skull, both drawn from the bundle's own sound-speed volume. Add
    ``atlas`` (a label-volume path or a loaded NIfTI image) with ``atlas_ids`` (the label
    values that make up the target structure) to draw the structure itself -- e.g. the
    Allen cerebellum warped into a mouse skull's frame. Without a bundle the section is
    skipped; with a bundle but no atlas the target is drawn as a sphere.

    ``movie`` (an mp4/gif path, or a directory of frame PNGs) adds the wave-propagation
    animation: it plays inline in the HTML, and in the PDF it is appended as a page built
    with LaTeX's ``animate`` package, so it plays in Acrobat and compatible viewers exactly
    as the manuscript's propagation figure does.

    ``forward`` adds the transcranial-vs-free-field section: pass the directory of a solved
    forward pair (or a ``focal_peaks.npz`` written by :func:`forward_peak_volumes`, with
    ``forward.json`` beside it). The section reports what the skull costs at the target and
    shows where it puts the energy instead.

    ``parameters`` appends rows to the parameter table -- ``(name, value, source, note)``
    with ``source`` one of Measured / Literature / Derived / Convention / Assumption, the
    classification the neuromodulation reports use so a reader can see which numbers are
    measured and which are assumptions that deserve a sensitivity check.

    ``frame_is_mni`` overrides the frame test that decides whether the human atlas
    conventions apply (EEG 10-20 sites, "MNI mm" labels). It is inferred from the bundle's
    recorded world frame, but a producer that stamped only a generic ``ras_mm`` will read as
    non-MNI even when the skull is MNI-aligned -- pass ``True`` for such a bundle."""
    want_pdf = (str(out_html).lower().endswith(".pdf")) if pdf is None else bool(pdf)
    out_html = Path(out_html).with_suffix(".html")
    from .position_tool import preview_placement
    import matplotlib
    matplotlib.use("Agg")

    out = Path(out_html)
    tgt = np.asarray(placement.target_mni_mm, float)
    win = np.asarray(placement.window_center_mni_mm, float)

    u = _UnwrapData(tmap, placement, bowl_radius_mm, theta_max_deg)
    if frame_is_mni is not None:
        u.is_mni = bool(frame_is_mni)
        u.sites, u.slon, u.slat = ([], [], []) if not u.is_mni else (u.sites, u.slon, u.slat)
    wf = str(getattr(tmap.registration, "world_frame", "mni_ras_mm") or "mni_ras_mm")
    frame_lbl = "MNI" if u.is_mni else f"world ({wf})"
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
    png_ortho = png_t3d = None
    if bundle is not None:
        c_map = None
        try:
            c_map = bundle.skull_c()
        except Exception:                        # a bundle without its c-map: skip anatomy
            c_map = None
        if c_map is not None:
            atlas_img = atlas
            if isinstance(atlas, (str, Path)):
                import nibabel as nib
                atlas_img = nib.load(str(atlas))
            thr = float(dict(getattr(bundle, "physics", {}) or {}).get("bone_threshold", 2200.0))
            phys = dict(getattr(bundle, "physics", {}) or {})
            png_ortho = _fig_png(_ortho_figure(tmap, placement, c_map, atlas_img, atlas_ids,
                                               atlas_label, c0=phys.get("c0")))
            fig_t3d = _target3d_figure(tmap, placement, c_map, thr, atlas_img, atlas_ids,
                                       atlas_label)
            if fig_t3d is not None:
                png_t3d = _fig_png(fig_t3d)
    _fwd_cmap = _fwd_c0 = _fwd_thr = None
    if bundle is not None:
        try:
            _fwd_cmap = bundle.skull_c()
            _p = dict(getattr(bundle, "physics", {}) or {})
            _fwd_c0, _fwd_thr = _p.get("c0"), _p.get("bone_threshold")
        except Exception:
            _fwd_cmap = None
    png_fwd, fwd_json = None, None
    if forward is not None:
        fp = Path(forward)
        peaks, jpath = {}, None
        if fp.is_dir():
            jpath = fp / "forward.json"
            npz = fp / "focal_peaks.npz"
            peaks = dict(np.load(npz)) if npz.exists() else forward_peak_volumes(fp)
        elif fp.suffix == ".npz":
            peaks = dict(np.load(fp))
            jpath = fp.with_name("forward.json")
        if jpath is not None and jpath.exists():
            import json as _json
            fwd_json = _json.loads(jpath.read_text())
        if peaks:
            fig_fwd = _forward_figure(peaks, placement, tmap.registration, fwd_json,
                                      c_map=_fwd_cmap, c0=_fwd_c0, bone_threshold=_fwd_thr)
            if fig_fwd is not None:
                png_fwd = _fig_png(fig_fwd)
    png_3d = _fig_png(_map3d_figure(tmap))
    png_sc = _fig_png(_scene3d_figure(tmap, placement, bowl_radius_mm, aperture_mm))
    png_scw = _fig_png(_scene3d_figure(tmap, placement, bowl_radius_mm, aperture_mm,
                                       frame="device"))
    tmp = out.with_suffix(".placement.png")
    preview_placement(tmap, placement, out_png=str(tmp), title=title)
    png_pl = base64.b64encode(tmp.read_bytes()).decode()
    tmp.unlink()

    fmt = lambda v: "(" + ", ".join(f"{x:.1f}" for x in np.asarray(v, float)) + ") mm"
    rows = [
        ("Target" + (f" ({target_name})" if target_name else ""), f"{fmt(tgt)} {frame_lbl}"),
        ("Window centre", f"{fmt(win)} {frame_lbl}"),
        *((("Window ≈ EEG 10-20 site",
            "%s  (%.0f mm away)" % nearest_eeg_site(win)),) if u.is_mni else ()),
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
        eeg = ""
        if u.is_mni:
            asite, adist = nearest_eeg_site(wm)
            eeg = f"<td>{asite} ({adist:.0f} mm)</td>"
        arows.append(f"<tr><td>{rank}</td><td>{fmtv(wm)} mm</td>{eeg}"
                     f"<td>{100 * u.obj[i]:.0f}%</td>"
                     f"<td>{u.inc_deg[i]:.0f}°</td>"
                     f"<td>{float(np.asarray(tmap.rad_mm, float)[i]):.1f} mm</td></tr>")
    alts_table = (f"<table><tr><td>#</td><td>window ({frame_lbl})</td>"
                  + ("<td>≈ EEG site</td>" if u.is_mni else "")
                  + "<td>objective vs best</td><td>incidence</td><td>target distance</td>"
                  "</tr>" + "".join(arows) + "</table>"
                  "<p class=\"note\">Ranked fallbacks (orange stars on the maps above), "
                  "each ≥30 mm from the chosen window and from each other — for when "
                  "hair, hardware, or a frame blocks the optimum. Columns follow the "
                  "manuscript Table 1 (window, incidence, distance).</p>"
                  ) if arows else "<p>(no separated alternatives found)</p>"
    movie_html = ""
    if movie is not None and Path(movie).exists() and Path(movie).suffix.lower() == ".gif":
        b64 = base64.b64encode(Path(movie).read_bytes()).decode()
        movie_html = (f'<img src="data:image/gif;base64,{b64}" alt="wave propagation"/>\n')
    elif movie is not None and Path(movie).exists():
        gif = Path(movie).with_suffix(".gif")             # prefer a sibling gif for the HTML
        if gif.exists():
            b64 = base64.b64encode(gif.read_bytes()).decode()
            movie_html = (f'<img src="data:image/gif;base64,{b64}" alt="wave propagation"/>\n')
    anatomy_html = ""
    if png_ortho:
        what = atlas_label if (atlas is not None and atlas_ids is not None) else "target"
        anatomy_html = (
            "<h2>2 · The target in the skull</h2>\n"
            f'<img src="data:image/png;base64,{png_ortho}" alt="anatomy ortho-slices"/>\n'
            + (f'<img src="data:image/png;base64,{png_t3d}" alt="target in 3-D"/>\n'
               if png_t3d else "")
            + '<p class="note">Sagittal, coronal and axial cuts through the target, taken '
              f'from the same sound-speed volume the solver used, with the {what} overlaid '
              'where an atlas was supplied. The 3-D scene is the manuscript Figure-1 view: '
              'the structure inside a semi-transparent skull, with the chosen window and '
              'the beam axis marked. Slices and coordinates are in the frame named in the '
              'footer.</p>\n')
    forward_html = ""
    if png_fwd or fwd_json:
        rows_f = []
        if fwd_json:
            tc, ff = fwd_json["transcranial"], fwd_json["free_field"]
            rows_f = [("Pressure at the target",
                       f"{tc['peak_at_target_pa']:.2f} Pa through the skull vs "
                       f"{ff['peak_at_target_pa']:.2f} Pa free field"),
                      ("Transmission at the target",
                       f"{fwd_json.get('transmission_at_target', float('nan')):.2f}x "
                       f"({fwd_json.get('transmission_at_target_db', float('nan')):+.1f} dB)"),
                      ("Peak anywhere in the box",
                       f"{tc['peak_pa']:.2f} Pa vs {ff['peak_pa']:.2f} Pa free field"),
                      ("That peak's distance from the target",
                       f"{tc['focal_shift_mm']:.2f} mm vs {ff['focal_shift_mm']:.2f} mm free field"),
                      ("-6 dB focal width",
                       " x ".join(f"{v:.2f}" for v in tc["fwhm_mm"]) + " mm vs " +
                       " x ".join(f"{v:.2f}" for v in ff["fwhm_mm"]) + " mm free field")]
        trs_f = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows_f)
        forward_html = (f"<h2>{_n if False else '{n_fwd}'} · Transcranial vs free field</h2>\n"
                        + (f"<table>{trs_f}</table>\n" if trs_f else "")
                        + (f'<img src="data:image/png;base64,{png_fwd}" alt="forward pair"/>\n'
                           if png_fwd else "")
                        + '<p class="note">Two forward solves on one grid with the same source '
                          'and drive, differing only in the medium: the skull, then water '
                          'everywhere. Their ratio at the target is the insertion loss of this '
                          'placement. Quote the at-target number, not the box maximum -- a '
                          'skull that aberrates more than it absorbs can put a brighter spot '
                          'off target and make the maximum look like a gain.</p>\n')
    _n = 2 + (1 if anatomy_html else 0)
    if forward_html:                          # numbered in DOCUMENT order: forward, then movie
        forward_html = forward_html.replace("{n_fwd}", str(_n))
        _n += 1
    if movie_html or movie is not None:
        cap = movie_caption or ("The outward wave leaving the target and sweeping through "
                                "the skull. It plays in this HTML, and in the PDF on the "
                                "appended animation page (Acrobat and compatible viewers; "
                                "other readers show one frame).")
        movie_html = (f"<h2>{_n} · Wave propagation</h2>\n" + movie_html
                      + f'<p class="note">{cap}</p>\n')
        _n += 1
    n_surf, n_inc, n_obj, n_alt, n_par = _n, _n + 1, _n + 2, _n + 3, _n + 4
    n_meth = _n + 5
    prov_html = _provenance_html(_provenance_rows(
        tmap, placement, bundle, bowl_radius_mm=bowl_radius_mm, aperture_mm=aperture_mm,
        theta_max_deg=theta_max_deg, extra=parameters))
    fwd_scope = ("" if fwd_json else
                 " (a transcranial-vs-free-field pair is one; see the forward module)")
    stamp = datetime.date.today().isoformat()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto;
        color: #1d1d1f; }}
 h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
 table {{ border-collapse: collapse; }} td {{ border: 1px solid #ccd; padding: 5px 12px; }}
 td:first-child {{ background: #f4f6f9; font-weight: 600; }}
 th {{ border: 1px solid #ccd; padding: 5px 12px; background: #e7ebf1; text-align: left; }}
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

<p class="note"><b>Scope.</b> This reports where to couple a transducer for one target on
one skull, and what that placement delivers. It comes from a single full-wave time-reversal
solve: a virtual source at the target radiates outward, and by reciprocity the field
recorded on the skull is the transmit coupling of every candidate window. Everything after
the solve -- the window search, the alternatives, the figures -- is post-processing on that
one recorded field. It does <i>not</i> establish the focal volume, off-target lobes, or
safety margins; those need the inward re-simulation{fwd_scope}.</p>

<h2>1 · Placement summary</h2>
<table>{trs}</table>

{anatomy_html}
{forward_html}
{movie_html}
<h2>{n_surf} · Surface coupling and bone transmission</h2>
<img src="data:image/png;base64,{png_3d}" alt="transparency map 3-D"/>
<img src="data:image/png;base64,{png_t}" alt="transparency unwrapped"/>
<p class="note">Transmitted amplitude in dB (1/r&sup2;-corrected) — for <em>seeing</em>
where the bone transmits, in the manuscript Fig.&nbsp;5 presentation. Bright = transparent
bone; grey = no direct-arrival coverage. This corrected map is a visualisation — placement
uses the raw quantity below.</p>

<h2>{n_inc} · Beam incidence and the legality limit</h2>
<img src="data:image/png;base64,{png_i}" alt="incidence unwrapped"/>
<p class="note">Angle between the target→patch ray and the bone's outward normal. Beyond
the legality limit the fluid model's longitudinal transmission gives way to unmodelled
shear conversion, so those windows are excluded from placement (manuscript, Discussion:
the water–bone longitudinal critical angle).</p>

<h2>{n_obj} · The placement objective and the chosen window</h2>
<img src="data:image/png;base64,{png_o}" alt="objective unwrapped"/>
<img src="data:image/png;base64,{png_sc}" alt="placement scene 3-D"/>
<img src="data:image/png;base64,{png_scw}" alt="placement scene 3-D, device scale"/>
<img src="data:image/png;base64,{png_pl}" alt="placement"/>
<p class="note">The scene shows the transducer surface seated at its pose on the
semi-transparent skull with the target marked inside the head (manuscript Figs.&nbsp;1
and&nbsp;8 style). The first pair is framed on the <em>head</em>, which reads the window
against the anatomy but crops the dish whenever the device is large next to the head; the
second pair is framed on the <em>device</em>, so the whole bowl and its stand-off are
visible at the scale the hardware actually has. &radic;J<sub>w</sub> — incidence-weighted <em>raw</em> delivered intensity
integrated over the bowl footprint, illegal patches excluded (manuscript Eqs.&nbsp;7
&amp;&nbsp;14; the field of Fig.&nbsp;8). The red star is the chosen window; the bright
lobe's breadth is the placement margin quoted in the summary.</p>

<h2>{n_alt} · Alternative windows (if the optimum is blocked)</h2>
{alts_table}

<h2>{n_par} · Parameters and their provenance</h2>
{prov_html}
<p class="note">Every number this result depends on, labelled by where it came from.
<b>Measured</b> is the device or the scan; <b>Literature</b> is a published value;
<b>Derived</b> follows from the others; <b>Convention</b> is a choice of the pipeline; and
<b>Assumption</b> is a value chosen by judgement, which is where a sensitivity check earns
its keep.</p>

<h2>{n_meth} · Methods</h2>
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
<footer>skull-transparency report — generated {stamp}. Coordinates are {frame_lbl} RAS mm. The map
comes from one full-wave time-reversal solve (virtual source at the target); placement and
every figure here are post-processing on that recorded field. Method, conventions, and the
figure/equation/table numbers cited above: {_MANUSCRIPT}. This file is self-contained —
print to PDF for archival.</footer>
</body></html>
"""
    out.write_text(html)
    if not want_pdf:
        return out
    pdf = html_to_pdf(out)
    if movie is not None and Path(movie).exists():
        append_animated_page(pdf, movie, caption=(movie_caption or
                             "Outward wave from the target through the skull."))
    return pdf
