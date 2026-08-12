"""One-command placement report: a self-contained HTML page with the transparency map,
the chosen placement, and the numbers an experimenter needs to write down (target and
window coordinates, incidence, score, nearest EEG 10-20 site). Everything is embedded
(base64 PNGs), so the file can be mailed or attached to a protocol as-is.

Headless-safe: uses the matplotlib Agg backend, no display needed.
"""
from __future__ import annotations

import base64
import datetime
import io
from pathlib import Path

import numpy as np

from .atlas_targets import nearest_eeg_site


def _fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=115, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _map_figure(tmap):
    """The transparency map on the skull surface from four viewpoints (amplitude dB)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    P = np.asarray(tmap.surf_mni_mm() if tmap.registration is not None else tmap.surf_vox, float)
    amp = np.sqrt(np.maximum(np.asarray(tmap.value, float), 0.0))
    ref = np.percentile(amp, 98.0) or 1.0
    db = 20.0 * np.log10(np.maximum(amp / ref, 1e-2))
    vlo, vhi = np.percentile(db, 25.0), np.percentile(db, 99.0)
    rhat = np.asarray(tmap.rhat, float)

    fig = plt.figure(figsize=(16, 4.2))
    sc = None
    for k, (elev, azim) in enumerate([(18, -60), (18, 60), (18, 180), (78, -90)]):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        e, a = np.radians(elev), np.radians(azim)
        cam = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        m = rhat @ cam > 0.0                    # camera-facing hemisphere only
        sc = ax.scatter(*P[m].T, c=db[m], cmap="inferno", vmin=vlo, vmax=vhi, s=4, linewidths=0)
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
                 target_name=None) -> Path:
    """Write the self-contained HTML report; returns its path."""
    from .position_tool import preview_placement
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_html)
    tgt = np.asarray(placement.target_mni_mm, float)
    win = np.asarray(placement.window_center_mni_mm, float)
    site, site_mm = nearest_eeg_site(win)
    map_png = _fig_png(_map_figure(tmap))
    # placement figure via the existing preview (rendered to a temp png buffer)
    tmp = out.with_suffix(".placement.png")
    preview_placement(tmap, placement, out_png=str(tmp), title=title)
    place_png = base64.b64encode(tmp.read_bytes()).decode()
    tmp.unlink()

    fmt = lambda v: "(" + ", ".join(f"{x:.1f}" for x in np.asarray(v, float)) + ") mm"
    rows = [
        ("Target" + (f" ({target_name})" if target_name else ""), fmt(tgt) + " MNI"),
        ("Window centre", fmt(win) + " MNI"),
        ("Window-to-target distance", f"{np.linalg.norm(win - tgt):.1f} mm"),
        ("Beam incidence at window", f"{float(placement.incidence_deg):.1f} deg"),
        ("Transparency score", f"{float(placement.transparency_score):.3f}"),
        ("Nearest EEG 10-20 site", f"{site}  ({site_mm:.0f} mm from the window centre)"),
        ("Surface patches", f"{len(np.asarray(tmap.surf_vox)):,}"),
        ("Median target depth", f"{float(np.median(tmap.rad_mm)):.0f} mm"),
    ]
    trs = "\n".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    stamp = datetime.date.today().isoformat()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 2em auto; color: #1d1d1f; }}
 h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 1.6em; }}
 table {{ border-collapse: collapse; }} td {{ border: 1px solid #ccd; padding: 5px 12px; }}
 td:first-child {{ background: #f4f6f9; font-weight: 600; }}
 img {{ max-width: 100%; }} footer {{ color: #778; font-size: 0.85em; margin-top: 2em; }}
</style></head><body>
<h1>{title}</h1>
<h2>Placement summary</h2>
<table>{trs}</table>
<h2>Skull transparency map</h2>
<img src="data:image/png;base64,{map_png}" alt="transparency map"/>
<h2>Chosen placement</h2>
<img src="data:image/png;base64,{place_png}" alt="placement"/>
<footer>skull-transparency report — generated {stamp}. Coordinates are MNI RAS mm; the
transparency map is referenced to a virtual source at the target (one full-wave
time-reversal solve; placement is post-processing on the recorded field).</footer>
</body></html>
"""
    out.write_text(html)
    return out
