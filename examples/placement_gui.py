#!/usr/bin/env python
"""Interactive napari control panel for transducer placement on a skull transparency map.

A *thin* GUI that DELEGATES to the toolkit (``compute_transparency_map``,
``place_bowl_optimal``, the footprint-sum score) — it adds no physics of its own. It
consumes a Field Bundle (produced by the sim/solve layer, or ``st.make_synthetic_bundle``)
and lets you explore where to put a transducer:

  * point-size slider        — surface point size (live)
  * footprint-radius slider  — the array half-aperture; live yellow footprint preview
  * [1] Find optimal placement — ``place_bowl_optimal`` (argmax surface-integral) -> green window
  * [2] Score marker          — drag the cyan marker onto the skull, read score vs the optimum
  * toggles                   — orientation labels (L/R/A/P/S/I), element overlay, white background
  * terminal log              — colour-coded, timestamped record of data/code paths + every action

Run (needs the ``[viz]`` extra: ``pip install -e '.[viz]'`` and a display):

    python examples/placement_gui.py --bundle <bundle_dir> [--aperture-mm 65] [--elements probe_xy.npy]

Zero-data demo (no sim needed) — make a synthetic bundle first:

    python -c "import skull_transparency as st; st.make_synthetic_bundle('synthetic_bundle')"
    python examples/placement_gui.py --bundle synthetic_bundle

``--elements`` is optional: an ``(N,3)`` array of device-frame element positions (planar,
mm, centred) for your probe; if given, the elements are overlaid on the chosen window.
"""
import argparse, os, sys, time
import numpy as np
import napari
from magicgui.widgets import Container, PushButton, FloatSlider, TextEdit, CheckBox

import skull_transparency as st
from skull_transparency.placement import BowlConstraints, place_bowl_optimal, _incidence_cos
from skull_transparency.position_tool import _surface_points
from skull_transparency.score import PositioningScore


# 16-anchor sampling of MATLAB 'parula' (napari interpolates to 256); embedded so the
# example is self-contained (no data file).
_PARULA = np.array([
    [0.2422, 0.1504, 0.6603], [0.2717, 0.2184, 0.8439], [0.2814, 0.3095, 0.9483],
    [0.2647, 0.4030, 0.9935], [0.1847, 0.5030, 0.9819], [0.1540, 0.5902, 0.9218],
    [0.1085, 0.6669, 0.8734], [0.0009, 0.7248, 0.7815], [0.1609, 0.7635, 0.6671],
    [0.2809, 0.7964, 0.5266], [0.5044, 0.7993, 0.3480], [0.7344, 0.7679, 0.1852],
    [0.9184, 0.7308, 0.1890], [0.9962, 0.7798, 0.2095], [0.9619, 0.8840, 0.1557],
    [0.9769, 0.9839, 0.0805]])


def _parula():
    """black- and white-anchored 'parula' napari colormaps (value 0 -> black / white)."""
    from napari.utils.colormaps import Colormap
    return (Colormap(np.vstack([[0, 0, 0], _PARULA]), name="parula_black"),
            Colormap(np.vstack([[1, 1, 1], _PARULA]), name="parula_white"))


def launch(bundle_dir, aperture_mm=65.0, elements_file=None):
    radius0 = aperture_mm / 2.0
    t_start = time.time()
    bundle = st.load_bundle(bundle_dir)
    tmap = st.compute_transparency_map(bundle)
    P, frame = _surface_points(tmap)
    coupling = np.asarray(tmap.Ipk_Wcm2, float)
    Tw = coupling * np.clip(_incidence_cos(tmap), 0.0, 1.0) ** 2     # incidence-weighted (== place_bowl)
    target = np.asarray(tmap.registration.target_mni_mm, float)

    # optional transducer element geometry (planar device-frame XY, mm); None -> no overlay
    elems = np.load(elements_file) if (elements_file and os.path.exists(elements_file)) else None

    # log-scale + percentile-clip coupling so the colour variation is visible
    cpos = coupling[coupling > 0]
    disp = np.log10(np.clip(coupling, cpos.min() if cpos.size else 1e-30, None))
    lo, hi = np.percentile(disp, [40, 99.5]); disp = np.clip(disp, lo, hi)
    cm_black, cm_white = _parula()

    v = napari.Viewer(title=f"skull-transparency placement — {os.path.basename(bundle_dir)}")
    v.dims.ndisplay = 3
    surf = v.add_points(P, features={"coupling": disp}, face_color="coupling",
                        face_colormap=cm_black, size=0.4, border_width=0, blending="translucent",
                        name="transparency (log coupling)", shading="none")
    tgt = v.add_points(target[None], face_color="white", symbol="x", size=7.0, border_width=0, name="target")
    win = v.add_points(np.empty((0, 3)), face_color="green", size=7.0, border_width=0, name="optimal window")
    foot = v.add_points(np.empty((0, 3)), face_color="yellow", size=1.0, border_width=0,
                        opacity=0.55, name="footprint")
    marker = v.add_points(P[int(np.argmax(Tw))][None], face_color="cyan", size=8.0,
                          border_width=0, name="marker (drag me)")
    arr = (v.add_points(np.empty((0, 3)), face_color="white", size=0.9, border_width=0,
                        name=f"array elements ({len(elems)})") if elems is not None else None)

    # anatomical orientation labels (assumes a RAS-like world: +x=R,-x=L,+y=A,-y=P,+z=S,-z=I)
    d = float(np.percentile(np.linalg.norm(P - target, axis=1), 97)) * 1.06
    axes = {"R": (d, 0, 0), "L": (-d, 0, 0), "A": (0, d, 0), "P": (0, -d, 0),
            "S": (0, 0, d), "I": (0, 0, -d)}
    lab_pos = np.array([target + np.array(o) for o in axes.values()])
    orient = v.add_points(lab_pos, features={"lab": list(axes.keys())},
                          text={"string": "{lab}", "size": 16, "color": "white", "anchor": "center"},
                          face_color="transparent", border_width=0, size=0.1, name="orientation")

    # ---------------- widgets + colour-coded terminal log ----------------
    psize = FloatSlider(value=0.4, min=0.1, max=3.0, step=0.1, label="point size")
    radius = FloatSlider(value=radius0, min=5.0, max=50.0, step=0.5, label="footprint radius mm")
    logw = TextEdit(value="")
    try:
        logw.native.setReadOnly(True); logw.native.setMinimumHeight(180)
        logw.native.setStyleSheet("background-color:#0e1116; font-family:monospace; font-size:13px;")
    except Exception: pass

    from datetime import datetime
    import html as _html
    _COL = {"hint": "#7fd6ff", "data": "#b8b8b8", "action": "#ffcc55",
            "result": "#76e36b", "info": "#e0e0e0", "warn": "#ff8a8a"}
    _entries = []
    def log(msg, kind="info"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        _entries.append((_COL.get(kind, "#e0e0e0"), ts, _html.escape(str(msg))))
        body = "<br>".join(f'<span style="color:{c}">[{t}] {m}</span>' for c, t, m in _entries[-300:])
        try:
            logw.native.setHtml(body)
            sb = logw.native.verticalScrollBar(); sb.setValue(sb.maximum())
        except Exception:
            logw.value = "\n".join(f"[{t}] {m}" for _, t, m in _entries[-300:])
        print(f"[{ts}] {msg}", flush=True)

    def _proc():
        try:
            from qtpy.QtWidgets import QApplication; QApplication.processEvents()
        except Exception: pass

    arr_name = os.path.basename(elements_file) if elems is not None else "(none — pass --elements)"
    for ln in ("── DATA ──",
               f"bundle:  {os.path.abspath(bundle_dir)}",
               f"array:   {arr_name}  (aperture {aperture_mm:.0f} mm)",
               f"target:  {np.round(target, 1)} mm  ({frame})",
               "── CODE ──",
               f"python:  {sys.executable}",
               f"toolkit: {os.path.dirname(st.__file__)}",
               "calls:   compute_transparency_map -> place_bowl_optimal -> footprint-sum score",
               f"surface: {len(P)} patches", "──────────"):
        log(ln, "data")
    log(f"loaded bundle ({len(P)} patches) in {time.time()-t_start:.1f}s", "info")

    def _footprint(center, r):
        return np.linalg.norm(P - center, axis=1) <= r

    def array_world(center):
        """Place the planar array centred at `center`, in the plane perpendicular to the
        beam (window->target), so it sits tangent to the surface facing the target."""
        if elems is None:
            return np.empty((0, 3))
        beam = target - center; n = np.linalg.norm(beam)
        if n == 0:
            return np.empty((0, 3))
        beam /= n
        up = np.array([0, 0, 1.0]) if abs(beam[2]) < 0.9 else np.array([1.0, 0, 0])
        t1 = np.cross(beam, up); t1 /= np.linalg.norm(t1)
        t2 = np.cross(beam, t1)
        return center + np.outer(elems[:, 0], t1) + np.outer(elems[:, 1], t2)

    def _place_array_at(center):
        if arr is not None:
            arr.data = array_world(center)

    def _on_marker_move(*_):
        c = np.asarray(marker.data)[-1]
        _place_array_at(P[int(np.argmin(np.linalg.norm(P - c, axis=1)))])
    try: marker.events.data.connect(_on_marker_move)
    except Exception: pass
    _place_array_at(P[int(np.argmax(Tw))])

    psize.changed.connect(lambda *_: setattr(surf, "size", float(psize.value)))

    _cache = {}
    def _best(r):
        if r not in _cache:
            log(f"running place_bowl_optimal(bowl_radius_mm={r}) ... surface-integral search", "action"); _proc()
            t = time.time()
            pl = place_bowl_optimal(tmap, BowlConstraints(bowl_radius_mm=r))
            cw = np.asarray(pl.window_center_mni_mm, float)
            _cache[r] = (float(Tw[_footprint(cw, r)].sum()), pl)
            log(f"  -> optimal window {np.round(cw,1)} mm  ({time.time()-t:.1f}s)", "result")
        return _cache[r]

    def preview(*_):
        c = np.asarray(marker.data)[-1]
        cw = P[int(np.argmin(np.linalg.norm(P - c, axis=1)))]
        foot.data = P[_footprint(cw, float(radius.value))]
    radius.changed.connect(preview)

    def find_optimal():
        r = float(radius.value); _, pl = _best(r)
        cw = np.asarray(pl.window_center_mni_mm, float); inside = _footprint(cw, r)
        win.data = cw[None]; foot.data = P[inside]; _place_array_at(cw)
        sc = PositioningScore.from_placement(pl, "target")
        log(f"FIND OPTIMAL @ r={r}mm: window {np.round(cw,1)} | score {sc.normalized:.3f} | "
            f"proxy {sc.focal_pressure_proxy:.3g} | incidence {sc.incidence_deg:.1f}° | {int(inside.sum())} patches", "result")

    def score_marker():
        c = np.asarray(marker.data)[-1]
        nearest = int(np.argmin(np.linalg.norm(P - c, axis=1)))
        cw = P[nearest]; r = float(radius.value); inside = _footprint(cw, r)
        J = float(Tw[inside].sum()); proxy = float(np.sqrt(max(J, 0.0)))
        win.data = cw[None]; foot.data = P[inside]; _place_array_at(cw)
        bestJ, _ = _best(r); norm = (J / bestJ) if bestJ > 0 else float("nan")
        inc = float(np.degrees(np.arccos(np.clip(_incidence_cos(tmap)[nearest], -1, 1))))
        log(f"SCORE MARKER {np.round(cw,1)}: score {norm:.3f} | proxy {proxy:.3g} | "
            f"incidence {inc:.1f}° | {int(inside.sum())} patches", "result")

    showlab = CheckBox(value=True, text="orientation labels (L/R/A/P/S/I)")
    showlab.changed.connect(lambda *_: setattr(orient, "visible", bool(showlab.value)))
    showarr = CheckBox(value=True, text="transducer elements overlay")
    if arr is not None:
        showarr.changed.connect(lambda *_: setattr(arr, "visible", bool(showarr.value)))
    bgwhite = CheckBox(value=False, text="white background (white = 0)")
    def set_bg(*_):
        white = bool(bgwhite.value)
        try: v.theme = "light" if white else "dark"
        except Exception: pass
        for attr in ("_qt_viewer", "qt_viewer"):
            qv = getattr(v.window, attr, None)
            if qv is not None:
                try: qv.canvas.background_color_override = "white" if white else "black"
                except Exception: pass
        try:
            surf.face_colormap = cm_white if white else cm_black; surf.blending = "translucent"; surf.refresh()
        except Exception: pass
        if arr is not None:
            arr.face_color = "black" if white else "white"
        tgt.face_color = "black" if white else "white"
        try: orient.text.color = "black" if white else "white"; orient.refresh()
        except Exception: pass
    bgwhite.changed.connect(set_bg)

    b1 = PushButton(text="① Find optimal placement"); b1.clicked.connect(find_optimal)
    b2 = PushButton(text="② Score marker (drag the cyan point first)"); b2.clicked.connect(score_marker)
    panel = Container(widgets=[psize, radius, showlab, showarr, bgwhite, b1, b2, logw], labels=False)
    dock = v.window.add_dock_widget(panel, name="placement", area="right")
    try:
        for w in (logw.native, panel.native):
            w.setMinimumWidth(110)
        dock.setMinimumWidth(115)
        v.window._qt_window.resizeDocks([dock], [320], 1)
    except Exception:
        pass

    for ln in ("HOW TO USE — quick start:",
               "1. In the layer list (left), click the 'marker (drag me)' layer to select it.",
               "2. Click the 'select points' arrow in the layer controls, then drag the CYAN",
               "   marker onto the surface where you'd place the array.",
               "3. Press [② Score marker] -> scores that window vs the optimum (green lines).",
               "4. Press [① Find optimal placement] -> best window (green) + footprint (yellow).",
               "5. Toggles: orientation labels - element overlay - white background.",
               "   Sliders: point size - footprint radius (= array half-aperture).",
               "Legend: cyan = marker - green = optimal window - yellow = footprint",
               "        white dots = transducer elements (--elements) - white X = target."):
        log(ln, "hint")
    napari.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Interactive transducer-placement GUI on a transparency map.")
    ap.add_argument("--bundle", required=True, help="Field Bundle directory")
    ap.add_argument("--aperture-mm", type=float, default=65.0, help="transducer aperture (mm)")
    ap.add_argument("--elements", default=None, help="optional (N,3) .npy of device-frame element positions (mm)")
    a = ap.parse_args()
    launch(a.bundle, a.aperture_mm, a.elements)
