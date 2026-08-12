#!/usr/bin/env python
"""Thin napari control panel for transducer placement on a skull transparency map.

DELEGATES to the toolkit (compute_transparency_map, place_bowl_optimal, footprint-sum
score) — no re-implemented physics. Widgets:
  * point-size slider   -> live point size for the surface
  * footprint radius slider -> live footprint preview (the array half-aperture)
  * ① Find optimal placement -> place_bowl_optimal (argmax surface-integral) + score
  * ② Score marker -> drag the magenta marker onto the skull, read score vs optimum
  * status panel -> data/code paths + a timestamped activity log of what's running

Usage: DISPLAY=:1 python skull_transparency_gui.py --bundle <bundle_dir> [--aperture-mm 65]
The heavy outward sim is a backend orta job; this panel consumes the Field Bundle.
"""
import argparse, os, sys, time
import numpy as np
import napari
from magicgui.widgets import Container, PushButton, Label, FloatSlider, TextEdit

import skull_transparency as st
from skull_transparency.placement import BowlConstraints, place_bowl_optimal, _incidence_cos
from skull_transparency.position_tool import _surface_points
from skull_transparency.score import PositioningScore


def launch(bundle_dir, aperture_mm=65.0):
    radius0 = aperture_mm / 2.0
    t_start = time.time()
    bundle = st.load_bundle(bundle_dir)
    tmap = st.compute_transparency_map(bundle)
    P, frame = _surface_points(tmap)
    coupling = np.asarray(tmap.Ipk_Wcm2, float)
    w_inc = np.clip(_incidence_cos(tmap), 0.0, 1.0) ** 2
    Tw = coupling * w_inc
    target = np.asarray(tmap.registration.target_mni_mm, float)

    # transducer array geometry (device-frame element positions, planar, centred, mm)
    elem_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sparse1024_elements_mm.npy")
    elems = np.load(elem_file) if os.path.exists(elem_file) else None

    # log-scale + percentile-clip coupling so colour variation is visible
    cpos = coupling[coupling > 0]
    disp = np.log10(np.clip(coupling, cpos.min() if cpos.size else 1e-30, None))
    lo, hi = np.percentile(disp, [40, 99.5]); disp = np.clip(disp, lo, hi)

    # modified MATLAB parula (vendored parula_256.npy beside this script — no external repo),
    # anchored at black (black bg, 0=black) or white (white bg, 0=white)
    from napari.utils.colormaps import Colormap
    _par_npy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parula_256.npy")
    try:
        par = np.load(_par_npy)[:, :3]
    except Exception:
        par = np.array([[0.2081,0.1663,0.5292],[0.0779,0.5040,0.8384],[0.0227,0.6418,0.7906],
                        [0.2186,0.7276,0.6196],[0.5044,0.7993,0.3480],[0.9959,0.7698,0.2031],
                        [0.9763,0.9831,0.0538]])
    cm_black = Colormap(np.vstack([[0, 0, 0], par]), name="parula_black")
    cm_white = Colormap(np.vstack([[1, 1, 1], par]), name="parula_white")

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

    # anatomical orientation labels L/R/A/P/S/I (Halle RAS: +x=R,-x=L,+y=A,-y=P,+z=S,-z=I)
    d = float(np.percentile(np.linalg.norm(P - target, axis=1), 97)) * 1.06
    axes = {"R": (d, 0, 0), "L": (-d, 0, 0), "A": (0, d, 0), "P": (0, -d, 0),
            "S": (0, 0, d), "I": (0, 0, -d)}
    lab_pos = np.array([target + np.array(o) for o in axes.values()])
    orient = v.add_points(lab_pos, features={"lab": list(axes.keys())},
                          text={"string": "{lab}", "size": 16, "color": "white", "anchor": "center"},
                          face_color="transparent", border_width=0, size=0.1, name="orientation")

    # ---------------- widgets ----------------
    psize = FloatSlider(value=0.4, min=0.1, max=3.0, step=0.1, label="point size")
    radius = FloatSlider(value=radius0, min=5.0, max=50.0, step=0.5, label="footprint radius mm")
    logw = TextEdit(value="")
    try:
        logw.native.setReadOnly(True); logw.native.setMinimumHeight(180)
        # fixed dark "terminal" box so the colour-coded text stays readable on either bg
        logw.native.setStyleSheet("background-color:#0e1116; font-family:monospace; font-size:13px;")
    except Exception: pass

    from datetime import datetime
    import html as _html
    _COL = {"hint": "#7fd6ff", "data": "#b8b8b8", "action": "#ffcc55",
            "result": "#76e36b", "info": "#e0e0e0", "warn": "#ff8a8a"}
    _entries = []
    def log(msg, kind="info"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]   # date + ms-precise time
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

    # provenance (data + code paths) -> the GUI terminal log (and stdout)
    for ln in (
        "── DATA ──",
        f"bundle:  {os.path.abspath(bundle_dir)}",
        f"c-map:   {os.path.join(os.path.abspath(bundle_dir), 'skull_fullres_c.npy')}",
        f"array:   sparse1024_elements_mm.npy  (aperture {aperture_mm:.0f} mm)",
        f"target:  {np.round(target,1)} mm  ({frame})",
        "── CODE ──",
        f"python:  {sys.executable}",
        f"toolkit: {os.path.dirname(st.__file__)}",
        "calls:   compute_transparency_map -> place_bowl_optimal -> footprint-sum score",
        f"surface: {len(P)} patches",
        "──────────",
    ):
        log(ln, "data")
    log(f"loaded bundle ({len(P)} patches) in {time.time()-t_start:.1f}s; coupling log-clip [{lo:.2f},{hi:.2f}]", "info")

    def _footprint(center, r):
        return np.linalg.norm(P - center, axis=1) <= r

    def array_world(center):
        """Place the planar array centred at `center`, in the plane perpendicular to the
        beam (window→target), so it sits tangent to the skull facing the target."""
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

    def _on_marker_move(*_):           # array follows the (snapped) marker live
        c = np.asarray(marker.data)[-1]
        _place_array_at(P[int(np.argmin(np.linalg.norm(P - c, axis=1)))])
    try: marker.events.data.connect(_on_marker_move)
    except Exception: pass
    _place_array_at(P[int(np.argmax(Tw))])     # show the overlay at the initial marker

    # live point size
    def set_psize(*_):
        surf.size = float(psize.value)
    psize.changed.connect(set_psize)

    # optimal-window cache per radius (place_bowl_optimal is the ~10-20 s search)
    _cache = {}
    def _best(r):
        if r not in _cache:
            log(f"running place_bowl_optimal(bowl_radius_mm={r}) ... surface-integral search", "action"); _proc()
            t = time.time()
            pl = place_bowl_optimal(tmap, BowlConstraints(bowl_radius_mm=r))
            cw = np.asarray(pl.window_center_mni_mm, float)
            J = float(Tw[_footprint(cw, r)].sum())
            _cache[r] = (J, pl)
            log(f"  -> optimal window {np.round(cw,1)} mm  ({time.time()-t:.1f}s)", "result")
        return _cache[r]

    # live footprint preview when the radius slider moves
    def preview(*_):
        c = np.asarray(marker.data)[-1]
        nearest = int(np.argmin(np.linalg.norm(P - c, axis=1)))
        cw = P[nearest]; r = float(radius.value); inside = _footprint(cw, r)
        foot.data = P[inside]   # live footprint; numbers go to the log on ② Score marker
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

    # ============================================================================
    # FORWARD FOCUSING SOLVE (true fullwave on the GPU host) + focal-field overlay
    # ============================================================================
    # The focal pressure field at a CHOSEN depth comes from a real bench_3d_opt forward
    # solve (run_focus_solve.py -> orta). Depth is measured along the beam axis
    # unit(target - window/marker center): depth 0 = at target, + = deeper, - = shallower.
    import subprocess, json, glob, threading
    from magicgui.widgets import PushButton as _PB
    fdepth = FloatSlider(value=0.0, min=-25.0, max=25.0, step=1.0, label="focus depth mm (▶solve; 0=target)")
    fdr = FloatSlider(value=20.0, min=3.0, max=40.0, step=1.0, label="focal dyn range dB (shown)")
    focal = v.add_points(np.empty((0, 3)), face_color="magenta", size=1.2, border_width=0,
                         opacity=0.8, name="focal pressure (sim)", shading="none")
    _focus_state = {"proc": None, "dir": None, "lastpos": 0}

    def _beam_axis():
        c = np.asarray(marker.data)[-1]
        cw = P[int(np.argmin(np.linalg.norm(P - c, axis=1)))]
        ax = target - cw; n = np.linalg.norm(ax)
        return (ax / n) if n > 0 else np.array([0, 0, -1.0]), cw

    def focus_at_target():
        fdepth.value = 0.0
        log("focus depth set to 0 (at target). Press ▶ Run forward solve to simulate.", "hint")

    _focal_cache = {"pk": None, "cm": None, "info": None}

    def overlay_focal(focus_dir):
        try:
            pk = np.load(os.path.join(focus_dir, "focal_Pmax.npy"))
            cm = np.load(os.path.join(focus_dir, "focal_coords_mni.npy"))
            info = json.load(open(os.path.join(focus_dir, "focal_gain.json")))
        except Exception as e:
            log(f"could not load focal field from {focus_dir}: {e}", "warn"); return
        _focal_cache.update(pk=pk, cm=cm, info=info)
        _render_focal()
        log(f"FOCAL FIELD overlaid: gain {info['gain']:.3g} | focal peak {info['focal_peak']:.3g} | "
            f"peak@MNI {np.round(info['peak_loc_mni'],1).tolist()} | full {int(pk.size)}-voxel focal box "
            f"({fdr.value:.0f} dB range; use the 'focal dyn range dB' slider)", "result")

    def _render_focal(*_):
        pk = _focal_cache["pk"]; cm = _focal_cache["cm"]
        if pk is None:
            return
        # show the WHOLE recorded focal box as a dB-graded, translucent cloud: pressure in dB
        # below the peak, keep everything within `fdr` dB, fade + shrink low values so the bright
        # focal lobe stands out without hiding the surrounding field.
        DR = float(fdr.value)
        pdb = 20.0 * np.log10(np.clip(pk, pk.max() * 1e-6, None) / (pk.max() + 1e-30))
        keep = pdb >= -DR
        pn = np.clip((pdb[keep] + DR) / DR, 0.0, 1.0)   # 0 at -DR dB, 1 at peak
        focal.data = cm[keep]
        focal.features = {"p": pn}
        focal.face_color = "p"; focal.face_colormap = "magma"
        focal.size = 0.4 + 1.6 * pn ** 2
        focal.blending = "translucent"
    fdr.changed.connect(_render_focal)   # live dynamic-range adjust on the shown focal field

    def _poll_focus():
        st_ = _focus_state
        proc = st_["proc"]
        if proc is None:
            return
        dl = os.path.join(st_["dir"], "driver.log")
        if os.path.exists(dl):
            with open(dl) as f:
                f.seek(st_["lastpos"]); new = f.read(); st_["lastpos"] = f.tell()
            for ln in new.splitlines():
                if ln.strip():
                    log("  " + ln, "info")
        if proc.poll() is not None:           # finished
            _focus_state["proc"] = None
            if proc.returncode == 0:
                overlay_focal(st_["dir"])
            else:
                log(f"forward solve FAILED (rc={proc.returncode}); see {dl}", "warn")
            try: _timer.stop()
            except Exception: pass

    def run_forward_solve():
        if _focus_state["proc"] is not None:
            log("a forward solve is already running — wait for it to finish.", "warn"); return
        axis, cw = _beam_axis()
        d = float(fdepth.value)
        # clamp the depth so the focus (and a margin) stays inside the FOV — the recorded focal
        # box needs to be in-grid or you get a clipped edge artifact, not a real focus.
        reg = bundle.registration
        if reg is not None:
            Ngrid = float(bundle.grid["N"]); m = 12.0           # voxel margin from each face
            tv = np.asarray(reg.mni_to_fullres(target), float)
            dirv = np.asarray(reg.mni_to_fullres(target + axis), float) - tv   # voxels per mm
            d_lo, d_hi = -25.0, 25.0
            for i in range(3):
                if abs(dirv[i]) < 1e-9:
                    continue
                lim = sorted(((m - tv[i]) / dirv[i], (Ngrid - 1 - m - tv[i]) / dirv[i]))
                d_lo = max(d_lo, lim[0]); d_hi = min(d_hi, lim[1])
            dc = float(np.clip(d, d_lo, d_hi))
            if abs(dc - d) > 0.5:
                log(f"focus depth {d:+.0f} mm would put the focus OUTSIDE the FOV — clamped to "
                    f"{dc:+.0f} mm (in-grid range here ≈ [{d_lo:+.0f},{d_hi:+.0f}] mm). "
                    f"Move the marker for a different beam axis to reach deeper.", "warn")
                d = dc; fdepth.value = dc
        focus_mni = target + d * axis
        fid = "target" if abs(d) < 1e-6 else f"d{d:+.0f}".replace("+", "p").replace("-", "m")
        outsub = f"focus_geo_{fid}"
        fdir = os.path.join(os.path.dirname(os.path.abspath(bundle_dir.rstrip("/"))), outsub)
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_focus_solve.py"),
               "--bundle", os.path.abspath(bundle_dir), "--mode", "geo",
               # '=' form: focus_mni leads with a negative number, which argparse would eat as a flag
               "--focus-mni=" + ",".join(f"{x:.2f}" for x in focus_mni), "--id", fid]
        log(f"▶ RUN forward solve: depth {d:+.0f} mm along beam → focus MNI "
            f"{np.round(focus_mni,1).tolist()} (window {np.round(cw,1).tolist()})", "action")
        log("  launching true fullwave solve on the GPU host — this takes a few minutes; "
            "progress streams below.", "hint")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            log(f"could not launch run_focus_solve.py: {e}", "warn"); return
        _focus_state.update(proc=proc, dir=fdir, lastpos=0)
        os.makedirs(fdir, exist_ok=True)
        try: _timer.start(1500)
        except Exception: pass

    from qtpy.QtCore import QTimer
    _timer = QTimer(); _timer.timeout.connect(_poll_focus)

    # overlay any focal solve already on disk (most recent by mtime), so reopening the GUI shows it
    _existing = [d for d in glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(bundle_dir.rstrip("/"))), "focus_geo_*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "focal_gain.json"))]
    _existing.sort(key=os.path.getmtime)
    if _existing:
        log(f"found {len(_existing)} prior focal solve(s); overlaying the latest "
            f"({os.path.basename(_existing[-1])})", "info")
        overlay_focal(_existing[-1])

    from magicgui.widgets import CheckBox
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
            surf.face_colormap = cm_white if white else cm_black
            surf.blending = "translucent"
            surf.refresh()
        except Exception: pass
        if arr is not None:
            arr.face_color = "black" if white else "white"   # keep elements visible on either bg
        tgt.face_color = "black" if white else "white"        # target X flips with bg too
        try:                                                 # L/R/A/P/S/I labels: flip with bg
            orient.text.color = "black" if white else "white"
            orient.refresh()
        except Exception: pass
    bgwhite.changed.connect(set_bg)

    b1 = PushButton(text="① Find optimal placement"); b1.clicked.connect(find_optimal)
    b2 = PushButton(text="② Score marker (drag magenta first)"); b2.clicked.connect(score_marker)
    b3 = PushButton(text="③ Focus at target (depth 0)"); b3.clicked.connect(focus_at_target)
    b4 = PushButton(text="▶ Run forward solve (focus → pressure map)"); b4.clicked.connect(run_forward_solve)
    panel = Container(widgets=[psize, radius, fdepth, fdr, showlab, showarr, bgwhite,
                              b1, b2, b3, b4, logw], labels=False)
    dock = v.window.add_dock_widget(panel, name="placement", area="right")
    # allow the panel to be dragged narrow so more of the 3D view is visible. The log
    # TextEdit scrolls (no fixed-width Label), so nothing forces a wide minimum.
    try:
        for w in (logw.native, panel.native):
            w.setMinimumWidth(110)
        logw.native.setMinimumHeight(170)
        dock.setMinimumWidth(115)
        v.window._qt_window.resizeDocks([dock], [320], 1)  # start ~320 px (Qt.Horizontal)
    except Exception:
        pass

    for ln in (
        "HOW TO USE — quick start:",
        "1. In the layer list (left), click the 'marker (drag me)' layer to select it.",
        "2. In the toolbar (top-left of the layer controls), click the 'select points' arrow,",
        "   then drag the CYAN marker onto the skull where you'd place the array.",
        "3. Press [② Score marker] -> scores that window vs the optimum (see green/result lines).",
        "4. Press [① Find optimal placement] -> marks the best window (green) + footprint (yellow).",
        "5. Toggles: orientation labels - transducer overlay - white background.",
        "   Sliders: point size - footprint radius (= array half-aperture).",
        "FOCAL PRESSURE MAP (true fullwave forward solve on the GPU host):",
        "6. Set [focus depth mm] (0 = target, + = deeper along the beam), or press",
        "   [③ Focus at target]. Then [▶ Run forward solve] -> launches a real bench_3d_opt",
        "   solve on orta (~3 min; progress streams here), then overlays the focal pressure",
        "   field (magma) + prints the FOCAL GAIN. Depth is along unit(target - your marker).",
        "Legend: cyan = your marker - green = optimal window - yellow = footprint",
        "        white dots = 1024 array elements - white X = target (VIM)",
        "        magma cloud = simulated focal pressure (the focus).",
    ):
        log(ln, "hint")
    napari.run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--aperture-mm", type=float, default=65.0)
    a = ap.parse_args()
    launch(a.bundle, a.aperture_mm)
