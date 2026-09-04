#!/usr/bin/env python3
"""
setup_overview_camera.py — Interactive overview camera positioning tool.

Shows a captured overview image with the robot workspace projected onto it as
a cyan grid. Adjust six sliders (x, y, z, roll, pitch, yaw) until the cyan
grid aligns with the physical table in the image. Save the result once and
real_robot.launch.py will read it automatically on every subsequent launch.

Workflow
--------
1. Mount the overview RealSense D435i and connect it to the robot PC.
2. Start the simulation or real robot system so topics are active.
3. Capture one overview frame (run in another terminal):
       docker exec vlm_ros2 bash -c \\
         "source /opt/ros/humble/setup.bash && \\
          python3 /workspace/scripts/_capture_scene.py"
4. Run this script on the host:
       python scripts/setup_overview_camera.py
5. Click "Refresh Image" to load the captured frame.
6. Adjust the sliders until the cyan grid lines up with the table surface.
   - cyan grid  = robot table plane at z=0 m (panda_link0 frame)
   - red line   = base frame X-axis (forward)
   - green line = base frame Y-axis (left)
   - blue line  = base frame Z-axis (up)
7. Click "Save Config" — values are written to data/overview_camera_setup.json
   and the exact ros2 launch command is printed to the terminal.

On the next launch, real_robot.launch.py automatically reads overview_camera_setup.json.
Re-run this script whenever the camera is remounted.
Usage
-----
    python scripts/setup_overview_camera.py
    python scripts/setup_overview_camera.py --serial 242322071571
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IMG_PATH  = _REPO_ROOT / "data" / "scene_overview.png"
_INFO_PATH = _REPO_ROOT / "data" / "overview_camera_info.json"
_CFG_PATH  = _REPO_ROOT / "data" / "overview_camera_setup.json"

# RealSense D435i typical intrinsics @ 640×480 — overridden by camera_info if available
_DEFAULT_K = np.array([
    [606.0,   0.0, 320.0],
    [  0.0, 606.0, 240.0],
    [  0.0,   0.0,   1.0],
])

# Defaults per camera overview reale: camera a 1 m sopra il tavolo, punta verso il basso.
# pitch ≈ π fa puntare l'asse ottico verso -Z (verso il tavolo).
# Ajusta x/y per centrare e yaw per ruotare.
_DEFAULTS = {
    "x": 0.0, "y": 0.0, "z": 1.0,
    "roll": 0.0, "pitch": 3.14, "yaw": 0.0,
}

# Table grid z-level — overridden at runtime by the z_table slider
_Z_TABLE_DEFAULT = 0.0
_N_GRID   = 11
_GRID_XS  = np.linspace(-1.0, 1.0, _N_GRID)   # ogni quadrato = 20cm
_GRID_YS  = np.linspace(-1.0, 1.0, _N_GRID)

# Base-frame axis arrow length (m)
_AXIS_LEN = 0.30


# ── Geometry ─────────────────────────────────────────────────────────────────

def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ROS RPY → 3×3 rotation matrix R where p_base = R @ p_cam + t."""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1, 0,  0 ], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0,  1,   0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0,  0,  1]])
    return Rz @ Ry @ Rx


def _proj(p_base: np.ndarray, R: np.ndarray, t: np.ndarray,
          K: np.ndarray) -> tuple[float, float] | None:
    """Project one 3-D base-frame point to (u, v) image pixel, or None if behind camera."""
    p_cam = R.T @ (p_base - t)
    if p_cam[2] <= 0.01:
        return None
    return (
        K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2],
        K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2],
    )


# ── RealSense capture ─────────────────────────────────────────────────────────

def _capture_from_realsense(serial: str) -> np.ndarray | None:
    """Capture one RGB frame from the RealSense, save to _IMG_PATH and update K.
    Returns the image array, or None on failure."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[WARN] pyrealsense2 non disponibile — ricarico da disco")
        return None

    from PIL import Image as _PIL
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device(serial)

    started = False
    for (w, h) in [(1280, 720), (848, 480), (640, 480)]:
        try:
            config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 15)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16,  15)
            profile = pipeline.start(config)
            started = True
            break
        except Exception:
            config.disable_all_streams()

    if not started:
        print(f"[WARN] Impossibile aprire RealSense serial={serial}")
        return None

    try:
        import time
        time.sleep(1.5)
        align = rs.align(rs.stream.color)
        for _ in range(20):          # warmup: auto-esposizione
            try:
                pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                pass
        frames      = pipeline.wait_for_frames(timeout_ms=10000)
        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame:
            return None

        arr = np.asanyarray(color_frame.get_data())
        img = _PIL.fromarray(arr, "RGB")
        img.save(str(_IMG_PATH))

        # Aggiorna K con i valori reali
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        K = [[intr.fx, 0.0, intr.ppx],
             [0.0, intr.fy, intr.ppy],
             [0.0, 0.0,     1.0     ]]
        with open(_INFO_PATH, "w") as f:
            json.dump({"serial": serial, "K": K}, f, indent=2)

        # Salva anche depth
        if depth_frame:
            depth_arr = np.asanyarray(depth_frame.get_data())
            np.save(str(_REPO_ROOT / "data" / "depth_overview.npy"), depth_arr)

        print(f"[OK] Frame catturato {img.width}×{img.height} da serial {serial}")
        return arr
    except Exception as e:
        print(f"[WARN] Cattura fallita: {e}")
        return None
    finally:
        pipeline.stop()


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_K() -> np.ndarray:
    if _INFO_PATH.exists():
        with open(_INFO_PATH) as f:
            return np.array(json.load(f)["K"])
    print(f"[WARN] {_INFO_PATH.name} not found — using RealSense D435i defaults")
    return _DEFAULT_K.copy()


def _load_init() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded saved config from {_CFG_PATH.name}")
        return cfg
    return _DEFAULTS.copy()


def _load_img() -> np.ndarray:
    from PIL import Image
    if not _IMG_PATH.exists():
        print(f"[WARN] {_IMG_PATH.name} not found — using dark placeholder.")
        print(f"       Capture a frame first, then click 'Refresh Image'.")
        arr = np.full((480, 640, 3), 25, dtype=np.uint8)
        arr[::40, :] = 45
        arr[:, ::40] = 45
        return arr
    return np.array(Image.open(_IMG_PATH).convert("RGB"))


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="Overview camera positioning tool")
    ap.add_argument("--serial", default=None,
                    help="Serial number della overview RealSense (es. 242322071571). "
                         "Se fornito, Refresh cattura un nuovo frame direttamente dalla camera.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Prova a leggere il serial dal camera_info se non passato esplicitamente
    _serial = args.serial
    if _serial is None and _INFO_PATH.exists():
        try:
            with open(_INFO_PATH) as f:
                _serial = json.load(f).get("serial")
        except Exception:
            pass

    try:
        import matplotlib
        # TkAgg works on Ubuntu without extra config; fall back silently.
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.widgets import Slider, Button, TextBox

        # Workaround: matplotlib TextBox + Python 3.13 crash.
        # @_call_with_reparented_event (widgets.py:184) accesses event.inaxes but
        # ResizeEvent has no such attribute → AttributeError on every window resize.
        # _resize is connected directly to 'resize_event'; _click/_motion/_release
        # guard against receiving unexpected event types from the dispatcher.
        class _SafeTextBox(TextBox):
            # The @_call_with_reparented_event decorator on every TextBox event
            # handler accesses event.inaxes, but ResizeEvent has no such attribute
            # (Python 3.13 + matplotlib 3.8+ bug).  Override the two methods that
            # receive ResizeEvent and bypass the decorator entirely.
            def _resize(self, event):
                # Original body: self.stop_typing() — no decorator needed here.
                self.stop_typing()
            def _motion(self, event):
                if not hasattr(event, "inaxes"): return
                super()._motion(event)
    except ImportError:
        sys.exit("[ERROR] matplotlib not installed — pip install matplotlib")
    try:
        from PIL import Image as _PIL  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] Pillow not installed — pip install Pillow")

    K      = _load_K()
    cfg    = _load_init()
    img_arr = _load_img()

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    fig.patch.set_facecolor("#0f0f1a")
    fig.canvas.manager.set_window_title("Overview Camera Setup")

    ax = fig.add_axes([0.04, 0.30, 0.92, 0.67])
    ax.set_facecolor("#0f0f1a")
    ax.axis("off")

    im_obj = ax.imshow(img_arr, origin="upper", interpolation="bilinear")
    H0, W0 = img_arr.shape[:2]
    ax.set_xlim(-0.5, W0 - 0.5)
    ax.set_ylim(H0 - 0.5, -0.5)

    # Grid line objects — pre-allocated so updates are fast (no redraw)
    _ctr = len(_GRID_YS) // 2   # index of the centre line (x=0 or y=0)
    _h_lines = [ax.plot([], [], color="cyan", lw=0.8, alpha=0.50)[0]
                for _ in _GRID_YS]
    _v_lines = [ax.plot([], [], color="cyan", lw=0.8, alpha=0.50)[0]
                for _ in _GRID_XS]
    # Centre crosshair lines (x=0, y=0) drawn brighter and thicker
    _h_lines[_ctr].set_color("#00ffff"); _h_lines[_ctr].set_linewidth(1.6); _h_lines[_ctr].set_alpha(0.85)
    _v_lines[_ctr].set_color("#00ffff"); _v_lines[_ctr].set_linewidth(1.6); _v_lines[_ctr].set_alpha(0.85)

    # Axis arrows with arrowheads and labels (X=red, Y=green, Z=blue)
    _ax_anns = []
    _ax_lbls = []
    for c, lbl in zip(["#ff4444", "#44ee44", "#4499ff"], ["X", "Y", "Z"]):
        ann = ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2,
                            mutation_scale=16, shrinkA=0, shrinkB=0),
            zorder=8,
        )
        txt = ax.text(0, 0, lbl, color=c, fontsize=11, fontweight="bold",
                      ha="center", va="center", zorder=10,
                      bbox=dict(boxstyle="round,pad=0.15", fc="#000000",
                                alpha=0.55, ec="none"))
        ann.set_visible(False); txt.set_visible(False)
        _ax_anns.append(ann); _ax_lbls.append(txt)

    _origin_dot = ax.scatter([], [], s=70, c="white", zorder=9, marker="o")

    title_obj = ax.set_title("", color="#ccccdd", fontsize=9, pad=4, loc="left")

    # ── Sliders ───────────────────────────────────────────────────────────────
    _SC = "#1e1e38"   # slider background
    _AC = "#5555cc"   # slider fill colour

    _slider_defs = [
        # label,        vmin,   vmax,  init,               fmt
        ("x (m)",       -2.00,  2.00,  cfg["x"],           "%.3f"),
        ("y (m)",       -2.00,  2.00,  cfg["y"],           "%.3f"),
        ("z (m)",        0.00,  3.00,  cfg["z"],           "%.3f"),
        ("roll (rad)",  -3.15,  3.15,  cfg["roll"],        "%.3f"),
        ("pitch (rad)", -3.15,  3.15,  cfg["pitch"],       "%.3f"),
        ("yaw (rad)",   -3.15,  3.15,  cfg["yaw"],         "%.3f"),
    ]
    # ── Layout verticale (dal basso):
    #   y=0.025–0.075  → pulsanti Refresh / Run DINO / Save  (h=0.050)
    #   y=0.085–0.115  → TextBox "Object:"                   (h=0.030)
    #   y=0.125–0.150  → z_table slider                      (h=0.025)
    #   y=0.165–0.190  → slider row 2: roll/pitch/yaw        (h=0.025)
    #   y=0.205–0.230  → slider row 1: x/y/z                 (h=0.025)
    #   y=0.290+        → immagine
    sliders: list[Slider] = []
    for i, (lbl, lo, hi, val, fmt) in enumerate(_slider_defs):
        col, row = i % 3, i // 3
        ax_sl = fig.add_axes(
            [0.08 + col * 0.31, 0.205 - row * 0.040, 0.25, 0.025],
            facecolor=_SC,
        )
        sl = Slider(ax_sl, lbl, lo, hi, valinit=val, color=_AC, valfmt=fmt)
        sl.label.set_color("white")
        sl.valtext.set_color("white")
        sliders.append(sl)

    sl_x, sl_y, sl_z, sl_r, sl_p, sl_yaw = sliders

    # z_table slider
    ax_zt = fig.add_axes([0.35, 0.125, 0.30, 0.025], facecolor="#2a1a1a")
    sl_zt = Slider(ax_zt, "z_table (m)", -0.50, 1.00,
                   valinit=cfg.get("z_table", _Z_TABLE_DEFAULT),
                   color="#cc5555", valfmt="%.3f")
    sl_zt.label.set_color("#ffaaaa")
    sl_zt.valtext.set_color("#ffaaaa")

    # ── TextBox per query DINO ────────────────────────────────────────────────
    ax_txt = fig.add_axes([0.22, 0.085, 0.56, 0.030], facecolor="#111122")
    txt_obj = _SafeTextBox(ax_txt, "Object: ", initial="",
                           color="#111122", hovercolor="#1e1e44")
    txt_obj.label.set_color("#aaaacc")
    txt_obj.text_disp.set_color("#ffffff")

    # Pulsanti
    ax_bref = fig.add_axes([0.04, 0.025, 0.22, 0.050], facecolor="#1e1e38")
    ax_bsav = fig.add_axes([0.74, 0.025, 0.22, 0.050], facecolor="#102010")
    btn_ref = Button(ax_bref, "↻  Refresh Image", color="#1e1e38", hovercolor="#3a3a70")
    btn_sav = Button(ax_bsav, "✔  Save Config",   color="#102010", hovercolor="#1a501a")
    btn_ref.label.set_color("white");  btn_ref.label.set_fontsize(10)
    btn_sav.label.set_color("white");  btn_sav.label.set_fontsize(10)

    # ── DINO detection state ──────────────────────────────────────────────────
    _dets: list = []          # list of (box, label, score)
    _det_rects: list = []     # Rectangle patches
    _det_lbl_txts: list = []  # label texts
    _det_pos_txts: list = []  # 3D position texts (updated on every slider change)

    def _median_depth_from_box(depth_img, box, shrink=0.35, min_valid=5):
        x0,y0,x1,y1 = box
        dx,dy = (x1-x0)*shrink, (y1-y0)*shrink
        rx0=int(max(0,x0+dx)); ry0=int(max(0,y0+dy))
        rx1=int(min(depth_img.shape[1]-1,x1-dx))
        ry1=int(min(depth_img.shape[0]-1,y1-dy))
        if rx1<=rx0 or ry1<=ry0: return None
        patch = depth_img[ry0:ry1,rx0:rx1].astype(np.float32)
        valid = patch[patch>0]
        return float(np.median(valid))/1000.0 if valid.size>=min_valid else None

    def _reproject_dets():
        """Recompute 3D position for every stored detection using current params."""
        if not _dets:
            return
        x,y,z = sl_x.val,sl_y.val,sl_z.val
        roll,pitch,yaw = sl_r.val,sl_p.val,sl_yaw.val
        z_table = sl_zt.val
        R = _rpy_to_R(roll,pitch,yaw)
        t = np.array([x,y,z])
        K_inv = np.linalg.inv(K)

        depth = None
        depth_path = _REPO_ROOT/"data"/"depth_overview.npy"
        if depth_path.exists():
            try: depth = np.load(str(depth_path))
            except Exception: pass

        for i,(box,label,score) in enumerate(_dets):
            if i >= len(_det_pos_txts): break
            u = (box[0]+box[2])/2; v = (box[1]+box[3])/2
            pos_str = ""

            if depth is not None:
                z_cam = _median_depth_from_box(depth, box)
                if z_cam is not None and z_cam > 0.05:
                    p_cam  = K_inv @ np.array([u,v,1.0]) * z_cam
                    p_base = R @ p_cam + t
                    pos_str = f"({p_base[0]:.2f},{p_base[1]:.2f},{p_base[2]:.2f})m"

            if not pos_str:  # ray-plane fallback with z_table
                d_cam  = K_inv @ np.array([u,v,1.0])
                d_base = R @ d_cam; n=np.linalg.norm(d_base)
                if n>1e-9: d_base/=n
                if abs(d_base[2])>1e-9:
                    lam = (z_table - t[2])/d_base[2]
                    if lam>0:
                        p_base = t + lam*d_base
                        pos_str = f"({p_base[0]:.2f},{p_base[1]:.2f},{p_base[2]:.2f})m*"

            _det_pos_txts[i].set_text(pos_str or "?")

    # ── Update callback ───────────────────────────────────────────────────────

    def _update(_=None) -> None:
        x, y, z         = sl_x.val, sl_y.val, sl_z.val
        roll, pitch, yaw = sl_r.val, sl_p.val, sl_yaw.val
        z_table          = sl_zt.val
        R = _rpy_to_R(roll, pitch, yaw)
        t = np.array([x, y, z])

        # Grid rows (constant y)
        for i, yg in enumerate(_GRID_YS):
            uvs = [_proj(np.array([xg, yg, z_table]), R, t, K) for xg in _GRID_XS]
            us = [uv[0] for uv in uvs if uv is not None]
            vs = [uv[1] for uv in uvs if uv is not None]
            _h_lines[i].set_data(us, vs)

        # Grid columns (constant x)
        for i, xg in enumerate(_GRID_XS):
            uvs = [_proj(np.array([xg, yg, z_table]), R, t, K) for yg in _GRID_YS]
            us = [uv[0] for uv in uvs if uv is not None]
            vs = [uv[1] for uv in uvs if uv is not None]
            _v_lines[i].set_data(us, vs)

        # Base frame origin + axes with arrowheads and labels
        orig_uv = _proj(np.zeros(3), R, t, K)
        if orig_uv is not None:
            _origin_dot.set_offsets(np.array([orig_uv]))
            for j, tip_base in enumerate([
                np.array([_AXIS_LEN, 0.0, 0.0]),
                np.array([0.0, _AXIS_LEN, 0.0]),
                np.array([0.0, 0.0, _AXIS_LEN]),
            ]):
                tip_uv = _proj(tip_base, R, t, K)
                ann = _ax_anns[j]; lbl = _ax_lbls[j]
                if tip_uv is not None:
                    ann.xy = tip_uv          # arrowhead at tip
                    ann.set_position(orig_uv) # arrow tail at origin
                    ann.set_visible(True)
                    # label just past the arrowhead
                    dx = tip_uv[0] - orig_uv[0]
                    dy = tip_uv[1] - orig_uv[1]
                    n  = max(1.0, (dx**2 + dy**2) ** 0.5)
                    lbl.set_position((tip_uv[0] + 16*dx/n, tip_uv[1] + 16*dy/n))
                    lbl.set_visible(True)
                else:
                    ann.set_visible(False); lbl.set_visible(False)
        else:
            _origin_dot.set_offsets(np.empty((0, 2)))
            for ann, lbl in zip(_ax_anns, _ax_lbls):
                ann.set_visible(False); lbl.set_visible(False)

        # Debug: quanti punti della griglia sono dentro l'immagine?
        H_img, W_img = img_arr.shape[:2]
        all_uvs = [_proj(np.array([xg, yg, z_table]), R, t, K)
                   for xg in _GRID_XS for yg in _GRID_YS]
        in_bounds = [uv for uv in all_uvs
                     if uv is not None and 0 <= uv[0] < W_img and 0 <= uv[1] < H_img]
        n_visible = len(in_bounds)

        # Dove proietta il centro tavolo (0,0,0)?
        center_uv = _proj(np.zeros(3), R, t, K)
        if center_uv is not None:
            cx_str = f"centro→({center_uv[0]:.0f},{center_uv[1]:.0f})px"
        else:
            p_cam_z = float((R.T @ (np.zeros(3) - t))[2])
            cx_str  = f"centro dietro camera (z_cam={p_cam_z:.2f})"

        status = f"griglia: {n_visible}/{len(all_uvs)} punti visibili   {cx_str}"

        title_obj.set_text(
            f"  x={x:.3f} m   y={y:.3f} m   z={z:.3f} m   "
            f"roll={roll:.3f}   pitch={pitch:.3f}   yaw={yaw:.3f} rad   "
            f"z_table={z_table:.3f} m\n"
            f"  {status}"
        )
        _reproject_dets()
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(_update)
    sl_zt.on_changed(_update)

    # ── Keyboard shortcuts (fine-tuning without touching sliders) ─────────────
    # Step: 1 cm / 0.5° — hold Shift for 5× step
    def _on_key(event):
        if txt_obj.capturekeystrokes:
            return  # TextBox attivo: non muovere gli slider
        step_m   = 0.05  if event.key and "shift" in event.key else 0.01
        step_rad = 0.05  if event.key and "shift" in event.key else 0.01

        key = (event.key or "").replace("shift+", "")
        mapping = {
            "left":  (sl_y,  +step_m),   "right": (sl_y,  -step_m),
            "up":    (sl_x,  +step_m),   "down":  (sl_x,  -step_m),
            "pageup":   (sl_z,  +step_m),  "pagedown": (sl_z,  -step_m),
            "w": (sl_p,  +step_rad),  "s": (sl_p,  -step_rad),
            "a": (sl_yaw,+step_rad),  "d": (sl_yaw,-step_rad),
            "q": (sl_r,  +step_rad),  "e": (sl_r,  -step_rad),
            "r": (sl_zt, +step_m),    "f": (sl_zt, -step_m),
        }
        if key in mapping:
            sl, delta = mapping[key]
            sl.set_val(float(np.clip(sl.val + delta, sl.valmin, sl.valmax)))

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_refresh(_) -> None:
        nonlocal K, img_arr
        if _serial:
            print(f"[INFO] Cattura da RealSense serial={_serial}…")
            new_arr = _capture_from_realsense(_serial)
            if new_arr is None:
                print("[WARN] Cattura fallita — ricarico da disco")
                new_arr = _load_img()
            else:
                K = _load_K()
        else:
            print("[INFO] Nessun serial configurato — ricarico da disco")
            new_arr = _load_img()

        img_arr = new_arr
        H, W = img_arr.shape[:2]
        im_obj.set_data(img_arr)
        im_obj.set_extent([-0.5, W - 0.5, H - 0.5, -0.5])
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        _update()

    def _on_save(_) -> None:
        vals = {
            k: float(s.val)
            for k, s in zip(["x", "y", "z", "roll", "pitch", "yaw"], sliders)
        }
        vals["z_table"] = float(sl_zt.val)
        with open(_CFG_PATH, "w") as f:
            json.dump(vals, f, indent=2)

        # Genera anche overview_camera_pose.json (matrice 4×4 cam_to_base)
        # usato da validate_depth_pose.py e run_loop_host.py
        R = _rpy_to_R(vals["roll"], vals["pitch"], vals["yaw"])
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3]  = [vals["x"], vals["y"], vals["z"]]
        pose_path = _REPO_ROOT / "data" / "overview_camera_pose.json"
        with open(pose_path, "w") as f:
            json.dump({"cam_to_base": mat.tolist()}, f, indent=2)
        print(f"[OK] Pose matrix saved → {pose_path.name}")

        v = vals
        print(f"\n{'─'*62}")
        print(f"  [OK] Config saved → {_CFG_PATH}")
        print(f"{'─'*62}")
        print("\n  Launch command (values will also be read automatically):\n")
        print(f"    ros2 launch vlm_robot_planner_bringup real_robot.launch.py \\")
        print(f"      robot_ip:=<ROBOT_IP> \\")
        print(f"      overview_x:={v['x']:.4f} overview_y:={v['y']:.4f} "
              f"overview_z:={v['z']:.4f} \\")
        print(f"      overview_roll:={v['roll']:.4f} "
              f"overview_pitch:={v['pitch']:.4f} "
              f"overview_yaw:={v['yaw']:.4f}")
        print(f"\n  (real_robot.launch.py reads {_CFG_PATH.name} automatically)")
        print(f"{'─'*62}\n")

    btn_ref.on_clicked(_on_refresh)
    btn_sav.on_clicked(_on_save)

    # ── DINO button
    ax_bdino  = fig.add_axes([0.30, 0.025, 0.40, 0.050], facecolor="#1a2e1a")
    btn_dino  = Button(ax_bdino, "⬡  Run DINO",
                       color="#1a2e1a", hovercolor="#2a6a2a")
    btn_dino.label.set_color("white")
    btn_dino.label.set_fontsize(10)

    def _on_dino(_):
        query = txt_obj.text.strip()
        if not query:
            print("[DINO] Scrivi il nome dell'oggetto nel campo 'Object:' e riprova")
            return
        print(f"[DINO] Ricerca '{query}' nell'immagine overview…")

        # Rimuovi rilevamenti precedenti
        for p in _det_rects:     p.remove()
        for t in _det_lbl_txts:  t.remove()
        for t in _det_pos_txts:  t.remove()
        _det_rects.clear(); _det_lbl_txts.clear(); _det_pos_txts.clear(); _dets.clear()

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            from PIL import Image as _PIL

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "IDEA-Research/grounding-dino-tiny"
            proc  = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
            model.eval()

            pil_img = _PIL.fromarray(img_arr)
            text    = query.lower().rstrip(".") + " ."
            inputs  = proc(images=pil_img, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            H_i,W_i = img_arr.shape[:2]
            res = proc.post_process_grounded_object_detection(
                outputs, inputs.input_ids,
                threshold=0.25, text_threshold=0.20,
                target_sizes=[(H_i,W_i)]
            )[0]

            for box,score,label in zip(res["boxes"],res["scores"],res["labels"]):
                box = [float(v) for v in box.tolist()]
                _dets.append((box, label, float(score)))

                rect = mpatches.Rectangle(
                    (box[0],box[1]), box[2]-box[0], box[3]-box[1],
                    fill=False, edgecolor="#00ff88", linewidth=2, zorder=6)
                ax.add_patch(rect); _det_rects.append(rect)

                lt = ax.text(box[0]+2, box[1]-5,
                             f"{label} {float(score):.2f}",
                             color="#00ff88", fontsize=8, fontweight="bold",
                             va="bottom", zorder=7,
                             bbox=dict(boxstyle="round,pad=0.1",fc="black",
                                       alpha=0.55,ec="none"))
                _det_lbl_txts.append(lt)

                pt = ax.text((box[0]+box[2])/2, (box[1]+box[3])/2, "…",
                             color="white", fontsize=8, ha="center", va="center",
                             zorder=7,
                             bbox=dict(boxstyle="round,pad=0.2",fc="#000055",
                                       alpha=0.75,ec="none"))
                _det_pos_txts.append(pt)

            print(f"[DINO] {len(_dets)} oggetto/i trovato/i")
            _reproject_dets()
            fig.canvas.draw_idle()

        except Exception as exc:
            print(f"[DINO ERROR] {exc}")

    btn_dino.on_clicked(_on_dino)

    # ── Initial render + help text ────────────────────────────────────────────
    _update()

    print(f"\n{'─'*62}")
    print("  Overview Camera Setup Tool")
    print(f"{'─'*62}")
    print(f"  Image  : {_IMG_PATH}")
    print(f"  Config : {_CFG_PATH}")
    if _serial:
        print(f"  Serial : {_serial}  ← Refresh cattura direttamente dalla camera")
    else:
        print(f"  Serial : non configurato  ← Refresh ricarica solo da disco")
        print(f"           Usa --serial <SN> per cattura live")
    print()
    print("  Steps:")
    print("  1. Posiziona la camera fisicamente.")
    print("  2. Clicca 'Refresh Image' per catturare un nuovo frame.")
    print("  3. Regola gli slider finché la griglia ciana copre il tavolo.")
    print("  4. Clicca 'Save Config'.")
    print()
    print("  Tip: inizia da z (altezza camera) e yaw (rotazione sinistra/destra),")
    print("       poi affina x/y per centrare la griglia sul tavolo.")
    print()
    print("  Keyboard shortcuts (clicca sull'immagine prima):")
    print("    ←/→/↑/↓      — sposta y/y/x/x  (Shift = 5× step)")
    print("    PgUp/PgDn    — alza/abbassa z (camera)")
    print("    W / S        — pitch +/-")
    print("    A / D        — yaw +/-")
    print("    Q / E        — roll +/-")
    print("    R / F        — z_table +/-  (alza/abbassa piano tavolo)")
    print(f"{'─'*62}\n")

    plt.show()


if __name__ == "__main__":
    main()
