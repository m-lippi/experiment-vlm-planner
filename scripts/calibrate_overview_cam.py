#!/usr/bin/env python3
"""
calibrate_overview_cam.py — Calibration tool for the overview camera.

Workflow:
  1. Capture current overview camera image from the simulation
  2. Project the workspace area onto the image (show what the robot can reach)
  3. Print current camera pose (from world file)
  4. Save annotated image for visual inspection
  5. If view is not ideal, edit the world file and re-run

Usage:
    # With sim running:
    source .venv/bin/activate
    python3 scripts/calibrate_overview_cam.py --world office

    # After capturing manually:
    python3 scripts/calibrate_overview_cam.py --world office --image data/scene_overview.png

For real robot deployment: position the external camera so the entire
manipulation workspace is visible, then record the mount position/orientation
and update the world file.
"""

from __future__ import annotations
import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _read_pose_from_world(world_name: str):
    import xml.etree.ElementTree as ET
    world_path = (_REPO_ROOT / "ros2_ws/src/vlm_robot_planner_bringup/worlds" /
                  f"{world_name}.world")
    if not world_path.exists():
        return None
    tree = ET.parse(str(world_path))
    for model in tree.iter("model"):
        if model.get("name") == "overview_camera":
            pe = model.find("pose")
            if pe is not None and pe.text:
                vals = list(map(float, pe.text.split()))
                if len(vals) == 6:
                    return vals
    return None


def _compute_cam_matrix(pose, K_override=None, W=1280, H=960, fov=1.047):
    """Given SDF pose [x,y,z,r,p,y], return (K, cam_to_base, R_world_to_cam, t_world_to_cam).
    K_override: use actual K from camera_info topic instead of computing from FOV."""
    px, py, pz, roll, pitch, yaw = pose
    ROBOT_BASE = np.array([0.20, 0.0, 0.770])

    if K_override is not None:
        K = K_override
    else:
        fx = fy = W / (2.0 * math.tan(fov / 2.0))
        K = np.array([[fx, 0, W/2.0], [0, fy, H/2.0], [0, 0, 1.0]])

    def rpy_matrix(r, p, y):
        Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
        Ry = np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
        Rz = np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
        return Rz @ Ry @ Rx

    R_W_G = rpy_matrix(roll, pitch, yaw)
    R_C_G = np.array([[0,-1,0],[0,0,-1],[1,0,0]])  # Gazebo +Y=left → OpenCV -X
    R_world_to_cam = R_C_G @ R_W_G.T
    t_world_to_cam = -R_world_to_cam @ np.array([px, py, pz])

    cam_pos_world = np.array([px, py, pz])
    cam_pos_base  = cam_pos_world - ROBOT_BASE
    cam_to_base   = np.eye(4)
    cam_to_base[:3,:3] = R_world_to_cam.T
    cam_to_base[:3, 3] = cam_pos_base

    return K, cam_to_base, R_world_to_cam, t_world_to_cam


def _project_point(p_world, K, R, t):
    """Project a world 3D point to pixel (u, v). Returns None if behind camera."""
    p_cam = R @ np.array(p_world) + t
    if p_cam[2] <= 0.01:
        return None
    u = K[0,0] * p_cam[0] / p_cam[2] + K[0,2]
    v = K[1,1] * p_cam[1] / p_cam[2] + K[1,2]
    return int(u), int(v)


def _capture_overview(container="vlm_ros2"):
    """Capture overview + wrist camera images using _capture_scene.py."""
    print("[INFO] Capturing images from simulation...")
    r = subprocess.run(
        ["docker", "exec", container, "bash", "-c",
         "source /opt/ros/humble/setup.bash && "
         "source /workspace/ros2_ws/install/setup.bash && "
         "python3 /workspace/scripts/_capture_scene.py"],
        capture_output=True, timeout=15
    )
    out = (r.stdout + r.stderr).decode().strip()
    for line in out.splitlines():
        if line.strip():
            print(f"  {line}")
    return r.returncode == 0


def _read_object_positions_from_world(world_name: str) -> dict:
    """
    Read all model/include object positions directly from the world SDF file.
    Returns {name: [x, y, z]} in world frame.
    """
    import xml.etree.ElementTree as ET
    world_path = (_REPO_ROOT / "ros2_ws/src/vlm_robot_planner_bringup/worlds" /
                  f"{world_name}.world")
    if not world_path.exists():
        return {}
    tree = ET.parse(str(world_path))
    positions = {}
    # Objects to SHOW: only desk/workspace objects, not furniture/decor
    _SHOW = {
        # office
        "pen","pen2","eraser","keyboard","mouse","coffee_cup","notebook",
        "paperweight","monitor_1","laptop",
        # workshop
        "hammer","monkey_wrench","cordless_drill","coke_can",
        "gauge_ball","target_tray","hex_bolt",
        # kitchen
        "bottle","glass","glass2","cup","can","plate","mug","spoon","knife",
        "cutting_board","target_tray",
    }
    _SKIP = {"room","floor","robot_pedestal","table","workbench","side_table",
             "overview_camera","trash_can","plant","office_chair","bookshelf",
             "sofa","coffee_table","wall_shelf","tool_cabinet","safety_cone",
             "toolbox","cabinet","shelf_b"}

    # Inline <model> elements
    for model in tree.iter("model"):
        name = model.get("name", "")
        if not name or name in _SKIP:
            continue
        if _SHOW and name not in _SHOW:
            continue
        pose_el = model.find("pose")
        if pose_el is not None and pose_el.text:
            v = list(map(float, pose_el.text.split()))
            if len(v) >= 3:
                positions[name] = v[:3]
    # <include> elements
    for inc in tree.iter("include"):
        name_el = inc.find("name")
        uri_el  = inc.find("uri")
        pose_el = inc.find("pose")
        name = (name_el.text if name_el is not None else
                (uri_el.text.split("/")[-1] if uri_el is not None else None))
        if not name or name in ("sun", "ground_plane"):
            continue
        if name in _SKIP or (_SHOW and name not in _SHOW):
            continue
        if pose_el is not None and pose_el.text:
            v = list(map(float, pose_el.text.split()))
            if len(v) >= 3:
                positions[name] = v[:3]
    return positions


def _annotate_workspace(image, K, R, t, world_name, cam_pose=None):
    """Draw workspace boundary and key points on the image."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    W, H = image.width, image.height

    # ── Workspace areas per scene (world frame) ────────────────────────────
    # Each scene has one or more rectangular surfaces: [(corners_list, color, label)]
    WORKSPACE_AREAS = {
        "kitchen": [
            (  # Counter: model center (0.85,0), 200x80cm — x:[0.45,1.25] y:[-1.00,+1.00]
                [[0.45,-1.00,0.770],[0.45,1.00,0.770],[1.25,1.00,0.770],[1.25,-1.00,0.770]],
                (0,220,220), "counter"
            ),
        ],
        "office": [
            (  # Main desk: center (0.85,0), 100×60cm
                [[0.35,-0.30,0.770],[0.35,0.30,0.770],[1.35,0.30,0.770],[1.35,-0.30,0.770]],
                (0,220,220), "desk"
            ),
            (  # Side table: center (1.10,-0.70), 49×72cm (legs at ±0.245, ±0.36)
                [[0.855,-1.06,0.755],[0.855,-0.34,0.755],[1.345,-0.34,0.755],[1.345,-1.06,0.755]],
                (0,220,120), "side_table"
            ),
        ],
        "workshop": [
            (
                [[-0.10,-0.375,0.770],[-0.10,0.375,0.770],[1.10,0.375,0.770],[1.10,-0.375,0.770]],
                (0,220,220), "workbench"
            ),
        ],
    }

    # Robot base (static, same for all scenes)
    robot_base_world = [0.20, 0.00, 0.77]

    objects_world = _read_object_positions_from_world(world_name)
    print(f"  Objects from world file: {list(objects_world.keys())}")

    # Draw workspace perimeter. PIL clips lines at image boundaries automatically;
    # only reject points behind the camera (depth <= 0).
    areas = WORKSPACE_AREAS.get(world_name, [])
    for corners_list, color, label in areas:
        corners_px = [_project_point(c, K, R, t) for c in corners_list]
        corners_px = [p for p in corners_px if p is not None]
        if len(corners_px) >= 3:
            for i in range(len(corners_px)):
                draw.line([corners_px[i], corners_px[(i+1) % len(corners_px)]],
                          fill=color, width=3)
            # Place label at the first corner that falls within the image frame.
            in_frame = [p for p in corners_px if 0 <= p[0] < W and 0 <= p[1] < H]
            lbl_pt = in_frame[0] if in_frame else corners_px[0]
            draw.text((lbl_pt[0]+4, lbl_pt[1]+4), label, fill=color)

    # Draw robot base (red dot)
    rb_px = _project_point(robot_base_world, K, R, t)
    if rb_px and 3 <= rb_px[0] < W-3 and 3 <= rb_px[1] < H-3:
        r = 6
        draw.ellipse([rb_px[0]-r, rb_px[1]-r, rb_px[0]+r, rb_px[1]+r], fill=(255,60,60))
        draw.text((rb_px[0]+7, rb_px[1]-6), "robot", fill=(255,60,60))

    # Draw key object positions (yellow dots + labels)
    for name, world_pos in objects_world.items():
        px = _project_point(world_pos, K, R, t)
        if px is None or not (3 <= px[0] < W-3 and 3 <= px[1] < H-3):
            continue
        r = 5
        draw.ellipse([px[0]-r, px[1]-r, px[0]+r, px[1]+r], fill=(255, 220, 0))
        draw.text((px[0]+6, px[1]-6), name[:10], fill=(255, 220, 0))

    # Draw image center crosshair (grey)
    cx, cy = W//2, H//2
    draw.line([cx-20, cy, cx+20, cy], fill=(128,128,128), width=1)
    draw.line([cx, cy-20, cx, cy+20], fill=(128,128,128), width=1)

    # ── Orientation compass — top-right corner, small ────────────────────
    ox, oy = W - 85, 80   # top-right
    ARROW = 28
    ref_world = np.array([0.85, 0.0, 0.77])   # desk center (always in front of cam)

    def _dir_to_img(wx, wy, wz=0.0, scale=0.5):
        tip = ref_world + np.array([wx, wy, wz]) * scale
        p_r = R @ ref_world + t;  p_t = R @ tip + t
        if p_r[2] <= 0.05 or p_t[2] <= 0.05:
            return None
        ur = K[0,0]*p_r[0]/p_r[2]+K[0,2]; vr = K[1,1]*p_r[1]/p_r[2]+K[1,2]
        ut = K[0,0]*p_t[0]/p_t[2]+K[0,2]; vt = K[1,1]*p_t[1]/p_t[2]+K[1,2]
        du, dv = ut-ur, vt-vr
        mag = math.sqrt(du**2+dv**2)+1e-9
        return (int(du/mag*ARROW), int(dv/mag*ARROW))

    draw.rectangle([ox-40, oy-68, W-3, oy+ARROW+18], fill=(25,25,25))
    draw.text((ox-37, oy-66), "BUSSOLA", fill=(180,180,180))
    for label, col, wx, wy in [
        ("X+ desk", (255,100,100),  1,  0),
        ("X- robot",(255,160, 80), -1,  0),
        ("Y+ sx",   (100,255,100),  0,  1),
        ("Y- dx",   (100,200,255),  0, -1),
    ]:
        dv2 = _dir_to_img(wx, wy)
        if dv2:
            draw.line([ox, oy, ox+dv2[0], oy+dv2[1]], fill=col, width=2)
            draw.text((ox+dv2[0]+3, oy+dv2[1]-6), label, fill=col)

    return image


def main():
    ap = argparse.ArgumentParser(description="Overview camera calibration tool")
    ap.add_argument("--world",     default="office", help="Scene name")
    ap.add_argument("--image",     default=None,    help="Use existing image instead of capturing")
    ap.add_argument("--container", default="vlm_ros2")
    ap.add_argument("--out",       default=None,    help="Output path (default: data/calibration_<world>.png)")
    args = ap.parse_args()

    from PIL import Image

    # ── 1. Read camera pose from world file ───────────────────────────────────
    pose = _read_pose_from_world(args.world)
    if pose is None:
        print(f"[WARN] overview_camera not found in {args.world}.world — using default")
        pose = [1.0, 0.7, 1.5, 0.0, 0.68, -2.19]

    print(f"\n{'='*55}")
    print(f"  Overview Camera Calibration — {args.world}")
    print(f"{'='*55}")
    print(f"  Current pose in {args.world}.world:")
    print(f"    position : x={pose[0]:.3f}  y={pose[1]:.3f}  z={pose[2]:.3f}")
    print(f"    rotation : roll={pose[3]:.3f}  pitch={pose[4]:.3f}  yaw={pose[5]:.3f}")

    # ── 2. Capture image (before loading K, so overview_camera_info.json is
    #        updated with the current camera resolution first) ─────────────────
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            sys.exit(f"[ERROR] Image not found: {img_path}")
        img = Image.open(str(img_path)).convert("RGB")
        print(f"\n  Using image: {img_path}")
    else:
        if not _capture_overview(args.container):
            print("[WARN] Could not capture from sim — using last saved overview if available")
        img_path = _REPO_ROOT / "data" / "scene_overview.png"
        if not img_path.exists():
            sys.exit("[ERROR] No overview image available. Run with sim active.")
        img = Image.open(str(img_path)).convert("RGB")

    # ── 3. Load K after capture (overview_camera_info.json is now up to date) ─
    import json
    K_real = None
    ov_info = _REPO_ROOT / "data" / "overview_camera_info.json"
    if ov_info.exists():
        with open(str(ov_info)) as f:
            K_real = np.array(json.load(f)["K"])
        print(f"  Intrinsics (from camera_info): fx={K_real[0,0]:.1f}  cx={K_real[0,2]:.1f}  ({img.width}x{img.height})")
    else:
        print(f"  Intrinsics (computed from FOV {img.width}x{img.height})")

    K, cam_to_base, R, t = _compute_cam_matrix(
        pose, K_override=K_real, W=img.width, H=img.height)
    print(f"  Using fx={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

    # ── 4. Annotate workspace ─────────────────────────────────────────────────
    img = _annotate_workspace(img, K, R, t, args.world, cam_pose=pose)

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_path = Path(args.out) if args.out else \
               _REPO_ROOT / "data" / f"calibration_{args.world}.png"
    img.save(str(out_path), optimize=True)

    print(f"\n  Annotated image saved: {out_path}")
    print(f"  Cyan rectangle = table workspace boundary")
    print(f"  Yellow dots    = key object positions")
    print(f"  Red dot        = robot base")
    print(f"\n  To adjust camera position:")
    print(f"  1. Edit {args.world}.world: find <model name=\"overview_camera\"> → <pose>")
    print(f"  2. Change position/orientation")
    print(f"  3. Re-run: python3 scripts/calibrate_overview_cam.py --world {args.world}")
    print(f"\n  Goal: workspace boundary visible + objects clearly identifiable")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
