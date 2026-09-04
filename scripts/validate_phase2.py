#!/usr/bin/env python3
"""
validate_phase2.py — Phase 2 hybrid approach validation (no ROS pipeline).

Runs on the host:
  1. VLM → plan with bbox
  2. VLM-bbox → 3D pose for pick target (ray-plane intersection)
  3. GroundingDINO → 3D pose for place location
  4. Gazebo oracle → ground-truth poses for error comparison
  5. Annotated debug image + text report

Usage:
    source .venv/bin/activate
    python3 scripts/validate_phase2.py \\
        --task "pick the hammer and place it on the yellow tray" \\
        [--image data/scene.png]  [--no-oracle]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_ROBOT_BASE_WORLD = np.array([0.20, 0.0, 0.770])   # panda_link0 in world frame

# Hardcoded oracle poses (panda_link0 = world - robot_base).
# Fallback when Gazebo is unreachable.
_ORACLE_HARDCODED = {
    # ── workshop ──────────────────────────────────────────────────────────────
    "hammer":         {"x": 0.550, "y":  0.017, "z": 0.025},
    "monkey_wrench":  {"x": 0.350, "y": -0.134, "z": 0.025},
    "coke_can":       {"x": 0.520, "y": -0.260, "z": 0.000},
    "gauge_ball":     {"x": 0.648, "y":  0.220, "z": 0.040},
    "target_tray":    {"x": 0.670, "y": -0.172, "z": 0.010},
    "cordless_drill": {"x": 0.373, "y":  0.220, "z":-0.135},
    "hex_bolt":       {"x": 0.450, "y": -0.050, "z": 0.011},
    # ── office ────────────────────────────────────────────────────────────────
    "coffee_cup":     {"x": 0.400, "y":  0.173, "z": 0.065},
    "glass":          {"x": 0.400, "y":  0.173, "z": 0.065},  # alias coffee_cup
    "notebook":       {"x": 0.420, "y": -0.060, "z": 0.013},
    "pen":            {"x": 0.540, "y":  0.270, "z": 0.008},
    "eraser":         {"x": 0.420, "y": -0.140, "z": 0.008},
    "paperweight":    {"x": 0.680, "y":  0.187, "z": 0.025},
    "keyboard":       {"x": 0.795, "y":  0.004, "z": 0.000},
    "mouse":          {"x": 1.100, "y": -0.090, "z": 0.000},
    "monitor_1":      {"x": 1.020, "y":  0.150, "z": 0.000},
}


# ── geometry helpers ──────────────────────────────────────────────────────────

def ray_plane_intersect(u, v, K, cam_to_base, z_plane=0.025):
    """Back-project pixel (u,v) to a 3D point on the plane z=z_plane in panda_link0."""
    K_inv = np.linalg.inv(K)
    d_cam = K_inv @ np.array([u, v, 1.0])
    base_to_cam = np.linalg.inv(cam_to_base)
    R = base_to_cam[:3, :3]
    t = base_to_cam[:3, 3]
    cam_origin_in_base = cam_to_base[:3, 3]
    d_base = cam_to_base[:3, :3] @ d_cam
    d_base /= np.linalg.norm(d_base)
    if abs(d_base[2]) < 1e-9:
        return None
    t_ray = (z_plane - cam_origin_in_base[2]) / d_base[2]
    if t_ray < 0:
        return None
    pt = cam_origin_in_base + t_ray * d_base
    return {"x": float(pt[0]), "y": float(pt[1]), "z": float(z_plane)}


def error_m(est, gt):
    """Euclidean distance (m) between estimated and ground-truth 3D positions."""
    return math.sqrt((est["x"]-gt["x"])**2 + (est["y"]-gt["y"])**2)


# ── oracle query via Gazebo ───────────────────────────────────────────────────

def query_oracle(objects: list[str], container="vlm_ros2") -> dict:
    """Query Gazebo /gazebo/model_states for ground-truth poses (world frame)."""
    script = """
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates
import json, time
rclpy.init()
node = Node('_val_oracle')
msg = None
def cb(m):
    global msg; msg = m
sub = node.create_subscription(ModelStates, '/gazebo/model_states', cb, 1)
deadline = time.time() + 3.0
while time.time() < deadline and msg is None:
    rclpy.spin_once(node, timeout_sec=0.1)
if msg:
    targets = set(""" + repr(objects) + """)
    out = {}
    for name, pose in zip(msg.name, msg.pose):
        if name in targets:
            p = pose.position
            out[name] = {"x": p.x, "y": p.y, "z": p.z}
    print(json.dumps(out))
node.destroy_node(); rclpy.shutdown()
"""
    try:
        r = subprocess.run(
            ["docker", "exec", container, "bash", "-c",
             "source /opt/ros/humble/setup.bash && "
             "source /workspace/ros2_ws/install/setup.bash && "
             f"python3 -c {repr(script)}"],
            capture_output=True, timeout=10
        )
        if r.returncode == 0:
            return json.loads(r.stdout.decode().strip())
    except Exception as e:
        print(f"[WARN] Oracle query failed: {e}")
    return {}


# ── VLM inference ─────────────────────────────────────────────────────────────

def run_vlm(task, images):
    from vlm.planner import VLMPlanner
    vlm = VLMPlanner()
    vlm.load()
    plan = vlm.plan_next_step(task, images, [])
    return plan


# ── GroundingDINO ─────────────────────────────────────────────────────────────

_DINO_PROC  = None
_DINO_MODEL = None

def _load_dino():
    global _DINO_PROC, _DINO_MODEL
    if _DINO_MODEL is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch
        _DINO_PROC  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        _DINO_MODEL = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-tiny").cuda().eval()


def grounding_dino_pose(query, image, K, cam_to_base, threshold=0.15,
                        roi_bbox=None):
    """
    Run GroundingDINO on image (or ROI crop if roi_bbox provided).
    roi_bbox = [x1,y1,x2,y2] in original image coords — cascade mode.
    Coordinates are mapped back to original image before 3D projection.
    """
    import torch
    _load_dino()
    W, H = image.width, image.height

    # Cascade: crop to VLM ROI so DINO only sees the relevant region
    offset_x, offset_y = 0, 0
    work_img = image
    if roi_bbox is not None and len(roi_bbox) == 4:
        rx1,ry1,rx2,ry2 = roi_bbox
        rx1=max(0,min(int(rx1),W-1)); rx2=max(0,min(int(rx2),W))
        ry1=max(0,min(int(ry1),H-1)); ry2=max(0,min(int(ry2),H))
        if rx2 > rx1 and ry2 > ry1:
            work_img = image.crop((rx1, ry1, rx2, ry2))
            offset_x, offset_y = rx1, ry1

    wW, wH = work_img.width, work_img.height
    inp = _DINO_PROC(images=work_img, text=query + " .",
                     return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = _DINO_MODEL(**inp)
    res = _DINO_PROC.post_process_grounded_object_detection(
        out, inp.input_ids,
        threshold=threshold, text_threshold=threshold*0.8,
        target_sizes=[(wH, wW)]
    )[0]
    if len(res["boxes"]) == 0:
        return None, None
    best_idx = int(res["scores"].argmax())
    box = res["boxes"][best_idx].tolist()
    score = float(res["scores"][best_idx])
    # Map back to original image coords
    box_orig = [box[0]+offset_x, box[1]+offset_y,
                box[2]+offset_x, box[3]+offset_y]
    u = (box_orig[0]+box_orig[2])/2
    v = (box_orig[1]+box_orig[3])/2
    pose = ray_plane_intersect(u, v, K, cam_to_base)
    return pose, {"box": box_orig, "score": score, "center": (u, v),
                  "cropped": roi_bbox is not None}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick the hammer and place it on the yellow tray")
    ap.add_argument("--image", default="data/scene.png")
    ap.add_argument("--no-oracle", action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"  PHASE 2 VALIDATION — ibrido VLM-bbox + GroundingDINO")
    print(f"  Task: {args.task}")
    print(f"{'='*60}\n")

    # ── Load image and camera calibration ────────────────────────────────────
    img_path = _REPO_ROOT / args.image
    if not img_path.exists():
        sys.exit(f"[ERROR] {img_path} non trovato. Esegui prima _capture_scene.py")

    image = Image.open(img_path).convert("RGB")
    W, H  = image.size
    print(f"[INFO] Immagine: {W}×{H}  ({img_path.name})")

    data_dir = _REPO_ROOT / "data"
    with open(data_dir / "camera_info.json") as f:
        ci = json.load(f)
    K = np.array(ci["K"])
    with open(data_dir / "camera_pose.json") as f:
        cam_to_base = np.array(json.load(f)["cam_to_base"])
    print(f"[INFO] K: fx={K[0,0]:.1f}  cam_to_base loaded")

    # ── 1. VLM inference ─────────────────────────────────────────────────────
    print("\n[1/4] VLM inference...")
    plan = run_vlm(args.task, [image])
    if not plan.steps:
        sys.exit("[ERROR] VLM non ha generato step")

    step = plan.steps[0]
    print(f"  Prossimo step: {step.primitive}({step.args})")
    vlm_bbox   = step.args.get("bbox")
    pick_target = step.args.get("object") or step.args.get("target")
    place_loc   = step.args.get("location")
    loc_bbox    = step.args.get("location_bbox")

    # ── 2. VLM-bbox → 3D pose (pick target) ──────────────────────────────────
    vlm_pose_est = None
    vlm_detection = None
    if vlm_bbox and pick_target:
        print(f"\n[2/4] VLM-bbox → 3D pose per '{pick_target}'")
        x1,y1,x2,y2 = vlm_bbox
        x1=max(0,min(x1,W)); x2=max(0,min(x2,W))
        y1=max(0,min(y1,H)); y2=max(0,min(y2,H))
        u_c = (x1+x2)/2.0; v_c = (y1+y2)/2.0
        print(f"  VLM bbox=[{x1},{y1},{x2},{y2}]  center=({u_c:.0f},{v_c:.0f})")
        vlm_pose_est = ray_plane_intersect(u_c, v_c, K, cam_to_base)
        vlm_detection = {"box": [x1,y1,x2,y2], "center": (u_c,v_c)}
        if vlm_pose_est:
            print(f"  → 3D: ({vlm_pose_est['x']:.3f}, {vlm_pose_est['y']:.3f}, {vlm_pose_est['z']:.3f}) panda_link0")
    else:
        print(f"\n[2/4] Nessuna VLM bbox per pick — step corrente è '{step.primitive}'")

    # ── 3. Confronto 3 approcci ───────────────────────────────────────────────
    target_for_dino = (place_loc or pick_target or "").replace("_", " ")
    print(f"\n[3/4] Confronto approcci su '{target_for_dino}'")

    # A) GroundingDINO standalone (full image)
    dino_pose_full, dino_det_full = grounding_dino_pose(
        target_for_dino, image, K, cam_to_base, threshold=0.12)
    if dino_pose_full:
        sc = dino_det_full['score']
        print(f"  A) DINO standalone:     ({dino_pose_full['x']:.3f},{dino_pose_full['y']:.3f}) — score={sc:.3f}")
    else:
        print(f"  A) DINO standalone:     non rilevato")

    # B) GroundingDINO in ROI (cascade: crop sul VLM bbox)
    roi = vlm_bbox
    dino_pose_roi, dino_det_roi = None, None
    if roi:
        # Skip if clipped ROI is degenerate (zero width or height)
        rx1,ry1,rx2,ry2 = roi
        rx1=max(0,min(rx1,W)); rx2=max(0,min(rx2,W))
        ry1=max(0,min(ry1,H)); ry2=max(0,min(ry2,H))
        if rx2 - rx1 > 10 and ry2 - ry1 > 10:
            dino_pose_roi, dino_det_roi = grounding_dino_pose(
                target_for_dino, image, K, cam_to_base, threshold=0.10, roi_bbox=roi)
        else:
            print(f"  B) DINO in VLM-ROI:    [skip — bbox clippato degenere {rx2-rx1}×{ry2-ry1}px]")
        if dino_pose_roi:
            sc = dino_det_roi['score']
            print(f"  B) DINO in VLM-ROI:    ({dino_pose_roi['x']:.3f},{dino_pose_roi['y']:.3f}) — score={sc:.3f}  [HYBRID CASCADE]")
        else:
            print(f"  B) DINO in VLM-ROI:    non rilevato nel ROI")

    # C) Place location using VLM location_bbox or DINO
    dino_loc_pose, dino_loc_det = None, None
    if step.primitive == "place" and place_loc:
        loc_q = place_loc.replace("_", " ")
        if loc_bbox:
            x1,y1,x2,y2=loc_bbox; u_c=(x1+x2)/2.0; v_c=(y1+y2)/2.0
            dino_loc_pose = ray_plane_intersect(u_c, v_c, K, cam_to_base)
            dino_loc_det  = {"box": loc_bbox, "center": (u_c,v_c), "score": None}
            print(f"  C) Place VLM-loc bbox: ({dino_loc_pose['x']:.3f},{dino_loc_pose['y']:.3f}) — '{loc_q}'")
        else:
            dino_loc_pose, dino_loc_det = grounding_dino_pose(loc_q, image, K, cam_to_base)
            if dino_loc_pose:
                print(f"  C) Place DINO:         ({dino_loc_pose['x']:.3f},{dino_loc_pose['y']:.3f}) — '{loc_q}' score={dino_loc_det['score']:.3f}")

    # ── 4. Oracle ground-truth ────────────────────────────────────────────────
    oracle_poses = {}
    if not args.no_oracle:
        objs_to_query = [o for o in [pick_target, place_loc] if o]
        print(f"\n[4/4] Oracle → ground-truth per {objs_to_query}")
        # Try Gazebo first, fallback to hardcoded
        world_poses = query_oracle(objs_to_query)
        for name, wp in world_poses.items():
            lp = {"x": wp["x"]-_ROBOT_BASE_WORLD[0],
                  "y": wp["y"]-_ROBOT_BASE_WORLD[1],
                  "z": wp["z"]-_ROBOT_BASE_WORLD[2]}
            oracle_poses[name] = lp
            print(f"  {name:20s}: ({lp['x']:.3f},{lp['y']:.3f}) [Gazebo]")
        # Fallback to hardcoded for missing objects
        for name in objs_to_query:
            if name not in oracle_poses and name in _ORACLE_HARDCODED:
                oracle_poses[name] = _ORACLE_HARDCODED[name]
                print(f"  {name:20s}: ({_ORACLE_HARDCODED[name]['x']:.3f},{_ORACLE_HARDCODED[name]['y']:.3f}) [hardcoded]")

    # ── Validation results ────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  RISULTATI VALIDAZIONE")
    print(f"{'─'*60}")

    gt_pick  = oracle_poses.get(pick_target or "")
    gt_place = oracle_poses.get(place_loc or "")

    def err_str(pose, gt):
        if pose and gt:
            return f"  errore XY: {error_m(pose,gt)*100:.1f} cm"
        return "  (no oracle)"

    obj_label = f"'{target_for_dino}'"
    if vlm_pose_est:
        print(f"  A) VLM-bbox:        ({vlm_pose_est['x']:.3f},{vlm_pose_est['y']:.3f}){err_str(vlm_pose_est, gt_pick or gt_place)}")
    if dino_pose_full:
        print(f"  B) DINO standalone: ({dino_pose_full['x']:.3f},{dino_pose_full['y']:.3f}){err_str(dino_pose_full, gt_pick or gt_place)}")
    if dino_pose_roi:
        print(f"  C) DINO in ROI:     ({dino_pose_roi['x']:.3f},{dino_pose_roi['y']:.3f}){err_str(dino_pose_roi, gt_pick or gt_place)}  ← HYBRID")
    if gt_pick:
        print(f"  ORACLE pick:        ({gt_pick['x']:.3f},{gt_pick['y']:.3f})  z={gt_pick['z']:.3f}")
    if dino_loc_pose and gt_place:
        print(f"  Place loc DINO:     ({dino_loc_pose['x']:.3f},{dino_loc_pose['y']:.3f}){err_str(dino_loc_pose, gt_place)}")
    if gt_place:
        print(f"  ORACLE place:       ({gt_place['x']:.3f},{gt_place['y']:.3f})")

    # ── Debug image ───────────────────────────────────────────────────────────
    # Line width scales with image resolution:
    #   640×480  (sim wrist cam) → lw=2
    #   3024×4032 (phone)        → lw=10
    lw = max(2, min(image.width, image.height) // 200)

    dbg  = image.copy()
    draw = ImageDraw.Draw(dbg)

    # VLM bbox (lime)
    if vlm_detection:
        b = vlm_detection["box"]
        draw.rectangle([b[0],b[1],b[2],b[3]], outline="lime", width=lw)
        gt = oracle_poses.get(pick_target or "")
        lbl = f"VLM {pick_target}"
        if vlm_pose_est and gt:
            lbl += f" err={error_m(vlm_pose_est,gt)*100:.0f}cm"
        draw.text((b[0], max(0,b[1]-14)), lbl, fill="lime")

    # DINO standalone (red — high visibility)
    if dino_det_full:
        b = dino_det_full["box"]
        draw.rectangle([b[0],b[1],b[2],b[3]], outline="red", width=lw)
        sc = dino_det_full.get("score", 0)
        gt = oracle_poses.get(pick_target or "")
        lbl = f"DINO {sc:.2f}"
        if dino_pose_full and gt:
            lbl += f" err={error_m(dino_pose_full,gt)*100:.0f}cm"
        draw.text((b[0], max(0,b[3]+2)), lbl, fill="red")

    # DINO in ROI cascade (cyan, thicker stroke)
    if dino_det_roi:
        b = dino_det_roi["box"]
        draw.rectangle([b[0],b[1],b[2],b[3]], outline="cyan", width=lw + 1)
        sc = dino_det_roi.get("score", 0)
        gt = oracle_poses.get(pick_target or "")
        lbl = f"HYBRID {sc:.2f}"
        if dino_pose_roi and gt:
            lbl += f" err={error_m(dino_pose_roi,gt)*100:.0f}cm"
        draw.text((b[0], max(0,b[1]-14)), lbl, fill="cyan")

    # Place location (magenta)
    if dino_loc_det:
        b = dino_loc_det["box"]
        draw.rectangle([b[0],b[1],b[2],b[3]], outline="magenta", width=lw)
        draw.text((b[0], max(0,b[1]-14)), f"place:{place_loc}", fill="magenta")

    # Output filename derived from task
    safe = args.task.lower().replace(" ", "_")[:40]
    out_path = _REPO_ROOT / "data" / f"val_{safe}.png"
    dbg.save(str(out_path))
    print(f"\n  Debug image: {out_path}")
    print(f"  Lime = VLM-bbox  |  Rosso = DINO standalone  |  Ciano = DINO@ROI hybrid")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
