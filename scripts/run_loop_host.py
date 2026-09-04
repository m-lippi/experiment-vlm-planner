#!/usr/bin/env python3
"""
run_loop_host.py — Closed-loop task execution (HOST side).

Implements the closed-loop architecture:
  [scan] -> [capture] -> [VLM next step] -> [inject] -> [wait complete] -> repeat

Each iteration:
  1. Pre-scan: move arm to scan pose via _pre_scan.py (wrist camera view)
  2. Capture: take image from wrist camera via _capture_scene.py
  3. VLM: plan_next_step(task, image, completed_steps) -> single action
  4. Ground: GroundingDINO -> correct object names and 3D poses
  5. Inject: send single-step plan to orchestrator
  6. Wait: _wait_step_complete.py -> get completion signal
  7. If complete: break; else: add step to completed_steps, repeat

Sim-to-real note: the same loop works on the real robot — the only difference
is that Gazebo oracle is replaced by RealSense depth in the PerceptionModule.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _docker(container: str, use_sudo: bool) -> list[str]:
    return (["sudo", "docker"] if use_sudo else ["docker"]) + ["exec", "-i", container]


def _run_in_container(args, bash_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        _docker(args.container, args.sudo_docker) + ["bash", "-c", bash_cmd],
        capture_output=True, timeout=timeout,
    )


def _pre_scan(args) -> bool:
    """Move arm to scan pose before capture."""
    print("[LOOP] Pre-scan: moving arm to scan pose...")
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_pre_scan.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=40)
    output = r.stdout.decode().strip()
    if output:
        for line in output.splitlines():
            print(f"       {line}")
    if r.returncode == 0:
        print("[OK]   Scan pose reached.")
        return True
    err = r.stderr.decode().strip()
    if err:
        print(f"[WARN] Pre-scan: {err}")
    print("[WARN] Pre-scan failed — continuing anyway (will use fallback camera)")
    return False


def _capture(args) -> Path | None:
    """Capture image from wrist camera."""
    scene_path = _REPO_ROOT / "data" / "scene.png"
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_capture_scene.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=15)
    output = r.stdout.decode().strip()
    if r.returncode != 0:
        print(f"[FAIL] Capture failed: {r.stderr.decode().strip()}")
        return None
    # Print capture output (includes which camera topic was used)
    for line in output.splitlines():
        print(f"       {line}")
    return scene_path


def _get_gazebo_models(args) -> dict:
    """Get Gazebo scene objects and their positions."""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_get_model_states.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=10)
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.decode().strip()).get("models", {})
        except Exception:
            pass
    return {}


def _read_overview_pose_from_world(world_name: str):
    """
    Parse the world SDF file and extract the overview_camera model pose.
    Returns (x, y, z, roll, pitch, yaw) or None if not found.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path
    world_path = (Path(__file__).resolve().parent.parent /
                  "ros2_ws/src/vlm_robot_planner_bringup/worlds" /
                  f"{world_name}.world")
    if not world_path.exists():
        return None
    try:
        tree = ET.parse(str(world_path))
        for model in tree.iter("model"):
            if model.get("name") == "overview_camera":
                pose_el = model.find("pose")
                if pose_el is not None and pose_el.text:
                    vals = list(map(float, pose_el.text.split()))
                    if len(vals) == 6:
                        return vals   # [x, y, z, roll, pitch, yaw]
    except Exception:
        pass
    return None


def _get_scene_objects(world_name: str) -> list[str]:
    """
    Read all named objects from the world SDF and return their names.
    Used to inject the exact object names into the VLM prompt so the model
    generates correct oracle-compatible names regardless of task wording.
    Skips structural models (walls, floor, pedestal, cameras, furniture).
    """
    import xml.etree.ElementTree as ET
    _SKIP = {
        "sun", "ground_plane", "floor", "room", "wall_back", "wall_left",
        "wall_right", "robot_pedestal", "overview_camera", "ceiling_lamp",
        "wall_cabinet_l", "wall_cabinet_r", "fridge", "stove", "kitchen_table",
        "chair_north", "chair_south", "chair_east", "counter", "workbench",
        "desk", "side_table", "laptop_stand", "monitor_stand", "cabinet",
        "shelf_b", "bookshelf", "sofa", "plant", "trash_can", "office_chair",
        "coffee_table", "safety_cone",
    }
    world_path = (Path(__file__).resolve().parent.parent /
                  "ros2_ws/src/vlm_robot_planner_bringup/worlds" /
                  f"{world_name}.world")
    if not world_path.exists():
        return []
    try:
        tree = ET.parse(str(world_path))
        names = []
        for model in tree.iter("model"):
            n = model.get("name", "")
            if n and n not in _SKIP:
                names.append(n)
        for inc in tree.iter("include"):
            name_el = inc.find("name")
            n = name_el.text.strip() if name_el is not None and name_el.text else ""
            if n and n not in _SKIP:
                names.append(n)
        return sorted(set(names))
    except Exception:
        return []


def _get_overview_cam_data(world_name: str = "office"):
    """
    Compute K and cam_to_base for the OVERVIEW camera.
    Reads pose from the world SDF file — update the world file to recalibrate.
    The overview camera is STATIC so this is computed once at startup.
    Returns (K, cam_to_base) or (None, None) on error.
    """
    try:
        import numpy as np, math

        # ── Read pose from world file ─────────────────────────────────────────
        pose = _read_overview_pose_from_world(world_name)
        if pose is None:
            # Fallback: hardcoded default
            pose = [1.0, 0.7, 1.5, 0.0, 0.68, -2.19]
            print(f"[WARN] overview_camera not found in {world_name}.world — using default")
        else:
            print(f"[INFO] Overview cam pose from {world_name}.world: "
                  f"pos=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}) "
                  f"rpy=({pose[3]:.2f},{pose[4]:.2f},{pose[5]:.2f})")

        _POS  = np.array(pose[:3])
        _RPY  = tuple(pose[3:])
        _W, _H, _FOV = 640, 480, 1.047
        _ROBOT_BASE = np.array([0.20, 0.0, 0.770])

        # ── Intrinsics — prefer actual K from camera_info topic ───────────────
        from pathlib import Path as _Path
        ov_info_path = _Path(__file__).resolve().parent.parent / "data" / "overview_camera_info.json"
        if ov_info_path.exists():
            import json as _json
            with open(str(ov_info_path)) as _f:
                K = np.array(_json.load(_f)["K"])
            print(f"[INFO] Overview K from camera_info: fx={K[0,0]:.1f}")
        else:
            fx = fy = _W / (2.0 * math.tan(_FOV / 2.0))
            K = np.array([[fx, 0, _W/2.0], [0, fy, _H/2.0], [0, 0, 1.0]])
            print(f"[INFO] Overview K computed from FOV: fx={K[0,0]:.1f} (run calibration first)")

        # ── Rotation: SDF RPY → world-to-OpenCV-camera ───────────────────────
        def _rpy(r, p, y):
            Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
            Ry = np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
            Rz = np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
            return Rz @ Ry @ Rx

        R_W_G = _rpy(*_RPY)       # world → Gazebo link (cols = cam axes in world)
        # Gazebo cam: +X=optical; OpenCV cam: +Z=optical
        R_C_G = np.array([[0,-1,0],[0,0,-1],[1,0,0]])  # Gazebo +Y=left → OpenCV -X
        R_world_to_cam = R_C_G @ R_W_G.T   # world → OpenCV camera

        # ── cam_to_base (camera → panda_link0) ───────────────────────────────
        # Real robot: prefer the TF-based pose saved by _capture_scene.py.
        # Simulation: fall back to computing it from the world SDF pose.
        from pathlib import Path as _Path2
        import json as _json2
        _ov_pose_path = _Path2(__file__).resolve().parent.parent / "data" / "overview_camera_pose.json"
        if _ov_pose_path.exists():
            with open(str(_ov_pose_path)) as _f2:
                cam_to_base = np.array(_json2.load(_f2)["cam_to_base"])
            print("[INFO] Overview cam_to_base from overview_camera_pose.json (TF-based)")
            return K, cam_to_base

        # Simulation fallback: compute from SDF pose
        R_cam_to_world = R_world_to_cam.T
        cam_pos_in_base = _POS - _ROBOT_BASE

        cam_to_base = np.eye(4)
        cam_to_base[:3, :3] = R_cam_to_world
        cam_to_base[:3,  3] = cam_pos_in_base

        return K, cam_to_base
    except Exception as _e:
        print(f"[WARN] overview cam calibration failed: {_e}")
        return None, None


def _annotate_handled_objects(
    image,
    placed_at: dict,
    data_dir: str,
    info_file: str = "camera_info.json",
    pose_file: str = "camera_pose.json",
) -> "PIL.Image.Image":
    """
    Annotate the image with already-handled objects using two non-obstructive elements:
    1. A small cross (+) at the projected 3D position of each placed object
    2. A text legend box in the top-left corner listing all handled objects

    The small cross minimally occludes the scene; the text box is fully readable
    by the VLM. This approach avoids covering nearby unhandled objects.
    """
    if not placed_at:
        return image

    import json
    import numpy as np
    from PIL import ImageDraw
    from pathlib import Path

    ci_path = Path(data_dir) / info_file
    cp_path = Path(data_dir) / pose_file

    K, cam_to_base, R, t = None, None, None, None
    if ci_path.exists() and cp_path.exists():
        try:
            with open(ci_path) as f:
                K = np.array(json.load(f)["K"])
            with open(cp_path) as f:
                cam_to_base = np.array(json.load(f)["cam_to_base"])
            base_to_cam = np.linalg.inv(cam_to_base)
            R = base_to_cam[:3, :3]
            t = base_to_cam[:3, 3]
        except Exception:
            pass

    dbg  = image.copy()
    draw = ImageDraw.Draw(dbg)
    W, H = dbg.width, dbg.height
    CS   = max(5, min(W, H) // 80)   # cross arm length (tiny)

    # ── 1. Small cross at each projected object position ─────────────────────
    if R is not None:
        for i, (name, (px, py)) in enumerate(placed_at.items(), 1):
            p_cam = R @ np.array([px, py, 0.025]) + t
            if p_cam[2] <= 0.05:
                continue
            u = int(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
            v = int(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])
            if not (CS <= u < W - CS and CS <= v < H - CS):
                continue
            draw.line([u - CS, v, u + CS, v], fill=(0, 220, 0), width=2)
            draw.line([u, v - CS, u, v + CS], fill=(0, 220, 0), width=2)
            draw.text((u + CS + 1, v - CS), str(i), fill=(0, 220, 0))

    # ── 2. Text legend box in top-left corner ────────────────────────────────
    PAD   = 6
    LH    = 14   # line height
    lines = ["DONE:"] + [f" {i}. {n}" for i, n in enumerate(placed_at, 1)]
    box_w = max(len(l) for l in lines) * 7 + PAD * 2
    box_h = len(lines) * LH + PAD * 2
    draw.rectangle([2, 2, box_w, box_h], fill=(0, 60, 0))
    draw.rectangle([2, 2, box_w, box_h], outline=(0, 200, 0), width=1)
    for i, line in enumerate(lines):
        color = (180, 255, 180) if i == 0 else (220, 255, 220)
        draw.text((PAD + 2, PAD + i * LH), line, fill=color)

    return dbg


def _publish_perception_pose(
    args, object_name: str, x: float, y: float, z: float,
    height_m: float | None = None,
) -> bool:
    """Publish a perception-estimated pose to /perception/object_pose.

    height_m: estimated object height in metres (from _estimate_object_height).
              Encoded in orientation.z; None → 0.0 → orchestrator uses fallback.
    """
    height_arg = f" --height_m {height_m:.4f}" if height_m is not None else ""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_publish_perception_pose.py "
        f"--object {object_name} --x {x:.6f} --y {y:.6f} --z {z:.6f}{height_arg}"
    )
    r = _run_in_container(args, bash_cmd, timeout=10)
    for line in r.stdout.decode().strip().splitlines():
        print(f"       {line}")
    return r.returncode == 0


def _estimate_object_height(
    detection: dict | None,
    obj_xyz: tuple,
    K,
    ctb,
) -> float | None:
    """Estimate object height from DINO bbox using the pinhole model.

    H ≈ bbox_height_px × dist(camera, object) / fy

    Works for both overview camera and wrist camera (ctb changes with arm pose).
    Phase 2 improvement: replace with depth-channel measurement from RealSense
    (sample depth at multiple rows of the bbox → more accurate, handles tilt).

    Returns None if inputs are unavailable or the estimate is out of range.
    """
    if detection is None or K is None or ctb is None:
        return None
    try:
        bbox_h_px = detection["box"][3] - detection["box"][1]   # y2 - y1
        if bbox_h_px < 5:   # < 5 pixels → unreliable
            return None
        cam_origin = ctb[:3, 3]                                 # camera in panda_link0
        dist = float(_np.linalg.norm(_np.array(obj_xyz) - cam_origin))
        fy = float(K[1, 1])
        h = bbox_h_px * dist / fy
        return h if 0.02 < h < 0.60 else None   # sanity: 2 cm – 60 cm
    except Exception:
        return None




def _wait_step_complete(args, timeout: int = 60, min_seq: int = 0) -> dict:
    """Wait for step completion signal from orchestrator.

    min_seq: ignore step_complete messages with seq < this value, preventing
    stale TRANSIENT_LOCAL (latched) messages from previous steps being accepted
    as the result of the current step.
    """
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_wait_step_complete.py --timeout {timeout} --min-seq {min_seq}"
    )
    r = _run_in_container(args, bash_cmd, timeout=timeout + 5)
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.decode().strip())
        except Exception:
            pass
    return {"success": False, "task_complete": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop task execution")
    parser.add_argument("--task",       required=True)
    parser.add_argument("--max-steps",  type=int, default=10)
    parser.add_argument("--container",  default="vlm_ros2")
    parser.add_argument("--sudo-docker", action="store_true")
    parser.add_argument("--world",      default="office",
                        help="Active Gazebo world (reads overview cam pose from world file)")
    args = parser.parse_args()

    print("[LOOP] Loading VLM (Qwen3-VL-8B-Instruct)…")
    from vlm.planner import VLMPlanner
    from vlm.perception import PerceptionModule
    from PIL import Image as PilImage

    vlm       = VLMPlanner()
    vlm.load()
    perception = PerceptionModule()
    perception.load()
    print("[OK]   VLM + PerceptionModule loaded.\n")

    completed_steps: list[str] = []
    docker_cmd = _docker(args.container, args.sudo_docker)

    # Replanning on failure state
    _current_plan       = None   # cached full VLMPlan (remaining steps)
    _last_failed_step   = None   # step that caused last replan
    _replan_count       = 0      # how many times we've replanned

    import datetime
    _ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _world_tag = getattr(args, "world", "unknown")
    _task_tag  = args.task[:30].replace(" ", "_").replace("/", "-")
    _RUN_DIR   = _REPO_ROOT / "data" / "runs" / f"{_ts}_{_world_tag}_{_task_tag}"
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(_RUN_DIR / "run_info.txt"), "w") as _rf:
        _rf.write(f"timestamp: {_ts}\n")
        _rf.write(f"world:     {_world_tag}\n")
        _rf.write(f"task:      {args.task}\n")
    print(f"[LOOP] Run dir: {_RUN_DIR.relative_to(_REPO_ROOT)}")

    # Overview camera calibration — computed once from world file (camera is static)
    _OV_K, _OV_CTB = _get_overview_cam_data(args.world)
    if _OV_K is not None:
        print("[OK]   Overview camera calibration: ready (static from SDF)")
    else:
        print("[WARN] Overview camera calibration failed — using wrist cam for VLM")

    # Tracks destinations of placed objects; new DINO detections within
    # EXCL_RADIUS of a recorded position are skipped to prevent re-picking.
    _placed_at: dict[str, tuple[float, float]] = {}
    _last_dino_est: dict[str, tuple[float, float]] = {}  # last DINO estimate per name
    _EXCL_RADIUS = 0.10  # 10cm — objects within this radius are treated as identical
    # Tracks the last dispatch_seq; _wait_step_complete uses min_seq=_last_seq+1
    # to ignore stale TRANSIENT_LOCAL (latched) messages from previous steps.
    _last_seq: int = -1
    # Persists domain enrichments across iterations. The VLM enriches the domain
    # only when it first encounters a novel action; subsequent iterations omit it.
    # generate_problem needs the accumulated definitions to infer the PDDL goal.
    _accumulated_da: dict = {}

    for iteration in range(args.max_steps):
        print(f"\n{'─'*60}")
        print(f"  ITERAZIONE {iteration+1} / {args.max_steps}")
        print(f"  Completati: {completed_steps or ['(nessuno)']}")
        print(f"{'─'*60}")

        # 1. Pre-scan — only when gripper is empty.
        # If holding an object, scan pose movement prevents place from succeeding
        # (MoveIt2 can't plan from scan+held_object to pre-place position).
        last_pick  = max((i for i,s in enumerate(completed_steps) if s.startswith("pick")),  default=-1)
        last_place = max((i for i,s in enumerate(completed_steps) if s.startswith("place") or s.startswith("stack")), default=-1)
        holding = last_pick > last_place

        if not holding:
            _pre_scan(args)
            time.sleep(1.0)
        else:
            print("[LOOP] Holding object — skip scan pose, capture from current arm position")

        # 2. Capture
        image_path = _capture(args)
        if image_path is None:
            print("[FAIL] No image — aborting loop")
            break
        image = PilImage.open(image_path).convert("RGB")

        # Wrist snapshot saved here temporarily; moved into the iter subfolder at debug-save time
        iter_path = _RUN_DIR / f"iter_{iteration+1:02d}_wrist.png"
        image.save(str(iter_path))
        print(f"[LOOP] Snapshot: {iter_path.name}")

        # Load overview camera image for VLM (fixed reference, better perspective)
        _ov_path = _REPO_ROOT / "data" / "scene_overview.png"
        if _ov_path.exists() and _OV_K is not None:
            image_vlm = PilImage.open(str(_ov_path)).convert("RGB")
        else:
            image_vlm = image   # fallback to wrist cam
        _using_overview = (_ov_path.exists() and _OV_K is not None)

        # Persist last scan-pose image + calibration for place location detection.
        # When arm is holding an object, the camera view is distorted by the arm.
        # Using the last FREE scan gives better geometry for location detection.
        if not holding:
            import shutil
            _data = _REPO_ROOT / "data"
            for fname in ("scene.png", "camera_info.json", "camera_pose.json"):
                src = _data / fname
                if src.exists():
                    shutil.copy2(str(src), str(_data / f"last_scan_{fname}"))
            print("[LOOP] Last scan saved (arm free → used for place location detection)")

        # 3. Get Gazebo models — filter scene infrastructure (never pick/place targets)
        _INFRA = frozenset({
            'floor', 'room', 'ground_plane', 'sun', 'robot_pedestal',
            'overview_camera', 'table', 'workbench',
        })
        gazebo_poses = {k: v for k, v in _get_gazebo_models(args).items()
                        if k not in _INFRA}
        gazebo_models = list(gazebo_poses.keys())
        print(f"[LOOP] Scene objects: {gazebo_models}")

        # 4. VLM: plan next single step (measure inference time)
        # Strip arrow notation from completed_steps before passing to VLM
        # so it doesn't echo back "cube->red_cup" and cause double-arrows.
        vlm_context = [s.split("->")[-1].rstrip(")") + ")" if "->" in s else s
                       for s in completed_steps
                       if not s.startswith("skip_")]

        # Annotate image for VLM with already-handled objects.
        # Use overview camera image (fixed reference) for stable annotations.
        # Wrist cam (image) continues to be used for DINO localization.
        _data_dir_annot = str(_REPO_ROOT / "data")
        if _using_overview and _OV_CTB is not None:
            # Annotate on overview image using overview cam_to_base (fixed reference)
            import json as _jjson
            _ov_info = {"K": _OV_K.tolist()}
            _ov_pose = {"cam_to_base": _OV_CTB.tolist()}
            _tmp_info = _REPO_ROOT / "data" / "_tmp_ov_info.json"
            _tmp_pose = _REPO_ROOT / "data" / "_tmp_ov_pose.json"
            with open(str(_tmp_info), "w") as _f: _jjson.dump(_ov_info, _f)
            with open(str(_tmp_pose), "w") as _f: _jjson.dump(_ov_pose, _f)
            image_for_vlm = _annotate_handled_objects(
                image_vlm, _placed_at, str(_REPO_ROOT / "data"),
                info_file="_tmp_ov_info.json", pose_file="_tmp_ov_pose.json")
        else:
            image_for_vlm = _annotate_handled_objects(image, _placed_at, _data_dir_annot)

        if _placed_at:
            annot_path = _RUN_DIR / f"iter_{iteration+1:02d}_annotated.png"
            image_for_vlm.save(str(annot_path))
            src = "overview" if _using_overview else "wrist"
            print(f"[LOOP] Annotated image [{src}]: {annot_path.name} "
                  f"({len(_placed_at)} marker(s): {list(_placed_at.keys())})")

        # Each iteration: VLM receives the current image and generates the full remaining plan.
        # If the scene state changed unexpectedly, the plan will differ from the previous iteration.
        t_vlm = time.time()
        action_label = "REPLAN" if _last_failed_step else ("PLAN" if not vlm_context else "VERIFY+PLAN")
        print(f"[LOOP] VLM {action_label} (piano completo rimanente) per: '{args.task}'")

        _prev_plan_steps = [f"{s.primitive}({s.args})" for s in (_current_plan.steps if _current_plan else [])]
        _prev_current_plan = _current_plan   # saved to inherit grasp_mode if VLM drops it

        # Pass both overview (annotated) + wrist camera to the VLM.
        # overview → global scene state with handled-object markers
        # wrist    → close-up of current arm position / grip
        _vlm_images = [image_for_vlm]
        if image is not None and image is not image_for_vlm:
            _vlm_images.append(image)

        _current_plan = vlm.plan_remaining(
            args.task, _vlm_images, vlm_context,
            failed_step=_last_failed_step,
            prior_enrichment=_accumulated_da if _accumulated_da else None,
        )
        _last_failed_step = None
        vlm_time = time.time() - t_vlm

        # Preserve grasp_mode from previous plan if VLM omitted it during replanning.
        # The VLM sometimes drops grasp_mode=side when updating the plan because it
        # doesn't remember it specified it earlier. We inherit it so the correct
        # physical grasp is preserved across iterations.
        if _prev_current_plan and _current_plan.steps:
            _prev_picks_by_obj = {
                s.args.get("object", ""): s
                for s in _prev_current_plan.steps
                if s.primitive == "pick" and s.args.get("object")
            }
            for _s in _current_plan.steps:
                if _s.primitive == "pick" and "grasp_mode" not in _s.args:
                    _obj = _s.args.get("object", "")
                    _prev_pick = _prev_picks_by_obj.get(_obj)
                    if _prev_pick and "grasp_mode" in _prev_pick.args:
                        _s.args = dict(_s.args)
                        _s.args["grasp_mode"] = _prev_pick.args["grasp_mode"]
                        print(f"[LOOP] Inherited grasp_mode='{_s.args['grasp_mode']}' "
                              f"for pick('{_obj}') from previous plan")
        print(f"[LOOP] VLM inference    : {vlm_time:.1f}s")

        if _current_plan.steps:
            _new_steps = [f"{s.primitive}({s.args})" for s in _current_plan.steps]
            # Detect if VLM changed the plan (state verification detected a change)
            if _prev_plan_steps and _new_steps != _prev_plan_steps:
                print(f"[LOOP] ⚡ Piano AGGIORNATO dalla VLM (stato cambiato):")
            else:
                print(f"[LOOP] Piano confermato ({len(_current_plan.steps)} passi rimanenti):")
            for _i, _s in enumerate(_current_plan.steps, 1):
                _args_str = ", ".join(f"{k}={v}" for k, v in _s.args.items())
                print(f"         {_i}. {_s.primitive}({_args_str})")

        # Extract only the NEXT step for execution this iteration
        from copy import deepcopy as _dc
        if _current_plan.steps:
            plan = _dc(_current_plan)
            plan.steps = [_current_plan.steps[0]]
        else:
            plan = _current_plan   # complete=True

        # ── VLM plan summary ──────────────────────────────────────────────
        print(f"[LOOP] Domain template  : {plan.domain_template}")

        # Show domain enrichment if the VLM added anything beyond the base template
        da = plan.domain_additions
        enriched = (da.get("new_predicates") or da.get("new_actions") or
                    da.get("new_types") or da.get("modified_preconditions"))
        if enriched:
            # Persist enrichment: merge new_actions/predicates into accumulator so
            # subsequent iterations can use them even when VLM says "no enrichment".
            for key in ("new_types", "new_predicates", "new_actions", "modified_preconditions"):
                if da.get(key):
                    existing = _accumulated_da.get(key, [])
                    existing_names = {
                        a.get("name") for a in existing
                        if isinstance(a, dict) and "name" in a
                    }
                    for item in da[key]:
                        name = item.get("name") if isinstance(item, dict) else None
                        if name not in existing_names:
                            existing.append(item)
                    _accumulated_da[key] = existing
            print(f"[LOOP] ⚡ DOMAIN ENRICHMENT:")
            if da.get("new_types"):
                print(f"         new_types      : {da['new_types']}")
            if da.get("new_predicates"):
                print(f"         new_predicates : {da['new_predicates']}")
            if da.get("new_actions"):
                for a in da["new_actions"]:
                    print(f"         new_action     : {a.get('name')} "
                          f"({a.get('parameters','')}) "
                          f"pre={a.get('precondition','')} "
                          f"eff={a.get('effect','')}")
            if da.get("modified_preconditions"):
                print(f"         mod_precond    : {da['modified_preconditions']}")
        else:
            print(f"[LOOP] Domain enrichment: none (base template sufficient)")

        if not plan.steps:
            # Save final state image (no bboxes — task is complete)
            try:
                final_path = _RUN_DIR / f"loop_iter_{iteration+1:02d}.png"
                image.save(str(final_path))
                print(f"[LOOP] Snapshot finale: {final_path.name}")
            except Exception:
                pass
            print("\n[LOOP] ✅  Task completato secondo VLM!")
            break

        step0 = plan.steps[0]
        print(f"[LOOP] Prossimo step: {step0.primitive}({step0.args})")

        # Prevent phantom pick: skip pick if already holding an object
        if step0.primitive == "pick":
            last_pick  = max((i for i, s in enumerate(completed_steps) if s.startswith("pick")),  default=-1)
            last_place = max((i for i, s in enumerate(completed_steps) if s.startswith("place") or s.startswith("stack")), default=-1)
            if last_pick > last_place:
                print(f"[WARN] Phantom pick detected (already holding) — skipping")
                completed_steps.append(f"skip_pick({step0.args.get('object','?')})")
                continue

        # Prevent phantom place: skip place if the gripper should be empty
        # (no pick in completed_steps since last place/gripper_open)
        if step0.primitive == "place":
            last_pick = max(
                (i for i, s in enumerate(completed_steps) if s.startswith("pick")),
                default=-1
            )
            last_place = max(
                (i for i, s in enumerate(completed_steps) if s.startswith("place")),
                default=-1
            )
            if last_pick < last_place:
                # Count consecutive skip_place to detect stuck loop
                consecutive_skips = sum(
                    1 for s in reversed(completed_steps)
                    if s.startswith("skip_place")
                    ) if completed_steps else 0
                if consecutive_skips >= 2:
                    print(f"[LOOP] ✅ {consecutive_skips} phantom places consecutivi → "
                          "task considerato completato (oggetto già depositato)")
                    break
                print(f"[WARN] Phantom place detected (no pick since last place) — skipping")
                completed_steps.append(f"skip_place({step0.args.get('object','?')})")
                continue

        # Phase 2: VLM object names are passed directly to DINO as queries.
        # ground_names() was a Phase 1 step for oracle name matching and is no longer called.
        from copy import deepcopy
        plan_grounded = deepcopy(plan)

        # 5c. DINO pose estimation — primary source is the overview camera.
        # The overview D435i operates within its optimal depth range (0.8-1.5 m)
        # with stable extrinsic calibration and a full view of the workspace.
        # The wrist camera is not used as the DINO source: its ~0.3 m depth is
        # borderline for D435i and hand-eye calibration is less reliable.
        # Falls back to wrist camera if overview is unavailable.
        _data_dir = str(_REPO_ROOT / "data")
        step0 = plan_grounded.steps[0] if plan_grounded.steps else None
        if step0:
            try:
                import numpy as _np
                from vlm.perception import PerceptionModule

                # Select primary camera source for DINO
                if _using_overview and _OV_K is not None and _OV_CTB is not None:
                    det_img_all   = image_vlm   # scene_overview.png — full workspace view
                    det_K_all     = _OV_K
                    det_ctb_all   = _OV_CTB
                    src_label_all = "overview"
                else:
                    # Fallback: wrist camera
                    _cam = PerceptionModule.load_camera_data(_data_dir)
                    if _cam:
                        det_img_all, det_K_all, det_ctb_all = image, _cam[0], _cam[1]
                        src_label_all = "wrist"
                    else:
                        det_img_all = det_K_all = det_ctb_all = None
                        src_label_all = "none"

                # Collect object names referenced in the current step
                names_to_estimate = {}
                for _key in ("target", "object", "location", "container"):
                    _n = step0.args.get(_key, "")
                    if _n and _n not in _INFRA and _n not in names_to_estimate:
                        names_to_estimate[_n] = _key

                # If currently holding an object, skip DINO for it regardless of step.
                # The held object is inside the gripper and not visible in the overview.
                # Identify it from the last pick(...) in completed_steps.
                if holding:
                    _held = next(
                        (s[5:].split(",")[0].rstrip(")")
                         for s in reversed(completed_steps)
                         if s.startswith("pick(")),
                        None,
                    )
                    if _held and _held in names_to_estimate:
                        names_to_estimate.pop(_held)
                        print(f"[LOOP] Holding '{_held}' — skip DINO (in gripper, not visible)")

                _dino_detections = []
                for name, name_key in names_to_estimate.items():
                    # SIM-ONLY: fuzzy name match against Gazebo model names.
                    # In simulation, Gazebo provides ground-truth poses for all models.
                    # If the VLM name matches a Gazebo model name (substring), use that
                    # pose directly — avoids DINO misidentifying large surfaces as locations
                    # (e.g. "tray" detected as the whole counter → wrong 3D point).
                    # On the real robot gazebo_poses is empty → this block never executes
                    # and DINO is always used for all object localisation.
                    _gz_name_match = None
                    if gazebo_poses and name not in gazebo_poses:
                        _name_lower = name.lower().replace("_", "")
                        _candidates = []
                        for _gz in gazebo_poses:
                            _gz_lower = _gz.lower().replace("_", "")
                            if _name_lower in _gz_lower or _gz_lower in _name_lower:
                                _candidates.append(_gz)
                        if len(_candidates) == 1:
                            _gz_name_match = _candidates[0]
                            _gp = gazebo_poses[_gz_name_match]
                            _rbase = _np.array([0.20, 0.0])
                            _resolved_pose = {
                                "x": _gp["x"] - _rbase[0],
                                "y": _gp["y"] - _rbase[1],
                                "z": 0.025,
                            }
                            print(f"[LOOP] NameMatch: '{name}' → '{_gz_name_match}' "
                                  f"(Gazebo pose, no DINO needed)")
                            _resolved_xy = (_resolved_pose["x"], _resolved_pose["y"])
                            _last_dino_est[name] = _resolved_xy
                            _last_dino_est[_gz_name_match] = _resolved_xy  # also under PDDL name
                            _publish_perception_pose(
                                args, _gz_name_match,
                                _resolved_pose["x"], _resolved_pose["y"], _resolved_pose["z"])
                            step0.args = dict(step0.args)
                            step0.args[name_key] = _gz_name_match
                            continue  # skip DINO for this name

                    if det_img_all is None or det_K_all is None:
                        print(f"[LOOP] No camera for '{name}' — skip")
                        continue

                    # Load depth array for real-robot depth-based unprojection.
                    # Both wrist and overview cameras are RealSense D435i → both have depth.
                    _depth_arr = None
                    _depth_file = {
                        "wrist":    _REPO_ROOT / "data" / "depth.npy",
                        "overview": _REPO_ROOT / "data" / "depth_overview.npy",
                    }.get(src_label_all)
                    if _depth_file is not None and _depth_file.exists():
                        try:
                            _depth_arr = _np.load(str(_depth_file))
                        except Exception:
                            pass

                    pose_est = perception.get_pose(
                        name, det_img_all, det_K_all, det_ctb_all,
                        vlm_description=name.replace("_", " "),
                        depth_image=_depth_arr,
                    )
                    if perception._last_detection:
                        _dino_detections.append(perception._last_detection.copy())
                    if pose_est:
                        print(f"[LOOP] DINO [{src_label_all}]: '{name}' → "
                              f"({pose_est['x']:.3f},{pose_est['y']:.3f},{pose_est['z']:.3f})")

                        # SIM-ONLY: snap DINO estimate to nearest Gazebo oracle position.
                        # DINO correctly identifies which object, but its 3D projection
                        # from a 2D image has cm-level errors → IK can fail on slightly
                        # off positions. Oracle gives ground-truth → use it in sim.
                        # On real robot gazebo_poses is empty → block never executes.
                        _pub_x, _pub_y, _pub_z = pose_est["x"], pose_est["y"], pose_est["z"]
                        _gz_resolved = None
                        if gazebo_poses:
                            _rbase = _np.array([0.20, 0.0])
                            _pxy   = _np.array([_pub_x, _pub_y])
                            _best_gz, _best_d = None, float("inf")
                            for _gz, _gp in gazebo_poses.items():
                                _d = float(_np.linalg.norm(
                                    _pxy - (_np.array([_gp["x"], _gp["y"]]) - _rbase)))
                                if _d < _best_d:
                                    _best_d, _best_gz = _d, _gz
                            if _best_gz and _best_d < 0.15:
                                _gz_resolved = _best_gz
                                _gp = gazebo_poses[_best_gz]
                                _pub_x = _gp["x"] - _rbase[0]
                                _pub_y = _gp["y"] - _rbase[1]
                                # Oracle z = world z of model origin (typically object
                                # centre) → converts to panda_link0 frame.
                                # On real robot this block never runs (gazebo_poses empty).
                                _pub_z = max(_gp.get("z", 0.770 + _pub_z) - 0.770, 0.0)
                                print(f"[LOOP] SIM snap: "
                                      f"DINO({pose_est['x']:.3f},{pose_est['y']:.3f},{pose_est['z']:.3f})"
                                      f" → oracle '{_best_gz}' "
                                      f"({_pub_x:.3f},{_pub_y:.3f},{_pub_z:.3f}) "
                                      f"Δxy={_best_d*100:.1f}cm")

                        _last_dino_est[name] = (_pub_x, _pub_y)
                        # Estimate height from the raw DINO bbox even in sim: the oracle
                        # snap overrides xyz, but _last_detection still holds the bbox.
                        _height_m = _estimate_object_height(
                            perception._last_detection,
                            (_pub_x, _pub_y, _pub_z),
                            det_K_all,
                            det_ctb_all,
                        )
                        if _height_m is not None:
                            print(f"[LOOP] height est: {_height_m*100:.1f} cm")
                        _publish_perception_pose(
                            args, name, _pub_x, _pub_y, _pub_z, height_m=_height_m)
                        if _gz_resolved and _gz_resolved != name:
                            _publish_perception_pose(
                                args, _gz_resolved, _pub_x, _pub_y, _pub_z,
                                height_m=_height_m)
                            step0.args = dict(step0.args)
                            step0.args[name_key] = _gz_resolved
                    else:
                        print(f"[LOOP] DINO [{src_label_all}]: '{name}' non rilevato — oracle fallback")

                # Save DINO bounding-box overlay for this iteration
                if _dino_detections and det_img_all is not None:
                    try:
                        from vlm.perception import PerceptionModule as _PM
                        _ann = _PM.draw_detections(det_img_all, _dino_detections)
                        _dino_path = _RUN_DIR / f"iter_{iteration+1:02d}_dino.png"
                        _ann.save(str(_dino_path))
                        print(f"[LOOP] DINO annotation saved: {_dino_path.name}")
                    except Exception as _ae:
                        print(f"[WARN] DINO annotation failed: {_ae}")

            except Exception as _pe:
                print(f"[WARN] pre-step perception failed: {_pe}")

        # Restore accumulated enrichment if current plan has none.
        # VLM correctly omits enrichment for repeat iterations, but generate_problem
        # needs the action definitions (e.g. pour effects) to infer the PDDL goal.
        _cur_da = plan_grounded.domain_additions
        _cur_enriched = (
            _cur_da.get("new_predicates") or _cur_da.get("new_actions") or
            _cur_da.get("new_types") or _cur_da.get("modified_preconditions")
        )
        if not _cur_enriched and _accumulated_da:
            plan_grounded.domain_additions = _accumulated_da

        # Generate PDDL + save comprehensive debug info for this iteration
        pddl_str = ""
        try:
            from planner.problem_generator import generate_problem
            pddl_str = generate_problem(plan_grounded)
            print("\n  PDDL PROBLEM:")
            for line in pddl_str.splitlines():
                print(f"    {line}")
            print()
        except Exception as _pe:
            pddl_str = f"# generation failed: {_pe}"

        # Save per-iteration debug package to run directory
        _iter_n = iteration + 1
        try:
            import json as _dbg_json
            from pathlib import Path as _PPath

            # 1. VLM plan JSON (raw + grounded)
            # Full remaining plan (all steps, before extracting current step)
            _full_plan_dict  = _dbg_json.loads(_current_plan.to_json()) if _current_plan else {}
            # Current step only (what gets executed this iteration)
            _plan_raw_dict   = _dbg_json.loads(plan.to_json())
            _plan_grnd_dict  = _dbg_json.loads(plan_grounded.to_json())

            # 2. PDDL domain content
            _domain_path = (_REPO_ROOT / "pddl" / "domains" /
                            f"{plan.domain_template}.pddl")
            _domain_str = (_domain_path.read_text()
                           if _domain_path.exists() else "# domain file not found")

            # 3. Comprehensive debug JSON
            _debug = {
                "iteration":       _iter_n,
                "task":            args.task,
                "world":           getattr(args, "world", "unknown"),
                "completed_steps": completed_steps,
                "vlm_time_s":      round(vlm_time, 2),
                "full_remaining_plan": _full_plan_dict,   # all remaining steps
                "plan_raw":        _plan_raw_dict,        # current step only
                "plan_grounded":   _plan_grnd_dict,
                "domain_template": plan.domain_template,
                "domain_additions": plan.domain_additions,
                "pddl_problem":    pddl_str,
                "step_primitive":  step0.primitive if step0 else None,
                "step_args":       dict(step0.args) if step0 else {},
                "dino_estimates":  dict(_last_dino_est),
                "placed_at":       {k: list(v) for k, v in _placed_at.items()},
                "using_overview_cam": _using_overview,
            }
            _iter_dir = _RUN_DIR / f"iter_{_iter_n:02d}"
            _iter_dir.mkdir(exist_ok=True)

            (_iter_dir / "debug.json").write_text(
                _dbg_json.dumps(_debug, indent=2, ensure_ascii=False))
            (_iter_dir / "full_remaining_plan.json").write_text(
                _dbg_json.dumps(_full_plan_dict, indent=2, ensure_ascii=False))
            (_iter_dir / "plan_current_step.json").write_text(
                _dbg_json.dumps(_plan_raw_dict, indent=2, ensure_ascii=False))
            (_iter_dir / "plan_grounded.json").write_text(
                _dbg_json.dumps(_plan_grnd_dict, indent=2, ensure_ascii=False))
            (_iter_dir / "problem.pddl").write_text(pddl_str)
            (_iter_dir / f"domain_{plan.domain_template}.pddl").write_text(_domain_str)

            # Move wrist snapshot into iter subfolder
            import shutil as _shutil
            _wrist_src = _RUN_DIR / f"iter_{_iter_n:02d}_wrist.png"
            if _wrist_src.exists():
                _shutil.move(str(_wrist_src), str(_iter_dir / "wrist.png"))
            _annot_src = _RUN_DIR / f"iter_{_iter_n:02d}_annotated.png"
            if _annot_src.exists():
                _shutil.move(str(_annot_src), str(_iter_dir / "overview_annotated.png"))
            # Also save current overview image
            _ov_src = _REPO_ROOT / "data" / "scene_overview.png"
            if _ov_src.exists():
                _shutil.copy2(str(_ov_src), str(_iter_dir / "overview.png"))

        except Exception as _save_err:
            print(f"[WARN] Debug save failed: {_save_err}")

        # 6. Serialize + inject.
        # Full PDDL pipeline (no direct flag): orchestrator runs FastDownward to
        # validate the single-step plan before dispatch.  The problem_generator
        # infers the live robot state from the VLM plan structure:
        #   - pick steps   → object starts on a surface
        #   - place steps without prior pick → arm is already holding the object
        #   - pour/tilt steps without prior pick → arm is already holding the source
        # This makes single-step validation correct for all mid-task states.
        payload = json.dumps({
            "command":  args.task,
            "vlm_plan": json.loads(plan_grounded.to_json()),
        })
        bash_cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/ros2_ws/install/setup.bash && "
            "python3 /workspace/scripts/_publish_plan.py"
        )
        inject_result = subprocess.run(
            docker_cmd + ["bash", "-c", bash_cmd],
            input=payload.encode(),
            capture_output=True,
        )
        if inject_result.returncode != 0:
            print(f"[FAIL] Injection failed: {inject_result.stderr.decode().strip()}")
            break
        print(f"[OK]   Step injected.")

        # 7. Wait for step completion
        print("[LOOP] Attendo completamento step...")
        result = _wait_step_complete(args, timeout=60, min_seq=_last_seq + 1)
        if "seq" in result:
            _last_seq = result["seq"]

        # Build step description — include original VLM name + grounded PDDL name
        # so the VLM can match its own terminology with the completed action.
        s0_orig = plan.steps[0]
        s0_grnd = plan_grounded.steps[0]
        # Always use ORIGINAL VLM names in completed_steps context.
        # The Gazebo resolution (glass→coffee_cup) is sim-internal — VLM should
        # see its own names so it recognises completed steps correctly.
        obj_orig = s0_orig.args.get("object", s0_orig.args.get("target", "?"))
        loc_orig = s0_orig.args.get("location", "")
        step_desc = (f"{s0_orig.primitive}({obj_orig}, {loc_orig})"
                     if loc_orig else f"{s0_orig.primitive}({obj_orig})")
        if result.get("success"):
            # Detect look_at loop: same look_at repeated → replan instead of break
            if s0_grnd.primitive == "look_at" and step_desc in completed_steps:
                print(f"[WARN] look_at('{obj_orig}') già eseguito — "
                      "DINO non riesce a trovare l'oggetto → replan")
                _current_plan = None
                _last_failed_step = f"look_at({obj_orig}) — object not detectable"
                _replan_count += 1
                continue

            completed_steps.append(step_desc)
            print(f"[OK]   Step completato: {step_desc}")

            # Track place destinations for annotation markers
            if s0_orig.primitive == "place":
                obj_placed = s0_orig.args.get("object", "")
                loc_placed = s0_grnd.args.get("location", "")
                if obj_placed and loc_placed in _last_dino_est:
                    px, py = _last_dino_est[loc_placed]
                    _placed_at[obj_placed] = (px, py)
                    print(f"[LOOP] Annotation: '{obj_placed}' placed at "
                          f"({px:.2f},{py:.2f}) → ✓ marker added to future images")
        else:
            # ── REPLANNING ON FAILURE ────────────────────────────────────────
            print(f"[FAIL] Step fallito: {step_desc}")
            _replan_count += 1
            if _replan_count > 3:
                print(f"[LOOP] ❌ Troppi replan ({_replan_count}) — task abortito")
                break
            print(f"[LOOP] ⚠️  Replan #{_replan_count} — rigenero piano completo...")
            _current_plan     = None          # force full replan next iteration
            _last_failed_step = step_desc     # context for VLM
            # Do NOT break — continue to next iteration which will replan
        # NOTE: task_complete from orchestrator = last step of CURRENT plan done.
        # In closed-loop, task completion is determined by the VLM (next iteration
        # returns complete=true or 0 steps), not by step count.  Do NOT break here.
    else:
        print(f"\n[WARN] Limite massimo di {args.max_steps} step raggiunto.")

    print(f"\n[LOOP] Steps completati: {completed_steps}")
    # Save completed steps to run directory
    with open(str(_RUN_DIR / "run_info.txt"), "a") as _rf:
        _rf.write(f"steps:     {completed_steps}\n")
        _rf.write(f"n_steps:   {len(completed_steps)}\n")
    print(f"[LOOP] Debug images saved in: {_RUN_DIR.relative_to(_REPO_ROOT)}")

    # Generate self-contained HTML report for this run
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_generate_report",
            str(_REPO_ROOT / "scripts" / "_generate_report.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _report = _mod.generate_html_report(_RUN_DIR)
        print(f"[LOOP] Report HTML: {_report.relative_to(_REPO_ROOT)}")
    except Exception as _re:
        print(f"[WARN] Report generation failed: {_re}")


if __name__ == "__main__":
    main()
