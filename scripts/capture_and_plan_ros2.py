#!/usr/bin/env python3
"""Capture ROS 2 RGB-D camera topics, plan, and localize detected objects.

Unlike ``capture_and_plan.py``, this script never opens RealSense devices
directly. Camera ownership remains with the ROS 2 RealSense nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image as PilImage


_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
_RUNS_DIR = _REPO_ROOT / "data" / "real_runs"


def _run_dir(task: str, parent: Path | None) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_") or "capture"
    root = parent.resolve() if parent else _RUNS_DIR
    result = root / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{slug}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def _docker(args: argparse.Namespace) -> list[str]:
    return (["sudo", "docker"] if args.sudo_docker else ["docker"]) + [
        "exec", args.container,
    ]


def _capture(args: argparse.Namespace, run_dir: Path) -> None:
    try:
        relative_dir = run_dir.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError("--output-dir must be inside the repository") from exc
    container_dir = f"/workspace/{relative_dir}"
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_capture_ros2_cameras.py "
        f"--output-dir {shlex.quote(container_dir)} --timeout {args.capture_timeout} "
        f"--overview-image-topic {shlex.quote(args.overview_image_topic)} "
        f"--overview-depth-topic {shlex.quote(args.overview_depth_topic)} "
        f"--overview-info-topic {shlex.quote(args.overview_info_topic)} "
        f"--wrist-image-topic {shlex.quote(args.wrist_image_topic)} "
        f"--wrist-depth-topic {shlex.quote(args.wrist_depth_topic)} "
        f"--wrist-info-topic {shlex.quote(args.wrist_info_topic)}"
    )
    if not args.no_wrist:
        command += " --include-wrist"
    result = subprocess.run(
        _docker(args) + ["bash", "-c", command], capture_output=True, text=True,
        timeout=args.capture_timeout + 10,
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ROS 2 overview capture failed")


def _load_camera(run_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.asarray(PilImage.open(run_dir / f"{name}.png").convert("RGB"))
    depth = np.load(run_dir / f"{name}_depth_mm.npy")
    with (run_dir / f"{name}_camera_info.json").open(encoding="utf-8") as stream:
        K = np.asarray(json.load(stream)["K"], dtype=float)
    if image.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"{name} aligned depth shape {depth.shape} does not match color "
            f"shape {image.shape[:2]}"
        )
    return image, depth, K


def _load_pose(run_dir: Path, name: str) -> np.ndarray | None:
    path = run_dir / f"{name}_camera_pose.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as stream:
        return np.asarray(json.load(stream)["cam_to_base"], dtype=float)


def _names_from_plan(plan, explicit: list[str]) -> list[str]:
    keys = {"object", "target", "location", "from", "to", "container"}
    infrastructure = {"table", "ground", "floor", "workspace"}
    values = list(explicit)
    values.extend(
        value
        for step in plan.steps
        for key, value in step.args.items()
        if key in keys and isinstance(value, str) and value not in infrastructure
    )
    return list(dict.fromkeys(values))


def _depth_pose(
    box: list[float], depth_mm: np.ndarray, K: np.ndarray, cam_to_base: np.ndarray
) -> tuple[dict | None, float | None]:
    from vlm.perception import PerceptionModule

    z_camera = PerceptionModule._median_depth_from_box(depth_mm, box)
    if z_camera is None or z_camera <= 0.05:
        return None, None
    u = (box[0] + box[2]) / 2.0
    v = (box[1] + box[3]) / 2.0
    point_camera = np.linalg.inv(K) @ np.array([u, v, 1.0]) * z_camera
    point_base = cam_to_base[:3, :3] @ point_camera + cam_to_base[:3, 3]
    return {
        "frame_id": "fr3_link0",
        "position": {
            "x": float(point_base[0]),
            "y": float(point_base[1]),
            "z": float(point_base[2]),
        },
        # RGB-D gives a 3-D point; object orientation is not inferred.
        "orientation": None,
    }, float(z_camera)


def _publish_pose(args: argparse.Namespace, name: str, pose: dict) -> None:
    position = pose["position"]
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_publish_perception_pose.py "
        f"--object {shlex.quote(name)} --x {position['x']:.8f} "
        f"--y {position['y']:.8f} --z {position['z']:.8f}"
    )
    result = subprocess.run(
        _docker(args) + ["bash", "-c", command], capture_output=True, text=True,
        timeout=10,
    )
    if result.returncode != 0:
        print(f"[WARN] Could not publish pose for {name}: {result.stderr.strip()}")


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to normalized (x, y, z, w)."""
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        values = ((rotation[2, 1] - rotation[1, 2]) / s,
                  (rotation[0, 2] - rotation[2, 0]) / s,
                  (rotation[1, 0] - rotation[0, 1]) / s, 0.25 * s)
    else:
        index = int(np.argmax(np.diag(rotation)))
        indices = (1, 2, 0) if index == 0 else ((2, 0, 1) if index == 1 else (0, 1, 2))
        j, k, _ = indices
        s = math.sqrt(1 + rotation[index, index] - rotation[j, j] - rotation[k, k]) * 2
        q = [0.0, 0.0, 0.0, 0.0]
        q[index] = 0.25 * s
        q[j] = (rotation[index, j] + rotation[j, index]) / s
        q[k] = (rotation[index, k] + rotation[k, index]) / s
        q[3] = (rotation[k, j] - rotation[j, k]) / s
        values = tuple(q)
    norm = math.sqrt(sum(value * value for value in values))
    return tuple(value / norm for value in values)


def _start_overview_tf_publisher(args: argparse.Namespace, run_dir: Path) -> bool:
    """Keep the calibrated static TF alive in the container when launch did not."""
    pose = _load_pose(run_dir, "overview")
    if pose is None:
        return False
    qx, qy, qz, qw = _rotation_to_quaternion(pose[:3, :3])
    x, y, z = pose[:3, 3]
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "ros2 run tf2_ros static_transform_publisher "
        f"--x {x:.12g} --y {y:.12g} --z {z:.12g} "
        f"--qx {qx:.12g} --qy {qy:.12g} --qz {qz:.12g} --qw {qw:.12g} "
        "--frame-id fr3_link0 --child-frame-id overview_camera_optical_frame "
        "--ros-args -r __node:=overview_camera_calibration_tf"
    )
    docker = ["sudo", "docker"] if args.sudo_docker else ["docker"]
    result = subprocess.run(
        docker + ["exec", "-d", args.container, "bash", "-c", command],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("[OK] Started persistent calibrated overview transform publisher")
        return True
    print(f"[WARN] Could not start overview transform publisher: {result.stderr.strip()}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="", help="Natural-language task")
    parser.add_argument("--objects", nargs="*", default=[], help="Extra DINO queries")
    parser.add_argument("--no-vlm", action="store_true", help="Capture/localize only")
    parser.add_argument("--no-wrist", action="store_true")
    parser.add_argument("--publish-poses", action="store_true")
    parser.add_argument(
        "--no-publish-overview-tf", action="store_true",
        help="Do not start a static TF publisher when the transform is missing",
    )
    parser.add_argument("--skip-pddl", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--container", default="vlm_ros2")
    parser.add_argument("--sudo-docker", action="store_true")
    parser.add_argument("--capture-timeout", type=float, default=10.0)
    parser.add_argument(
        "--overview-image-topic",
        default="/overview_camera/overview_camera/color/image_raw",
    )
    parser.add_argument(
        "--overview-depth-topic",
        default="/overview_camera/overview_camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--overview-info-topic",
        default="/overview_camera/overview_camera/aligned_depth_to_color/camera_info",
    )
    parser.add_argument("--wrist-image-topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--wrist-depth-topic", default="/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--wrist-info-topic", default="/camera/color/camera_info")
    args = parser.parse_args()
    if not args.task and not args.no_vlm:
        parser.error("--task is required unless --no-vlm is used")
    if args.no_vlm and not args.objects:
        parser.error("--no-vlm requires at least one --objects query")
    if args.capture_timeout <= 0:
        parser.error("--capture-timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    run_dir = _run_dir(args.task, args.output_dir)
    print(f"[INFO] Run directory: {run_dir}")
    try:
        _capture(args, run_dir)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    with (run_dir / "capture_manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    print(f"[INFO] Overview TF: {manifest['overview_tf_source']}")
    if (
        manifest["overview_tf_source"] == "published_from_calibration"
        and not args.no_publish_overview_tf
    ):
        _start_overview_tf_publisher(args, run_dir)

    overview_array, overview_depth, overview_K = _load_camera(run_dir, "overview")
    overview_image = PilImage.fromarray(overview_array)
    images = [overview_image]
    if manifest["cameras"]["wrist"]["available"]:
        images.append(PilImage.open(run_dir / "wrist.png").convert("RGB"))

    plan = None
    if not args.no_vlm:
        from vlm.planner import VLMPlanner

        planner = VLMPlanner()
        planner.load()
        plan = planner.plan_remaining(args.task, images, completed_steps=[])
        (run_dir / "plan.json").write_text(plan.to_json(), encoding="utf-8")
        print(f"[OK] Plan saved with {len(plan.steps)} step(s)")
        for index, step in enumerate(plan.steps, 1):
            arguments = ", ".join(f"{key}={value}" for key, value in step.args.items())
            print(f"  {index}. {step.primitive}({arguments})")
        if not args.skip_pddl:
            try:
                from planner.problem_generator import generate_problem

                (run_dir / "problem.pddl").write_text(
                    generate_problem(plan), encoding="utf-8"
                )
            except Exception as exc:
                print(f"[WARN] PDDL generation failed: {exc}")

    names = _names_from_plan(plan, args.objects) if plan is not None else args.objects
    from vlm.perception import PerceptionModule

    perception = PerceptionModule()
    perception.load()
    cam_to_base = _load_pose(run_dir, "overview")
    if cam_to_base is None:
        print("[ERROR] Overview cam_to_base is unavailable", file=sys.stderr)
        return 1

    detections = []
    for name in names:
        # get_pose performs GroundingDINO and records its best box. We calculate
        # the output point explicitly so a missing depth value never silently
        # falls back to a table-plane assumption.
        perception.get_pose(
            name, overview_image, overview_K, cam_to_base,
            vlm_description=name.replace("_", " "), depth_image=overview_depth,
        )
        detection = perception._last_detection
        if detection is None:
            detections.append({"name": name, "detected": False, "pose": None})
            print(f"[WARN] Not detected: {name}")
            continue
        pose, z_camera = _depth_pose(
            detection["box"], overview_depth, overview_K, cam_to_base
        )
        record = {
            "name": name,
            "detected": True,
            "score": float(detection["score"]),
            "box_xyxy": [float(value) for value in detection["box"]],
            "depth_m": z_camera,
            "pose": pose,
            "pose_method": "aligned_depth_bbox_median" if pose else None,
            "camera": "overview",
        }
        detections.append(record)
        if pose is None:
            print(f"[WARN] Detected {name}, but no valid aligned depth was available")
        else:
            p = pose["position"]
            print(f"[OK] {name}: ({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f}) fr3_link0")
            if args.publish_poses:
                _publish_pose(args, name, pose)

    write_data = {
        "frame_id": "fr3_link0",
        "source_camera": "overview",
        "depth_units": "metres",
        "detections": detections,
    }
    (run_dir / "detections_with_poses.json").write_text(
        json.dumps(write_data, indent=2) + "\n", encoding="utf-8"
    )
    drawable = [
        {"name": d["name"], "box": d["box_xyxy"], "score": d["score"]}
        for d in detections if d.get("detected")
    ]
    if drawable:
        PerceptionModule.draw_detections(overview_image, drawable).save(
            run_dir / "overview_detections.png"
        )
    print(f"[OK] Results saved to {run_dir / 'detections_with_poses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
