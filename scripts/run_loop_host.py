#!/usr/bin/env python3
"""
run_loop_host.py — Closed-loop task execution (HOST side).

Implements two closed-loop policies:
  default: [capture] -> [VLM verify/replan] -> [inject] -> repeat
  --replan-on-failure-only: [VLM full plan] -> [execute cached steps]; replan on failure

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
import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _docker(container: str, use_sudo: bool) -> list[str]:
    return (["sudo", "docker"] if use_sudo else ["docker"]) + ["exec", "-i", container]


def _world_label(args) -> str:
    """Return truthful environment metadata for logs."""
    return "real_ros2" if args.real_ros2 else getattr(args, "world", "unknown")


def _run_in_container(args, bash_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        _docker(args.container, args.sudo_docker) + ["bash", "-c", bash_cmd],
        capture_output=True, timeout=timeout,
    )


def _capture_robot_state(args, timeout: float = 2.0) -> dict:
    """Read joints and end-effector TF from ROS 2 without moving the robot."""
    robot = "fr3" if args.real_ros2 else "panda"
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_capture_robot_state.py "
        f"--robot {robot} --timeout {timeout:.3f}"
    )
    try:
        result = _run_in_container(args, command, timeout=int(timeout) + 4)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "errors": [f"state capture failed: {exc}"]}

    for line in reversed(result.stdout.decode(errors="replace").splitlines()):
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(state, dict) and "available" in state:
            if result.returncode != 0:
                state.setdefault("errors", []).append(
                    result.stderr.decode(errors="replace").strip()
                    or f"capture exited with code {result.returncode}"
                )
            return state
    error = result.stderr.decode(errors="replace").strip()
    return {
        "available": False,
        "errors": [error or "robot-state helper returned no JSON"],
    }


def _update_iteration_debug(iter_dir: Path, updates: dict) -> None:
    """Merge post-dispatch facts into an iteration debug file."""
    path = iter_dir / "debug.json"
    try:
        debug = json.loads(path.read_text(encoding="utf-8"))
        debug.update(updates)
        path.write_text(
            json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[WARN] Could not update {path.name} after execution: {exc}")


class _ExperimentVideoRecording:
    """Own the docker-exec recorder process and finalize its MP4 safely."""

    def __init__(
        self, process: subprocess.Popen, log_stream, output_path: Path,
        label: str, log_name: str, *, container: str, use_sudo: bool,
        pid_path: Path,
    ):
        self.process = process
        self.log_stream = log_stream
        self.output_path = output_path
        self.label = label
        self.log_name = log_name
        self.container = container
        self.use_sudo = use_sudo
        self.pid_path = pid_path
        self._stopped = False

    def _signal_recorder(self) -> bool:
        """Signal the recorder inside Docker, not the disposable exec client."""
        try:
            pid = self.pid_path.read_text(encoding="utf-8").strip()
        except OSError:
            return False
        if not pid.isdigit() or int(pid) <= 1:
            return False
        try:
            result = subprocess.run(
                _docker(self.container, self.use_sudo) + ["kill", "-INT", pid],
                capture_output=True,
                timeout=3,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        signaled_in_container = self._signal_recorder()
        if self.process.poll() is None:
            if not signaled_in_container:
                self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        try:
            self.pid_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"[WARN] Could not remove recorder PID file: {exc}")
        self.log_stream.close()
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            print(f"[LOOP] {self.label} video: {self.output_path.name}")
        else:
            print(
                f"[WARN] {self.label} produced no video; check "
                f"{self.log_name} and its ROS image topic"
            )


def _format_ros_double(value: float) -> str:
    """Serialize a ROS CLI double without it being inferred as an integer."""
    return f"{float(value):.6f}"


def _start_ros_video(
    args, run_dir: Path, *, enabled: bool, topic: str, fps: float,
    filename: str, log_name: str, label: str, node_name: str,
) -> _ExperimentVideoRecording | None:
    """Start a ROS 2 topic recorder writing into this experiment directory."""
    if not enabled:
        return None
    try:
        relative_dir = run_dir.relative_to(_REPO_ROOT)
    except ValueError:
        print("[WARN] Run directory is outside the shared workspace; video disabled")
        return None

    output_path = run_dir / filename
    container_output = Path("/workspace") / relative_dir / output_path.name
    pid_path = run_dir / f".{node_name}.pid"
    container_pid_path = Path("/workspace") / relative_dir / pid_path.name
    try:
        pid_path.unlink()
    except FileNotFoundError:
        pass
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"echo $$ > {shlex.quote(str(container_pid_path))} && "
        "exec /workspace/ros2_ws/install/vlm_robot_planner/"
        "lib/vlm_robot_planner/webcam_recorder --ros-args "
        f"-r __node:={shlex.quote(node_name)} "
        f"-p image_topic:={shlex.quote(topic)} "
        f"-p output_path:={shlex.quote(str(container_output))} "
        f"-p fps:={_format_ros_double(fps)}"
    )
    log_stream = (run_dir / log_name).open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            _docker(args.container, args.sudo_docker) + ["bash", "-c", command],
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        log_stream.close()
        print(f"[WARN] Could not start {label}: {exc}")
        return None

    recording = _ExperimentVideoRecording(
        process,
        log_stream,
        output_path,
        label,
        log_name,
        container=args.container,
        use_sudo=args.sudo_docker,
        pid_path=pid_path,
    )
    time.sleep(1.0)
    if process.poll() is not None:
        recording.stop()
        print(
            f"[WARN] {label} recorder exited during startup; rebuild the ROS 2 "
            f"workspace and inspect {log_name}"
        )
        return None
    atexit.register(recording.stop)
    print(f"[LOOP] Recording {topic} -> {output_path.name}")
    return recording


def _start_experiment_video(
    args, run_dir: Path,
) -> _ExperimentVideoRecording | None:
    return _start_ros_video(
        args,
        run_dir,
        enabled=args.record_webcam,
        topic=args.webcam_topic,
        fps=args.webcam_fps,
        filename="experiment_webcam.mp4",
        log_name="webcam_recorder.log",
        label="Experiment webcam",
        node_name="experiment_webcam_recorder",
    )


def _capture_overview_image(args, run_dir: Path, label: str) -> bool:
    """Capture a single frame from the overview camera and save it to run_dir."""
    try:
        if getattr(args, "real_ros2", False):
            capture_dir = run_dir / "_overview_capture"
            capture_dir.mkdir(exist_ok=True)
            container_dir = "/workspace" / run_dir.relative_to(_REPO_ROOT) / "_overview_capture"
            command = (
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/ros2_ws/install/setup.bash && "
                "python3 /workspace/scripts/_capture_ros2_cameras.py "
                f"--output-dir {container_dir} --timeout {args.capture_timeout} "
                f"--overview-image-topic {shlex.quote(args.overview_image_topic)} "
                f"--overview-depth-topic {shlex.quote(args.overview_depth_topic)} "
                f"--overview-info-topic {shlex.quote(args.overview_info_topic)}"
            )
            result = _run_in_container(args, command, timeout=int(args.capture_timeout) + 8)
            if result.returncode == 0:
                src = capture_dir / "overview.png"
                if src.exists():
                    dst = run_dir / f"overview_{label}.png"
                    import shutil
                    shutil.move(str(src), str(dst))
                    print(f"[LOOP] Overview snapshot ({label}): {dst.name}")
                    return True
            return False
        else:
            bash_cmd = (
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/ros2_ws/install/setup.bash && "
                "python3 /workspace/scripts/_capture_scene.py"
            )
            r = _run_in_container(args, bash_cmd, timeout=15)
            if r.returncode == 0:
                src = _REPO_ROOT / "data" / "scene_overview.png"
                if src.exists():
                    import shutil
                    dst = run_dir / f"overview_{label}.png"
                    shutil.copy2(str(src), str(dst))
                    print(f"[LOOP] Overview snapshot ({label}): {dst.name}")
                    return True
            return False
    except Exception as exc:
        print(f"[WARN] Overview capture ({label}) failed: {exc}")
        return False


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
    if getattr(args, "real_ros2", False):
        return _capture_real_ros2(args)
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


def _capture_real_ros2(args) -> Path | None:
    """Capture aligned RGB-D topics and map them to the loop's data contract."""
    data_dir = _REPO_ROOT / "data"
    capture_dir = data_dir / "_loop_ros2_capture"
    container_dir = "/workspace/data/_loop_ros2_capture"
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_capture_ros2_cameras.py "
        f"--output-dir {container_dir} --timeout {args.capture_timeout} "
        "--include-wrist "
        f"--overview-image-topic {shlex.quote(args.overview_image_topic)} "
        f"--overview-depth-topic {shlex.quote(args.overview_depth_topic)} "
        f"--overview-info-topic {shlex.quote(args.overview_info_topic)} "
        f"--wrist-image-topic {shlex.quote(args.wrist_image_topic)} "
        f"--wrist-depth-topic {shlex.quote(args.wrist_depth_topic)} "
        f"--wrist-info-topic {shlex.quote(args.wrist_info_topic)}"
    )
    result = _run_in_container(args, command, timeout=int(args.capture_timeout) + 8)
    for line in result.stdout.decode().strip().splitlines():
        print(f"       {line}")
    if result.returncode != 0:
        print(f"[FAIL] ROS 2 RGB-D capture: {result.stderr.decode().strip()}")
        return None

    try:
        with (capture_dir / "capture_manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)

        wrist = manifest["cameras"].get("wrist", {})
        if wrist.get("available") and wrist.get("pose_available"):
            primary_image = capture_dir / "wrist.png"
            primary_label = "wrist"
        else:
            # The closed loop can still operate from the fixed calibrated view.
            primary_image = capture_dir / "overview.png"
            primary_label = "overview fallback"
        # Keep all ROS captures run-local. Files created in the bind mount may
        # be owned by the container user, and calibrated files are immutable
        # inputs rather than capture outputs.
        args.ros2_capture_dir = capture_dir
        args.ros2_primary_camera = primary_label
        print(f"[OK]   ROS 2 capture ready (primary: {primary_label})")
        return primary_image
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] Invalid ROS 2 capture output: {exc}")
        return None


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


def _get_overview_cam_data(world_name: str = "office", real_ros2: bool = False):
    """
    Compute K and cam_to_base for the OVERVIEW camera.
    Reads pose from the world SDF file — update the world file to recalibrate.
    The overview camera is STATIC so this is computed once at startup.
    Returns (K, cam_to_base) or (None, None) on error.
    """
    try:
        import numpy as np, math

        if real_ros2:
            info_path = _REPO_ROOT / "data" / "overview_camera_info.json"
            pose_path = _REPO_ROOT / "data" / "overview_camera_pose.json"
            with info_path.open(encoding="utf-8") as stream:
                K = np.asarray(json.load(stream)["K"], dtype=float)
            with pose_path.open(encoding="utf-8") as stream:
                cam_to_base = np.asarray(
                    json.load(stream)["cam_to_base"], dtype=float
                )
            if K.shape != (3, 3) or cam_to_base.shape != (4, 4):
                raise ValueError("invalid overview calibration matrix shape")
            print("[INFO] Overview calibration loaded from AprilTag output files")
            return K, cam_to_base

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
    *,
    camera_matrix=None,
    cam_to_base=None,
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

    K, R, t = None, None, None
    if camera_matrix is not None and cam_to_base is not None:
        try:
            K = np.asarray(camera_matrix, dtype=float)
            cam_to_base = np.asarray(cam_to_base, dtype=float)
            base_to_cam = np.linalg.inv(cam_to_base)
            R = base_to_cam[:3, :3]
            t = base_to_cam[:3, 3]
        except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
            print(f"[WARN] Invalid annotation calibration: {exc}")
    else:
        ci_path = Path(data_dir) / info_file
        cp_path = Path(data_dir) / pose_file
        if ci_path.exists() and cp_path.exists():
            try:
                with open(ci_path) as f:
                    K = np.array(json.load(f)["K"])
                with open(cp_path) as f:
                    cam_to_base = np.array(json.load(f)["cam_to_base"])
                base_to_cam = np.linalg.inv(cam_to_base)
                R = base_to_cam[:3, :3]
                t = base_to_cam[:3, 3]
            except (OSError, KeyError, TypeError, ValueError,
                    json.JSONDecodeError, np.linalg.LinAlgError) as exc:
                print(f"[WARN] Cannot load annotation calibration: {exc}")

    dbg  = image.copy()
    draw = ImageDraw.Draw(dbg)
    W, H = dbg.width, dbg.height
    CS   = max(5, min(W, H) // 80)   # cross arm length (tiny)

    # ── 1. Small cross at each projected object position ─────────────────────
    if R is not None:
        for i, (name, position) in enumerate(placed_at.items(), 1):
            px, py = position[:2]
            # Keep compatibility with older XY-only state, but use measured Z
            # whenever available. Real work surfaces are not at z=0.025 m.
            pz = position[2] if len(position) >= 3 else 0.025
            p_cam = R @ np.array([px, py, pz]) + t
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


def _placed_object_position(
    original_location: str,
    resolved_location: str,
    estimates: dict[str, tuple[float, float]],
    poses: dict[str, dict],
    gazebo_poses: dict[str, dict],
) -> tuple[float, float, float] | tuple[float, float] | None:
    """Resolve a successful place destination for handled-object annotation."""
    names = tuple(dict.fromkeys((resolved_location, original_location)))
    for name in names:
        pose = poses.get(name, {}).get("position", {})
        if all(axis in pose for axis in ("x", "y", "z")):
            return float(pose["x"]), float(pose["y"]), float(pose["z"])
        if name in estimates:
            return tuple(estimates[name])
        # Gazebo reports world coordinates; panda_link0 is at world x=0.20.
        gazebo_pose = gazebo_poses.get(name)
        if gazebo_pose and "x" in gazebo_pose and "y" in gazebo_pose:
            position = (
                float(gazebo_pose["x"]) - 0.20,
                float(gazebo_pose["y"]),
            )
            if "z" in gazebo_pose:
                # Gazebo world table top is z=0.770; panda_link0 is at table height.
                return position + (max(float(gazebo_pose["z"]) - 0.770, 0.0),)
            return position
    return None


def _publish_perception_pose(
    args, object_name: str, x: float, y: float, z: float,
    height_m: float | None = None,
) -> bool:
    """Publish a perception-estimated pose to /perception/object_pose.

    height_m: estimated object height in metres (from _estimate_object_height).
              Encoded in orientation.z; None → 0.0 → orchestrator uses fallback.
    """
    if not args.execute:
        return True

    height_arg = f" --height_m {height_m:.4f}" if height_m is not None else ""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_publish_perception_pose.py "
        f"--object {shlex.quote(object_name)} "
        f"--x {x:.6f} --y {y:.6f} --z {z:.6f}{height_arg}"
    )
    # Allows up to 10 s for DDS discovery plus 30 s of retrying cache ACKs.
    r = _run_in_container(args, bash_cmd, timeout=45)
    for line in r.stdout.decode().strip().splitlines():
        print(f"       {line}")
    if r.returncode != 0:
        error = r.stderr.decode().strip()
        if error:
            print(f"[WARN] Perception pose delivery failed: {error}")
    return r.returncode == 0


def _publish_dino_annotated_image(
    args, image_path: Path, camera_source: str,
) -> bool:
    """Publish a saved GroundingDINO overlay through the ROS 2 relay."""
    try:
        container_path = Path("/workspace") / image_path.relative_to(_REPO_ROOT)
    except ValueError:
        print(f"[WARN] DINO image is outside the shared workspace: {image_path}")
        return False

    frame_id = (
        "overview_camera_color_optical_frame"
        if camera_source == "overview"
        else "camera_color_optical_frame"
    )
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_publish_dino_annotated_image.py "
        f"--image {shlex.quote(str(container_path))} "
        f"--frame-id {shlex.quote(frame_id)}"
    )
    r = _run_in_container(args, bash_cmd, timeout=30)
    output = r.stdout.decode().strip()
    if output:
        for line in output.splitlines():
            print(f"       {line}")
    if r.returncode != 0:
        error = r.stderr.decode().strip()
        if error:
            print(f"[WARN] DINO annotated-image publication failed: {error}")
        return False
    return True


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




def _wait_step_complete(
    args, timeout: int = 60, min_seq: int = 0, request_id: str = "",
) -> dict:
    """Wait for step completion signal from orchestrator.

    min_seq: ignore step_complete messages with seq < this value, preventing
    stale TRANSIENT_LOCAL (latched) messages from previous steps being accepted
    as the result of the current step.
    """
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_wait_step_complete.py "
        f"--timeout {timeout} --min-seq {min_seq} "
        f"--request-id {shlex.quote(request_id)}"
    )
    r = _run_in_container(args, bash_cmd, timeout=timeout + 5)
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.decode().strip())
        except Exception:
            pass
    return {"success": False, "task_complete": False}


def main(real_ros2_default: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Closed-loop task execution")
    parser.add_argument("--task",       required=True)
    parser.add_argument("--max-steps",  type=int, default=20)
    parser.add_argument(
        "--container",
        default="vlm_ros2_real" if real_ros2_default else "vlm_ros2",
    )
    parser.add_argument("--sudo-docker", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute", dest="execute", action="store_true", default=True,
        help="Execute planned robot motions (default)",
    )
    execution.add_argument(
        "--no-execute", dest="execute", action="store_false",
        help="Capture, plan, and localize, but do not move the robot",
    )
    parser.add_argument(
        "--confirm-actions", action="store_true",
        help="Require y/yes confirmation immediately before injecting each action",
    )
    parser.add_argument(
        "--replan-on-failure-only",
        action="store_true",
        help=(
            "Reuse the remaining steps of the current VLM plan after a successful "
            "action; call the VLM again only after an action fails"
        ),
    )
    parser.add_argument(
        "--real-ros2", action="store_true", default=real_ros2_default,
        help="Use calibrated real-camera ROS 2 RGB-D topics; disable Gazebo lookup",
    )
    parser.add_argument("--world",      default="office",
                        help="Active Gazebo world (reads overview cam pose from world file)")
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
    overview_recording = parser.add_mutually_exclusive_group()
    overview_recording.add_argument(
        "--record-overview-video",
        dest="record_overview_video",
        action="store_true",
        help="Record the overview camera into the run directory (default)",
    )
    overview_recording.add_argument(
        "--no-record-overview-video",
        dest="record_overview_video",
        action="store_false",
        help="Disable overview-camera video recording",
    )
    parser.set_defaults(record_overview_video=True)
    parser.add_argument(
        "--overview-video-topic",
        default=None,
        help="ROS 2 overview topic to record; defaults according to run mode",
    )
    parser.add_argument(
        "--overview-video-fps", type=float, default=None,
        help="Output overview-video FPS (default: 30 real, 10 simulation)",
    )
    parser.add_argument("--wrist-image-topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--wrist-depth-topic", default="/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--wrist-info-topic", default="/camera/color/camera_info")
    webcam_recording = parser.add_mutually_exclusive_group()
    webcam_recording.add_argument(
        "--record-webcam", dest="record_webcam", action="store_true",
        help="Record the external ROS 2 experiment camera",
    )
    webcam_recording.add_argument(
        "--no-record-webcam", dest="record_webcam", action="store_false",
        help="Disable external experiment-camera recording",
    )
    parser.set_defaults(record_webcam=None)
    parser.add_argument(
        "--webcam-topic", default="/experiment_camera/image_raw",
        help="ROS 2 image topic recorded into the run directory",
    )
    parser.add_argument("--webcam-fps", type=float, default=20.0)
    args = parser.parse_args()
    if args.capture_timeout <= 0:
        parser.error("--capture-timeout must be positive")
    if args.webcam_fps <= 0:
        parser.error("--webcam-fps must be positive")
    if args.overview_video_fps is not None and args.overview_video_fps <= 0:
        parser.error("--overview-video-fps must be positive")
    if args.record_webcam is None:
        args.record_webcam = args.real_ros2
    if args.overview_video_topic is None:
        args.overview_video_topic = (
            args.overview_image_topic
            if args.real_ros2
            else "/overview_camera/image_raw"
        )
    if args.overview_video_fps is None:
        args.overview_video_fps = 30.0 if args.real_ros2 else 10.0

    execution_mode = "ENABLED" if args.execute else "DISABLED (observation only)"
    print(f"[LOOP] Robot execution: {execution_mode}")
    planning_mode = (
        "cached plan; VLM replans only after failure"
        if args.replan_on_failure_only
        else "VLM verifies/replans before every action"
    )
    print(f"[LOOP] Planning policy: {planning_mode}")

    # Create the run and start its external video before model loading so the
    # recording covers the complete closed-loop experiment invocation.
    import datetime
    _ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _world_tag = _world_label(args)
    _task_tag = args.task[:30].replace(" ", "_").replace("/", "-")
    _runs_root = "real_runs" if args.real_ros2 else "runs"
    _RUN_DIR = (
        _REPO_ROOT / "data" / _runs_root
        / f"{_ts}_{_world_tag}_{_task_tag}"
    )
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(_RUN_DIR / "run_info.txt"), "w") as _rf:
        _rf.write(f"timestamp: {_ts}\n")
        _rf.write(f"world:     {_world_tag}\n")
        _rf.write(f"task:      {args.task}\n")
        _rf.write(f"execute:   {args.execute}\n")
        _rf.write(f"replan_on_failure_only: {args.replan_on_failure_only}\n")
    print(f"[LOOP] Run dir: {_RUN_DIR.relative_to(_REPO_ROOT)}")
    _loop_started_monotonic = time.monotonic()
    _loop_started_at = datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(timespec="milliseconds")
    _video_recordings = []
    _overview_recording = _start_ros_video(
        args,
        _RUN_DIR,
        enabled=args.record_overview_video,
        topic=args.overview_video_topic,
        fps=args.overview_video_fps,
        filename="overview_camera.mp4",
        log_name="overview_camera_recorder.log",
        label="Overview camera",
        node_name="overview_camera_recorder",
    )
    if _overview_recording is not None:
        _video_recordings.append(_overview_recording)
    _experiment_recording = _start_experiment_video(args, _RUN_DIR)
    if _experiment_recording is not None:
        _video_recordings.append(_experiment_recording)

    print("[LOOP] Loading VLM (Qwen3-VL-8B-Instruct)…")
    from vlm.planner import VLMPlanner
    from vlm.perception import PerceptionModule
    from PIL import Image as PilImage

    vlm       = VLMPlanner()
    vlm.load()
    perception = PerceptionModule()
    perception.load()
    print("[OK]   VLM + PerceptionModule loaded.\n")

    # Capture overview camera at start of experiment
    _capture_overview_image(args, _RUN_DIR, "start")

    # Register shutdown handler to capture overview camera when program exits
    def _shutdown_capture():
        try:
            _capture_overview_image(args, _RUN_DIR, "shutdown")
        except Exception:
            pass
    atexit.register(_shutdown_capture)

    completed_steps: list[str] = []
    docker_cmd = _docker(args.container, args.sudo_docker)

    # Replanning on failure state
    _current_plan       = None   # cached full VLMPlan (remaining steps)
    _last_failed_step   = None   # step that caused last replan
    _replan_count       = 0      # how many times we've replanned

    # Overview camera calibration — computed once from world file (camera is static)
    _OV_K, _OV_CTB = _get_overview_cam_data(args.world, real_ros2=args.real_ros2)
    if _OV_K is not None:
        source = "AprilTag calibration" if args.real_ros2 else "static from SDF"
        print(f"[OK]   Overview camera calibration: ready ({source})")
    else:
        print("[WARN] Overview camera calibration failed — using wrist cam for VLM")

    # Tracks destinations of placed objects; new DINO detections within
    # EXCL_RADIUS of a recorded position are skipped to prevent re-picking.
    _placed_at: dict[
        str, tuple[float, float] | tuple[float, float, float]
    ] = {}
    _last_dino_est: dict[str, tuple[float, float]] = {}  # last DINO estimate per name
    _last_dino_pose: dict[str, dict] = {}  # full XYZ pose used by execution
    _EXCL_RADIUS = 0.10  # 10cm — objects within this radius are treated as identical
    # Tracks the last dispatch_seq; _wait_step_complete uses min_seq=_last_seq+1
    # to ignore stale TRANSIENT_LOCAL (latched) messages from previous steps.
    _last_seq: int = -1
    # Persists domain enrichments across iterations. The VLM enriches the domain
    # only when it first encounters a novel action; subsequent iterations omit it.
    # generate_problem needs the accumulated definitions to infer the PDDL goal.
    _accumulated_da: dict = {}

    for iteration in range(args.max_steps):
        _iteration_started_monotonic = time.monotonic()
        _iteration_started_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="milliseconds")
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
        _held_object = next(
            (
                step[5:].split(",")[0].rstrip(")")
                for step in reversed(completed_steps)
                if step.startswith("pick(")
            ),
            None,
        ) if holding else None

        # REMOVED Pre-scan

        # if not args.execute:
        #     print("[LOOP] --no-execute: pre-scan robot motion skipped")
        # elif not holding:
        #     _pre_scan(args)
        #     time.sleep(1.0)
        # else:
        #     print("[LOOP] Holding object — skip scan pose, capture from current arm position")

        # 2. Capture
        _capture_started_monotonic = time.monotonic()
        image_path = _capture(args)
        _capture_time_s = time.monotonic() - _capture_started_monotonic
        if image_path is None:
            print("[FAIL] No image — aborting loop")
            break
        image = PilImage.open(image_path).convert("RGB")

        # Wrist snapshot saved here temporarily; moved into the iter subfolder at debug-save time
        iter_path = _RUN_DIR / f"iter_{iteration+1:02d}_wrist.png"
        image.save(str(iter_path))
        print(f"[LOOP] Snapshot: {iter_path.name}")

        # Load overview camera image for VLM (fixed reference, better perspective)
        _ov_path = (
            args.ros2_capture_dir / "overview.png"
            if args.real_ros2
            else _REPO_ROOT / "data" / "scene_overview.png"
        )
        if _ov_path.exists() and _OV_K is not None:
            image_vlm = PilImage.open(str(_ov_path)).convert("RGB")
        else:
            image_vlm = image   # fallback to wrist cam
        _using_overview = (_ov_path.exists() and _OV_K is not None)

        # The fixed extrinsic was estimated from the color image and therefore
        # uses color/camera_info.  Runtime 3-D unprojection, however, consumes
        # aligned_depth_to_color/image_raw and must use the CameraInfo captured
        # from that aligned-depth stream.  Do not reuse the calibration K here,
        # even though RealSense commonly publishes identical matrices for both.
        _ov_depth_K = _OV_K
        if args.real_ros2 and _using_overview:
            _runtime_info = args.ros2_capture_dir / "overview_camera_info.json"
            try:
                with _runtime_info.open(encoding="utf-8") as _stream:
                    _runtime_data = json.load(_stream)
                _candidate_K = np.asarray(_runtime_data["K"], dtype=float)
                _info_size = (
                    int(_runtime_data["width"]), int(_runtime_data["height"])
                )
                if _candidate_K.shape != (3, 3) or not np.isfinite(_candidate_K).all():
                    raise ValueError("K is not a finite 3x3 matrix")
                if _info_size != image_vlm.size:
                    raise ValueError(
                        f"CameraInfo size {_info_size} does not match color image "
                        f"size {image_vlm.size}"
                    )
                _ov_depth_K = _candidate_K
                print(
                    "[INFO] Overview depth K from current "
                    "aligned_depth_to_color/camera_info"
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                # A stale color K must not be used with depth merely because it
                # happens to have the same dimensions.
                _ov_depth_K = None
                print(f"[WARN] Invalid runtime overview depth CameraInfo: {exc}")

        # Persist last scan-pose image + calibration for place location detection.
        # When arm is holding an object, the camera view is distorted by the arm.
        # Using the last FREE scan gives better geometry for location detection.
        if not holding and not args.real_ros2:
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
        raw_gazebo_poses = {} if args.real_ros2 else _get_gazebo_models(args)
        gazebo_poses = {k: v for k, v in raw_gazebo_poses.items() if k not in _INFRA}
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
            # Annotate on overview image using its fixed calibration directly.
            image_for_vlm = _annotate_handled_objects(
                image_vlm, _placed_at, str(_REPO_ROOT / "data"),
                camera_matrix=_OV_K, cam_to_base=_OV_CTB)
        else:
            image_for_vlm = _annotate_handled_objects(image, _placed_at, _data_dir_annot)

        if _placed_at:
            annot_path = _RUN_DIR / f"iter_{iteration+1:02d}_annotated.png"
            image_for_vlm.save(str(annot_path))
            src = "overview" if _using_overview else "wrist"
            print(f"[LOOP] Annotated image [{src}]: {annot_path.name} "
                  f"({len(_placed_at)} marker(s): {list(_placed_at.keys())})")

        # In the default mode the VLM verifies the scene and regenerates the
        # remaining plan before every action. With --replan-on-failure-only,
        # successful actions consume the cached plan locally; a new VLM call is
        # made only for the initial plan or after execution failure.
        _prev_plan_steps = []
        _use_cached_plan = (
            args.replan_on_failure_only
            and _current_plan is not None
            and bool(_current_plan.steps)
            and _last_failed_step is None
        )
        if _use_cached_plan:
            vlm_time = 0.0
            print(
                f"[LOOP] Piano VLM in cache: prossimo di "
                f"{len(_current_plan.steps)} step rimanenti "
                "(nessuna nuova inferenza)"
            )
        else:
            t_vlm = time.time()
            action_label = (
                "REPLAN" if _last_failed_step
                else ("PLAN" if not vlm_context else "VERIFY+PLAN")
            )
            print(
                f"[LOOP] VLM {action_label} (piano completo rimanente) "
                f"per: '{args.task}'"
            )

            _prev_plan_steps = [
                f"{s.primitive}({s.args})"
                for s in (_current_plan.steps if _current_plan else [])
            ]
            # Saved to inherit grasp_mode if the VLM drops it during replanning.
            _prev_current_plan = _current_plan

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

            # Preserve grasp_mode from the prior plan if replanning omitted it.
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
                            print(
                                f"[LOOP] Inherited grasp_mode="
                                f"'{_s.args['grasp_mode']}' for pick('{_obj}') "
                                "from previous plan"
                            )
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
                if args.replan_on_failure_only and _current_plan.steps:
                    _current_plan.steps.pop(0)
                    if not _current_plan.steps:
                        print("[LOOP] Piano VLM in cache esaurito.")
                        break
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
                if args.replan_on_failure_only and _current_plan.steps:
                    _current_plan.steps.pop(0)
                    if not _current_plan.steps:
                        print("[LOOP] Piano VLM in cache esaurito.")
                        break
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
                    det_K_all     = _ov_depth_K
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
                    if _held_object and _held_object in names_to_estimate:
                        names_to_estimate.pop(_held_object)
                        print(
                            f"[LOOP] Holding '{_held_object}' — skip DINO "
                            "(in gripper, not visible)"
                        )

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
                            _last_dino_pose[name] = {
                                "frame_id": "fr3_link0",
                                "position": dict(_resolved_pose),
                                "source": "gazebo",
                            }
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
                    if args.real_ros2:
                        _depth_file = {
                            "wrist": args.ros2_capture_dir / "wrist_depth_mm.npy",
                            "overview": (
                                args.ros2_capture_dir / "overview_depth_mm.npy"
                            ),
                        }.get(src_label_all)
                    else:
                        _depth_file = {
                            "wrist": _REPO_ROOT / "data" / "depth.npy",
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
                        excluded_positions=(
                            list(_placed_at.values())
                            if name_key in ("object", "target") else None
                        ),
                        exclusion_radius=_EXCL_RADIUS,
                    )
                    if perception._last_detection:
                        _dino_detections.append(perception._last_detection.copy())
                    if args.real_ros2 and pose_est:
                        # Real execution must never silently substitute a table-plane
                        # intersection when aligned depth is absent or invalid.
                        _valid_depth = None
                        if _depth_arr is not None and perception._last_detection:
                            _valid_depth = perception._median_depth_from_box(
                                _depth_arr, perception._last_detection["box"]
                            )
                        if _valid_depth is None:
                            print(
                                f"[LOOP] DINO [{src_label_all}]: '{name}' has no "
                                "valid aligned depth — pose rejected"
                            )
                            pose_est = None
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
                        _last_dino_pose[name] = {
                            "frame_id": "fr3_link0",
                            "position": {
                                "x": float(_pub_x),
                                "y": float(_pub_y),
                                "z": float(_pub_z),
                            },
                            "source": (
                                "overview_aligned_depth"
                                if args.real_ros2 and src_label_all == "overview"
                                else src_label_all
                            ),
                        }
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

                # Publish handled-object crosses too. Previously this overlay was
                # rebuilt from the raw frame, so RViz never showed the marks that
                # were present in the private VLM input image.
                if det_img_all is not None and (_dino_detections or _placed_at):
                    try:
                        from vlm.perception import PerceptionModule as _PM
                        if src_label_all == "overview":
                            _ann_base = _annotate_handled_objects(
                                det_img_all, _placed_at, _data_dir,
                                camera_matrix=_OV_K, cam_to_base=_OV_CTB,
                            )
                        else:
                            _ann_base = _annotate_handled_objects(
                                det_img_all, _placed_at, _data_dir,
                            )
                        _ann = _PM.draw_detections(_ann_base, _dino_detections)
                        _dino_path = _RUN_DIR / f"iter_{iteration+1:02d}_dino.png"
                        _ann.save(str(_dino_path))
                        print(f"[LOOP] DINO annotation saved: {_dino_path.name}")
                        _publish_dino_annotated_image(
                            args, _dino_path, src_label_all
                        )
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
            # print("\n  PDDL PROBLEM:")
            # for line in pddl_str.splitlines():
            #     print(f"    {line}")
            # print()
        except Exception as _pe:
            pddl_str = f"# generation failed: {_pe}"

        # Save per-iteration debug package to run directory
        _iter_n = iteration + 1
        _state_before_step = _capture_robot_state(args)
        _debug_monotonic = time.monotonic()
        _debug_timestamp = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="milliseconds")
        _iter_dir = _RUN_DIR / f"iter_{_iter_n:02d}"
        try:
            import json as _dbg_json

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
                "debug_schema_version": 2,
                "run_id": _RUN_DIR.name,
                "iteration":       _iter_n,
                "task":            args.task,
                "world":           _world_tag,
                "environment": {
                    "mode": "real_robot" if args.real_ros2 else "simulation",
                    "gazebo_world": None if args.real_ros2 else args.world,
                    "robot": "fr3" if args.real_ros2 else "panda",
                    "execute": bool(args.execute),
                },
                "timestamp": _debug_timestamp,
                "loop_started_at": _loop_started_at,
                "iteration_started_at": _iteration_started_at,
                "timing": {
                    "loop_elapsed_s": round(
                        _debug_monotonic - _loop_started_monotonic, 3
                    ),
                    "iteration_elapsed_s": round(
                        _debug_monotonic - _iteration_started_monotonic, 3
                    ),
                    "capture_s": round(_capture_time_s, 3),
                    "vlm_s": round(vlm_time, 3),
                },
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
                "detected_object_poses": dict(_last_dino_pose),
                "placed_at":       {k: list(v) for k, v in _placed_at.items()},
                "using_overview_cam": _using_overview,
                "robot_state_before_step": _state_before_step,
                "loop_state_before_step": {
                    "holding_object": holding,
                    "held_object_name": _held_object,
                    "completed_step_count": len(completed_steps),
                    "replan_count": _replan_count,
                    "using_cached_plan": _use_cached_plan,
                    "remaining_plan_step_count": len(
                        _current_plan.steps if _current_plan else []
                    ),
                    "last_dispatch_seq": _last_seq,
                },
                "camera_state": {
                    "primary_perception_camera": (
                        "overview" if _using_overview else "wrist"
                    ),
                    "overview_available": bool(_using_overview),
                    "wrist_snapshot": str(image_path),
                    "overview_snapshot": str(_ov_path) if _ov_path.exists() else None,
                    "overview_image_topic": args.overview_image_topic,
                    "wrist_image_topic": args.wrist_image_topic,
                    "overview_recording_enabled": bool(
                        args.record_overview_video
                    ),
                    "overview_video_topic": args.overview_video_topic,
                    "overview_video_fps": args.overview_video_fps,
                    "overview_video_file": "overview_camera.mp4",
                },
                "scene_state": {
                    "known_models": gazebo_models,
                    "detected_object_count": len(_last_dino_pose),
                    "placed_object_count": len(_placed_at),
                },
                "execution": {
                    "attempted": False,
                    "result": None,
                },
            }
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
            _ov_src = (
                args.ros2_capture_dir / "overview.png"
                if args.real_ros2
                else _REPO_ROOT / "data" / "scene_overview.png"
            )
            if _ov_src.exists():
                _shutil.copy2(str(_ov_src), str(_iter_dir / "overview.png"))

        except Exception as _save_err:
            print(f"[WARN] Debug save failed: {_save_err}")

        if not args.execute:
            print(
                "[LOOP] --no-execute: plan and perception saved; "
                "plan injection skipped"
            )
            break

        if args.confirm_actions:
            action = plan_grounded.steps[0]
            action_args = ", ".join(
                f"{key}={value!r}" for key, value in action.args.items()
            )
            action_text = f"{action.primitive}({action_args})"
            print(f"\n[CONFIRM] Next robot action: {action_text}")
            try:
                answer = input("[CONFIRM] Execute this action? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                answer = ""
            if answer not in {"y", "yes"}:
                print("[LOOP] Action cancelled by operator; plan was not injected.")
                _update_iteration_debug(_iter_dir, {
                    "execution": {
                        "attempted": False,
                        "cancelled_by_operator": True,
                        "result": None,
                    }
                })
                break

        # 6. Serialize + inject.
        # Full PDDL pipeline (no direct flag): orchestrator runs FastDownward to
        # validate the single-step plan before dispatch.  The problem_generator
        # infers the live robot state from the VLM plan structure:
        #   - pick steps   → object starts on a surface
        #   - place steps without prior pick → arm is already holding the object
        #   - pour/tilt steps without prior pick → arm is already holding the source
        # This makes single-step validation correct for all mid-task states.
        request_id = str(uuid.uuid4())
        _dispatch_started_monotonic = time.monotonic()
        _dispatch_started_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="milliseconds")
        payload = json.dumps({
            "request_id": request_id,
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
            _inject_error = inject_result.stderr.decode().strip()
            print(f"[FAIL] Injection failed: {_inject_error}")
            _update_iteration_debug(_iter_dir, {
                "execution": {
                    "attempted": True,
                    "request_id": request_id,
                    "dispatch_started_at": _dispatch_started_at,
                    "dispatch_loop_elapsed_s": round(
                        _dispatch_started_monotonic - _loop_started_monotonic, 3
                    ),
                    "injected": False,
                    "error": _inject_error,
                    "result": None,
                }
            })
            break
        print(f"[OK]   Step injected.")

        # 7. Wait for step completion
        print("[LOOP] Attendo completamento step...")
        result = _wait_step_complete(
            args, timeout=60, min_seq=_last_seq + 1,
            request_id=request_id,
        )
        _step_completed_monotonic = time.monotonic()
        _step_completed_at = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="milliseconds")
        _state_after_step = _capture_robot_state(args)
        _update_iteration_debug(_iter_dir, {
            "execution": {
                "attempted": True,
                "request_id": request_id,
                "dispatch_started_at": _dispatch_started_at,
                "completed_at": _step_completed_at,
                "dispatch_loop_elapsed_s": round(
                    _dispatch_started_monotonic - _loop_started_monotonic, 3
                ),
                "completion_loop_elapsed_s": round(
                    _step_completed_monotonic - _loop_started_monotonic, 3
                ),
                "duration_s": round(
                    _step_completed_monotonic - _dispatch_started_monotonic, 3
                ),
                "injected": True,
                "result": result,
                "robot_state_after_step": _state_after_step,
            }
        })
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
            _holding_after_step = (
                True if s0_grnd.primitive == "pick"
                else False if s0_grnd.primitive in ("place", "stack")
                else holding
            )
            _held_after_step = (
                obj_orig if s0_grnd.primitive == "pick"
                else None if s0_grnd.primitive in ("place", "stack")
                else _held_object
            )
            _update_iteration_debug(_iter_dir, {
                "completed_steps": list(completed_steps),
                "loop_state_after_step": {
                    "holding_object": _holding_after_step,
                    "held_object_name": _held_after_step,
                    "completed_step_count": len(completed_steps),
                    "replan_count": _replan_count,
                    "last_dispatch_seq": _last_seq,
                },
            })

            # Track place destinations for annotation markers
            if s0_orig.primitive == "place":
                obj_placed = s0_orig.args.get("object", "")
                loc_placed = s0_grnd.args.get("location", "")
                loc_original = s0_orig.args.get("location", "")
                placed_position = _placed_object_position(
                    loc_original, loc_placed, _last_dino_est,
                    _last_dino_pose, raw_gazebo_poses,
                )
                if obj_placed and placed_position is not None:
                    px, py = placed_position[:2]
                    _placed_at[obj_placed] = placed_position
                    print(f"[LOOP] Annotation: '{obj_placed}' placed at "
                          f"({px:.2f},{py:.2f}) → ✓ marker added to future images")
                    # Update the ROS/RViz image now as well. Otherwise a final
                    # place exits before another iteration can publish the mark.
                    try:
                        if _using_overview and _OV_CTB is not None:
                            placed_image = _annotate_handled_objects(
                                image_vlm, _placed_at, _data_dir_annot,
                                camera_matrix=_OV_K, cam_to_base=_OV_CTB,
                            )
                            placed_source = "overview"
                        else:
                            placed_image = _annotate_handled_objects(
                                image, _placed_at, _data_dir_annot,
                            )
                            placed_source = "wrist"
                        placed_path = (
                            _RUN_DIR / f"iter_{iteration+1:02d}_placed.png"
                        )
                        placed_image.save(str(placed_path))
                        _publish_dino_annotated_image(
                            args, placed_path, placed_source,
                        )
                    except Exception as exc:
                        print(f"[WARN] Post-place annotation failed: {exc}")
                elif obj_placed:
                    print(
                        f"[WARN] Annotation: no pose for place destination "
                        f"'{loc_placed or loc_original}'; cannot mark '{obj_placed}'"
                    )

            if args.replan_on_failure_only:
                # The first cached step is exactly the action just completed.
                # Consume it locally instead of asking the VLM to regenerate
                # the remaining plan on the next iteration.
                if _current_plan is not None and _current_plan.steps:
                    _current_plan.steps.pop(0)
                if not _current_plan or not _current_plan.steps:
                    print("\n[LOOP] ✅ Piano VLM completato con successo!")
                    break
                print(
                    f"[LOOP] Avanzo al prossimo step in cache "
                    f"({len(_current_plan.steps)} rimanenti)"
                )
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

    # Capture overview camera at end of experiment
    _capture_overview_image(args, _RUN_DIR, "end")

    for _video_recording in _video_recordings:
        _video_recording.stop()
        atexit.unregister(_video_recording.stop)
    # Save completed steps to run directory
    with open(str(_RUN_DIR / "run_info.txt"), "a") as _rf:
        _rf.write(f"steps:     {completed_steps}\n")
        _rf.write(f"n_steps:   {len(completed_steps)}\n")
        if (_RUN_DIR / "experiment_webcam.mp4").exists():
            _rf.write("video:     experiment_webcam.mp4\n")
        if (_RUN_DIR / "overview_camera.mp4").exists():
            _rf.write("overview_video: overview_camera.mp4\n")
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
