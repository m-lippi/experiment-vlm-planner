#!/usr/bin/env python3
"""
capture_and_plan.py — HOST-SIDE: capture from real RealSense D435i cameras
and test VLM plan generation without robot execution.

Two RealSense D435i cameras are expected:
  - Overview camera: fixed/external, sees the full workspace
  - Wrist camera:    on the gripper (or hand-held during tests)

Both cameras operate as the primary DINO source in the full pipeline:
the overview at its optimal depth range (0.8-1.5 m) with stable extrinsic
calibration. This script is for plan-generation validation only — no robot
motion, no Docker, no ROS.

Prerequisites:
    pip install pyrealsense2          # Intel librealsense Python bindings
    source .venv/bin/activate

Usage:
    # Auto-detect cameras (first device = overview, second = wrist)
    python3 scripts/capture_and_plan.py \\
        --task "pick the red cup and place it next to the pen"

    # List connected RealSense devices and exit
    python3 scripts/capture_and_plan.py --list

    # Specify serial numbers explicitly
    python3 scripts/capture_and_plan.py \\
        --task "pick the cup" \\
        --overview-serial 012345678901 \\
        --wrist-serial    098765432109

    # Only overview camera (no wrist)
    python3 scripts/capture_and_plan.py \\
        --task "pick the cup" \\
        --no-wrist

    # Use pre-saved images (no camera hardware needed)
    python3 scripts/capture_and_plan.py \\
        --task "pick the cup" \\
        --overview-image data/overview.jpg \\
        --wrist-image    data/wrist.jpg

    # Skip VLM inference (fast capture-only test)
    python3 scripts/capture_and_plan.py --task "..." --no-vlm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_DATA_DIR = _REPO_ROOT / "data"
_RUNS_DIR = _REPO_ROOT / "data" / "real_runs"

# RealSense capture: colour resolution and warmup
_RS_WIDTH     = 1280
_RS_HEIGHT    = 720
_RS_FPS       = 15   # 15 fps is more stable when two D435i share a USB controller
_WARMUP_FRAMES = 30   # ~1 s for auto-exposure to settle


# ── RealSense helpers ──────────────────────────────────────────────────────────

def _import_rs():
    try:
        import pyrealsense2 as rs
        return rs
    except ImportError:
        print(
            "[ERROR] pyrealsense2 not found.\n"
            "        Install: pip install pyrealsense2\n"
            "        Docs:    https://github.com/IntelRealSense/librealsense",
            file=sys.stderr,
        )
        sys.exit(1)


def list_devices() -> list[dict]:
    """Return info dicts for all connected RealSense devices."""
    rs  = _import_rs()
    ctx = rs.context()
    out = []
    for dev in ctx.devices:
        out.append({
            "serial": dev.get_info(rs.camera_info.serial_number),
            "name":   dev.get_info(rs.camera_info.name),
        })
    return out


def capture_frame(serial: str | None = None) -> dict | None:
    """
    Capture one aligned RGB + depth frame from a RealSense D435i.

    Opens the pipeline exclusively — call sequentially for each camera.
    Returns {"rgb": PIL Image, "depth_mm": np.ndarray uint16,
             "K": np.ndarray 3×3, "serial": str}  or None on failure.
    """
    import numpy as np
    from PIL import Image as PilImage

    rs       = _import_rs()
    pipeline = rs.pipeline()
    config   = rs.config()

    if serial:
        config.enable_device(serial)

    # Try preferred resolution; fall back to 640×480 if unsupported
    started = False
    for (w, h) in [(_RS_WIDTH, _RS_HEIGHT), (848, 480), (640, 480)]:
        try:
            config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, _RS_FPS)
            config.enable_stream(rs.stream.depth, w, h, rs.format.z16,  _RS_FPS)
            profile = pipeline.start(config)
            res_w, res_h = w, h
            started = True
            break
        except Exception:
            config.disable_all_streams()
            continue

    if not started:
        print(f"[WARN] Could not start RealSense pipeline (serial={serial})", file=sys.stderr)
        return None

    try:
        import time as _time
        align = rs.align(rs.stream.color)

        # Brief pause after start — lets the USB bus settle when capturing
        # two cameras sequentially on the same controller.
        _time.sleep(1.5)

        # Discard warmup frames so auto-exposure settles; skip on timeout
        for _ in range(_WARMUP_FRAMES):
            try:
                pipeline.wait_for_frames(timeout_ms=8000)
            except RuntimeError:
                pass

        frames      = pipeline.wait_for_frames(timeout_ms=10000)
        aligned     = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            print("[WARN] Empty frame received", file=sys.stderr)
            return None

        rgb_img   = PilImage.fromarray(
            np.asanyarray(color_frame.get_data()), "RGB"
        )
        depth_arr = np.asanyarray(depth_frame.get_data())   # uint16, mm

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        K = np.array([
            [intr.fx,  0.0,     intr.ppx],
            [0.0,      intr.fy, intr.ppy],
            [0.0,      0.0,     1.0     ],
        ])

        dev_serial = profile.get_device().get_info(rs.camera_info.serial_number)
        print(f"[OK]   Captured {res_w}×{res_h} from device {dev_serial}")
        return {"rgb": rgb_img, "depth_mm": depth_arr, "K": K, "serial": dev_serial}

    finally:
        pipeline.stop()


# ── Run folder helpers ─────────────────────────────────────────────────────────

def _make_run_dir(task: str, parent: Path | None = None) -> Path:
    import re
    from datetime import datetime
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower())[:40].strip("_")
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (parent or _RUNS_DIR) / f"{ts}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_camera_data(run_dir: Path, label: str, frame: dict) -> None:
    """Save RGB image and camera intrinsics K; note that cam_to_base
    (extrinsic) requires a separate calibration step with the robot."""
    img_path = run_dir / f"{label}.png"
    frame["rgb"].save(str(img_path))
    print(f"[OK]   {label}.png saved ({frame['rgb'].width}×{frame['rgb'].height})")

    k_path = run_dir / f"{label}_K.json"
    with open(k_path, "w") as f:
        json.dump({
            "serial": frame["serial"],
            "K": frame["K"].tolist(),
            "note": "cam_to_base not yet set — calibrate before robot execution",
        }, f, indent=2)
    print(f"[OK]   {label}_K.json saved (serial={frame['serial']})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture from RealSense D435i cameras and test VLM planning."
    )
    parser.add_argument("--task", default="",
        help="Natural language task description")
    parser.add_argument("--list", action="store_true",
        help="List connected RealSense devices and exit")

    cam_group = parser.add_argument_group("camera selection")
    cam_group.add_argument("--overview-serial", metavar="SN",
        help="Serial number of the overview (fixed) RealSense")
    cam_group.add_argument("--wrist-serial", metavar="SN",
        help="Serial number of the wrist (eye-in-hand) RealSense")
    cam_group.add_argument("--no-wrist", action="store_true",
        help="Only capture overview camera (no wrist camera)")

    img_group = parser.add_argument_group("pre-saved images (skip capture)")
    img_group.add_argument("--overview-image", metavar="PATH",
        help="Use this image as overview instead of capturing from camera")
    img_group.add_argument("--wrist-image", metavar="PATH",
        help="Use this image as wrist view instead of capturing from camera")

    parser.add_argument("--output-dir", "-o", metavar="DIR", default=None,
        help="Parent folder where the run directory is created "
             "(default: data/real_runs/)")
    parser.add_argument("--no-vlm", action="store_true",
        help="Skip VLM inference (capture only)")
    parser.add_argument("--skip-pddl", action="store_true",
        help="Skip PDDL problem generation")
    args = parser.parse_args()

    # ── List devices ──────────────────────────────────────────────────────────
    if args.list:
        devices = list_devices()
        if not devices:
            print("No RealSense devices found.")
        else:
            print(f"Found {len(devices)} RealSense device(s):")
            for i, d in enumerate(devices):
                print(f"  [{i}] serial={d['serial']}  name={d['name']}")
        return

    if not args.task and not args.no_vlm:
        parser.error("--task is required (or use --no-vlm for capture-only)")

    # ── Determine capture strategy ────────────────────────────────────────────
    devices = list_devices() if not (args.overview_image and
                                     (args.wrist_image or args.no_wrist)) else []

    # Assign serial numbers automatically when not specified
    if not args.overview_serial and not args.overview_image:
        if not devices:
            print("[ERROR] No RealSense cameras found and no --overview-image provided.",
                  file=sys.stderr)
            sys.exit(1)
        args.overview_serial = devices[0]["serial"]
        print(f"[INFO] Auto-selected overview camera: {args.overview_serial}")

    if not args.wrist_serial and not args.wrist_image and not args.no_wrist:
        if len(devices) >= 2:
            args.wrist_serial = devices[1]["serial"]
            print(f"[INFO] Auto-selected wrist camera:    {args.wrist_serial}")
        else:
            print("[INFO] Only one RealSense found — running overview-only.")
            args.no_wrist = True

    # ── Capture / load images ─────────────────────────────────────────────────
    from PIL import Image as PilImage

    out_parent = Path(args.output_dir) if args.output_dir else None
    run_dir = _make_run_dir(args.task or "capture_test", parent=out_parent)
    print(f"[INFO] Run folder: {run_dir}")

    # Overview
    if args.overview_image:
        ov_rgb = PilImage.open(args.overview_image).convert("RGB")
        ov_K   = None
        print(f"[INFO] Loaded overview from {args.overview_image} ({ov_rgb.width}×{ov_rgb.height})")
        ov_rgb.save(str(run_dir / "overview.png"))
    else:
        print(f"[INFO] Capturing overview from serial {args.overview_serial}…")
        ov_frame = capture_frame(args.overview_serial)
        if ov_frame is None:
            print("[ERROR] Overview capture failed.", file=sys.stderr)
            sys.exit(1)
        _save_camera_data(run_dir, "overview", ov_frame)
        ov_rgb = ov_frame["rgb"]
        ov_K   = ov_frame["K"]

    # Wrist
    wrist_rgb = None
    if not args.no_wrist:
        if args.wrist_image:
            wrist_rgb = PilImage.open(args.wrist_image).convert("RGB")
            print(f"[INFO] Loaded wrist from {args.wrist_image} ({wrist_rgb.width}×{wrist_rgb.height})")
            wrist_rgb.save(str(run_dir / "wrist.png"))
        else:
            import time as _time
            _time.sleep(3.0)   # let USB bus fully release after first pipeline
            print(f"[INFO] Capturing wrist from serial {args.wrist_serial}…")
            wrist_frame = capture_frame(args.wrist_serial)
            if wrist_frame is None:
                print("[WARN] Wrist capture failed — continuing overview-only.")
            else:
                _save_camera_data(run_dir, "wrist", wrist_frame)
                wrist_rgb = wrist_frame["rgb"]

    images = [img for img in [ov_rgb, wrist_rgb] if img is not None]
    print(f"[INFO] Images ready: {len(images)} "
          f"({'overview + wrist' if wrist_rgb else 'overview only'})")

    if args.no_vlm:
        print("[INFO] --no-vlm: capture complete, skipping VLM.")
        return

    # ── VLM inference ─────────────────────────────────────────────────────────
    print(f"\n[INFO] Loading VLM (Qwen3-VL-8B-Instruct)…")
    from vlm.planner import VLMPlanner
    vlm = VLMPlanner()
    vlm.load()
    print("[OK]   VLM loaded.\n")

    print(f"[INFO] Running inference for: '{args.task}'")
    # Use plan_remaining (replanning prompt) — same path as the real loop,
    # which includes the few-shot enrichment examples. plan() uses the simpler
    # system_prompt.txt that does not instruct the VLM to produce domain_additions.
    plan = vlm.plan_remaining(args.task, images, completed_steps=[])

    # ── Plan summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print(f"  PIANO VLM — '{plan.goal}'")
    print("=" * 62)
    print(f"  Domain template : {plan.domain_template}")
    if plan.domain_additions.get("new_predicates") or \
       plan.domain_additions.get("new_actions"):
        print(f"  Domain additions: {plan.domain_additions}")
    print(f"  Steps ({len(plan.steps)}):")
    for i, step in enumerate(plan.steps, 1):
        args_str = ", ".join(f"{k}={v}" for k, v in step.args.items())
        print(f"    {i:2d}. {step.primitive}({args_str})")
    print("=" * 62)

    # ── PDDL problem generation ───────────────────────────────────────────────
    if not args.skip_pddl:
        try:
            from planner.problem_generator import generate_problem
            pddl_str = generate_problem(plan)
            print("\n  PDDL PROBLEM:")
            for line in pddl_str.splitlines():
                print(f"    {line}")
            print()
            pddl_path = run_dir / "problem.pddl"
            pddl_path.write_text(pddl_str)
            print(f"[OK]   PDDL saved: {pddl_path}")
        except Exception as e:
            print(f"[WARN] PDDL generation failed: {e}")

    # ── DINO detection on overview image ─────────────────────────────────────
    # Extract all object/target names from plan steps and run GroundingDINO
    # on the overview image. No cam_to_base needed — detection only, no pose.
    _OBJ_KEYS = ("object", "target", "location", "from", "to", "container")
    _INFRA     = {"table", "ground", "floor", "workspace"}
    names_to_detect = list(dict.fromkeys(
        v for step in plan.steps
        for k, v in step.args.items()
        if k in _OBJ_KEYS and isinstance(v, str) and v not in _INFRA
    ))

    if names_to_detect:
        print(f"\n[INFO] Running DINO detection on overview for: {names_to_detect}")
        try:
            from vlm.perception import PerceptionModule
            perception = PerceptionModule()
            perception.load()

            boxes = perception._detect(names_to_detect, ov_rgb)

            detections = []
            found, missing = [], []
            for name in names_to_detect:
                if boxes.get(name):
                    best = max(boxes[name], key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                    detections.append({"name": name, "box": best, "score": 1.0})
                    found.append(name)
                else:
                    missing.append(name)

            print(f"[OK]   Detected : {found}")
            if missing:
                print(f"[WARN] Not found: {missing}")

            if detections:
                ann_img = PerceptionModule.draw_detections(ov_rgb, detections)
                ann_path = run_dir / "overview_dino.png"
                ann_img.save(str(ann_path))
                print(f"[OK]   Annotated overview saved: {ann_path}")
        except Exception as e:
            print(f"[WARN] DINO detection failed: {e}")

    # ── Save plan JSON ────────────────────────────────────────────────────────
    plan_path = run_dir / "plan.json"
    plan_path.write_text(plan.to_json())
    print(f"[OK]   Plan JSON saved: {plan_path}")
    print(f"\n[INFO] All outputs in: {run_dir}")


if __name__ == "__main__":
    main()
