#!/usr/bin/env python3
"""
run_vlm_host.py — HOST-SIDE script: VLM inference → plan injection.

Runs on the Ubuntu host (GPU available). Performs VLM inference with
Qwen3-VL-8B-Instruct, then pipes the resulting VLMPlan JSON into the
Docker container via stdin so the ROS 2 Orchestrator can execute it.

Prerequisites:
  - .venv activated (pip install -r requirements.txt)
  - Docker container 'vlm_ros2' running with the simulation launched

Usage:
    source .venv/bin/activate
    python3 scripts/run_vlm_host.py \\
        --task "pick the red cup and place it on the shelf" \\
        --image path/to/scene.jpg

    # Dry-run: show plan without publishing (no Docker required)
    python3 scripts/run_vlm_host.py --task "..." --dry-run

    # Synthetic image (for testing without a real scene)
    python3 scripts/run_vlm_host.py --task "..." --synthetic

    # Capture scene from Gazebo camera and run VLM (full end-to-end)
    python3 scripts/run_vlm_host.py \\
        --task "pick the red cup and place it on the shelf" \\
        --capture \\
        --container vlm_ros2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _make_synthetic_image():
    """Return a minimal PIL Image for testing (no real camera needed)."""
    from PIL import Image as PilImage
    import numpy as np
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    return PilImage.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VLM on host and inject plan into ROS 2 container."
    )
    parser.add_argument("--task", required=True, help="Natural language task")
    parser.add_argument(
        "--image", nargs="*", default=[], metavar="PATH",
        help="Scene image file paths (can specify multiple)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use a synthetic noise image instead of a real scene image"
    )
    parser.add_argument(
        "--no-vlm", action="store_true",
        help="Skip VLM inference; inject a hardcoded pick+place plan (for testing injection only)"
    )
    parser.add_argument(
        "--capture", action="store_true",
        help="Capture one frame from /wrist_camera/image_raw inside the container "
             "before running VLM (saved to data/scene.png)"
    )
    parser.add_argument(
        "--container", default="vlm_ros2",
        help="Docker container name (default: vlm_ros2)"
    )
    parser.add_argument(
        "--sudo-docker", action="store_true",
        help="Prepend sudo to docker commands (use if not in docker group)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print JSON plan without publishing to the container"
    )
    parser.add_argument(
        "--use-bbox-grounding", action="store_true",
        help="Use VLM bbox + camera projection for grounding (Phase 2 / real robot). "
             "Disabled by default in simulation (overview camera perspective too unusual)."
    )
    parser.add_argument(
        "--items", nargs="*", default=[], metavar="NAME",
        help="Known PDDL item names passed to VLM (e.g. --items red_cup blue_box)"
    )
    parser.add_argument(
        "--locations", nargs="*", default=[], metavar="NAME",
        help="Known PDDL location names passed to VLM (e.g. --locations shelf_b)"
    )
    args = parser.parse_args()

    # ── Capture scene from Gazebo camera ─────────────────────────────────────
    _SCENE_PATH = _REPO_ROOT / "data" / "scene.png"

    _docker = ["sudo", "docker"] if args.sudo_docker else ["docker"]

    if args.capture:
        print("[INFO] Capturing scene from /wrist_camera/image_raw…")
        bash_cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/ros2_ws/install/setup.bash && "
            "python3 /workspace/scripts/_capture_scene.py"
        )
        cap_result = subprocess.run(
            _docker + ["exec", args.container, "bash", "-c", bash_cmd],
            capture_output=True,
        )
        output = cap_result.stdout.decode().strip()
        if cap_result.returncode != 0:
            err = cap_result.stderr.decode().strip()
            print(f"[FAIL] Capture failed: {err}")
            sys.exit(1)
        print(f"[OK]   {output}")
        if not args.image:
            args.image = [str(_SCENE_PATH)]

    # ── Build plan ────────────────────────────────────────────────────────────
    from vlm.planner import VLMPlan, PlanStep

    if args.no_vlm:
        # Hardcoded plan for testing the injection mechanism without GPU
        print("[INFO] --no-vlm: using hardcoded pick+place plan.")
        plan = VLMPlan(
            goal=args.task,
            steps=[
                PlanStep(primitive="pick",  args={"object": "red_cup"}),
                PlanStep(primitive="place", args={"object": "red_cup", "location": "shelf_b"}),
            ],
            raw_output="[mock — no VLM inference]",
            domain_template="manipulation_base",
        )
    else:
        # ── Load images ───────────────────────────────────────────────────────
        images = []
        if args.synthetic:
            print("[INFO] Using synthetic image.")
            images = [_make_synthetic_image()]
        elif args.image:
            from PIL import Image as PilImage
            for p in args.image:
                img = PilImage.open(p).convert("RGB")
                images.append(img)
                print(f"[INFO] Loaded image: {p} ({img.size[0]}×{img.size[1]})")
        else:
            print("[WARN] No image provided — VLM will reason without visual context.")

        # ── VLM inference ─────────────────────────────────────────────────────
        print(f"\n[INFO] Loading VLM (Qwen3-VL-8B-Instruct)…")
        from vlm.planner import VLMPlanner
        vlm = VLMPlanner()
        vlm.load()
        print("[OK]   VLM loaded.\n")

        # ── Discover scene objects from Gazebo ────────────────────────────────
        # Query /gazebo/model_states for scene object names and positions.
        # Used as a candidate list for visual name grounding (OWL-ViT / DINO).
        gazebo_models: list[str]       = []   # names only (for OWL-ViT fallback)
        gazebo_poses:  dict[str, dict] = {}   # name → {x,y,z} (for bbox grounding)
        if args.capture or not args.no_vlm:
            bash_cmd = (
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/ros2_ws/install/setup.bash && "
                "python3 /workspace/scripts/_get_model_states.py"
            )
            ms_result = subprocess.run(
                _docker + ["exec", args.container, "bash", "-c", bash_cmd],
                capture_output=True,
            )
            if ms_result.returncode == 0:
                import json as _json
                ms_data      = _json.loads(ms_result.stdout.decode().strip())
                gazebo_poses = ms_data.get("models", {})
                gazebo_models = list(gazebo_poses.keys())
                print(f"[INFO] Gazebo scene objects: {gazebo_models}")
            else:
                print("[WARN] Could not query Gazebo model states — grounding disabled.")

        # ── VLM inference: no vocabulary hints (fully adaptive) ───────────────
        # The VLM reasons from the image alone. Names are corrected post-inference
        # via PerceptionModule grounding against Gazebo scene objects.
        # The hint-based mode is still available via --items/--locations for ablation.
        scene_context = None
        if args.items or args.locations:
            scene_context = {"items": args.items, "locations": args.locations}
            print(f"[INFO] Scene context (Modalità A): items={args.items}, locations={args.locations}")
        else:
            print(f"[INFO] Scene context: none (Modalità B — VLM reasons freely)")

        print(f"[INFO] Running inference for: '{args.task}'")
        plan = vlm.plan(args.task, images, scene_context=scene_context)

        # ── Visual grounding: name-based matching ─────────────────────────────
        # Match VLM-generated object names to known scene objects using OWL-ViT.
        from vlm.perception import PerceptionModule
        perception = PerceptionModule()

        grounded = False
        if not grounded:
            vocab = args.items or gazebo_models
            locs  = args.locations if args.locations else gazebo_models
            if images and (vocab or locs):
                perception.load()
                plan = perception.ground_names(
                    plan, images[0],
                    known_items=vocab,
                    known_locations=locs,
                )

    # ── Verbose plan summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  PIANO VLM — '{plan.goal}'")
    print("=" * 60)
    print(f"  Domain template : {plan.domain_template}")
    if plan.domain_additions.get("new_predicates") or \
       plan.domain_additions.get("new_actions"):
        print(f"  Domain additions: {plan.domain_additions}")
    print(f"  Steps ({len(plan.steps)}):")
    for i, step in enumerate(plan.steps, 1):
        args_str = ", ".join(f"{k}={v}" for k, v in step.args.items())
        print(f"    {i}. {step.primitive}({args_str})")
    print("=" * 60)

    # ── PDDL problem preview (from pipeline) ─────────────────────────────────
    if not args.no_vlm and not args.dry_run:
        try:
            from planner.problem_generator import generate_problem
            pddl_problem = generate_problem(plan)
            print("\n  PDDL PROBLEM generato:")
            print("  " + "\n  ".join(pddl_problem.splitlines()))
            print()
        except Exception as _e:
            pass  # non-fatal: PDDL generation requires the planner module

    # VLM bboxes are no longer used — no bbox-based debug images are generated

    # ── Serialize ─────────────────────────────────────────────────────────────
    payload = json.dumps({
        "command":  args.task,
        "vlm_plan": json.loads(plan.to_json()),
    })
    print(f"\n[INFO] Serialized plan: {len(payload)} chars")

    if args.dry_run:
        print("\n[DRY-RUN] Plan JSON (not published):")
        print(payload)
        return

    # ── Inject via docker exec ────────────────────────────────────────────────
    # The container has /opt/ros/humble and /workspace/ros2_ws sourced via
    # entrypoint.sh; we source them explicitly here for robustness.
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_publish_plan.py"
    )
    cmd = _docker + ["exec", "-i", args.container, "bash", "-c", bash_cmd]

    print(f"[INFO] Injecting into container '{args.container}'…")
    result = subprocess.run(
        cmd,
        input=payload.encode(),
        capture_output=True,
    )

    if result.returncode == 0:
        output = result.stdout.decode().strip()
        print(f"[OK]   {output}")
    else:
        stderr = result.stderr.decode().strip()
        stdout = result.stdout.decode().strip()
        print(f"[FAIL] docker exec returned {result.returncode}")
        if stdout:
            print(f"       stdout: {stdout}")
        if stderr:
            print(f"       stderr: {stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
