"""
Entry point for the planning pipeline.
Works standalone (no ROS needed) for testing VLM + PDDL generation + plan parsing.

Usage:
    # Dry-run (no GPU needed — tests the full pipeline with a mock VLM plan):
    python scripts/run_pipeline.py \
        --task "pick the red cup and place it on the shelf" \
        --images tests/images/scene.jpg \
        --dry-run

    # Real VLM inference (requires GPU + model weights):
    python scripts/run_pipeline.py \
        --task "pick the red cup and place it on the shelf" \
        --images tests/images/scene.jpg
"""

import argparse
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from planner.plan_parser import parse_plan
from planner.problem_generator import write_problem


def run_vlm_plan(task: str, image_paths: list[str]):
    from vlm.planner import VLMPlanner
    planner = VLMPlanner()
    print("[VLM] Loading model (first run ~30s)...")
    planner.load()
    return planner.plan(task, image_paths)


def mock_vlm_plan(task: str):
    from vlm.planner import VLMPlan, PlanStep
    return VLMPlan(
        goal=task,
        steps=[
            PlanStep(primitive="look_at", args={"object": "red_cup"}),
            PlanStep(primitive="pick",    args={"object": "red_cup"}),
            PlanStep(primitive="place",   args={"object": "red_cup", "location": "shelf"}),
        ],
        raw_output="[dry-run mock]",
    )


def main():
    parser = argparse.ArgumentParser(description="VLM Robot Planner — standalone test")
    parser.add_argument("--task", required=True)
    parser.add_argument("--images", nargs="+", required=True, metavar="IMG")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip VLM inference, use a mock plan")
    parser.add_argument("--skip-pddl", action="store_true",
                        help="Skip Fast Downward (trust VLM plan directly)")
    parser.add_argument("--domain", default="pddl/domain/manipulation.pddl")
    args = parser.parse_args()

    for p in args.images:
        if not os.path.exists(p):
            print(f"[ERROR] Image not found: {p}")
            sys.exit(1)

    print(f"\nTask  : {args.task}")
    print(f"Images: {args.images}\n")

    # ── Step 1: VLM planning ───────────────────────────────────────────────────
    if args.dry_run:
        print("[dry-run] Using mock VLM plan.")
        vlm_plan = mock_vlm_plan(args.task)
    else:
        vlm_plan = run_vlm_plan(args.task, args.images)

    print(f"[VLM] Goal : {vlm_plan.goal}")
    print("[VLM] Steps:")
    for i, s in enumerate(vlm_plan.steps, 1):
        print(f"  {i}. {s.primitive}({s.args})")

    if not vlm_plan.steps:
        print(f"[WARN] Empty plan. Raw VLM output:\n{vlm_plan.raw_output}")
        sys.exit(1)

    # ── Step 2: PDDL problem generation + symbolic validation ─────────────────
    if not args.skip_pddl:
        from planner.fast_downward import FastDownwardPlanner

        with tempfile.NamedTemporaryFile(suffix=".pddl", delete=False, mode="w") as f:
            problem_path = f.name

        write_problem(vlm_plan, problem_path, domain_name="manipulation")
        print(f"\n[PDDL] Problem written to: {problem_path}")

        print("[PDDL] Running Fast Downward...")
        pddl_plan = FastDownwardPlanner().solve(args.domain, problem_path)

        if pddl_plan is None:
            print("[ERROR] Symbolic planner found no valid plan.")
            sys.exit(1)

        primitives = parse_plan(pddl_plan)
        print(f"[PDDL] Validated plan ({len(primitives)} steps):")
    else:
        # Trust VLM plan directly (no symbolic validation)
        from planner.plan_parser import PrimitiveCall
        primitives = [
            PrimitiveCall(name=s.primitive, args=list(s.args.values()))
            for s in vlm_plan.steps
        ]
        print("\n[PDDL] Skipped (--skip-pddl). Executing VLM plan directly.")

    # ── Step 3: Primitive dispatch (mock) ──────────────────────────────────────
    print("\n[Executor] Dispatching primitives (mock):")
    for i, p in enumerate(primitives, 1):
        print(f"  {i}. → {p.name}({', '.join(str(a) for a in p.args)})")

    print("\n[Done] Pipeline completed (mock execution).")


if __name__ == "__main__":
    main()
