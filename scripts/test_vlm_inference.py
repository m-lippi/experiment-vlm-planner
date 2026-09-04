"""
Full VLM inference test — requires GPU (RTX 3090 Ti) and model weights.
Run this on the Linux lab machine after cloning the repo.

Usage:
    # With a real scene image:
    python scripts/test_vlm_inference.py --image tests/images/scene.jpg

    # With a synthetic image (no camera needed — validates the full inference stack):
    python scripts/test_vlm_inference.py --synthetic

    # With multiple images:
    python scripts/test_vlm_inference.py --image img1.jpg img2.jpg
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def make_synthetic_image(path: str) -> str:
    """Generate a minimal scene image (colored rectangles on a table background)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (640, 480), color=(180, 160, 140))  # table surface
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 150, 180, 300], fill=(200, 50, 50))   # red object
    draw.rectangle([300, 170, 380, 310], fill=(50, 50, 200))   # blue object
    img.save(path)
    print(f"[Synthetic image saved to {path}]")
    return path


def main():
    parser = argparse.ArgumentParser(description="Full VLM inference test (GPU required)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", nargs="+", metavar="IMG", help="Path(s) to scene image(s)")
    group.add_argument("--synthetic", action="store_true", help="Generate and use a synthetic scene")
    parser.add_argument("--task", default="pick the red object and place it next to the blue object")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--verbose", action="store_true", help="Print raw VLM output")
    args = parser.parse_args()

    if args.synthetic:
        synthetic_path = "/tmp/vlm_test_scene.jpg"
        image_paths = [make_synthetic_image(synthetic_path)]
    else:
        image_paths = args.image
        for p in image_paths:
            if not os.path.exists(p):
                print(f"[ERROR] Image not found: {p}")
                sys.exit(1)

    print(f"\nModel : {args.model}")
    print(f"Task  : {args.task}")
    print(f"Images: {image_paths}\n")

    from vlm.planner import VLMPlanner
    planner = VLMPlanner(model_name=args.model)

    print("[1/3] Loading model weights...")
    t0 = time.time()
    planner.load()
    print(f"      Done in {time.time() - t0:.1f}s\n")

    print("[2/3] Running inference...")
    t1 = time.time()
    plan = planner.plan(args.task, image_paths)
    elapsed = time.time() - t1
    print(f"      Done in {elapsed:.1f}s\n")

    print("[3/3] Results")
    print(f"  Goal : {plan.goal}")
    print(f"  Steps: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        print(f"    {i}. {step.primitive}({step.args})")

    if args.verbose:
        print("\n  Raw VLM output:")
        print("  " + "-" * 60)
        print(f"  {plan.raw_output}")
        print("  " + "-" * 60)

    if not plan.steps:
        print("\n[WARN] VLM returned no valid steps.")
        if not args.verbose:
            print(f"  Raw output: {plan.raw_output}")
        sys.exit(1)

    print("\n[OK] Inference test passed.")


if __name__ == "__main__":
    main()
