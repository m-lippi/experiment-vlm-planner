#!/usr/bin/env python3
"""
debug_owlvit.py — Diagnostic for OWL-ViT detection on the last captured scene.

Runs on the host (GPU). Shows raw detection scores for multiple text queries
so we can tune thresholds and query phrasing for the wrist-camera perspective.

Usage:
    source .venv/bin/activate
    python3 scripts/debug_owlvit.py                          # uses data/scene.png
    python3 scripts/debug_owlvit.py --image data/loop_iter_01.png
    python3 scripts/debug_owlvit.py --threshold 0.0          # show ALL detections
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _draw_boxes(image, detections: dict, out_path: Path) -> None:
    from PIL import ImageDraw, ImageFont
    img = image.copy()
    draw = ImageDraw.Draw(img)
    colors = ["red", "blue", "green", "orange", "purple", "cyan", "yellow"]
    for ci, (query, boxes) in enumerate(detections.items()):
        color = colors[ci % len(colors)]
        for x0, y0, x1, y1, score in boxes:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0, max(0, y0 - 14)), f"{query[:12]} {score:.2f}", fill=color)
    img.save(out_path)
    print(f"[OK] Annotated image saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/scene.png")
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Show all detections at or above this score (0.0 = show everything)"
    )
    args = parser.parse_args()

    img_path = _REPO / args.image
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    from PIL import Image as PilImage
    image = PilImage.open(img_path).convert("RGB")
    print(f"[INFO] Image: {img_path} ({image.width}×{image.height})")

    # Candidate queries — covers both simulation and real-world phrasing
    queries = [
        # Specific (good for real robot)
        "red cylindrical cup",
        "small blue cube",
        "small green platform",
        # Generic fallbacks
        "red cup",
        "blue box",
        "green shelf",
        "cup",
        "cube",
        "box",
        "cylinder",
        "shelf",
        "platform",
    ]

    print("[INFO] Loading OWL-ViT …")
    from vlm.perception import PerceptionModule
    pm = PerceptionModule()
    pm.load()
    print("[OK]   Model loaded.\n")

    # ── Get raw sigmoid scores directly (bypass threshold) ───────────────────
    import torch
    print(f"{'Query':<20} {'Max score':>10}  {'p50 score':>10}  {'# > 0.01':>10}  {'# > 0.05':>10}")
    print("─" * 70)

    best_boxes: dict[str, tuple] = {}   # query → (score, x0,y0,x1,y1) for TOP-1

    processor = pm._processor
    model     = pm._model
    device    = pm._device

    for q in queries:
        readable = q.replace("_", " ")
        inputs = processor(text=[[readable]], images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits    = outputs.logits[0].cpu()       # [patches, 1]
        pred_boxes = outputs.pred_boxes[0].cpu()  # [patches, 4]  cx,cy,w,h normalised
        scores    = torch.sigmoid(logits[:, 0])   # [patches]
        W, H      = float(image.width), float(image.height)

        max_s  = scores.max().item()
        med_s  = scores.median().item()
        n_01   = (scores >= 0.01).sum().item()
        n_05   = (scores >= 0.05).sum().item()

        # TOP-1 box
        top_idx = scores.argmax().item()
        cx, cy, w, h = pred_boxes[top_idx].tolist()
        x0, y0 = (cx-w/2)*W, (cy-h/2)*H
        x1, y1 = (cx+w/2)*W, (cy+h/2)*H
        best_boxes[q] = (max_s, x0, y0, x1, y1)

        print(f"{q:<20} {max_s:>10.4f}  {med_s:>10.4f}  {n_01:>10}  {n_05:>10}")

    # ── Show top-5 best queries by max score ──────────────────────────────────
    print("\n── TOP-5 queries by max score ──")
    sorted_q = sorted(best_boxes.items(), key=lambda x: x[1][0], reverse=True)
    for q, (s, x0, y0, x1, y1) in sorted_q[:5]:
        print(f"  {q:<20} score={s:.4f}  box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    print(f"\n── Suggestion ──")
    top_score = sorted_q[0][1][0]
    if top_score < 0.01:
        print("  All scores < 0.01: objects likely NOT VISIBLE in this image.")
        print("  → Check data/scene.png — does the wrist camera actually see the table?")
    elif top_score < 0.05:
        print(f"  Max score = {top_score:.4f} → below current threshold (0.05).")
        print("  → Lowering DETECTION_THRESHOLD to ~0.01 or using TOP-1 strategy would work.")
    else:
        print(f"  Max score = {top_score:.4f} → detection should work at current threshold.")

    # ── Save annotated output with TOP-1 per query ───────────────────────────
    out = _REPO / "data" / "debug_owlvit.png"
    top1_dets = {q: [(x0, y0, x1, y1, s)] for q, (s, x0, y0, x1, y1) in sorted_q[:6]}
    _draw_boxes(image, top1_dets, out)


if __name__ == "__main__":
    main()
