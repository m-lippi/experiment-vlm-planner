#!/usr/bin/env python3
"""
validate_depth_pose.py — Validate depth-based 3D pose estimation off-line.

Runs GroundingDINO on a captured image + depth file and shows:
  - detected bounding box
  - estimated 3D position in panda_link0 frame
  - back-projection of the 3D point onto the image (sanity check)
  - comparison with ray-plane fallback (fixed z) to quantify the difference

Usage:
    # From the wrist camera (uses data/scene.png + data/depth.npy)
    python scripts/validate_depth_pose.py --object "pen"

    # From the overview camera
    python scripts/validate_depth_pose.py --object "pen" --camera overview

    # Custom files
    python scripts/validate_depth_pose.py --object "pen" \\
        --image data/scene.png --depth data/depth.npy \\
        --camera-info data/camera_info.json --camera-pose data/camera_pose.json

    # Dry-run without GPU (prints geometry only, skips DINO)
    python scripts/validate_depth_pose.py --object "pen" --no-dino \\
        --bbox 280 210 340 270
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    ap = argparse.ArgumentParser(description="Validate depth-based 3D pose estimation")
    ap.add_argument("--object",      default="pen",        help="Object name / DINO query")
    ap.add_argument("--camera",      default="wrist",      choices=["wrist", "overview"],
                    help="Which camera data to use (default: wrist)")
    ap.add_argument("--image",       default=None,         help="Path to RGB image (overrides --camera default)")
    ap.add_argument("--depth",       default=None,         help="Path to depth .npy file (overrides --camera default)")
    ap.add_argument("--camera-info", default=None,         help="Path to camera_info.json")
    ap.add_argument("--camera-pose", default=None,         help="Path to camera_pose.json")
    ap.add_argument("--world",       default="office",     help="World name (for overview cam pose, sim only)")
    ap.add_argument("--z-base",      type=float, default=0.025,
                    help="Fixed table height above panda_link0 for ray-plane fallback (m)")
    ap.add_argument("--no-dino",     action="store_true",  help="Skip DINO, use --bbox instead")
    ap.add_argument("--bbox",        type=float, nargs=4,  metavar=("X0","Y0","X1","Y1"),
                    help="Manual bounding box pixels (with --no-dino)")
    ap.add_argument("--out",         default=None,         help="Save annotated image to this path")
    ap.add_argument("--threshold",   type=float, default=0.25, help="DINO detection threshold")
    return ap.parse_args()


# ── Camera data loading ───────────────────────────────────────────────────────

def _load_wrist_data(args):
    data = _REPO_ROOT / "data"
    img_path   = Path(args.image)   if args.image       else data / "scene.png"
    depth_path = Path(args.depth)   if args.depth       else data / "depth.npy"
    info_path  = Path(args.camera_info) if args.camera_info else data / "camera_info.json"
    pose_path  = Path(args.camera_pose) if args.camera_pose else data / "camera_pose.json"

    for p, name in [(img_path, "image"), (info_path, "camera_info")]:
        if not p.exists():
            sys.exit(f"[ERROR] {name} not found: {p}\n"
                     "  Run the simulation and capture a frame first.")

    with open(info_path) as f:
        K = np.array(json.load(f)["K"])
    with open(pose_path) as f:
        cam_to_base = np.array(json.load(f)["cam_to_base"])

    depth = np.load(str(depth_path)) if depth_path.exists() else None
    if depth is None:
        print(f"[WARN] Depth file not found: {depth_path} — will use ray-plane fallback only")

    return Image.open(img_path).convert("RGB"), depth, K, cam_to_base


def _load_overview_data(args):
    import math
    data = _REPO_ROOT / "data"
    img_path   = Path(args.image) if args.image else data / "scene_overview.png"
    depth_path = Path(args.depth) if args.depth else data / "depth_overview.npy"

    if not img_path.exists():
        sys.exit(f"[ERROR] Overview image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    depth = np.load(str(depth_path)) if depth_path.exists() else None
    if depth is None:
        print(f"[WARN] Overview depth not found: {depth_path} — will use ray-plane fallback only")

    # K: prefer camera_info file, else compute from FOV
    info_path = data / "overview_camera_info.json"
    if info_path.exists():
        with open(info_path) as f:
            K = np.array(json.load(f)["K"])
        print(f"[INFO] Overview K from camera_info: fx={K[0,0]:.1f}")
    else:
        W, H, fov = img.width, img.height, 1.047
        fx = fy = W / (2.0 * math.tan(fov / 2.0))
        K = np.array([[fx, 0, W/2.0], [0, fy, H/2.0], [0, 0, 1.0]])
        print(f"[INFO] Overview K from FOV: fx={K[0,0]:.1f}")

    # cam_to_base: prefer JSON file, else compute from world SDF
    pose_path = data / "overview_camera_pose.json"
    if pose_path.exists():
        with open(pose_path) as f:
            cam_to_base = np.array(json.load(f)["cam_to_base"])
        print("[INFO] Overview cam_to_base from overview_camera_pose.json")
    else:
        # Sim path: compute from world file
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from run_loop_host import _get_overview_cam_data
        K_ov, cam_to_base = _get_overview_cam_data(args.world)
        if cam_to_base is None:
            sys.exit("[ERROR] Could not determine overview camera pose. "
                     "Create data/overview_camera_pose.json for real robot.")
        K = K_ov if K_ov is not None else K

    return img, depth, K, cam_to_base


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _median_depth(depth_image: np.ndarray, box: list[float],
                  shrink: float = 0.35) -> float | None:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
    rx0 = int(max(0, x0 + dx));  ry0 = int(max(0, y0 + dy))
    rx1 = int(min(depth_image.shape[1]-1, x1 - dx))
    ry1 = int(min(depth_image.shape[0]-1, y1 - dy))
    if rx1 <= rx0 or ry1 <= ry0:
        return None
    patch = depth_image[ry0:ry1, rx0:rx1].astype(np.float32)
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size >= 5 else None


def _unproject_depth(u, v, z_cam, K, cam_to_base):
    K_inv  = np.linalg.inv(K)
    p_cam  = K_inv @ np.array([u, v, 1.0]) * z_cam
    R, t   = cam_to_base[:3, :3], cam_to_base[:3, 3]
    return R @ p_cam + t


def _ray_plane(u, v, z_base, K, cam_to_base):
    K_inv  = np.linalg.inv(K)
    d_cam  = K_inv @ np.array([u, v, 1.0])
    R, t   = cam_to_base[:3, :3], cam_to_base[:3, 3]
    d_base = R @ d_cam;  d_base /= np.linalg.norm(d_base)
    if abs(d_base[2]) < 1e-9:
        return None
    lam = (z_base - t[2]) / d_base[2]
    if lam < 0:
        return None
    return t + lam * d_base


def _project_to_image(p_base, K, cam_to_base):
    """Back-project a 3D base-frame point to image pixel."""
    R, t   = cam_to_base[:3, :3], cam_to_base[:3, 3]
    base_to_cam = np.eye(4)
    base_to_cam[:3, :3] = R.T
    base_to_cam[:3,  3] = -R.T @ t
    p_cam = base_to_cam[:3, :3] @ p_base + base_to_cam[:3, 3]
    if p_cam[2] <= 0.01:
        return None
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return int(u), int(v)


# ── Annotation ────────────────────────────────────────────────────────────────

def _annotate(img, box, p_depth, p_rayplane, K, cam_to_base):
    out  = img.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.width, out.height

    # Bounding box
    draw.rectangle([box[0], box[1], box[2], box[3]], outline=(0, 220, 0), width=3)

    # Centre cross
    u = (box[0] + box[2]) / 2;  v = (box[1] + box[3]) / 2
    CS = 10
    draw.line([u-CS, v, u+CS, v], fill=(0, 220, 0), width=2)
    draw.line([u, v-CS, u, v+CS], fill=(0, 220, 0), width=2)

    # Depth-based back-projection (cyan dot)
    if p_depth is not None:
        bp = _project_to_image(p_depth, K, cam_to_base)
        if bp and 0 <= bp[0] < W and 0 <= bp[1] < H:
            r = 7
            draw.ellipse([bp[0]-r, bp[1]-r, bp[0]+r, bp[1]+r], fill=(0, 220, 255))
            draw.text((bp[0]+9, bp[1]-8), "depth", fill=(0, 220, 255))

    # Ray-plane back-projection (orange dot)
    if p_rayplane is not None:
        bp = _project_to_image(p_rayplane, K, cam_to_base)
        if bp and 0 <= bp[0] < W and 0 <= bp[1] < H:
            r = 7
            draw.ellipse([bp[0]-r, bp[1]-r, bp[0]+r, bp[1]+r], fill=(255, 160, 0))
            draw.text((bp[0]+9, bp[1]-8), "ray-plane", fill=(255, 160, 0))

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # Load camera data
    if args.camera == "wrist":
        img, depth, K, cam_to_base = _load_wrist_data(args)
    else:
        img, depth, K, cam_to_base = _load_overview_data(args)

    print(f"\n{'='*55}")
    print(f"  Depth Pose Validation — camera={args.camera}  object='{args.object}'")
    print(f"{'='*55}")
    print(f"  Image  : {img.width}×{img.height}")
    print(f"  Depth  : {'available' if depth is not None else 'NOT FOUND'}"
          + (f"  shape={depth.shape}  dtype={depth.dtype}" if depth is not None else ""))
    print(f"  K fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    t = cam_to_base[:3, 3]
    print(f"  cam origin in base frame: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")

    # Detect bounding box
    if args.no_dino:
        if args.bbox is None:
            sys.exit("[ERROR] --no-dino requires --bbox x0 y0 x1 y1")
        box = list(args.bbox)
        score = None
        print(f"\n  Bounding box (manual): {box}")
    else:
        print(f"\n  Running GroundingDINO (threshold={args.threshold})...")
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "IDEA-Research/grounding-dino-tiny"
        processor = AutoProcessor.from_pretrained(model_id)
        model     = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        model.eval()

        text = args.object.replace("_", " ").lower() + " ."
        inputs = processor(images=img, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        H, W = img.height, img.width
        res = processor.post_process_grounded_object_detection(
            out, inputs.input_ids,
            threshold=args.threshold,
            text_threshold=args.threshold * 0.8,
            target_sizes=[(H, W)],
        )[0]

        if len(res["boxes"]) == 0:
            print(f"  [FAIL] '{args.object}' not detected at threshold={args.threshold}")
            print("  Tip: lower --threshold or try a different description")
            sys.exit(1)

        best_idx = int(np.argmax([float(s) for s in res["scores"]]))
        box  = res["boxes"][best_idx].tolist()
        score = float(res["scores"][best_idx])
        print(f"  Detected: box={[f'{v:.0f}' for v in box]}  score={score:.3f}")

    u = (box[0] + box[2]) / 2.0
    v = (box[1] + box[3]) / 2.0
    print(f"  Bbox centre: ({u:.0f}, {v:.0f}) px")

    # Depth-based estimation
    p_depth = None
    if depth is not None:
        z_cam = _median_depth(depth, box)
        if z_cam is not None:
            p_depth = _unproject_depth(u, v, z_cam, K, cam_to_base)
            print(f"\n  [DEPTH]     z_camera = {z_cam:.4f} m")
            print(f"              p_base   = ({p_depth[0]:.4f}, {p_depth[1]:.4f}, {p_depth[2]:.4f}) m")
        else:
            print("\n  [DEPTH]     insufficient valid depth pixels in bbox centre region")
    else:
        print("\n  [DEPTH]     skipped (no depth file)")

    # Ray-plane fallback
    p_ray = _ray_plane(u, v, args.z_base, K, cam_to_base)
    if p_ray is not None:
        print(f"  [RAY-PLANE] z_base   = {args.z_base:.4f} m (fixed assumption)")
        print(f"              p_base   = ({p_ray[0]:.4f}, {p_ray[1]:.4f}, {p_ray[2]:.4f}) m")

    # Difference
    if p_depth is not None and p_ray is not None:
        delta = np.linalg.norm(p_depth - p_ray)
        print(f"\n  Δ depth vs ray-plane: {delta*100:.1f} cm")
        if delta > 0.05:
            print("  [NOTE] >5 cm difference — depth-based is more accurate "
                  "(ray-plane assumes flat surface at z_base)")

    # Annotate and save
    ann = _annotate(img, box, p_depth, p_ray, K, cam_to_base)
    out_path = Path(args.out) if args.out else \
               _REPO_ROOT / "data" / f"validate_depth_{args.camera}_{args.object}.png"
    ann.save(str(out_path))
    print(f"\n  Annotated image saved: {out_path}")
    print(f"  Cyan dot  = depth-based back-projection (should overlap bbox centre)")
    print(f"  Orange dot = ray-plane back-projection  (should also overlap, may differ)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
