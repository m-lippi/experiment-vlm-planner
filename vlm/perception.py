"""
vlm/perception.py — Perception Module: visual grounding for object name resolution.

Phase 1 (simulation):
  Maps VLM-generated object names to known PDDL names using GroundingDINO open-vocabulary
  object detection.  Runs on the host GPU alongside the VLM.  The GazeboOracle still
  provides 3D poses — this module only corrects the names before PDDL planning.

Phase 2 (real robot — future):
  Extend get_pose() to use RealSense D435i depth instead of the oracle.  The
  grounding logic (GroundingDINO inference) is identical; only the 3D localization
  backend changes.  No other code needs to be modified.

The VLM generates object names from the task text ("blue cube", "cylinder", …)
which may differ from the PDDL names ("blue_box", "red_cup").  Without grounding,
the oracle lookup fails.  With grounding, names are corrected visually before
reaching the PDDL pipeline — regardless of how the VLM describes the object.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from PIL import Image

from vlm.planner import PlanStep, VLMPlan


# ── Camera calibration (overview_camera, from tabletop.world SDF) ────────────
# These are constants for Phase 1 simulation.  Phase 2 (real robot) reads
# calibration from the RealSense factory calibration and hand-eye procedure.

_CAM_POS  = np.array([1.0, 0.7, 1.5])        # position in world frame (m)
_CAM_RPY  = (0.0, 0.68, -2.19)               # roll, pitch, yaw (rad) from SDF
_IMG_W, _IMG_H = 640, 480                     # image resolution (px)
_FOV_H    = 1.047                             # horizontal FOV (rad)


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """SDF-convention RPY -> 3×3 rotation matrix (R = Rz @ Ry @ Rx)."""
    Rx = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],
                   [0,math.sin(roll),math.cos(roll)]])
    Ry = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],
                   [-math.sin(pitch),0,math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw),-math.sin(yaw),0],
                   [math.sin(yaw),math.cos(yaw),0],[0,0,1]])
    return Rz @ Ry @ Rx


def _build_camera_matrix() -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Build the world-to-OpenCV-camera projection components.

    In Gazebo, the camera link's +X axis is the optical axis.
    OpenCV/ROS convention: +Z forward, +X right, +Y down.
    The transform from Gazebo-link to OpenCV camera frame is:
        R_C_G = [[0,-1,0],[0,0,-1],[1,0,0]]

    Returns: (R, t, fx, fy, cx, cy)
      R, t  — 3×3 rotation and 3-vector translation for world->camera projection
      fx,fy — focal lengths (px)
      cx,cy — principal point (px)
    """
    # Rotation: world -> Gazebo link -> OpenCV camera
    R_W_G  = _rpy_to_matrix(*_CAM_RPY)   # camera frame axes as world columns
    R_G_W  = R_W_G.T                      # world -> Gazebo link
    # Gazebo camera link: +X=forward(optical), +Y=right, +Z=up
    # OpenCV camera frame: +Z=forward, +X=right, +Y=down
    # Mapping: G_x->C_z, G_y->C_x, G_z->C_(-y)
    R_C_G  = np.array([[0, 1, 0],
                        [0, 0,-1],
                        [1, 0, 0]])   # Gazebo link -> OpenCV cam
    R      = R_C_G @ R_G_W               # world -> OpenCV camera

    t      = -R @ _CAM_POS               # translation in OpenCV camera frame

    # Intrinsics from SDF (square pixels, principal point at image centre)
    fx = fy = _IMG_W / (2.0 * math.tan(_FOV_H / 2.0))
    cx, cy  = _IMG_W / 2.0, _IMG_H / 2.0

    return R, t, fx, fy, cx, cy


_R, _T, _FX, _FY, _CX, _CY = _build_camera_matrix()


def world_to_pixel(xyz: np.ndarray) -> tuple[float, float] | None:
    """
    Project a world 3D point to image pixel (u, v).
    Returns None if the point is behind the camera.
    """
    P_cam = _R @ xyz + _T
    if P_cam[2] <= 0:        # behind camera
        return None
    u = _FX * P_cam[0] / P_cam[2] + _CX
    v = _FY * P_cam[1] / P_cam[2] + _CY
    return u, v


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-Union of two [x0, y0, x1, y1] boxes."""
    x0 = max(a[0], b[0]);  y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]);  y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes: list[list[float]], iou_threshold: float = 0.5) -> list[list[float]]:
    """Greedy NMS: remove boxes with IoU > threshold against a higher-area box."""
    if len(boxes) <= 1:
        return boxes
    ranked = sorted(boxes, key=lambda b: -(b[2]-b[0])*(b[3]-b[1]))
    kept: list[list[float]] = []
    for box in ranked:
        if not any(_iou(box, k) > iou_threshold for k in kept):
            kept.append(box)
    return kept


class PerceptionModule:
    """
    Visual grounding module: maps free-form VLM names to known PDDL names
    and estimates 3D object poses for the robot.

    Uses GroundingDINO for open-vocabulary object detection.
    GroundingDINO significantly outperforms OWL-ViT on both synthetic
    (Gazebo) and real (RealSense) images for robotics manipulation tasks.

    Usage:
        perception = PerceptionModule()
        perception.load()
        corrected_plan = perception.ground_names(plan, image,
                                                  known_items=["red_cup", "blue_box"],
                                                  known_locations=["shelf_b"])
    """

    MODEL_NAME          = "IDEA-Research/grounding-dino-tiny"
    # GroundingDINO thresholds — lower than typical (0.3) to handle the
    # domain gap between Gazebo-rendered images and real photos.
    DETECTION_THRESHOLD = 0.15   # box + text confidence cutoff
    IOU_MATCH_THRESHOLD = 0.05   # min IoU to accept a visual match

    def __init__(self, threshold: float | None = None) -> None:
        self._processor = None
        self._model     = None
        self._device    = "cuda" if torch.cuda.is_available() else "cpu"
        self._last_detection: dict | None = None   # set after each get_pose()
        # Allow per-instance threshold override.
        # Typical values: 0.15 (simulation), 0.20-0.25 (real robot).
        if threshold is not None:
            self.DETECTION_THRESHOLD = threshold
            self.GET_POSE_THRESHOLD  = threshold * 0.67

    def load(self) -> None:
        """Load GroundingDINO-Tiny weights (~700 MB)."""
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print(f"[INFO] Loading PerceptionModule ({self.MODEL_NAME})…")
        self._processor = AutoProcessor.from_pretrained(self.MODEL_NAME)
        self._model     = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.MODEL_NAME
        ).to(self._device)
        self._model.eval()
        print("[OK]   PerceptionModule loaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    # Words that hint at a model being a location (surface/stand/shelf)
    # rather than a graspable item.
    _LOCATION_HINTS: frozenset[str] = frozenset({
        "shelf", "table", "surface", "platform", "stand",
        "rack", "tray", "bin", "ground", "floor",
    })

    def ground_names(
        self,
        plan:            VLMPlan,
        image:           Image.Image,
        known_items:     list[str],
        known_locations: list[str],
    ) -> VLMPlan:
        """
        Correct object/location names in *plan* using visual grounding.

        Names already present in known_items / known_locations are left unchanged.
        Unknown names are matched using:
          1. OWL-ViT visual detection (bounding-box IoU)
          2. Token-overlap fallback
          3. Semantic role hint (object vs location) to avoid conflicts

        The matching is role-aware: names used as pick targets ("object" key)
        are matched separately from names used as place targets ("location" key).
        This prevents e.g. "green_platform" and "red_cylinder" from both
        resolving to "red_cup".

        Returns a new VLMPlan; the original is not modified.
        """
        if self._model is None:
            raise RuntimeError("Call load() before ground_names()")

        # Deduplicate while preserving order (items before locations)
        seen: set[str] = set()
        known_all: list[str] = []
        for n in known_items + known_locations:
            if n not in seen:
                known_all.append(n)
                seen.add(n)

        # Separate unknown names by their semantic role in the plan
        to_ground_items: set[str] = set()    # used as pick/look_at targets
        to_ground_locs:  set[str] = set()    # used as place destinations

        for step in plan.steps:
            for key, val in step.args.items():
                if not isinstance(val, str) or val in known_all:
                    continue
                if key in ("object", "target"):
                    to_ground_items.add(val)
                elif key == "location":
                    to_ground_locs.add(val)

        all_to_ground = to_ground_items | to_ground_locs
        if not all_to_ground:
            return plan

        print(f"[INFO] PerceptionModule: grounding {all_to_ground} -> {known_all}")

        # Pre-detect all known objects once (visual boxes)
        known_boxes = self._detect(known_all, image)

        name_map: dict[str, str] = {}
        used_pddl: set[str] = set()

        def _match(vlm_name: str, prefer_location: bool) -> str:
            """Match one VLM name, avoiding already-used candidates."""
            available = [c for c in known_all if c not in used_pddl]
            if not available:
                available = known_all

            # Visual match
            vlm_boxes = self._detect([vlm_name], image)
            vis_match = self._best_iou_match(
                vlm_boxes.get(vlm_name, []),
                {k: known_boxes[k] for k in available if k in known_boxes},
            )
            if vis_match:
                return vis_match

            # Token-overlap fallback
            best_tok  = self._token_fallback(vlm_name, available)
            tok_score = len(
                set(vlm_name.lower().replace("_", " ").split()) &
                set(best_tok.lower().replace("_", " ").split())
            )

            # If token score is 0 and we need a location, prefer candidates
            # whose name contains a location-hint word (shelf, table, …).
            if tok_score == 0 and prefer_location:
                loc_scored = [
                    (c, self._location_score(c)) for c in available
                ]
                loc_scored.sort(key=lambda x: x[1], reverse=True)
                if loc_scored[0][1] > 0:
                    return loc_scored[0][0]

            return best_tok

        # Match items first (pick / look_at targets)
        for name in sorted(to_ground_items):
            match = _match(name, prefer_location=False)
            tag   = "visual" if self._best_iou_match(
                self._detect([name], image).get(name, []),
                {k: known_boxes[k] for k in known_all if k in known_boxes},
            ) else "token fallback"
            name_map[name] = match
            used_pddl.add(match)
            icon = "[OK]  " if "visual" in tag else "[WARN]"
            print(f"{icon} '{name}' -> '{match}' ({tag})")

        # Match locations second (prefer candidates not used by items)
        for name in sorted(to_ground_locs):
            match = _match(name, prefer_location=True)
            tag   = "visual" if self._best_iou_match(
                self._detect([name], image).get(name, []),
                {k: known_boxes[k] for k in known_all if k in known_boxes},
            ) else "token fallback"
            name_map[name] = match
            used_pddl.add(match)
            icon = "[OK]  " if "visual" in tag else "[WARN]"
            print(f"{icon} '{name}' -> '{match}' ({tag})")

        corrected = deepcopy(plan)
        corrected.steps = [
            PlanStep(
                primitive=step.primitive,
                args={
                    k: (name_map.get(v, v) if isinstance(v, str) else v)
                    for k, v in step.args.items()
                },
            )
            for step in plan.steps
        ]
        return corrected

    # Best GroundingDINO query per PDDL name.
    # First entry is the primary query used in _detect(); the rest are
    # Fallback queries used ONLY when no VLM description is available.
    # In Phase 2+, the natural-language description from the VLM is passed
    # directly as the GroundingDINO query — no hardcoding needed.
    # This dict is kept as a last resort for PDDL-name-only calls (e.g. tests).
    _QUERY_SYNONYMS: dict[str, list[str]] = {}

    GET_POSE_THRESHOLD = 0.10

    @staticmethod
    def _median_depth_from_box(
        depth_image: np.ndarray,
        box: list[float],
        shrink: float = 0.35,
        min_valid: int = 5,
    ) -> float | None:
        """
        Estimate depth of an object from its bounding box using RealSense depth data.

        Samples the central (1-2*shrink) region of the box to avoid edge noise and
        background pixels. Returns the median of valid (non-zero) depth values in
        metres, or None if fewer than min_valid valid readings are found.

        depth_image: uint16 H×W array, values in mm (RealSense D435i convention).
        box:         [x0, y0, x1, y1] in pixels.
        shrink:      fraction to crop from each side of the box before sampling.
        """
        x0, y0, x1, y1 = box
        dx = (x1 - x0) * shrink
        dy = (y1 - y0) * shrink
        rx0 = int(max(0, x0 + dx))
        ry0 = int(max(0, y0 + dy))
        rx1 = int(min(depth_image.shape[1] - 1, x1 - dx))
        ry1 = int(min(depth_image.shape[0] - 1, y1 - dy))
        if rx1 <= rx0 or ry1 <= ry0:
            return None
        patch = depth_image[ry0:ry1, rx0:rx1].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size < min_valid:
            return None
        return float(np.median(valid)) / 1000.0  # mm → metres

    def get_pose(
        self,
        object_name: str,
        image: Image.Image,
        K: np.ndarray,
        cam_to_base: np.ndarray,
        obj_z_base: float = 0.025,
        vlm_description: str | None = None,
        depth_image: np.ndarray | None = None,
    ) -> dict | None:
        """
        Estimate a 3D object pose in panda_link0 frame.

        Two modes depending on whether a depth image is supplied:

        - depth_image provided (real robot, RealSense D435i): samples the central
          region of the detected bounding box in the depth channel, takes the median
          of valid readings, and unprojects using K.  No fixed-z assumption.

        - depth_image=None (simulation fallback): ray-plane intersection at z=obj_z_base
          (table surface height in panda_link0 frame).

        The DINO query is built as:
          1. vlm_description if provided (e.g. "the red cup on the table")
          2. Otherwise: object_name converted to natural language ("red cup")
          3. Fallback synonyms from _QUERY_SYNONYMS if defined

        Returns {"x": …, "y": …, "z": …} in panda_link0 frame, or None.
        """
        if self._model is None:
            raise RuntimeError("Call load() before get_pose()")

        readable = object_name.replace("_", " ")
        synonyms = self._QUERY_SYNONYMS.get(object_name, [])
        primary  = vlm_description.lower() if vlm_description else readable
        queries  = list(dict.fromkeys([primary, readable] + synonyms))

        best_score = -1.0
        best_box   = None
        for q in queries:
            readable_q = q.replace("_", " ").lower() + " ."
            inputs_q = self._processor(
                images=image, text=readable_q, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                out_q = self._model(**inputs_q)
            H, W = image.height, image.width
            res = self._processor.post_process_grounded_object_detection(
                out_q,
                inputs_q.input_ids,
                threshold=self.GET_POSE_THRESHOLD,
                text_threshold=self.GET_POSE_THRESHOLD * 0.8,
                target_sizes=[(H, W)],
            )[0]
            for box, score in zip(res["boxes"], res["scores"]):
                s = float(score)
                if s > best_score:
                    best_score = s
                    best_box   = box.tolist()

        if best_box is None:
            self._last_detection = None
            return None

        u = (best_box[0] + best_box[2]) / 2.0
        v = (best_box[1] + best_box[3]) / 2.0
        print(f"[Perception] get_pose '{object_name}': GroundingDINO "
              f"score={best_score:.3f} → center=({u:.0f},{v:.0f})")
        self._last_detection = {
            "name":  object_name,
            "box":   best_box,
            "score": best_score,
        }

        R     = cam_to_base[:3, :3]
        t     = cam_to_base[:3, 3]
        K_inv = np.linalg.inv(K)

        # ── Mode 1: depth-based unprojection (real robot, RealSense) ─────────
        if depth_image is not None:
            z_cam = self._median_depth_from_box(depth_image, best_box)
            if z_cam is not None and z_cam > 0.05:
                p_cam = K_inv @ np.array([u, v, 1.0]) * z_cam
                point = R @ p_cam + t
                print(f"[Perception] depth median z_cam={z_cam:.3f}m "
                      f"→ base ({point[0]:.3f},{point[1]:.3f},{point[2]:.3f})")
                return {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
            print(f"[Perception] depth sampling failed for '{object_name}' "
                  f"— falling back to ray-plane")

        # ── Mode 2: ray-plane intersection at fixed z (simulation fallback) ──
        d_cam  = K_inv @ np.array([u, v, 1.0])
        d_base = R @ d_cam
        norm   = np.linalg.norm(d_base)
        if norm < 1e-9:
            return None
        d_base /= norm
        if abs(d_base[2]) < 1e-9:
            return None
        t_ray = (obj_z_base - t[2]) / d_base[2]
        if t_ray < 0:
            return None
        point = t + t_ray * d_base
        return {"x": float(point[0]), "y": float(point[1]), "z": float(obj_z_base)}

    @staticmethod
    def draw_detections(
        image: "Image.Image",
        detections: list[dict],
    ) -> "Image.Image":
        """
        Draw DINO detection boxes on a copy of image.

        detections: list of {"name": str, "box": [x1,y1,x2,y2], "score": float}
        Returns annotated PIL Image.
        """
        from PIL import ImageDraw, ImageFont
        import random

        img = image.copy().convert("RGB")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
            font_small = font

        palette = ["#FF4444", "#44FF44", "#4444FF", "#FF9900", "#FF44FF",
                   "#00CCFF", "#FFFF00", "#FF6699", "#99FF66", "#6699FF"]
        color_map: dict[str, str] = {}
        for i, det in enumerate(detections):
            name = det["name"]
            if name not in color_map:
                color_map[name] = palette[len(color_map) % len(palette)]

        for det in detections:
            name  = det["name"]
            box   = det["box"]          # [x1, y1, x2, y2]
            score = det["score"]
            color = color_map[name]
            x1, y1, x2, y2 = [int(v) for v in box]

            for d in range(3):
                draw.rectangle([x1-d, y1-d, x2+d, y2+d], outline=color)

            label = f"{name}  {score:.2f}"
            bbox_txt = draw.textbbox((x1, y1 - 24), label, font=font)
            draw.rectangle(bbox_txt, fill=color)
            draw.text((x1, y1 - 24), label, fill="black", font=font)

            # Centre crosshair
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            draw.line([cx - 8, cy, cx + 8, cy], fill=color, width=2)
            draw.line([cx, cy - 8, cx, cy + 8], fill=color, width=2)

        return img

    @staticmethod
    def load_camera_data(
        data_dir: str = "/workspace/data",
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Load K (3×3) and cam_to_base (4×4) from camera_info.json / camera_pose.json.
        Returns (K, cam_to_base) or None if either file is missing.
        """
        import json
        import os

        cam_info_path = os.path.join(data_dir, "camera_info.json")
        cam_pose_path = os.path.join(data_dir, "camera_pose.json")
        if not (os.path.exists(cam_info_path) and os.path.exists(cam_pose_path)):
            return None

        with open(cam_info_path) as f:
            K = np.array(json.load(f)["K"], dtype=np.float64)
        with open(cam_pose_path) as f:
            cam_to_base = np.array(json.load(f)["cam_to_base"], dtype=np.float64)
        return K, cam_to_base

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect(
        self,
        names:     list[str],
        image:     Image.Image,
        threshold: Optional[float] = None,
    ) -> dict[str, list[list[float]]]:
        """
        Run GroundingDINO and return bounding boxes per name.

        GroundingDINO takes a single dot-separated text string and returns
        bounding boxes with matched phrase labels.  Each detected label is
        matched back to the input name list via token overlap.

        Returns: {name: [[x0, y0, x1, y1], ...]}
        """
        if threshold is None:
            threshold = self.DETECTION_THRESHOLD

        # Convert PDDL names to natural-language queries.
        # No hardcoded synonyms — GroundingDINO handles arbitrary descriptions.
        # The VLM's original text (passed via ground_names / get_pose callers)
        # is already in natural language; PDDL names just need underscore removal.
        readable = [n.replace("_", " ").lower() for n in names]
        text = " . ".join(readable) + " ."

        inputs = self._processor(
            images=image, text=text, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        H, W = image.height, image.width
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold * 0.8,
            target_sizes=[(H, W)],
        )[0]

        boxes: dict[str, list] = {n: [] for n in names}

        label_key = "text_labels" if "text_labels" in results else "labels"
        for box, label in zip(results["boxes"], results[label_key]):
            x0, y0, x1, y1 = box.tolist()
            label_tokens = set(label.lower().split())
            # Match label to the input name with maximum token overlap
            best_name, best_overlap = None, 0
            for orig, read in zip(names, readable):
                overlap = len(label_tokens & set(read.split()))
                if overlap > best_overlap:
                    best_overlap, best_name = overlap, orig
            if best_name:
                boxes[best_name].append([x0, y0, x1, y1])

        # NMS per object: keep only non-overlapping boxes
        return {n: _nms(b) for n, b in boxes.items()}

    def _best_iou_match(
        self,
        query_boxes: list[list[float]],
        known_boxes: dict[str, list[list[float]]],
    ) -> str | None:
        """Return the known name with highest IoU against any query box."""
        best_name: str | None = None
        best_iou  = self.IOU_MATCH_THRESHOLD  # minimum to qualify

        for known_name, kboxes in known_boxes.items():
            for qb in query_boxes:
                for kb in kboxes:
                    score = _iou(qb, kb)
                    if score > best_iou:
                        best_iou = score
                        best_name = known_name
        return best_name

    def _location_score(self, name: str) -> int:
        """How many location-hint words appear in the candidate name."""
        words = set(name.lower().replace("_", " ").split())
        return len(words & self._LOCATION_HINTS)

    # Shape/function synonyms for token matching — MUST be scene-independent.
    # Only geometric/functional equivalences, NOT color-object mappings.
    # Color mappings (e.g. "red"->"cup") would overfit to specific scenes.
    _SYNONYMS: dict[str, list[str]] = {
        "cube":      ["box"],
        "box":       ["cube"],
        "cylinder":  ["cup"],
        "cup":       ["cylinder"],
        "platform":  ["shelf"],
        "shelf":     ["platform", "rack", "stand"],
        "surface":   ["shelf", "platform", "table"],
        "container": ["box", "bin"],
        "object":    [],   # too generic — no expansion
        "thing":     [],   # too generic — no expansion
    }

    @staticmethod
    def _token_fallback(name: str, candidates: list[str]) -> str:
        """Fallback: pick candidate with most token overlap (+ synonym expansion)."""
        raw_tokens = set(name.lower().replace("_", " ").split())
        expanded: set[str] = set(raw_tokens)
        for token in raw_tokens:
            expanded.update(PerceptionModule._SYNONYMS.get(token, []))

        scored = []
        for c in candidates:
            c_tokens = set(c.lower().replace("_", " ").split())
            c_expanded: set[str] = set(c_tokens)
            for token in c_tokens:
                c_expanded.update(PerceptionModule._SYNONYMS.get(token, []))
            score = len(expanded & c_expanded)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
