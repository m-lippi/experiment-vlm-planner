#!/usr/bin/env python3
"""Calibrate the fixed overview camera from a table-mounted AprilTag.

The script reads the overview RealSense intrinsics from CameraInfo, detects the
same tag previously located with ``save_table_apriltag_transform.py``, and
writes ``overview_camera_info.json`` plus ``overview_camera_setup.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from _apriltag_calibration_common import (
    AprilTagDetector,
    average_transforms,
    camera_matrix,
    estimate_camera_from_tag,
    matrix_to_rpy,
    ros_image_to_gray,
    validate_transform,
    write_json,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_tag_transform(path: Path, expected_tag_id: int) -> tuple[np.ndarray, dict]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("tag_id") is not None and int(data["tag_id"]) != expected_tag_id:
        raise ValueError(
            f"{path} contains tag {data['tag_id']}, but --tag-id is {expected_tag_id}"
        )
    if "tag_to_base" not in data:
        raise ValueError(f"{path} has no 'tag_to_base' matrix")
    transform = np.asarray(data["tag_to_base"], dtype=np.float64)
    validate_transform(transform, "tag_to_base")
    return transform, data


class OverviewCalibrationNode(Node):
    def __init__(self, args: argparse.Namespace, tag_to_base: np.ndarray) -> None:
        super().__init__("calibrate_overview_camera_apriltag")
        self.args = args
        self.tag_to_base = tag_to_base
        self.K: np.ndarray | None = None
        self.distortion = np.empty(0, dtype=np.float64)
        self.width: int | None = None
        self.height: int | None = None
        self.distortion_model: str | None = None
        self.samples: list[np.ndarray] = []
        self.errors: list[float] = []
        self.detector = AprilTagDetector(args.family)
        self.create_subscription(
            CameraInfo, args.camera_info_topic, self._camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, args.image_topic, self._image, qos_profile_sensor_data
        )
        self.get_logger().info(f"Waiting for tag {args.tag_id} on {args.image_topic}")

    @property
    def done(self) -> bool:
        return len(self.samples) >= self.args.samples

    def _camera_info(self, msg: CameraInfo) -> None:
        try:
            self.K, self.distortion = camera_matrix(msg)
            self.width, self.height = int(msg.width), int(msg.height)
            self.distortion_model = msg.distortion_model
        except ValueError as exc:
            self.get_logger().error(str(exc))

    def _image(self, msg: Image) -> None:
        if self.done or self.K is None:
            return
        try:
            corners = self.detector.corners(ros_image_to_gray(msg), self.args.tag_id)
            if corners is None:
                return
            camera_from_tag, error = estimate_camera_from_tag(
                corners, self.args.tag_size, self.K, self.distortion
            )
            if error > self.args.max_reprojection_error:
                self.get_logger().warning(
                    f"Rejected detection with {error:.2f}px reprojection error"
                )
                return
            # T_base_camera = T_base_tag * inverse(T_camera_tag).
            camera_to_base = self.tag_to_base @ np.linalg.inv(camera_from_tag)
            self.samples.append(camera_to_base)
            self.errors.append(error)
            self.get_logger().info(
                f"Accepted sample {len(self.samples)}/{self.args.samples} "
                f"(error {error:.2f}px)"
            )
        except Exception as exc:
            self.get_logger().warning(f"Could not use image: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag-id", type=int, required=True)
    parser.add_argument(
        "--tag-size", type=float, default=None,
        help="Outer black-square edge length in metres (defaults to saved value)",
    )
    parser.add_argument("--family", default=None, help="Defaults to saved tag family")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument(
        "--image-topic",
        default="/overview_camera/overview_camera/color/image_raw",
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/overview_camera/overview_camera/aligned_depth_to_color/camera_info",
    )
    parser.add_argument(
        "--tag-transform",
        type=Path,
        default=_REPO_ROOT / "data" / "table_apriltag_transform.json",
    )
    parser.add_argument(
        "--info-output",
        type=Path,
        default=_REPO_ROOT / "data" / "overview_camera_info.json",
    )
    parser.add_argument(
        "--setup-output",
        type=Path,
        default=_REPO_ROOT / "data" / "overview_camera_setup.json",
    )
    parser.add_argument(
        "--pose-output",
        type=Path,
        default=_REPO_ROOT / "data" / "overview_camera_pose.json",
        help="Also save the 4x4 cam_to_base matrix here",
    )
    parser.add_argument(
        "--table-z", type=float, default=None,
        help="Table height in fr3_link0; default is the saved tag origin's Z",
    )
    args = parser.parse_args()
    if args.samples <= 0 or args.timeout <= 0 or args.max_reprojection_error <= 0:
        parser.error(
            "--samples, --timeout, and --max-reprojection-error must be positive"
        )
    return args


def main() -> int:
    args = parse_args()
    try:
        tag_to_base, tag_data = load_tag_transform(args.tag_transform, args.tag_id)
        saved_size = tag_data.get("tag_size_m")
        saved_family = tag_data.get("tag_family", "tag36h11")
        args.tag_size = args.tag_size if args.tag_size is not None else saved_size
        args.family = args.family if args.family is not None else saved_family
        if args.tag_size is None or args.tag_size <= 0:
            raise ValueError("tag size is missing; pass --tag-size")
        if saved_size is not None and not np.isclose(args.tag_size, float(saved_size)):
            raise ValueError(
                f"--tag-size {args.tag_size} differs from saved size {saved_size}; "
                "both calibrations must use the same physical size"
            )
        if args.family != saved_family:
            raise ValueError(
                f"--family {args.family} differs from saved family {saved_family}"
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Cannot load tag transform: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    try:
        node = OverviewCalibrationNode(args, tag_to_base)
    except Exception as exc:
        rclpy.shutdown()
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.done and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if not node.done:
            print(
                f"[ERROR] Timed out with {len(node.samples)}/{args.samples} valid samples. "
                "Check topics, tag ID/size, lighting, and tag visibility.",
                file=sys.stderr,
            )
            return 1

        camera_to_base = average_transforms(node.samples)
        validate_transform(camera_to_base, "camera_to_base")
        roll, pitch, yaw = matrix_to_rpy(camera_to_base[:3, :3])
        translation = camera_to_base[:3, 3]
        table_z = float(tag_to_base[2, 3] if args.table_z is None else args.table_z)

        info = {
            "K": node.K.tolist(),
            "width": node.width,
            "height": node.height,
            "D": node.distortion.tolist(),
            "distortion_model": node.distortion_model,
        }
        setup = {
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]),
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "z_table": table_z,
        }
        pose = {
            "cam_to_base": camera_to_base.tolist(),
            "parent_frame": "fr3_link0",
            "child_frame": "overview_camera_optical_frame",
            "calibration_tag_id": args.tag_id,
            "samples": len(node.samples),
            "mean_reprojection_error_px": float(np.mean(node.errors)),
        }
        write_json(args.info_output, info)
        write_json(args.setup_output, setup)
        write_json(args.pose_output, pose)
        print(f"[OK] Saved intrinsics to {args.info_output}")
        print(f"[OK] Saved launch setup to {args.setup_output}")
        print(f"[OK] Saved matrix pose to {args.pose_output}")
        print(json.dumps(setup, indent=2))
        print("Restart real_robot.launch.py to load the new static transform.")
        return 0
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
