#!/usr/bin/env python3
"""Save an AprilTag pose in fr3_link0 using the wrist camera.

Keep the robot stationary with the whole tag visible, then run this script in
the ROS 2 environment. The output is consumed by
``calibrate_overview_camera_apriltag.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros

from _apriltag_calibration_common import (
    AprilTagDetector,
    average_transforms,
    camera_matrix,
    estimate_camera_from_tag,
    ros_image_to_gray,
    transform_from_ros,
    write_json,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


class TagInBaseNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("save_table_apriltag_transform")
        self.args = args
        self.K: np.ndarray | None = None
        self.distortion = np.empty(0, dtype=np.float64)
        self.samples: list[np.ndarray] = []
        self.errors: list[float] = []
        self.detector = AprilTagDetector(args.family)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
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
            tf = self.tf_buffer.lookup_transform(
                self.args.base_frame, self.args.camera_frame, Time()
            )
            base_from_camera = transform_from_ros(tf.transform)
            self.samples.append(base_from_camera @ camera_from_tag)
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
        "--tag-size", type=float, required=True,
        help="Outer black-square edge length in metres",
    )
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-reprojection-error", type=float, default=2.0)
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "table_apriltag_transform.json",
    )
    args = parser.parse_args()
    if (
        args.tag_size <= 0
        or args.samples <= 0
        or args.timeout <= 0
        or args.max_reprojection_error <= 0
    ):
        parser.error(
            "--tag-size, --samples, --timeout, and --max-reprojection-error "
            "must be positive"
        )
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    try:
        node = TagInBaseNode(args)
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
                "Check topics, TF, tag ID/size, lighting, and tag visibility.",
                file=sys.stderr,
            )
            return 1
        tag_to_base = average_transforms(node.samples)
        write_json(
            args.output,
            {
                "tag_to_base": tag_to_base.tolist(),
                "parent_frame": args.base_frame,
                "child_frame": f"apriltag_{args.tag_id}",
                "tag_id": args.tag_id,
                "tag_family": args.family,
                "tag_size_m": args.tag_size,
                "samples": len(node.samples),
                "mean_reprojection_error_px": float(np.mean(node.errors)),
            },
        )
        print(f"[OK] Saved {args.base_frame} <- tag transform to {args.output}")
        print(np.array2string(tag_to_base, precision=6, suppress_small=True))
        return 0
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
