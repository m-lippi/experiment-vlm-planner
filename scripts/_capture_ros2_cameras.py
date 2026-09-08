#!/usr/bin/env python3
"""Capture overview and wrist RGB-D data from ROS 2 topics.

This helper runs in the ROS 2 environment. It is normally invoked by
``capture_and_plan_ros2.py`` rather than directly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image as PilImage
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from _apriltag_calibration_common import transform_from_ros, write_json


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _rgb_image(msg: Image) -> PilImage.Image:
    encodings = {
        "rgb8": (3, False),
        "r8g8b8": (3, False),
        "bgr8": (3, True),
        "rgba8": (4, False),
        "bgra8": (4, True),
        "mono8": (1, False),
    }
    encoding = msg.encoding.lower()
    if encoding not in encodings:
        raise ValueError(f"unsupported color encoding {msg.encoding!r}")
    channels, bgr = encodings[encoding]
    rows = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    array = rows[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = array[:, :, :3]
    if bgr:
        array = array[:, :, ::-1]
    return PilImage.fromarray(np.ascontiguousarray(array), "RGB")


def _depth_mm(msg: Image) -> np.ndarray:
    encoding = msg.encoding.lower()
    if encoding in ("16uc1", "mono16"):
        dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
        itemsize = 2
        scale = 1.0
    elif encoding == "32fc1":
        dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
        itemsize = 4
        scale = 1000.0
    else:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}")
    row_items = msg.step // itemsize
    rows = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, row_items)
    depth = rows[:, : msg.width].astype(np.float64) * scale
    depth[~np.isfinite(depth)] = 0.0
    return np.clip(np.rint(depth), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _camera_info(msg: CameraInfo) -> dict:
    return {
        "K": np.asarray(msg.k, dtype=float).reshape(3, 3).tolist(),
        "width": int(msg.width),
        "height": int(msg.height),
        "D": list(msg.d),
        "distortion_model": msg.distortion_model,
        "frame_id": msg.header.frame_id,
    }


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to normalized (x, y, z, w)."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
            0.25 * s,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            q = (0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s,
                 (rotation[2, 1] - rotation[1, 2]) / s)
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            q = ((rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                 (rotation[1, 2] + rotation[2, 1]) / s,
                 (rotation[0, 2] - rotation[2, 0]) / s)
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            q = ((rotation[0, 2] + rotation[2, 0]) / s,
                 (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s,
                 (rotation[1, 0] - rotation[0, 1]) / s)
    norm = math.sqrt(sum(value * value for value in q))
    return tuple(value / norm for value in q)


class CaptureNode(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("capture_ros2_cameras")
        self.args = args
        self.data: dict[str, dict] = {"overview": {}, "wrist": {}}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.overview_tf_source = "unavailable"
        self._tf_checked = False

        topics = {
            "overview": (args.overview_image_topic, args.overview_depth_topic,
                         args.overview_info_topic),
            "wrist": (args.wrist_image_topic, args.wrist_depth_topic,
                      args.wrist_info_topic),
        }
        for camera, (image_topic, depth_topic, info_topic) in topics.items():
            if camera == "wrist" and not args.include_wrist:
                continue
            self.create_subscription(
                Image, image_topic,
                lambda msg, c=camera: self._color(msg, c), qos_profile_sensor_data,
            )
            self.create_subscription(
                Image, depth_topic,
                lambda msg, c=camera: self._depth(msg, c), qos_profile_sensor_data,
            )
            self.create_subscription(
                CameraInfo, info_topic,
                lambda msg, c=camera: self._info(msg, c), qos_profile_sensor_data,
            )

        self.create_timer(0.5, self._ensure_overview_tf)

    def _color(self, msg: Image, camera: str) -> None:
        if "image" not in self.data[camera]:
            try:
                self.data[camera]["image"] = _rgb_image(msg)
            except Exception as exc:
                self.get_logger().warning(f"{camera} color: {exc}")

    def _depth(self, msg: Image, camera: str) -> None:
        if "depth" not in self.data[camera]:
            try:
                self.data[camera]["depth"] = _depth_mm(msg)
            except Exception as exc:
                self.get_logger().warning(f"{camera} depth: {exc}")

    def _info(self, msg: CameraInfo, camera: str) -> None:
        if "info" not in self.data[camera]:
            self.data[camera]["info"] = _camera_info(msg)

    def _ensure_overview_tf(self) -> None:
        if self._tf_checked:
            return
        calibrated_matrix = None
        pose_path = self.args.overview_pose
        if pose_path.exists():
            try:
                with pose_path.open(encoding="utf-8") as stream:
                    pose_data = json.load(stream)
                calibrated_matrix = np.asarray(pose_data["cam_to_base"], dtype=float)
                if calibrated_matrix.shape != (4, 4):
                    raise ValueError("cam_to_base is not 4x4")
            except Exception as exc:
                self.get_logger().warning(f"Cannot read overview calibration: {exc}")
        try:
            transform = self.tf_buffer.lookup_transform(
                self.args.base_frame, self.args.overview_frame, Time()
            )
            tf_matrix = transform_from_ros(transform.transform)
            matrix = calibrated_matrix if calibrated_matrix is not None else tf_matrix
            self.data["overview"]["pose"] = {
                "cam_to_base": matrix.tolist()
            }
            self.overview_tf_source = "existing_tf"
            if calibrated_matrix is not None:
                translation_error = np.linalg.norm(
                    calibrated_matrix[:3, 3] - tf_matrix[:3, 3]
                )
                rotation_error = np.linalg.norm(
                    calibrated_matrix[:3, :3] - tf_matrix[:3, :3]
                )
                if translation_error > 0.005 or rotation_error > 0.01:
                    self.get_logger().warning(
                        "Existing overview TF differs from overview_camera_pose.json; "
                        "depth poses will use the calibrated file. Restart the launch "
                        "stack to refresh its static transform."
                    )
            self._tf_checked = True
            return
        except Exception:
            pass

        if calibrated_matrix is None:
            return
        try:
            matrix = calibrated_matrix
            qx, qy, qz, qw = _rotation_to_quaternion(matrix[:3, :3])
            msg = TransformStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.args.base_frame
            msg.child_frame_id = self.args.overview_frame
            msg.transform.translation.x = float(matrix[0, 3])
            msg.transform.translation.y = float(matrix[1, 3])
            msg.transform.translation.z = float(matrix[2, 3])
            msg.transform.rotation.x = qx
            msg.transform.rotation.y = qy
            msg.transform.rotation.z = qz
            msg.transform.rotation.w = qw
            self.static_broadcaster.sendTransform(msg)
            self.data["overview"]["pose"] = {"cam_to_base": matrix.tolist()}
            self.overview_tf_source = "published_from_calibration"
            self._tf_checked = True
            self.get_logger().info(
                f"Published {self.args.base_frame} -> {self.args.overview_frame} "
                f"from {pose_path}"
            )
        except Exception as exc:
            self.get_logger().warning(f"Cannot publish overview transform: {exc}")

    def capture_complete(self) -> bool:
        required = ("image", "depth", "info")
        return all(key in self.data["overview"] for key in required)

    def save(self) -> dict:
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "base_frame": self.args.base_frame,
            "overview_tf_source": self.overview_tf_source,
            "cameras": {},
        }
        for camera, values in self.data.items():
            available = all(key in values for key in ("image", "depth", "info"))
            manifest["cameras"][camera] = {"available": available}
            if not available:
                continue
            values["image"].save(self.args.output_dir / f"{camera}.png")
            np.save(self.args.output_dir / f"{camera}_depth_mm.npy", values["depth"])
            write_json(self.args.output_dir / f"{camera}_camera_info.json", values["info"])
            if "pose" in values:
                write_json(self.args.output_dir / f"{camera}_camera_pose.json", values["pose"])

        # Wrist pose is dynamic, so resolve it immediately before saving.
        if manifest["cameras"]["wrist"]["available"]:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.args.base_frame, self.args.wrist_frame, Time()
                )
                wrist_pose = {
                    "cam_to_base": transform_from_ros(transform.transform).tolist()
                }
                write_json(self.args.output_dir / "wrist_camera_pose.json", wrist_pose)
                manifest["cameras"]["wrist"]["pose_available"] = True
            except Exception as exc:
                manifest["cameras"]["wrist"]["pose_available"] = False
                self.get_logger().warning(f"Cannot resolve wrist transform: {exc}")

        manifest["cameras"]["overview"]["pose_available"] = (
            "pose" in self.data["overview"]
        )
        write_json(self.args.output_dir / "capture_manifest.json", manifest)
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--include-wrist", action="store_true")
    parser.add_argument("--base-frame", default="fr3_link0")
    parser.add_argument("--overview-frame", default="overview_camera_optical_frame")
    parser.add_argument("--wrist-frame", default="camera_color_optical_frame")
    parser.add_argument(
        "--overview-pose",
        type=Path,
        default=_REPO_ROOT / "data" / "overview_camera_pose.json",
    )
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
    parser.add_argument("--wrist-image-topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--wrist-depth-topic", default="/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--wrist-info-topic", default="/camera/color/camera_info")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = CaptureNode(args)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            wrist_ready = (
                not args.include_wrist
                or all(k in node.data["wrist"] for k in ("image", "depth", "info"))
            )
            if node.capture_complete() and wrist_ready and node._tf_checked:
                break
        manifest = node.save()
        print(json.dumps(manifest))
        return 0 if manifest["cameras"]["overview"]["available"] else 1
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
