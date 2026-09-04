#!/usr/bin/env python3
"""
_capture_scene.py — Runs INSIDE the Docker container.

Subscribes to /wrist_camera/image_raw, captures one frame, and saves it
as /workspace/data/scene.png (bind-mounted from the host).

Decodes the ROS2 Image message directly with numpy + PIL — does NOT use
cv_bridge to avoid the NumPy 1.x/2.x ABI incompatibility.

Usage (called by run_vlm_host.py via docker exec):
    docker exec vlm_ros2 bash -c "source ... && python3 /workspace/scripts/_capture_scene.py"

Exit codes:
    0 — image saved successfully
    1 — timeout or error
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
from PIL import Image as PilImage

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros

_OUTPUT_PATH          = "/workspace/data/scene.png"
_OVERVIEW_PATH        = "/workspace/data/scene_overview.png"
_OVERVIEW_INFO_PATH   = "/workspace/data/overview_camera_info.json"
_CAM_INFO_PATH        = "/workspace/data/camera_info.json"
_CAM_POSE_PATH        = "/workspace/data/camera_pose.json"
_DEPTH_PATH              = "/workspace/data/depth.npy"
_DEPTH_OVERVIEW_PATH     = "/workspace/data/depth_overview.npy"
_OVERVIEW_POSE_PATH      = "/workspace/data/overview_camera_pose.json"
_TIMEOUT_SEC       = 8.0
_TF_RETRIES        = 5
_TF_RETRY_SLEEP    = 0.3

# Primary: wrist camera (eye-in-hand) — matches real robot deployment.
# Fallback: overview camera (fixed, when wrist cam offline or scan pose unreachable).
_TOPICS = [
    "/wrist_camera/image_raw",
    "/overview_camera/image_raw",
]

# Supported encodings: (channels, dtype, bgr_flag)
# Keys are lowercase; Gazebo uses R8G8B8 which maps to rgb8.
_ENCODINGS = {
    "rgb8":   (3, np.uint8,  False),
    "r8g8b8": (3, np.uint8,  False),  # Gazebo Classic alias for rgb8
    "bgr8":   (3, np.uint8,  True),
    "rgba8":  (4, np.uint8,  False),
    "bgra8":  (4, np.uint8,  True),
    "mono8":  (1, np.uint8,  False),
}


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3×3 rotation matrix (no scipy)."""
    x2, y2, z2 = x*x, y*y, z*z
    return np.array([
        [1 - 2*(y2 + z2),   2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x2 + z2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x2 + y2)],
    ])


def _ros_image_to_pil(msg: Image) -> PilImage.Image:
    enc = msg.encoding.lower()
    if enc not in _ENCODINGS:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    channels, dtype, is_bgr = _ENCODINGS[enc]
    arr = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.width, channels)
    if is_bgr:
        arr = arr[:, :, ::-1].copy()  # BGR → RGB  (or BGRA → RGBA)
    if channels == 1:
        arr = np.repeat(arr, 3, axis=2)  # mono → RGB
    elif channels == 4:
        arr = arr[:, :, :3]  # drop alpha
    return PilImage.fromarray(arr, "RGB")


_PRIMARY_TOPIC_TIMEOUT = 5.0   # seconds to wait for primary before trying fallbacks


class _CaptureNode(Node):
    def __init__(self) -> None:
        super().__init__("_scene_capture")
        self.saved            = False
        self.overview_saved   = False   # overview camera saved separately
        self._primary         = _TOPICS[0]
        self._primary_deadline = time.time() + _PRIMARY_TOPIC_TIMEOUT
        self._K: np.ndarray | None = None
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        for topic in _TOPICS:
            self.create_subscription(
                Image, topic,
                lambda msg, t=topic: self._cb(msg, t),
                qos_profile_sensor_data,
            )
            self.get_logger().info(f"Listening on {topic}")

        # Always capture overview camera independently (fixed reference for VLM)
        self.create_subscription(
            Image, "/overview_camera/image_raw",
            self._cb_overview,
            qos_profile_sensor_data,
        )
        self._overview_K: np.ndarray | None = None
        self.create_subscription(
            CameraInfo, "/overview_camera/camera_info",
            self._overview_cam_info_cb,
            qos_profile_sensor_data,
        )

        self.create_subscription(
            CameraInfo, "/wrist_camera/camera_info",
            self._cam_info_cb,
            qos_profile_sensor_data,
        )

        self.depth_saved = False
        self.depth_overview_saved = False
        # Wrist depth: sim uses /wrist_camera/depth/..., real robot uses bridged topic.
        for depth_topic in ["/wrist_camera/depth/image_rect_raw",
                             "/camera/depth/image_rect_raw"]:
            self.create_subscription(
                Image, depth_topic,
                self._cb_depth,
                qos_profile_sensor_data,
            )
        # Overview depth: same camera model (RealSense D435i), same depth channel.
        self.create_subscription(
            Image, "/overview_camera/depth/image_rect_raw",
            self._cb_depth_overview,
            qos_profile_sensor_data,
        )

    def _cam_info_cb(self, msg: CameraInfo) -> None:
        if self._K is not None:
            return
        k = msg.k  # row-major 3×3
        self._K = np.array([
            [k[0], k[1], k[2]],
            [k[3], k[4], k[5]],
            [k[6], k[7], k[8]],
        ])
        data = {
            "K": self._K.tolist(),
            "width": msg.width,
            "height": msg.height,
        }
        with open(_CAM_INFO_PATH, "w") as f:
            json.dump(data, f)
        print(f"[OK] Camera info saved: {_CAM_INFO_PATH}")

    def _save_camera_pose(self) -> None:
        for attempt in range(_TF_RETRIES):
            try:
                tf = self._tf_buffer.lookup_transform(
                    "panda_link0", "wrist_camera_optical_frame", Time()
                )
                tr = tf.transform.translation
                q  = tf.transform.rotation
                R  = _quat_to_matrix(q.x, q.y, q.z, q.w)
                mat = np.eye(4)
                mat[:3, :3] = R
                mat[:3,  3] = [tr.x, tr.y, tr.z]
                with open(_CAM_POSE_PATH, "w") as f:
                    json.dump({"cam_to_base": mat.tolist()}, f)
                print(f"[OK] Camera pose saved: {_CAM_POSE_PATH}")
                return
            except Exception as exc:
                if attempt < _TF_RETRIES - 1:
                    time.sleep(_TF_RETRY_SLEEP)
                else:
                    print(f"[WARN] TF lookup failed after {_TF_RETRIES} attempts: {exc}",
                          file=sys.stderr)

    def _save_overview_camera_pose(self) -> None:
        """Save overview camera extrinsic via TF lookup (static TF from launch file)."""
        for attempt in range(_TF_RETRIES):
            try:
                tf = self._tf_buffer.lookup_transform(
                    "panda_link0", "overview_camera_optical_frame", Time()
                )
                tr = tf.transform.translation
                q  = tf.transform.rotation
                R  = _quat_to_matrix(q.x, q.y, q.z, q.w)
                mat = np.eye(4)
                mat[:3, :3] = R
                mat[:3,  3] = [tr.x, tr.y, tr.z]
                with open(_OVERVIEW_POSE_PATH, "w") as f:
                    json.dump({"cam_to_base": mat.tolist()}, f)
                print(f"[OK] Overview camera pose saved: {_OVERVIEW_POSE_PATH}")
                return
            except Exception as exc:
                if attempt < _TF_RETRIES - 1:
                    time.sleep(_TF_RETRY_SLEEP)
                else:
                    print(f"[WARN] Overview TF lookup failed after {_TF_RETRIES} attempts: {exc}",
                          file=sys.stderr)

    def _overview_cam_info_cb(self, msg: CameraInfo) -> None:
        if self._overview_K is not None:
            return
        k = msg.k
        self._overview_K = np.array([
            [k[0], k[1], k[2]],
            [k[3], k[4], k[5]],
            [k[6], k[7], k[8]],
        ])
        data = {"K": self._overview_K.tolist(), "width": msg.width, "height": msg.height}
        with open(_OVERVIEW_INFO_PATH, "w") as f:
            import json
            json.dump(data, f)
        print(f"[OK] Overview camera info saved: {_OVERVIEW_INFO_PATH}")

    def _cb_depth(self, msg: Image) -> None:
        """Save wrist camera depth image as uint16 numpy array (values in mm)."""
        if self.depth_saved:
            return
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(
                msg.height, msg.width
            )
            np.save(_DEPTH_PATH, arr)
            self.depth_saved = True
            print(f"[OK] Depth saved: {_DEPTH_PATH} ({msg.width}×{msg.height})")
        except Exception as exc:
            print(f"[WARN] Depth save failed: {exc}", file=sys.stderr)

    def _cb_depth_overview(self, msg: Image) -> None:
        """Save overview camera depth image as uint16 numpy array (values in mm)."""
        if self.depth_overview_saved:
            return
        try:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(
                msg.height, msg.width
            )
            np.save(_DEPTH_OVERVIEW_PATH, arr)
            self.depth_overview_saved = True
            print(f"[OK] Depth overview saved: {_DEPTH_OVERVIEW_PATH} ({msg.width}×{msg.height})")
        except Exception as exc:
            print(f"[WARN] Depth overview save failed: {exc}", file=sys.stderr)

    def _cb_overview(self, msg: Image) -> None:
        """Save overview camera image — fixed external reference for VLM."""
        if self.overview_saved:
            return
        try:
            img = _ros_image_to_pil(msg)
            img.save(_OVERVIEW_PATH)
            self.overview_saved = True
            print(f"[OK] Overview saved: {_OVERVIEW_PATH} ({img.width}×{img.height})")
            self._save_overview_camera_pose()
        except Exception:
            pass

    def _cb(self, msg: Image, topic: str) -> None:
        if self.saved:
            return
        # Give the primary topic priority for _PRIMARY_TOPIC_TIMEOUT seconds.
        # After that, accept any topic (fallback to overview camera).
        if topic != self._primary and time.time() < self._primary_deadline:
            return
        try:
            img = _ros_image_to_pil(msg)
            img.save(_OUTPUT_PATH)
            self.saved = True
            label = "primary" if topic == self._primary else "fallback"
            print(f"[OK] Scene saved: {_OUTPUT_PATH} ({img.width}×{img.height}) "
                  f"[{label}: {topic}]")
            self._save_camera_pose()
        except Exception as exc:
            print(f"[ERROR] Failed to save image: {exc}", file=sys.stderr)
            sys.exit(1)


def main() -> None:
    rclpy.init()
    node     = _CaptureNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    t0 = time.time()
    while (time.time() - t0) < _TIMEOUT_SEC:
        executor.spin_once(timeout_sec=0.1)
        if node.saved and node.overview_saved:
            break

    node.destroy_node()
    rclpy.shutdown()

    if not node.saved:
        print(
            f"[ERROR] No frame received within {_TIMEOUT_SEC}s "
            "(is the simulation running and camera active?)",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
