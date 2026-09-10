"""ROS 2 image publisher for the external Trust experiment webcam."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def find_video_device(name_match: str = "trust") -> str | None:
    """Find index-0 V4L2 capture device matching its USB or kernel name."""
    needle = name_match.casefold()
    by_id_matches = []
    for path in sorted(glob.glob("/dev/v4l/by-id/*")):
        label = Path(path).name.casefold()
        if needle in label and "index0" in label.replace("-", ""):
            by_id_matches.append(path)
    if by_id_matches:
        return by_id_matches[0]

    for sys_path in sorted(Path("/sys/class/video4linux").glob("video*")):
        try:
            label = (sys_path / "name").read_text(encoding="utf-8").casefold()
        except OSError:
            continue
        if needle in label:
            return f"/dev/{sys_path.name}"
    return None


class UsbWebcam(Node):
    def __init__(self) -> None:
        super().__init__("experiment_usb_webcam")
        self.declare_parameter("device", "auto")
        self.declare_parameter("device_name_match", "trust")
        self.declare_parameter("image_topic", "/experiment_camera/image_raw")
        self.declare_parameter("frame_id", "experiment_camera_optical_frame")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 20.0)

        self._bridge = CvBridge()
        self._capture = None
        self._device_path = None
        self._last_missing_log_ns = 0
        self._publisher = self.create_publisher(
            Image,
            str(self.get_parameter("image_topic").value),
            qos_profile_sensor_data,
        )
        fps = max(float(self.get_parameter("fps").value), 1.0)
        self._timer = self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f"Trust webcam publisher ready on {self.get_parameter('image_topic').value}"
        )

    def _resolve_device(self) -> str | None:
        configured = str(self.get_parameter("device").value)
        if configured != "auto":
            return configured if os.path.exists(configured) else None
        return find_video_device(str(self.get_parameter("device_name_match").value))

    def _open(self) -> bool:
        device = self._resolve_device()
        if device is None:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_missing_log_ns > 10_000_000_000:
                self.get_logger().warning(
                    "Trust webcam not found; connect it and publishing will start automatically"
                )
                self._last_missing_log_ns = now_ns
            return False

        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.get_parameter("width").value))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter("height").value))
        capture.set(cv2.CAP_PROP_FPS, float(self.get_parameter("fps").value))
        if not capture.isOpened():
            capture.release()
            return False
        self._capture = capture
        self._device_path = device
        self.get_logger().info(f"Publishing Trust webcam from {device}")
        return True

    def _tick(self) -> None:
        if self._capture is None and not self._open():
            return
        ok, frame = self._capture.read()
        if not ok:
            self.get_logger().warning(f"Lost webcam {self._device_path}; retrying")
            self._capture.release()
            self._capture = None
            self._device_path = None
            return
        message = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        self._publisher.publish(message)

    def destroy_node(self) -> bool:
        if self._capture is not None:
            self._capture.release()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = UsbWebcam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
