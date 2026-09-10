"""Record a ROS 2 image topic to a playable MP4 experiment artifact."""

from __future__ import annotations

from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class WebcamRecorder(Node):
    def __init__(self) -> None:
        super().__init__("experiment_webcam_recorder")
        self.declare_parameter("image_topic", "/experiment_camera/image_raw")
        self.declare_parameter("output_path", "")
        self.declare_parameter("fps", 20.0)
        output = str(self.get_parameter("output_path").value)
        if not output:
            raise ValueError("output_path parameter is required")
        self._output = Path(output)
        self._output.parent.mkdir(parents=True, exist_ok=True)
        self._fps = max(float(self.get_parameter("fps").value), 1.0)
        self._bridge = CvBridge()
        self._writer = None
        self._frames = 0
        topic = str(self.get_parameter("image_topic").value)
        self._subscription = self.create_subscription(
            Image, topic, self._on_image, qos_profile_sensor_data
        )
        self.get_logger().info(f"Recording {topic} to {self._output}")

    def _on_image(self, message: Image) -> None:
        frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self._output),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                (width, height),
            )
            if not self._writer.isOpened():
                self._writer.release()
                self._writer = None
                raise RuntimeError(f"Cannot open MP4 writer for {self._output}")
        self._writer.write(frame)
        self._frames += 1

    def destroy_node(self) -> bool:
        if self._writer is not None:
            self._writer.release()
            if rclpy.ok():
                self.get_logger().info(
                    f"Saved {self._frames} webcam frames to {self._output}"
                )
        elif self._output.exists():
            self._output.unlink()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = WebcamRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy's SIGINT handler may already have shut the context down.
        if rclpy.ok():
            rclpy.shutdown()
