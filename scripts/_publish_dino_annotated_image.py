#!/usr/bin/env python3
"""Send a host-generated DINO overlay to the orchestrator's image relay."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import rclpy
from PIL import Image as PilImage
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--frame-id", default="overview_camera_color_optical_frame")
    parser.add_argument("--ack-timeout", type=float, default=15.0)
    args = parser.parse_args()

    try:
        rgb = np.asarray(PilImage.open(args.image).convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        print(f"[ERROR] Cannot load annotated image: {exc}", file=sys.stderr)
        raise SystemExit(1)

    rclpy.init()
    node = Node("_dino_annotated_image_sender")
    pub = node.create_publisher(
        Image, "/perception/dino_annotated_image_input", 10
    )

    msg = Image()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = args.frame_id
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = rgb.tobytes()

    acknowledged = False

    def _on_ack(ack_msg: String) -> None:
        nonlocal acknowledged
        try:
            data = json.loads(ack_msg.data)
            acknowledged = (
                data.get("stamp_sec") == msg.header.stamp.sec
                and data.get("stamp_nanosec") == msg.header.stamp.nanosec
                and data.get("published") is True
            )
        except Exception:
            pass

    from rclpy.qos import DurabilityPolicy, QoSProfile
    ack_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ack_sub = node.create_subscription(
        String, "/perception/dino_annotated_image_ack", _on_ack, ack_qos
    )

    discovery_deadline = time.monotonic() + 10.0
    while (
        (
            pub.get_subscription_count() == 0
            or node.count_publishers(
                "/perception/dino_annotated_image_ack"
            ) == 0
        )
        and time.monotonic() < discovery_deadline
    ):
        rclpy.spin_once(node, timeout_sec=0.05)

    if pub.get_subscription_count() == 0:
        print("[ERROR] DINO image relay subscriber not found", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)
    if node.count_publishers("/perception/dino_annotated_image_ack") == 0:
        print("[ERROR] DINO image ACK publisher not found", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    settle_deadline = time.monotonic() + 0.25
    while time.monotonic() < settle_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    ack_deadline = time.monotonic() + args.ack_timeout
    next_publish = 0.0
    publish_count = 0
    while not acknowledged and time.monotonic() < ack_deadline:
        now = time.monotonic()
        if now >= next_publish:
            pub.publish(msg)
            publish_count += 1
            next_publish = now + 0.5
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()

    if not acknowledged:
        print(
            f"[ERROR] DINO image was not acknowledged after "
            f"{publish_count} publishes",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(
        f"[OK] DINO annotated image published: {msg.width}x{msg.height} "
        f"after {publish_count} publish(es)"
    )


if __name__ == "__main__":
    main()
