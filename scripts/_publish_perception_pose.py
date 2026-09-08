#!/usr/bin/env python3
"""
_publish_perception_pose.py — Runs INSIDE the Docker container.

Publishes one PoseStamped to /perception/object_pose so the orchestrator
can use the perception-estimated 3D pose instead of the GazeboOracle.

The object name is encoded in header.frame_id.
The pose is in panda_link0 frame.

Object height encoding: orientation.z carries the estimated object height in
metres (from DINO bbox + camera geometry, Phase 2+).  orientation.z == 0.0
means height unknown → orchestrator uses the fixed fallback.
The quaternion is intentionally non-unit when height_m > 0 — the orchestrator
reads orientation.z as a sideband, not as a rotation.

Usage (called by run_loop_host.py via docker exec):
    python3 _publish_perception_pose.py --object red_cup --x 0.3 --y 0.1 --z 0.06
    python3 _publish_perception_pose.py --object red_cup --x 0.3 --y 0.1 --z 0.06 --height_m 0.12
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object",    required=True, help="PDDL object name")
    parser.add_argument("--x",         type=float, required=True)
    parser.add_argument("--y",         type=float, required=True)
    parser.add_argument("--z",         type=float, required=True)
    parser.add_argument("--height_m",  type=float, default=None,
                        help="Estimated object height in metres (Phase 2+). "
                             "Encoded in orientation.z; 0.0 means unknown.")
    parser.add_argument(
        "--ack-timeout", type=float, default=30.0,
        help="Seconds to wait for confirmation that the pose was cached",
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("_perception_pose_pub")
    pub  = node.create_publisher(PoseStamped, "/perception/object_pose", 10)
    ack: dict | None = None

    msg                      = PoseStamped()
    msg.header.stamp         = node.get_clock().now().to_msg()
    msg.header.frame_id      = args.object   # object name carried in frame_id
    msg.pose.position.x      = args.x
    msg.pose.position.y      = args.y
    msg.pose.position.z      = args.z
    # Sideband: orientation.z carries object height (0.0 = unknown).
    msg.pose.orientation.z   = float(args.height_m) if args.height_m else 0.0
    msg.pose.orientation.w   = 1.0

    def _on_ack(ack_msg) -> None:
        nonlocal ack
        try:
            candidate = json.loads(ack_msg.data)
            if (
                candidate.get("object") == args.object
                and candidate.get("stamp_sec") == msg.header.stamp.sec
                and candidate.get("stamp_nanosec") == msg.header.stamp.nanosec
            ):
                ack = candidate
        except Exception:
            pass

    from std_msgs.msg import String
    from rclpy.qos import QoSProfile, DurabilityPolicy
    ack_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ack_sub = node.create_subscription(
        String, "/perception/object_pose_ack", _on_ack, ack_qos
    )

    discovery_deadline = time.monotonic() + 10.0
    while (
        (
            pub.get_subscription_count() == 0
            or node.count_publishers("/perception/object_pose_ack") == 0
        )
        and time.monotonic() < discovery_deadline
    ):
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        print(
            "[ERROR] No perception subscriber discovered after 10s",
            file=sys.stderr,
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    if node.count_publishers("/perception/object_pose_ack") == 0:
        print(
            "[ERROR] No perception acknowledgement publisher discovered "
            "after 10s",
            file=sys.stderr,
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    discovery_settle_deadline = time.monotonic() + 0.25
    while time.monotonic() < discovery_settle_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    # Pose updates are idempotent. Keep the writer alive and resend the exact
    # same timestamped pose until its matching cache acknowledgement arrives.
    # This tolerates loss in either direction without creating a new detection.
    ack_deadline = time.monotonic() + args.ack_timeout
    next_publish = 0.0
    publish_count = 0
    while ack is None and time.monotonic() < ack_deadline:
        now = time.monotonic()
        if now >= next_publish:
            pub.publish(msg)
            publish_count += 1
            next_publish = now + 0.5
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()
    if ack is None:
        print(
            f"[ERROR] Orchestrator did not acknowledge pose for {args.object} "
            f"within {args.ack_timeout:.1f}s after {publish_count} publishes",
            file=sys.stderr,
        )
        sys.exit(1)
    height_str = f", h={args.height_m:.3f}m" if args.height_m else ""
    print(
        f"[OK] Perception pose published: {args.object} → "
        f"({args.x:.3f}, {args.y:.3f}, {args.z:.3f}) {height_str}"
        f"(acknowledged after {publish_count} publish(es))"
    )


if __name__ == "__main__":
    main()
