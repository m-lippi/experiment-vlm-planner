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
    args = parser.parse_args()

    rclpy.init()
    node = Node("_perception_pose_pub")
    pub  = node.create_publisher(PoseStamped, "/perception/object_pose", 10)

    msg                      = PoseStamped()
    msg.header.stamp         = node.get_clock().now().to_msg()
    msg.header.frame_id      = args.object   # object name carried in frame_id
    msg.pose.position.x      = args.x
    msg.pose.position.y      = args.y
    msg.pose.position.z      = args.z
    # Sideband: orientation.z carries object height (0.0 = unknown).
    msg.pose.orientation.z   = float(args.height_m) if args.height_m else 0.0
    msg.pose.orientation.w   = 1.0

    # Publish several times: the orchestrator subscriber may need one spin
    # cycle before the message lands in its callback queue.
    for _ in range(5):
        pub.publish(msg)
        time.sleep(0.05)

    node.destroy_node()
    rclpy.shutdown()
    height_str = f", h={args.height_m:.3f}m" if args.height_m else ""
    print(
        f"[OK] Perception pose published: {args.object} → "
        f"({args.x:.3f}, {args.y:.3f}, {args.z:.3f}) panda_link0{height_str}"
    )


if __name__ == "__main__":
    main()
