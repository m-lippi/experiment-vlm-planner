#!/usr/bin/env python3
"""
_publish_plan.py — Runs INSIDE the Docker container.

Reads a JSON plan payload from stdin and publishes it to
/vlm_planner/inject_plan so the Orchestrator picks it up.

Called by run_vlm_host.py via:
    docker exec -i vlm_ros2 bash -c "source ... && python3 /workspace/scripts/_publish_plan.py"

Stdin format:
    {"command": "<task>", "vlm_plan": { ... VLMPlan fields ... }}
"""

from __future__ import annotations

import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main() -> None:
    payload = sys.stdin.read().strip()
    if not payload:
        print("[ERROR] No data on stdin.", file=sys.stderr)
        sys.exit(1)

    # Validate JSON before publishing
    try:
        data = json.loads(payload)
        command = data.get("command", "?")
        n_steps = len(data.get("vlm_plan", {}).get("steps", []))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = Node("_plan_injector")
    pub  = node.create_publisher(String, "/vlm_planner/inject_plan", 10)

    # Wait until at least 1 subscriber (orchestrator) is discovered — much
    # more reliable than a fixed sleep, which can fail when DDS discovery
    # takes longer than expected (many active ROS2 nodes in Gazebo sessions).
    deadline = time.time() + 10.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        print("[WARN] No subscriber discovered after 10s — publishing anyway",
              file=sys.stderr)

    msg      = String()
    msg.data = payload
    pub.publish(msg)
    # Spin briefly so DDS flushes the outgoing message before the node exits
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)

    node.destroy_node()
    rclpy.shutdown()

    print(f"[OK] Plan injected: '{command}' ({n_steps} steps)")


if __name__ == "__main__":
    main()
