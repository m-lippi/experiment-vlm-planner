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

import argparse
import json
import sys
import time
import uuid

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ack-timeout", type=float, default=30.0,
        help="Seconds to retry until the orchestrator acknowledges the plan",
    )
    args = parser.parse_args()

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

    request_id = str(data.get("request_id") or uuid.uuid4())
    data["request_id"] = request_id
    payload = json.dumps(data)
    ack: dict | None = None

    def _on_ack(msg: String) -> None:
        nonlocal ack
        try:
            candidate = json.loads(msg.data)
            if candidate.get("request_id") == request_id:
                ack = candidate
        except Exception:
            pass

    from rclpy.qos import QoSProfile, DurabilityPolicy
    ack_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ack_sub = node.create_subscription(
        String, "/vlm_planner/plan_ack", _on_ack, ack_qos
    )

    # Wait until at least 1 subscriber (orchestrator) is discovered — much
    # more reliable than a fixed sleep, which can fail when DDS discovery
    # takes longer than expected (many active ROS2 nodes in Gazebo sessions).
    deadline = time.monotonic() + 10.0
    while (
        (
            pub.get_subscription_count() == 0
            or node.count_publishers("/vlm_planner/plan_ack") == 0
        )
        and time.monotonic() < deadline
    ):
        rclpy.spin_once(node, timeout_sec=0.05)
    if pub.get_subscription_count() == 0:
        print("[ERROR] No orchestrator subscriber discovered after 10s",
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)
    if node.count_publishers("/vlm_planner/plan_ack") == 0:
        print(
            "[ERROR] No orchestrator acknowledgement publisher discovered "
            "after 10s",
            file=sys.stderr,
        )
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Give the reverse endpoint match a moment to propagate to the writer.
    discovery_settle_deadline = time.monotonic() + 0.25
    while time.monotonic() < discovery_settle_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    msg      = String()
    msg.data = payload

    # Retry the same UUID until acknowledged. The orchestrator deduplicates the
    # UUID, so loss of the ACK cannot execute the robot action more than once.
    deadline = time.monotonic() + args.ack_timeout
    next_publish = 0.0
    publish_count = 0
    while ack is None and time.monotonic() < deadline:
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
            f"[ERROR] No plan acknowledgement for request {request_id} "
            f"within {args.ack_timeout:.1f}s after {publish_count} publishes",
            file=sys.stderr,
        )
        sys.exit(1)
    if not ack.get("accepted", False):
        print(
            f"[ERROR] Plan rejected: {ack.get('reason', 'unknown reason')}",
            file=sys.stderr,
        )
        sys.exit(2)

    print(
        f"[OK] Plan accepted: '{command}' ({n_steps} steps), "
        f"request_id={request_id}, publishes={publish_count}"
    )


if __name__ == "__main__":
    main()
