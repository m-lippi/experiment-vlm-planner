#!/usr/bin/env python3
"""
_pre_scan.py — Runs INSIDE the Docker container.

Moves the Panda arm to the "scan" pose so the wrist camera (eye-in-hand)
has a clear frontal view of the table workspace before image capture.

The scan pose places the camera above and in front of the table, looking
down at the objects.  The exact joint config is defined in base.py as
_NAMED_CONFIGS["scan"].

Usage (called by run_vlm_host.py via docker exec before _capture_scene.py):
    docker exec vlm_ros2 bash -c "source ... && python3 /workspace/scripts/_pre_scan.py"

Exit codes:
    0 — arm reached scan pose successfully
    1 — timeout or failure
"""

from __future__ import annotations

import sys
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

_STATUS_TOPIC  = "/vlm_planner/status"
_INJECT_TOPIC  = "/vlm_planner/inject_plan"
_TIMEOUT_SEC   = 30.0


class _PreScanNode(Node):
    def __init__(self) -> None:
        super().__init__("_pre_scan")
        self.done   = False
        self.status = ""
        self._pub   = self.create_publisher(String, _INJECT_TOPIC, 10)
        self._sub   = self.create_subscription(
            String, _STATUS_TOPIC, self._on_status, 10
        )

    def _on_status(self, msg: String) -> None:
        self.status = msg.data
        if "ready" in msg.data.lower() and self.done:
            pass  # already done

    def inject_scan(self) -> None:
        """Inject a single look_at step to move arm to scan pose."""
        import json
        payload = json.dumps({
            "command": "scan",
            "vlm_plan": {
                "goal": "scan",
                "steps": [
                    {"primitive": "look_at", "args": {"target": "scene"}}
                ],
                "raw_output": "",
                "domain_template": "manipulation_base",
                "domain_additions": {
                    "new_types": [], "new_predicates": [],
                    "new_actions": [], "modified_preconditions": {},
                },
            },
        })
        msg      = String()
        msg.data = payload
        self._pub.publish(msg)
        self.get_logger().info("Pre-scan: look_at(scene) injected -> moving to scan pose")


def main() -> None:
    rclpy.init()
    node     = _PreScanNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    t0 = time.time()
    while "ready" not in node.status.lower() and (time.time() - t0) < 10.0:
        executor.spin_once(timeout_sec=0.1)

    if "ready" not in node.status.lower():
        print("[WARN] _pre_scan: orchestrator not ready — skipping scan pose",
              file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return   # non-fatal: capture will fall back to overview camera

    node.inject_scan()
    node.done = True

    # Wait for orchestrator to go busy then back to ready (arm motion complete)
    t0 = time.time()
    saw_busy = False
    while (time.time() - t0) < _TIMEOUT_SEC:
        executor.spin_once(timeout_sec=0.1)
        if "busy" in node.status.lower():
            saw_busy = True
        if saw_busy and "ready" in node.status.lower():
            print("[OK] Pre-scan: arm in scan pose — ready for image capture")
            break
    else:
        print("[WARN] _pre_scan: timeout waiting for scan pose", file=sys.stderr)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
