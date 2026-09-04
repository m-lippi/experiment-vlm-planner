#!/usr/bin/env python3
"""
_get_model_states.py — Runs INSIDE the Docker container.

Reads the current Gazebo scene from /gazebo/model_states and outputs a
JSON list of "interesting" model names (non-infrastructure objects).

Used by run_vlm_host.py (on the host) via docker exec to discover what
objects are in the scene WITHOUT hardcoding a vocabulary — enabling
fully adaptive scene understanding.

Output (stdout):
    {"names": ["red_cup", "blue_box", "shelf_b"]}

Sim-to-real note:
    On the real robot this file is not needed — the Perception Module
    uses OWL-ViT + RealSense depth to ground names without any object
    database.  This script is a simulation-only bridge.
"""

from __future__ import annotations

import json
import sys
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

# Models that are always present in Gazebo but are never graspable objects.
_SKIP = frozenset({
    "ground_plane", "sun", "panda", "overview_camera",
    "table", "world", "sky", "default",
    
})

_TIMEOUT_SEC = 5.0


class _ModelStateReader(Node):
    def __init__(self) -> None:
        super().__init__("_model_state_reader")
        self.names: list[str] | None = None
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self._cb, 1
        )

    def _cb(self, msg: ModelStates) -> None:
        if self.names is not None:
            return
        self.names = {}
        for name, pose in zip(msg.name, msg.pose):
            if name in _SKIP or name.startswith("_"):
                continue
            self.names[name] = {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
            }


def main() -> None:
    rclpy.init()
    node     = _ModelStateReader()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    t0 = time.time()
    while node.names is None and (time.time() - t0) < _TIMEOUT_SEC:
        executor.spin_once(timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    if node.names is None:
        print(
            "[ERROR] /gazebo/model_states not received — is Gazebo running?",
            file=sys.stderr,
        )
        sys.exit(1)

    # Output: {"models": {"red_cup": {"x":…, "y":…, "z":…}, …}}
    print(json.dumps({"models": node.names}))


if __name__ == "__main__":
    main()
