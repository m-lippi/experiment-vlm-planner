#!/usr/bin/env python3
"""
test_pick_place.py — Execution layer test: GazeboOracle → PickPrimitive → PlacePrimitive.

Bypasses VLM and PDDL — tests only the robot execution stack:
  1. GazeboOracle queries Gazebo for object poses
  2. PickPrimitive executes a top-down grasp on 'red_cup'
  3. PlacePrimitive deposits the cup on 'shelf_b'

Prerequisites (all in the container):
  - simulation.launch.py running (Gazebo + MoveIt2 + Orchestrator or standalone)
  - ROS 2 workspace sourced
  - VLMRP_REPO_ROOT=/workspace (default in docker-compose)

Usage:
  python3 /workspace/scripts/test_pick_place.py
"""

from __future__ import annotations

import sys
import os
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive

# Allow imports from /workspace (simulation/, planner/, vlm/)
_REPO_ROOT = os.environ.get("VLMRP_REPO_ROOT", "/workspace")
sys.path.insert(0, _REPO_ROOT)

try:
    from vlm_robot_planner.moveit2_client import MoveIt2Client as MoveIt2
    from vlm_robot_planner.primitives.base import ARM_JOINT_NAMES
    from vlm_robot_planner.primitives.pick import PickPrimitive
    from vlm_robot_planner.primitives.place import PlacePrimitive
    from simulation.oracle.world_state import GazeboOracle, GazeboAttach
except ImportError as e:
    sys.exit(
        f"[ERROR] Import failed — run inside the container with the workspace sourced.\n{e}"
    )


# Objects to test
_PICK_OBJECT   = "red_cup"
_PLACE_OBJECT  = "shelf_b"

# Ready joint configuration from SRDF
_READY_JOINTS = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]


def setup_planning_scene(node: Node) -> None:
    """Publish table as collision object to MoveIt2's planning scene."""
    pub = node.create_publisher(CollisionObject, "/collision_object", 10)
    time.sleep(0.3)

    co = CollisionObject()
    co.header.frame_id = "panda_link0"
    co.header.stamp    = node.get_clock().now().to_msg()
    co.id              = "table"

    box            = SolidPrimitive()
    box.type       = SolidPrimitive.BOX
    box.dimensions = [1.20, 1.00, 0.77]

    p              = Pose()
    p.position.x   = 0.30
    p.position.y   = 0.00
    p.position.z   = -0.385
    p.orientation.w = 1.0

    co.primitives      = [box]
    co.primitive_poses = [p]
    co.operation       = CollisionObject.ADD

    for _ in range(5):
        pub.publish(co)
        time.sleep(0.1)

    print("[OK]   Table collision object published.")
    time.sleep(0.5)


def main() -> None:
    rclpy.init()
    node = Node("pick_place_test")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print("\n=== Pick + Place execution test ===\n")
    print("[INFO] Waiting 2 s for MoveIt2 to settle…")
    time.sleep(2.0)

    # ── Planning scene ────────────────────────────────────────────────────────
    print("[INFO] Publishing table collision object…")
    setup_planning_scene(node)

    # ── MoveIt2 client ─────────────────────────────────────────────────────────
    print("[INFO] Creating MoveIt2 client…")
    cb_group = ReentrantCallbackGroup()
    moveit2  = MoveIt2(
        node              = node,
        joint_names       = ARM_JOINT_NAMES,
        base_link_name    = "panda_link0",
        end_effector_name = "panda_hand",
        group_name        = "panda_arm",
        callback_group    = cb_group,
    )
    moveit2.max_velocity     = 0.3
    moveit2.max_acceleration = 0.3
    print("[OK]   MoveIt2 ready.\n")

    # ── GazeboOracle ──────────────────────────────────────────────────────────
    print("[INFO] Connecting to GazeboOracle…")
    oracle = GazeboOracle(node=node, reference_frame="panda_link0")
    time.sleep(0.5)

    # ── Query object poses ────────────────────────────────────────────────────
    print(f"\n[INFO] Querying pose of '{_PICK_OBJECT}'…")
    pick_pose = oracle.get_pose(_PICK_OBJECT)
    if pick_pose is None:
        print(f"[FAIL] Oracle returned None for '{_PICK_OBJECT}' — aborting.")
        _shutdown(executor, spin_thread, node)
        sys.exit(1)
    p = pick_pose.position
    print(f"[OK]   {_PICK_OBJECT}: panda_link0 ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")

    print(f"[INFO] Querying pose of '{_PLACE_OBJECT}'…")
    place_pose = oracle.get_pose(_PLACE_OBJECT)
    if place_pose is None:
        print(f"[FAIL] Oracle returned None for '{_PLACE_OBJECT}' — aborting.")
        _shutdown(executor, spin_thread, node)
        sys.exit(1)
    p = place_pose.position
    print(f"[OK]   {_PLACE_OBJECT}: panda_link0 ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")

    # ── Build pose_data dicts (same format as Orchestrator._dispatch) ─────────
    pick_pose_data  = {"position": pick_pose.position,  "orientation": pick_pose.orientation}
    place_pose_data = {"position": place_pose.position, "orientation": place_pose.orientation}

    # ── Simulated attachment ──────────────────────────────────────────────────
    print("[INFO] Connecting to GazeboAttach…")
    attach = GazeboAttach(node=node, eef_frame="panda_hand")
    time.sleep(0.5)

    # ── Primitives ────────────────────────────────────────────────────────────
    pick  = PickPrimitive(node, moveit2, attach=attach)
    place = PlacePrimitive(node, moveit2, attach=attach)

    results: dict[str, bool] = {}

    # Republish planning scene after MoveIt2 is connected
    setup_planning_scene(node)

    # ── Pick ──────────────────────────────────────────────────────────────────
    print(f"\n--- PICK: '{_PICK_OBJECT}' ---")
    ok = pick.execute(_PICK_OBJECT, pick_pose_data)
    results["pick"] = ok
    print(f"  {'[OK]  ' if ok else '[FAIL]'} pick('{_PICK_OBJECT}')\n")

    if not ok:
        print("[WARN] Pick failed — skipping place.")
        _print_summary(results)
        _shutdown(executor, spin_thread, node)
        sys.exit(1)

    time.sleep(1.0)

    # ── Place ─────────────────────────────────────────────────────────────────
    print(f"--- PLACE: '{_PICK_OBJECT}' onto '{_PLACE_OBJECT}' ---")
    ok = place.execute(_PLACE_OBJECT, place_pose_data)
    results["place"] = ok
    print(f"  {'[OK]  ' if ok else '[FAIL]'} place('{_PLACE_OBJECT}')\n")

    _print_summary(results)
    _shutdown(executor, spin_thread, node)
    sys.exit(0 if all(results.values()) else 1)


def _print_summary(results: dict[str, bool]) -> None:
    print("=== Results ===")
    all_pass = True
    for label, passed in results.items():
        t = "PASS" if passed else "FAIL"
        print(f"  {t}  {label}")
        if not passed:
            all_pass = False
    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}\n")


def _shutdown(executor, spin_thread, node) -> None:
    executor.shutdown(timeout_sec=2.0)
    spin_thread.join(timeout=3.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
