#!/usr/bin/env python3
"""Interactive real-robot pick/place test from the current gripper pose.

The object must already be positioned between the open fingers. The current
``fr3_hand`` pose is treated as the pick pose; the place pose is the same
height and orientation with a configurable XY offset.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import threading

os.environ.setdefault("VLM_ROBOT", "fr3")

import rclpy
from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

from vlm_robot_planner.moveit2_client import MoveIt2Client
from vlm_robot_planner.primitives.base import (
    ARM_GROUP,
    ARM_JOINT_NAMES,
    BASE_FRAME,
    EEF_LINK,
)


class PickPlaceTest(Node):
    def __init__(self, velocity: float) -> None:
        super().__init__("real_pick_place_test")
        callback_group = ReentrantCallbackGroup()
        self.moveit = MoveIt2Client(
            node=self,
            joint_names=ARM_JOINT_NAMES,
            base_link_name=BASE_FRAME,
            end_effector_name=EEF_LINK,
            group_name=ARM_GROUP,
            callback_group=callback_group,
        )
        self.moveit.max_velocity = velocity
        self.moveit.max_acceleration = velocity
        self.gripper = ActionClient(
            self,
            GripperCommand,
            "/franka_gripper/gripper_action",
            callback_group=callback_group,
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def current_pose(self) -> Pose | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME, EEF_LINK, Time(), timeout=Duration(seconds=10.0)
            ).transform
        except Exception as error:
            self.get_logger().error(f"Cannot read current {EEF_LINK} pose: {error}")
            return None
        pose = Pose()
        pose.position.x = transform.translation.x
        pose.position.y = transform.translation.y
        pose.position.z = transform.translation.z
        pose.orientation = transform.rotation
        return pose

    def command_gripper(self, width: float, effort: float) -> bool:
        if not self.gripper.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("ROS 2 gripper adapter is unavailable")
            return False
        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = effort
        accepted = threading.Event()
        completed = threading.Event()
        holder = {"goal": None, "result": None}

        def on_result(future):
            holder["result"] = future.result()
            completed.set()

        def on_goal(future):
            holder["goal"] = future.result()
            accepted.set()
            if holder["goal"] is not None and holder["goal"].accepted:
                holder["goal"].get_result_async().add_done_callback(on_result)
            else:
                completed.set()

        self.gripper.send_goal_async(goal).add_done_callback(on_goal)
        if not accepted.wait(5.0) or holder["goal"] is None or not holder["goal"].accepted:
            self.get_logger().error("Gripper goal was rejected")
            return False
        if not completed.wait(20.0):
            self.get_logger().error("Gripper command timed out")
            holder["goal"].cancel_goal_async()
            return False
        response = holder["result"]
        return response is not None and response.status == GoalStatus.STATUS_SUCCEEDED

    def move_linear(self, pose: Pose, label: str) -> bool:
        self.get_logger().info(
            f"Moving {label}: ({pose.position.x:.3f}, "
            f"{pose.position.y:.3f}, {pose.position.z:.3f})"
        )
        self.moveit.move_cartesian_waypoints([pose], max_step=0.005, min_fraction=0.95)
        if not self.moveit.wait_until_executed(timeout=45.0):
            self.get_logger().error(f"Motion failed: {label}")
            return False
        return True


def run(args: argparse.Namespace) -> int:
    rclpy.init()
    node = PickPlaceTest(args.velocity)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        pick = node.current_pose()
        if pick is None:
            return 1

        lift_pick = copy.deepcopy(pick)
        lift_pick.position.z += args.lift
        lift_place = copy.deepcopy(lift_pick)
        lift_place.position.x += args.place_x
        lift_place.position.y += args.place_y
        place = copy.deepcopy(pick)
        place.position.x += args.place_x
        place.position.y += args.place_y

        print("\nReal pick/place test")
        print(f"  pick       : ({pick.position.x:+.3f}, {pick.position.y:+.3f}, {pick.position.z:+.3f}) m")
        print(f"  lift       : {args.lift:.3f} m")
        print(f"  place delta: x={args.place_x:+.3f}, y={args.place_y:+.3f} m")
        print(f"  open/close : {args.open_width:.3f}/{args.close_width:.3f} m")

        if args.check_only:
            if not node.gripper.wait_for_server(timeout_sec=5.0):
                print("[FAIL] Gripper adapter action is unavailable.")
                return 1
            print("[OK] TF, MoveIt client, and gripper adapter are available; no motion sent.")
            return 0

        print("\nPlace the object between the fingers at the CURRENT pose.")
        input("Press ENTER to open the gripper and execute the full sequence (Ctrl+C aborts): ")

        if not node.command_gripper(args.open_width, 0.0):
            print("[FAIL] Could not open gripper.")
            return 1
        input("Insert/verify the object between the fingers, then press ENTER to grasp: ")
        if not node.command_gripper(args.close_width, args.effort):
            print("[FAIL] Could not close gripper.")
            return 1
        if not node.move_linear(lift_pick, "vertical lift"):
            return 1
        if not node.move_linear(lift_place, "transfer above place"):
            return 1
        if not node.move_linear(place, "vertical place descent"):
            return 1
        if not node.command_gripper(args.open_width, 0.0):
            print("[FAIL] Could not release object.")
            return 1
        if not node.move_linear(lift_place, "post-place retreat"):
            return 1
        print("[OK] Pick and place completed.")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nAborted by operator.")
        return 130
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--velocity", type=float, default=0.1)
    parser.add_argument("--lift", type=float, default=0.10)
    parser.add_argument("--place-x", type=float, default=0.0)
    parser.add_argument("--place-y", type=float, default=0.10)
    parser.add_argument("--open-width", type=float, default=0.08)
    parser.add_argument("--close-width", type=float, default=0.02)
    parser.add_argument("--effort", type=float, default=20.0)
    args = parser.parse_args()
    if not 0.0 < args.velocity <= 0.2:
        parser.error("--velocity must be in (0, 0.2] for this physical test")
    if not 0.02 <= args.lift <= 0.20:
        parser.error("--lift must be between 0.02 and 0.20 m")
    if abs(args.place_x) > 0.25 or abs(args.place_y) > 0.25:
        parser.error("place offsets are limited to +/-0.25 m")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
