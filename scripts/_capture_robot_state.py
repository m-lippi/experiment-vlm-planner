#!/usr/bin/env python3
"""Capture arm joint state and base-to-end-effector TF as one JSON object."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


def _stamp(stamp) -> dict:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


class RobotStateCapture(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("capture_robot_state")
        self.args = args
        self.joint_message: JointState | None = None
        self.transform = None
        self.tf_error = "not received"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            JointState,
            args.joint_topic,
            self._on_joints,
            qos_profile_sensor_data,
        )

    def _on_joints(self, message: JointState) -> None:
        if all(name in message.name for name in self.args.arm_joints):
            self.joint_message = message

    def try_transform(self) -> None:
        try:
            self.transform = self.tf_buffer.lookup_transform(
                self.args.base_frame,
                self.args.eef_frame,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            self.tf_error = ""
        except Exception as exc:
            self.tf_error = str(exc)

    def result(self) -> dict:
        errors = []
        joints = None
        if self.joint_message is None:
            errors.append(f"no complete arm state on {self.args.joint_topic}")
        else:
            message = self.joint_message
            index = {name: offset for offset, name in enumerate(message.name)}

            def values(sequence) -> dict:
                return {
                    name: float(sequence[index[name]])
                    for name in self.args.arm_joints
                    if index[name] < len(sequence)
                }

            joints = {
                "topic": self.args.joint_topic,
                "frame_id": message.header.frame_id,
                "message_stamp": _stamp(message.header.stamp),
                "position_rad": values(message.position),
                "velocity_rad_s": values(message.velocity),
                "effort": values(message.effort),
            }

        eef_pose = None
        if self.transform is None:
            errors.append(
                f"no TF {self.args.base_frame} -> {self.args.eef_frame}: "
                f"{self.tf_error}"
            )
        else:
            transform = self.transform
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            eef_pose = {
                "parent_frame": transform.header.frame_id,
                "child_frame": transform.child_frame_id,
                "message_stamp": _stamp(transform.header.stamp),
                "position_m": {
                    "x": float(translation.x),
                    "y": float(translation.y),
                    "z": float(translation.z),
                },
                "orientation_xyzw": {
                    "x": float(rotation.x),
                    "y": float(rotation.y),
                    "z": float(rotation.z),
                    "w": float(rotation.w),
                },
            }

        return {
            "available": not errors,
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "robot": self.args.robot,
            "arm_joints": joints,
            "end_effector_pose": eef_pose,
            "errors": errors,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("fr3", "panda"), required=True)
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    prefix = "fr3" if args.robot == "fr3" else "panda"
    args.arm_joints = [f"{prefix}_joint{index}" for index in range(1, 8)]
    args.base_frame = f"{prefix}_link0"
    args.eef_frame = "fr3_EE" if args.robot == "fr3" else "panda_hand"
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = RobotStateCapture(args)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
            node.try_transform()
            if node.joint_message is not None and node.transform is not None:
                break
        print(json.dumps(node.result(), separators=(",", ":")))
        return 0
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
