#!/usr/bin/env python3
"""Interactive smoke test for every robot primitive on the real FR3.

The test uses the same MoveIt and GripperCommand interfaces as the
orchestrator, so arm trajectories and gripper goals pass through the ROS 2 to
ROS 1 adapters.  Motion primitives that need a target use a virtual point near
the end-effector pose captured at startup.  Pick and place run last and require
the operator to put a lightweight object between the fingers.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections.abc import Callable

# base.py selects joint names and action endpoints when it is imported.
os.environ["VLM_ROBOT"] = "fr3"

import rclpy
from geometry_msgs.msg import Point, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import JointState
import tf2_ros

from vlm_robot_planner.moveit2_client import MoveIt2Client
from vlm_robot_planner.primitives.base import (
    ARM_GROUP,
    ARM_JOINT_NAMES,
    BASE_FRAME,
    EEF_LINK,
)
from vlm_robot_planner.primitives.cut import CutPrimitive
from vlm_robot_planner.primitives.look_at import LookAtPrimitive
from vlm_robot_planner.primitives.navigate_to import NavigateToPrimitive
from vlm_robot_planner.primitives.pick import PickPrimitive
from vlm_robot_planner.primitives.place import PlacePrimitive
from vlm_robot_planner.primitives.pour import PourPrimitive
from vlm_robot_planner.primitives.stir import StirPrimitive
from vlm_robot_planner.primitives.tilt import TiltPrimitive


ALL_TESTS = [
    "navigate_to",
    "gripper",
    "look_at",
    "tilt",
    "pour",
    "stir",
    "cut",
    "pick",
    "place",
]

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


def pose_data(point: Point) -> dict:
    return {
        "position": point,
        "orientation": Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    }


class PrimitiveTestNode(Node):
    def __init__(self, velocity: float) -> None:
        super().__init__("real_primitive_test")
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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pick = PickPrimitive(self, self.moveit, tf_buffer=self.tf_buffer)
        self.place = PlacePrimitive(self, self.moveit, tf_buffer=self.tf_buffer)
        self.look_at = LookAtPrimitive(self, self.moveit, tf_buffer=self.tf_buffer)
        self.navigate_to = NavigateToPrimitive(self)
        self.pour = PourPrimitive(self, self.moveit, tf_buffer=self.tf_buffer)
        self.stir = StirPrimitive(self, self.moveit, tf_buffer=self.tf_buffer)
        self.tilt = TiltPrimitive(self, self.moveit, tf_buffer=self.tf_buffer)
        self.cut = CutPrimitive(self, self.moveit, tf_buffer=self.tf_buffer)

        self._joint_event = threading.Event()
        self._joints: list[float] | None = None
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_state,
            SENSOR_QOS,
            callback_group=callback_group,
        )

    def _on_joint_state(self, msg: JointState) -> None:
        positions = dict(zip(msg.name, msg.position))
        if not all(name in positions for name in ARM_JOINT_NAMES):
            return
        self._joints = [float(positions[name]) for name in ARM_JOINT_NAMES]
        self._joint_event.set()

    def wait_for_joints(self, timeout: float = 10.0) -> list[float] | None:
        self._joint_event.wait(timeout)
        return list(self._joints) if self._joints is not None else None

    def current_eef(self) -> Point | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                EEF_LINK,
                Time(),
                timeout=Duration(seconds=5.0),
            ).transform.translation
        except Exception as error:
            self.get_logger().error(f"Cannot resolve {BASE_FRAME} -> {EEF_LINK}: {error}")
            return None
        return Point(x=transform.x, y=transform.y, z=transform.z)

    def interfaces_ready(self) -> list[str]:
        missing = []
        if not self.moveit._client.wait_for_server(timeout_sec=5.0):
            missing.append("/move_action")
        if not self.moveit._execute_client.wait_for_server(timeout_sec=5.0):
            missing.append("/execute_trajectory")
        if not self.moveit._cartesian_client.wait_for_service(timeout_sec=5.0):
            missing.append("/compute_cartesian_path")
        if not self.pick._gripper_client.wait_for_server(timeout_sec=5.0):
            missing.append("/franka_gripper/gripper_action")
        return missing

    def restore_joints(self, joints: list[float]) -> bool:
        self.get_logger().info("Restoring the startup joint configuration")
        self.moveit.move_to_configuration(joints)
        return self.moveit.wait_until_executed(timeout=45.0)


def confirm(name: str, description: str, no_confirm: bool) -> bool:
    print(f"\n{'=' * 72}")
    print(f"PRIMITIVE TEST: {name}")
    print(description)
    if no_confirm:
        return True
    try:
        answer = input("Press ENTER to execute, or type 's' to skip: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer not in {"s", "skip"}


def run_step(
    name: str,
    description: str,
    action: Callable[[], bool],
    no_confirm: bool,
) -> bool | None:
    if not confirm(name, description, no_confirm):
        print(f"[SKIP] {name}")
        return None
    try:
        result = bool(action())
    except Exception as error:
        print(f"[FAIL] {name}: {type(error).__name__}: {error}")
        return False
    print(f"[{'OK' if result else 'FAIL'}] {name}")
    return result


def run(args: argparse.Namespace) -> int:
    rclpy.init()
    node = PrimitiveTestNode(args.velocity)
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        joints = node.wait_for_joints()
        eef = node.current_eef()
        missing = node.interfaces_ready()
        if joints is None:
            print("[FAIL] No complete FR3 state received on /joint_states.")
            return 1
        if eef is None:
            print(f"[FAIL] No TF from {BASE_FRAME} to {EEF_LINK}.")
            return 1
        if missing:
            print("[FAIL] Missing interfaces: " + ", ".join(missing))
            return 1

        selected = args.only or ALL_TESTS
        print("\nReal FR3 primitive test")
        print(f"  selected : {', '.join(selected)}")
        print(f"  velocity : {args.velocity:.0%}")
        print(f"  startup EEF: ({eef.x:+.3f}, {eef.y:+.3f}, {eef.z:+.3f}) m")
        print("[OK] Joint states, TF, MoveIt actions/services, and gripper action are ready.")

        if args.check_only:
            print("[OK] Check-only completed; no commands were sent.")
            return 0

        # Pick's top-down grasp pose is object.z + 0.15. This makes the grasp
        # pose equal to the EEF translation captured above.
        # [0.411, 0.054, 0.515]

        # object_point = Point(x=eef.x, y=eef.y, z=eef.z - 0.15)

        
        #NOTA: su z sottraggo 0.05 dato che grasp è +0.05 rispetto alla posizione dell'oggetto
        object_point = Point(x=0.427, y=0.055, z=0.497-0.05)
        # Place's release pose is target.z + 0.22.
        # place_point = Point(
        #     x=eef.x + args.place_x,
        #     y=eef.y + args.place_y,
        #     z=eef.z - 0.1,
        # )
        place_point = Point(x=object_point.x + args.place_x, y=object_point.y + args.place_y, z=0.497)
        # Stir and cut are exercised in free space around the startup EEF,
        # without a tool or physical surface.
        work_point = Point(x=eef.x, y=eef.y, z=eef.z)

        results: dict[str, bool | None] = {}

        def execute_and_restore(action: Callable[[], bool]) -> bool:
            ok = False
            try:
                ok = bool(action())
            finally:
                restored = node.restore_joints(joints)
            return ok and restored

        def test_gripper() -> bool:
            if not node.pick.open_gripper():
                return False
            if not args.no_confirm:
                input(
                    "Insert/verify the object between the fingers, then press "
                    "ENTER to grasp: "
                )
            if not node.pick.close_gripper():
                return False
            return node.pick.open_gripper()

        def test_prepositioned_pick() -> bool:
            def confirm_object_ready() -> bool:
                if args.no_confirm:
                    return True
                input(
                    "Insert/verify the object between the fingers, then press "
                    "ENTER to grasp: "
                )
                return True

            return node.pick.execute_prepositioned(
                "primitive_test_object",
                lift_m=args.pick_lift,
                before_grasp=confirm_object_ready,
            )

        actions: dict[str, tuple[str, Callable[[], bool]]] = {
            "navigate_to": (
                "Calls the current navigation stub; the mobile base will not move.",
                lambda: node.navigate_to.execute("test_destination"),
            ),
            "gripper": (
                "Opens, grasps, then opens the gripper after a successful grasp. "
                "Keep fingers clear until prompted to insert the object.",
                test_gripper,
            ),
            "look_at": (
                "Moves to the look-at configuration, then restores the startup joints.",
                lambda: execute_and_restore(
                    lambda: node.look_at.execute("virtual_target", pose_data(object_point))
                ),
            ),
            "tilt": (
                f"Tilts at the current position by {args.tilt_angle:.1f} degrees, "
                "then restores the startup joints.",
                lambda: execute_and_restore(
                    lambda: node.tilt.execute(
                        "virtual_object", angle_deg=args.tilt_angle
                    )
                ),
            ),
            "pour": (
                "Runs the pour tilt without an object or target, then restores startup joints.",
                lambda: execute_and_restore(
                    lambda: node.pour.execute("virtual_target")
                ),
            ),
            "stir": (
                "Runs the stirring path in free space around the startup EEF, then restores.",
                lambda: execute_and_restore(
                    lambda: node.stir.execute("virtual_container", pose_data(work_point))
                ),
            ),
            "cut": (
                "Runs three cutting strokes in free space near the startup EEF, then restores.",
                lambda: execute_and_restore(
                    lambda: node.cut.execute("virtual_object", pose_data(work_point))
                ),
            ),
            # "pick": (
            #     "Place a LIGHTWEIGHT object between the fingers at the startup EEF pose. "
            #     f"PickPrimitive grasps without repositioning, then lifts "
            #     f"{args.pick_lift:.3f} m vertically.",
            #     test_prepositioned_pick,
            # ),
            "pick": (
                "Place a LIGHTWEIGHT object between the fingers at the startup EEF pose. "
                "The test opens, approaches, grasps, attaches it in MoveIt, and retreats.",
                lambda: node.pick.execute(
                    "primitive_test_object",
                    pose_data(object_point),
                    grasp_mode="top_down",
                    object_height_m=args.object_height,
                ),
            ),
            "place": (
                f"Places the held object at offset x={args.place_x:+.3f}, "
                f"y={args.place_y:+.3f} m and returns to ready.",
                lambda: node.place.execute(
                    "primitive_test_place", pose_data(place_point)
                ),
            ),
        }

        for name in selected:
            description, action = actions[name]
            results[name] = run_step(name, description, action, args.no_confirm)
            if results[name] is False and not args.continue_on_failure:
                print("Stopping after failure. Use --continue-on-failure to continue.")
                break

        print(f"\n{'=' * 72}")
        print("SUMMARY")
        for name in selected:
            if name not in results:
                state = "NOT RUN"
            elif results[name] is None:
                state = "SKIPPED"
            else:
                state = "PASS" if results[name] else "FAIL"
            print(f"  {name:12s} {state}")

        return 1 if any(result is False for result in results.values()) else 0
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        choices=ALL_TESTS,
        help="Run only this test; repeat the option to select several.",
    )
    parser.add_argument("--velocity", type=float, default=0.10)
    parser.add_argument("--place-x", type=float, default=0.0)
    parser.add_argument("--place-y", type=float, default=0.10)
    parser.add_argument("--object-height", type=float, default=0.08)
    parser.add_argument("--pick-lift", type=float, default=0.05)
    parser.add_argument("--tilt-angle", type=float, default=15.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--no-confirm", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.velocity <= 0.15:
        parser.error("--velocity must be in (0, 0.15] for the physical test")
    if abs(args.place_x) > 0.20 or abs(args.place_y) > 0.20:
        parser.error("place offsets are limited to +/-0.20 m")
    if not 0.04 <= args.object_height <= 0.25:
        parser.error("--object-height must be between 0.04 and 0.25 m")
    if not 0.02 <= args.pick_lift <= 0.15:
        parser.error("--pick-lift must be between 0.02 and 0.15 m")
    if not 1.0 <= abs(args.tilt_angle) <= 30.0:
        parser.error("absolute --tilt-angle must be between 1 and 30 degrees")
    sys.exit(run(args))


if __name__ == "__main__":
    main()
