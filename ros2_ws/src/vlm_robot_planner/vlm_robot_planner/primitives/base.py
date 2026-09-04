"""
Shared MoveIt2 helpers for all arm primitives — uses pymoveit2.

All primitives (pick, place, look_at) inherit from ArmPrimitive.
A single pymoveit2.MoveIt2 instance is created in the Orchestrator
and passed to each primitive to avoid multiple competing planning
interfaces.

The node's executor MUST be MultiThreadedExecutor for pymoveit2 to work.
Gripper control goes through the GripperCommand action directly, using
threading.Event callbacks so it is safe to call from any background thread.

Robot type is selected via VLM_ROBOT environment variable:
  VLM_ROBOT=panda  (default) — Gazebo simulation, panda_joint1…7
  VLM_ROBOT=fr3             — Real FR3, fr3_joint1…7, /franka_gripper
"""

from __future__ import annotations

import os
import threading
from copy import deepcopy

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose, Quaternion
from rclpy.action import ActionClient
from rclpy.node import Node

# ── Robot-specific constants — set at import time from VLM_ROBOT env var ─────
_ROBOT = os.environ.get("VLM_ROBOT", "panda")

if _ROBOT == "fr3":
    ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
    ARM_GROUP       = "fr3_arm"
    BASE_FRAME      = "fr3_link0"
    EEF_LINK        = "fr3_hand"
    _HELD_OBJ_LINK  = "fr3_hand"
    _TOUCH_LINKS    = [
        "fr3_hand", "fr3_leftfinger", "fr3_rightfinger",
        "fr3_link6", "fr3_link7", "fr3_link8",
    ]
    _GRIPPER_ACTION = "/franka_gripper/gripper_action"
else:  # panda — simulation default
    ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
    ARM_GROUP       = "panda_arm"
    BASE_FRAME      = "panda_link0"
    EEF_LINK        = "panda_hand"
    _HELD_OBJ_LINK  = "panda_hand"
    _TOUCH_LINKS    = [
        "panda_hand", "panda_leftfinger", "panda_rightfinger",
        "panda_link6", "panda_link7", "panda_link8",
    ]
    _GRIPPER_ACTION = "/panda_hand_controller/gripper_cmd"

# Named joint configurations (radians) — sourced from moveit_resources_panda SRDF
_NAMED_CONFIGS = {
    "ready": [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854],
    # Scan pose: arm forward-low with camera aimed at the table centre.
    # j1=-0.70 (forward tilt), j3=-2.10 (elbow down), j5=1.40 (wrist).
    # Used as fallback when look_at has no target pose.
    "scan":  [0.0, -0.70, 0.0, -2.10, 0.0, 1.40, 0.7854],

    # Safe retreat: arm pulled up and back, EEF well above table surface.
    # Used as intermediate waypoint after place/pick before returning to "ready".
    # Prevents PILZ PTP from taking paths that dip near table-level objects.
    # Geometry rationale: joint2=-0.9 + joint4=-1.8 → EEF at z≈0.65m above
    # panda_link0 (table surface), safely above any tabletop object.
    # This config is general — does NOT depend on specific scene objects.
    "safe_retreat": [0.0, -0.9, 0.0, -1.8, 0.0, 0.9, 0.7854],

    # Side-grasp seed: "ready" with j7 shifted by -π for the Ry(90°)·Rz(180°)
    # quaternion (x=0.707, y=0, z=0.707, w=0) used in side-grasp pick/pour.
    "side_ready": [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, -2.4],

    # Pour approach seed for tray target (y≈-0.50 in kitchen scene).
    # Problem: safe_retreat has j1=0 → IK for the tray target (0.43,-0.50,0.40)
    # with _SIDE_QUAT converges to j1=+1.13 (arm swings LEFT, "wrong" IK branch).
    # From j1=+1.13 the descent Cartesian path fails at ~40% → OMPL fallback
    # (10-15s) → random joint config → cascading failures.
    # Fix: seed from j1=-0.9 (arm already pointing toward y=-0.50 side) so KDL
    # converges to the natural negative-j1 branch for all subsequent moves.
    # j2/j4/j6 match safe_retreat to keep the arm at a safe height.
    # j7=-2.42 matches side_approach so _SIDE_QUAT IK is trivial from this seed.
    "pour_approach_tray": [-0.9, -0.9, 0.0, -1.8, 0.0, 0.9, -2.42],

    # Pour tilt: joint config for EEF ≈(0.428,-0.498,0.249) with _POUR_TILT_QUAT
    # over the kitchen scene tray (x=0.43, y=-0.50). Derived from the OMPL
    # solution in the kitchen scene. Bypasses KDL IK entirely: the transit
    # lands at j5≈+2.09, but the tilt needs j5≈-0.92 — a 3-rad gap that KDL
    # gradient descent cannot bridge from that seed. OMPL finds the right config
    # but takes ≈12s and leaves the arm in a chaotic state. Using a direct
    # joint-space goal (PILZ PTP, no IK) is deterministic and at _POUR_VEL=0.12
    # the arm sweeps slowly ≈11s, making the tilt motion clearly visible.
    "pour_pre_tilt": [-1.20, 0.80, 0.60, -1.44, -0.92, 1.40, 0.36],

    # Side-grasp natural approach configuration (office scene, can at y≈-0.20).
    # These are the exact joint angles from a successful side-grasp with the OLD
    # quaternion (x=0, y=0.707, z=0, w=0.707), with j7 shifted by -π to match
    # the new quaternion (x=0.707, y=0, z=0.707, w=0) — camera UP.
    # j7 shift: 0.72 - π ≈ -2.42 (within Panda j7 limits [-2.8973, 2.8973]).
    # j1-j6 unchanged → EEF position is identical to the old pre-grasp.
    # Used via move_to_configuration (direct joint control, bypasses IK entirely)
    # so KDL self-collision issues are irrelevant. From this config, PILZ PTP
    # to the grasp pose uses a small joint delta and finds natural IK trivially.
    # Phase 4: replace with a general IK solution seeded from this config.
    "side_approach": [-0.52, 0.58, -0.62, -2.90, 2.03, 1.17, -2.42],
}

_GRIPPER_OPEN   = 0.04   # metres per finger (8 cm total opening)
# Grip position: 0.020 m per finger = 4 cm total opening.
# Covers Phase 2 objects (hammer handle ~4cm, cup ~5cm, cylinder ~4cm).
# GazeboAttach handles the actual attachment regardless of exact closure.
_GRIPPER_CLOSED = 0.020
_GRIPPER_EFFORT = 20.0   # N — enough for lightweight objects

# Top-down grasp orientation: 180° rotation around x → end-effector points down.
_TOP_DOWN_QUAT = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)


# ── W5: AttachedCollisionObject parameters ────────────────────────────────────
# Conservative geometry for any grasped object — NOT specific to scene objects.
# Real robot Phase 2: replace with object geometry from PerceptionModule.
# The cylinder approximates "something held in the gripper" in a general way.
_HELD_OBJ_RADIUS = 0.04   # m — covers graspable objects up to ~8 cm diameter
_HELD_OBJ_HEIGHT = 0.14   # m — covers objects up to ~14 cm tall
# Object centre offset in hand frame (+Z = toward fingertips/object).
_HELD_OBJ_Z_IN_HAND = 0.13   # m — in hand +Z (toward object, away from arm)


class ArmPrimitive:
    # Class-level shared state: which object is currently held in the gripper.
    # Shared across ALL primitive instances (pick and place are separate objects
    # but must coordinate on what is attached in the MoveIt2 planning scene).
    _HELD_ACO_ID: str | None = None
    """
    Base class for arm primitives.

    Args:
        node:    rclpy Node (the Orchestrator).
        moveit2: pymoveit2.MoveIt2 instance (shared across all primitives).
    """

    def __init__(self, node: Node, moveit2, tf_buffer=None) -> None:
        self._node    = node
        self._moveit2 = moveit2
        # GripperCommand action — exposed by franka_gripper_node (franka_ros2).
        self._gripper_client = ActionClient(node, GripperCommand, _GRIPPER_ACTION)

        # Shared TF buffer passed from orchestrator — one listener for all primitives.
        # If not provided (e.g. tests), create a local one as fallback.
        if tf_buffer is not None:
            self._tf_buffer = tf_buffer
        else:
            import tf2_ros
            self._tf_buffer = tf2_ros.Buffer()
            tf2_ros.TransformListener(self._tf_buffer, node)
        # Publisher for AttachedCollisionObject — W5
        from moveit_msgs.msg import AttachedCollisionObject as _ACO
        self._aco_pub = node.create_publisher(
            _ACO, "/attached_collision_object", 10
        )
        # Publisher to remove objects from the world collision scene.
        # Needed because MoveIt2 re-adds a detached ACO as a world object
        # at the gripper position, causing false collisions during retreat.
        from moveit_msgs.msg import CollisionObject as _ColObj
        self._col_pub = node.create_publisher(_ColObj, "/collision_object", 10)

        self._held_object_id: str | None = None   # instance-level (legacy compat)

        # Persistent joint-state cache — updated by a long-lived subscription so
        # _log_eef never creates/destroys subscriptions (which races with the
        # executor's QoS wait_set and causes InvalidHandle crashes in Humble).
        from sensor_msgs.msg import JointState as _JS
        self._last_joint_state = None
        self._js_sub = node.create_subscription(
            _JS, "/joint_states",
            lambda msg: setattr(self, "_last_joint_state", msg),
            10,
        )

    # ── Arm motions ───────────────────────────────────────────────────────────

    def move_to_pose(
        self,
        pose:       Pose,
        frame_id:   str   = BASE_FRAME,
        timeout_sec: float = 15.0,
    ) -> bool:
        """Plan and execute a Cartesian goal (OMPL, free-space path)."""
        self._moveit2.move_to_pose(
            position=[pose.position.x, pose.position.y, pose.position.z],
            quat_xyzw=[
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            cartesian=False,
        )
        result = self._moveit2.wait_until_executed(timeout=timeout_sec)
        if not result:
            self._node.get_logger().warn("ArmPrimitive: move_to_pose failed.")
            return False
        return True

    def move_to_pose_linear(
        self,
        pose:        Pose,
        timeout_sec: float = 15.0,
    ) -> bool:
        """Plan a smooth Cartesian pose motion (PILZ PTP, OMPL fallback).

        Primary: PILZ PTP — deterministic, smooth joint-space interpolation
        that produces near-straight Cartesian paths for small displacements.
        Fallback: OMPL — used if PILZ rejects the goal.
        """
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        pos  = [pose.position.x, pose.position.y, pose.position.z]

        self._moveit2.move_to_pose_linear(position=pos, quat_xyzw=quat)
        result = self._moveit2.wait_until_executed(timeout=timeout_sec)

        if not result:
            self._node.get_logger().warn(
                "ArmPrimitive: PILZ PTP failed — falling back to OMPL."
            )
            self._moveit2.move_to_pose(position=pos, quat_xyzw=quat)
            result = self._moveit2.wait_until_executed(timeout=timeout_sec)
            if not result:
                self._node.get_logger().warn(
                    "ArmPrimitive: move_to_pose_linear (OMPL fallback) also failed."
                )
                return False

        return True

    def move_to_pose_cartesian(
        self,
        pose:        Pose,
        timeout_sec: float = 15.0,
    ) -> bool:
        """Move to pose via computeCartesianPath (geometrically straight line).

        Falls back to PILZ PTP → OMPL when Cartesian fraction < 90% (e.g. large
        workspace moves or paths through kinematic singularities).
        """
        self._moveit2.move_cartesian_waypoints([pose])
        result = self._moveit2.wait_until_executed(timeout=timeout_sec)
        if not result:
            self._node.get_logger().warn(
                "ArmPrimitive: Cartesian path incomplete — falling back to PILZ PTP."
            )
            return self.move_to_pose_linear(pose, timeout_sec=timeout_sec)
        return True

    def move_to_configuration(self, joint_positions: list, timeout_sec: float = 15.0) -> bool:
        """Move to an explicit list of 7 joint positions (radians)."""
        self._moveit2.move_to_configuration(joint_positions)
        result = self._moveit2.wait_until_executed(timeout=timeout_sec)
        if not result:
            self._node.get_logger().warn("ArmPrimitive: move_to_configuration failed.")
        return result

    def move_to_named(self, config_name: str) -> bool:
        """Move to a named joint configuration defined in _NAMED_CONFIGS."""
        joint_positions = _NAMED_CONFIGS.get(config_name)
        if joint_positions is None:
            self._node.get_logger().warn(
                f"ArmPrimitive: unknown named config '{config_name}' — skipping."
            )
            return False
        self._moveit2.move_to_configuration(joint_positions)
        result = self._moveit2.wait_until_executed()
        if not result:
            self._node.get_logger().warn(
                f"ArmPrimitive: move_to_named('{config_name}') failed."
            )
            return False
        return True

    def _get_current_joints(self) -> list:
        """Return current arm joint positions from /joint_states. Falls back to side_approach."""
        js = self._last_joint_state
        if js is None:
            return list(_NAMED_CONFIGS["side_approach"])
        name_to_pos = dict(zip(js.name, js.position))
        return [name_to_pos.get(n, 0.0) for n in ARM_JOINT_NAMES]

    # ── Gripper motions ───────────────────────────────────────────────────────

    def open_gripper(self, max_effort: float = 0.0) -> bool:
        return self._send_gripper_goal(position=_GRIPPER_OPEN, max_effort=max_effort)

    def close_gripper(self, effort: float = _GRIPPER_EFFORT) -> bool:
        return self._send_gripper_goal(position=_GRIPPER_CLOSED, max_effort=effort)

    def _send_gripper_goal(self, position: float, max_effort: float) -> bool:
        """Send a GripperCommand action goal to franka_gripper_node (franka_ros2).

        Uses threading.Event + callbacks so it is safe to call from any
        background thread without conflicting with the MultiThreadedExecutor.
        """
        if not self._gripper_client.wait_for_server(timeout_sec=3.0):
            self._node.get_logger().warn("ArmPrimitive: gripper action server not ready.")
            return False

        goal = GripperCommand.Goal()
        goal.command.position   = position
        goal.command.max_effort = max_effort

        done          = threading.Event()
        result_holder: list = [None]

        def _on_result(future):
            result_holder[0] = future.result()
            done.set()

        def _on_goal(future):
            gh = future.result()
            if gh is None or not gh.accepted:
                self._node.get_logger().warn("ArmPrimitive: gripper goal rejected.")
                done.set()
                return
            gh.get_result_async().add_done_callback(_on_result)

        self._gripper_client.send_goal_async(goal).add_done_callback(_on_goal)
        done.wait(timeout=10.0)

        res     = result_holder[0]
        success = res is not None and res.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            self._node.get_logger().warn("ArmPrimitive: gripper action did not succeed.")
        return success

    # ── W5: AttachedCollisionObject ───────────────────────────────────────────

    def attach_object(
        self,
        object_id: str,
        support_surface: str | None = None,
    ) -> None:
        """Notify MoveIt2 that the robot is now holding an object (W5).

        Adds a conservative cylinder attached to panda_hand so MoveIt2
        includes the held object in all subsequent collision checks.

        Args:
            object_id:       PDDL/Gazebo name of the grasped object.
            support_surface: Name of the MoveIt2 collision object the item was
                             resting on (e.g. "table", "shelf_b").  Added to
                             touch_links so MoveIt2 doesn't block the initial
                             lift due to ACO-surface overlap at grasp height.

        Phase 2: replace the fixed cylinder geometry with the actual object
        shape from the PerceptionModule (e.g. from a point cloud segment).
        """
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose as _Pose

        aco = AttachedCollisionObject()
        aco.link_name              = _HELD_OBJ_LINK    # robot-type specific
        aco.object.id              = object_id
        aco.object.header.frame_id = _HELD_OBJ_LINK

        cyl            = SolidPrimitive()
        cyl.type       = SolidPrimitive.CYLINDER
        cyl.dimensions = [_HELD_OBJ_HEIGHT, _HELD_OBJ_RADIUS]

        p              = _Pose()
        p.position.z   = _HELD_OBJ_Z_IN_HAND
        p.orientation.w = 1.0

        aco.object.primitives      = [cyl]
        aco.object.primitive_poses = [p]
        aco.object.operation       = CollisionObject.ADD
        # Allow physical contact with gripper and wrist links.
        gripper_links = _TOUCH_LINKS
        surface_links = [support_surface] if support_surface else []
        aco.touch_links = gripper_links + surface_links
        self._held_object_id = object_id          # instance-level (compat)
        ArmPrimitive._HELD_ACO_ID = object_id     # class-level (shared with place)
        self._aco_pub.publish(aco)
        self._node.get_logger().debug(
            f"W5 AttachedCollisionObject: '{object_id}' attached to {_HELD_OBJ_LINK}"
        )

    def detach_object(self, object_id: str | None = None) -> None:
        """Notify MoveIt2 that the robot has released the held object (W5).

        Uses class-level _HELD_ACO_ID so PlacePrimitive can detach even though
        it is a different instance from the PickPrimitive that attached the object.

        Args:
            object_id: explicit ID to detach; falls back to class-level shared state.
        """
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

        # Resolve ID: explicit > instance > class-level shared
        oid = object_id or self._held_object_id or ArmPrimitive._HELD_ACO_ID
        if oid is None:
            return
        aco                  = AttachedCollisionObject()
        aco.link_name        = _HELD_OBJ_LINK    # robot-type specific
        aco.object.id        = oid
        aco.object.operation = CollisionObject.REMOVE
        self._aco_pub.publish(aco)

        # MoveIt2 re-adds the detached object as a world collision object at
        # the gripper position (ACO detach behaviour).  Immediately remove it
        # from the world scene so it doesn't block the retreat trajectory.
        from moveit_msgs.msg import CollisionObject as _CO
        import time as _t
        _t.sleep(0.05)   # brief pause so ACO detach is processed first
        world_rm          = _CO()
        world_rm.id       = oid
        world_rm.operation = _CO.REMOVE
        self._col_pub.publish(world_rm)

        self._held_object_id     = None
        ArmPrimitive._HELD_ACO_ID = None          # clear shared state too
        self._node.get_logger().debug(
            f"W5 AttachedCollisionObject: '{oid}' detached + world object removed"
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def _get_current_eef_pos(self, timeout_sec: float = 1.0) -> list | None:
        """Return current EEF position [x, y, z] in BASE_FRAME via TF lookup."""
        try:
            import rclpy.time
            import rclpy.duration
            tf = self._tf_buffer.lookup_transform(
                BASE_FRAME, EEF_LINK,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec),
            )
            t = tf.transform.translation
            return [t.x, t.y, t.z]
        except Exception as exc:
            self._node.get_logger().warn(f"_get_current_eef_pos: TF lookup failed: {exc}")
            return None

    def _make_pre_grasp_pose(self, grasp_pose: Pose, lift_m: float = 0.12) -> Pose:
        """Return a pose `lift_m` above the grasp pose (pre-grasp approach)."""
        pre = deepcopy(grasp_pose)
        pre.position.z += lift_m
        return pre

    def _log(self, msg: str) -> None:
        self._node.get_logger().info(f"[{self.__class__.__name__}] {msg}")

    def _log_eef(self, label: str) -> None:
        """Log EEF pose (TF) and joint angles at the given step label."""
        if self._tf_buffer is not None:
            try:
                from rclpy.time import Time
                t = self._tf_buffer.lookup_transform(BASE_FRAME, EEF_LINK, Time())
                r = t.transform.rotation
                p = t.transform.translation
                self._log(
                    f"  [DBG {label}] EEF pos=({p.x:.3f},{p.y:.3f},{p.z:.3f}) "
                    f"quat=({r.x:.3f},{r.y:.3f},{r.z:.3f},{r.w:.3f})"
                )
            except Exception as e:
                self._log(f"  [DBG {label}] TF lookup failed: {e}")

        try:
            msg = self._last_joint_state
            if msg is not None:
                arm_joints = [
                    (msg.name.index(n), idx + 1)
                    for idx, n in enumerate(ARM_JOINT_NAMES)
                    if n in msg.name
                ]
                vals = [f"j{i}={msg.position[idx]:.2f}" for idx, i in arm_joints]
                self._log(f"  [DBG {label}] joints: {' '.join(vals)}")
        except Exception as e:
            self._log(f"  [DBG {label}] joint_states failed: {e}")
