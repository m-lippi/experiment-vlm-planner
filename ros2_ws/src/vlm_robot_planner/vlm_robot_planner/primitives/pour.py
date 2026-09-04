"""
pour(source, target) — Pour liquid from source container into target container.

Simplified motion (single joint tilt):
  1. Optional Cartesian move to above target (skipped if pose unavailable or fails).
  2. Unlock GazeboAttach orientation.
  3. Tilt: decrement panda_joint5 by _POUR_TILT_DELTA rad (~86 deg).
  4. Hold _POUR_DURATION_S seconds.
  5. Un-tilt: restore pre-tilt joint config.
  6. Re-lock orientation.
  7. Release object and return to ready.

Design rationale: the thesis evaluates the VLM planner, not motion complexity.
One joint tilt clearly demonstrates the pour action without relying on fragile
multi-step IK chains or slow 11-second named-config moves.
"""
from __future__ import annotations
import time

from geometry_msgs.msg import Pose, Quaternion
from rclpy.node import Node
from vlm_robot_planner.primitives.base import ArmPrimitive

_POUR_DURATION_S  = 2.5    # seconds to hold tilted
_POUR_VEL         = 0.30   # 30% max velocity — visible but not sluggish
_POUR_TILT_DELTA  = 1.5    # radians removed from panda_joint5 (index 4) for tilt
_POUR_CLEARANCE_M = 0.25   # height above target surface when carrying can
_FINGER_REACH_M   = 0.10   # EEF to finger-pad offset along EEF Z

_SIDE_QUAT = Quaternion(x=0.7071, y=0.0, z=0.7071, w=0.0)

# Return-to-source geometry (matches pick.py side-grasp offsets).
_RETURN_REACH_M    = 0.10
_RETURN_Z_OFFSET_M = 0.10
_RETURN_RETREAT_M  = 0.15


class PourPrimitive(ArmPrimitive):
    """Pour contents of held object into target container, then release."""

    def __init__(self, node: Node, moveit, attach=None, tf_buffer=None) -> None:
        super().__init__(node, moveit, tf_buffer=tf_buffer)
        self._attach = attach

    def execute(
        self,
        target_name: str,
        pose_data: dict | None = None,
        source_name: str | None = None,
        source_pose_data: dict | None = None,
    ) -> bool:
        self._log(f"pour('{source_name or 'object'}' -> '{target_name}'): starting")

        _saved_vel = self._moveit2.max_velocity
        _saved_acc = self._moveit2.max_acceleration
        self._moveit2.max_velocity     = _POUR_VEL
        self._moveit2.max_acceleration = _POUR_VEL

        try:
            # ── 1. Optional: carry can above target ───────────────────────────
            if pose_data is not None:
                pos = pose_data["position"]
                above = Pose()
                above.position.x = pos.x - _FINGER_REACH_M
                above.position.y = pos.y
                above.position.z = pos.z + _POUR_CLEARANCE_M
                above.orientation = _SIDE_QUAT
                self._log(
                    f"  -> carrying to above target "
                    f"({above.position.x:.2f},{above.position.y:.2f},{above.position.z:.2f})"
                )
                if not self.move_to_pose_cartesian(above):
                    self._log("  -> transit failed -- pouring in current position")

            # ── 2. Tilt ───────────────────────────────────────────────────────
            ok = self._do_pour_motion()

        finally:
            self._moveit2.max_velocity     = _saved_vel
            self._moveit2.max_acceleration = _saved_acc

        # ── 3. Release ────────────────────────────────────────────────────────
        if source_pose_data is not None:
            self._return_to_source(source_name or target_name, source_pose_data)
        else:
            if self._attach is not None:
                self._attach.detach()
            self.detach_object()
            self.open_gripper()
            self.move_to_named("safe_retreat")
            self.move_to_named("ready")

        self._log("  -> pour sequence complete")
        return ok

    # ── Pour motion ───────────────────────────────────────────────────────────

    def _do_pour_motion(self) -> bool:
        """Read current joints, apply delta on j5, hold, reverse."""
        pre_tilt = self._get_current_joints()

        tilt = list(pre_tilt)
        tilt[5] += _POUR_TILT_DELTA   # panda_joint6 index = 5

        if self._attach is not None:
            self._attach.lock_orientation(False)

        self._log(
            f"  -> tilting j6 by +{_POUR_TILT_DELTA:.2f} rad "
            f"({pre_tilt[5]:.2f} -> {tilt[5]:.2f})"
        )
        tilt_ok = self.move_to_configuration(tilt)
        if not tilt_ok:
            self._log("  -> tilt move failed")

        self._log(f"  -> pouring ({_POUR_DURATION_S}s)...")
        time.sleep(_POUR_DURATION_S)

        self._log("  -> un-tilting")
        self.move_to_configuration(pre_tilt)

        if self._attach is not None:
            self._attach.lock_orientation(True)

        return tilt_ok

    # ── Return to source ──────────────────────────────────────────────────────

    def _return_to_source(self, source_name: str, source_pose_data: dict) -> None:
        pos = source_pose_data["position"]
        self._log(
            f"  -> returning '{source_name}' to source "
            f"({pos.x:.2f},{pos.y:.2f},{pos.z:.2f})"
        )

        return_pose = Pose()
        return_pose.position.x = pos.x - _RETURN_REACH_M
        return_pose.position.y = pos.y
        return_pose.position.z = pos.z + _RETURN_Z_OFFSET_M
        return_pose.orientation = _SIDE_QUAT

        if not self.move_to_pose_cartesian(return_pose):
            self._log("  -> return move failed -- releasing at current position")

        if self._attach is not None:
            self._attach.detach()
        self.detach_object()
        self.open_gripper()

        retreat = Pose()
        retreat.position.x = return_pose.position.x - _RETURN_RETREAT_M
        retreat.position.y = return_pose.position.y
        retreat.position.z = return_pose.position.z
        retreat.orientation = _SIDE_QUAT
        self.move_to_pose_cartesian(retreat)

        self.move_to_named("safe_retreat")
        self.move_to_named("ready")
        self._log(f"  -> '{source_name}' returned")
