"""
tilt(object, angle_deg) — Tilt held object by a fixed angle.

Phase 2: pre-programmed rotation of end-effector around Y axis.
Useful for: tipping containers slightly, pouring dry ingredients, etc.
"""
from __future__ import annotations
import math
from rclpy.node import Node
from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

_DEFAULT_TILT_DEG = 45


class TiltPrimitive(ArmPrimitive):

    def __init__(self, node: Node, moveit, tf_buffer=None) -> None:
        super().__init__(node, moveit, tf_buffer=tf_buffer)

    def execute(
        self,
        object_name: str,
        pose_data: dict | None = None,
        angle_deg: float = _DEFAULT_TILT_DEG,
    ) -> bool:
        self._log(f"tilt('{object_name}', {angle_deg}°)")
        from geometry_msgs.msg import Quaternion

        # Resolve current EEF position via TF so we change only orientation
        pos = self._get_current_eef_pos()
        if pos is None:
            self._log("  → TF lookup failed — cannot tilt")
            return False

        half = math.radians(angle_deg) / 2.0
        tilt_q = Quaternion(x=0.0, y=math.sin(half), z=0.0, w=math.cos(half))
        self._moveit2.move_to_pose_linear(
            position=pos,
            quat_xyzw=[tilt_q.x, tilt_q.y, tilt_q.z, tilt_q.w],
        )
        ok = self._moveit2.wait_until_executed(timeout=5.0)
        self._log(f"  → tilt {'complete' if ok else 'failed'}")
        return ok
