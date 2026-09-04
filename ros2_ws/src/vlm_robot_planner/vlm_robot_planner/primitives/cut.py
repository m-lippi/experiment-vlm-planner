"""
cut(object) — Cutting motion: repeated downward strokes at object position.

Phase 2: pre-programmed fixed vertical stroke motion.
Requires: cutting tool in gripper, object on stable surface.
"""
from __future__ import annotations
import time
from geometry_msgs.msg import Pose
from rclpy.node import Node
from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

_CUT_STROKES  = 3
_CUT_DEPTH_M  = 0.03   # depth of each stroke below contact point


class CutPrimitive(ArmPrimitive):

    def __init__(self, node: Node, moveit, tf_buffer=None) -> None:
        super().__init__(node, moveit, tf_buffer=tf_buffer)

    def execute(self, object_name: str, pose_data: dict | None = None) -> bool:
        self._log(f"cut('{object_name}'): {_CUT_STROKES} strokes")
        if pose_data is None:
            self._log("  → no object pose — cannot cut")
            return False
        pos = pose_data["position"]
        for i in range(_CUT_STROKES):
            up = Pose(); up.position.x = pos.x; up.position.y = pos.y
            up.position.z = pos.z + 0.05; up.orientation = _TOP_DOWN_QUAT
            self.move_to_pose_linear(up)
            dn = Pose(); dn.position.x = pos.x; dn.position.y = pos.y
            dn.position.z = pos.z - _CUT_DEPTH_M; dn.orientation = _TOP_DOWN_QUAT
            self.move_to_pose_linear(dn)
            time.sleep(0.2)
        self._log("  → cut complete")
        return True
