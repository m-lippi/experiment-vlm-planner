"""
stir(container) — Stir contents of a container with held object (spoon/tool).

Physical motion: circular path in XY plane inside the container,
N revolutions at fixed depth.

Phase 2 (sim): pre-programmed fixed circular motion.
"""
from __future__ import annotations
import math
import time

from geometry_msgs.msg import Pose
from rclpy.node import Node
from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

_STIR_RADIUS_M    = 0.03    # 3cm radius circle inside container
_STIR_DEPTH_M     = 0.04    # depth below container rim
_STIR_REVOLUTIONS = 3
_STIR_PERIOD_S    = 1.5     # seconds per revolution


class StirPrimitive(ArmPrimitive):
    """Stir contents of container using circular end-effector motion."""

    def __init__(self, node: Node, moveit, tf_buffer=None) -> None:
        super().__init__(node, moveit, tf_buffer=tf_buffer)

    def execute(
        self,
        container_name: str,
        pose_data: dict | None = None,
    ) -> bool:
        self._log(f"stir('{container_name}'): circular stirring motion")

        if pose_data is None:
            self._log("  → no container pose — cannot stir")
            return False

        pos = pose_data["position"]
        cx, cy, cz = pos.x, pos.y, pos.z + _STIR_DEPTH_M

        self._log(f"  → stirring at ({cx:.2f},{cy:.2f}) r={_STIR_RADIUS_M*100:.0f}cm "
                  f"× {_STIR_REVOLUTIONS} rev")

        center = Pose()
        center.position.x = cx; center.position.y = cy; center.position.z = cz + 0.05
        center.orientation = _TOP_DOWN_QUAT
        if not self.move_to_pose_linear(center):
            self._log("  → failed to reach container")
            return False

        steps = 16
        for rev in range(_STIR_REVOLUTIONS):
            for i in range(steps):
                angle = 2.0 * math.pi * i / steps
                via = Pose()
                via.position.x = cx + _STIR_RADIUS_M * math.cos(angle)
                via.position.y = cy + _STIR_RADIUS_M * math.sin(angle)
                via.position.z = cz
                via.orientation = _TOP_DOWN_QUAT
                self.move_to_pose_linear(via)
                time.sleep(_STIR_PERIOD_S / steps)

        self._log(f"  → stir complete ({_STIR_REVOLUTIONS} revolutions)")
        return True
