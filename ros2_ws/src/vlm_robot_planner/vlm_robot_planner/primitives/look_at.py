"""
look_at — Phase 2: DINO-based directional observation.

Flow:
  1. run_loop_host.py runs DINO on scan image → publishes pose to /perception/object_pose
  2. Orchestrator _dispatch checks perception cache → passes pose_data to execute()
  3. execute() uses pose_data.position to compute j0 = atan2(y, x) → rotates toward object
  4. Fallback: scan pose (j0=0) if no pose available

Oracle-free: pose comes from GroundingDINO, not oracle.
Phase 4 (real robot): same flow, RealSense depth for z.
"""

from __future__ import annotations

import math

from rclpy.node import Node
from geometry_msgs.msg import Pose, Quaternion
from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

_TABLE_VIEW_JOINTS = [0.0, -0.70, 0.0, -2.10, 0.0, 1.40, 0.7854]
_J0_MAX = 1.30   # ±75° clamp


_APPROACH_X_M   = 0.07   # x clearance above object
_APPROACH_HEIGHT_M   = 0.25   # vertical clearance above object


class LookAtPrimitive(ArmPrimitive):

    def __init__(self, node: Node, moveit, tf_buffer=None) -> None:
        super().__init__(node, moveit, tf_buffer=tf_buffer)

    def execute(self, target_name: str, pose_data: dict | None = None) -> bool:
        """
        Move arm toward target using DINO-estimated pose_data.
        Falls back to scan pose (j0=0) when pose unavailable.
        """
        if pose_data is None:
            self._log(f"look_at('{target_name}'): no pose — scan pose fallback")
            return self.move_to_named("scan")


        
        pos = pose_data["position"]
        obs = Pose()
        obs.position.x = pos.x - _APPROACH_X_M; obs.position.y = pos.y
        obs.position.z = pos.z + _APPROACH_HEIGHT_M
        obs.orientation = _TOP_DOWN_QUAT
        

        self._log(f"look_at('{target_name}'): DINO pose "
            f"({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})")

        self._log(f"look_at('{target_name}'): Target pose "
            f"({obs.position.x:.3f},{obs.position.y:.3f},{obs.position.z:.3f})")

        if not self.move_to_pose_cartesian(obs):
            self._log("look at approach failed — aborting look at")
            return False
        return True
        
        
        # j0  = math.atan2(pos.y, pos.x)
        # j0  = max(-_J0_MAX, min(_J0_MAX, j0))

        # joints    = list(_TABLE_VIEW_JOINTS)
        # joints[0] = j0

        # self._log(
        #     f"look_at('{target_name}'): DINO pose "
        #     f"({pos.x:.3f},{pos.y:.3f}) → j0={math.degrees(j0):.1f}°"
        # )

        # self._moveit2.move_to_configuration(joints)
        # ok = self._moveit2.wait_until_executed(timeout=15.0)
        # if not ok:
        #     self._log(f"  → j0 rotation failed, fallback scan pose")
        #     ok = self.move_to_named("scan")

        # if ok:
        #     self._log(f"look_at('{target_name}'): camera aimed at target area")
        # return ok

        # ── Phase 4: move directly above object (RealSense depth required) ──
        # obs = Pose()
        # obs.position.x = pos.x; obs.position.y = pos.y
        # obs.position.z = pos.z + 0.35
        # obs.orientation = _TOP_DOWN_QUAT
        # return self.move_to_pose_linear(obs) or self.move_to_named("scan")
        # ─────────────────────────────────────────────────────────────────────
