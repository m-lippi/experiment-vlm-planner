"""
'place' primitive: deposit the held object at a target location via MoveIt2.

Motion sequence:
  1. move_to_pose(pre_place)        — OMPL free-space approach above target
  2. move_to_pose_linear(place)     — PILZ LIN straight descent to release height
  3. open_gripper()                 — release object
  4. move_to_pose_linear(pre_place) — PILZ LIN straight retreat upward
  5. move_to_named("ready")         — PILZ PTP return to neutral

The place pose uses the same top-down orientation as the pick.
"""

from __future__ import annotations

from geometry_msgs.msg import Pose, Quaternion
from rclpy.node import Node

from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

# Height of panda_hand above the target surface centre at the release pose.
# With two-finger Boeing joints, cup follows panda_hand at the same
# _GRASP_OFFSET_Z_M (0.13 m) captured during pick.
#
# Geometry (panda_link0 reference, table surface = z=0.00 m):
#   cup_z       = panda_hand_z − 0.13
#   cup_bottom  = cup_z − 0.06
#   shelf top   = shelf_centre_z + 0.01 = 0.01 + 0.01 = 0.02 m
#
# _RELEASE_HEIGHT_M = 0.22: panda_hand at 0.23 m → cup bottom at 0.04 m
#   → 2 cm above shelf top ✓
#   ACO bottom at 0.23−0.13−0.07 = 0.03 m > table top 0.00 m ✓
# Cup falls ~2 cm onto shelf after Boeing detach.
_RELEASE_HEIGHT_M  = 0.22
# Approach height above the release pose (pre-place clearance)
_APPROACH_HEIGHT_M = 0.15


class PlacePrimitive(ArmPrimitive):
    """
    Deposits the held object at a target location.

    Args:
        node:   rclpy Node (Orchestrator).
        moveit: MoveIt2Client instance shared with all other primitives.
        attach: Optional GazeboAttach shared with PickPrimitive.
                If provided, the object continues following the EEF during
                the approach, is detached after open_gripper, then Gazebo
                physics drops it the remaining ~1 cm onto the surface.
    """

    def __init__(self, node: Node, moveit, attach=None, tf_buffer=None) -> None:
        super().__init__(node, moveit, tf_buffer=tf_buffer)
        self._attach = attach

    def execute(self, location_name: str, pose_data: dict) -> bool:
        """
        Execute a top-down place at the named location.

        Args:
            location_name: Symbolic location name (for logging).
            pose_data:     Pose dict from GazeboOracle:
                           {"position": Position, "orientation": Orientation}

        Returns:
            True if the full place sequence completed successfully.
        """
        if pose_data is None:
            self._log(f"place('{location_name}'): no pose available — aborting")
            return False
        pos = pose_data["position"]
        self._log(f"place('{location_name}'): pos=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")

        # Sanity-check oracle coordinates: Panda workspace is ≤ 0.85 m from base.
        # Values outside ±2 m indicate a corrupted Gazebo state (physics explosion).
        if abs(pos.x) > 2.0 or abs(pos.y) > 2.0 or abs(pos.z) > 2.0:
            self._log(
                f"place('{location_name}'): oracle pose outside workspace "
                f"({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}) — Gazebo state corrupted, aborting"
            )
            return False

        place_pose = self._build_release_pose(pose_data)
        pre_place  = self._make_pre_grasp_pose(place_pose, lift_m=_APPROACH_HEIGHT_M)

        # ── 1. Move above target — Cartesian approach (OMPL fallback) ────
        self._log(f"  → pre-place (z={pre_place.position.z:.3f})")
        if not self.move_to_pose_cartesian(pre_place):
            self._log("pre-place planning failed — aborting place")
            return False

        # ── 2. Descend to release height — PILZ LIN (straight vertical) ──
        self._log(f"  → descending to release (z={place_pose.position.z:.3f})")
        if not self.move_to_pose_linear(place_pose):
            self._log("place descend failed — aborting place")
            return False

        # ── 3. Release: GazeboAttach stop → ACO detach → open gripper ────
        # Stop the 100Hz teleport timer before opening the gripper so the
        # object stays at the place position while the fingers withdraw.

        # 3a. Simulation-only: release physics joints first
        if self._attach is not None:
            self._attach.detach()

        # 3b. Notify MoveIt2 that the object is released (W5)
        self.detach_object()

        # 3c. Open gripper — fingers now free to withdraw
        if not self.open_gripper():
            self._log("open_gripper failed during place — object may not be released")

        # ── 4. Retreat upward — PILZ LIN ──────────────────────────────────
        self._log("  → retreating")
        if not self.move_to_pose_linear(pre_place):
            self._log("retreat after place failed")
            return False

        # ── 5. Safe retreat → ready (two-step return) ─────────────────────
        # Go through "safe_retreat" first (arm high above table) to avoid
        # PILZ PTP paths that could dip near table-level objects.
        # Falls back to direct "ready" if safe_retreat is unreachable.
        if not self.move_to_named("safe_retreat"):
            self._log("safe_retreat unreachable — returning directly to ready")
        self.move_to_named("ready")

        self._log(f"place('{location_name}'): SUCCESS")
        return True

    def _build_release_pose(self, pose_data: dict) -> Pose:
        """Build release pose: target surface + clearance, top-down orientation."""
        pos = pose_data["position"]
        pose = Pose()
        pose.position.x = pos.x
        pose.position.y = pos.y
        pose.position.z = pos.z + _RELEASE_HEIGHT_M
        pose.orientation = _TOP_DOWN_QUAT
        return pose
