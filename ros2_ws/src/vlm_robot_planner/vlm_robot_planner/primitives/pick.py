"""
'pick' primitive: grasp an object from the table via MoveIt2.

Supports two grasp modes (selected by the VLM via grasp_mode arg):
  top_down (default): approach from above → retreat upward.   Good for place/stack/stir/cut.
  side:               pure horizontal approach from behind (Ry(90°)) → retreat along same axis.
                      Contacts cylindrical body laterally — natural for pour/tilt.
  handle:             same as top_down in Phase 1; reserved for Phase 4 tool-grip refinement.

The 'side' orientation (Ry(90°)) points EEF Z along +X in panda_link0:
  EEF Z = (1.0, 0.0, 0.0) — gripper approaches the object from behind, purely horizontally.

Side grasp geometry:
  The finger pads are ~0.1034m along EEF Z from the panda_hand frame origin.
  To place the pads AT the object, panda_hand must be offset in the −EEF_Z direction:
    panda_hand.x = obj.x − 0.1034 × 1.0 = obj.x − 0.1034
    panda_hand.z = obj.z + grip_height        (no z component: approach is horizontal)
  Finger tips land at (obj.x, obj.y, obj.z + grip_height).
"""

from __future__ import annotations

from geometry_msgs.msg import Pose, Quaternion
from rclpy.node import Node

from vlm_robot_planner.primitives.base import ArmPrimitive, _TOP_DOWN_QUAT

# Grasp approach clearance (pre-grasp offset)
_APPROACH_HEIGHT_M   = 0.15   # vertical clearance above grasp for top_down
_APPROACH_LATERAL_M  = 0.15   # lateral clearance before grasp for side

# Height of panda_hand above the detected object z at the grasp pose (top_down only).
# Franka finger length below panda_hand frame ≈ 0.133 m.
# In sim: detected_z = oracle object centre → finger tips ~1.5 cm above centre.
# On real robot (Phase 2+): detected_z from RealSense depth → same formula applies.
# finger_tips = detected_z + _GRASP_OFFSET_Z_M - 0.133
_GRASP_OFFSET_Z_M = 0.15

# Side grasp: Ry(90°) × Rz(180°) body rotation.
# EEF Z = [1,0,0] (world +X) — gripper approaches from behind along +X, unchanged.
# EEF X = [0,0,1] (world +Z) — camera faces UP (camera mount is along panda_hand +X).
# Ry(90°) alone gives EEF X = [0,0,-1] (camera floor); adding Rz(180°) flips it UP.
# Quaternion: (x=0.7071, y=0, z=0.7071, w=0) = Ry(90°)·Rz(180°).
_SIDE_GRASP_QUAT = Quaternion(x=0.7071, y=0.0, z=0.7071, w=0.0)

# Distance from panda_hand origin to the object centre along EEF Z (panda_hand +Z).
# URDF: finger_joint at 0.0584 m, pad centre ~0.045 m further → total ~0.10 m.
# Also used as grasp_offset_forward_m for GazeboAttach so the attached object
# appears at the finger pads (not at the palm), regardless of arm depth.
_FINGER_REACH_M = 0.10

# Desired finger-pad grip height above oracle z (object base on table surface).
# 0.06m ≈ centre of a standard 330ml can (12 cm tall).
# Oracle reports the object base: z=0 means base is at the counter surface.
_SIDE_GRASP_Z_OFFSET_M = 0.10   # Phase 1 fallback: safe for 8–25 cm containers

# Known grasp modes — used in _dispatch to distinguish from source_location arg.
GRASP_MODES = {"top_down", "side", "handle"}


class PickPrimitive(ArmPrimitive):
    """
    Grasps an object given its symbolic name and 3D pose from the oracle.

    Args:
        node:   rclpy Node (Orchestrator).
        moveit: MoveIt2Client instance shared with all other primitives.
        attach: Optional GazeboAttach for simulated object attachment.
                If provided, the object will follow the EEF during the lift.
    """

    def __init__(self, node: Node, moveit, attach=None, tf_buffer=None) -> None:
        super().__init__(node, moveit, tf_buffer=tf_buffer)
        self._attach = attach

    def execute(
        self,
        object_name: str,
        pose_data: dict,
        support_surface: str | None = None,
        grasp_mode: str = "top_down",
        object_height_m: float | None = None,
    ) -> bool:
        """
        Grasp the named object using the requested grasp_mode.

        object_height_m: estimated object height in metres (Phase 2+ perception).
          Grip height = 50% of height; falls back to _SIDE_GRASP_Z_OFFSET_M when None.
        """
        pos = pose_data["position"]
        self._log(
            f"pick('{object_name}', mode={grasp_mode}"
            + (f", h={object_height_m:.2f}m" if object_height_m else "")
            + f"): pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})"
        )

        if grasp_mode == "side":
            grasp_pose = self._build_side_grasp_pose(pose_data, object_height_m)
            pre_grasp  = self._make_side_pre_grasp_pose(grasp_pose)
            # panda_hand is (grip_h + reach*0.707) above obj.z — derive from pose delta.
            # GazeboAttach formula: world_z = 0.77 + hand_z - offset_z → offset_z = hand_z - obj.z
            _side_offset_z = grasp_pose.position.z - pose_data["position"].z
        else:  # top_down or handle — same physical approach in Phase 1
            grasp_pose = self._build_top_down_pose(pose_data)
            pre_grasp  = self._make_pre_grasp_pose(grasp_pose, lift_m=_APPROACH_HEIGHT_M)
            _side_offset_z = None   # unused for top_down

        # ── 0. Clean up stale state from previous incomplete operations ──────
        self.detach_object()
        if self._attach is not None:
            self._attach.detach()

        # ── 0b. Side grasp: go directly to natural pre-grasp joint configuration ──
        # Problem: for quaternion (0.707,0,0.707,0) at position (0.174,-0.20,0.13),
        # KDL gradient-descent IK consistently converges to a self-colliding
        # configuration (panda_link5↔panda_link1) regardless of j7 seed.
        # OMPL then finds a valid but non-natural config with j1≈154° that causes
        # the arm to rotate on itself during the approach and carry.
        #
        # Fix: use "side_approach" (joint config from old working pick with j7-π)
        # via move_to_configuration — direct joint control, zero IK computation.
        # The arm arrives AT the pre-grasp EEF position with j7=-2.42 and natural
        # j1-j6 angles. Subsequent PILZ to grasp has a small joint delta and finds
        # the natural IK trivially from this seed.
        #
        # Phase 4: replace "side_approach" with a general IK solution seeded from
        # the natural configuration computed from the object position.
        if grasp_mode == "side":
            self._log_eef("before-approach")
            if not self.move_to_named("side_approach"):
                self._log("failed to reach side_approach pose — aborting side grasp")
                return False
            self._log_eef("after-approach")

        # ── 1. Open gripper ────────────────────────────────────────────────
        if not self.open_gripper():
            self._log("open_gripper failed — aborting pick")
            return False

        # ── 2. Pre-grasp ───────────────────────────────────────────────────
        if grasp_mode == "side":
            q = pre_grasp.orientation
            self._log(
                f"  → side pre-grasp "
                f"(x={pre_grasp.position.x:.3f}, y={pre_grasp.position.y:.3f}, "
                f"z={pre_grasp.position.z:.3f}) "
                f"quat_cmd=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})"
            )
        else:
            self._log(f"  → pre-grasp above object (z={pre_grasp.position.z:.3f})")
        if not self.move_to_pose_cartesian(pre_grasp):
            self._log("pre-grasp planning failed — aborting pick")
            return False
        if grasp_mode == "side":
            self._log_eef("after-pre-grasp")

        # ── 3. Approach to grasp (Cartesian straight line) ─────────────────
        # Use computeCartesianPath (not PILZ PTP) for the grasp approach.
        # PILZ PTP interpolates in joint space and swings panda_link6 toward
        # the table mid-path (→ collision reject → OMPL fallback → j1=-1.05).
        # computeCartesianPath keeps a geometrically straight line so j1 stays
        # at the natural value from side_approach (-0.52).  For 15 cm the
        # Cartesian fraction is 100% from the side_approach seed configuration.
        if grasp_mode == "side":
            q = grasp_pose.orientation
            self._log(
                f"  → Cartesian approach to grasp "
                f"(x={grasp_pose.position.x:.3f}, y={grasp_pose.position.y:.3f}, "
                f"z={grasp_pose.position.z:.3f}) "
                f"quat_cmd=({q.x:.3f},{q.y:.3f},{q.z:.3f},{q.w:.3f})"
            )
        else:
            self._log(f"  → descend to grasp (z={grasp_pose.position.z:.3f})")
        if not self.move_to_pose_cartesian(grasp_pose):
            self._log("grasp approach failed — aborting pick")
            return False
        if grasp_mode == "side":
            self._log_eef("after-grasp-approach")

        # ── 4. Close gripper ───────────────────────────────────────────────
        if not self.close_gripper():
            self._log("close_gripper failed — object may have slipped")

        # ── 4b. Notify MoveIt2 ─────────────────────────────────────────────
        self.attach_object(object_name, support_surface=support_surface)

        # ── 4c. Simulation-only: physics attachment (BoeingAttach) ─────────
        if self._attach is not None:
            if grasp_mode == "side":
                self._attach.attach(
                    object_name,
                    grasp_offset_z=_side_offset_z,
                    grasp_offset_forward_m=_FINGER_REACH_M,
                )
            else:
                self._attach.attach(object_name)   # top_down: defaults (0.13, 0.0)

        # ── 5. Retreat (Cartesian straight line) ───────────────────────────
        # Use computeCartesianPath so the EEF moves in a geometrically straight
        # line. move_to_pose_linear (PILZ PTP) interpolates in joint space and
        # can dip the EEF during the retreat → can teleports down with the hand
        # → visual glitch; Cartesian keeps the retreat perfectly horizontal.
        self._log("  → retreating with object")
        retreat = self._make_side_retreat_pose(grasp_pose) if grasp_mode == "side" else pre_grasp
        if not self.move_to_pose_cartesian(retreat):
            self._log("retreat failed — object may be stuck")
            self.detach_object(object_name)
            if self._attach is not None:
                self._attach.detach()
            return False

        self._log(f"pick('{object_name}', mode={grasp_mode}): SUCCESS")
        return True

    # ── Pose builders ────────────────────────────────────────────────────────

    def _build_top_down_pose(self, pose_data: dict) -> Pose:
        """Vertical approach from above: panda_hand Z pointing down."""
        pos = pose_data["position"]
        pose = Pose()
        pose.position.x = pos.x
        pose.position.y = pos.y
        pose.position.z = pos.z + _GRASP_OFFSET_Z_M
        pose.orientation = _TOP_DOWN_QUAT
        return pose

    def _build_side_grasp_pose(
        self, pose_data: dict, object_height_m: float | None = None
    ) -> Pose:
        """Horizontal approach (Ry(90°), EEF Z = (1.0, 0.0, 0.0)).

        The finger pads are _FINGER_REACH_M ahead of panda_hand along EEF Z = +X.
        To land pads at (obj.x, obj.y, obj.z + grip_h), panda_hand is offset back:
          panda_hand.x = obj.x - _FINGER_REACH_M
          panda_hand.z = obj.z + grip_h    (no z component: approach is horizontal)
        Finger tips land at (obj.x, obj.y, obj.z + grip_h).

        grip_h selection:
          Phase 1 (object_height_m=None): fixed fallback _SIDE_GRASP_Z_OFFSET_M (0.10 m).
          Phase 2+ (perception height available): grip at 50% of object height.
          Minimum grip_h = 0.03 m (keeps finger tips at least ~1 cm above table).

        Phase 1: fixed approach direction (objects at positive X in panda_link0).
        Phase 4: compute approach direction dynamically from object position.
        """
        pos = pose_data["position"]
        reach = _FINGER_REACH_M

        if object_height_m is not None and object_height_m > 0.04:
            grip_h = object_height_m * 0.5
        else:
            grip_h = _SIDE_GRASP_Z_OFFSET_M
        grip_h = max(grip_h, 0.03)

        pose = Pose()
        pose.position.x = pos.x - reach        # back-offset along -X so pads reach obj.x
        pose.position.y = pos.y
        pose.position.z = pos.z + grip_h        # horizontal approach: no z component from reach
        pose.orientation = _SIDE_GRASP_QUAT
        return pose

    def _make_side_pre_grasp_pose(self, grasp_pose: Pose) -> Pose:
        """Pre-grasp for side mode: offset backward along the Ry(90°) approach axis.

        Approach axis: EEF Z = (1.0, 0.0, 0.0) = +X.
        Pull back by _APPROACH_LATERAL_M in the -X direction at constant height:
          Δx = -1.0 * dist = -0.15 m  (backward)
          Δz = 0                       (horizontal: no height change)
        """
        from copy import deepcopy
        pre = deepcopy(grasp_pose)
        pre.position.x -= _APPROACH_LATERAL_M
        return pre

    def _make_side_retreat_pose(self, grasp_pose: Pose) -> Pose:
        """Retreat after side grasp: reverse along approach axis (same as pre-grasp)."""
        return self._make_side_pre_grasp_pose(grasp_pose)

