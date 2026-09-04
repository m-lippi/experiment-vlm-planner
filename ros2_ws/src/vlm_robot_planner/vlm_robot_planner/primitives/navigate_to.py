"""
'navigate_to' primitive: move the mobile base to a target location.

Phase 1 stub — the robot operates with a STATIONARY base in Phase 1.
Navigation via Nav2 is planned for Phase 3 (mobile base integration).

This stub logs the navigation request and returns True so the pipeline
does not fail on navigation steps during simulation. When Phase 3 begins,
replace the body of execute() with a Nav2 NavigateToPose action client call.
"""

from __future__ import annotations

from rclpy.node import Node


class NavigateToPrimitive:
    """
    Phase 1 stub: logs the destination and returns True immediately.
    Phase 3: send NavigateToPose goal to Nav2 action server.

    Note: This primitive does NOT inherit from ArmPrimitive because
    it does not use MoveIt2 — it drives the mobile base.
    """

    def __init__(self, node: Node) -> None:
        self._node = node

    def execute(self, destination: str, pose_data: dict | None = None) -> bool:
        """
        Phase 1 stub: logs the destination and returns True immediately.

        Args:
            destination: Symbolic location name (e.g. "shelf_area").
            pose_data:   3D pose for navigation goal (Phase 3).
        """
        self._node.get_logger().warn(
            f"[NavigateToPrimitive] navigate_to('{destination}'): "
            "STUB — mobile base navigation not implemented in Phase 1. "
            "Robot stays stationary. Returning True."
        )
        return True
