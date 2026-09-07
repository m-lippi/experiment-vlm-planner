"""
MoveIt2 Python client — wraps the /move_group action server.

MoveIt 2 communicates through its standard MoveGroup and ExecuteTrajectory
actions. MoveGroup plans without executing; the plan is published for RViz,
held for a configurable preview interval, and then sent to ExecuteTrajectory.
In simulation execution ends at ros2_control; on the real robot it ends at the
local FollowJointTrajectory-to-ROS-1 topic adapter.

Public interface (mirrors pymoveit2.MoveIt2):
  move_to_pose(position, quat_xyzw)
  move_to_pose_linear(position, quat_xyzw)
  move_to_configuration(joint_positions)
  move_cartesian_waypoints(waypoints, max_step, min_fraction)
  wait_until_executed(timeout) → bool
  max_velocity, max_acceleration  (float attributes)

Requires MultiThreadedExecutor on the host node.
"""

from __future__ import annotations

import math
import threading
from typing import List, Optional

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    DisplayTrajectory,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    WorkspaceParameters,
)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

_SUCCESS = 1   # MoveItErrorCodes.SUCCESS


class MoveIt2Client:
    """
    Drop-in replacement for pymoveit2.MoveIt2.

    Communicates with move_group via the /move_group action (planning +
    execution) and /execute_trajectory action (Cartesian path execution).
    Both actions are served by MoveIt 2; the configured controller manager
    selects either the simulation controller or the real-robot adapter.
    """

    def __init__(
        self,
        node:              Node,
        joint_names:       List[str],
        base_link_name:    str,
        end_effector_name: str,
        group_name:        str,
        callback_group:    Optional[CallbackGroup] = None,
    ) -> None:
        self._node        = node
        self._joint_names = joint_names
        self._base_link   = base_link_name
        self._eef_link    = end_effector_name
        self._group_name  = group_name

        self.max_velocity     = 0.3
        self.max_acceleration = 0.3

        self._client = ActionClient(
            node, MoveGroup, "move_action",
            callback_group=callback_group,
        )
        self._execute_client = ActionClient(
            node, ExecuteTrajectory, "/execute_trajectory",
            callback_group=callback_group,
        )
        self._cartesian_client = node.create_client(
            GetCartesianPath, "/compute_cartesian_path",
            callback_group=callback_group,
        )
        self._display_pub = node.create_publisher(
            DisplayTrajectory, "/display_planned_path", 10
        )

        parameter_name = "plan_preview_duration"
        if not node.has_parameter(parameter_name):
            node.declare_parameter(parameter_name, 3.0)
        self.plan_preview_duration = max(
            0.0, float(node.get_parameter(parameter_name).value)
        )

        self._lock               = threading.Lock()
        self._done_event         = threading.Event()
        self._last_success       = False
        self._active_goal_handle = None
        self._cancel_event: threading.Event | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def move_to_pose(
        self,
        position:  List[float],
        quat_xyzw: List[float],
        cartesian: bool = False,
    ) -> None:
        self._send_goal_async(self._build_pose_goal(position, quat_xyzw))

    def move_to_pose_linear(
        self,
        position:  List[float],
        quat_xyzw: List[float],
    ) -> None:
        """PILZ PTP for near-straight Cartesian motions."""
        goal = self._build_pose_goal(position, quat_xyzw)
        goal.request.pipeline_id           = "pilz_industrial_motion_planner"
        goal.request.planner_id            = "PTP"
        goal.request.num_planning_attempts = 1
        self._send_goal_async(goal)

    def move_to_configuration(self, joint_positions: List[float]) -> None:
        self._send_goal_async(self._build_joint_goal(joint_positions))

    def wait_until_executed(self, timeout: float = 60.0) -> bool:
        # The preview is intentional idle time before controller execution and
        # should not consume the caller's existing motion timeout budget.
        effective_timeout = timeout + self.plan_preview_duration
        signalled = self._done_event.wait(timeout=effective_timeout)
        if not signalled:
            self._node.get_logger().warn("MoveIt2Client: wait_until_executed timed out.")
            with self._lock:
                cancel_event = self._cancel_event
                gh = self._active_goal_handle
            if cancel_event is not None:
                # In particular, prevent a preview thread from executing after
                # its caller has already timed out.
                cancel_event.set()
            if gh is not None:
                try:
                    gh.cancel_goal_async()
                except Exception:
                    pass
        return self._last_success

    def _begin_operation(self) -> threading.Event:
        """Start one motion operation and invalidate any older callbacks."""
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._active_goal_handle = None
            self._done_event.clear()
            self._last_success = False
        return cancel_event

    def _is_current(self, cancel_event: threading.Event) -> bool:
        with self._lock:
            return self._cancel_event is cancel_event

    def _finish_operation(
        self, cancel_event: threading.Event, success: bool
    ) -> None:
        with self._lock:
            if self._cancel_event is not cancel_event:
                return
            self._active_goal_handle = None
            self._last_success = bool(success)
            self._done_event.set()

    # ── Goal builders ─────────────────────────────────────────────────────────

    def _build_pose_goal(
        self, position: List[float], quat_xyzw: List[float]
    ) -> MoveGroup.Goal:
        request = self._base_request()

        sphere            = SolidPrimitive()
        sphere.type       = SolidPrimitive.SPHERE
        sphere.dimensions = [0.002]

        centre               = Pose()
        centre.position.x    = float(position[0])
        centre.position.y    = float(position[1])
        centre.position.z    = float(position[2])
        centre.orientation.w = 1.0

        bv                 = BoundingVolume()
        bv.primitives      = [sphere]
        bv.primitive_poses = [centre]

        pos_c                   = PositionConstraint()
        pos_c.header.frame_id   = self._base_link
        pos_c.link_name         = self._eef_link
        pos_c.constraint_region = bv
        pos_c.weight            = 1.0

        ori_c                           = OrientationConstraint()
        ori_c.header.frame_id           = self._base_link
        ori_c.link_name                 = self._eef_link
        ori_c.orientation.x             = float(quat_xyzw[0])
        ori_c.orientation.y             = float(quat_xyzw[1])
        ori_c.orientation.z             = float(quat_xyzw[2])
        ori_c.orientation.w             = float(quat_xyzw[3])
        ori_c.absolute_x_axis_tolerance = 0.1
        ori_c.absolute_y_axis_tolerance = 0.1
        ori_c.absolute_z_axis_tolerance = 0.1
        ori_c.weight                    = 1.0

        goal_c = Constraints()
        goal_c.position_constraints    = [pos_c]
        goal_c.orientation_constraints = [ori_c]
        request.goal_constraints = [goal_c]
        return self._wrap_request(request)

    def _build_joint_goal(self, joint_positions: List[float]) -> MoveGroup.Goal:
        request = self._base_request()
        request.pipeline_id = "pilz_industrial_motion_planner"
        request.planner_id  = "PTP"

        goal_c = Constraints()
        for name, pos in zip(self._joint_names, joint_positions):
            jc                 = JointConstraint()
            jc.joint_name      = name
            jc.position        = float(pos)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight          = 1.0
            goal_c.joint_constraints.append(jc)

        request.goal_constraints = [goal_c]
        return self._wrap_request(request)

    def _base_request(self) -> MotionPlanRequest:
        r = MotionPlanRequest()
        r.group_name                      = self._group_name
        r.num_planning_attempts           = 5
        r.allowed_planning_time           = 10.0
        r.max_velocity_scaling_factor     = float(self.max_velocity)
        r.max_acceleration_scaling_factor = float(self.max_acceleration)
        r.start_state.is_diff             = True

        ws                 = WorkspaceParameters()
        ws.header.frame_id = self._base_link
        ws.min_corner.x    = -1.5; ws.min_corner.y = -1.5; ws.min_corner.z = -1.5
        ws.max_corner.x    =  1.5; ws.max_corner.y =  1.5; ws.max_corner.z =  1.5
        r.workspace_parameters = ws
        return r

    @staticmethod
    def _wrap_request(request: MotionPlanRequest) -> MoveGroup.Goal:
        opts                 = PlanningOptions()
        # Planning and execution are intentionally separate. This gives RViz
        # time to animate /display_planned_path before hardware starts moving.
        opts.plan_only       = True
        opts.replan          = False
        opts.replan_attempts = 0

        goal                  = MoveGroup.Goal()
        goal.request          = request
        goal.planning_options = opts
        return goal

    # ── Cartesian path ────────────────────────────────────────────────────────

    def move_cartesian_waypoints(
        self,
        waypoints:    List[Pose],
        max_step:     float = 0.005,
        min_fraction: float = 0.90,
    ) -> None:
        cancel_event = self._begin_operation()

        threading.Thread(
            target=self._run_cartesian,
            args=(waypoints, max_step, min_fraction, cancel_event),
            daemon=True,
        ).start()

    def _run_cartesian(
        self,
        waypoints:    List[Pose],
        max_step:     float,
        min_fraction: float,
        cancel_event: threading.Event,
    ) -> None:
        if not self._cartesian_client.wait_for_service(timeout_sec=5.0):
            self._node.get_logger().error(
                "MoveIt2Client: /compute_cartesian_path not available."
            )
            self._finish_operation(cancel_event, False)
            return

        req                     = GetCartesianPath.Request()
        req.header.frame_id     = self._base_link
        req.start_state.is_diff = True
        req.group_name          = self._group_name
        req.link_name           = self._eef_link
        req.waypoints           = list(waypoints)
        req.max_step            = float(max_step)
        req.jump_threshold      = 5.0
        req.avoid_collisions    = True

        svc_done   = threading.Event()
        svc_result = [None]

        def _on_svc(future):
            svc_result[0] = future.result()
            svc_done.set()

        self._cartesian_client.call_async(req).add_done_callback(_on_svc)
        if not svc_done.wait(timeout=15.0):
            self._node.get_logger().warn("MoveIt2Client: GetCartesianPath timed out.")
            self._finish_operation(cancel_event, False)
            return

        if cancel_event.is_set() or not self._is_current(cancel_event):
            return

        response = svc_result[0]
        fraction = response.fraction if response else 0.0
        if response is None or fraction < min_fraction:
            error_code = response.error_code.val if response is not None else "none"
            self._node.get_logger().warn(
                f"MoveIt2Client: Cartesian path {fraction:.0%} "
                f"(need ≥{min_fraction:.0%}, error_code={error_code})."
            )
            self._finish_operation(cancel_event, False)
            return

        self._apply_trajectory_scaling(response.solution)
        self._node.get_logger().info(
            f"MoveIt2Client: Cartesian path {fraction:.0%}, retimed to "
            f"velocity={self.max_velocity:.0%}, "
            f"acceleration={self.max_acceleration:.0%} — previewing."
        )
        self._preview_then_execute(
            response.start_state,
            response.solution,
            cancel_event,
        )

    def _apply_trajectory_scaling(self, trajectory) -> None:
        """Conservatively enforce the configured limits on a Cartesian path.

        ``GetCartesianPath`` in ROS 2 Humble has no velocity or acceleration
        scaling fields. Its returned trajectory is therefore timed at the
        robot-model limits, unlike a normal ``MotionPlanRequest``. Stretching
        time uniformly makes the existing path obey both configured limits:
        velocities scale with 1/time and accelerations with 1/time**2.
        """
        velocity_scale = min(1.0, max(float(self.max_velocity), 1e-6))
        acceleration_scale = min(1.0, max(float(self.max_acceleration), 1e-6))
        time_scale = max(
            1.0 / velocity_scale,
            1.0 / math.sqrt(acceleration_scale),
        )

        for point in trajectory.joint_trajectory.points:
            total_nanoseconds = (
                point.time_from_start.sec * 1_000_000_000
                + point.time_from_start.nanosec
            )
            scaled_nanoseconds = math.ceil(total_nanoseconds * time_scale)
            point.time_from_start.sec = scaled_nanoseconds // 1_000_000_000
            point.time_from_start.nanosec = scaled_nanoseconds % 1_000_000_000
            if point.velocities:
                point.velocities = [value / time_scale for value in point.velocities]
            if point.accelerations:
                point.accelerations = [
                    value / (time_scale * time_scale)
                    for value in point.accelerations
                ]

    def _publish_preview(self, start_state, trajectory) -> None:
        display = DisplayTrajectory()
        display.trajectory_start = start_state
        display.trajectory = [trajectory]
        self._display_pub.publish(display)

    def _preview_then_execute(
        self, start_state, trajectory, cancel_event: threading.Event
    ) -> None:
        """Publish a trajectory, wait for RViz preview, then execute it."""
        if cancel_event.is_set() or not self._is_current(cancel_event):
            return

        self._publish_preview(start_state, trajectory)
        delay = self.plan_preview_duration
        if delay > 0.0:
            self._node.get_logger().info(
                f"MoveIt2Client: displaying plan in RViz for {delay:.1f} s "
                "before execution."
            )
            if cancel_event.wait(delay):
                return

        if not self._is_current(cancel_event):
            return

        if not self._execute_client.wait_for_server(timeout_sec=5.0):
            self._node.get_logger().error(
                "MoveIt2Client: /execute_trajectory not available."
            )
            self._finish_operation(cancel_event, False)
            return

        if cancel_event.is_set() or not self._is_current(cancel_event):
            return

        exec_goal            = ExecuteTrajectory.Goal()
        exec_goal.trajectory = trajectory
        self._node.get_logger().info("MoveIt2Client: preview complete — executing plan.")
        self._execute_client.send_goal_async(exec_goal).add_done_callback(
            lambda future: self._on_exec_goal(future, cancel_event)
        )

    def _on_exec_goal(self, future, cancel_event: threading.Event) -> None:
        gh = future.result()
        if cancel_event.is_set() or not self._is_current(cancel_event):
            if gh is not None and gh.accepted:
                gh.cancel_goal_async()
            return
        if gh is None or not gh.accepted:
            self._node.get_logger().warn("MoveIt2Client: ExecuteTrajectory goal rejected.")
            self._finish_operation(cancel_event, False)
            return
        with self._lock:
            if self._cancel_event is not cancel_event:
                return
            self._active_goal_handle = gh
        gh.get_result_async().add_done_callback(
            lambda result_future: self._on_exec_result(result_future, cancel_event)
        )

    def _on_exec_result(self, future, cancel_event: threading.Event) -> None:
        if not self._is_current(cancel_event):
            return
        response = future.result()
        success = False
        if response is None:
            self._node.get_logger().warn("MoveIt2Client: ExecuteTrajectory null result.")
        else:
            code = response.result.error_code.val
            success = (
                response.status == GoalStatus.STATUS_SUCCEEDED and code == _SUCCESS
            )
            if not success:
                self._node.get_logger().warn(
                    f"MoveIt2Client: ExecuteTrajectory failed — "
                    f"status={response.status}, code={code}"
                )
        self._finish_operation(cancel_event, success)

    # ── Async dispatch ────────────────────────────────────────────────────────

    def _send_goal_async(self, goal: MoveGroup.Goal) -> None:
        cancel_event = self._begin_operation()

        if not self._client.wait_for_server(timeout_sec=10.0):
            self._node.get_logger().error(
                "MoveIt2Client: /move_action not available (is move_group running?)."
            )
            self._finish_operation(cancel_event, False)
            return

        self._client.send_goal_async(goal).add_done_callback(
            lambda future: self._on_goal_response(future, cancel_event)
        )

    def _on_goal_response(self, future, cancel_event: threading.Event) -> None:
        gh = future.result()
        if cancel_event.is_set() or not self._is_current(cancel_event):
            if gh is not None and gh.accepted:
                gh.cancel_goal_async()
            return
        if gh is None or not gh.accepted:
            self._node.get_logger().warn("MoveIt2Client: goal rejected by move_group.")
            self._finish_operation(cancel_event, False)
            return
        with self._lock:
            if self._cancel_event is not cancel_event:
                return
            self._active_goal_handle = gh
        gh.get_result_async().add_done_callback(
            lambda result_future: self._on_plan_result(
                result_future, cancel_event
            )
        )

    def _on_plan_result(self, future, cancel_event: threading.Event) -> None:
        if cancel_event.is_set() or not self._is_current(cancel_event):
            return
        with self._lock:
            if self._cancel_event is cancel_event:
                self._active_goal_handle = None
        response = future.result()
        if response is None:
            self._node.get_logger().warn("MoveIt2Client: null result from move_group.")
            self._finish_operation(cancel_event, False)
            return

        code = response.result.error_code.val
        success = (
            response.status == GoalStatus.STATUS_SUCCEEDED and code == _SUCCESS
        )
        if not success:
            self._node.get_logger().warn(
                f"MoveIt2Client: planning failed — "
                f"status={response.status}, error_code={code}"
            )
            self._finish_operation(cancel_event, False)
            return

        self._node.get_logger().info("MoveIt2Client: planning succeeded.")
        self._preview_then_execute(
            response.result.trajectory_start,
            response.result.planned_trajectory,
            cancel_event,
        )
