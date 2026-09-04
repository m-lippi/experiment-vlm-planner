"""Expose a ROS 2 FollowJointTrajectory action over a bridged command topic.

``ros1_bridge`` translates messages and services, but the ROS 1 and ROS 2
action transports are different.  This node keeps MoveIt 2 on its native
action interface and sends the trajectory itself through the bridge as a
``trajectory_msgs/JointTrajectory`` message.  Completion is monitored from
the bridged ``/joint_states`` topic.
"""

from __future__ import annotations

import math
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class TrajectoryTopicAdapter(Node):
    """ROS 2 action server backed by a ROS 1 controller command topic."""

    def __init__(self) -> None:
        super().__init__("trajectory_topic_adapter")

        self.declare_parameter(
            "action_name",
            "/effort_joint_trajectory_controller/follow_joint_trajectory",
        )
        self.declare_parameter(
            "command_topic", "/effort_joint_trajectory_controller/command"
        )
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("filtered_joint_state_topic", "/fr3_joint_states")
        self.declare_parameter(
            "gripper_joint_state_topic", "/ros1_gripper/joint_states"
        )
        self.declare_parameter(
            "joint_names", [f"fr3_joint{i}" for i in range(1, 8)]
        )
        self.declare_parameter("goal_tolerance", 0.01)
        self.declare_parameter("connection_timeout", 10.0)
        self.declare_parameter("execution_timeout_margin", 5.0)

        self._action_name = str(self.get_parameter("action_name").value)
        self._command_topic = str(self.get_parameter("command_topic").value)
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._filtered_joint_state_topic = str(
            self.get_parameter("filtered_joint_state_topic").value
        )
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._finger_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
        self._finger_positions = dict.fromkeys(self._finger_names, 0.02)
        self._default_tolerance = float(self.get_parameter("goal_tolerance").value)
        self._connection_timeout = float(
            self.get_parameter("connection_timeout").value
        )
        self._timeout_margin = float(
            self.get_parameter("execution_timeout_margin").value
        )

        self._state_lock = threading.Lock()
        self._goal_lock = threading.Lock()
        self._goal_reserved = False
        self._positions: dict[str, float] = {}
        self._state_generation = 0

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        callback_group = ReentrantCallbackGroup()
        self._command_pub = self.create_publisher(
            JointTrajectory, self._command_topic, 10
        )
        self._filtered_state_pub = self.create_publisher(
            JointState, self._filtered_joint_state_topic, sensor_qos
        )
        self.create_subscription(
            JointState,
            self._joint_state_topic,
            self._on_joint_state,
            sensor_qos,
            callback_group=callback_group,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("gripper_joint_state_topic").value),
            self._on_gripper_joint_state,
            sensor_qos,
            callback_group=callback_group,
        )
        self._server = ActionServer(
            self,
            FollowJointTrajectory,
            self._action_name,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            execute_callback=self._execute,
            callback_group=callback_group,
        )

        self.get_logger().info(
            f"MoveIt action {self._action_name} -> bridged topic "
            f"{self._command_topic}"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        name_to_index = {name: index for index, name in enumerate(msg.name)}
        with self._state_lock:
            self._positions.update(zip(msg.name, msg.position))
            self._state_generation += 1

        # The ROS 1 mobile-manipulator stack publishes arm and Husky wheel
        # joints on the same topic. MoveIt must only receive joints belonging
        # to its FR3 robot model.
        if not all(name in name_to_index for name in self._joint_names):
            return
        filtered = JointState()
        filtered.header = msg.header
        filtered.name = list(self._joint_names) + self._finger_names
        indices = [name_to_index[name] for name in self._joint_names]
        with self._state_lock:
            finger_positions = [
                self._finger_positions[name] for name in self._finger_names
            ]
        filtered.position = [msg.position[index] for index in indices] + finger_positions
        self._filtered_state_pub.publish(filtered)

    def _on_gripper_joint_state(self, msg: JointState) -> None:
        with self._state_lock:
            for name, position in zip(msg.name, msg.position):
                if name in self._finger_positions:
                    self._finger_positions[name] = float(position)

    def _on_goal(self, request: FollowJointTrajectory.Goal) -> GoalResponse:
        trajectory = request.trajectory
        if not trajectory.points:
            self.get_logger().error("Rejected an empty trajectory")
            return GoalResponse.REJECT
        if set(trajectory.joint_names) != set(self._joint_names):
            self.get_logger().error(
                "Rejected trajectory with joints %s; expected %s"
                % (trajectory.joint_names, self._joint_names)
            )
            return GoalResponse.REJECT
        if any(len(point.positions) != len(trajectory.joint_names)
               for point in trajectory.points):
            self.get_logger().error("Rejected trajectory with incomplete positions")
            return GoalResponse.REJECT

        with self._goal_lock:
            if self._goal_reserved:
                self.get_logger().warning("Rejected concurrent trajectory goal")
                return GoalResponse.REJECT
            self._goal_reserved = True
        return GoalResponse.ACCEPT

    @staticmethod
    def _on_cancel(_goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _result(self, code: int, message: str):
        result = FollowJointTrajectory.Result()
        result.error_code = code
        result.error_string = message
        return result

    def _hold_current_position(self) -> None:
        with self._state_lock:
            if not all(name in self._positions for name in self._joint_names):
                return
            positions = [self._positions[name] for name in self._joint_names]
        hold = JointTrajectory()
        hold.joint_names = self._joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = 1
        hold.points = [point]
        self._command_pub.publish(hold)

    def _goal_tolerances(self, request) -> dict[str, float]:
        tolerances = {name: self._default_tolerance for name in self._joint_names}
        for tolerance in request.goal_tolerance:
            if tolerance.name in tolerances and tolerance.position > 0.0:
                tolerances[tolerance.name] = tolerance.position
        return tolerances

    def _publish_feedback(self, goal_handle, names, target, actual) -> None:
        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = names
        feedback.desired.positions = target
        feedback.actual.positions = actual
        feedback.error.positions = [d - a for d, a in zip(target, actual)]
        goal_handle.publish_feedback(feedback)

    def _execute(self, goal_handle):
        try:
            deadline = time.monotonic() + self._connection_timeout
            while self._command_pub.get_subscription_count() == 0:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    return self._result(0, "Canceled before dispatch")
                if time.monotonic() >= deadline:
                    goal_handle.abort()
                    return self._result(
                        FollowJointTrajectory.Result.INVALID_GOAL,
                        f"No bridge subscriber on {self._command_topic}",
                    )
                time.sleep(0.05)

            request = goal_handle.request
            trajectory = request.trajectory
            names = list(trajectory.joint_names)
            target = list(trajectory.points[-1].positions)
            tolerances = self._goal_tolerances(request)

            with self._state_lock:
                start_generation = self._state_generation

            # Publish only after the dynamic bridge has created its ROS 2
            # subscriber, otherwise the one-shot command could be lost.
            self._command_pub.publish(trajectory)

            duration = trajectory.points[-1].time_from_start
            goal_time = request.goal_time_tolerance
            allowed = (
                float(duration.sec)
                + duration.nanosec / 1e9
                + float(goal_time.sec)
                + goal_time.nanosec / 1e9
                + self._timeout_margin
            )
            deadline = time.monotonic() + max(allowed, self._timeout_margin)

            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    self._hold_current_position()
                    goal_handle.canceled()
                    return self._result(0, "Trajectory canceled; hold command sent")

                with self._state_lock:
                    generation = self._state_generation
                    actual = [self._positions.get(name, math.nan) for name in names]

                if generation > start_generation and all(math.isfinite(v) for v in actual):
                    self._publish_feedback(goal_handle, names, target, actual)
                    errors = [abs(d - a) for d, a in zip(target, actual)]
                    if all(error <= tolerances[name]
                           for name, error in zip(names, errors)):
                        goal_handle.succeed()
                        return self._result(
                            FollowJointTrajectory.Result.SUCCESSFUL,
                            "Final joint state is within goal tolerance",
                        )

                if time.monotonic() >= deadline:
                    goal_handle.abort()
                    return self._result(
                        FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED,
                        "Timed out waiting for the bridged joint state to reach the goal",
                    )
                time.sleep(0.02)

            goal_handle.abort()
            return self._result(FollowJointTrajectory.Result.INVALID_GOAL, "ROS shutdown")
        finally:
            with self._goal_lock:
                self._goal_reserved = False


def main() -> None:
    rclpy.init()
    node = TrajectoryTopicAdapter()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
