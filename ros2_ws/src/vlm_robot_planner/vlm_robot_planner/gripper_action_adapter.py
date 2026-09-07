"""Expose a ROS 2 GripperCommand action through bridge-friendly Float64 topics."""

from __future__ import annotations

import threading
import time

import rclpy
from control_msgs.action import GripperCommand as GripperCommandAction
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Float64


class GripperActionAdapter(Node):
    def __init__(self) -> None:
        super().__init__("gripper_action_adapter")

        self.declare_parameter(
            "action_name",
            "/franka_gripper/gripper_action",
        )
        self.declare_parameter(
            "command_topic",
            "/ros1_gripper/command",
        )
        self.declare_parameter(
            "result_topic",
            "/ros1_gripper/result",
        )
        self.declare_parameter(
            "timeout",
            15.0,
        )

        callback_group = ReentrantCallbackGroup()

        # Bridge-friendly command:
        #
        # ROS 2 Float64
        #       |
        #       v
        # ros1_bridge
        #       |
        #       v
        # ROS 1 Float64
        #
        # The Float64 value is the desired gripper width in meters.
        self._command_pub = self.create_publisher(
            Float64,
            str(self.get_parameter("command_topic").value),
            10,
        )

        self._result_lock = threading.Lock()
        self._result_generation = 0
        self._last_result = False

        self._goal_lock = threading.Lock()
        self._goal_reserved = False

        self._timeout = float(
            self.get_parameter("timeout").value
        )

        # Result coming back from the ROS 1 adapter.
        self.create_subscription(
            Bool,
            str(self.get_parameter("result_topic").value),
            self._on_result,
            10,
            callback_group=callback_group,
        )

        # Keep the ROS 2-facing API as a normal GripperCommand action.
        self._server = ActionServer(
            self,
            GripperCommandAction,
            str(self.get_parameter("action_name").value),
            goal_callback=self._on_goal,
            cancel_callback=lambda _goal: CancelResponse.ACCEPT,
            execute_callback=self._execute,
            callback_group=callback_group,
        )

        self.get_logger().info(
            "Gripper action adapter started"
        )
        self.get_logger().info(
            f"ROS 2 action: {self.get_parameter('action_name').value}"
        )
        self.get_logger().info(
            "Bridge command topic: "
            f"{self.get_parameter('command_topic').value} "
            "[std_msgs/msg/Float64]"
        )
        self.get_logger().info(
            "Result topic: "
            f"{self.get_parameter('result_topic').value} "
            "[std_msgs/msg/Bool]"
        )

    def _on_result(self, msg: Bool) -> None:
        with self._result_lock:
            self._last_result = bool(msg.data)
            self._result_generation += 1

    def _on_goal(self, _request) -> GoalResponse:
        with self._goal_lock:
            if self._goal_reserved:
                self.get_logger().warn(
                    "Rejecting gripper goal: another goal is active"
                )
                return GoalResponse.REJECT

            self._goal_reserved = True

        return GoalResponse.ACCEPT

    def _execute(self, goal_handle):
        result = GripperCommandAction.Result()

        try:
            # Wait until ros1_bridge has created the subscription.
            deadline = time.monotonic() + 5.0

            while self._command_pub.get_subscription_count() == 0:
                if time.monotonic() >= deadline:
                    self.get_logger().error(
                        "No subscriber on bridge command topic"
                    )
                    goal_handle.abort()
                    return result

                time.sleep(0.05)

            # Get the current result generation BEFORE publishing.
            #
            # This prevents an old result from being interpreted as
            # the result of the new command.
            with self._result_lock:
                generation = self._result_generation

            # Extract the requested width from the ROS 2 action goal.
            width = float(
                goal_handle.request.command.position
            )

            self.get_logger().info(
                f"Gripper goal received: width={width:.4f} m"
            )

            # Basic safety check.
            if width < 0.0 or width > 0.08:
                self.get_logger().error(
                    f"Rejecting invalid gripper width: {width:.4f} m"
                )
                goal_handle.abort()
                return result

            # Convert:
            #
            # control_msgs/action/GripperCommand
            #
            # into:
            #
            # std_msgs/msg/Float64
            #
            msg = Float64()
            msg.data = width

            self._command_pub.publish(msg)

            self.get_logger().info(
                f"Published Float64 gripper command: {width:.4f} m"
            )

            deadline = time.monotonic() + self._timeout

            while (
                rclpy.ok()
                and time.monotonic() < deadline
            ):
                if goal_handle.is_cancel_requested:
                    self.get_logger().warn(
                        "Gripper goal cancelled"
                    )
                    goal_handle.canceled()
                    return result

                with self._result_lock:
                    if self._result_generation > generation:
                        success = self._last_result

                        if success:
                            result.reached_goal = True
                            goal_handle.succeed()

                            self.get_logger().info(
                                "Gripper goal succeeded"
                            )
                        else:
                            goal_handle.abort()

                            self.get_logger().error(
                                "Gripper goal failed"
                            )

                        return result

                time.sleep(0.02)

            self.get_logger().error(
                "Timeout waiting for ROS 1 gripper result"
            )

            goal_handle.abort()
            return result

        finally:
            with self._goal_lock:
                self._goal_reserved = False


def main() -> None:
    rclpy.init()

    node = GripperActionAdapter()

    executor = MultiThreadedExecutor(
        num_threads=2
    )

    executor.add_node(node)

    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

# """Expose a ROS 2 GripperCommand action through bridge-friendly topics."""

# from __future__ import annotations

# import threading
# import time

# import rclpy
# from control_msgs.action import GripperCommand as GripperCommandAction
# from control_msgs.msg import GripperCommand
# from rclpy.action import ActionServer, CancelResponse, GoalResponse
# from rclpy.callback_groups import ReentrantCallbackGroup
# from rclpy.executors import MultiThreadedExecutor
# from rclpy.node import Node
# from std_msgs.msg import Bool


# class GripperActionAdapter(Node):
#     def __init__(self) -> None:
#         super().__init__("gripper_action_adapter")
#         self.declare_parameter("action_name", "/franka_gripper/gripper_action")
#         self.declare_parameter("command_topic", "/ros1_gripper/command")
#         self.declare_parameter("result_topic", "/ros1_gripper/result")
#         self.declare_parameter("timeout", 15.0)

#         callback_group = ReentrantCallbackGroup()
#         self._command_pub = self.create_publisher(
#             GripperCommand, str(self.get_parameter("command_topic").value), 10
#         )
#         self._result_lock = threading.Lock()
#         self._result_generation = 0
#         self._last_result = False
#         self._goal_lock = threading.Lock()
#         self._goal_reserved = False
#         self._timeout = float(self.get_parameter("timeout").value)
#         self.create_subscription(
#             Bool,
#             str(self.get_parameter("result_topic").value),
#             self._on_result,
#             10,
#             callback_group=callback_group,
#         )
#         self._server = ActionServer(
#             self,
#             GripperCommandAction,
#             str(self.get_parameter("action_name").value),
#             goal_callback=self._on_goal,
#             cancel_callback=lambda _goal: CancelResponse.ACCEPT,
#             execute_callback=self._execute,
#             callback_group=callback_group,
#         )

#     def _on_result(self, msg: Bool) -> None:
#         with self._result_lock:
#             self._last_result = bool(msg.data)
#             self._result_generation += 1

#     def _on_goal(self, _request) -> GoalResponse:
#         with self._goal_lock:
#             if self._goal_reserved:
#                 return GoalResponse.REJECT
#             self._goal_reserved = True
#         return GoalResponse.ACCEPT

#     def _execute(self, goal_handle):
#         result = GripperCommandAction.Result()
#         try:
#             deadline = time.monotonic() + 5.0
#             while self._command_pub.get_subscription_count() == 0:
#                 if time.monotonic() >= deadline:
#                     goal_handle.abort()
#                     return result
#                 time.sleep(0.05)

#             with self._result_lock:
#                 generation = self._result_generation
#             self._command_pub.publish(goal_handle.request.command)
#             deadline = time.monotonic() + self._timeout

#             while rclpy.ok() and time.monotonic() < deadline:
#                 if goal_handle.is_cancel_requested:
#                     goal_handle.canceled()
#                     return result
#                 with self._result_lock:
#                     if self._result_generation > generation:
#                         success = self._last_result
#                         if success:
#                             result.reached_goal = True
#                             goal_handle.succeed()
#                         else:
#                             goal_handle.abort()
#                         return result
#                 time.sleep(0.02)

#             goal_handle.abort()
#             return result
#         finally:
#             with self._goal_lock:
#                 self._goal_reserved = False


# def main() -> None:
#     rclpy.init()
#     node = GripperActionAdapter()
#     executor = MultiThreadedExecutor(num_threads=2)
#     executor.add_node(node)
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         executor.shutdown()
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
