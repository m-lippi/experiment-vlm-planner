#!/usr/bin/env python3

import threading

import actionlib
import rospy

from actionlib_msgs.msg import GoalStatus
from franka_gripper.msg import (
    GraspAction,
    GraspGoal,
    HomingAction,
    HomingGoal,
    GraspEpsilon,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
from std_srvs.srv import Trigger, TriggerResponse


class GripperAdapter:
    """
    ROS 1 gripper adapter.

    ROS 1 bridge-facing interface:

        /ros1_gripper/command
            std_msgs/Float64
            desired grasp width [m]

    This is converted into:

        /franka_gripper/grasp
            franka_gripper/GraspAction

    Homing is exposed as a ROS 1 service:

        /ros1_gripper/homing
            std_srvs/Trigger

    which internally calls:

        /franka_gripper/homing
            franka_gripper/HomingAction
    """

    def __init__(self):
        self._grasp_action_name = rospy.get_param(
            "~grasp_action_name",
            "/franka_gripper/grasp",
        )

        self._homing_action_name = rospy.get_param(
            "~homing_action_name",
            "/franka_gripper/homing",
        )

        self._finger_names = rospy.get_param(
            "~finger_joint_names",
            ["fr3_finger_joint1", "fr3_finger_joint2"],
        )

        # Default grasp parameters.
        self._speed = float(
            rospy.get_param("~speed", 0.05)
        )

        self._force = float(
            rospy.get_param("~force", 10.0)
        )

        self._epsilon_inner = float(
            rospy.get_param("~epsilon_inner", 0.005)
        )

        self._epsilon_outer = float(
            rospy.get_param("~epsilon_outer", 0.005)
        )

        # Maximum allowed command width.
        self._max_width = float(
            rospy.get_param("~max_width", 0.08)
        )

        # ------------------------------------------------------------
        # Grasp action client
        # ------------------------------------------------------------

        self._grasp_client = actionlib.SimpleActionClient(
            self._grasp_action_name,
            GraspAction,
        )

        # ------------------------------------------------------------
        # Homing action client
        # ------------------------------------------------------------

        self._homing_client = actionlib.SimpleActionClient(
            self._homing_action_name,
            HomingAction,
        )

        # ------------------------------------------------------------
        # Result topic
        # ------------------------------------------------------------

        self._result_pub = rospy.Publisher(
            "/ros1_gripper/result",
            Bool,
            queue_size=5,
        )

        # ------------------------------------------------------------
        # Synthetic gripper joint state
        # ------------------------------------------------------------

        self._state_pub = rospy.Publisher(
            "/ros1_gripper/joint_states",
            JointState,
            queue_size=10,
        )

        self._width = float(
            rospy.get_param("~initial_width", 0.04)
        )

        # ------------------------------------------------------------
        # Locks
        # ------------------------------------------------------------

        self._lock = threading.Lock()
        self._busy = False

        # ------------------------------------------------------------
        # Initial synthetic state
        # ------------------------------------------------------------

        self._publish_width(self._width)

        self._state_timer = rospy.Timer(
            rospy.Duration(0.1),
            self._publish_timer_callback,
        )

        # ------------------------------------------------------------
        # Bridge-facing command
        # ------------------------------------------------------------

        rospy.Subscriber(
            "/ros1_gripper/command",
            Float64,
            self._on_command,
            queue_size=1,
        )

        # ------------------------------------------------------------
        # Homing service
        # ------------------------------------------------------------

        self._homing_service = rospy.Service(
            "/ros1_gripper/homing",
            Trigger,
            self._on_homing,
        )

        rospy.loginfo(
            "ROS 1 gripper adapter started"
        )

        rospy.loginfo(
            "  Grasp action: %s",
            self._grasp_action_name,
        )

        rospy.loginfo(
            "  Homing action: %s",
            self._homing_action_name,
        )

        rospy.loginfo(
            "  Command topic: /ros1_gripper/command [Float64]"
        )

        rospy.loginfo(
            "  Homing service: /ros1_gripper/homing [Trigger]"
        )

        rospy.loginfo(
            "  Speed: %.3f m/s",
            self._speed,
        )

        rospy.loginfo(
            "  Force: %.3f N",
            self._force,
        )

        rospy.loginfo(
            "  Epsilon: inner=%.4f outer=%.4f",
            self._epsilon_inner,
            self._epsilon_outer,
        )

    # ================================================================
    # Joint state publishing
    # ================================================================

    def _publish_timer_callback(self, _event):
        self._publish_width(self._width)

    def _publish_width(self, width):
        """
        Publish the current assumed gripper width.

        For the Franka finger joint convention:

            finger_joint1 = width / 2
            finger_joint2 = width / 2
        """

        self._width = max(
            0.0,
            float(width),
        )

        state = JointState()

        state.header.stamp = rospy.Time.now()

        state.name = list(
            self._finger_names
        )

        state.position = [
            self._width / 2.0,
            self._width / 2.0,
        ]

        self._state_pub.publish(state)

    # ================================================================
    # Grasp feedback
    # ================================================================

    def _on_grasp_feedback(self, feedback):
        """
        Feedback from /franka_gripper/grasp.
        """

        self._publish_width(
            feedback.width
        )

        rospy.loginfo(
            "Grasp feedback: width=%.4f m",
            feedback.width,
        )

    # ================================================================
    # Grasp result
    # ================================================================

    def _on_grasp_done(self, status, result):
        """
        Called when the Franka grasp action finishes.
        """

        self._publish_width(
            result.width
        )

        success = (
            status == GoalStatus.SUCCEEDED
            and bool(result.success)
        )

        rospy.loginfo(
            "Grasp finished: status=%d "
            "width=%.4f success=%s",
            status,
            result.width,
            result.success,
        )

        self._result_pub.publish(
            Bool(data=success)
        )

        with self._lock:
            self._busy = False

    # ================================================================
    # Gripper command
    # ================================================================

    def _on_command(self, msg):
        """
        Receive desired width from ROS 2 through ros1_bridge.

        Float64.data = desired grasp width [m].
        """

        width = float(msg.data)

        rospy.loginfo(
            "Received gripper command: width=%.4f m",
            width,
        )

        # ------------------------------------------------------------
        # Safety checks
        # ------------------------------------------------------------

        if width < 0.0:
            rospy.logerr(
                "Ignoring negative gripper width: %.4f",
                width,
            )

            self._result_pub.publish(
                Bool(data=False)
            )

            return

        if width > self._max_width:
            rospy.logerr(
                "Ignoring gripper width %.4f m "
                "(maximum %.4f m)",
                width,
                self._max_width,
            )

            self._result_pub.publish(
                Bool(data=False)
            )

            return

        # ------------------------------------------------------------
        # Prevent concurrent commands
        # ------------------------------------------------------------

        with self._lock:
            if self._busy:
                rospy.logwarn(
                    "Ignoring concurrent gripper command"
                )

                self._result_pub.publish(
                    Bool(data=False)
                )

                return

            self._busy = True

        # ------------------------------------------------------------
        # Wait for grasp action server
        # ------------------------------------------------------------

        rospy.loginfo(
            "Waiting for Franka grasp action server..."
        )

        if not self._grasp_client.wait_for_server(
            rospy.Duration(5.0)
        ):
            rospy.logerr(
                "Franka grasp action server is unavailable"
            )

            self._result_pub.publish(
                Bool(data=False)
            )

            with self._lock:
                self._busy = False

            return

        # ------------------------------------------------------------
        # Construct GraspGoal
        # ------------------------------------------------------------

        goal = GraspGoal()

        goal.width = width
        goal.speed = self._speed
        goal.force = self._force

        goal.epsilon = GraspEpsilon(
            inner=self._epsilon_inner,
            outer=self._epsilon_outer,
        )

        rospy.loginfo(
            "Sending grasp goal: "
            "width=%.4f m, speed=%.4f m/s, "
            "force=%.2f N, epsilon=(%.4f, %.4f)",
            goal.width,
            goal.speed,
            goal.force,
            goal.epsilon.inner,
            goal.epsilon.outer,
        )

        self._grasp_client.send_goal(
            goal,
            done_cb=self._on_grasp_done,
            feedback_cb=self._on_grasp_feedback,
        )

    # ================================================================
    # Homing service
    # ================================================================

    def _on_homing(self, _request):
        """
        ROS service callback.

        ROS 1:

            rosservice call /ros1_gripper/homing

        Internally executes:

            /franka_gripper/homing
            franka_gripper/HomingAction
        """

        with self._lock:
            if self._busy:
                rospy.logwarn(
                    "Cannot home gripper: another command is active"
                )

                return TriggerResponse(
                    success=False,
                    message="Gripper is busy",
                )

            self._busy = True

        rospy.loginfo(
            "Homing requested"
        )

        try:
            rospy.loginfo(
                "Waiting for Franka homing action server..."
            )

            if not self._homing_client.wait_for_server(
                rospy.Duration(5.0)
            ):
                rospy.logerr(
                    "Franka homing action server is unavailable"
                )

                return TriggerResponse(
                    success=False,
                    message="Homing action server unavailable",
                )

            goal = HomingGoal()

            rospy.loginfo(
                "Sending homing goal..."
            )

            self._homing_client.send_goal(goal)

            finished = self._homing_client.wait_for_result(
                rospy.Duration(15.0)
            )

            if not finished:
                rospy.logerr(
                    "Homing action timed out"
                )

                self._homing_client.cancel_goal()

                return TriggerResponse(
                    success=False,
                    message="Homing timed out",
                )

            state = self._homing_client.get_state()

            result = self._homing_client.get_result()

            success = (
                state == GoalStatus.SUCCEEDED
            )

            rospy.loginfo(
                "Homing finished: status=%d success=%s",
                state,
                success,
            )

            if success:
                # Homing changes the known maximum opening.
                # We use the configured maximum width as our
                # synthetic state representation.
                self._publish_width(
                    self._max_width
                )

                return TriggerResponse(
                    success=True,
                    message="Gripper homing succeeded",
                )

            return TriggerResponse(
                success=False,
                message="Gripper homing failed",
            )

        finally:
            with self._lock:
                self._busy = False


if __name__ == "__main__":
    rospy.init_node(
        "ros1_gripper_adapter"
    )

    adapter = GripperAdapter()

    rospy.loginfo(
        "ROS 1 gripper adapter ready."
    )

    rospy.spin()




# #!/usr/bin/env python3
# """Translate bridged GripperCommand messages into a ROS 1 action client."""

# import threading

# import actionlib
# import rospy
# from actionlib_msgs.msg import GoalStatus
# from control_msgs.msg import GripperCommand, GripperCommandAction, GripperCommandGoal
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Bool


# class GripperAdapter:
#     def __init__(self):
#         action_name = rospy.get_param(
#             "~action_name", "/franka_gripper/gripper_action"
#         )
#         self._finger_names = rospy.get_param(
#             "~finger_joint_names", ["fr3_finger_joint1", "fr3_finger_joint2"]
#         )
#         self._client = actionlib.SimpleActionClient(action_name, GripperCommandAction)
#         self._result_pub = rospy.Publisher(
#             "/ros1_gripper/result", Bool, queue_size=5
#         )
#         self._state_pub = rospy.Publisher(
#             "/ros1_gripper/joint_states", JointState, queue_size=10
#         )
#         self._lock = threading.Lock()
#         self._busy = False

#         self._width = float(rospy.get_param("~initial_width", 0.04))
#         self._publish_width(self._width)
#         self._state_timer = rospy.Timer(
#             rospy.Duration(0.1), lambda _event: self._publish_width(self._width)
#         )
#         rospy.Subscriber(
#             "/ros1_gripper/command",
#             GripperCommand,
#             self._on_command,
#             queue_size=1,
#         )
#         rospy.loginfo("ROS 1 gripper adapter targeting %s", action_name)

#     def _publish_width(self, width):
#         self._width = max(0.0, float(width))
#         state = JointState()
#         state.header.stamp = rospy.Time.now()
#         state.name = list(self._finger_names)
#         state.position = [self._width / 2.0] * 2
#         self._state_pub.publish(state)

#     def _on_feedback(self, feedback):
#         self._publish_width(feedback.position)

#     def _on_done(self, status, result):
#         self._publish_width(result.position)
#         # Closing on an object normally finishes as "stalled": the fingers
#         # cannot reach the requested width because the object is being held.
#         success = status == GoalStatus.SUCCEEDED and bool(
#             result.reached_goal or result.stalled
#         )
#         self._result_pub.publish(Bool(data=success))
#         with self._lock:
#             self._busy = False

#     def _on_command(self, command):
#         with self._lock:
#             if self._busy:
#                 rospy.logwarn("Ignoring concurrent gripper command")
#                 self._result_pub.publish(Bool(data=False))
#                 return
#             self._busy = True

#         if not self._client.wait_for_server(rospy.Duration(5.0)):
#             rospy.logerr("ROS 1 gripper action server is unavailable")
#             self._result_pub.publish(Bool(data=False))
#             with self._lock:
#                 self._busy = False
#             return

#         goal = GripperCommandGoal()
#         goal.command = command
#         self._client.send_goal(
#             goal, done_cb=self._on_done, feedback_cb=self._on_feedback
#         )


# if __name__ == "__main__":
#     rospy.init_node("ros1_gripper_adapter")
#     GripperAdapter()
#     rospy.spin()
