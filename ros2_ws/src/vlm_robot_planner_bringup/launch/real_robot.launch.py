"""
real_robot.launch.py — Bringup for the real Franka FR3 via franka_ros2.

Architecture (franka_ros2 + franka_hardware, no ROS 1 bridge):
  franka_hardware plugin → libfranka (FCI over Ethernet) → FR3 robot
  ros2_control controller_manager → fr3_arm_controller (FollowJointTrajectory)
  franka_gripper_node → FCI → gripper
  MoveIt 2 move_group ← standard FollowJointTrajectory action (no bridge)

Prerequisites:
  - Robot network: FR3 reachable at robot_ip, FCI enabled in Franka Desk
  - Docker container: vlm_ros2 (includes franka_ros2 built in /opt/franka_ws/)
  - No ROS 1 bridge required

Usage:
  ros2 launch vlm_robot_planner_bringup real_robot.launch.py robot_ip:=192.168.1.100
  ros2 launch vlm_robot_planner_bringup real_robot.launch.py robot_ip:=192.168.1.100 rviz:=false
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


# Auto-load overview camera config saved by scripts/setup_overview_camera.py.
_CAM_CFG_PATH = (
    Path(__file__).resolve()
    .parent.parent.parent.parent.parent  # repo root
    / "data" / "overview_camera_setup.json"
)
_cam_cfg: dict = {}
if _CAM_CFG_PATH.exists():
    try:
        with open(_CAM_CFG_PATH) as _f:
            _cam_cfg = json.load(_f)
        print(f"[launch] overview camera config: {_CAM_CFG_PATH.name}")
    except Exception as _e:
        print(f"[launch] WARNING: could not read {_CAM_CFG_PATH.name}: {_e}")


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ────────────────────────────────────────────────────────────
    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        description="IP address of the Franka FR3 robot (FCI interface).",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 for visualization",
    )
    overview_x_arg     = DeclareLaunchArgument(
        "overview_x",     default_value=str(_cam_cfg.get("x",     0.65)))
    overview_y_arg     = DeclareLaunchArgument(
        "overview_y",     default_value=str(_cam_cfg.get("y",     0.70)))
    overview_z_arg     = DeclareLaunchArgument(
        "overview_z",     default_value=str(_cam_cfg.get("z",     0.73)))
    overview_roll_arg  = DeclareLaunchArgument(
        "overview_roll",  default_value=str(_cam_cfg.get("roll",  0.0)))
    overview_pitch_arg = DeclareLaunchArgument(
        "overview_pitch", default_value=str(_cam_cfg.get("pitch", 0.68)))
    overview_yaw_arg   = DeclareLaunchArgument(
        "overview_yaw",   default_value=str(_cam_cfg.get("yaw",   -2.19)))

    # ── Paths ─────────────────────────────────────────────────────────────────
    bringup_share  = get_package_share_directory("vlm_robot_planner_bringup")
    fr3_desc_share = get_package_share_directory("franka_description")
    fr3_cfg_share  = get_package_share_directory("franka_fr3_moveit_config")
    rviz_path      = os.path.join(bringup_share, "config", "moveit.rviz")

    # Local franka_description (used for MoveIt kinematics — no ros2_control support)
    urdf_xacro = os.path.join(fr3_desc_share, "robots", "fr3", "fr3.urdf.xacro")
    srdf_xacro = os.path.join(fr3_desc_share, "robots", "fr3", "fr3.srdf.xacro")

    # fr3_hw.urdf.xacro — our local URDF that composes FR3 kinematics with the
    # franka_hardware ros2_control plugin (patched v0.1.0, POSITION interface).
    # Lives in ros2_ws/src/franka_description (built into the container workspace)
    # so $(find franka_description) resolves correctly at xacro time.
    urdf_xacro_hw = os.path.join(fr3_desc_share, "robots", "fr3", "fr3_hw.urdf.xacro")

    # ── Robot description for ros2_control_node ───────────────────────────────
    # Includes the <ros2_control> hardware block (FrankaHardwareInterface, position).
    robot_description_hw = {
        "robot_description": ParameterValue(
            Command([
                FindExecutable(name="xacro"), " ", urdf_xacro_hw,
                " robot_ip:=", LaunchConfiguration("robot_ip"),
                " hand:=true",
            ]),
            value_type=str,
        )
    }

    # ── MoveIt 2 configuration ────────────────────────────────────────────────
    # robot_description without ros2_control tags — MoveIt 2 only needs the
    # kinematic chain and joint limits, not the hardware plugin definition.
    # FR3-specific overrides applied on top of the panda moveit config base:
    #   kinematics  → kinematics_fr3.yaml   (pick_ik, fr3_arm group)
    #   joint_limits → joint_limits_fr3.yaml
    #   controllers  → real_robot_moveit_controllers.yaml
    #                  (fr3_arm_controller / franka_gripper)
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(
            file_path=urdf_xacro,
            mappings={"robot_type": "fr3", "hand": "true"},
        )
        .robot_description_semantic(
            file_path=srdf_xacro,
            mappings={"robot_type": "fr3", "hand": "true"},
        )
        .robot_description_kinematics(
            file_path=os.path.join(bringup_share, "config", "kinematics_fr3.yaml")
        )
        .joint_limits(
            file_path=os.path.join(bringup_share, "config", "joint_limits_fr3.yaml")
        )
        .trajectory_execution(
            file_path=os.path.join(
                bringup_share, "config", "real_robot_moveit_controllers.yaml"
            )
        )
        .to_moveit_configs()
    )

    moveit_params = moveit_config.to_dict()
    moveit_params["use_sim_time"] = False

    # ── 1. ros2_control controller_manager ────────────────────────────────────
    # Loads franka_hardware plugin (FCI connection) and all controller configs.
    # Must start before robot_state_publisher and spawners.
    controller_config = os.path.join(
        fr3_cfg_share, "config", "fr3_ros_controllers.yaml"
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description_hw, controller_config],
        output="screen",
        remappings=[
            ("~/robot_description", "/robot_description"),
        ],
    )

    # ── 2. robot_state_publisher ──────────────────────────────────────────────
    # Uses robot_description_hw (with ros2_control:=true) so that the
    # /robot_description topic received by ros2_control_node via the remapping
    # contains the <ros2_control> hardware block. MoveIt ignores ros2_control tags.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description_hw, {"use_sim_time": False}],
    )

    # ── 3. Controller spawners ────────────────────────────────────────────────
    # Spawned after ros2_control_node is running (OnProcessStart event handler).
    def make_spawner(controller_name: str) -> Node:
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[controller_name, "--controller-manager", "/controller_manager"],
            output="screen",
        )

    joint_state_broadcaster_spawner = make_spawner("joint_state_broadcaster")
    fr3_arm_controller_spawner      = make_spawner("fr3_arm_controller")

    # Start spawners only after ros2_control_node has started.
    spawn_controllers = RegisterEventHandler(
        OnProcessStart(
            target_action=ros2_control_node,
            on_start=[
                joint_state_broadcaster_spawner,
                fr3_arm_controller_spawner,
            ],
        )
    )

    # ── 4. franka_gripper ─────────────────────────────────────────────────────
    # ROS 2 node that connects to the gripper via FCI and exposes:
    #   /franka_gripper/gripper_action  (control_msgs/GripperCommand action)
    #   /franka_gripper/move            (franka_msgs/Move action)
    #   /franka_gripper/grasp           (franka_msgs/Grasp action)
    franka_gripper = Node(
        package="franka_gripper",
        executable="franka_gripper_node",
        name="franka_gripper",
        parameters=[{
            "robot_ip":  LaunchConfiguration("robot_ip"),
            "use_sim_time": False,
            "default_speed": 0.1,
            "default_grasp_epsilon": {
                "inner": 0.005,
                "outer": 0.005,
            },
            "joint_names": ["fr3_finger_joint1", "fr3_finger_joint2"],
        }],
        output="screen",
    )

    # ── 5. Static TF: world → base ────────────────────────────────────────────
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_to_base",
        arguments=["0.20", "0", "0.77", "0", "0", "0", "world", "base"],
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # ── 5b. Static TF: fr3_link0 → overview_camera_optical_frame ─────────────
    static_tf_overview = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_fr3_to_overview_cam",
        arguments=[
            LaunchConfiguration("overview_x"),
            LaunchConfiguration("overview_y"),
            LaunchConfiguration("overview_z"),
            LaunchConfiguration("overview_roll"),
            LaunchConfiguration("overview_pitch"),
            LaunchConfiguration("overview_yaw"),
            "fr3_link0",
            "overview_camera_optical_frame",
        ],
        output="screen",
        parameters=[{"use_sim_time": False}],
    )

    # ── 6. MoveIt 2 move_group ────────────────────────────────────────────────
    # No /joint_states remapping: ros2_control publishes only FR3 joints,
    # so the default /joint_states topic is already correct.
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_params],
    )

    # ── 7. RViz2 (optional) ───────────────────────────────────────────────────
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_path] if os.path.exists(rviz_path) else [],
        parameters=[moveit_params],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # ── 8. Orchestrator ───────────────────────────────────────────────────────
    # Delayed 10s: waits for move_group + controllers to be fully ready.
    orchestrator = TimerAction(
        period=10.0,
        actions=[
            Node(
                package="vlm_robot_planner",
                executable="orchestrator",
                output="screen",
                parameters=[moveit_params, {"use_sim": False}],
                additional_env={"VLM_ROBOT": "fr3"},
            )
        ],
    )

    return LaunchDescription([
        robot_ip_arg,
        rviz_arg,
        overview_x_arg, overview_y_arg, overview_z_arg,
        overview_roll_arg, overview_pitch_arg, overview_yaw_arg,
        ros2_control_node,
        robot_state_publisher,
        spawn_controllers,
        franka_gripper,
        static_tf,
        static_tf_overview,
        move_group,
        rviz2,
        orchestrator,
    ])
