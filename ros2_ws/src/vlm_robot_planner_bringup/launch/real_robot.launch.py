"""MoveIt 2 bringup for an FR3 controlled by a ROS 1 computer.

The ROS 1 host owns the hardware and runs
``effort_joint_trajectory_controller``. A separate ``ros1_bridge`` process
bridges ``/joint_states`` and the controller's ``command`` topic. The local
trajectory adapter presents the ROS 2 action expected by MoveIt and forwards
its trajectory through that command topic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder



_CAM_CFG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "data"
    / "overview_camera_setup.json"
)
_cam_cfg: dict = {}
if _CAM_CFG_PATH.exists():
    try:
        with open(_CAM_CFG_PATH, encoding="utf-8") as config_file:
            _cam_cfg = json.load(config_file)
    except (OSError, ValueError) as error:
        print(f"[launch] WARNING: cannot read {_CAM_CFG_PATH.name}: {error}")


def generate_launch_description() -> LaunchDescription:
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true", description="Launch RViz2"
    )
    preview_arg = DeclareLaunchArgument(
        "plan_preview_duration",
        default_value="2.0",
        description="Seconds to display a MoveIt plan in RViz before execution",
    )
    allowed_start_tolerance_arg = DeclareLaunchArgument(
        "allowed_start_tolerance",
        default_value="0.02",
        description="Maximum joint error allowed when trajectory execution starts",
    )
    joint_state_topic_arg = DeclareLaunchArgument(
        "joint_state_topic",
        default_value="/joint_states",
        description="ROS 1 joint-state topic, as exposed by ros1_bridge",
    )
    command_topic_arg = DeclareLaunchArgument(
        "command_topic",
        default_value="/effort_joint_trajectory_controller/command",
        description="ROS 1 JointTrajectory command topic, as exposed by ros1_bridge",
    )
    overview_args = [
        DeclareLaunchArgument("overview_x", default_value=str(_cam_cfg.get("x", 0.65))),
        DeclareLaunchArgument("overview_y", default_value=str(_cam_cfg.get("y", 0.70))),
        DeclareLaunchArgument("overview_z", default_value=str(_cam_cfg.get("z", 0.73))),
        DeclareLaunchArgument(
            "overview_roll", default_value=str(_cam_cfg.get("roll", 0.0))
        ),
        DeclareLaunchArgument(
            "overview_pitch", default_value=str(_cam_cfg.get("pitch", 0.68))
        ),
        DeclareLaunchArgument(
            "overview_yaw", default_value=str(_cam_cfg.get("yaw", -2.19))
        ),
    ]

    bringup_share = get_package_share_directory("vlm_robot_planner_bringup")
    description_share = get_package_share_directory("franka_description")
    urdf_xacro = os.path.join(
        description_share, "robots", "fr3", "fr3.urdf.xacro"
    )
    srdf_xacro = os.path.join(
        description_share, "robots", "fr3", "fr3.srdf.xacro"
    )

    robot_description = {
        "robot_description": ParameterValue(
            Command(
                [
                    FindExecutable(name="xacro"),
                    " ",
                    urdf_xacro,
                    " robot_type:=fr3 hand:=true",
                ]
            ),
            value_type=str,
        )
    }
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(
            file_path=urdf_xacro, mappings={"robot_type": "fr3", "hand": "true"}
        )
        .robot_description_semantic(
            file_path=srdf_xacro, mappings={"robot_type": "fr3", "hand": "true"}
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

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": False}],
        remappings=[("/joint_states", "/fr3_joint_states")],
    )
    trajectory_adapter = Node(
        package="vlm_robot_planner",
        executable="trajectory_topic_adapter",
        output="screen",
        parameters=[
            {
                "action_name": (
                    "/effort_joint_trajectory_controller/follow_joint_trajectory"
                ),
                "command_topic": LaunchConfiguration("command_topic"),
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "filtered_joint_state_topic": "/fr3_joint_states",
                "joint_names": [f"fr3_joint{i}" for i in range(1, 8)],
            }
        ],
    )
    # realsense_node = Node(
    #     package="realsense2_camera",
    #     executable="realsense2_camera_node",
    #     namespace="overview_camera",
    #     name="realsense2_camera",
    #     parameters=[{
    #         "enable_color": True,
    #         "enable_depth": True,
    #         "align_depth.enable": True,
    #         "enable_gyro": False,
    #         "enable_accel": False,
    #         "enable_sync": True,
    #     }],
    #     output="screen",
    # )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "examples/align_depth",
                "rs_align_depth_launch.py",
            )
        ),
        launch_arguments={
            "camera_namespace": "overview_camera",
            "camera_name": "overview_camera",
            "enable_color": "true",
            "enable_depth": "true",
            "align_depth.enable": "true",
            "enable_sync": "true",
            "enable_gyro": "false",
            "enable_accel": "false",
        }.items(),
    )

    gripper_adapter = Node(
        package="vlm_robot_planner",
        executable="gripper_action_adapter",
        output="screen",
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_to_fr3",
        arguments=["0.20", "0", "0.77", "0", "0", "0", "world", "fr3_link0"],
        parameters=[{"use_sim_time": False}],
        output="screen",
    )
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
        parameters=[{"use_sim_time": False}],
        output="screen",
    )
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_params,
            {
                "trajectory_execution.allowed_start_tolerance": ParameterValue(
                    LaunchConfiguration("allowed_start_tolerance"),
                    value_type=float,
                )
            },
        ],
        remappings=[("/joint_states", "/fr3_joint_states")],
    )
    rviz_path = os.path.join(bringup_share, "config", "moveit.rviz")
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_path] if os.path.exists(rviz_path) else [],
        parameters=[moveit_params],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    orchestrator = TimerAction(
        period=10.0,
        actions=[
            Node(
                package="vlm_robot_planner",
                executable="orchestrator",
                output="screen",
                parameters=[
                    moveit_params,
                    {
                        "use_sim": False,
                        "plan_preview_duration": ParameterValue(
                            LaunchConfiguration("plan_preview_duration"),
                            value_type=float,
                        ),
                    },
                ],
                additional_env={"VLM_ROBOT": "fr3"},
            )
        ],
    )

    return LaunchDescription(
        [
            rviz_arg,
            preview_arg,
            allowed_start_tolerance_arg,
            joint_state_topic_arg,
            command_topic_arg,
            *overview_args,
            robot_state_publisher,
            trajectory_adapter,
            gripper_adapter,
            # realsense_launch,
            static_tf,
            static_tf_overview,
            move_group,
            rviz,
            orchestrator,
        ]
    )
