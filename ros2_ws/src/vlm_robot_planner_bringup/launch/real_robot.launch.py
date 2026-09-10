"""MoveIt 2 bringup for an FR3 controlled by a ROS 1 computer.

The ROS 1 host owns the hardware and runs
``effort_joint_trajectory_controller``. A separate ``ros1_bridge`` process
bridges ``/joint_states`` and the controller's ``command`` topic. The local
trajectory adapter presents the ROS 2 action expected by MoveIt and forwards
its trajectory through that command topic.
"""

from __future__ import annotations

import json
import math
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


_TAG_TF_PATH = _CAM_CFG_PATH.with_name("table_apriltag_transform.json")


def _rotation_matrix_to_quaternion(rotation):
    """Return normalized (x, y, z, w) for a 3x3 rotation matrix."""
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]
    trace = r00 + r11 + r22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r21 - r12) / scale
        qy = (r02 - r20) / scale
        qz = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        qw = (r21 - r12) / scale
        qx = 0.25 * scale
        qy = (r01 + r10) / scale
        qz = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        qw = (r02 - r20) / scale
        qx = (r01 + r10) / scale
        qy = 0.25 * scale
        qz = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        qw = (r10 - r01) / scale
        qx = (r02 + r20) / scale
        qy = (r12 + r21) / scale
        qz = 0.25 * scale

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        raise ValueError("rotation matrix produced a zero quaternion")
    return qx / norm, qy / norm, qz / norm, qw / norm


def _load_table_apriltag_tf():
    """Load the saved T_base_tag transform for a static TF broadcaster."""
    if not _TAG_TF_PATH.exists():
        print(f"[launch] WARNING: {_TAG_TF_PATH.name} not found; AprilTag TF disabled")
        return None

    try:
        with _TAG_TF_PATH.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        transform = config["tag_to_base"]
        if len(transform) != 4 or any(len(row) != 4 for row in transform):
            raise ValueError("tag_to_base must be a 4x4 matrix")

        translation = tuple(float(transform[index][3]) for index in range(3))
        rotation = [
            [float(transform[row][column]) for column in range(3)]
            for row in range(3)
        ]
        quaternion = _rotation_matrix_to_quaternion(rotation)
        parent = str(config.get("parent_frame", "fr3_link0"))
        child = str(
            config.get("child_frame", f"apriltag_{int(config.get('tag_id', 0))}")
        )
        if not parent or not child or parent == child:
            raise ValueError("invalid parent_frame/child_frame")
        return translation, quaternion, parent, child
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"[launch] WARNING: cannot read {_TAG_TF_PATH.name}: {error}")
        return None


def generate_launch_description() -> LaunchDescription:
    rviz_arg = DeclareLaunchArgument(
        "rviz", default_value="true", description="Launch RViz2"
    )
    preview_arg = DeclareLaunchArgument(
        "plan_preview_duration",
        default_value="0.0",
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
    experiment_webcam_device_arg = DeclareLaunchArgument(
        "experiment_webcam_device",
        default_value="auto",
        description="Trust webcam V4L2 path, or 'auto' for name-based discovery",
    )
    experiment_webcam_match_arg = DeclareLaunchArgument(
        "experiment_webcam_match",
        default_value="trust",
        description="Case-insensitive device-name match used in auto mode",
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
            file_path=urdf_xacro,
            mappings={
                "robot_type": "fr3",
                "hand": "true",
                "planning_ee_frame": "true",
            },
        )
        .robot_description_semantic(
            file_path=srdf_xacro,
            mappings={
                "robot_type": "fr3",
                "hand": "true",
                "planning_tip_link": "fr3_EE",
            },
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
    realsense_node = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="overview_camera",
        name="overview_camera",
        parameters=[{
            "enable_color": True,
            "enable_depth": True,
            "align_depth.enable": True,
            "enable_gyro": False,
            "enable_accel": False,
            "enable_sync": True,
            "enable_infra1": False,
            "enable_infra2": False,

            "overview_camera.color.image_raw.enable_pub_plugins": [
                "image_transport/raw",
            ],

            "overview_camera.depth.image_rect_raw.enable_pub_plugins": [
                "image_transport/raw",
            ],

            "overview_camera.aligned_depth_to_color.image_raw.enable_pub_plugins": [
                "image_transport/raw",
            ],
        }],
        output="screen",
    )
    experiment_webcam = Node(
        package="vlm_robot_planner",
        executable="usb_webcam",
        name="experiment_usb_webcam",
        output="screen",
        parameters=[{
            "device": LaunchConfiguration("experiment_webcam_device"),
            "device_name_match": LaunchConfiguration("experiment_webcam_match"),
            "image_topic": "/experiment_camera/image_raw",
            "frame_id": "experiment_camera_optical_frame",
            "width": 640,
            "height": 480,
            "fps": 10.0,
        }],
    )

    # realsense_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory("realsense2_camera"),
    #             "examples/align_depth",
    #             "rs_align_depth_launch.py",
    #         )
    #     ),
    #     launch_arguments={
    #         "camera_namespace": "overview_camera",
    #         "camera_name": "overview_camera",
    #         "enable_color": "true",
    #         "enable_depth": "true",
    #         "align_depth.enable": "true",
    #         "enable_sync": "true",
    #         "enable_gyro": "false",
    #         "enable_accel": "false",
    #         "overview_camera.color.image_raw.enable_pub_plugins":
    #             "['image_transport/raw']",
    #     }.items(),
    # )

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
            "--x",
            LaunchConfiguration("overview_x"),
            "--y",
            LaunchConfiguration("overview_y"),
            "--z",
            LaunchConfiguration("overview_z"),
            "--roll",
            LaunchConfiguration("overview_roll"),
            "--pitch",
            LaunchConfiguration("overview_pitch"),
            "--yaw",
            LaunchConfiguration("overview_yaw"),
            "--frame-id",
            "fr3_link0",
            "--child-frame-id",
            "overview_camera_optical_frame",
        ],
        parameters=[{"use_sim_time": False}],
        output="screen",
    )
    table_apriltag_tf = _load_table_apriltag_tf()
    table_apriltag_tf_nodes = []
    if table_apriltag_tf is not None:
        translation, quaternion, parent_frame, child_frame = table_apriltag_tf
        table_apriltag_tf_nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_tf_fr3_to_table_apriltag",
                arguments=[
                    "--x", str(translation[0]),
                    "--y", str(translation[1]),
                    "--z", str(translation[2]),
                    "--qx", str(quaternion[0]),
                    "--qy", str(quaternion[1]),
                    "--qz", str(quaternion[2]),
                    "--qw", str(quaternion[3]),
                    "--frame-id", parent_frame,
                    "--child-frame-id", child_frame,
                ],
                parameters=[{"use_sim_time": False}],
                output="screen",
            )
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
            experiment_webcam_device_arg,
            experiment_webcam_match_arg,
            *overview_args,
            robot_state_publisher,
            trajectory_adapter,
            gripper_adapter,
            realsense_node,
            experiment_webcam,
            static_tf,
            static_tf_overview,
            *table_apriltag_tf_nodes,
            move_group,
            rviz,
            orchestrator,
        ]
    )
