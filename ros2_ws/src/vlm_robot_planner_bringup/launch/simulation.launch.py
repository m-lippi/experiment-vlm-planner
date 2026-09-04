"""
simulation.launch.py — Full simulation bringup for VLM-RobotPlanner (Phase 1).

Starts (in order):
  1. Gazebo Classic 11  with tabletop.world
  2. robot_state_publisher   (publishes /robot_description + TF)
  3. spawn_entity            (spawns Franka Panda into Gazebo)
  4. controller_manager      (via gazebo_ros2_control plugin, already in URDF)
  5. joint_state_broadcaster (publishes /joint_states)
  6. panda_arm_controller    (JointTrajectoryController — MoveIt2 target)
  7. panda_hand_controller   (GripperActionController — pick/place target)
  8. MoveIt2 move_group      (motion planning server)
  9. RViz2                   (visualization, optional — set rviz:=false to skip)
 10. Orchestrator node       (VLM→PDDL→primitive dispatch)

Usage:
  ros2 launch vlm_robot_planner_bringup simulation.launch.py
  ros2 launch vlm_robot_planner_bringup simulation.launch.py rviz:=false
"""

from __future__ import annotations

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory, get_package_share_path
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description() -> LaunchDescription:

    # ── Arguments ────────────────────────────────────────────────────────────
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 for visualization",
    )
    world_arg = DeclareLaunchArgument(
        "world_name",
        default_value="tabletop",
        description=(
            "World file to load (without .world extension). "
            "Available: tabletop (default), fetchit_manipulation, office_manipulation. "
            "Run bin/download_extra_scenes.sh first to get the extra worlds."
        ),
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    bringup_share   = get_package_share_directory("vlm_robot_planner_bringup")
    gazebo_ros_share = get_package_share_directory("gazebo_ros")

    world_path = PathJoinSubstitution([
        bringup_share, "worlds",
        [LaunchConfiguration("world_name"), ".world"]
    ])
    urdf_path  = os.path.join(bringup_share, "urdf",   "panda_sim.urdf.xacro")
    rviz_path  = os.path.join(bringup_share, "config", "moveit.rviz")

    # ── Robot description (xacro → URDF string) ───────────────────────────────
    # ParameterValue(..., value_type=str) prevents ROS2 from trying to parse
    # the URDF XML as YAML (common pitfall in Humble).
    robot_description_content = ParameterValue(
        Command([FindExecutable(name="xacro"), " ", urdf_path]),
        value_type=str,
    )
    robot_description = {"robot_description": robot_description_content}

    # ── MoveIt 2 configuration ────────────────────────────────────────────────
    # Use moveit_resources_panda_moveit_config for SRDF + kinematics,
    # but override robot_description with our custom xacro (adds camera).
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="panda",
            package_name="moveit_resources_panda_moveit_config",
        )
        .robot_description(file_path=urdf_path)
        .robot_description_semantic(
            file_path=str(
                get_package_share_path("moveit_resources_panda_moveit_config")
                / "config" / "panda.srdf"
            )
        )
        .robot_description_kinematics(
            # Use our local kinematics.yaml (TRAC-IK Distance) instead of the
            # KDL default from moveit_resources_panda_moveit_config.
            # TRAC-IK prevents joint-space jumps in Cartesian paths for
            # non-standard EEF orientations (e.g. side grasp mode).
            file_path=os.path.join(bringup_share, "config", "kinematics.yaml")
        )
        .to_moveit_configs()
    )

    # Add sim_time to all MoveIt2 parameters
    moveit_params = moveit_config.to_dict()
    moveit_params["use_sim_time"] = True

    # ── 1. Gazebo Classic ─────────────────────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_share, "launch", "gazebo.launch.py")
        ),
        launch_arguments={
            "world":   world_path,
            "verbose": "false",
            "pause":   "false",
        }.items(),
    )

    # ── 2. robot_state_publisher ──────────────────────────────────────────────
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    # ── 2b. Static TF: world → panda_link0 ───────────────────────────────────
    # MoveIt2 SRDF defines virtual_joint (fixed, world→panda_link0).
    # Table: 90×60 cm centred at world x=0.50 m → back edge at x=0.05 m.
    # Robot base radius ≈ 12 cm → spawn at x=0.20 m gives 3 cm clearance past
    # the table back edge (0.05 + 0.12 + 0.03 = 0.20 m), base fully on table.
    # z=0.77 m = table surface height (standard Franka-on-table setup).
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_world_to_panda",
        arguments=["0.20", "0", "0.77", "0", "0", "0", "world", "panda_link0"],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 3. Spawn Franka Panda into Gazebo ─────────────────────────────────────
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "panda",
            "-x", "0.20", "-y", "0.0", "-z", "0.77",
        ],
        output="screen",
    )

    # ── 4–5. Controllers (spawned after robot is in Gazebo) ───────────────────
    spawn_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    spawn_arm_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_joint_state_broadcaster,
            on_exit=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=["panda_arm_controller", "--controller-manager", "/controller_manager"],
                    output="screen",
                )
            ],
        )
    )

    spawn_hand_controller = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_joint_state_broadcaster,
            on_exit=[
                Node(
                    package="controller_manager",
                    executable="spawner",
                    arguments=["panda_hand_controller", "--controller-manager", "/controller_manager"],
                    output="screen",
                )
            ],
        )
    )

    # Delay controllers until robot is spawned
    controller_bringup = TimerAction(
        period=3.0,
        actions=[spawn_joint_state_broadcaster],
    )

    # ── 6. MoveIt 2 move_group ────────────────────────────────────────────────
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

    # ── 8. Orchestrator node ──────────────────────────────────────────────────
    # Delay to ensure MoveIt2 move_group is ready before the orchestrator starts.
    orchestrator = TimerAction(
        period=8.0,
        actions=[
            Node(
                package="vlm_robot_planner",
                executable="orchestrator",
                output="screen",
                parameters=[
                    moveit_params,
                    {"use_sim_time": True},
                ],
            )
        ],
    )

    return LaunchDescription([
        rviz_arg,
        world_arg,
        # Gazebo + robot
        gazebo,
        robot_state_publisher,
        static_tf,
        spawn_robot,
        # Controllers (delayed)
        controller_bringup,
        spawn_arm_controller,
        spawn_hand_controller,
        # MoveIt2 + RViz2
        move_group,
        rviz2,
        # Orchestrator (delayed until MoveIt2 is up)
        orchestrator,
    ])
