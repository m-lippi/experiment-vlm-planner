cmake_minimum_required(VERSION 3.0.2)
project(franka_gripper)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  message_generation
)

add_message_files(
  FILES
    GraspEpsilon.msg
)

add_action_files(
  FILES
    Grasp.action
    Homing.action
    Move.action
)

generate_messages(
  DEPENDENCIES
    actionlib_msgs
)

catkin_package(
  CATKIN_DEPENDS
    actionlib_msgs
    message_runtime
)
