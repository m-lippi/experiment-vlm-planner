cmake_minimum_required(VERSION 3.8)
project(vlm_robot_planner_bringup)

find_package(ament_cmake REQUIRED)

# Install all resource directories (launch, config, worlds, urdf)
install(
  DIRECTORY launch config worlds urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
