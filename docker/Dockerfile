# ROS 2 Humble + Gazebo Classic 11 + MoveIt 2 + Franka
# Ubuntu 22.04 inside Docker — host stays on Ubuntu 20.04
# VLM weights and GPU inference run on the HOST (not in this container).
# This container handles only: ROS 2, Gazebo, MoveIt 2, robot control.

FROM osrf/ros:humble-desktop-full

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# --allow-unauthenticated: GPG keys in the base image become invalid after
# upstream key rotations (Ubuntu + ROS2). Acceptable for a dev/research container.
RUN apt-get update \
      -o Acquire::AllowInsecureRepositories=true \
      -o Acquire::AllowDowngradeToInsecureRepositories=true \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
    # Build tools
    python3-pip \
    python3-colcon-common-extensions \
    python3-vcstool \
    python3-rosdep \
    # Fast Downward build dependencies
    cmake \
    g++ \
    libglib2.0-dev \
    # MoveIt 2 (arm motion planning)
    ros-humble-moveit \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-resources-panda-moveit-config \
    ros-humble-moveit-configs-utils \
    ros-humble-moveit-ros-move-group \
    # pick_ik: PickNik's recommended IK solver for MoveIt2 on ROS2 Humble.
    # mode=local uses gradient descent starting from the seed (current joint state),
    # so Cartesian interpolation naturally selects nearby solutions — prevents
    # joint-space jumps for non-standard EEF orientations (e.g. side grasp).
    # (trac_ik_kinematics_plugin is not an apt package in Humble; build-from-source)
    ros-humble-pick-ik \
    # Gazebo Classic 11 + ros2_control bridge
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    # ros2_control stack
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-gripper-controllers \
    ros-humble-position-controllers \
    # moveit_resources panda description (URDF + meshes, used by moveit_config)
    ros-humble-moveit-resources-panda-description \
    # Visualization & utilities
    ros-humble-rviz2 \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-tf2-ros \
    ros-humble-tf2-tools \
    # cv_bridge for camera image conversion
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-realsense2-*\
    # Mesa software rendering — fixes libGL/libGLX version mismatch that causes
    # Gazebo/OGRE black screen after Docker image rebuild.
    # The rebuild updates libGLX_mesa (Oct 2024) but leaves libGL.so at Jan 2022,
    # causing silent OGRE rendering failure.
    # Explicitly installing all Mesa packages together ensures consistent versions.
    libglvnd0 \
    libgl1 \
    libglx0 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    mesa-utils \
    x11-utils \
    # Utils
    git wget vim curl \
    # libfranka build dependencies (libfranka built from source below).
    libpoco-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# libfranka 0.13.3 — built from source to match the FR3 controller firmware.
# Protocol version mapping (verified empirically via libfranka-common kVersion):
#   0.10.0 – 0.13.2 → library version 6 (incompatible with server v7)
#   0.13.3 – 0.13.6 → library version 7 (server v7) — CORRECT TARGET
#   0.14.0           → library version 8
#   0.15.0+          → library version 9 (ros-humble-libfranka APT)
# git clone --recursive è necessario: il submodulo 'common' (libfranka-common)
# è obbligatorio per la build principale di 0.12.0+.
# Richiede --network=host nel docker build per risolvere github.com.
RUN git clone --recursive --branch 0.13.3 --depth 1 \
      https://github.com/frankaemika/libfranka.git /opt/libfranka_src \
    && mkdir /opt/libfranka_src/build \
    && cd /opt/libfranka_src/build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_TESTS=OFF \
             -DBUILD_EXAMPLES=OFF \
             .. \
    && make -j"$(nproc)" \
    && make install \
    && cp -r /opt/libfranka_src/common/include/research_interface /usr/local/include/ \
    && ldconfig

# Python packages needed inside the container (non-VLM)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    pillow \
    transforms3d

# Build gazebo_ros2_control from source with a targeted patch that fixes the
# Humble bug where rcl_parse_arguments chokes on --param robot_description:=<xml>
# (long URDFs contain characters that break the RCL YAML arg parser).
# The patch writes robot_description to a temp YAML file and uses --params-file.
COPY docker/patch_gazebo_plugin.py /tmp/patch_gazebo_plugin.py

RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    git clone --branch humble --depth 1 \
      https://github.com/ros-controls/gazebo_ros2_control.git \
      /opt/gcr2c_ws/src/gazebo_ros2_control && \
    python3 /tmp/patch_gazebo_plugin.py && \
    cd /opt/gcr2c_ws && \
    colcon build \
      --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --packages-select gazebo_ros2_control \
      --event-handlers console_direct+"

# Boeing gazebo_model_attachment_plugin — physics-based object grasping.
# Replaces the teleport-based GazeboAttach (which oscillates due to ODE gravity
# fighting kinematic updates). Creates a real Gazebo fixed joint at grasp time.
RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    git clone --branch humble --depth 1 \
      https://github.com/Boeing/gazebo_model_attachment_plugin.git \
      /opt/boeing_ws/src/gazebo_model_attachment_plugin && \
    cd /opt/boeing_ws && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build \
      --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release \
      --event-handlers console_direct+"

# franka_ros2 v0.1.0 — ROS 2 Humble drivers for Franka robots.
# Version 0.1.0 is used because it was released for libfranka 0.10.x
# (no minimum version constraint in CMakeLists).  The current 'humble'
# branch requires libfranka 0.13.2 which is incompatible with the FR3
# controller firmware at server version 7.
#
# Packages built:
#   franka_msgs      — ROS 2 message types (GraspEpsilon, Errors, FrankaRobotState)
#   franka_hardware  — ros2_control plugin (patched: POSITION control via libfranka 0.10.0)
#   franka_gripper   — gripper ROS 2 node (direct FCI connection)
#
# NOT built (franka_semantic_components depends on controller_interface
# semantic_components header that moved between Humble patch releases):
#   franka_semantic_components     — optional diagnostics
#   franka_robot_state_broadcaster — optional diagnostics
#
# franka_description and franka_fr3_moveit_config are NOT built here because
# they live in ros2_ws/src/ with our custom overrides.
COPY docker/patch_franka_hardware.py /tmp/patch_franka_hardware.py
# franka_ros2 v0.1.0 source — bundled as tarball to avoid git network dependency.
# Extract top-level directory (franka_ros2-0.1.0/) to /opt/franka_ws/src/franka_ros2/.
COPY docker/franka_ros2_v010.tar.gz /tmp/franka_ros2_v010.tar.gz

RUN bash -c "\
    source /opt/ros/humble/setup.bash && \
    mkdir -p /opt/franka_ws/src && \
    tar -xzf /tmp/franka_ros2_v010.tar.gz -C /opt/franka_ws/src && \
    mv /opt/franka_ws/src/franka_ros2-0.1.0 /opt/franka_ws/src/franka_ros2 && \
    \
    python3 /tmp/patch_franka_hardware.py && \
    \
    for pkg in franka_hardware franka_gripper franka_msgs; do \
      f=/opt/franka_ws/src/franka_ros2/\${pkg}/CMakeLists.txt; \
      [ -f \"\${f}\" ] && \
        sed -i 's/find_package(ament_cmake_clang_format/# find_package(ament_cmake_clang_format/g' \"\${f}\" && \
        sed -i 's/find_package(ament_cmake_clang_tidy/# find_package(ament_cmake_clang_tidy/g'   \"\${f}\" && \
        sed -i 's/ament_clang_format(/# ament_clang_format(/g'  \"\${f}\" && \
        sed -i 's/ament_clang_tidy(/# ament_clang_tidy(/g'      \"\${f}\"; \
    done && \
    cd /opt/franka_ws && \
    colcon build \
      --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
      --packages-select \
        franka_msgs \
        franka_hardware \
        franka_gripper \
      --event-handlers console_direct+"

# Fast Downward PDDL planner — build from source (takes ~2-3 min)
RUN git clone --depth 1 https://github.com/aibasel/downward.git /opt/fast-downward \
 && cd /opt/fast-downward && python3 build.py \
 && ln -sf /opt/fast-downward/fast-downward.py /usr/local/bin/fast-downward

WORKDIR /workspace

# Build ROS 2 workspace (our vlm_robot_planner + vlm_robot_planner_bringup packages).
# MoveIt2 motion planning uses pymoveit2 (pure Python, pip-installed above) —
# no C++ moveit_py compilation required.
COPY ros2_ws/ ros2_ws/
RUN bash -c "source /opt/ros/humble/setup.bash && \
    cd /workspace/ros2_ws && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Copy pure-Python planning modules (accessible from container for debugging)
COPY vlm/      vlm/
COPY pddl/     pddl/
COPY planner/  planner/
COPY simulation/ simulation/
COPY scripts/  scripts/

# Auto-source ROS 2 overlays in every interactive shell (docker exec bash).
# The workspace overlay is built at container startup by entrypoint.sh, so
# guard it with [ -f ] to avoid errors if it doesn't exist yet.
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
 && echo "[ -f /opt/gcr2c_ws/install/setup.bash ]  && source /opt/gcr2c_ws/install/setup.bash"  >> /root/.bashrc \
 && echo "[ -f /opt/boeing_ws/install/setup.bash ]  && source /opt/boeing_ws/install/setup.bash"  >> /root/.bashrc \
 && echo "[ -f /opt/franka_ws/install/setup.bash ]  && source /opt/franka_ws/install/setup.bash"  >> /root/.bashrc \
 && echo "[ -f /workspace/ros2_ws/install/setup.bash ] && source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc

# Entrypoint sources both ROS 2 overlays
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
