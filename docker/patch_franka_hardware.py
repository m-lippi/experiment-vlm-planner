#!/usr/bin/env python3
"""
patch_franka_hardware.py — patches franka_hardware v0.1.0 to expose a
POSITION command interface instead of the default EFFORT (torque) interface.

Background:
  franka_hardware v0.1.0 only exports effort command interfaces via
  libfranka torque control.  The FR3's controller firmware (server v7)
  requires libfranka 0.12.0, which supports franka::JointPositions
  control (internal joint impedance controller with position setpoints).
  This is safer and simpler than torque control for trajectory execution.

Files patched (all under /opt/franka_ws/src/franka_ros2/franka_hardware/):
  include/franka_hardware/robot.hpp           — add position control declarations
  src/robot.cpp                               — add position control implementation
  src/franka_hardware_interface.cpp           — switch effort → position throughout
"""

import pathlib
import sys

BASE = pathlib.Path("/opt/franka_ws/src/franka_ros2/franka_hardware")


def patch_file(path: pathlib.Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text()
    for old, new in replacements:
        if old not in text:
            print(f"  WARNING: string not found in {path.name}: {old[:70]!r}", flush=True)
            continue
        text = text.replace(old, new)
        print(f"  OK: {old[:60]!r}", flush=True)
    path.write_text(text)


# ── 1. robot.hpp — declare initializePositionControl / writePositions ─────────
print("Patching robot.hpp …")
patch_file(
    BASE / "include/franka_hardware/robot.hpp",
    [
        # Add position control declarations after write()
        (
            "  virtual void write(const std::array<double, 7>& efforts);",
            """\
  virtual void write(const std::array<double, 7>& efforts);

  /**
   * Starts a joint-position control loop using libfranka's internal impedance
   * controller (franka::JointPositions).  Compatible with libfranka 0.10.x.
   * Safe: the robot's internal controller handles smooth tracking.
   */
  virtual void initializePositionControl();

  /**
   * Sends new joint position setpoints in a thread-safe way.
   * @param[in] positions  target joint angles [rad] for each of the 7 joints.
   */
  virtual void writePositions(const std::array<double, 7>& positions);""",
        ),
        # Add pos_command_ member alongside tau_command_
        (
            "  std::array<double, 7> tau_command_{};",
            """\
  std::array<double, 7> tau_command_{};
  std::array<double, 7> pos_command_{};""",
        ),
    ],
)

# ── 2. robot.cpp — implement initializePositionControl / writePositions ────────
print("Patching robot.cpp …")
patch_file(
    BASE / "src/robot.cpp",
    [
        # Insert position control implementation before initializeContinuousReading
        (
            "void Robot::initializeContinuousReading() {",
            """\
void Robot::initializePositionControl() {
  assert(isStopped());
  stopped_ = false;
  // Seed pos_command_ from the current robot state so the first command does
  // not jump away from wherever the robot is resting.
  {
    franka::RobotState init_state = robot_->readOnce();
    std::lock_guard<std::mutex> lock(write_mutex_);
    pos_command_ = init_state.q;
  }
  const auto kPositionControl = [this]() {
    robot_->control(
        [this](const franka::RobotState& state,
               const franka::Duration& /*period*/) -> franka::JointPositions {
          {
            std::lock_guard<std::mutex> lock(read_mutex_);
            current_state_ = state;
          }
          std::lock_guard<std::mutex> lock(write_mutex_);
          franka::JointPositions out(pos_command_);
          out.motion_finished = finish_;
          return out;
        });
  };
  control_thread_ = std::make_unique<std::thread>(kPositionControl);
}

void Robot::writePositions(const std::array<double, 7>& positions) {
  std::lock_guard<std::mutex> lock(write_mutex_);
  pos_command_ = positions;
}

void Robot::initializeContinuousReading() {""",
        ),
    ],
)

# ── 3. franka_hardware_interface.cpp — switch effort → position throughout ────
print("Patching franka_hardware_interface.cpp …")
patch_file(
    BASE / "src/franka_hardware_interface.cpp",
    [
        # export_command_interfaces(): EFFORT → POSITION
        (
            "        info_.joints[i].name, hardware_interface::HW_IF_EFFORT, &hw_commands_.at(i)));",
            "        info_.joints[i].name, hardware_interface::HW_IF_POSITION, &hw_commands_.at(i)));",
        ),
        # write(): use writePositions instead of write (torque)
        (
            "  robot_->write(hw_commands_);",
            "  robot_->writePositions(hw_commands_);",
        ),
        # on_init(): command interface validation — EFFORT → POSITION
        (
            "    if (joint.command_interfaces[0].name != hardware_interface::HW_IF_EFFORT) {",
            "    if (joint.command_interfaces[0].name != hardware_interface::HW_IF_POSITION) {",
        ),
        (
            # RCLCPP_FATAL that says Expected 'effort' (on_init)
            "                   hardware_interface::HW_IF_EFFORT);\n      return CallbackReturn::ERROR;\n    }\n    if (joint.state_interfaces.size() != 3)",
            "                   hardware_interface::HW_IF_POSITION);\n      return CallbackReturn::ERROR;\n    }\n    if (joint.state_interfaces.size() != 3)",
        ),
        # perform_command_mode_switch(): initializeTorqueControl → initializePositionControl
        (
            "    robot_->initializeTorqueControl();",
            "    robot_->initializePositionControl();",
        ),
        # prepare_command_mode_switch(): is_effort_interface lambda → HW_IF_POSITION
        (
            "    return interface.find(hardware_interface::HW_IF_EFFORT) != std::string::npos;",
            "    return interface.find(hardware_interface::HW_IF_POSITION) != std::string::npos;",
        ),
        # Error messages in prepare_command_mode_switch
        (
            '"Expected %ld effort interfaces to stop, but got %ld instead."',
            '"Expected %ld position interfaces to stop, but got %ld instead."',
        ),
        (
            '"Invalid number of effort interfaces to stop. Expected "',
            '"Invalid number of position interfaces to stop. Expected "',
        ),
        (
            '"Expected %ld effort interfaces to start, but got %ld instead."',
            '"Expected %ld position interfaces to start, but got %ld instead."',
        ),
        (
            '"Invalid number of effort interfaces to start. Expected "',
            '"Invalid number of position interfaces to start. Expected "',
        ),
    ],
)

# ── 4. franka_hardware_interface.hpp — fix k_robot_name for FR3 ───────────────
# v0.1.0 hardcodes k_robot_name = "panda".  The franka_robot_state_broadcaster
# queries state interfaces under this name (via arm_id param).  Changing it to
# "fr3" makes the broadcaster's arm_id: fr3 config consistent.
print("Patching franka_hardware_interface.hpp …")
patch_file(
    BASE / "include/franka_hardware/franka_hardware_interface.hpp",
    [
        (
            '  const std::string k_robot_name{"panda"};',
            '  const std::string k_robot_name{"fr3"};',
        ),
    ],
)

# ── 5. franka_semantic_components headers — fix include path ──────────────────
# In v0.1.0 the header is included as "semantic_components/...". In ROS 2 Humble
# (current) it moved to "controller_interface/semantic_components/...".
print("Patching franka_semantic_components include paths …")
SCOMP_BASE = pathlib.Path(
    "/opt/franka_ws/src/franka_ros2/franka_semantic_components/include/franka_semantic_components"
)
for hpp_name in ("franka_robot_state.hpp", "franka_robot_model.hpp"):
    patch_file(
        SCOMP_BASE / hpp_name,
        [
            (
                '"semantic_components/semantic_component_interface.hpp"',
                '"controller_interface/semantic_components/semantic_component_interface.hpp"',
            ),
        ],
    )

# Also ensure controller_interface is in the ament_target_dependencies for the
# production library (it was only in the test section in v0.1.0).
cmake_scomp = pathlib.Path(
    "/opt/franka_ws/src/franka_ros2/franka_semantic_components/CMakeLists.txt"
)
text = cmake_scomp.read_text()
old_dep = "set(THIS_PACKAGE_INCLUDE_DEPENDS franka_hardware franka_msgs hardware_interface rclcpp_lifecycle)"
new_dep = "set(THIS_PACKAGE_INCLUDE_DEPENDS franka_hardware franka_msgs hardware_interface controller_interface rclcpp_lifecycle)"
if old_dep in text:
    text = text.replace(old_dep, new_dep)
    cmake_scomp.write_text(text)
    print("  OK: added controller_interface to THIS_PACKAGE_INCLUDE_DEPENDS")
else:
    print("  WARNING: could not patch franka_semantic_components CMakeLists.txt")

print("\nAll patches applied successfully.")
