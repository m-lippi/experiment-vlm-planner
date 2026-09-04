#!/usr/bin/env python3
"""
Patch gazebo_ros2_control_plugin.cpp to write robot_description to a temp YAML
file instead of passing it as --param <full_urdf>.

Root cause (Humble):
  gazebo_ros2_control builds an args vector like:
      {"--ros-args", "--param", "robot_description:=<full xml>", "--params-file", "..."}
  and then calls rcl_parse_arguments() on it.  The YAML parser inside rcl chokes
  on the long XML string, logging "Couldn't parse parameter override rule".
  As a result, robot_description never reaches the ControllerManager node and the
  controllers never start (/controller_manager service never appears).

Fix: write robot_description to /tmp/gz_ros2_ctrl_rd.yaml in "/**" params-file
format and pass --params-file instead — rcl handles file-based params perfectly.
"""
import re
import sys

FILEPATH = (
    "/opt/gcr2c_ws/src/gazebo_ros2_control"
    "/gazebo_ros2_control/src/gazebo_ros2_control_plugin.cpp"
)

# ---------------------------------------------------------------------------
# Replacement: write to temp YAML, use --params-file
# Note: the C++ string literals use \\n so the *written* file has real newlines.
# ---------------------------------------------------------------------------
NEW = (
    '  // Write robot_description to a temp YAML file so rcl_parse_arguments can\n'
    '  // load it via --params-file (long URDFs passed via --param crash rcl in Humble).\n'
    '  {\n'
    '    const std::string rd_yaml_path("/tmp/gz_ros2_ctrl_rd.yaml");\n'
    '    std::ofstream rd_yaml_file(rd_yaml_path);\n'
    '    rd_yaml_file << "/**:\\n  ros__parameters:\\n    robot_description: |\\n";\n'
    '    std::istringstream rd_ss(urdf_string);\n'
    '    std::string rd_ln;\n'
    '    while (std::getline(rd_ss, rd_ln)) {\n'
    '      rd_yaml_file << "      " << rd_ln << "\\n";\n'
    '    }\n'
    '  }\n'
    '  arguments.push_back(RCL_PARAM_FILE_FLAG);\n'
    '  arguments.push_back("/tmp/gz_ros2_ctrl_rd.yaml");'
)

# ---------------------------------------------------------------------------
# Apply patch — regex so the trailing comment on the RCL_PARAM_FLAG line
# does not need to match exactly (it differs across patch levels).
# ---------------------------------------------------------------------------
with open(FILEPATH, "r") as fh:
    content = fh.read()

# Ensure required headers are present
for header in ("<fstream>", "<sstream>"):
    if header not in content:
        anchor = "#include <hardware_interface/resource_manager.hpp>"
        if anchor in content:
            content = content.replace(anchor, f"#include {header}\n" + anchor, 1)
        else:
            idx = content.index("#include ")
            content = content[:idx] + f"#include {header}\n" + content[idx:]
        print(f"Inserted {header}")

# Pattern: match the whole "set the robot description parameter" block.
# [^\n]* allows any trailing comment (or none) on the RCL_PARAM_FLAG line.
PATTERN = re.compile(
    r'  // set the robot description parameter\n'
    r'  // to propagate it among controller manager and controllers\n'
    r'  std::string rb_arg = std::string\("robot_description:="\) \+ urdf_string;\n'
    r'  arguments\.push_back\(RCL_PARAM_FLAG\);[^\n]*\n'
    r'  arguments\.push_back\(rb_arg\);',
    re.MULTILINE,
)

m = PATTERN.search(content)
if m is None:
    print(f"ERROR: regex pattern not found in {FILEPATH}")
    print("Context around 'rb_arg' in file:")
    for i, line in enumerate(content.splitlines(), 1):
        if "rb_arg" in line or "robot_description" in line:
            print(f"  {i:4d}: {line}")
    sys.exit(1)

# Use direct string slicing — NOT re.subn(pattern, repl_str, ...) because Python's
# regex engine processes \n in string replacements and would turn the intended C++
# escape sequences into actual newlines, producing invalid C++ source.
content = content[: m.start()] + NEW + content[m.end() :]
print("Main patch applied (regex)")

with open(FILEPATH, "w") as fh:
    fh.write(content)

print(f"Written: {FILEPATH}")
