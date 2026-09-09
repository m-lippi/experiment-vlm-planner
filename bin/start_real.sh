#!/bin/bash
# Start the ROS 1 bridge and the ROS 2 MoveIt stack for the physical FR3.

set -euo pipefail

ROBOT_IP="192.168.131.1"
LOCAL_IP=""
RVIZ="true"
PLAN_PREVIEW_DURATION="3.0"
FAIL_ON_EMPTY_GRASP="false"
EMPTY_GRASP_WIDTH_THRESHOLD="0.005"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip) ROBOT_IP="$2"; shift 2 ;;
        --local-ip) LOCAL_IP="$2"; shift 2 ;;
        --no-rviz) RVIZ="false"; shift ;;
        --preview-seconds) PLAN_PREVIEW_DURATION="$2"; shift 2 ;;
        --fail-on-empty-grasp) FAIL_ON_EMPTY_GRASP="true"; shift ;;
        --empty-grasp-threshold)
            FAIL_ON_EMPTY_GRASP="true"
            EMPTY_GRASP_WIDTH_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--robot-ip IP] [--local-ip IP] [--no-rviz] [--preview-seconds N] [--fail-on-empty-grasp] [--empty-grasp-threshold M]"
            echo "Default ROS 1 computer: 192.168.131.1"
            echo "Default empty-grasp threshold: 0.005 m (check disabled)"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$LOCAL_IP" ]]; then
    LOCAL_IP=$(ip route get "$ROBOT_IP" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')
fi
if [[ -z "$LOCAL_IP" ]]; then
    echo "[ERROR] Cannot determine this computer's IP on the robot network."
    echo "        Pass it explicitly with --local-ip."
    exit 1
fi

if ! ping -c 1 -W 2 "$ROBOT_IP" >/dev/null 2>&1; then
    echo "[WARN] $ROBOT_IP did not answer ping; attempting startup anyway."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "ROS 1 robot computer : $ROBOT_IP"
echo "Local ROS 1 address : $LOCAL_IP"
echo "ROS 1 controller    : /effort_joint_trajectory_controller"
echo "ROS 2 domain        : 42"
echo "Empty-grasp check   : $FAIL_ON_EMPTY_GRASP (threshold ${EMPTY_GRASP_WIDTH_THRESHOLD} m)"
echo
echo "The robot computer must already run roscore, publish /joint_states,"
echo "and have effort_joint_trajectory_controller in the running state."

export ROBOT_IP LOCAL_IP RVIZ PLAN_PREVIEW_DURATION
export FAIL_ON_EMPTY_GRASP EMPTY_GRASP_WIDTH_THRESHOLD
cd "$REPO_ROOT/docker"

# Use compose for both processes so Ctrl+C tears down the bridge and MoveIt
# together. No process on this machine opens an FCI/libfranka connection.
exec docker compose --profile real up --abort-on-container-exit \
    ros1_gripper_adapter ros1_bridge  ros2_real
