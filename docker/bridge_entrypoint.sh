#!/bin/bash
set -e

: "${ROS_MASTER_URI:=http://192.168.131.1:11311}"
: "${ROS_DOMAIN_ID:=42}"

export ROS_MASTER_URI ROS_DOMAIN_ID

if [[ -z "${ROS_IP:-}" ]]; then
    echo "[bridge] ROS_IP is required so the remote ROS 1 master can call back."
    exit 2
fi

source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash

echo "[bridge] ROS 1 master: ${ROS_MASTER_URI}"
echo "[bridge] ROS 1 local IP: ${ROS_IP}"
echo "[bridge] ROS 2 domain: ${ROS_DOMAIN_ID}"

# Force creation of ROS 1 -> ROS 2 topics such as /joint_states even before a
# ROS 2 CLI subscriber exists. ROS 2 -> ROS 1 command topics are created when
# the ROS 1 effort controller subscriber is discovered.
exec ros2 run ros1_bridge dynamic_bridge -- \
    --bridge-all-topics
