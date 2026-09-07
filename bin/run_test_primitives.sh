#!/bin/bash
# Run the guarded all-primitives test in the active real-robot container.

set -euo pipefail

CONTAINER_ID=$(docker ps \
    --filter "name=^/vlm_ros2_real$" \
    --format "{{.ID}}" | head -1)

if [[ -z "$CONTAINER_ID" ]]; then
    echo "[ERROR] vlm_ros2_real is not running. Start bin/start_real.sh first."
    exit 1
fi

exec docker exec -it "$CONTAINER_ID" bash -c '
    source /opt/ros/humble/setup.bash
    source /workspace/ros2_ws/install/setup.bash
    export ROS_DOMAIN_ID=42
    export VLM_ROBOT=fr3
    exec python3 /workspace/scripts/test_real_primitives.py "$@"
' -- "$@"
