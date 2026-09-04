#!/bin/bash
# run_test_move.sh — Run the real-robot movement test inside the active ros2 container.
#
# Usage:
#   bin/run_test_move.sh [--velocity 0.1] [--check-only] [--no-confirm]
#
# Prerequisites:
#   - bin/start_real.sh is running in another terminal (bridge + MoveIt ready)

VELOCITY="0.1"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --velocity)
            VELOCITY="$2"
            shift 2
            ;;
        --check-only)
            EXTRA_ARGS="$EXTRA_ARGS --check-only"
            shift
            ;;
        --no-confirm)
            EXTRA_ARGS="$EXTRA_ARGS --no-confirm"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--velocity N] [--check-only] [--no-confirm]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Find the running ros2 container.
CONTAINER_ID=$(docker ps --filter "ancestor=vlm-robot-planner:latest" --format "{{.ID}}" | head -1)

if [[ -z "$CONTAINER_ID" ]]; then
    echo "[ERROR] Nessun container ros2 in esecuzione."
    echo "Avvia prima: bin/start_real.sh"
    exit 1
fi

echo "Container ros2: $CONTAINER_ID"
echo "Velocità: ${VELOCITY} ($(echo "$VELOCITY * 100" | bc)%)"
echo ""

docker exec -it "$CONTAINER_ID" bash -c "
  export ROS_DOMAIN_ID=42 &&
  source /opt/ros/humble/setup.bash &&
  source /workspace/ros2_ws/install/setup.bash &&
  VLM_ROBOT=fr3 python3 /workspace/scripts/test_real_move.py \
    --velocity $VELOCITY $EXTRA_ARGS
"
