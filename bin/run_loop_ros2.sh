#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?Usage: $0 \"task description\" [options]}"
shift

options=("$@")
container="vlm_ros2_real"
execute=true
docker_cmd=(docker)
for ((i=0; i<${#options[@]}; i++)); do
    case "${options[i]}" in
        --container)
            i=$((i + 1))
            if ((i >= ${#options[@]})); then
                echo "[ERROR] --container requires a value." >&2
                exit 2
            fi
            container="${options[i]}"
            ;;
        --container=*) container="${options[i]#--container=}" ;;
        --no-execute) execute=false ;;
        --execute) execute=true ;;
        --sudo-docker) docker_cmd=(sudo docker) ;;
    esac
done

if [[ "$execute" == true ]]; then
    echo "[LOOP] Waiting for the real-robot orchestrator in '$container'..."
    ready=false
    ready_deadline=$((SECONDS + 60))
    while (( SECONDS < ready_deadline )); do
        if "${docker_cmd[@]}" exec "$container" bash -c \
            "source /opt/ros/humble/setup.bash 2>/dev/null; \
             source /workspace/ros2_ws/install/setup.bash 2>/dev/null; \
             timeout 2 ros2 topic echo /vlm_planner/ready std_msgs/msg/Bool \
               --once --qos-reliability reliable \
               --qos-durability transient_local --qos-history keep_last \
               --qos-depth 1 2>/dev/null | \
               grep -q '^data: true$'" 2>/dev/null; then
            ready=true
            break
        fi
        sleep 1
    done
    if [[ "$ready" != true ]]; then
        echo "[ERROR] Orchestrator in '$container' did not report ready after 60 seconds." >&2
        exit 1
    fi
else
    echo "[LOOP] Observation-only mode: orchestrator readiness is not required."
fi

source "$REPO_ROOT/.venv/bin/activate"
exec python3 "$REPO_ROOT/scripts/run_loop_ros2_host.py" \
    --task "$TASK" \
    --container "$container" \
    "$@"
