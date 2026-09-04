#!/bin/bash
# Start only the ROS 1 <-> ROS 2 topic bridge (diagnostics/development helper).

set -euo pipefail

ROBOT_IP="192.168.131.1"
LOCAL_IP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip) ROBOT_IP="$2"; shift 2 ;;
        --local-ip) LOCAL_IP="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--robot-ip IP] [--local-ip IP]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$LOCAL_IP" ]]; then
    LOCAL_IP=$(ip route get "$ROBOT_IP" 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')
fi
if [[ -z "$LOCAL_IP" ]]; then
    echo "[ERROR] Cannot determine the local IP used to reach $ROBOT_IP."
    echo "        Pass it explicitly with --local-ip."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export ROBOT_IP LOCAL_IP
cd "$REPO_ROOT/docker"
exec docker compose --profile real up ros1_bridge
