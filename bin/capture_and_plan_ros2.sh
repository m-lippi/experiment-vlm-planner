#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    echo "[ERROR] Missing $REPO_ROOT/.venv" >&2
    exit 1
fi

source "$REPO_ROOT/.venv/bin/activate"
exec python3 "$REPO_ROOT/scripts/capture_and_plan_ros2.py" "$@"
