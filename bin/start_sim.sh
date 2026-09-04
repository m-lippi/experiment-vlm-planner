#!/bin/bash
# start_sim.sh — Avvia il container Docker e la simulazione Gazebo + MoveIt2.
#
# Uso:
#   bin/start_sim.sh                    # scena default (tabletop)
#   bin/start_sim.sh --world workshop   # officina
#   bin/start_sim.sh --world office     # ufficio
#   bin/start_sim.sh rviz:=true         # con RViz2
#
# Strategia per stato pulito garantito:
#   - NON distruggere il container (docker compose down + ipc:host lasciano
#     IPC orfani sull'host → conflitti DDS → gazebo_ros2_control si blocca).
#   - Mantenere il container in esecuzione e uccidere solo i processi
#     simulazione vecchi al suo interno.
#   - Pulire shared memory DDS (/dev/shm/ros_*) che causa conflitti.
#   - Reinizializzare il daemon ROS2 per stato fresco.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse argomenti ────────────────────────────────────────────────────────
WORLD_ARG=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --world)
            WORLD_ARG="world_name:=$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

LAUNCH_ARGS="${WORLD_ARG} ${EXTRA_ARGS[*]:-rviz:=false}"

echo "=== VLM-RobotPlanner: avvio simulazione ==="
[ -n "$WORLD_ARG" ] && echo "    Scena: ${WORLD_ARG#world_name:=}" || echo "    Scena: tabletop (default)"

# ── 1. Permessi X11 ────────────────────────────────────────────────────────
xhost +local: > /dev/null 2>&1

# ── 2. Pulizia shared memory sull'host PRIMA di avviare il container ───────
# Con ipc:host il container condivide /dev/shm con l'host. FastRTPS (ROS2 DDS)
# crea file fastrtps_* e SEMAFORI sem.fastrtps_* per ogni nodo ROS2.
# Se il container si ferma senza "docker compose down", questi restano con PID
# orfani → il prossimo ros2 launch trova semafori bloccati → deadlock.
# IMPORTANTE: devono essere rimossi PRIMA che docker compose up avvii il
# container, altrimenti entrypoint.sh li trova già presenti.
echo "    Pulizia shared memory FastRTPS sull'host..."
# File fastrtps_* (segmenti shared memory dei partecipanti DDS)
rm -f /dev/shm/fastrtps_* 2>/dev/null || true
# Semafori sem.fastrtps_* — prefisso "sem." diverso da "fastrtps_*", erano mancanti
rm -f /dev/shm/sem.fastrtps_* 2>/dev/null || true
# Altri eventuali residui ROS2/Gazebo
rm -f /dev/shm/ros_* /dev/shm/*.shm 2>/dev/null || true
rm -f /tmp/.gazebo_master.lock /tmp/gazebo_*.lock 2>/dev/null || true

# ── 3. Avvia container (se non è già in esecuzione) ────────────────────────
cd "$REPO_ROOT/docker"
docker compose up -d

# Attendi che il container sia pronto (entrypoint.sh fa colcon build ~2s)
sleep 2.0

# ── 4. Pulizia processi precedenti DENTRO il container ─────────────────────
echo "    Pulizia processi precedenti nel container..."
docker exec vlm_ros2 bash -c "
    # Kill processi simulazione rimasti
    pkill -9 -f gzserver        2>/dev/null || true
    pkill -9 -f gzclient        2>/dev/null || true
    pkill -9 -f 'ros2 launch'   2>/dev/null || true
    pkill -9 -f orchestrator    2>/dev/null || true
    pkill -9 -f move_group      2>/dev/null || true
    pkill -9 -f spawner         2>/dev/null || true
    pkill -9 -f controller_manager 2>/dev/null || true
    pkill -9 -f robot_state_pub 2>/dev/null || true
    pkill -9 -f static_transform 2>/dev/null || true
    sleep 0.5

    # Pulisci file temporanei ROS2
    rm -f /tmp/ros_* /tmp/fastdds_* /tmp/.ros_* 2>/dev/null || true

    # Reinizializza daemon ROS2
    source /opt/ros/humble/setup.bash 2>/dev/null || true
    ros2 daemon stop 2>/dev/null || true
    sleep 0.3
    ros2 daemon start 2>/dev/null || true
" 2>/dev/null || true

# ── 4. Lancia simulazione — foreground (Ctrl+C per fermare) ────────────────
echo "Lancio Gazebo + MoveIt2 + orchestratore... (Ctrl+C per fermare)"
echo ""
exec docker exec -it vlm_ros2 bash -c \
  "source /opt/ros/humble/setup.bash && \
   source /workspace/ros2_ws/install/setup.bash && \
   ros2 launch vlm_robot_planner_bringup simulation.launch.py ${LAUNCH_ARGS}"
