#!/bin/bash
# start_real.sh — Avvia lo stack completo per il robot reale FR3 via franka_ros2.
#
# Architettura (nessun bridge ROS 1 richiesto):
#   franka_hardware (ros2_control plugin) → libfranka → FCI → FR3
#   franka_gripper_node → libfranka → gripper
#   ros2_control controller_manager → fr3_arm_controller (JointTrajectoryController)
#   MoveIt 2 move_group → FollowJointTrajectory → fr3_arm_controller
#
# Prerequisiti:
#   - FCI abilitata in Franka Desk (schermata "Settings → End-Effector")
#   - Robot raggiungibile all'IP indicato (ping funzionante)
#   - Docker image costruita: docker compose build ros2
#   - Nessun ROS 1 bridge necessario
#
# Utilizzo:
#   bin/start_real.sh --robot-ip 192.168.131.1

set -e

ROBOT_IP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot-ip)
            ROBOT_IP="$2"
            shift 2
            ;;
        -h|--help)
            echo "Utilizzo: $0 --robot-ip <IP>"
            exit 0
            ;;
        *)
            echo "Argomento sconosciuto: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ROBOT_IP" ]]; then
    echo "Errore: --robot-ip è obbligatorio."
    echo "Utilizzo: $0 --robot-ip <IP>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Stack robot reale (franka_ros2 + franka_hardware) ==="
echo "Robot IP : $ROBOT_IP"
echo ""

# Verifica connettività base prima di avviare il container
if ! ping -c 1 -W 2 "$ROBOT_IP" &>/dev/null; then
    echo "[WARN] Impossibile raggiungere $ROBOT_IP — verificare la rete."
    echo "       Continuo comunque (il ping potrebbe essere bloccato dal firewall)."
fi

echo "Prerequisiti da verificare prima di continuare:"
echo "  1. FCI abilitata in Franka Desk (pulsante 'Activate FCI')"
echo "  2. Robot non in stato di errore (LED blu fisso)"
echo "  3. Nessun altro processo connesso via FCI (libfranka)"
echo ""
read -rp "Premi INVIO per avviare..."

echo ""
echo "[1/1] Avvio stack franka_ros2 nel container ros2..."
echo "      (ros2_control_node + franka_gripper + MoveIt 2 move_group)"
echo ""

cd "$REPO_ROOT/docker"
docker compose run --rm \
    -e VLM_ROBOT=fr3 \
    ros2 \
    ros2 launch vlm_robot_planner_bringup real_robot.launch.py \
        robot_ip:="$ROBOT_IP"
