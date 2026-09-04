#!/bin/bash
set -e

# ── Pulizia shared memory stantia ────────────────────────────────────────────
# Con ipc: host il container condivide /dev/shm con l'host. FastRTPS (ROS2 DDS)
# crea file fastrtps_* e semafori sem.fastrtps_* per ogni partecipante DDS.
# Se il container viene stoppato senza "docker compose down", questi file rimangono
# con PID non più validi → il prossimo avvio trova semafori bloccati → deadlock.
# Rimuovere i file orfani prima di avviare qualsiasi nodo ROS2 risolve il problema.
for _f in /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*; do
    [ -e "$_f" ] || continue
    # Estrai il PID dalla naming convention FastRTPS (solo per i file _el che
    # contengono il PID del processo owning). Per i file senza PID nel nome,
    # rimuoviamo sempre quelli che appartengono al dominio ROS_DOMAIN_ID corrente.
    rm -f "$_f" 2>/dev/null || true
done
# Rimuovi anche eventuali socket/lock di Gazebo da run precedenti
rm -f /tmp/.gazebo_master.lock 2>/dev/null || true
rm -f /tmp/gazebo_*.lock 2>/dev/null || true
# ─────────────────────────────────────────────────────────────────────────────

source /opt/ros/humble/setup.bash

# Overlay source-built gazebo_ros2_control (fixes long-URDF bug in Humble apt package)
if [ -f /opt/gcr2c_ws/install/setup.bash ]; then
    source /opt/gcr2c_ws/install/setup.bash
fi

# Boeing gazebo_model_attachment_plugin (physics-based object grasping)
if [ -f /opt/boeing_ws/install/setup.bash ]; then
    source /opt/boeing_ws/install/setup.bash
fi

# franka_ros2 (franka_hardware, franka_gripper, franka_msgs, …)
if [ -f /opt/franka_ws/install/setup.bash ]; then
    source /opt/franka_ws/install/setup.bash
fi

# Rebuild the Python package so entry point scripts are always current.
# ament_python packages have no C++ — this takes ~2s.
if [ -d /workspace/ros2_ws/src ]; then
    cd /workspace/ros2_ws
    colcon build \
        --symlink-install \
        --packages-select vlm_robot_planner vlm_robot_planner_bringup \
        --event-handlers console_direct- #\
        # 2>&1 | grep -E '(Summary|ERROR|error:)' || true

    # colcon-ros ament_python puts console_scripts in bin/ via pip editable install,
    # but ros2 launch's executable finder looks in lib/<pkg_name>/.
    # Create the missing symlink manually.
    for EXECUTABLE in orchestrator trajectory_topic_adapter gripper_action_adapter; do
        BIN="/workspace/ros2_ws/install/vlm_robot_planner/bin/${EXECUTABLE}"
        LIBEXEC="/workspace/ros2_ws/install/vlm_robot_planner/lib/vlm_robot_planner/${EXECUTABLE}"
        if [ -f "$BIN" ]; then
            mkdir -p "$(dirname "$LIBEXEC")"
            ln -sfn "$BIN" "$LIBEXEC"
        fi
    done

    cd /
fi

# Source workspace overlay
if [ -f /workspace/ros2_ws/install/setup.bash ]; then
    source /workspace/ros2_ws/install/setup.bash
fi

exec "$@"
