#!/bin/bash
# reset_scene.sh — Resetta la scena Gazebo (oggetti alle posizioni iniziali).
# Uso: bin/reset_scene.sh
# Richiede: simulazione già in esecuzione (bin/start_sim.sh)

docker exec vlm_ros2 bash -c \
  "source /opt/ros/humble/setup.bash && \
   ros2 service call /reset_world std_srvs/srv/Empty" > /dev/null 2>&1

echo "[OK] Scena resettata — attendi 2s per stabilizzazione fisica..."
sleep 2
echo "[OK] Pronto per il prossimo test."
