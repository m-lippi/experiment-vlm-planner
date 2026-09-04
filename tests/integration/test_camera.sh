#!/bin/bash
# test_camera.sh — Verifica camera topics disponibili e cattura un frame.
# Uso: tests/integration/test_camera.sh
# Richiede: simulazione running (bin/start_sim.sh)

echo "=== TEST CAMERA ==="
echo ""

echo "1. Topic camera disponibili:"
docker exec vlm_ros2 bash -c "source /opt/ros/humble/setup.bash && ros2 topic list | grep -i camera"
echo ""

echo "2. Frequenza pubblicazione (3 secondi per topic):"
for TOPIC in /wrist_camera/image_raw /overview_camera/image_raw /camera/color/image_raw; do
    printf "   %-40s → " "$TOPIC"
    RESULT=$(docker exec vlm_ros2 bash -c \
        "source /opt/ros/humble/setup.bash && timeout 3 ros2 topic hz '$TOPIC' 2>&1 | tail -1")
    echo "$RESULT"
done
echo ""

echo "3. Cattura frame dalla wrist camera:"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
docker exec vlm_ros2 bash -c \
    "source /opt/ros/humble/setup.bash && \
     source /workspace/ros2_ws/install/setup.bash && \
     python3 /workspace/scripts/_capture_scene.py"
echo ""

echo "4. Apertura immagine catturata:"
eog "$REPO_ROOT/data/scene.png" 2>/dev/null &
echo "   Aperta in eog (o controlla data/scene.png)"
