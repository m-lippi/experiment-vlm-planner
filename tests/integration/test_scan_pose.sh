#!/bin/bash
# test_scan_pose.sh — Prova diverse scan pose per trovare quella ottimale.
# Ogni pose muove il braccio, cattura un'immagine e la salva con nome descrittivo.
# Uso: tests/integration/test_scan_pose.sh
# Richiede: simulazione + orchestratore running

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/scan_pose_test"
mkdir -p "$DATA_DIR"

echo "=== TEST SCAN POSE ==="
echo "Salvo immagini in $DATA_DIR/"
echo ""

# Configurazioni da testare: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
# La wrist camera ha 30° di tilt fisso. joint6 aggiunge flessione al polso.
declare -A POSES=(
    ["ready"]="0.0 -0.785 0.0 -2.356 0.0 1.571 0.785"
    ["scan_j6_1.5"]="0.0 -0.3 0.0 -1.5 0.0 1.5 0.785"
    ["scan_j6_2.0"]="0.0 -0.3 0.0 -1.5 0.0 2.0 0.785"
    ["scan_j6_2.5"]="0.0 -0.3 0.0 -1.5 0.0 2.5 0.785"
    ["scan_forward"]="0.0 -0.5 0.0 -1.8 0.0 2.0 0.0"
    ["scan_low"]="0.0 -0.1 0.0 -1.2 0.0 1.8 0.785"
)

for NAME in "${!POSES[@]}"; do
    JOINTS="${POSES[$NAME]}"
    echo "Pose: $NAME → joints [$JOINTS]"

    # Inietta look_at con questa pose (modificando temporaneamente _NAMED_CONFIGS)
    # Per semplicità: inietta direttamente un piano con joint goal
    PAYLOAD=$(python3 -c "
import json
joints = list(map(float, '$JOINTS'.split()))
print(json.dumps({
    'command': 'test_scan',
    'vlm_plan': {
        'goal': 'test',
        'steps': [{'primitive': 'look_at', 'args': {'target': 'scene'}}],
        'raw_output': '',
        'domain_template': 'manipulation_base',
        'domain_additions': {'new_types':[],'new_predicates':[],'new_actions':[],'modified_preconditions':{}}
    }
}))")

    # Aspetta orchestratore ready, poi inietta
    for i in $(seq 1 10); do
        STATUS=$(docker exec vlm_ros2 bash -c \
            "source /opt/ros/humble/setup.bash && \
             timeout 1 ros2 topic echo /vlm_planner/status --once 2>/dev/null | head -2" 2>/dev/null)
        if echo "$STATUS" | grep -q "ready"; then break; fi
        sleep 1
    done

    echo "$PAYLOAD" | docker exec -i vlm_ros2 bash -c \
        "source /opt/ros/humble/setup.bash && \
         source /workspace/ros2_ws/install/setup.bash && \
         python3 /workspace/scripts/_publish_plan.py" > /dev/null 2>&1

    sleep 4  # attendi arm motion

    # Cattura immagine
    docker exec vlm_ros2 bash -c \
        "source /opt/ros/humble/setup.bash && \
         source /workspace/ros2_ws/install/setup.bash && \
         python3 /workspace/scripts/_capture_scene.py" > /dev/null 2>&1

    cp "$REPO_ROOT/data/scene.png" "$DATA_DIR/pose_${NAME}.png"
    echo "   → $DATA_DIR/pose_${NAME}.png"
    echo ""
done

echo "Apertura immagini per confronto..."
eog "$DATA_DIR"/ 2>/dev/null &
echo "Controlla le immagini in $DATA_DIR/ e scegli la posa migliore."
