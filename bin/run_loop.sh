#!/bin/bash
# run_loop.sh — Closed-loop task execution.
#
# Architettura:
#   [scan pose] → [cattura wrist cam] → [VLM: prossimo step] → [esegui] → [attendi] → [ricattura] → ...
#
# Ad ogni iterazione:
#   1. Muovi braccio in scan pose (look_at)
#   2. Cattura immagine dalla wrist camera
#   3. VLM decide il PROSSIMO SINGOLO step (dato task + step completati)
#   4. Inietta lo step nell'orchestratore
#   5. Aspetta il segnale di completamento step
#   6. Se task completo → exit; altrimenti → prossima iterazione
#
# Uso:
#   bin/run_loop.sh "pick the red cup and place it on the shelf"
#   bin/run_loop.sh "stack the blue box on the red cup"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?Uso: $0 \"descrizione task\"}"
MAX_STEPS=10   # safety limit

# Aspetta orchestratore
echo "[LOOP] Attendo orchestratore pronto..."
for i in $(seq 1 30); do
    if docker exec vlm_ros2 bash -c \
        "source /opt/ros/humble/setup.bash 2>/dev/null; \
         ros2 topic list 2>/dev/null | grep -q '/vlm_planner/inject_plan'" 2>/dev/null; then
        echo "[OK]   Orchestratore pronto."
        break
    fi
    sleep 2
done

source "$REPO_ROOT/.venv/bin/activate"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  CLOSED-LOOP TASK: $TASK"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Chiama lo script Python del loop
python3 "$REPO_ROOT/scripts/run_loop_host.py" --task "$TASK" --max-steps "$MAX_STEPS"
