#!/bin/bash
# run_task.sh — Cattura scena, lancia VLM e inietta il piano nel container.
#
# Modalità B (default, adaptive): nessun vocabolario pre-definito.
#   Il sistema scopre gli oggetti da Gazebo (/gazebo/model_states) e
#   usa OWL-ViT per il grounding visivo dei nomi VLM.
#
# Modalità A (ablazione, vocabolario esplicito):
#   ./run_task.sh "..." --items red_cup blue_box --locations shelf_b
#
# Altri flag:
#   --dry-run   mostra il piano senza eseguirlo
#   --no-vlm    usa un piano mock (test iniezione)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK="${1:?Uso: $0 \"descrizione task\" [--dry-run|--no-vlm]}"
shift  # rimuovi il task dagli argomenti

# Aspetta che l'orchestratore sia operativo (max 60s)
echo "[INFO] Attendo orchestratore pronto..."
for i in $(seq 1 30); do
    if docker exec vlm_ros2 bash -c \
        "source /opt/ros/humble/setup.bash 2>/dev/null; \
         ros2 topic list 2>/dev/null | grep -q '/vlm_planner/inject_plan'" 2>/dev/null; then
        echo "[OK]   Orchestratore pronto."
        break
    fi
    sleep 2
    if [ $i -eq 30 ]; then
        echo "[WARN] Orchestratore non trovato dopo 60s — continuo comunque."
    fi
done

# Attiva venv e lancia
source "$REPO_ROOT/.venv/bin/activate"

python3 "$REPO_ROOT/scripts/run_vlm_host.py" \
  --task "$TASK" \
  --capture \
  "$@"
