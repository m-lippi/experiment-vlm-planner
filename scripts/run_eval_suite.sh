#!/usr/bin/env bash
# run_eval_suite.sh — Lancia la suite A-F con capture_and_plan.py
#
# Utilizzo:
#   bash scripts/run_eval_suite.sh <nome>
#
# Struttura output:
#   appunti/<nome>/A/  2026-07-06_14-00-00_pick_the_pen.../
#                  B/  2026-07-06_14-01-30_pick_the_bottle.../
#                  C/  ...
#                  D/  ...
#                  E/  ...
#                  F/  ...
#
# I seriali camera possono essere sovrascritti via env:
#   OVERVIEW_SERIAL=xxx WRIST_SERIAL=yyy bash scripts/run_eval_suite.sh nome

# Non usiamo -e per non abortire sull'errore di un singolo task
set -uo pipefail

# ─── argomenti ───────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo "Utilizzo: bash scripts/run_eval_suite.sh <nome>"
    echo "  <nome>  nome dell'esperimento (es. run_01, baseline, dopo_fix)"
    exit 1
fi

NOME="$1"
OVERVIEW_SERIAL="${OVERVIEW_SERIAL:-242322071571}"
WRIST_SERIAL="${WRIST_SERIAL:-241122072695}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$REPO_DIR/appunti/$NOME"
VENV="$REPO_DIR/.venv/bin/activate"
PY="python3 $REPO_DIR/scripts/capture_and_plan.py"

# ─── venv ────────────────────────────────────────────────────────────────────
if [[ ! -f "$VENV" ]]; then
    echo "[ERROR] venv non trovato: $VENV"
    exit 1
fi
# shellcheck disable=SC1090
source "$VENV"

mkdir -p "$OUT_BASE"

# ─── contatori ───────────────────────────────────────────────────────────────
ok=0; fail=0; total=0

run_task() {
    local group="$1"; shift
    local task="$*"
    total=$((total + 1))
    printf "\n  [%s] %s\n" "$group" "$task"
    if $PY \
            --task         "$task" \
            --overview-serial "$OVERVIEW_SERIAL" \
            --wrist-serial    "$WRIST_SERIAL" \
            -o "$OUT_BASE/$group"; then
        ok=$((ok + 1))
    else
        fail=$((fail + 1))
        echo "  [WARN] task fallito — continuo"
    fi
}

# ─── suite ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  VLM EVAL SUITE — $NOME"
echo "  Output : $OUT_BASE"
echo "  Overview serial : $OVERVIEW_SERIAL"
echo "  Wrist serial    : $WRIST_SERIAL"
echo "================================================================"

echo ""
echo "── GRUPPO A: core primitives (no enrichment) ───────────────────"
run_task A "pick the pen and place it next to the keyboard"
run_task A "put the smartphone on the notebook"
run_task A "move the mouse to the other side of the keyboard"

echo ""
echo "── GRUPPO B: enrichment richiesto (verbo non-core) ─────────────"
run_task B "pick the bottle and pour the water in the cup"
run_task B "cut the paper on the desk with the scissors"
run_task B "write on the notebook with the pen"
run_task B "flip the notebook upside down on the desk"
run_task B "scan the barcode on the bottle with the smartphone"

echo ""
echo "── GRUPPO C: grasp mode reasoning ──────────────────────────────"
run_task C "tilt the bottle to check if it has water"
run_task C "pour from the bottle into the cup"
run_task C "pick up the cup and place it on the keyboard"
run_task C "pick up the pen and drop it into the cup"

echo ""
echo "── GRUPPO D: multi-oggetto / multi-step ────────────────────────"
run_task D "put the pen and the mouse on the notebook"
run_task D "clear the keyboard: move the mouse and the smartphone away from it"
run_task D "pick the pen, place it on the notebook, then put the bottle on the desk"

echo ""
echo "── GRUPPO E: template selection ────────────────────────────────"
run_task E "stack the notebook on top of the laptop"
run_task E "open the laptop and place the smartphone inside it"

echo ""
echo "── GRUPPO F: edge cases ────────────────────────────────────────"
run_task F "hand me the scissors"
run_task F "make a phone call with the smartphone"
run_task F "put everything on the desk in order"

# ─── riepilogo ───────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
printf "  COMPLETATO — %d/%d ok, %d falliti\n" "$ok" "$total" "$fail"
echo "  Risultati in: $OUT_BASE"
echo "================================================================"
