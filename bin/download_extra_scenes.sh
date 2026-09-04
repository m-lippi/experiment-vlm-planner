#!/bin/bash
# download_extra_scenes.sh — Download world files and models from
# leonhartyao/gazebo_models_worlds_collection for Phase 2 diversity testing.
#
# Run ONCE (inside the repo root):
#   bin/download_extra_scenes.sh
#
# What it does:
#   1. Downloads 2 world files into ros2_ws/.../worlds/
#   2. Downloads the model dependencies into ros2_ws/.../models/
#      (already in GAZEBO_MODEL_PATH via docker-compose.yml)
#
# Requirements: curl, unzip (already in the Docker container via host network)

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLDS_DIR="$REPO_ROOT/ros2_ws/src/vlm_robot_planner_bringup/worlds"
MODELS_DIR="$REPO_ROOT/ros2_ws/src/vlm_robot_planner_bringup/models"
BASE_URL="https://raw.githubusercontent.com/leonhartyao/gazebo_models_worlds_collection/master"

mkdir -p "$MODELS_DIR"

echo "=== Downloading world files ==="

# Fetchit challenge (raw world, for reference)
curl -fsSL "$BASE_URL/worlds/fetchit_challenge_simple.world" \
     -o "$WORLDS_DIR/fetchit_manipulation.world" && \
     echo "[OK] fetchit_manipulation.world" || echo "[SKIP] fetchit_manipulation.world"

# Office small — desks, cups, bottles, books — great pick&place variety
curl -fsSL "$BASE_URL/worlds/office_small.world" \
     -o "$WORLDS_DIR/office_small_raw.world" && \
     echo "[OK] office_small_raw.world" || echo "[SKIP] office_small_raw.world"

# Workshop — tools (drill, wrench, hammer) — industrial scenario
curl -fsSL "$BASE_URL/worlds/workshop_example.world" \
     -o "$WORLDS_DIR/workshop_raw.world" && \
     echo "[OK] workshop_raw.world" || echo "[SKIP] workshop_raw.world"

echo ""
echo "=== Downloading model dependencies ==="

# All models used by the downloaded world files
for model in fetchit_table fetchit_simple_env caddy_green dropoff_box \
             gearbox_bolt gearbox_bottom gearbox_top \
             100mmbin 50mmbin cpr_office; do
    MODEL_URL="$BASE_URL/models/$model"
    MODEL_DIR="$MODELS_DIR/$model"
    if [ -d "$MODEL_DIR" ]; then
        echo "[SKIP] $model (already exists)"
        continue
    fi
    mkdir -p "$MODEL_DIR"
    for file in model.config model.sdf; do
        curl -fsSL "$MODEL_URL/$file" -o "$MODEL_DIR/$file" 2>/dev/null && \
            echo "[OK]   $model/$file" || echo "[WARN] $model/$file not found"
    done
    # meshes subfolder (optional — skip silently if not available)
    mkdir -p "$MODEL_DIR/meshes"
    for mesh in $(curl -fsSL "https://api.github.com/repos/leonhartyao/gazebo_models_worlds_collection/contents/models/$model/meshes" 2>/dev/null | python3 -c "import sys,json; [print(f['name']) for f in json.load(sys.stdin) if f['type']=='file']" 2>/dev/null); do
        curl -fsSL "$MODEL_URL/meshes/$mesh" -o "$MODEL_DIR/meshes/$mesh" 2>/dev/null && \
            echo "[OK]   $model/meshes/$mesh" || true
    done
done

echo ""
echo "=== Done ==="
echo "Restart Docker container to pick up new models (GAZEBO_MODEL_PATH already includes models/)."
echo "Use 'bin/start_sim.sh --world fetchit_manipulation' or '--world office_manipulation' to launch."

# ═══════════════════════════════════════════════════════════════════════════════
# 3DGEMS Dataset — York University (Rasouli & Tsotsos, 2017)
# https://data.nvision2.eecs.yorku.ca/3DGEMS/
# 270+ modelli 3D per robotica, formato SDF/Gazebo.
# NOTA: modelli STATICI (no inertia/friction) → usare approccio hybrid:
#       visual=mesh 3DGEMS + collision=SDF primitiva (nostra).
# Categorie scaricate: Stationery, Kitchen, Tools, Worlds (~40 MB totale)
# ═══════════════════════════════════════════════════════════════════════════════

GEMS_URL="https://data.nvision2.eecs.yorku.ca/3DGEMS/data"
GEMS_DIR="$MODELS_DIR/3dgems"
mkdir -p "$GEMS_DIR"

echo ""
echo "=== 3DGEMS Dataset (York University) ==="

for category in stationery kitchen tools worlds furniture food electronics; do
    ARCHIVE="$GEMS_DIR/${category}.tar.gz"
    if [ -d "$GEMS_DIR/$category" ] && [ "$(ls -A "$GEMS_DIR/$category" 2>/dev/null)" ]; then
        echo "[SKIP] $category (già estratto)"
        continue
    fi
    echo -n "[DOWNLOAD] $category... "
    if curl -fsSL "$GEMS_URL/${category}.tar.gz" -o "$ARCHIVE" 2>/dev/null; then
        mkdir -p "$GEMS_DIR/$category"
        tar -xzf "$ARCHIVE" -C "$GEMS_DIR/$category" --strip-components=0 2>/dev/null || \
            tar -xzf "$ARCHIVE" -C "$GEMS_DIR/$category" 2>/dev/null || true
        rm -f "$ARCHIVE"
        N=$(find "$GEMS_DIR/$category" -name "model.config" | wc -l)
        echo "[OK] $N modelli estratti"
    else
        echo "[SKIP] non disponibile o errore di rete"
    fi
done

echo ""
echo "=== Modelli 3DGEMS disponibili ==="
for category in stationery kitchen tools furniture food electronics; do
    if [ -d "$GEMS_DIR/$category" ]; then
        models=$(find "$GEMS_DIR/$category" -name "model.config" -exec dirname {} \; | xargs -I{} basename {} 2>/dev/null | sort | tr '\n' ' ')
        echo "  $category: $models"
    fi
done

echo ""
echo "=== Setup GAZEBO_MODEL_PATH per 3DGEMS ==="
echo "Aggiungere a docker-compose.yml GAZEBO_MODEL_PATH:"
echo "  :/workspace/ros2_ws/src/vlm_robot_planner_bringup/models/3dgems/stationery"
echo "  :/workspace/ros2_ws/src/vlm_robot_planner_bringup/models/3dgems/kitchen"
echo "  :/workspace/ros2_ws/src/vlm_robot_planner_bringup/models/3dgems/tools"
echo "(già aggiunto se hai rilanciato lo script dopo questa modifica)"
