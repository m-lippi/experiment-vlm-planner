# VLM-RobotPlanner

A hybrid planning system for robot manipulation tasks: a Vision-Language Model (Qwen3-VL-8B-Instruct) interprets natural-language commands and scene images, generates a PDDL plan, and dispatches it to a Franka Emika Panda arm via ROS 2 / MoveIt 2.

Extends the approach of *"Look Before You Leap: Unveiling the Power of GPT-4V in Robotic Vision-Language Planning"* using an entirely open-source stack.

---

## System requirements

| Component | Requirement |
|-----------|-------------|
| OS (host) | Ubuntu 20.04 LTS |
| GPU | NVIDIA GPU with ≥16 GB VRAM (tested on RTX 3090 Ti, 24 GB) |
| CUDA | 11.8 |
| Docker | ≥ 24.0 with `docker compose` v2 |
| Python | 3.10+ (host venv) |

The **host** runs VLM inference and the Python planning layer.  
The **Docker container** (Ubuntu 22.04) runs ROS 2 Humble, Gazebo Classic 11, and MoveIt 2.

---

## Repository layout

```
vlm/                    # VLM inference — Qwen3-VL-8B, image preprocessing, GroundingDINO-tiny
planner/                # PDDL pipeline — DomainEnricher, ProblemGenerator, FastDownward wrapper
pddl/domains/           # Four PDDL domain templates (manipulation_base, stacking, containers, navigation)
simulation/oracle/      # GazeboOracle: ground-truth object poses from Gazebo (sim only)
ros2_ws/src/
  vlm_robot_planner/    # ROS 2 package — Orchestrator node, primitives, MoveIt2Client
  vlm_robot_planner_bringup/  # Launch files, world files, robot URDF/SRDF, 3D models
scripts/                # Host-side entry points (run_loop_host.py, capture_and_plan.py, setup_overview_camera.py, …)
bin/                    # Shell convenience wrappers (start_sim.sh, run_task.sh, run_loop.sh)
tests/                  # Unit tests (no GPU or ROS required)
docker/                 # Dockerfile and docker-compose.yml
data/                   # Runtime output — captured images, run logs, pose JSON
```

---

## Setup

### 1 — Clone and enter the repository

```bash
git clone <repo-url> VLM-RobotPlanner
cd VLM-RobotPlanner
```

### 2 — Host Python environment (VLM + planning layer)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> **Note:** Qwen3-VL-8B-Instruct requires `transformers>=5.0` — this is satisfied by `requirements.txt`. The model weights (~16 GB) are downloaded automatically from HuggingFace on first inference. Set `HF_HOME` to a directory with enough space if needed.

### 3 — Docker container (ROS 2 + Gazebo + MoveIt 2)

```bash
# Build the image (first time only — takes ~10 min)
docker compose -f docker/docker-compose.yml build

# Start the container in the background
docker compose -f docker/docker-compose.yml up -d
```

The container mounts `planner/`, `vlm/`, `simulation/`, `pddl/`, and `data/` from the host, so changes to those modules take effect immediately without rebuilding.

Fast Downward is compiled inside the container during the build step and linked as `fast-downward` on the container PATH.

### 4 — Download additional Gazebo models (optional)

Required for the **workshop** and **kitchen** simulation worlds:

```bash
bin/download_extra_scenes.sh
```

---

## Running the simulation

All commands below are run from the repository root on the **host** with `.venv` active.

### Start Gazebo + MoveIt 2

```bash
# Default scene (tabletop)
bin/start_sim.sh

# Specific scene
bin/start_sim.sh --world office
bin/start_sim.sh --world workshop
bin/start_sim.sh --world kitchen

# With RViz2
bin/start_sim.sh --world office rviz:=true
```

Available worlds: `tabletop`, `workshop`, `office`, `kitchen`.

### Run a single task (open-loop)

Captures the scene, runs VLM planning, and executes the primitive sequence once.

```bash
bin/run_task.sh "pick the pen and place it on the notebook"
```

### Run closed-loop execution

Re-observes the scene and re-plans after each primitive. Recommended for multi-step tasks.

```bash
bin/run_loop.sh "pick the pen and place it next to the keyboard"
```

Each iteration: scan pose → wrist camera capture → VLM next step → inject → wait for completion signal → repeat.

### Reset the scene

```bash
bin/reset_scene.sh
```

---

## Tests

Unit tests cover the pure-Python pipeline (VLM parser, problem generator, PDDL validation, domain enricher). No GPU or ROS installation required.

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

---

## Evaluation scripts

### VLM inference smoke test (GPU required)

Verifies that the model loads and produces a valid plan on a synthetic image.

```bash
python scripts/test_vlm_inference.py --synthetic
```

### Domain enrichment evaluation

Runs the enrichment pipeline on a set of tasks that require non-standard actions and reports recall, PDDL validity, and per-task breakdowns.

```bash
python scripts/eval_enrichment.py --image data/scene_overview.png --n-runs 3
```

Output is written to `data/eval_runs/<timestamp>_eval/` with an HTML report and per-task PDDL files.

### Phase 2 perception validation

Compares GroundingDINO bounding-box poses against oracle ground truth across object positions.

```bash
python scripts/validate_phase2.py --world office
```

---

## PDDL domain templates

Four templates are available in `pddl/domains/`. The VLM selects the appropriate one at planning time.

| Template | When used | Extensions over base |
|----------|-----------|----------------------|
| `manipulation_base` | Flat-surface pick-and-place | — |
| `manipulation_stacking` | Tasks involving stacking or spatial relationships | `stacked-on`, `clear` predicates, `stack`/`unstack` actions |
| `containers_manipulation` | Container access (drawers, boxes) | `container` type, `open`/`close` predicates and actions |
| `navigation_manipulation` | Mobile manipulation across zones | `zone` type, `navigate-to` action |

For tasks requiring verbs beyond the base primitive set (e.g. `pour`, `cut`, `stir`), the VLM populates the `domain_additions` field in its JSON output; `DomainEnricher` merges these additions into the selected template before planning.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ROS_DOMAIN_ID` | `42` | ROS 2 DDS domain — set in `docker-compose.yml`; must match on host if subscribing to topics directly |
| `VLMRP_REPO_ROOT` | `/workspace` | Repo root inside the container; used by the Orchestrator to import `planner/`, `vlm/`, `simulation/` |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory for model weights |
| `DISPLAY` | inherited from host | Required for Gazebo GUI; set automatically by `docker-compose.yml` |

---

## Architecture overview

The system is split across two environments that communicate through a shared `data/` volume and `docker exec` calls.

### Perception and planning — host (Ubuntu 20.04, GPU)

Two Intel RealSense D435i cameras feed the pipeline:

| Camera | Mount | Used for |
|--------|-------|----------|
| Overview | Fixed stand above table | GroundingDINO object detection + 3D pose via depth; VLM scene context |
| Wrist | End-effector (eye-in-hand) | VLM close-up context only — depth not used for 3D pose |

At each iteration `run_loop_host.py` runs this sequence on the host:

```
Overview image + Wrist image
        │
        ├──► GroundingDINO-tiny          — detects objects, returns 2D bounding boxes
        │    + RealSense depth + K⁻¹     — unprojects boxes to 3D poses in panda_link0
        │              │
        └──► VLMPlanner (Qwen3-VL-8B)   — receives both images + natural-language task
                       │                   outputs a structured JSON action plan
                       ▼
             DomainEnricher              — extends PDDL domain if novel actions needed
                       │
             ProblemGenerator            — binds detected objects to PDDL problem file
                       │
             FastDownward                — validates and orders the plan symbolically
                       │
             Grounded plan (pick A, place A on B, …)
                       │
               docker exec ──────────────────────────────────►
```

### Execution — Docker container (Ubuntu 22.04, ROS 2 Humble)

```
Orchestrator node  (receives grounded plan via stdin)
        │
        ├── pick(obj)       — MoveIt 2 grasp trajectory
        ├── place(obj, loc) — MoveIt 2 place trajectory
        ├── look_at(obj)    — reorients wrist camera toward target
        └── navigate_to(loc)— Nav2 base motion (Phase 3+)
                │
        MoveIt 2 ──► Franka HW  /  Gazebo Classic 11 (simulation)
```

The container writes a completion signal to `data/` after each primitive; `run_loop_host.py` waits for it before capturing the next frame and replanning.

---

## Real robot deployment

The real Franka Panda runs **franka_ros** (ROS 1 Noetic) on the robot PC. The planning stack uses ROS 2 Humble. A dedicated bridge container translates between the two.

### Physical setup (measured values)

| Reference point | Height from floor | Height from `panda_link0` |
|-----------------|-------------------|--------------------------|
| `panda_link0` (robot base) | 47 cm | 0 m (origin) |
| Table surface | 82 cm | +0.35 m |
| Overview camera | 132 cm | +0.85 m |

The overview RealSense D435i is mounted on a fixed stand above the workspace (~50 cm above the table surface). The wrist RealSense D435i is mounted on the end-effector (eye-in-hand).

### Identifying camera serials

Both cameras are the same model (Intel RealSense D435i), so the serial number is the only way to tell them apart. List all connected cameras:

```bash
source .venv/bin/activate
python scripts/capture_and_plan.py --list
```

Example output:
```
[0] serial=242322071571  name=Intel RealSense D435I   ← overview (fixed stand)
[1] serial=241122072695  name=Intel RealSense D435I   ← wrist (on end-effector)
```

The **overview** camera is the one physically mounted on the fixed stand above the table; the **wrist** camera is the one attached to the end-effector. Note the serials — you will need them for the calibration step below and for `capture_and_plan.py`.

### Overview camera calibration (one-time, run on host)

The overview camera must be calibrated once after mounting. The calibration stores the camera pose relative to `panda_link0` in `data/overview_camera_setup.json`; `real_robot.launch.py` reads this file automatically on every subsequent launch.

```bash
source .venv/bin/activate
python scripts/setup_overview_camera.py --serial <OVERVIEW_SERIAL>
```

**Calibration steps:**

1. Click **Refresh Image** to capture a live frame from the overview camera.
2. Adjust the six pose sliders (x, y, z, roll, pitch, yaw) until the cyan grid overlays the physical table surface.  
   Use `z_table` (red slider) to set the table height in `panda_link0` frame — for the measured setup this is **0.35 m**.
3. Type an object name in the **Object:** field and click **Run DINO** to verify that detected 3D positions are physically plausible (X forward, Y left, Z ≈ table height).
4. Click **Save Config** — writes `data/overview_camera_setup.json` and `data/overview_camera_pose.json` (4×4 cam-to-base matrix).

Keyboard shortcuts (active when the image area has focus, disabled while typing in the Object field):

| Key | Action |
|-----|--------|
| `←` / `→` | y − / y + |
| `↑` / `↓` | x + / x − |
| `PgUp` / `PgDn` | z + / z − |
| `W` / `S` | pitch + / − |
| `A` / `D` | yaw + / − |
| `Q` / `E` | roll + / − |
| `R` / `F` | z\_table + / − |

### Network requirements

- Development machine and robot PC must be on the same LAN.
- The robot PC (`ROBOT_IP`) runs: `roscore`, `franka_ros`, and the RealSense driver.
- Both machines must be able to ping each other.

### Step 1 — Build the bridge image (once)

```bash
docker compose --profile real build ros1_bridge
```

This builds `docker/Dockerfile.bridge` (Ubuntu 20.04 + ROS Noetic + ROS 2 Foxy + `ros1_bridge`). It takes several minutes the first time.

### Step 2 — Launch the full real-robot stack

```bash
bin/start_real.sh --robot-ip <ROBOT_PC_IP>
```

This script:
1. Starts the ROS 1 ↔ ROS 2 bridge (`bin/start_bridge.sh`) — bridges `/joint_states`, camera topics, and trajectory goals between the two ROS versions.
2. Launches MoveIt 2 in the `ros2` container with `real_robot.launch.py` (no Gazebo).

### Step 3 — Run a task (same as simulation)

In a separate terminal, activate the venv on the host and run:

```bash
source .venv/bin/activate
bin/run_task.sh --task "pick the pen and place it on the notebook"
```

Or for a closed-loop session:

```bash
bin/run_loop.sh --task "pick the pen and place it on the notebook"
```

The pipeline is identical to simulation: VLM inference runs on the host GPU, the PDDL plan is dispatched to the Orchestrator node inside the container, and MoveIt 2 sends trajectories to the real arm via the bridge.

### Bridge mode options

| Mode | Command | When to use |
|------|---------|-------------|
| `dynamic` (default) | `bin/start_bridge.sh --robot-ip <IP>` | Development — bridges all matching topic types automatically |
| `static` | `bin/start_bridge.sh --robot-ip <IP> --mode static` | Production — bridges only the topics listed in `docker/bridge_topics.yaml` |
| `pairs` | `bin/start_bridge.sh --robot-ip <IP> --mode pairs` | Diagnostic — prints matched/unmatched topic types and exits |

### Environment variables for real robot

| Variable | Example | Description |
|----------|---------|-------------|
| `ROS_MASTER_URI` | `http://192.168.1.100:11311` | Set automatically by `start_bridge.sh` from `--robot-ip` |
| `ROS_IP` | `192.168.1.50` | Set automatically to the local machine's IP; prevents ROS 1 from advertising the wrong address |
