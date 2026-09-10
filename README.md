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

MoveIt trajectories are previewed in RViz for 3 seconds before execution. To
change the preview interval:

```bash
bin/start_sim.sh rviz:=true plan_preview_duration:=6
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

The real Franka FR3 is owned by a **ROS 1 Noetic** robot PC. The planning
stack uses ROS 2 Humble, and a dedicated container translates common topics
between the two systems.

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

#### AprilTag calibration (non-interactive alternative)

An AprilTag fixed to the table can replace the manual slider calibration. Use
the same tag ID, family, and measured black-square edge size in both steps, and
keep the robot stationary while recording the wrist-camera measurement.

Run the scripts in the ROS 2 environment (inside `vlm_ros2` when using the
container):

```bash
# 1. Put the tag in view of /camera/color/image_raw and save its pose in fr3_link0.
python3 scripts/save_table_apriltag_transform.py \
  --tag-id 0 --tag-size 0.080

# 2. Put the same fixed tag in view of the overview camera.
python3 scripts/calibrate_overview_camera_apriltag.py --tag-id 0
```

Step 1 writes `data/table_apriltag_transform.json`. For step 2, AprilTag PnP
uses `/overview_camera/overview_camera/color/camera_info`, matching the color
image containing the detected corners. The script separately reads
`/overview_camera/overview_camera/aligned_depth_to_color/camera_info` and saves
those runtime depth intrinsics in `data/overview_camera_info.json`. It also
writes `data/overview_camera_setup.json` and `data/overview_camera_pose.json`.
By default, the tag origin's Z coordinate is used as `z_table`; pass
`--table-z <metres>` when the tag is not flush with the table surface. Restart
`real_robot.launch.py` after step 2 so it reloads the static overview-camera
transform.

#### ROS 2 RGB-D capture and planning

Once calibrated, use the ROS 2-native capture path instead of opening the two
RealSense devices directly with `pyrealsense2`:

```bash
bin/capture_and_plan_ros2.sh \
  --task "pick the red cup and place it next to the pen" \
  --publish-poses
```

The command captures aligned overview color/depth and camera intrinsics from
the `/overview_camera/overview_camera/...` topics. It optionally captures the
wrist camera from `/camera/...`, generates the plan, runs GroundingDINO for all
objects referenced by that plan, and writes each depth-derived position in
`fr3_link0` to `data/ros2_runs/<run>/detections_with_poses.json`.

If the calibrated overview transform is missing from TF, the capture helper
publishes it from `data/overview_camera_pose.json`, and the host entry point
starts a persistent static publisher in `vlm_ros2`. Use
`--no-publish-overview-tf` to disable that fallback. For capture and detection
without VLM inference:

```bash
bin/capture_and_plan_ros2.sh --no-vlm --objects red_cup pen
```

For closed-loop execution on the real robot, start `real_robot.launch.py` and
use the corresponding loop entry point:

```bash
bin/run_loop_ros2.sh "pick the red cup and place it next to the pen"
```

This uses the same replanning and execution state machine as
`run_loop_host.py`, but captures both cameras through ROS 2, loads the
AprilTag-derived overview calibration, uses aligned depth for object poses, and
does not query or snap positions to Gazebo. The latest GroundingDINO overlay is
published on `/perception/dino_annotated_image`.

When a Trust USB webcam is connected before the real-robot container starts,
the launch file finds its stable `/dev/v4l/by-id` entry and publishes it on
`/experiment_camera/image_raw`. Every real closed-loop run records that topic
to `experiment_webcam.mp4` in its own `data/real_runs/<run>/` directory. The
publisher waits and retries if the camera is temporarily disconnected. Disable
recording with `--no-record-webcam`, or override the topic/FPS with
`--webcam-topic` and `--webcam-fps`. If the USB descriptor does not contain
"Trust", start the stack with
`EXPERIMENT_WEBCAM_DEVICE=/dev/videoN bin/start_real.sh`; alternatively set
`EXPERIMENT_WEBCAM_MATCH` to another
case-insensitive device-name fragment.

The overview ROS camera is also recorded by default for every closed-loop run.
Its video is saved as `overview_camera.mp4` in the same run directory. Use
`--no-record-overview-video` to disable it, or `--overview-video-topic` and
`--overview-video-fps` to override the source topic and output frame rate.

Camera topic overrides can be passed after the task, for example:

```bash
bin/run_loop_ros2.sh "pick the cup" \
  --wrist-depth-topic /camera/aligned_depth_to_color/image_raw \
  --max-steps 8
```

Robot execution is enabled by default. To run one observation/planning cycle,
save its perception/debug outputs, and stop without moving the arm, use:

```bash
bin/run_loop_ros2.sh "pick the cup" --no-execute
```

`--no-execute` also suppresses the pre-scan movement and never injects the plan
into the orchestrator. `--execute` can be passed explicitly when desired.

To require operator approval immediately before every planned action is sent
to the robot, use:

```bash
bin/run_loop_ros2.sh "pick the cup" --confirm-actions
```

Only `y` or `yes` executes the displayed action. Any other response cancels the
loop without injecting it.

### Network requirements

- Development machine and robot PC must be on the same LAN.
- The ROS 1 robot PC is `192.168.131.1` by default.
- The robot PC runs `roscore`, the Franka hardware stack, and a running
  `/effort_joint_trajectory_controller`.
- It publishes `sensor_msgs/JointState` on `/joint_states` and subscribes to
  `trajectory_msgs/JointTrajectory` on
  `/effort_joint_trajectory_controller/command`.
- Both machines must be able to ping each other.
- Their clocks should be synchronized (NTP/chrony), because trajectory header
  timestamps cross the machine boundary.

### Step 1 — Build the bridge image (once)

```bash
docker compose --profile real build ros1_bridge
```

This builds `docker/Dockerfile.bridge` (Ubuntu 20.04 + ROS Noetic + ROS 2
Foxy + `ros1_bridge`). The bridge carries common ROS messages; it does not try
to bridge the incompatible ROS 1 and ROS 2 action transports directly.

### Step 2 — Launch the full real-robot stack

```bash
bin/start_real.sh

# Override either address when needed:
bin/start_real.sh --robot-ip 192.168.131.1 --local-ip 192.168.131.2
```

MoveIt plans are displayed on `/display_planned_path` before execution. The
default preview is 3 seconds; increase it when you want more inspection time:

```bash
bin/start_real.sh --preview-seconds 6
```

This script:

1. Detects this computer's IP on the robot network and exports it as `ROS_IP`.
2. Starts the ROS 1 ↔ ROS 2 topic bridge connected to
   `http://192.168.131.1:11311`.
3. Launches MoveIt 2, `robot_state_publisher`, the Orchestrator, and a local
   trajectory action adapter (no Gazebo and no local hardware driver).

The command path is:

```
MoveIt 2
  -> ROS 2 FollowJointTrajectory action
  -> trajectory_topic_adapter
  -> /effort_joint_trajectory_controller/command
  -> ros1_bridge
  -> ROS 1 effort_joint_trajectory_controller
  -> robot
```

The return path is ROS 1 `/joint_states` → `ros1_bridge` → MoveIt 2 and the
adapter. The adapter filters the shared Husky/FR3 state stream to
`/fr3_joint_states` for MoveIt and reports action success only when every arm
joint reaches the configured goal tolerance.

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

The planning pipeline is identical to simulation. Only trajectory execution is
routed through the ROS 1 computer.

### Validate before moving

In a second terminal, first verify that bridged states use the expected FR3
joint names without commanding motion:

```bash
bin/run_test_move.sh --check-only
```

Then run the interactive 10%-speed movement test:

```bash
bin/run_test_move.sh --velocity 0.1
```

If the ROS 1 stack publishes `panda_joint1` … `panda_joint7` instead of
`fr3_joint1` … `fr3_joint7`, do not move the robot: the MoveIt model and adapter
must first be switched to the Panda naming convention.

### Physical pick/place smoke test

The real stack includes a dedicated gripper adapter because ROS 1 and ROS 2
action transports are incompatible. It targets the ROS 1 action
`/franka_gripper/gripper_action` and adds the two synthetic finger states MoveIt
needs to complete the FR3 state.

With an object already positioned at the current gripper pose, validate the
interfaces without motion and then run the guarded sequence:

```bash
bin/run_test_pick_place.sh --check-only
bin/run_test_pick_place.sh --lift 0.10 --place-y 0.10
```

The sequence opens and closes the gripper at its current pose, lifts vertically,
moves to the requested XY offset, descends to the original height, releases the
object, and retreats. It defaults to 10% velocity and asks for confirmation
before grasping.

### Test every primitive on the real robot

First validate all ROS 1/ROS 2 interfaces without sending commands:

```bash
bin/run_test_primitives.sh --check-only
```

The interactive full test covers `navigate_to`, gripper open/close, `look_at`,
`tilt`, `pour`, `stir`, `cut`, `pick`, and `place`. It asks for confirmation
before every physical step and uses the startup end-effector pose as a virtual
free-space work point:

```bash
bin/run_test_primitives.sh
```

Run one primitive, or a selected subset, while debugging:

```bash
bin/run_test_primitives.sh --only gripper
bin/run_test_primitives.sh --only pick --only place --place-y 0.10
```

Keep the workspace clear during the free-space motion tests. Insert a
lightweight object only when prompted for the `pick` test.

### Environment variables for real robot

| Variable | Example | Description |
|----------|---------|-------------|
| `ROS_MASTER_URI` | `http://192.168.131.1:11311` | Set in the bridge container from `--robot-ip` |
| `ROS_IP` | e.g. `192.168.131.2` | Auto-detected or set with `--local-ip`; lets the ROS 1 PC call back into the bridge |
| `ROS_DOMAIN_ID` | `42` | Shared DDS domain used by the bridge and ROS 2 container |
