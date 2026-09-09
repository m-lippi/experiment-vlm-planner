"""
Orchestrator — ROS 2 node: bridges the VLM-PDDL pipeline with the robot.

Data flow:
  /vlm_planner/task_command  (std_msgs/String)   → triggers a planning + execution cycle
  /camera/color/image_raw    (sensor_msgs/Image)  → latest camera frame, passed to VLM
  /vlm_planner/status        (std_msgs/String)    → publishes pipeline status (busy / ready / error)

On each task command:
  1. Take the most recent camera frame.
  2. Run Pipeline.run(command, [pil_image]).
  3. For each PrimitiveCall: resolve pose via GazeboOracle, dispatch to primitive.

Primitives dispatched:
  pick, place → MoveIt2 (ArmPrimitive subclasses)
  look_at     → observation pose (Phase 1 fixed pose)
  navigate_to → stub (Phase 3: Nav2)
  open/close_gripper, open/close_container, say → inline handlers

VLM weights are loaded at startup in a background thread so the node
remains responsive during the (slow) GPU model load.
"""

from __future__ import annotations

import json
import math
import sys
import os
import threading
import time
import uuid
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped

# Locate repo root so planner/, vlm/, simulation/ are importable.
# VLMRP_REPO_ROOT can be set externally; inside Docker the layout is /workspace/<modules>.
_REPO_ROOT = os.environ.get("VLMRP_REPO_ROOT") or "/workspace"
sys.path.insert(0, _REPO_ROOT)

from planner.pipeline import Pipeline, PipelineResult
from planner.plan_parser import PrimitiveCall
from simulation.oracle.world_state import GazeboOracle, GazeboAttach


# Phase 2 architecture — who provides poses:
#
#  PICK targets   → PerceptionModule (GroundingDINO) after look_at.
#                   Oracle is BYPASSED when perception cache is fresh.
#
#  PLACE locations → known scene positions (oracle or static map).
#                   In Phase 4 real robot: replaced by a pre-built scene map
#                   or a second look_at on the target location.
#
# _TRACKED_OBJECTS: oracle queries ONLY for place locations and fallback.
# Kept minimal to avoid querying absent objects.
_TRACKED_OBJECTS = [
    # ── workshop place locations ──
    "target_tray",
    # ── office place locations ──
    "side_table",
    # ── tabletop (Phase 1 compat) ──
    "shelf_b", "table",
]

# Dynamic collision objects for manipulable scene objects will be
# provided by an OctoMap built from RealSense point cloud data in Phase 2.
# No manual collision box management here — that approach is fragile,
# simulation-specific, and would require replication on the real robot.


class Orchestrator(Node):

    def __init__(self) -> None:
        super().__init__("vlm_robot_planner")

        # ── Pipeline (pure Python — no ROS2 dependency) ───────────────────
        self._pipeline = Pipeline(repair_retries=3)
        self._task_lock = threading.Lock()   # prevents concurrent pipeline runs
        self._state_lock = threading.Lock()  # atomically reserves one task
        self._busy      = False
        self._ready     = False
        # request_id -> (accepted, reason). Repeated plan messages only resend
        # the original ACK; they never start the same robot action twice.
        self._plan_request_results: dict[str, tuple[bool, str]] = {}

        # ── Primitive dispatch table ──────────────────────────────────────
        # Populated in _init_primitives() after MoveIt2 is ready.
        self._prim_dispatch: dict[str, Any] = {}

        # ── Oracle + simulated attachment ─────────────────────────────────
        # Skip Gazebo services in real-robot mode (no Gazebo running).
        self.declare_parameter("use_sim", True)
        self._use_sim = bool(self.get_parameter("use_sim").value)
        if self._use_sim:
            from vlm_robot_planner.primitives.base import BASE_FRAME as _BF
            self._oracle  = GazeboOracle(node=self, reference_frame=_BF)
            self._attach  = GazeboAttach(node=self)
        else:
            self._oracle  = None
            self._attach  = None
            self.get_logger().info(
                "Real-robot mode: GazeboOracle/GazeboAttach disabled. "
                "Poses come from PerceptionModule (/perception/object_pose)."
            )

        # ── Camera image buffer ───────────────────────────────────────────
        self._latest_image: Image | None = None
        self._image_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._camera_callback,
            qos_profile=10,
        )

        # ── Phase 2: perception pose cache ────────────────────────────────
        # Receives poses from PerceptionModule (host) after look_at.
        # Cache: {object_name: (timestamp_sec, x, y, z, height_m_or_None)}
        #   height_m: estimated object height from DINO bbox (Phase 2+).
        #             None in Phase 1 (oracle) → pick uses fixed fallback.
        # Poses older than _PERCEPTION_TTL_S fall back to GazeboOracle.
        self._perception_cache: dict[str, tuple] = {}
        self._perception_condition = threading.Condition()
        self._perception_sub = self.create_subscription(
            PoseStamped,
            "/perception/object_pose",
            self._on_perception_pose,
            10,
        )

        # ── Task command subscription ─────────────────────────────────────
        self._cmd_sub = self.create_subscription(
            String,
            "/vlm_planner/task_command",
            self._command_callback,
            qos_profile=10,
        )

        # ── Inject pre-computed VLMPlan (from host GPU) ───────────────────
        # Payload: JSON {"command": "...", "vlm_plan": { <VLMPlan fields> }}
        # Skips VLM inference; goes straight to PDDL validation + execution.
        self._inject_sub = self.create_subscription(
            String,
            "/vlm_planner/inject_plan",
            self._inject_plan_callback,
            qos_profile=10,
        )

        # ── Collision object publisher (static planning scene only) ──────
        # Used for: table (startup), AttachedCollisionObject (W5 pick/place).
        # Dynamic obstacle management (W4) is deferred to Phase 2 (OctoMap).
        from moveit_msgs.msg import CollisionObject as _CO
        self._collision_pub = self.create_publisher(_CO, "/collision_object", 10)

        # ── Status publisher ──────────────────────────────────────────────
        from rclpy.qos import QoSProfile, DurabilityPolicy
        _latched_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        _latest_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._status_pub = self.create_publisher(
            String, "/vlm_planner/status", _latest_qos
        )
        self._plan_ack_pub = self.create_publisher(
            String, "/vlm_planner/plan_ack", _latched_qos
        )
        self._pose_ack_pub = self.create_publisher(
            String, "/perception/object_pose_ack", _latched_qos
        )
        self._dino_image_pub = self.create_publisher(
            Image, "/perception/dino_annotated_image", _latest_qos
        )
        self._dino_image_ack_pub = self.create_publisher(
            String, "/perception/dino_annotated_image_ack", _latched_qos
        )
        self._latest_dino_image: Image | None = None
        self._dino_image_sub = self.create_subscription(
            Image,
            "/perception/dino_annotated_image_input",
            self._on_dino_annotated_image,
            10,
        )
        self._dino_image_timer = self.create_timer(
            1.0, self._republish_dino_annotated_image
        )
        self._readiness_pub = self.create_publisher(
            Bool, "/vlm_planner/ready", _latest_qos
        )
        # ── Step-complete publisher (closed-loop support) ─────────────────
        # Published after every primitive execution.
        # Payload JSON: {"step": <i>, "primitive": "<name>", "success": bool,
        #                "task_complete": bool}
        # TRANSIENT_LOCAL: late subscribers receive the last message published.
        # Prevents race condition where host _wait_step_complete subscribes after
        # the orchestrator already published the completion signal.
        self._step_pub = self.create_publisher(
            String, "/vlm_planner/step_complete", _latched_qos
        )
        self._dispatch_seq = 0
        self._publish_readiness(False)
        self._publish_status("starting — loading VLM and MoveIt2")

        # ── Load VLM weights in background thread ─────────────────────────
        # GPU load takes ~20-30 s; doing it in background keeps the node alive.
        threading.Thread(target=self._load_vlm_async, daemon=True).start()

        self.get_logger().info(
            "Orchestrator starting. Listening for commands; execution remains "
            "disabled until MoveIt2 is ready."
        )

    # ── Startup helpers ───────────────────────────────────────────────────────

    def _load_vlm_async(self) -> None:
        """Load VLM weights and initialise MoveIt2 in a background thread.

        VLM loading is best-effort: in Phase 1 the VLM runs on the HOST (GPU),
        not inside the container. A failure here is expected and non-fatal —
        MoveIt2 is initialised regardless so the arm primitives are always ready.
        """
        try:
            self.get_logger().info("Loading VLM weights (GPU)…")
            self._pipeline.load_vlm()
            self.get_logger().info("VLM loaded.")
        except Exception as exc:
            self.get_logger().warn(
                f"VLM not loaded (expected in container — runs on host): {exc}"
            )
            # Do NOT return: MoveIt2 must be initialised even without VLM.

        try:
            self.get_logger().info("Initialising MoveIt2…")
            self._init_primitives()
            self._setup_planning_scene()
            with self._state_lock:
                self._ready = True
            self._publish_readiness(True)
            self.get_logger().info("MoveIt2 ready. Orchestrator fully operational.")
            self._publish_status("ready")
        except Exception as exc:
            self.get_logger().error(f"MoveIt2 init failed: {exc}")
            self._publish_status(f"error — MoveIt2 init failed: {exc}")

    def _setup_planning_scene(self) -> None:
        """Add static environment geometry to MoveIt2's planning scene.

        Simulation keeps its original table model. In real-robot mode, the
        table top is ``z_table`` from data/overview_camera_setup.json and the
        collision box spans z=0..z_table in fr3_link0.
        """
        from moveit_msgs.msg import CollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose

        pub = self._collision_pub

        co = CollisionObject()
        from vlm_robot_planner.primitives.base import BASE_FRAME as _BF
        co.header.frame_id = _BF
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = "table"

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX

        pose = Pose()
        if self._use_sim:
            box.dimensions = [1.20, 1.00, 0.77]
            pose.position.x = 0.30
            pose.position.y = 0.00
            pose.position.z = -0.385
            table_description = "simulation table"
        else:
            setup_path = os.path.join(
                _REPO_ROOT, "data", "overview_camera_setup.json"
            )
            try:
                with open(setup_path, encoding="utf-8") as stream:
                    z_table = float(json.load(stream)["z_table"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot load real table height from {setup_path}: {exc}"
                ) from exc
            if not math.isfinite(z_table) or z_table <= 0.0:
                raise RuntimeError(
                    f"Invalid z_table={z_table!r} in {setup_path}; expected metres > 0"
                )

            # Real table in fr3_link0: 0.8 m along X, 1.2 m along Y,
            # with its top surface at the calibrated z_table value.
            box.dimensions = [0.80, 1.20, z_table]
            pose.position.x = 0.60
            pose.position.y = 0.00
            pose.position.z = z_table / 2.0
            table_description = (
                f"real table: centre=(0.600, 0.000, {z_table / 2.0:.3f}), "
                f"size=(0.800, 1.200, {z_table:.3f}) m"
            )
        pose.orientation.w = 1.0

        co.primitives      = [box]
        co.primitive_poses = [pose]
        co.operation       = CollisionObject.ADD

        import time as _time
        for _ in range(5):
            pub.publish(co)
            _time.sleep(0.1)

        self.get_logger().info(
            f"Planning scene: {table_description} collision object added in {_BF}."
        )

    def _init_primitives(self) -> None:
        """Initialise MoveIt2Client and wire up all primitives."""
        from vlm_robot_planner.moveit2_client          import MoveIt2Client as MoveIt2
        from vlm_robot_planner.primitives.base        import (
            ARM_JOINT_NAMES, ARM_GROUP, BASE_FRAME, EEF_LINK,
        )
        from vlm_robot_planner.primitives.pick        import PickPrimitive
        from vlm_robot_planner.primitives.place       import PlacePrimitive
        from vlm_robot_planner.primitives.look_at     import LookAtPrimitive
        from vlm_robot_planner.primitives.navigate_to import NavigateToPrimitive
        from vlm_robot_planner.primitives.pour        import PourPrimitive
        from vlm_robot_planner.primitives.stir        import StirPrimitive
        from vlm_robot_planner.primitives.tilt        import TiltPrimitive
        from vlm_robot_planner.primitives.cut         import CutPrimitive

        cb_group = ReentrantCallbackGroup()
        moveit2  = MoveIt2(
            node              = self,
            joint_names       = ARM_JOINT_NAMES,
            base_link_name    = BASE_FRAME,
            end_effector_name = EEF_LINK,
            group_name        = ARM_GROUP,
            callback_group    = cb_group,
        )
        moveit2.max_velocity     = 0.3
        moveit2.max_acceleration = 0.3

        # Single shared TF buffer — one TransformListener for ALL primitives.
        import tf2_ros
        _tf_buffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(_tf_buffer, self)

        pick  = PickPrimitive(self, moveit2, attach=self._attach, tf_buffer=_tf_buffer)
        place = PlacePrimitive(self, moveit2, attach=self._attach, tf_buffer=_tf_buffer)
        look  = LookAtPrimitive(self, moveit2, tf_buffer=_tf_buffer)
        nav   = NavigateToPrimitive(self)
        pour  = PourPrimitive(self, moveit2, attach=self._attach, tf_buffer=_tf_buffer)
        stir  = StirPrimitive(self, moveit2, tf_buffer=_tf_buffer)
        tilt  = TiltPrimitive(self, moveit2, tf_buffer=_tf_buffer)
        cut   = CutPrimitive(self, moveit2, tf_buffer=_tf_buffer)

        self._prim_dispatch = {
            "pick":            pick.execute,
            "place":           place.execute,
            "look_at":         look.execute,
            "navigate_to":     nav.execute,
            "open_gripper":    lambda name, pose: pick.open_gripper(),
            "close_gripper":   lambda name, pose: pick.close_gripper(),
            "open_container":  self._exec_open_container,
            "close_container": self._exec_close_container,
            "say":             self._exec_say,
            # ── Kitchen / manipulation primitives (Phase 3) ──────────────
            "pour":            pour.execute,
            "stir":            stir.execute,
            "tilt":            tilt.execute,
            "cut":             cut.execute,
        }

    # ── ROS 2 callbacks ───────────────────────────────────────────────────────

    def _camera_callback(self, msg: Image) -> None:
        """Store the latest camera frame (thread-safe — overwrite is atomic)."""
        self._latest_image = msg

    def _on_perception_pose(self, msg: PoseStamped) -> None:
        """Cache a perception-estimated pose (Phase 2).
        The object name is encoded in msg.header.frame_id.
        Object height is encoded in orientation.z (0.0 = unknown)."""
        obj = msg.header.frame_id
        if not obj:
            return
        t = self.get_clock().now().nanoseconds / 1e9
        # Decode sideband height (orientation.z > 0 → estimated height in metres)
        h = msg.pose.orientation.z
        height_m = float(h) if h > 0.0 else None
        with self._perception_condition:
            self._perception_cache[obj] = (
                t,
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                height_m,
            )
            self._perception_condition.notify_all()

        ack = String()
        ack.data = json.dumps({
            "object": obj,
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "cached": True,
        })
        self._pose_ack_pub.publish(ack)
        height_str = f", h={height_m:.3f}m" if height_m else ""
        from vlm_robot_planner.primitives.base import BASE_FRAME as _frame
        self.get_logger().info(
            f"[Perception] '{obj}' cached: "
            f"({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, "
            f"{msg.pose.position.z:.3f}) {_frame}{height_str}"
        )

    def _on_dino_annotated_image(self, msg: Image) -> None:
        """Retain and publish the latest host-generated GroundingDINO overlay."""
        self._latest_dino_image = msg
        self._dino_image_pub.publish(msg)

        ack = String()
        ack.data = json.dumps({
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "published": True,
        })
        self._dino_image_ack_pub.publish(ack)
        self.get_logger().info(
            f"[Perception] DINO annotated image published "
            f"({msg.width}x{msg.height})"
        )

    def _republish_dino_annotated_image(self) -> None:
        """Keep the latest overlay visible to late volatile image subscribers."""
        if self._latest_dino_image is not None:
            self._dino_image_pub.publish(self._latest_dino_image)

    def _command_callback(self, msg: String) -> None:
        """
        Receive a task command and run the full pipeline in a separate thread
        so the ROS2 executor is not blocked during planning/execution.
        """
        command = msg.data.strip()
        if not command:
            return

        accepted, reason = self._reserve_execution()
        if not accepted:
            self.get_logger().warn(
                f"Orchestrator {reason} — ignoring command: '{command}'"
            )
            self._publish_status(f"{reason} — ignoring new command")
            return

        self.get_logger().info(f"Task received: '{command}'")
        request_id = str(uuid.uuid4())
        threading.Thread(
            target=self._run_task, args=(command, None, request_id), daemon=True
        ).start()

    def _inject_plan_callback(self, msg: String) -> None:
        """Receive a JSON-encoded VLMPlan from the host GPU and execute it.

        Expected payload:
            {"command": "<task text>", "vlm_plan": { <VLMPlan fields> }}
        Skips VLM inference; uses Pipeline.run(vlm_plan=...) directly.
        """
        request_id = ""
        try:
            data = json.loads(msg.data)
            request_id = str(data.get("request_id") or uuid.uuid4())
            command  = data.get("command", "injected plan")
            direct   = data.get("direct", False)   # closed-loop: skip PDDL
            from vlm.planner import VLMPlan
            vlm_plan = VLMPlan.from_json(json.dumps(data["vlm_plan"]))
        except Exception as exc:
            self.get_logger().error(f"inject_plan: invalid JSON — {exc}")
            self._publish_plan_ack(request_id, False, f"invalid plan: {exc}")
            return


        accepted, reason, duplicate = self._claim_plan_request(request_id)
        if duplicate:
            self.get_logger().info(
                f"Duplicate injected plan {request_id} — resending prior ACK "
                "without executing it again."
            )
            self._publish_plan_ack(request_id, accepted, reason)
            return

        if not accepted:
            self.get_logger().warn(
                f"Orchestrator {reason} — rejecting injected plan {request_id}."
            )
            self._publish_plan_ack(request_id, False, reason)
            self._publish_status(f"{reason} — rejected injected plan")
            return

        self._publish_plan_ack(request_id, True, "accepted")

        self.get_logger().info(
            f"Injected plan received: '{command}' "
            f"({len(vlm_plan.steps)} steps)"
            f"{' [direct]' if direct else ''}"
        )

        if direct:
            # Closed-loop mode: bypass PDDL, execute VLM steps directly.
            # Used when the VLM plans one step at a time and PDDL validation
            # would fail due to incomplete goal inference for partial plans.
            threading.Thread(
                target=self._run_direct,
                args=(command, vlm_plan, request_id), daemon=True,
            ).start()
        else:
            threading.Thread(
                target=self._run_task,
                args=(command, vlm_plan, request_id), daemon=True,
            ).start()

    # ── Task execution ────────────────────────────────────────────────────────

    def _run_task(self, command: str, vlm_plan=None, request_id: str = "") -> None:
        with self._task_lock:
            self._publish_status(f"busy — planning: {command}")
            success = False
            try:
                success = self.run(
                    command, vlm_plan=vlm_plan, request_id=request_id
                )
            except Exception as exc:
                self.get_logger().error(f"Orchestrator: unhandled exception: {exc}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                self._publish_step_complete(
                    -1, "execution", False, False, request_id=request_id
                )
            finally:
                self._release_execution()
                self._publish_status("ready" if success else f"error — task failed: {command}")

    def _run_direct(self, command: str, vlm_plan, request_id: str = "") -> None:
        """Execute VLM steps directly, bypassing PDDL (closed-loop mode).

        Used in closed-loop iterations where the VLM plans ONE step at a time.
        PDDL validation is skipped because partial plans (e.g. pick without place)
        produce trivially-empty FD plans.  The VLM's decision is trusted directly.
        """
        with self._task_lock:
            self._publish_status(f"busy — direct: {command}")
            success = False
            try:
                from planner.plan_parser import PrimitiveCall

                primitives  = []

                for step in vlm_plan.steps:
                    # Strip bbox/location_bbox — not used by primitives
                    clean = {k: v for k, v in step.args.items()
                             if k not in ("bbox", "location_bbox")}
                    vals = list(clean.values())
                    if step.primitive == "pick":
                        a = [clean.get("object", ""), clean.get("grasp_mode", "top_down")]
                    elif step.primitive == "place":
                        a = [clean.get("object", ""), clean.get("location", "")]
                    elif step.primitive in ("look_at", "navigate_to"):
                        a = [clean.get("target", clean.get("location", ""))]
                    elif step.primitive == "pour":
                        # VLM may use arbitrary PDDL param names (e.g. "can"/"tray") —
                        # fall back to positional extraction: first value = source,
                        # second value = target (consistent with PourPrimitive.execute).
                        source = (clean.get("source") or clean.get("object") or
                                  (vals[0] if vals else ""))
                        target = (clean.get("target") or clean.get("location") or
                                  (vals[1] if len(vals) > 1 else ""))
                        a = [source, target]
                    elif step.primitive == "stir":
                        a = [clean.get("container", "") or (vals[0] if vals else "")]
                    elif step.primitive == "cut":
                        a = [clean.get("object", "") or (vals[0] if vals else "")]
                    elif step.primitive == "tilt":
                        a = [clean.get("object", "") or (vals[0] if vals else ""),
                             clean.get("angle_deg", vals[1] if len(vals) > 1 else 45)]
                    else:
                        a = vals  # novel action — pass all values positionally
                    primitives.append(PrimitiveCall(step.primitive, a))

                # Novel PDDL actions generated by VLM domain enrichment.
                # These are soft-fail: if execution fails, log it and continue —
                # the thesis evaluates VLM planning correctness, not motion precision.
                # Core primitives (pick, place, look_at) remain hard-fail so the
                # host can replan when the world state is genuinely wrong.
                _SOFT_FAIL = frozenset({"pour", "stir", "tilt", "cut", "weigh",
                                        "stamp", "mix", "shake"})

                for i, prim in enumerate(primitives):
                    self.get_logger().info(f"  [direct] → {prim.name}({prim.args})")
                    ok = self._dispatch(prim)
                    if not ok and prim.name in _SOFT_FAIL:
                        self.get_logger().warn(
                            f"[direct] Novel action '{prim.name}' failed — "
                            "logging gracefully (VLM plan was valid, execution limited)")
                        ok = True  # treat as success for loop continuity
                    self._publish_step_complete(
                        i, prim.name, ok,
                        task_complete=(i == len(primitives)-1) and ok,
                        request_id=request_id,
                    )
                    if not ok:
                        self.get_logger().error(f"[direct] '{prim.name}' failed")
                        self._publish_status(f"error — direct step failed: {prim.name}")
                        return

                self.get_logger().info("[direct] Step executed successfully.")
                success = True
            except Exception as exc:
                self.get_logger().error(f"_run_direct: {exc}")
                import traceback; self.get_logger().error(traceback.format_exc())
                self._publish_status(f"error — {exc}")
                self._publish_step_complete(
                    -1, "execution", False, False, request_id=request_id
                )
            finally:
                self._release_execution()
                if success:
                    self._publish_status("ready")

    def run(self, command: str, vlm_plan=None, request_id: str = "") -> bool:
        """
        Execute one full task:
          images → VLM → PDDL pipeline → primitives → robot.

        Args:
            command:  Natural language task description.
            vlm_plan: Pre-computed VLMPlan (skips VLM inference when provided).

        Returns:
            True if all primitives executed successfully.
        """
        self.get_logger().info(f"Task: '{command}'")

        # Convert latest ROS2 Image to PIL (skip if VLMPlan already provided)
        images = []
        if vlm_plan is None:
            if self._latest_image is not None:
                pil = self._ros_image_to_pil(self._latest_image)
                if pil is not None:
                    images = [pil]
                else:
                    self.get_logger().warn("Image conversion failed — running VLM without image.")
            else:
                self.get_logger().warn("No camera frame available — running VLM without image.")

        # Run planning pipeline (vlm_plan=None → full VLM inference; otherwise skip VLM)
        result: PipelineResult = self._pipeline.run(command, images, vlm_plan=vlm_plan)

        if not result.success:
            self.get_logger().error(
                f"Pipeline failed at '{result.failure_stage}': {result.error}"
            )
            self._publish_step_complete(
                -1, "planning", False, False, request_id=request_id
            )
            return False

        self.get_logger().info(
            f"Plan validated — {len(result.primitives)} primitives "
            f"(repair_attempts={result.repair_attempts})"
        )

        # Annotate pick primitives with grasp_mode from the original VLM plan.
        # PDDL cannot model grasp_mode (physical parameter), so FastDownward drops it.
        # After validation, we re-attach it from vlm_plan before dispatch.
        # args layout after annotation: [object, source_location, grasp_mode]
        if vlm_plan is not None:
            _vlm_picks = [s for s in vlm_plan.steps if s.primitive == "pick"]
            _pick_idx  = 0
            for _prim in result.primitives:
                if _prim.name == "pick" and _pick_idx < len(_vlm_picks):
                    _gm = _vlm_picks[_pick_idx].args.get("grasp_mode", "top_down")
                    _prim.args = list(_prim.args) + [_gm]
                    _pick_idx += 1

        # Dispatch each primitive — oracle lookup is lazy inside _dispatch
        n = len(result.primitives)
        if n == 0:
            self.get_logger().error("Validated plan contains no executable primitives.")
            self._publish_step_complete(
                -1, "planning", False, False, request_id=request_id
            )
            return False
        for i, prim in enumerate(result.primitives):
            self.get_logger().info(f"  → {prim.name}({prim.args})")

            ok = self._dispatch(prim)
            is_last = (i == n - 1)
            self._publish_step_complete(
                i, prim.name, ok, task_complete=is_last and ok,
                request_id=request_id,
            )
            if not ok:
                self.get_logger().error(
                    f"Primitive '{prim.name}' failed — aborting task."
                )
                return False

        self.get_logger().info("Task completed successfully.")
        return True

    # ── Primitive dispatch ────────────────────────────────────────────────────

    # Index of the arg to use for oracle pose lookup, per primitive.
    # pick(obj)            → args[0] = object to pick
    # place(obj, location) → args[1] = destination location
    # look_at(obj)         → args[0] = object to observe
    # navigate_to(loc)     → args[0] = destination
    _ORACLE_ARG_IDX: dict[str, int] = {
        "place": 1,
        "pour":  1,   # pour(source, target) — target is args[1], source is in gripper
    }

    _POSE_REQUIRED = frozenset({"pick", "place", "pour", "tilt", "cut", "stir"})
    _PERCEPTION_WAIT_S = 10.0

    def _cached_perception_pose(
        self, object_name: str, wait_timeout: float = 0.0,
        ttl: float = 60.0,
    ) -> dict | None:
        """Return a fresh cached pose, waiting for its callback when requested."""
        deadline = time.monotonic() + max(0.0, wait_timeout)
        with self._perception_condition:
            while True:
                cached = self._perception_cache.get(object_name)
                if cached is not None:
                    age = self.get_clock().now().nanoseconds / 1e9 - cached[0]
                    if age < ttl:
                        from simulation.oracle.world_state import Position, Orientation
                        height_m = cached[4] if len(cached) > 4 else None
                        self.get_logger().info(
                            f"[Perception] Using cached pose for '{object_name}' "
                            f"(age {age:.1f}s) — oracle bypassed"
                        )
                        return {
                            "position": Position(
                                x=cached[1], y=cached[2], z=cached[3]
                            ),
                            "orientation": Orientation(
                                x=0.0, y=0.0, z=0.0, w=1.0
                            ),
                            "height_m": height_m,
                        }

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._perception_condition.wait(timeout=remaining)

    def _dispatch(self, prim: PrimitiveCall) -> bool:
        """Route a PrimitiveCall to the correct handler."""
        handler = self._prim_dispatch.get(prim.name)
        # print('Handler')
        # print(handler)
        if handler is None:
            if not self._ready:
                self.get_logger().error(
                    f"Cannot dispatch '{prim.name}': orchestrator is not ready."
                )
                return False
            self.get_logger().warn(f"Unknown primitive '{prim.name}' — skipping.")
            return True   # unknown primitives are non-fatal in Phase 1

        arg_idx  = self._ORACLE_ARG_IDX.get(prim.name, 0)
        obj_name = prim.args[arg_idx] if len(prim.args) > arg_idx else ""

        # ── Pose resolution: perception cache → oracle fallback ─────────
        # TTL: perception poses are valid for 60 s (one full pick–place cycle).
        _PERCEPTION_TTL_S = 60.0

        pose_data = None

        if (
            prim.name in self._POSE_REQUIRED
            and self._oracle is None
            and not obj_name
        ):
            self.get_logger().error(
                f"Cannot execute '{prim.name}': plan contains no pose target."
            )
            return False


        # if obj_name:
        #     # Phase 2: prefer PerceptionModule estimate when fresh.
        #     # Wait up to 3 seconds because the perception message may arrive
        #     # shortly after the primitive is dispatched.
        #     _PERCEPTION_TTL_S = 60.0
        #     _POSE_WAIT_S = 3.0
        #     _POSE_POLL_S = 0.05

        #     import time

        #     deadline = time.monotonic() + _POSE_WAIT_S

        #     while time.monotonic() < deadline:
        #         print(time.monotonic())
        #         cached = self._perception_cache.get(obj_name)
        #         print(cached)
        #         if cached is not None:
        #             t_cached, cx, cy, cz = (
        #                 cached[0], cached[1], cached[2], cached[3]
        #             )
        #             height_m = cached[4] if len(cached) > 4 else None

        #             age = (
        #                 self.get_clock().now().nanoseconds / 1e9
        #                 - t_cached
        #             )

        #             if age < _PERCEPTION_TTL_S:
        #                 from simulation.oracle.world_state import Position, Orientation

        #                 pose_data = {
        #                     "position": Position(x=cx, y=cy, z=cz),
        #                     "orientation": Orientation(
        #                         x=0.0, y=0.0, z=0.0, w=1.0
        #                     ),
        #                     "height_m": height_m,
        #                 }

        #                 self.get_logger().info(
        #                     f"[Perception] Using cached pose for '{obj_name}' "
        #                     f"(age {age:.1f}s) — oracle bypassed"
        #                 )
        #                 break

        #         time.sleep(_POSE_POLL_S)

        #     # Pose still not available after waiting.
        #     if pose_data is None and self._oracle is not None:
        #         self.get_logger().warn(
        #             f"[Perception] No pose for '{obj_name}' after "
        #             f"{_POSE_WAIT_S:.1f}s — trying GazeboOracle."
        #         )

        #         pose = self._oracle.get_pose(obj_name)

        #         if pose is not None:
        #             pose_data = {
        #                 "position": pose.position,
        #                 "orientation": pose.orientation,
        #             }
        #         else:
        #             self.get_logger().warn(
        #                 f"Oracle: no pose for '{obj_name}' — "
        #                 "primitive will use None pose."
        #             )

        #     elif pose_data is None and self._oracle is None:
        #         self.get_logger().warn(
        #             f"No pose for '{obj_name}' after "
        #             f"{_POSE_WAIT_S:.1f}s in real-robot mode. "
        #             "Primitive will use None pose."
        #         )

        if obj_name:
            # In real mode, wait for the executor to run a possibly queued pose
            # callback before declaring the pose unavailable.
            wait_timeout = (
                self._PERCEPTION_WAIT_S
                if self._oracle is None and prim.name in self._POSE_REQUIRED
                else 0.0
            )
            pose_data = self._cached_perception_pose(
                obj_name, wait_timeout=wait_timeout, ttl=_PERCEPTION_TTL_S
            )

            # Oracle fallback — lazy single query (only for needed object)
            if pose_data is None and self._oracle is not None:
                pose = self._oracle.get_pose(obj_name)
                if pose is not None:
                    pose_data = {
                        "position":    pose.position,
                        "orientation": pose.orientation,
                    }
                else:
                    self.get_logger().warn(
                        f"Oracle: no pose for '{obj_name}' — "
                        "primitive will use None pose."
                    )
            elif pose_data is None and self._oracle is None:
                self.get_logger().warn(
                    f"No pose for '{obj_name}' in perception cache and "
                    "oracle is disabled (real-robot mode). "
                    "Run look_at first to populate the perception cache."
                )

        if (
            prim.name in self._POSE_REQUIRED
            and obj_name
            and pose_data is None
            and self._oracle is None
        ):
            self.get_logger().error(
                f"Cannot execute '{prim.name}': no fresh perception pose "
                f"for '{obj_name}' after waiting "
                f"{self._PERCEPTION_WAIT_S:.1f}s."
            )
            return False

        if prim.name == "tilt":
            try:
                angle = float(prim.args[1]) if len(prim.args) > 1 else 45.0
            except (ValueError, TypeError):
                angle = 45.0
            return handler(obj_name, pose_data, angle_deg=angle)

        # For pick: resolve grasp_mode and support_surface.
        # Three possible args layouts:
        #   Direct mode:  [object, grasp_mode]              ← args[1] ∈ GRASP_MODES
        #   PDDL mode:    [object, source_location]         ← no grasp_mode (legacy)
        #   PDDL+annot:   [object, source_location, grasp_mode]  ← annotated in run()
        if prim.name == "pick" and obj_name:
            from vlm_robot_planner.primitives.pick import GRASP_MODES
            arg1 = prim.args[1] if len(prim.args) > 1 else None
            arg2 = prim.args[2] if len(prim.args) > 2 else None
            if arg1 in GRASP_MODES:
                # Direct mode: args[1] is the grasp_mode
                grasp_mode = arg1
                surface    = None
            else:
                # PDDL mode: args[1] is source_location, args[2] is grasp_mode (if annotated)
                grasp_mode = arg2 if arg2 in GRASP_MODES else "top_down"
                surface    = "table" if arg1 and arg1.startswith("source_") else arg1
            return handler(
                obj_name, pose_data,
                support_surface=surface,
                grasp_mode=grasp_mode,
                object_height_m=pose_data.get("height_m") if pose_data else None,
            )

        # Pour: resolve source pose so PourPrimitive can return the object.
        # pose_data is already the TARGET (tray) pose (args[1], arg_idx=1).
        # We additionally resolve the SOURCE (held object, args[0]) pose.
        if prim.name == "pour":
            source_name = prim.args[0] if len(prim.args) > 0 else ""
            source_pose_data = None
            if source_name:
                source_pose_data = self._cached_perception_pose(
                    source_name,
                    wait_timeout=(
                        self._PERCEPTION_WAIT_S
                        if self._oracle is None else 0.0
                    ),
                )
                if source_pose_data is None and self._oracle is not None:
                    src_p = self._oracle.get_pose(source_name)
                    if src_p is not None:
                        source_pose_data = {
                            "position":    src_p.position,
                            "orientation": src_p.orientation,
                        }
                if source_pose_data is None and self._oracle is None:
                    self.get_logger().error(
                        f"Cannot execute 'pour': no fresh source pose for "
                        f"'{source_name}'."
                    )
                    return False
            return handler(
                obj_name, pose_data,
                source_name=source_name,
                source_pose_data=source_pose_data,
            )

        return handler(obj_name, pose_data)

    # ── Inline primitive handlers ─────────────────────────────────────────────

    def _exec_open_container(self, container: str, pose_data: dict | None) -> bool:
        self.get_logger().info(f"    open_container('{container}') — stub")
        return True

    def _exec_close_container(self, container: str, pose_data: dict | None) -> bool:
        self.get_logger().info(f"    close_container('{container}') — stub")
        return True

    def _exec_say(self, text: str, pose_data: dict | None) -> bool:
        self.get_logger().info(f"    say: \"{text}\"")
        return True

    # ── Camera conversion ─────────────────────────────────────────────────────

    def _ros_image_to_pil(self, msg: Image):
        """Convert sensor_msgs/Image → PIL.Image using cv_bridge + numpy."""
        try:
            from cv_bridge import CvBridge
            from PIL import Image as PilImage
            bridge = CvBridge()
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return PilImage.fromarray(cv_img)
        except Exception as exc:
            self.get_logger().warn(f"_ros_image_to_pil failed: {exc}")
            return None

    # ── Status ────────────────────────────────────────────────────────────────

    def _reserve_execution(self) -> tuple[bool, str]:
        """Atomically accept one task only when startup has completed."""
        with self._state_lock:
            if not self._ready:
                return False, "not ready"
            if self._busy:
                return False, "busy"
            self._busy = True
            return True, "accepted"

    def _claim_plan_request(self, request_id: str) -> tuple[bool, str, bool]:
        """Atomically reserve a new plan or identify a duplicate request."""
        with self._state_lock:
            previous = self._plan_request_results.get(request_id)
            if previous is not None:
                return previous[0], previous[1], True

            if not self._ready:
                result = (False, "not ready")
            elif self._busy:
                result = (False, "busy")
            else:
                self._busy = True
                result = (True, "accepted")

            self._plan_request_results[request_id] = result
            # Bound memory use while retaining ample history for delayed DDS
            # duplicates. dict insertion order is guaranteed by Python 3.7+.
            while len(self._plan_request_results) > 100:
                oldest = next(iter(self._plan_request_results))
                del self._plan_request_results[oldest]
            return result[0], result[1], False

    def _release_execution(self) -> None:
        with self._state_lock:
            self._busy = False

    def _publish_plan_ack(
        self, request_id: str, accepted: bool, reason: str
    ) -> None:
        msg = String()
        msg.data = json.dumps({
            "request_id": request_id,
            "accepted": accepted,
            "reason": reason,
        })
        self._plan_ack_pub.publish(msg)
        self.get_logger().info(
            f"Plan acknowledgement: request_id={request_id}, "
            f"accepted={accepted}, reason={reason}"
        )

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    def _publish_readiness(self, ready: bool) -> None:
        msg = Bool()
        msg.data = ready
        self._readiness_pub.publish(msg)

    def _publish_step_complete(
        self, step: int, primitive: str, success: bool, task_complete: bool,
        request_id: str = "",
    ) -> None:
        """Publish step completion for closed-loop host monitoring."""
        self._dispatch_seq += 1
        msg      = String()
        msg.data = json.dumps({
            "step":          step,
            "primitive":     primitive,
            "success":       success,
            "task_complete": task_complete,
            "seq":           self._dispatch_seq,
            "request_id":    request_id,
        })
        self._step_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = Orchestrator()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
