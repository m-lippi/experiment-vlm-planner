"""
Oracle world state: provides ground-truth object poses from the simulator (Gazebo).

Role in the pipeline:
  - NOT used as VLM input (the VLM reasons from images directly).
  - Used at execution time only: when a primitive needs a 3D pose
    (e.g. MoveIt pick target), the oracle resolves the symbolic object name
    to an actual pose in the robot base frame (panda_link0).

GazeboOracle queries the Gazebo Classic service /gazebo/get_entity_state.
GazeboAttach simulates gripper attachment by continuously setting the object
pose to track the EEF via TF + /gazebo/set_entity_state.

Both services are exposed by the libgazebo_ros_state.so plugin (loaded in
tabletop.world with <namespace>/gazebo</namespace>).

WorldState is the pure-Python data container used by the Orchestrator.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


# ── Lightweight pose types (no ROS2 dependency) ───────────────────────────────

@dataclass
class Position:
    x: float
    y: float
    z: float


@dataclass
class Orientation:
    x: float
    y: float
    z: float
    w: float


@dataclass
class Pose:
    position:    Position
    orientation: Orientation

    def as_geometry_msg(self):
        """Convert to geometry_msgs.msg.Pose (lazy import — only in ROS2 context)."""
        from geometry_msgs.msg import Pose as RosPose, Point, Quaternion
        p              = RosPose()
        p.position     = Point(x=self.position.x, y=self.position.y, z=self.position.z)
        p.orientation  = Quaternion(
            x=self.orientation.x, y=self.orientation.y,
            z=self.orientation.z, w=self.orientation.w,
        )
        return p


# ── World state container ─────────────────────────────────────────────────────

@dataclass
class ObjectState:
    name:     str
    pose:     Pose
    location: str = ""   # symbolic label (e.g. "table_a") — filled by planner


@dataclass
class WorldState:
    objects:       list[ObjectState] = field(default_factory=list)
    gripper_empty: bool              = True

    def get_pose(self, object_name: str) -> Pose | None:
        """Resolve a symbolic object name to its current 3D pose."""
        for obj in self.objects:
            if obj.name == object_name:
                return obj.pose
        return None

    def to_pddl_init(self) -> list[str]:
        """PDDL :init facts derived from world state (for plan validation)."""
        facts = []
        for obj in self.objects:
            if obj.location:
                facts.append(f"(on {obj.name} {obj.location})")
        if self.gripper_empty:
            facts.append("(gripper-empty)")
        return facts


# ── Gazebo oracle ─────────────────────────────────────────────────────────────

class GazeboOracle:
    """
    Queries Gazebo Classic for ground-truth object poses via ROS 2 service.

    Requires the libgazebo_ros_state.so plugin in the world file with
    <namespace>/gazebo</namespace>, which exposes:
      /gazebo/get_entity_state  (gazebo_msgs/srv/GetEntityState)

    The reference_frame should match the robot base link so that MoveIt2
    can use the pose directly without an extra TF lookup.
    """

    SERVICE = "/gazebo/get_entity_state"

    def __init__(self, node, reference_frame: str = "panda_link0") -> None:
        from gazebo_msgs.srv import GetEntityState

        self._node   = node
        self._frame  = reference_frame
        self._client = node.create_client(GetEntityState, self.SERVICE)

        node.get_logger().info(f"GazeboOracle: waiting for {self.SERVICE} ...")
        if not self._client.wait_for_service(timeout_sec=60.0):
            node.get_logger().warn(
                f"GazeboOracle: {self.SERVICE} not available. "
                "Is the gazebo_ros_state plugin loaded in the world file?"
            )
        else:
            node.get_logger().info("GazeboOracle: service ready.")

    def get_pose(self, object_name: str) -> Pose | None:
        """
        Query Gazebo for the current pose of a named model.

        Uses threading.Event + callbacks so it is safe to call from any
        background thread with a MultiThreadedExecutor already spinning.

        Args:
            object_name: Gazebo model name (e.g. "red_cup", "shelf_b").

        Returns:
            Pose in the robot base frame, or None if the model is not found.
        """
        from gazebo_msgs.srv import GetEntityState

        req                  = GetEntityState.Request()
        req.name             = object_name
        req.reference_frame  = self._frame

        done          = threading.Event()
        result_holder: list = [None]

        def _on_response(future):
            result_holder[0] = future.result()
            done.set()

        self._client.call_async(req).add_done_callback(_on_response)
        done.wait(timeout=5.0)

        resp = result_holder[0]
        if resp is None:
            self._node.get_logger().warn(
                f"GazeboOracle: service call timed out for '{object_name}'"
            )
            return None

        if not resp.success:
            # status_message absent in some gazebo_msgs versions — use getattr
            msg = getattr(resp, "status_message", "not found")
            self._node.get_logger().warn(
                f"GazeboOracle: '{object_name}' not found in Gazebo. {msg}"
            )
            return None

        p = resp.state.pose.position
        q = resp.state.pose.orientation
        return Pose(
            position    = Position(x=p.x, y=p.y, z=p.z),
            orientation = Orientation(x=q.x, y=q.y, z=q.z, w=q.w),
        )

    def get_world_state(self, tracked_objects: list[str]) -> WorldState:
        """
        Query all tracked objects and return a WorldState snapshot.

        Args:
            tracked_objects: List of Gazebo model names to query.

        Returns:
            WorldState with current poses. Objects not found are skipped.
        """
        object_states = []
        for name in tracked_objects:
            pose = self.get_pose(name)
            if pose is not None:
                object_states.append(ObjectState(name=name, pose=pose))
            else:
                self._node.get_logger().warn(
                    f"GazeboOracle: skipping '{name}' — pose unavailable."
                )
        return WorldState(objects=object_states, gripper_empty=True)


# ── Physics-based attachment (Boeing plugin) ──────────────────────────────────

class BoeingAttach:
    """
    Physics-based gripper attachment via Boeing's gazebo_model_attachment_plugin.

    Creates a Gazebo fixed joint between panda_hand and the object at the
    moment of attachment. Gazebo's ODE solver then moves the object exactly
    with the arm — no 100 Hz teleport, no offset arithmetic, no orientation
    locking needed. Generalises to any arm motion including OMPL paths.

    Prerequisites (already in place):
      - Plugin loaded in every world file:
            <plugin name="model_attachment"
                    filename="libboeing_gazebo_model_attachment_plugin_lib.so"/>
      - boeing_gazebo_model_attachment_plugin compiled in /opt/boeing_ws/

    Interface:
        attach(object_name) — call AFTER close_gripper(); joint is created at
            the current relative transform → object stays at finger pads.
        detach()            — removes the joint; physics resumes on the object.

    Real-robot note: this class is simulation-only. On the real Franka the
    gripper holds the object mechanically — no equivalent is needed.
    """

    ATTACH_SRV = "/gazebo/attach"
    DETACH_SRV = "/gazebo/detach"

    _ROBOT_MODEL = "panda"    # Gazebo entity name (from -entity "panda" in spawn)
    # Gazebo Classic merges fixed joints into compound bodies.
    # panda_hand and panda_link8 are connected via fixed joints to panda_link7,
    # so Gazebo folds them all into the "panda_link7" physics body.
    _EEF_LINK    = "panda_link7"
    _JOINT_NAME  = "grasp_joint"
    _OBJECT_LINK = "link"     # standard link name for all manipulable objects

    def __init__(self, node) -> None:
        from boeing_gazebo_model_attachment_plugin_msgs.srv import Attach, Detach
        from gazebo_msgs.srv import GetEntityState, SetEntityState
        from std_srvs.srv import Empty

        self._node     = node
        self._attached: str | None = None

        self._attach_client   = node.create_client(Attach, self.ATTACH_SRV)
        self._detach_client   = node.create_client(Detach, self.DETACH_SRV)
        self._get_state_client = node.create_client(GetEntityState, "/gazebo/get_entity_state")
        self._set_state_client = node.create_client(SetEntityState, "/gazebo/set_entity_state")
        self._pause_client     = node.create_client(Empty, "/gazebo/pause_physics")
        self._unpause_client   = node.create_client(Empty, "/gazebo/unpause_physics")

        # Gazebo world plugins register their services only after the physics
        # simulation is fully initialised — typically 50-80s after the
        # orchestrator node starts (see launch sequence logs). Use 120s timeout
        # to cover slow machines without blocking indefinitely.
        node.get_logger().info("BoeingAttach: waiting for services (up to 120 s) ...")
        if not self._attach_client.wait_for_service(timeout_sec=120.0):
            node.get_logger().warn(
                f"BoeingAttach: {self.ATTACH_SRV} not available — "
                "is the Boeing plugin loaded in the world file?"
            )
        else:
            node.get_logger().info("BoeingAttach: services ready.")

    def attach(
        self,
        object_name: str,
        **kwargs,               # absorb grasp_offset_z etc. — not needed here
    ) -> None:
        """
        Create a Gazebo fixed joint between panda_hand and object_name.

        Must be called AFTER close_gripper() so that the relative transform
        captured by the joint matches the actual grasp pose.

        Extra kwargs from GazeboAttach callers (grasp_offset_z,
        grasp_offset_forward_m, lock_orientation) are silently ignored —
        the physics engine handles everything.
        """
        import time as _t
        from boeing_gazebo_model_attachment_plugin_msgs.srv import Attach
        from std_srvs.srv import Empty

        # Lift object off the table before creating the joint.
        # ODE cannot stably solve a fixed joint constraint + table contact
        # normal force simultaneously: the arm gets launched by the reaction
        # force. The teleport is imperceptible and happens only once.
        self._pre_lift(object_name, dz=0.020)

        # Pause Gazebo physics before calling Boeing attach.
        # The plugin's attachCallback immediately acquires GetPhysicsUpdateMutex().
        # While any controller (gripper, arm) is actively executing a trajectory,
        # the physics thread holds that mutex during each step → the Boeing
        # callback blocks forever waiting to acquire it → 5s timeout.
        # Pausing physics releases the mutex so attachCallback can proceed.
        _pause_done = threading.Event()
        def _pause_cb(f): _pause_done.set()
        self._pause_client.call_async(Empty.Request()).add_done_callback(_pause_cb)
        _pause_done.wait(timeout=2.0)
        _t.sleep(0.05)   # one physics tick to let the pause take effect

        req              = Attach.Request()
        req.joint_name   = self._JOINT_NAME
        req.model_name_1 = self._ROBOT_MODEL
        req.link_name_1  = self._EEF_LINK
        req.model_name_2 = object_name
        req.link_name_2  = self._OBJECT_LINK

        done   = threading.Event()
        result = [None]

        def _cb(future):
            result[0] = future.result()
            done.set()

        self._attach_client.call_async(req).add_done_callback(_cb)
        done.wait(timeout=10.0)

        # Always unpause physics (physics was paused above to unblock the mutex).
        _unpause_done = threading.Event()
        def _unpause_cb(f): _unpause_done.set()
        self._unpause_client.call_async(Empty.Request()).add_done_callback(_unpause_cb)
        _unpause_done.wait(timeout=2.0)

        resp = result[0]
        if resp and resp.success:
            self._attached = object_name
            self._node.get_logger().info(
                f"BoeingAttach: '{object_name}' attached via Gazebo joint (panda_link7)"
            )
            # Wait for ODE to integrate the new constraint before arm motion.
            _t.sleep(0.6)
        else:
            msg = getattr(resp, "message", "timeout")
            self._node.get_logger().warn(
                f"BoeingAttach: attach '{object_name}' failed — {msg}"
            )

    def _pre_lift(self, object_name: str, dz: float = 0.010) -> None:
        """Teleport object dz metres up to break table contact before Boeing joint creation."""
        import time
        from gazebo_msgs.srv import GetEntityState, SetEntityState
        from gazebo_msgs.msg import EntityState

        done = threading.Event()
        result = [None]
        def _cb(f): result[0] = f.result(); done.set()

        get_req = GetEntityState.Request()
        get_req.name = object_name
        get_req.reference_frame = "world"
        self._get_state_client.call_async(get_req).add_done_callback(_cb)
        done.wait(timeout=5.0)

        resp = result[0]
        if resp is None or not resp.success:
            self._node.get_logger().warn(
                f"BoeingAttach._pre_lift: could not get state of '{object_name}' — skipping"
            )
            return

        state = EntityState()
        state.name = object_name
        state.reference_frame = "world"
        state.pose = resp.state.pose
        state.pose.position.z += dz

        done2 = threading.Event()
        def _cb2(f): done2.set()
        set_req = SetEntityState.Request()
        set_req.state = state
        self._set_state_client.call_async(set_req).add_done_callback(_cb2)
        done2.wait(timeout=5.0)

        time.sleep(0.05)   # one physics tick for contact state to update

    def detach(self) -> str | None:
        """Remove the Gazebo fixed joint; physics resumes on the object."""
        from boeing_gazebo_model_attachment_plugin_msgs.srv import Detach

        obj = self._attached
        if obj is None:
            return None

        req              = Detach.Request()
        req.joint_name   = self._JOINT_NAME
        req.model_name_1 = self._ROBOT_MODEL
        req.model_name_2 = obj

        done   = threading.Event()
        result = [None]

        def _cb(future):
            result[0] = future.result()
            done.set()

        self._detach_client.call_async(req).add_done_callback(_cb)
        done.wait(timeout=5.0)

        self._attached = None
        resp = result[0]
        if resp and resp.success:
            self._node.get_logger().info(
                f"BoeingAttach: '{obj}' detached — physics resumes."
            )
        else:
            msg = getattr(resp, "message", "timeout")
            self._node.get_logger().warn(
                f"BoeingAttach: detach '{obj}' — {msg}"
            )
        return obj

    def lock_orientation(self, lock: bool = True) -> None:
        """No-op: Gazebo physics handles orientation correctly without locking."""
        pass

    def set_world_pose(self, *args, **kwargs) -> None:
        """Not applicable: Boeing plugin uses physics joints, not teleport."""
        self._node.get_logger().warn(
            "BoeingAttach.set_world_pose(): not applicable with physics attachment"
        )


# ── Teleport-based attachment (legacy fallback) ───────────────────────────────

class GazeboAttach:
    """
    Simulates gripper attachment by teleporting the object to track the EEF.

    Calls /gazebo/set_entity_state at 100 Hz (vs the old 20 Hz).
    At 100 Hz, the object falls only ~1.2 mm between teleport calls
    (0.5 * 9.81 * 0.01²) — imperceptible vs the ~12 mm visible bounce
    at 20 Hz.  Velocity is also zeroed each tick to prevent accumulation.

    Sim-to-real note: this class is simulation-only.
    """

    SET_SERVICE = "/gazebo/set_entity_state"

    # Offset from world origin to panda_link0 (must match launch file).
    _ROBOT_BASE_WORLD_X = 0.20
    _ROBOT_BASE_WORLD_Y = 0.00
    _ROBOT_BASE_WORLD_Z = 0.77

    def __init__(
        self,
        node,
        eef_frame: str = "panda_hand",
    ) -> None:
        from gazebo_msgs.srv import SetEntityState
        import tf2_ros

        self._node      = node
        self._eef_frame = eef_frame

        self._client      = node.create_client(SetEntityState, self.SET_SERVICE)
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, node)

        self._timer:                object | None  = None
        self._attached:             str | None     = None
        self._grasp_offset_z:       float          = 0.0
        self._grasp_offset_forward: float          = 0.0  # along EEF Z (panda_hand +Z)
        self._grasp_eef_quat:       tuple | None   = None  # (x,y,z,w) at attach time
        self._orientation_locked:   bool           = True  # True → object stays upright

        node.get_logger().info(
            f"GazeboAttach: waiting for {self.SET_SERVICE} ..."
        )
        if not self._client.wait_for_service(timeout_sec=60.0):
            node.get_logger().warn(
                f"GazeboAttach: {self.SET_SERVICE} not available — "
                "simulated attachment disabled."
            )
        else:
            node.get_logger().info("GazeboAttach: service ready.")

    def attach(
        self,
        object_name: str,
        grasp_offset_z: float = 0.13,
        grasp_offset_forward_m: float = 0.0,
    ) -> None:
        """Start teleporting object_name to follow the EEF at 100 Hz.

        Args:
            grasp_offset_z:        vertical (world Z) distance from panda_hand origin
                                   to the object centre, subtracted each tick.
            grasp_offset_forward_m: distance from panda_hand origin to the object
                                   centre along panda_hand +Z (EEF approach direction).
                                   For side grasp = _FINGER_REACH_M; for top_down = 0.
                                   The offset is rotated into world frame each tick, so
                                   it follows the gripper orientation during pour tilt.
        """
        self._attached             = object_name
        self._grasp_offset_z       = grasp_offset_z
        self._grasp_offset_forward = grasp_offset_forward_m
        self._grasp_eef_quat       = None   # captured on first _update tick
        self._orientation_locked   = True   # locked by default; unlock for intentional tilts
        if self._timer is not None:
            self._timer.cancel()
        self._timer = self._node.create_timer(0.01, self._update)   # 100 Hz
        self._node.get_logger().info(
            f"GazeboAttach: '{object_name}' tracking '{self._eef_frame}' "
            f"at 100 Hz (offset z={grasp_offset_z:.3f} m)"
        )

    def detach(self) -> str | None:
        """Stop teleporting. Object stays at its last pose."""
        obj = self._attached
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._attached            = None
        self._grasp_eef_quat      = None
        self._orientation_locked  = True   # reset to safe default
        if obj:
            self._node.get_logger().info(
                f"GazeboAttach: '{obj}' detached — physics resumes."
            )
        return obj

    def lock_orientation(self, lock: bool = True) -> None:
        """Lock or unlock the attached object's world-frame orientation.

        Locked (True, default after attach):
            Object stays upright regardless of EEF rotation.
            Safe for all transit/carry motions — OMPL can swing j1 freely
            without the object spinning and hitting nearby objects.

        Unlocked (False):
            Object orientation follows EEF delta since grasp time.
            Use only for intentional tilts (e.g. pour motion) where the
            arm is already at the stable target position.
        """
        self._orientation_locked = lock

    def set_world_pose(
        self, object_name: str, x: float, y: float, z: float
    ) -> None:
        """Teleport object to a specific world-frame position (upright)."""
        self._call_set_state(object_name, x, y, z)

    # ── Quaternion helpers (no external deps) ─────────────────────────────────

    @staticmethod
    def _quat_conj(q: tuple) -> tuple:
        x, y, z, w = q
        return (-x, -y, -z, w)

    @staticmethod
    def _quat_mul(q1: tuple, q2: tuple) -> tuple:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        )

    @staticmethod
    def _rotate_vec(v: tuple, q: tuple) -> tuple:
        """Rotate vector v = (x,y,z) by unit quaternion q via v' = q * v_pure * q⁻¹."""
        vq = (v[0], v[1], v[2], 0.0)
        q_inv = GazeboAttach._quat_conj(q)
        r = GazeboAttach._quat_mul(
            GazeboAttach._quat_mul(q, vq), q_inv
        )
        return (r[0], r[1], r[2])

    def _update(self) -> None:
        if self._attached is None:
            return
        try:
            from rclpy.time import Time
            t = self._tf_buffer.lookup_transform(
                "panda_link0", self._eef_frame, Time()
            )
            hand_x = t.transform.translation.x
            hand_y = t.transform.translation.y
            hand_z = t.transform.translation.z
            r      = t.transform.rotation
            eef_q  = (r.x, r.y, r.z, r.w)

            # Capture EEF orientation at first tick after attach.
            # panda_link0 shares its axes with world (robot base is axis-aligned),
            # so panda_link0-frame orientation = world-frame orientation.
            if self._grasp_eef_quat is None:
                self._grasp_eef_quat = eef_q

            # Forward offset: locked to the GRASP EEF orientation (not current).
            #
            # Using the current eef_q here causes severe visual detachment during
            # OMPL transit paths: OMPL explores intermediate joint configurations
            # where EEF Z points in arbitrary directions, so _rotate_vec would
            # send the offset (and thus the can) flying in the wrong direction.
            # For example, if OMPL briefly puts EEF Z toward world +Z, the can
            # appears 10 cm ABOVE the hand instead of in front of it.
            #
            # Fix: always rotate the offset by _grasp_eef_quat (= _SIDE_QUAT at
            # attach time). For the side grasp this gives (dx=0.10, dy=0, dz=0)
            # — a fixed world-X offset that does not wobble during any arm motion.
            # Pour tilt: the can ORIENTATION still follows the gripper via obj_q;
            # only the position offset direction is locked to the grasp frame.
            if self._grasp_offset_forward != 0.0 and self._grasp_eef_quat is not None:
                dx, dy, dz_fwd = self._rotate_vec(
                    (0.0, 0.0, self._grasp_offset_forward), self._grasp_eef_quat
                )
            else:
                dx, dy, dz_fwd = 0.0, 0.0, 0.0

            world_x = self._ROBOT_BASE_WORLD_X + hand_x + dx
            world_y = self._ROBOT_BASE_WORLD_Y + hand_y + dy
            world_z = self._ROBOT_BASE_WORLD_Z + hand_z + dz_fwd - self._grasp_offset_z

            # Object orientation.
            # Locked (default): identity — object stays upright in world frame.
            #   Prevents the object from spinning when OMPL chooses j1=154° paths
            #   during transit. The gripper may swing, but the can stays upright
            #   and does not hit nearby objects.
            # Unlocked: R_obj = R_eef(t) × R_eef_grasp⁻¹ — object follows EEF.
            #   Used only when PourPrimitive explicitly calls lock_orientation(False)
            #   for the intentional pour tilt (arm is already stable at the tray).
            if self._orientation_locked:
                obj_q = (0.0, 0.0, 0.0, 1.0)
            else:
                obj_q = self._quat_mul(eef_q, self._quat_conj(self._grasp_eef_quat))

            self._call_set_state(self._attached, world_x, world_y, world_z, obj_q)
        except Exception:
            pass

    def _call_set_state(
        self, name: str, x: float, y: float, z: float,
        orientation: tuple = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        from gazebo_msgs.srv import SetEntityState
        from gazebo_msgs.msg import EntityState
        from geometry_msgs.msg import Twist

        ox, oy, oz, ow = orientation

        state                    = EntityState()
        state.name               = name
        state.reference_frame    = "world"
        state.pose.position.x    = x
        state.pose.position.y    = y
        state.pose.position.z    = z
        state.pose.orientation.x = ox
        state.pose.orientation.y = oy
        state.pose.orientation.z = oz
        state.pose.orientation.w = ow
        # Zero velocity each tick — prevents velocity accumulation between
        # teleport calls that would cause the object to drift.
        state.twist              = Twist()

        req       = SetEntityState.Request()
        req.state = state
        self._client.call_async(req)
