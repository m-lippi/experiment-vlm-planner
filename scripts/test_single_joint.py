#!/usr/bin/env python3
"""
test_single_joint.py — Test base: muove un singolo giunto (default: joint7) di un
piccolo offset e verifica che /joint_states rispecchi il movimento.

Valida bidirezionalità ros2_control:
  • Lettura stato:  /joint_states  (franka_hardware → joint_state_broadcaster)
  • Scrittura cmd:  fr3_arm_controller/follow_joint_trajectory (action)

Uso nel container (con start_real.sh già attivo):
  python3 /workspace/scripts/test_single_joint.py
  python3 /workspace/scripts/test_single_joint.py --joint 4 --delta -5
  python3 /workspace/scripts/test_single_joint.py --no-return
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

JOINT_NAMES = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

_SEP = "-" * 60


class SingleJointTestNode(Node):

    def __init__(self) -> None:
        super().__init__("single_joint_test")
        self._js_lock = threading.Lock()
        self._current: list[float] | None = None

        self.create_subscription(
            JointState, "/joint_states", self._on_js, _SENSOR_QOS
        )
        self._action = ActionClient(
            self, FollowJointTrajectory,
            "/fr3_arm_controller/follow_joint_trajectory"
        )

    def _on_js(self, msg: JointState) -> None:
        n2p = dict(zip(msg.name, msg.position))
        pos = [n2p.get(n) for n in JOINT_NAMES]
        if None not in pos:
            with self._js_lock:
                self._current = [float(p) for p in pos]

    def read_joints(self, timeout: float = 5.0) -> list[float] | None:
        """Attende che almeno un messaggio /joint_states valido sia arrivato."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._js_lock:
                if self._current is not None:
                    return list(self._current)
            time.sleep(0.05)
        return None

    def move_joint(
        self,
        positions_all: list[float],
        duration_sec: float = 3.0,
        timeout: float = 15.0,
    ) -> bool:
        """
        Invia FollowJointTrajectory goal e attende il risultato.
        Usa solo callbacks — nessun spin_until_future_complete per evitare
        conflitti con l'executor che gira in background.
        """
        if not self._action.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server non disponibile.")
            return False

        point = JointTrajectoryPoint()
        point.positions = positions_all
        point.velocities = [0.0] * 7
        sec = int(duration_sec)
        nsec = int((duration_sec - sec) * 1_000_000_000)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.points = [point]

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = traj

        result_event = threading.Event()
        success_holder = [False]

        def _on_result(future):
            try:
                res = future.result()
                success_holder[0] = (
                    res.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL
                )
            except Exception as exc:
                self.get_logger().error(f"Errore nel risultato: {exc}")
            result_event.set()

        def _on_goal_response(future):
            try:
                handle = future.result()
                if not handle or not handle.accepted:
                    self.get_logger().error("Goal rifiutato dal controller.")
                    result_event.set()
                    return
                handle.get_result_async().add_done_callback(_on_result)
            except Exception as exc:
                self.get_logger().error(f"Errore nella risposta al goal: {exc}")
                result_event.set()

        self._action.send_goal_async(goal_msg).add_done_callback(_on_goal_response)

        if not result_event.wait(timeout=timeout):
            self.get_logger().error(f"Timeout ({timeout}s) attesa risultato movimento.")
            return False
        return success_holder[0]


def _fmt(positions: list[float]) -> str:
    return "\n".join(
        f"  {n}: {p:+.4f} rad ({math.degrees(p):+.2f}°)"
        for n, p in zip(JOINT_NAMES, positions)
    )


def _confirm(msg: str) -> bool:
    print(f"\n  >>> {msg}  (Ctrl+C = annulla) <<<")
    try:
        input()
        return True
    except KeyboardInterrupt:
        print("\nAnnullato.")
        return False


def run(args: argparse.Namespace) -> int:
    rclpy.init()
    node = SingleJointTestNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    idx = args.joint - 1
    delta_rad = math.radians(args.delta)

    # ── STEP 1: leggi stato iniziale ──────────────────────────────────────────
    print(f"\n{_SEP}")
    print("STEP 1 — Lettura /joint_states (timeout 5s)…")
    q0 = node.read_joints(timeout=5.0)
    if q0 is None:
        print("[FAIL] Nessun joint state ricevuto.")
        return 1
    print(f"[OK]  Stato iniziale:\n{_fmt(q0)}")

    # ── STEP 2: calcola target e chiedi conferma ───────────────────────────────
    target = list(q0)
    target[idx] += delta_rad
    jname = JOINT_NAMES[idx]

    print(f"\n{_SEP}")
    print(f"STEP 2 — Target: {jname}  "
          f"{math.degrees(q0[idx]):+.2f}° → {math.degrees(target[idx]):+.2f}°  "
          f"(Δ = {args.delta:+.1f}°)")
    print(f"         Durata: {args.duration:.1f}s")

    if not _confirm("Premi INVIO per muovere"):
        return 0

    # ── STEP 3: esegui movimento ───────────────────────────────────────────────
    print(f"[→]  Invio traiettoria…")
    t0 = time.time()
    ok = node.move_joint(target, duration_sec=args.duration)
    elapsed = time.time() - t0

    if not ok:
        print(f"[FAIL] Movimento fallito (elapsed {elapsed:.1f}s).")
        return 1
    print(f"[OK]  Movimento completato in {elapsed:.1f}s.")

    # ── STEP 4: verifica bidirezionale ────────────────────────────────────────
    time.sleep(0.5)
    q1 = node.read_joints(timeout=3.0)
    print(f"\n{_SEP}")
    print("STEP 3 — Verifica /joint_states dopo il movimento:")
    if q1:
        err_deg = abs(math.degrees(q1[idx] - target[idx]))
        print(f"  {jname}: atteso {math.degrees(target[idx]):+.2f}°, "
              f"letto {math.degrees(q1[idx]):+.2f}°  "
              f"(errore {err_deg:.3f}°)")
        if err_deg < 1.0:
            print("[OK]  Bidirezionalità confermata: stato rispecchia il comando.")
        else:
            print(f"[WARN] Errore {err_deg:.2f}° > 1° — verificare.")

    # ── STEP 5: ritorno ───────────────────────────────────────────────────────
    if not args.no_return:
        print(f"\n{_SEP}")
        print(f"STEP 4 — Ritorno: {jname} → {math.degrees(q0[idx]):+.2f}°")
        if _confirm("Premi INVIO per tornare"):
            ok2 = node.move_joint(q0, duration_sec=args.duration)
            print("[OK]  Ritorno completato." if ok2 else "[WARN] Ritorno fallito.")

    print(f"\n{_SEP}\nTEST COMPLETATO.\n")
    rclpy.shutdown()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Test singolo giunto su robot reale FR3")
    p.add_argument("--joint",    type=int,   default=7,   metavar="N",
                   help="Numero giunto 1-7 (default: 7 = polso)")
    p.add_argument("--delta",    type=float, default=5.0, metavar="DEG",
                   help="Spostamento in gradi (default: +5°)")
    p.add_argument("--duration", type=float, default=3.0, metavar="SEC",
                   help="Durata del movimento in secondi (default: 3s)")
    p.add_argument("--no-return", action="store_true",
                   help="Non tornare alla posizione iniziale")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
