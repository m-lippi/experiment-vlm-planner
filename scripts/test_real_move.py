#!/usr/bin/env python3
"""
test_real_move.py — Valida la catena MoveIt 2 → bridge → FR3.

Sequenza di test:
  1. Connettività: verifica che /joint_states arrivi con nomi fr3_joint*
  2. Movimento: PTP alla "ready pose" al 10% velocita', poi ritorno
     (interattivo — chiede conferma prima di ogni mossa)

Esecuzione nel container (con move_group gia' avviato da start_real.sh):
  docker compose run --rm ros2 python3 /workspace/scripts/test_real_move.py

Opzioni:
  --check-only    solo check connettivita', nessun movimento
  --velocity N    scaling velocita' 0.0-1.0 (default: 0.1 = 10%)
  --no-confirm    salta le richieste "premi Invio" (uso automatizzato)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading

# VLM_ROBOT deve essere impostato PRIMA che base.py venga importato
# (legge os.environ al momento dell'import, non a runtime).
os.environ.setdefault("VLM_ROBOT", "fr3")

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

# franka_hardware / joint_state_broadcaster pubblica con BEST_EFFORT.
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

_REPO_ROOT = os.environ.get("VLMRP_REPO_ROOT", "/workspace")
sys.path.insert(0, _REPO_ROOT)

try:
    from vlm_robot_planner.moveit2_client import MoveIt2Client
    from vlm_robot_planner.primitives.base import (
        ARM_JOINT_NAMES, ARM_GROUP, BASE_FRAME, EEF_LINK,
    )
except ImportError as e:
    sys.exit(
        "[ERROR] Import fallito — esegui nel container con il workspace sourced.\n"
        f"Dettaglio: {e}"
    )

# ── FR3 "ready pose" (rad) — configurazione di sicurezza standard Franka ─────
# j1=0°  j2=-45°  j3=0°  j4=-135°  j5=0°  j6=+90°  j7=+45°
_READY_POSE = [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398]

_SEP = "=" * 62


class _RealMoveTestNode(Node):

    def __init__(self, velocity: float) -> None:
        super().__init__("real_move_test")
        cb = ReentrantCallbackGroup()

        self._client = MoveIt2Client(
            node=self,
            joint_names=ARM_JOINT_NAMES,
            base_link_name=BASE_FRAME,
            end_effector_name=EEF_LINK,
            group_name=ARM_GROUP,
            callback_group=cb,
        )
        self._client.max_velocity = velocity
        self._client.max_acceleration = velocity

        self._joint_event = threading.Event()
        self._current_joints: list[float] | None = None

        # /joint_states is published by joint_state_broadcaster (ros2_control).
        # Only FR3 joints are included — no filter node needed.
        self.create_subscription(
            JointState, "/joint_states", self._on_js, _SENSOR_QOS, callback_group=cb
        )
        self.get_logger().info(
            f"[test] VLM_ROBOT={os.environ.get('VLM_ROBOT')}  "
            f"group={ARM_GROUP}  base={BASE_FRAME}  eef={EEF_LINK}"
        )

    # ── joint states ─────────────────────────────────────────────────────────

    def _on_js(self, msg: JointState) -> None:
        if self._joint_event.is_set():
            return
        name_to_pos = dict(zip(msg.name, msg.position))
        positions = [name_to_pos.get(n) for n in ARM_JOINT_NAMES]
        if None not in positions:
            self._current_joints = [float(p) for p in positions]
            self._joint_event.set()

    def wait_for_joints(self, timeout: float = 10.0) -> list[float] | None:
        self._joint_event.wait(timeout=timeout)
        return self._current_joints

    # ── motion helpers ────────────────────────────────────────────────────────

    def move_to(self, joints: list[float], label: str, timeout: float = 30.0) -> bool:
        self.get_logger().info(f"[test] Invio PTP → {label} …")
        self._client.move_to_configuration(joints)
        ok = self._client.wait_until_executed(timeout=timeout)
        if ok:
            self.get_logger().info(f"[test] OK — raggiunto {label}")
        else:
            self.get_logger().error(f"[test] FALLITO — {label}")
        return ok


# ── helpers di stampa ─────────────────────────────────────────────────────────

def _print_joints(label: str, joints: list[float]) -> None:
    print(f"\n  {label}:")
    for name, val in zip(ARM_JOINT_NAMES, joints):
        print(f"    {name}: {val:+.4f} rad  ({val * 57.2958:+7.2f}°)")


def _print_delta(current: list[float], target: list[float]) -> None:
    print("\n  Delta (target - corrente):")
    for name, c, t in zip(ARM_JOINT_NAMES, current, target):
        d = (t - c) * 57.2958
        flag = "  ← grande" if abs(d) > 45 else ""
        print(f"    {name}: {d:+7.2f}°{flag}")


def _confirm(msg: str, no_confirm: bool) -> bool:
    if no_confirm:
        return True
    print(f"\n  >>> {msg}  (Ctrl+C = annulla) <<<")
    try:
        input()
        return True
    except KeyboardInterrupt:
        print("\nAnnullato.")
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    rclpy.init()
    node = _RealMoveTestNode(velocity=args.velocity)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # ── STEP 1: verifica /joint_states ────────────────────────────────────────
    print(f"\n{_SEP}")
    print("STEP 1 — Attendo /joint_states (timeout 10s) …")
    joints_now = node.wait_for_joints(timeout=10.0)

    if joints_now is None:
        print("[FAIL] Nessun joint state ricevuto su /joint_states.")
        print("       Verificare:")
        print("         1. robot raggiungibile via rete (ping robot_ip)")
        print("         2. FCI abilitata in Franka Desk")
        print("         3. launch file avviato: ros2 launch vlm_robot_planner_bringup")
        print("            real_robot.launch.py robot_ip:=<IP>")
        rclpy.shutdown()
        return 1

    # Verifica che i nomi giunti corrispondano al robot configurato
    robot = os.environ.get("VLM_ROBOT", "panda")
    expected_prefix = "fr3_joint" if robot == "fr3" else "panda_joint"
    if not ARM_JOINT_NAMES[0].startswith(expected_prefix[:-1]):
        print(f"[WARN] ARM_JOINT_NAMES inizia con '{ARM_JOINT_NAMES[0]}' "
              f"ma VLM_ROBOT='{robot}' — possibile mismatch.")

    print(f"[OK]  Joint states ricevuti.")
    print(f"      Nomi giunti: {ARM_JOINT_NAMES}")
    _print_joints("Posizione corrente", joints_now)

    if args.check_only:
        print(f"\n--check-only: nessun movimento eseguito.")
        print(f"{_SEP}\n")
        rclpy.shutdown()
        return 0

    # ── STEP 2: PTP → ready pose ──────────────────────────────────────────────
    print(f"\n{_SEP}")
    print(f"STEP 2 — PTP verso 'ready pose' al {args.velocity * 100:.0f}% velocita'")
    _print_joints("Target (ready pose)", _READY_POSE)
    _print_delta(joints_now, _READY_POSE)

    if not _confirm("Premi INVIO per eseguire il movimento", args.no_confirm):
        rclpy.shutdown()
        return 0

    ok = node.move_to(_READY_POSE, "ready pose")
    if not ok:
        print("[FAIL] Movimento fallito — controlla i log di move_group.")
        rclpy.shutdown()
        return 1
    print("[OK]  Ready pose raggiunta.")

    # ── STEP 3: ritorno alla posizione iniziale ───────────────────────────────
    print(f"\n{_SEP}")
    print("STEP 3 — Attesa 3s, poi ritorno alla posizione iniziale …")
    time.sleep(3.0)

    _print_joints("Posizione di ritorno", joints_now)

    if not _confirm("Premi INVIO per tornare alla posizione iniziale", args.no_confirm):
        rclpy.shutdown()
        return 0

    ok = node.move_to(joints_now, "posizione iniziale")
    if not ok:
        print("[FAIL] Ritorno fallito.")
        rclpy.shutdown()
        return 1

    print("[OK]  Posizione iniziale raggiunta.")
    print(f"\n{_SEP}")
    print("TUTTI I TEST SUPERATI.")
    print("La catena MoveIt 2 → bridge → controller → FR3 funziona correttamente.")
    print(f"{_SEP}\n")
    rclpy.shutdown()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Testa movimenti base sul robot reale FR3 via MoveIt 2"
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Solo verifica /joint_states, nessun movimento",
    )
    parser.add_argument(
        "--velocity", type=float, default=0.1,
        metavar="N",
        help="Scaling velocita' 0.0-1.0 (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Salta le conferme interattive",
    )
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
