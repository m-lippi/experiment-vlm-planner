"""
Parses a Fast Downward plan (list of PDDL action strings)
into a list of primitive calls ready for execution.
"""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class PrimitiveCall:
    name: str
    args: list[str]


# Maps PDDL domain action names → canonical robot primitive names.
# Multiple PDDL actions can map to the same primitive (e.g. unstack and pick
# both call PickPrimitive — the difference is in preconditions, not execution).
PDDL_TO_PRIMITIVE: dict[str, str] = {
    "pick":                 "pick",
    "unstack":              "pick",             # top-of-stack grasp → PickPrimitive
    "place":                "place",
    "stack":                "place",            # place-on-object → PlacePrimitive
    "look-at":              "look_at",
    "open-container":       "open_container",
    "close-container":      "close_container",
    "pick-from-container":  "pick",             # → PickPrimitive
    "place-in-container":   "place",            # → PlacePrimitive
    "navigate-to":          "navigate_to",
    "scan-scene":           "scan_scene",
}


def parse_plan(pddl_actions: list[str]) -> list[PrimitiveCall]:
    """
    Converts PDDL action strings to PrimitiveCalls, preserving the original
    PDDL action name. Use normalize_to_primitives() afterwards for execution.

    Example:
        "(pick red_cup table_a)" → PrimitiveCall(name="pick", args=["red_cup", "table_a"])
        "(unstack red_cup blue_box table_a)" → PrimitiveCall(name="unstack", args=[...])
    """
    calls = []
    for action in pddl_actions:
        inner = re.sub(r"[()]", "", action).strip()
        parts = inner.split()
        if not parts:
            continue
        calls.append(PrimitiveCall(name=parts[0], args=parts[1:]))
    return calls


def normalize_to_primitives(calls: list[PrimitiveCall]) -> list[PrimitiveCall]:
    """
    Translates PDDL action names to canonical robot primitive names using
    PDDL_TO_PRIMITIVE. Unknown action names are kept as-is.

    Use this before dispatching to the robot primitive layer.

    Example:
        PrimitiveCall("unstack", ["red_cup", "blue_box", "table_a"])
        → PrimitiveCall("pick",  ["red_cup", "blue_box", "table_a"])
    """
    return [
        PrimitiveCall(
            name=PDDL_TO_PRIMITIVE.get(call.name, call.name),
            args=call.args,
        )
        for call in calls
    ]
