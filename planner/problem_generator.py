"""
Generates a PDDL problem file dynamically from a VLMPlan.

The VLM identifies objects, their locations, and goal state from images.
This module converts that structured output into a valid PDDL problem
compatible with the domain templates in pddl/domains/.

Limitation (Phase 1): initial state is inferred from plan steps (pick → object
must be somewhere). Explicit scene state from the VLM will be added in a future
iteration when the VLMPlan output is extended with a dedicated scene description.
"""

from __future__ import annotations
from pathlib import Path

from vlm.planner import VLMPlan, PlanStep


_PICK_PRIMITIVES = {"pick", "unstack", "pick-from-container"}
_PLACE_PRIMITIVES = {"place", "stack", "place-in-container"}
# When hold primitives appear with no corresponding pick, infer_init_state treats the
# operated object as already held (arm continues from a prior iteration). Goal inference
# relies on VLM enrichment (new_actions effects) rather than hardcoded logic.
_HOLD_PRIMITIVES = {"pour", "tilt"}
# Standard primitives handled by dedicated goal inference (no enrichment needed)
_STANDARD_PRIMITIVES = _PICK_PRIMITIVES | _PLACE_PRIMITIVES | {"look_at"}

# Maps VLMPlan.domain_template → PDDL domain name (must match (define (domain ...)) in file)
DOMAIN_TEMPLATE_TO_NAME: dict[str, str] = {
    "manipulation_base":       "manipulation-base",
    "manipulation_stacking":   "manipulation-stacking",
    "containers_manipulation": "manipulation-containers",
    "navigation_manipulation": "manipulation-navigation",
}


def extract_objects_and_locations(steps: list[PlanStep]) -> tuple[set[str], set[str]]:
    """
    Walk the plan steps and collect all object and location names referenced.

    Returns:
        (objects, locations) — two sets of symbolic names.
    """
    objects:   set[str] = set()
    locations: set[str] = set()

    for step in steps:
        args = step.args
        if "object" in args:
            objects.add(args["object"])
        if "location" in args:
            locations.add(args["location"])
        if step.primitive == "look_at" and "target" in args:
            objects.add(args["target"])
        # Enrichment primitives: all arg values are items (not locations)
        if step.primitive not in _STANDARD_PRIMITIVES:
            for v in args.values():
                if isinstance(v, str) and v:
                    objects.add(v)

    return objects, locations


def infer_init_state(steps: list[PlanStep]) -> tuple[list[tuple], set[str]]:
    """
    Infer the minimal initial state.

    Handles three cases:
    - Normal (pick + place): objects start on source locations, gripper empty.
    - Pick-only: objects start on source locations, gripper empty.
    - Place-only (closed-loop iteration 2+): arm is already holding the objects
      that appear in place steps but have no corresponding pick step.

    Returns:
        (on_pairs, held_objects)
        on_pairs    — List of (object, location) for :init (on ...) facts.
        held_objects — Set of objects already held (no (on) fact, no gripper-empty).
    """
    on_pairs:    list[tuple[str, str]] = []
    held_objects: set[str]             = set()
    seen:         set[str]             = set()

    picked = {
        step.args.get("object", "")
        for step in steps
        if step.primitive in _PICK_PRIMITIVES and step.args.get("object", "")
    }

    # Objects placed but NOT picked in this plan → arm must already be holding them
    for step in steps:
        if step.primitive in _PLACE_PRIMITIVES:
            obj = step.args.get("object", "")
            if obj and obj not in picked:
                held_objects.add(obj)

    # Same for hold primitives (pour source, tilt object) — operated on a held item.
    # The VLM may use arbitrary PDDL parameter names (e.g. "can" instead of "source")
    # so fall back to positional extraction: first arg value = the held object.
    for step in steps:
        if step.primitive in _HOLD_PRIMITIVES:
            vals = list(step.args.values())
            obj = (step.args.get("source") or step.args.get("object") or
                   (vals[0] if vals else ""))
            if obj and obj not in picked:
                held_objects.add(obj)

    for step in steps:
        if step.primitive in _PICK_PRIMITIVES:
            obj = step.args.get("object", "")
            if obj and obj not in seen:
                loc = step.args.get("source", f"source_{obj}")
                on_pairs.append((obj, loc))
                seen.add(obj)

    return on_pairs, held_objects


def _goal_from_enrichment_action(
    action_def: dict, step_args: dict
) -> str | None:
    """
    Given a VLM-generated action definition and the concrete step arguments,
    derive the PDDL goal fact by finding the first positive (non-negated) predicate
    in the action's effect and substituting the actual argument values.

    Works with any predicate name the VLM chose — no hardcoded names.
    Returns a PDDL fact string like '(poured bottle glass)', or None if derivation fails.
    """
    import re

    effect_str = action_def.get("effect", "")
    params_str = action_def.get("parameters", "")

    # Parse parameters: "(?source - item ?target - item)" → [source, target]
    param_names = re.findall(r"\?([a-zA-Z][\w-]*)", params_str)
    arg_values  = list(step_args.values())
    binding = {pn: str(arg_values[i]) for i, pn in enumerate(param_names) if i < len(arg_values)}

    _PDDL_KW = {"and", "or", "not", "when", "forall", "exists"}

    def _top_level_exprs(text: str) -> list[str]:
        """Extract direct children expressions (respects nested parens)."""
        exprs, depth, start = [], 0, -1
        for i, c in enumerate(text):
            if c == "(":
                if depth == 0: start = i
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and start != -1:
                    exprs.append(text[start: i + 1])
                    start = -1
        return exprs

    def _find_positive_pred(expr: str) -> str | None:
        """Recursively search for the first positive predicate fact."""
        expr = expr.strip()
        if not expr.startswith("("):
            return None
        inner = expr[1:-1].strip()
        head  = inner.split()[0] if inner else ""

        if head in ("not",):        # negated — skip entirely
            return None
        if head in ("and", "or"):   # recurse into children
            for child in _top_level_exprs(inner[len(head):].strip()):
                result = _find_positive_pred(child)
                if result:
                    return result
            return None
        if head in _PDDL_KW:        # other keyword — skip
            return None

        # Leaf predicate — substitute ?vars
        tokens = inner.split()
        pred_name = tokens[0]
        pred_args = []
        for tok in tokens[1:]:
            if tok.startswith("?"):
                pred_args.append(binding.get(tok[1:], tok))
            elif tok not in ("-", "item", "location", "object"):
                pred_args.append(tok)
        return f"({pred_name} {' '.join(pred_args)})" if pred_args else f"({pred_name})"

    return _find_positive_pred(effect_str)


def infer_goal_state(
    steps: list[PlanStep],
    domain_additions: dict | None = None,
) -> list[tuple[str, ...]]:
    """
    Infer the goal state.

    Returns a list of goal facts as tuples:
    - ("on", obj, loc)      for place goals
    - ("holding", obj)      for pick-only plans (no place step)

    The "holding" sentinel is used when the plan has picks but no places —
    otherwise Fast Downward would return an empty plan because the default
    goal (gripper-empty) is already satisfied in the initial state.
    """
    enrichment_actions: dict[str, dict] = {}
    if domain_additions:
        for act in domain_additions.get("new_actions", []):
            name = act.get("name", "")
            if name:
                enrichment_actions[name] = act

    goal: list[tuple[str, ...]] = []

    for step in steps:
        if step.primitive in _PLACE_PRIMITIVES:
            obj = step.args.get("object", "")
            loc = step.args.get("location", "")
            if obj and loc:
                goal.append(("on", obj, loc))

    # Enrichment primitives: derive goal from the VLM-generated action definition.
    # The VLM is free to choose any predicate name — we extract it from its effect.
    for step in steps:
        if step.primitive not in _STANDARD_PRIMITIVES:
            act_def = enrichment_actions.get(step.primitive)
            if act_def:
                fact = _goal_from_enrichment_action(act_def, step.args)
                if fact:
                    goal.append(("_raw_fact", fact))  # raw PDDL fact, rendered as-is

    # Pick-only plan: goal = holding the picked object(s)
    if not goal:
        for step in steps:
            if step.primitive in _PICK_PRIMITIVES:
                obj = step.args.get("object", "")
                if obj:
                    goal.append(("holding", obj))

    # look_at-only plan: goal = camera aimed at target
    if not goal:
        for step in steps:
            if step.primitive == "look_at":
                target = step.args.get("target", "")
                if target:
                    goal.append(("camera-aimed-at", target))

    return goal


def generate_problem(
    plan: VLMPlan,
    domain_name: str | None = None,
    problem_name: str = "generated_problem",
) -> str:
    """
    Generate a PDDL problem string from a VLMPlan.

    The domain name is resolved automatically from plan.domain_template
    unless overridden by the domain_name parameter.

    Args:
        plan:         Output of VLMPlanner.plan().
        domain_name:  Override the domain name. If None, derived from
                      plan.domain_template via DOMAIN_TEMPLATE_TO_NAME.
        problem_name: Label for the generated problem (informational).

    Returns:
        PDDL problem as a string, ready to be written to a .pddl file.
    """
    if domain_name is None:
        domain_name = DOMAIN_TEMPLATE_TO_NAME.get(
            plan.domain_template, "manipulation-base"
        )

    objects, locations = extract_objects_and_locations(plan.steps)
    init_pairs, held_objects = infer_init_state(plan.steps)
    goal_facts = infer_goal_state(plan.steps, domain_additions=plan.domain_additions)

    all_locations = locations | {loc for _, loc in init_pairs}
    # Held objects are items but have no (on ...) fact and no source location
    all_objects = objects | held_objects

    lines: list[str] = []
    lines.append(f"(define (problem {problem_name})")
    lines.append(f"  (:domain {domain_name})")
    lines.append("")

    obj_str = " ".join(sorted(all_objects)) + " - item"      if all_objects  else ""
    loc_str = " ".join(sorted(all_locations)) + " - location" if all_locations else ""
    lines.append("  (:objects")
    if obj_str:
        lines.append(f"    {obj_str}")
    if loc_str:
        lines.append(f"    {loc_str}")
    lines.append("  )")
    lines.append("")

    # Init
    lines.append("  (:init")
    for obj, loc in init_pairs:
        lines.append(f"    (on {obj} {loc})")
    for obj in sorted(held_objects):
        lines.append(f"    (holding {obj})")   # arm already holds these (place-only plan)
    for obj in sorted(all_objects):
        lines.append(f"    (clear {obj})")
    for loc in sorted(all_locations):
        lines.append(f"    (reachable {loc})")
    # Items are also reachable on the table. Required when VLM-enriched actions
    # (e.g. pour, tilt) use (reachable ?target) where ?target is an item.
    # The domain now declares (reachable ?o - object) so this type-checks.
    for obj in sorted(all_objects):
        lines.append(f"    (reachable {obj})")
    if not held_objects:
        lines.append("    (gripper-empty)")    # omit if arm is holding something

    # Phase 2: camera-aimed-at for each pick object NOT covered by a look_at
    # in THIS plan.  If look_at IS in the plan it will ACHIEVE camera-aimed-at
    # (so it must NOT be in init — FD would skip look_at as redundant).
    # If look_at is absent the prior iteration already aimed the camera → add to init.
    look_at_targets = {
        step.args.get("target", "")
        for step in plan.steps
        if step.primitive == "look_at"
    }
    pick_objects = {
        step.args.get("object", "")
        for step in plan.steps
        if step.primitive in _PICK_PRIMITIVES and step.args.get("object", "")
    }
    for obj in sorted(pick_objects - look_at_targets):
        lines.append(f"    (camera-aimed-at {obj})")

    lines.append("  )")
    lines.append("")

    # Goal
    lines.append("  (:goal")
    on_goals      = [f for f in goal_facts if f[0] == "on"]
    holding_goals = [f for f in goal_facts if f[0] == "holding"]
    camera_goals  = [f for f in goal_facts if f[0] == "camera-aimed-at"]
    raw_goals     = [f for f in goal_facts if f[0] == "_raw_fact"]

    all_goal_facts = []
    for f in on_goals:
        all_goal_facts.append(f"(on {f[1]} {f[2]})")
    for f in holding_goals:
        all_goal_facts.append(f"(holding {f[1]})")
    for f in camera_goals:
        all_goal_facts.append(f"(camera-aimed-at {f[1]})")
    for f in raw_goals:
        all_goal_facts.append(f[1])  # already a full PDDL fact string

    if not all_goal_facts:
        all_goal_facts = ["(gripper-empty)  ; no explicit goal inferred"]

    if len(all_goal_facts) == 1:
        lines.append(f"    {all_goal_facts[0]}")
    else:
        lines.append("    (and")
        for fact in all_goal_facts:
            lines.append(f"      {fact}")
        lines.append("    )")
    lines.append("  )")
    lines.append(")")

    return "\n".join(lines)


def write_problem(
    plan: VLMPlan,
    output_path: str | Path,
    domain_name: str | None = None,
) -> Path:
    """
    Generate and write the PDDL problem to a file.

    Returns:
        Path to the written file.
    """
    content = generate_problem(plan, domain_name=domain_name)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
