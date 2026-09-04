"""
Tests for PDDL problem generation from VLM plan output.
No GPU, no ROS, no model loading required.

Run with -s to see the showcase tests print generated PDDL:
    pytest tests/test_problem_generator.py -v -s
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vlm.planner import VLMPlan, PlanStep
from planner.problem_generator import (
    extract_objects_and_locations,
    infer_init_state,
    infer_goal_state,
    generate_problem,
    DOMAIN_TEMPLATE_TO_NAME,
)


def _plan(*steps: tuple, template: str = "manipulation_base") -> VLMPlan:
    """Helper: build a VLMPlan from (primitive, args_dict) tuples."""
    return VLMPlan(
        goal="test",
        steps=[PlanStep(primitive=p, args=a) for p, a in steps],
        raw_output="",
        domain_template=template,
    )


# ── extract_objects_and_locations ─────────────────────────────────────────────

def test_extract_single_pick_place():
    plan = _plan(
        ("pick",  {"object": "red_cup"}),
        ("place", {"object": "red_cup", "location": "shelf"}),
    )
    objects, locations = extract_objects_and_locations(plan.steps)
    assert objects   == {"red_cup"}
    assert locations == {"shelf"}


def test_extract_multiple_objects():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup",  "location": "table_b"}),
        ("pick",  {"object": "box"}),
        ("place", {"object": "box",  "location": "shelf"}),
    )
    objects, _ = extract_objects_and_locations(plan.steps)
    assert objects == {"cup", "box"}


def test_extract_ignores_look_at():
    plan = _plan(
        ("look_at", {"object": "cup"}),
        ("pick",    {"object": "cup"}),
        ("place",   {"object": "cup", "location": "shelf"}),
    )
    objects, locations = extract_objects_and_locations(plan.steps)
    assert "cup"   in objects
    assert "shelf" in locations


# ── infer_init_state ──────────────────────────────────────────────────────────

def test_init_assigns_source_location():
    plan = _plan(("pick", {"object": "red_cup"}))
    on_pairs, held = infer_init_state(plan.steps)
    assert len(on_pairs) == 1
    obj, loc = on_pairs[0]
    assert obj == "red_cup"
    assert "red_cup" in loc   # default: source_red_cup


def test_init_no_duplicates_for_same_object():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
        ("pick",  {"object": "cup"}),
    )
    on_pairs, _ = infer_init_state(plan.steps)
    assert [o for o, _ in on_pairs].count("cup") == 1


# ── infer_goal_state ──────────────────────────────────────────────────────────

def test_goal_from_place():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
    )
    # Returns 3-tuples: ("on", obj, loc)
    assert ("on", "cup", "shelf") in infer_goal_state(plan.steps)


def test_no_goal_if_no_place():
    # pick-only plan → goal is (holding obj) so the planner doesn't return empty plan
    plan = _plan(("pick", {"object": "cup"}))
    assert ("holding", "cup") in infer_goal_state(plan.steps)


# ── domain template name mapping ──────────────────────────────────────────────

def test_domain_template_to_name_mapping():
    assert DOMAIN_TEMPLATE_TO_NAME["manipulation_base"]       == "manipulation-base"
    assert DOMAIN_TEMPLATE_TO_NAME["manipulation_stacking"]   == "manipulation-stacking"
    assert DOMAIN_TEMPLATE_TO_NAME["containers_manipulation"] == "manipulation-containers"
    assert DOMAIN_TEMPLATE_TO_NAME["navigation_manipulation"] == "manipulation-navigation"


def test_generate_problem_uses_plan_domain_template():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
        template="manipulation_stacking",
    )
    pddl = generate_problem(plan)
    assert "(:domain manipulation-stacking)" in pddl


def test_generate_problem_domain_name_override():
    plan = _plan(("pick", {"object": "cup"}), ("place", {"object": "cup", "location": "shelf"}))
    pddl = generate_problem(plan, domain_name="my-custom-domain")
    assert "(:domain my-custom-domain)" in pddl


# ── generate_problem — structural correctness ─────────────────────────────────

def test_generated_pddl_uses_item_type():
    plan = _plan(
        ("pick",  {"object": "red_cup"}),
        ("place", {"object": "red_cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "red_cup - item" in pddl        # new type name
    assert "red_cup - object" not in pddl  # old type name must NOT appear


def test_generated_pddl_uses_location_type():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "- location" in pddl
    assert "shelf" in pddl


def test_generated_pddl_has_clear_facts():
    plan = _plan(
        ("pick",  {"object": "red_cup"}),
        ("place", {"object": "red_cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "(clear red_cup)" in pddl


def test_generated_pddl_has_reachable_facts():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "(reachable shelf)" in pddl


def test_generated_pddl_has_gripper_empty():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "(gripper-empty)" in pddl


def test_generated_pddl_goal_single():
    plan = _plan(
        ("pick",  {"object": "red_cup"}),
        ("place", {"object": "red_cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "(on red_cup shelf)" in pddl


def test_generated_pddl_multi_goal():
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup",  "location": "shelf"}),
        ("pick",  {"object": "box"}),
        ("place", {"object": "box",  "location": "table_b"}),
    )
    pddl = generate_problem(plan)
    assert "(and"           in pddl
    assert "(on cup shelf)"   in pddl
    assert "(on box table_b)" in pddl


def test_generated_pddl_full_structure():
    plan = _plan(
        ("look_at", {"object": "red_cup"}),
        ("pick",    {"object": "red_cup"}),
        ("place",   {"object": "red_cup", "location": "shelf"}),
    )
    pddl = generate_problem(plan)
    assert "(define (problem"          in pddl
    assert "(:domain manipulation-base)" in pddl
    assert "(:objects"                 in pddl
    assert "(:init"                    in pddl
    assert "(:goal"                    in pddl


# ── ════════════════════════════════════════════════════════════════════════ ──
#    SHOWCASE TESTS — run with `pytest -s` to see formatted PDDL output
#    These show what the system actually generates at each pipeline stage
# ── ════════════════════════════════════════════════════════════════════════ ──

def test_showcase_base_problem(capsys):
    """Showcase: simple pick-place problem with manipulation-base domain."""
    plan = _plan(
        ("look_at", {"object": "red_cup"}),
        ("pick",    {"object": "red_cup"}),
        ("place",   {"object": "red_cup", "location": "shelf_b"}),
        template="manipulation_base",
    )
    pddl = generate_problem(plan, problem_name="pick-red-cup")

    _print_showcase("GENERATED PDDL — base pick/place", pddl)
    assert "(on red_cup shelf_b)" in pddl


def test_showcase_stacking_problem(capsys):
    """Showcase: stacking problem — inferred from unstack primitive."""
    plan = _plan(
        ("look_at", {"object": "red_cup"}),
        ("pick",    {"object": "red_cup", "source": "blue_box"}),   # source = stacked on
        ("place",   {"object": "red_cup", "location": "shelf_b"}),
        template="manipulation_stacking",
    )
    pddl = generate_problem(plan, problem_name="unstack-red-cup")

    _print_showcase("GENERATED PDDL — stacking template", pddl)
    assert "manipulation-stacking" in pddl
    assert "(on red_cup blue_box)"  in pddl  # inferred source


def test_showcase_multi_object_problem(capsys):
    """Showcase: two objects, two goals, multi-goal PDDL."""
    plan = _plan(
        ("pick",  {"object": "cup"}),
        ("place", {"object": "cup",  "location": "shelf_a"}),
        ("pick",  {"object": "box"}),
        ("place", {"object": "box",  "location": "shelf_b"}),
        template="manipulation_base",
    )
    pddl = generate_problem(plan, problem_name="tidy-table")

    _print_showcase("GENERATED PDDL — multi-object, multi-goal", pddl)
    assert "(on cup shelf_a)" in pddl
    assert "(on box shelf_b)" in pddl


def test_showcase_full_pipeline(capsys):
    """
    Showcase: full pipeline from VLM JSON → VLMPlan → PDDL problem → domain enrichment.
    Shows all three outputs side by side.
    """
    from pathlib import Path
    from planner.domain_enricher import DomainEnricher

    # Simulate the JSON the VLM would output for a locked-object scene
    vlm_json = """{
        "goal": "pick the locked screwdriver from the drawer",
        "domain_template": "manipulation_base",
        "domain_additions": {
            "new_types": [],
            "new_predicates": ["(locked ?i - item)"],
            "new_actions": [{
                "name": "unlock",
                "parameters": "(?i - item)",
                "precondition": "(and (locked ?i) (gripper-empty))",
                "effect": "(not (locked ?i))"
            }],
            "modified_preconditions": {"pick": ["(not (locked ?i))"]}
        },
        "steps": [
            {"primitive": "look_at", "args": {"target": "screwdriver"}},
            {"primitive": "pick",    "args": {"object": "screwdriver"}},
            {"primitive": "place",   "args": {"object": "screwdriver", "location": "table_a"}}
        ]
    }"""

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from vlm.planner import VLMPlanner
    parser = VLMPlanner.__new__(VLMPlanner)
    plan = parser._parse_output("pick locked screwdriver", vlm_json)

    problem_pddl = generate_problem(plan, problem_name="locked-screwdriver")

    domain_path = Path(__file__).parent.parent / "pddl" / "domains" / "manipulation_base.pddl"
    enricher, base_domain = DomainEnricher.from_file(domain_path)
    result = enricher.enrich(base_domain, plan.to_domain_additions())

    _print_showcase("VLM OUTPUT — parsed plan", (
        f"  goal:            {plan.goal}\n"
        f"  domain_template: {plan.domain_template}\n"
        f"  steps:           {[s.primitive for s in plan.steps]}\n"
        f"  enrichment:      {plan.to_domain_additions()}"
    ))
    _print_showcase("GENERATED PDDL PROBLEM", problem_pddl)
    _print_showcase(
        f"ENRICHED DOMAIN  (valid={result.is_valid})\n"
        f"  applied:  {result.additions_applied}\n"
        f"  skipped:  {result.additions_skipped}",
        result.domain_text,
        truncate=40,
    )

    assert plan.domain_template == "manipulation_base"
    assert result.is_valid
    assert "(:action unlock" in result.domain_text
    assert "(on screwdriver source_screwdriver)" in problem_pddl


# ── helper ────────────────────────────────────────────────────────────────────

def _print_showcase(title: str, content: str, truncate: int | None = None) -> None:
    border = "═" * 62
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    lines = content.splitlines()
    if truncate and len(lines) > truncate:
        for line in lines[:truncate]:
            print(line)
        print(f"  ... ({len(lines) - truncate} more lines)")
    else:
        print(content)
    print(border)
