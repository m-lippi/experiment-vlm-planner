"""
Tests for the 4 PDDL domain templates and plan_parser normalization.
Does NOT require Fast Downward — validates structure and primitive mapping only.
"""

import re
from pathlib import Path
import pytest
from planner.plan_parser import (
    parse_plan,
    normalize_to_primitives,
    PrimitiveCall,
    PDDL_TO_PRIMITIVE,
)

DOMAINS_DIR = Path(__file__).parent.parent / "pddl" / "domains"
PROBLEMS_DIR = Path(__file__).parent.parent / "pddl" / "problems"

DOMAIN_FILES = {
    "base":        DOMAINS_DIR / "manipulation_base.pddl",
    "stacking":    DOMAINS_DIR / "manipulation_stacking.pddl",
    "containers":  DOMAINS_DIR / "containers_manipulation.pddl",
    "navigation":  DOMAINS_DIR / "navigation_manipulation.pddl",
}

PROBLEM_FILES = {
    "base":        PROBLEMS_DIR / "base_example.pddl",
    "stacking":    PROBLEMS_DIR / "stacking_example.pddl",
    "containers":  PROBLEMS_DIR / "containers_example.pddl",
    "navigation":  PROBLEMS_DIR / "navigation_example.pddl",
}


# ── Domain file existence ─────────────────────────────────────────────────────

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_domain_file_exists(key):
    assert DOMAIN_FILES[key].exists(), f"Missing domain file: {DOMAIN_FILES[key]}"


@pytest.mark.parametrize("key", PROBLEM_FILES)
def test_problem_file_exists(key):
    assert PROBLEM_FILES[key].exists(), f"Missing problem file: {PROBLEM_FILES[key]}"


# ── Domain structure (syntactic checks without a PDDL parser) ─────────────────

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_domain_has_define(key):
    content = DOMAIN_FILES[key].read_text()
    assert "(define" in content

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_domain_has_requirements(key):
    content = DOMAIN_FILES[key].read_text()
    assert ":requirements" in content

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_domain_has_types(key):
    content = DOMAIN_FILES[key].read_text()
    assert ":types" in content

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_domain_has_predicates(key):
    content = DOMAIN_FILES[key].read_text()
    assert ":predicates" in content

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_domain_has_pick_action(key):
    content = DOMAIN_FILES[key].read_text()
    assert "(:action pick" in content

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_domain_has_place_action(key):
    content = DOMAIN_FILES[key].read_text()
    assert "(:action place" in content


# ── Domain-specific action presence ──────────────────────────────────────────

def test_stacking_domain_has_stack_and_unstack():
    content = DOMAIN_FILES["stacking"].read_text()
    assert "(:action stack"   in content
    assert "(:action unstack" in content

def test_stacking_domain_has_clear_predicate():
    content = DOMAIN_FILES["stacking"].read_text()
    assert "(clear " in content

def test_stacking_domain_has_stacked_on_predicate():
    content = DOMAIN_FILES["stacking"].read_text()
    assert "(stacked-on " in content

def test_containers_domain_has_container_type():
    content = DOMAIN_FILES["containers"].read_text()
    assert "container" in content

def test_containers_domain_has_open_close_actions():
    content = DOMAIN_FILES["containers"].read_text()
    assert "(:action open-container"  in content
    assert "(:action close-container" in content

def test_containers_domain_has_pick_place_container_actions():
    content = DOMAIN_FILES["containers"].read_text()
    assert "(:action pick-from-container"  in content
    assert "(:action place-in-container"   in content

def test_navigation_domain_has_zone_type():
    content = DOMAIN_FILES["navigation"].read_text()
    assert "zone" in content

def test_navigation_domain_has_navigate_to():
    content = DOMAIN_FILES["navigation"].read_text()
    assert "(:action navigate-to" in content

def test_navigation_domain_has_at_robot_predicate():
    content = DOMAIN_FILES["navigation"].read_text()
    assert "(at-robot " in content


# ── Problem / domain name consistency ────────────────────────────────────────

def _extract_domain_ref(problem_text: str) -> str:
    m = re.search(r"\(:domain\s+([\w-]+)\)", problem_text)
    return m.group(1) if m else ""

def _extract_domain_name(domain_text: str) -> str:
    m = re.search(r"\(define\s+\(domain\s+([\w-]+)\)", domain_text)
    return m.group(1) if m else ""

@pytest.mark.parametrize("key", DOMAIN_FILES)
def test_problem_references_correct_domain(key):
    domain_name = _extract_domain_name(DOMAIN_FILES[key].read_text())
    problem_ref  = _extract_domain_ref(PROBLEM_FILES[key].read_text())
    assert domain_name == problem_ref, (
        f"{key}: problem references '{problem_ref}' but domain is '{domain_name}'"
    )


# ── normalize_to_primitives ───────────────────────────────────────────────────

def test_normalize_pick_unchanged():
    calls = parse_plan(["(pick red_cup table_a)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "pick"

def test_normalize_unstack_becomes_pick():
    calls = parse_plan(["(unstack red_cup blue_box table_a)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "pick"
    assert normed[0].args == ["red_cup", "blue_box", "table_a"]

def test_normalize_stack_becomes_place():
    calls = parse_plan(["(stack red_cup blue_box table_a)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "place"

def test_normalize_look_at():
    calls = parse_plan(["(look-at red_cup)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "look_at"

def test_normalize_navigate_to():
    calls = parse_plan(["(navigate-to kitchen_zone storage_zone)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "navigate_to"

def test_normalize_open_container():
    calls = parse_plan(["(open-container drawer)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "open_container"

def test_normalize_pick_from_container_becomes_pick():
    calls = parse_plan(["(pick-from-container screwdriver drawer)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "pick"

def test_normalize_unknown_action_kept_as_is():
    calls = parse_plan(["(some-future-action obj1 obj2)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].name == "some-future-action"

def test_normalize_preserves_args():
    calls = parse_plan(["(unstack top_obj bot_obj surface_loc)"])
    normed = normalize_to_primitives(calls)
    assert normed[0].args == ["top_obj", "bot_obj", "surface_loc"]

def test_normalize_full_stacking_plan():
    raw = [
        "(unstack red_cup blue_box table_a)",
        "(place red_cup shelf_b)",
    ]
    normed = normalize_to_primitives(parse_plan(raw))
    assert normed[0].name == "pick"
    assert normed[1].name == "place"

def test_all_pddl_actions_in_mapping_are_strings():
    for k, v in PDDL_TO_PRIMITIVE.items():
        assert isinstance(k, str) and isinstance(v, str)
