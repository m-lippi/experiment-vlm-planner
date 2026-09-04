"""
Tests for VLM output parsing — no model loading required.
These verify that _parse_output handles VLM responses correctly,
including malformed JSON, markdown code fences, domain_template,
and domain_additions fields.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vlm.planner import VLMPlanner, VLMPlan, PlanStep


def _parser():
    """Return a planner instance without loading weights."""
    return VLMPlanner.__new__(VLMPlanner)


# ── Existing plan parsing (backward-compatible) ───────────────────────────────

def test_clean_json():
    raw = '{"goal": "pick the cup", "steps": [{"primitive": "pick", "args": {"object": "cup"}}]}'
    plan = _parser()._parse_output("pick the cup", raw)
    assert plan.goal == "pick the cup"
    assert len(plan.steps) == 1
    assert plan.steps[0] == PlanStep(primitive="pick", args={"object": "cup"})


def test_json_with_markdown_fences():
    raw = '```json\n{"goal": "pick", "steps": [{"primitive": "look_at", "args": {"object": "box"}}]}\n```'
    plan = _parser()._parse_output("pick", raw)
    assert len(plan.steps) == 1
    assert plan.steps[0].primitive == "look_at"


def test_malformed_json_returns_empty_plan():
    raw = "Sorry, I cannot do that."
    plan = _parser()._parse_output("do something", raw)
    assert plan.steps == []
    assert plan.raw_output == raw


def test_empty_steps_list():
    raw = '{"goal": "impossible task", "steps": [], "error": "no valid primitives"}'
    plan = _parser()._parse_output("impossible task", raw)
    assert plan.steps == []


def test_multi_step_plan():
    raw = """{
        "goal": "pick red cup and place on shelf",
        "steps": [
            {"primitive": "look_at", "args": {"object": "red_cup"}},
            {"primitive": "pick",    "args": {"object": "red_cup"}},
            {"primitive": "place",   "args": {"object": "red_cup", "location": "shelf"}}
        ]
    }"""
    plan = _parser()._parse_output("pick red cup and place on shelf", raw)
    assert len(plan.steps) == 3
    assert plan.steps[1].primitive == "pick"
    assert plan.steps[2].args == {"object": "red_cup", "location": "shelf"}


# ── domain_template field ─────────────────────────────────────────────────────

def test_domain_template_parsed_correctly():
    raw = '{"goal": "stack boxes", "domain_template": "manipulation_stacking", "domain_additions": {"new_types": [], "new_predicates": [], "new_actions": [], "modified_preconditions": {}}, "steps": []}'
    plan = _parser()._parse_output("stack boxes", raw)
    assert plan.domain_template == "manipulation_stacking"


def test_domain_template_defaults_to_base_when_missing():
    raw = '{"goal": "pick the cup", "steps": []}'
    plan = _parser()._parse_output("pick the cup", raw)
    assert plan.domain_template == "manipulation_base"


def test_domain_template_defaults_on_malformed_json():
    plan = _parser()._parse_output("task", "not json at all")
    assert plan.domain_template == "manipulation_base"


# ── domain_additions field ────────────────────────────────────────────────────

def test_domain_additions_parsed_correctly():
    raw = """{
        "goal": "pick locked box",
        "domain_template": "manipulation_base",
        "domain_additions": {
            "new_types": [],
            "new_predicates": ["(locked ?i - item)"],
            "new_actions": [
                {
                    "name": "unlock",
                    "parameters": "(?i - item)",
                    "precondition": "(locked ?i)",
                    "effect": "(not (locked ?i))"
                }
            ],
            "modified_preconditions": {"pick": ["(not (locked ?i))"]}
        },
        "steps": [
            {"primitive": "pick", "args": {"object": "box"}}
        ]
    }"""
    plan = _parser()._parse_output("pick locked box", raw)
    assert plan.domain_additions["new_predicates"] == ["(locked ?i - item)"]
    assert len(plan.domain_additions["new_actions"]) == 1
    assert plan.domain_additions["new_actions"][0]["name"] == "unlock"
    assert plan.domain_additions["modified_preconditions"] == {"pick": ["(not (locked ?i))"]}


def test_domain_additions_defaults_to_empty_when_missing():
    raw = '{"goal": "pick the cup", "steps": []}'
    plan = _parser()._parse_output("pick the cup", raw)
    assert plan.domain_additions["new_types"]      == []
    assert plan.domain_additions["new_predicates"] == []
    assert plan.domain_additions["new_actions"]    == []
    assert plan.domain_additions["modified_preconditions"] == {}


def test_domain_additions_defaults_on_malformed_json():
    plan = _parser()._parse_output("task", "not json")
    assert plan.domain_additions["new_types"] == []


# ── to_domain_additions() integration ────────────────────────────────────────

def test_to_domain_additions_returns_correct_type():
    from planner.domain_enricher import DomainAdditions
    raw = """{
        "goal": "test",
        "domain_template": "manipulation_base",
        "domain_additions": {
            "new_types": ["container - location"],
            "new_predicates": ["(locked ?i - item)"],
            "new_actions": [],
            "modified_preconditions": {}
        },
        "steps": []
    }"""
    plan = _parser()._parse_output("test", raw)
    additions = plan.to_domain_additions()
    assert isinstance(additions, DomainAdditions)
    assert additions.new_types == ["container - location"]
    assert additions.new_predicates == ["(locked ?i - item)"]


def test_to_domain_additions_empty_plan():
    from planner.domain_enricher import DomainAdditions
    plan = _parser()._parse_output("task", "not json")
    additions = plan.to_domain_additions()
    assert isinstance(additions, DomainAdditions)
    assert additions.new_types == []
    assert additions.new_predicates == []
    assert additions.new_actions == []
    assert additions.modified_preconditions == {}


# ── Full pipeline: parse → enrich ────────────────────────────────────────────

def test_parse_then_enrich_pipeline():
    """VLMPlan.to_domain_additions() feeds directly into DomainEnricher.enrich()."""
    from pathlib import Path
    from planner.domain_enricher import DomainEnricher

    raw = """{
        "goal": "unlock and pick the locked box",
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
            {"primitive": "pick", "args": {"object": "locked_box"}}
        ]
    }"""
    plan = _parser()._parse_output("unlock and pick", raw)

    domain_path = Path(__file__).parent.parent / "pddl" / "domains" / "manipulation_base.pddl"
    enricher, domain_text = DomainEnricher.from_file(domain_path)
    result = enricher.enrich(domain_text, plan.to_domain_additions())

    assert result.is_valid
    assert "(:action unlock" in result.domain_text
    assert "(locked ?i - item)" in result.domain_text
