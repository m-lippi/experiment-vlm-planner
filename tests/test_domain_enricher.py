"""
Tests for DomainEnricher.
Does NOT require Fast Downward — validates enrichment logic only.
"""

from pathlib import Path
import pytest
from planner.domain_enricher import DomainEnricher, DomainAdditions, EnrichmentResult

DOMAINS_DIR = Path(__file__).parent.parent / "pddl" / "domains"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_domain() -> str:
    return (DOMAINS_DIR / "manipulation_base.pddl").read_text()


@pytest.fixture
def stacking_domain() -> str:
    return (DOMAINS_DIR / "manipulation_stacking.pddl").read_text()


@pytest.fixture
def enricher() -> DomainEnricher:
    return DomainEnricher()


# ── Empty additions → domain unchanged ───────────────────────────────────────

def test_empty_additions_returns_original(enricher, base_domain):
    result = enricher.enrich(base_domain, DomainAdditions())
    assert result.domain_text == base_domain
    assert result.is_valid
    assert result.additions_applied == []
    assert result.additions_skipped == []


# ── New types ─────────────────────────────────────────────────────────────────

def test_add_new_type(enricher, base_domain):
    additions = DomainAdditions(new_types=["container - location"])
    result = enricher.enrich(base_domain, additions)
    assert "container - location" in result.domain_text
    assert result.is_valid
    assert any("type" in a for a in result.additions_applied)


def test_duplicate_type_is_skipped(enricher, base_domain):
    additions = DomainAdditions(new_types=["item - object"])  # already exists
    result = enricher.enrich(base_domain, additions)
    assert any("already exists" in s for s in result.additions_skipped)
    assert result.domain_text == base_domain  # unchanged


# ── New predicates ────────────────────────────────────────────────────────────

def test_add_new_predicate(enricher, base_domain):
    additions = DomainAdditions(new_predicates=["(locked ?i - item)"])
    result = enricher.enrich(base_domain, additions)
    assert "(locked ?i - item)" in result.domain_text
    assert result.is_valid


def test_duplicate_predicate_is_skipped(enricher, base_domain):
    additions = DomainAdditions(new_predicates=["(on ?i - item ?l - location)"])
    result = enricher.enrich(base_domain, additions)
    assert any("already exists" in s for s in result.additions_skipped)


def test_add_multiple_predicates(enricher, base_domain):
    additions = DomainAdditions(
        new_predicates=["(locked ?i - item)", "(fragile ?i - item)"]
    )
    result = enricher.enrich(base_domain, additions)
    assert "(locked ?i - item)" in result.domain_text
    assert "(fragile ?i - item)" in result.domain_text
    assert len(result.additions_applied) == 2


# ── New actions ───────────────────────────────────────────────────────────────

def test_add_new_action(enricher, base_domain):
    additions = DomainAdditions(
        new_predicates=["(locked ?i - item)"],
        new_actions=[{
            "name":         "unlock",
            "parameters":   "(?i - item)",
            "precondition": "(locked ?i)",
            "effect":       "(not (locked ?i))",
        }]
    )
    result = enricher.enrich(base_domain, additions)
    assert "(:action unlock" in result.domain_text
    assert result.is_valid


def test_duplicate_action_is_skipped(enricher, base_domain):
    additions = DomainAdditions(
        new_actions=[{
            "name":         "pick",    # already in domain
            "parameters":   "(?i - item ?l - location)",
            "precondition": "(on ?i ?l)",
            "effect":       "(holding ?i)",
        }]
    )
    result = enricher.enrich(base_domain, additions)
    assert any("already exists" in s for s in result.additions_skipped)


def test_action_with_missing_name_is_skipped(enricher, base_domain):
    additions = DomainAdditions(
        new_actions=[{"parameters": "(?i - item)", "precondition": "(on ?i ?l)", "effect": "(holding ?i)"}]
    )
    result = enricher.enrich(base_domain, additions)
    assert any("missing" in s for s in result.additions_skipped)


def test_action_with_unbalanced_parens_is_skipped(enricher, base_domain):
    additions = DomainAdditions(
        new_actions=[{
            "name":         "broken",
            "parameters":   "(?i - item",  # unbalanced
            "precondition": "(on ?i ?l)",
            "effect":       "(holding ?i)",
        }]
    )
    result = enricher.enrich(base_domain, additions)
    assert any("unbalanced" in s for s in result.additions_skipped)


# ── Modified preconditions ────────────────────────────────────────────────────

def test_extend_precondition_wraps_single_condition(enricher):
    """Base domain's pick precondition is (and ...) — verify new cond is inserted."""
    domain = (DOMAINS_DIR / "manipulation_base.pddl").read_text()
    additions = DomainAdditions(
        new_predicates=["(locked ?i - item)"],
        modified_preconditions={"pick": ["(not (locked ?i))"]}
    )
    result = enricher.enrich(domain, additions)
    assert "(not (locked ?i))" in result.domain_text
    assert result.is_valid


def test_extend_precondition_on_nonexistent_action(enricher, base_domain):
    additions = DomainAdditions(
        modified_preconditions={"fly": ["(wings ?i)"]}
    )
    result = enricher.enrich(base_domain, additions)
    assert any("not found" in s for s in result.additions_skipped)


def test_condition_with_unbalanced_parens_is_skipped(enricher, base_domain):
    additions = DomainAdditions(
        modified_preconditions={"pick": ["(not (locked ?i)"]}  # missing closing )
    )
    result = enricher.enrich(base_domain, additions)
    assert any("unbalanced" in s for s in result.additions_skipped)


# ── Validation ────────────────────────────────────────────────────────────────

def test_validation_catches_unbalanced_domain():
    enricher = DomainEnricher()
    broken = "(define (domain test) (:types item - object) (:predicates (on ?i - item)) (:action pick :parameters (?i) :precondition (on ?i) :effect (holding ?i)"
    result = enricher.enrich(broken, DomainAdditions())
    assert not result.is_valid
    assert any("unbalanced" in e for e in result.errors)


def test_valid_enriched_domain_passes_validation(enricher, base_domain):
    result = enricher.enrich(base_domain, DomainAdditions())
    assert result.is_valid
    assert result.errors == []


# ── from_file convenience ─────────────────────────────────────────────────────

def test_from_file_loads_domain():
    enricher, domain_text = DomainEnricher.from_file(
        DOMAINS_DIR / "manipulation_base.pddl"
    )
    assert isinstance(enricher, DomainEnricher)
    assert "(define" in domain_text


# ── Realistic end-to-end scenario ────────────────────────────────────────────

def test_realistic_locked_object_scenario(enricher, base_domain):
    """
    Scenario: VLM sees a locked box on the table.
    Enrichment: add 'locked' predicate + 'unlock' action + constraint on pick.

    Expected plan after enrichment:
        unlock(box) → pick(box, table) → place(box, shelf)
    """
    additions = DomainAdditions(
        new_predicates=["(locked ?i - item)"],
        new_actions=[{
            "name":         "unlock",
            "parameters":   "(?i - item)",
            "precondition": "(and (locked ?i) (gripper-empty))",
            "effect":       "(not (locked ?i))",
        }],
        modified_preconditions={
            "pick": ["(not (locked ?i))"]
        },
    )
    result = enricher.enrich(base_domain, additions)

    assert result.is_valid
    assert "(locked ?i - item)" in result.domain_text
    assert "(:action unlock"   in result.domain_text
    assert "(not (locked ?i))" in result.domain_text
    assert len(result.additions_applied) == 3
    assert result.additions_skipped == []


def test_containers_enrichment_on_stacking_template(enricher, stacking_domain):
    """
    Scenario: VLM sees a drawer in a stacking-type scene.
    Enrichment: promote stacking template to container-aware.
    """
    additions = DomainAdditions(
        new_types=["container - location"],
        new_predicates=[
            "(open ?c - container)",
            "(closed ?c - container)",
            "(in-container ?i - item ?c - container)",
        ],
        new_actions=[
            {
                "name":         "open-container",
                "parameters":   "(?c - container)",
                "precondition": "(and (closed ?c) (gripper-empty) (reachable ?c))",
                "effect":       "(and (open ?c) (not (closed ?c)))",
            },
            {
                "name":         "pick-from-container",
                "parameters":   "(?i - item ?c - container)",
                "precondition": "(and (in-container ?i ?c) (clear ?i) (open ?c) (gripper-empty) (reachable ?c))",
                "effect":       "(and (holding ?i) (not (gripper-empty)) (not (in-container ?i ?c)))",
            },
        ],
    )
    result = enricher.enrich(stacking_domain, additions)

    assert result.is_valid
    assert "container - location" in result.domain_text
    assert "(:action open-container"       in result.domain_text
    assert "(:action pick-from-container"  in result.domain_text
    assert len(result.additions_applied) == 6  # 1 type + 3 predicates + 2 actions
