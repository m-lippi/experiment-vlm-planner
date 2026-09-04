"""
Tests for the Pipeline class (planner/pipeline.py).

No GPU, no ROS 2, no Fast Downward binary required.
VLM and FastDownward are replaced by lightweight stubs so tests run instantly.
Showcase tests (test_showcase_*) print pipeline traces; run with pytest -s to see them.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from planner.pipeline import Pipeline, PipelineResult, DOMAIN_TEMPLATE_FILES
from planner.plan_parser import PrimitiveCall
from vlm.planner import VLMPlan, PlanStep


DOMAINS_DIR = Path(__file__).parent.parent / "pddl" / "domains"


# ── Stubs ─────────────────────────────────────────────────────────────────────

class _StubVLM:
    """Replaces VLMPlanner — returns a fixed VLMPlan without touching the GPU."""

    def __init__(self, plan: VLMPlan) -> None:
        self._plan = plan

    def plan(self, command: str, images: list) -> VLMPlan:
        return self._plan


class _StubFD:
    """Replaces FastDownwardPlanner — returns a fixed action list without running FD."""

    def __init__(self, actions: list[str] | None = None, raise_error: bool = False):
        self._actions     = actions
        self._raise_error = raise_error
        self.call_count   = 0

    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        self.call_count += 1
        if self._raise_error:
            raise RuntimeError("fake FD error")
        return self._actions


def _make_plan(template: str = "manipulation_base") -> VLMPlan:
    return VLMPlan(
        goal="pick red_cup and place on shelf",
        steps=[
            PlanStep("pick",  {"object": "red_cup"}),
            PlanStep("place", {"object": "red_cup", "location": "shelf_b"}),
        ],
        raw_output="",
        domain_template=template,
    )


_DEFAULT_FD_ACTIONS = object()   # sentinel: "use default success actions"


def _make_pipeline(
    plan:        VLMPlan | None  = None,
    fd_actions                   = _DEFAULT_FD_ACTIONS,
    fd_error:    bool            = False,
    repair_retries: int          = 3,
) -> Pipeline:
    if plan is None:
        plan = _make_plan()
    if fd_actions is _DEFAULT_FD_ACTIONS:
        fd_actions = [
            "(pick red_cup source_red_cup)",
            "(place red_cup shelf_b)",
        ]
    return Pipeline(
        vlm=_StubVLM(plan),
        fd_planner=_StubFD(fd_actions, raise_error=fd_error),
        domains_dir=DOMAINS_DIR,
        repair_retries=repair_retries,
    )


# ── Stage 1: VLM ──────────────────────────────────────────────────────────────

def test_pipeline_fails_when_vlm_not_loaded():
    """
    Pipeline with vlm=None returns success=False, failure_stage="vlm", and an error
    referencing load_vlm(). Prevents silent failures when the model was not started.
    """
    pipeline = Pipeline(vlm=None, fd_planner=_StubFD([]), domains_dir=DOMAINS_DIR)
    result = pipeline.run("pick the cup", images=[])

    assert not result.success
    assert result.failure_stage == "vlm"
    assert "load_vlm" in result.error


def test_pipeline_accepts_precomputed_vlm_plan():
    """
    A pre-built VLMPlan passed via vlm_plan= bypasses VLM inference.
    Useful for dry-run tests and replaying stored plans.
    """
    plan = _make_plan()
    pipeline = _make_pipeline(plan=plan)
    result = pipeline.run("ignored command", images=[], vlm_plan=plan)

    assert result.success
    assert result.vlm_plan is plan


# ── Stage 2: Domain selection + enrichment ────────────────────────────────────

def test_pipeline_selects_correct_domain_file():
    """
    When domain_template="manipulation_stacking", the pipeline loads
    manipulation_stacking.pddl. A wrong template would cause planning failure.
    """
    plan = _make_plan(template="manipulation_stacking")
    pipeline = _make_pipeline(plan=plan)
    result = pipeline.run("pick", images=[], vlm_plan=plan)

    assert result.success
    assert "manipulation-stacking" in result.enrichment_result.domain_text


def test_pipeline_applies_domain_enrichment():
    """
    VLM-suggested domain additions (new predicates) are applied before planning.
    Core thesis innovation — verifies enrichment is wired into the pipeline.
    """
    plan = _make_plan()
    plan.domain_additions["new_predicates"] = ["(locked ?i - item)"]
    pipeline = _make_pipeline(plan=plan)
    result = pipeline.run("pick", images=[], vlm_plan=plan)

    assert result.success
    assert "(locked ?i - item)" in result.enrichment_result.domain_text


def test_pipeline_fails_on_missing_domain_file():
    """A non-existent template name causes failure_stage="enrichment"."""
    plan = _make_plan(template="nonexistent_template")
    pipeline = _make_pipeline(plan=plan)
    result = pipeline.run("pick", images=[], vlm_plan=plan)

    assert not result.success
    assert result.failure_stage == "enrichment"


# ── Stage 3: Problem generation ───────────────────────────────────────────────

def test_pipeline_generates_pddl_problem():
    """
    The generated PDDL problem contains the objects and goal extracted from the VLMPlan.
    Validates that ProblemGenerator is wired into the pipeline.
    """
    plan = _make_plan()
    pipeline = _make_pipeline(plan=plan)
    result = pipeline.run("pick", images=[], vlm_plan=plan)

    assert result.success
    assert "red_cup - item" in result.pddl_problem
    assert "(on red_cup shelf_b)" in result.pddl_problem   # goal


# ── Stage 4: Fast Downward + repair loop ─────────────────────────────────────

def test_pipeline_success_end_to_end():
    """
    All 5 stages succeed — VLM, enrichment, problem generation, FD, and parse.
    Returns success=True with pick and place primitives and repair_attempts=0.
    """
    result = _make_pipeline().run("pick red cup", images=[], vlm_plan=_make_plan())

    assert result.success
    assert result.repair_attempts == 0
    assert len(result.primitives) == 2
    assert result.primitives[0].name == "pick"
    assert result.primitives[1].name == "place"


def test_pipeline_fails_when_fd_returns_no_plan():
    """
    FastDownward returning None (unsolvable problem) causes success=False.
    failure_stage is "planning" or "repair_exhausted".
    """
    pipeline = _make_pipeline(fd_actions=None, repair_retries=0)
    result = pipeline.run("pick", images=[], vlm_plan=_make_plan())

    assert not result.success
    assert result.failure_stage in ("planning", "repair_exhausted")


def test_pipeline_repair_loop_retries_on_failure():
    """With repair_retries=2 and FD always returning None, FD is called 3 times (1 initial + 2 retries)."""
    fd_stub = _StubFD(actions=None)
    pipeline = Pipeline(
        vlm=_StubVLM(_make_plan()),
        fd_planner=fd_stub,
        domains_dir=DOMAINS_DIR,
        repair_retries=2,
    )
    result = pipeline.run("pick", images=[], vlm_plan=_make_plan())

    assert not result.success
    assert fd_stub.call_count == 3          # 1 + 2 retries
    assert result.repair_attempts == 2


def test_pipeline_repair_loop_stops_on_first_success():
    """FD is called exactly once when it succeeds on the first attempt; no unnecessary retries."""
    fd_stub = _StubFD(actions=["(pick red_cup source_red_cup)", "(place red_cup shelf_b)"])
    pipeline = Pipeline(
        vlm=_StubVLM(_make_plan()),
        fd_planner=fd_stub,
        domains_dir=DOMAINS_DIR,
        repair_retries=3,
    )
    result = pipeline.run("pick", images=[], vlm_plan=_make_plan())

    assert result.success
    assert fd_stub.call_count == 1
    assert result.repair_attempts == 0


# ── Stage 5: Normalization ────────────────────────────────────────────────────

def test_pipeline_normalizes_pddl_actions_to_primitives():
    """
    PDDL action "unstack" (from the stacking template) is normalized to primitive "pick".
    The robot only knows the 7 primitives — normalization bridges the PDDL action name gap.
    """
    plan = _make_plan(template="manipulation_stacking")
    fd_stub = _StubFD(actions=[
        "(unstack red_cup blue_box table_a)",
        "(place red_cup shelf_b table_a)",
    ])
    pipeline = Pipeline(
        vlm=_StubVLM(plan),
        fd_planner=fd_stub,
        domains_dir=DOMAINS_DIR,
    )
    result = pipeline.run("pick", images=[], vlm_plan=plan)

    assert result.success
    assert result.primitives[0].name == "pick"   # "unstack" normalized to "pick"
    assert result.primitives[1].name == "place"


# ── Domain file coverage ──────────────────────────────────────────────────────

def test_all_domain_templates_resolve_to_existing_files():
    """All entries in DOMAIN_TEMPLATE_FILES resolve to existing .pddl files. Catches renames."""
    for template_name, filename in DOMAIN_TEMPLATE_FILES.items():
        path = DOMAINS_DIR / filename
        assert path.exists(), f"Missing domain file for template '{template_name}': {path}"


# ── PipelineResult completeness ───────────────────────────────────────────────

def test_pipeline_result_carries_all_intermediates():
    """
    On success, PipelineResult carries all intermediate artifacts (vlm_plan,
    enrichment_result, pddl_problem, pddl_actions). Required for per-stage debugging.
    """
    result = _make_pipeline().run("pick", images=[], vlm_plan=_make_plan())

    assert result.vlm_plan is not None
    assert result.enrichment_result is not None
    assert result.pddl_problem != ""
    assert result.pddl_actions is not None
    assert result.error is None


# ── ════════════════════════════════════════════════════════════════════════ ──
#    SHOWCASE — run with `pytest -s` to see the full pipeline trace
# ── ════════════════════════════════════════════════════════════════════════ ──

def test_showcase_successful_pipeline(capsys):
    """Showcase: traces every stage of a successful base pick/place pipeline."""
    plan   = _make_plan()
    result = _make_pipeline(plan=plan).run("pick red cup", images=[], vlm_plan=plan)

    _section("STAGE 1 — VLM OUTPUT", (
        f"  goal:            {result.vlm_plan.goal}\n"
        f"  domain_template: {result.vlm_plan.domain_template}\n"
        f"  steps:           {[s.primitive for s in result.vlm_plan.steps]}"
    ))
    _section("STAGE 2 — DOMAIN ENRICHMENT", (
        f"  template file:   manipulation_base.pddl\n"
        f"  additions applied: {result.enrichment_result.additions_applied}\n"
        f"  valid: {result.enrichment_result.is_valid}"
    ))
    _section("STAGE 3 — PDDL PROBLEM", result.pddl_problem)
    _section("STAGE 4 — FAST DOWNWARD OUTPUT (stub)", "\n".join(f"  {a}" for a in result.pddl_actions))
    _section("STAGE 5 — PRIMITIVES READY FOR DISPATCH", "\n".join(
        f"  {i+1}. {p.name}({p.args})" for i, p in enumerate(result.primitives)
    ))
    _section("RESULT", f"  success={result.success}  repair_attempts={result.repair_attempts}")

    assert result.success


def test_showcase_enrichment_pipeline(capsys):
    """Showcase: domain enrichment with a locked-object scenario — VLM suggestion modifies the domain before planning."""
    plan = VLMPlan(
        goal="unlock and pick the locked screwdriver",
        steps=[
            PlanStep("look_at", {"target": "screwdriver"}),
            PlanStep("pick",    {"object": "screwdriver"}),
            PlanStep("place",   {"object": "screwdriver", "location": "workbench"}),
        ],
        raw_output="",
        domain_template="manipulation_base",
        domain_additions={
            "new_types": [],
            "new_predicates": ["(locked ?i - item)"],
            "new_actions": [{
                "name": "unlock",
                "parameters": "(?i - item)",
                "precondition": "(and (locked ?i) (gripper-empty))",
                "effect": "(not (locked ?i))",
            }],
            "modified_preconditions": {"pick": ["(not (locked ?i))"]},
        },
    )
    fd_stub = _StubFD(actions=[
        "(unlock screwdriver)",
        "(pick screwdriver source_screwdriver)",
        "(place screwdriver workbench)",
    ])
    pipeline = Pipeline(
        vlm=_StubVLM(plan), fd_planner=fd_stub, domains_dir=DOMAINS_DIR
    )
    result = pipeline.run("pick locked screwdriver", images=[], vlm_plan=plan)

    _section("VLM ENRICHMENT SUGGESTIONS", (
        f"  predicates: {plan.domain_additions['new_predicates']}\n"
        f"  actions:    {[a['name'] for a in plan.domain_additions['new_actions']]}\n"
        f"  modified:   {list(plan.domain_additions['modified_preconditions'].keys())}"
    ))
    _section("ENRICHMENT APPLIED", "\n".join(
        f"  ✓ {a}" for a in result.enrichment_result.additions_applied
    ))
    _section("PDDL PROBLEM", result.pddl_problem)
    _section("FAST DOWNWARD PLAN (stub)", "\n".join(f"  {a}" for a in result.pddl_actions))
    _section("NORMALIZED PRIMITIVES", "\n".join(
        f"  {i+1}. {p.name}({p.args})" for i, p in enumerate(result.primitives)
    ))

    assert result.success
    assert "(:action unlock" in result.enrichment_result.domain_text
    assert len(result.primitives) == 3


def test_showcase_failure_modes(capsys):
    """
    Showcase: what the pipeline returns in each failure scenario.
    """
    cases = [
        ("VLM not loaded",
         Pipeline(vlm=None, fd_planner=_StubFD([]), domains_dir=DOMAINS_DIR),
         None),
        ("Unknown domain template",
         _make_pipeline(plan=_make_plan(template="does_not_exist")),
         _make_plan(template="does_not_exist")),
        ("FD finds no plan (unsolvable)",
         _make_pipeline(fd_actions=None, repair_retries=1),
         _make_plan()),
    ]

    print(f"\n{'═'*62}")
    print("  PIPELINE FAILURE MODES")
    print(f"{'═'*62}")
    for label, pipeline, plan in cases:
        result = pipeline.run("test", images=[], vlm_plan=plan)
        status = "✗ FAILED" if not result.success else "✓ OK"
        print(f"  {status}  [{result.failure_stage}]  {label}")
        print(f"          error: {result.error}")
    print(f"{'═'*62}")


# ── helper ────────────────────────────────────────────────────────────────────

def _section(title: str, content: str) -> None:
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")
    print(content)
