#!/usr/bin/env python3
"""
eval_enrichment.py — Stress test for VLM domain enrichment capability.

For each test task the script runs VLM planning, validates PDDL enrichment,
and optionally runs FastDownward.  Results are saved in a timestamped run
directory with per-task subfolders and an HTML report.

Usage:
    source .venv/bin/activate
    python3 scripts/eval_enrichment.py --image data/scene_overview.png --n-runs 3

Output structure:
    data/eval_runs/<timestamp>_eval/
        report.html
        summary_metrics.json
        config.json
        scene.png
        tasks/
            001_pour_bottle_glass_run0/
                metadata.json
                plan_vlm.json
                domain_enriched.pddl
                problem.pddl
                plan_fd.json
            ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


# ─────────────────────────────────────────────────────────────────────────────
#  Test suite
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    task: str
    needs_enrichment: bool
    expected_primitive: str | None
    category: str                       # "enrichment" | "standard" | "novel"
    group: str = ""                     # "A" | "B" | "C" | "D" | "E" | "F"
    expected_grasp_mode: str | None = None  # "top_down" | "side" | "handle"


GENERIC_SUITE: list[TestCase] = [
    # ── ENRICHMENT: pour ────────────────────────────────────────────────────
    TestCase("pour the bottle into the glass",
             True,  "pour",  "enrichment"),
    TestCase("pour from the bottle into the mug",
             True,  "pour",  "enrichment"),
    TestCase("pour water from the bottle into both glasses",
             True,  "pour",  "enrichment"),
    TestCase("fill the glass by pouring from the bottle until it is half full",
             True,  "pour",  "enrichment"),
    TestCase("pour the contents of the can into the bowl",
             True,  "pour",  "enrichment"),
    TestCase("transfer the liquid from the mug into the glass",
             True,  "pour",  "enrichment"),

    # ── ENRICHMENT: stir ────────────────────────────────────────────────────
    TestCase("pick the spoon and stir the contents of the mug",
             True,  "stir",  "enrichment"),
    TestCase("use the spoon to mix the liquid in the glass",
             True,  "stir",  "enrichment"),
    TestCase("stir the bowl contents after picking up the spoon",
             True,  "stir",  "enrichment"),
    TestCase("pick the spoon, stir the mug, then place the spoon on the tray",
             True,  "stir",  "enrichment"),

    # ── ENRICHMENT: cut ─────────────────────────────────────────────────────
    TestCase("use the knife to cut on the cutting board",
             True,  "cut",   "enrichment"),
    TestCase("pick up the knife and make a cut on the cutting board",
             True,  "cut",   "enrichment"),
    TestCase("cut the food on the cutting board using the knife",
             True,  "cut",   "enrichment"),

    # ── ENRICHMENT: tilt ────────────────────────────────────────────────────
    TestCase("tilt the bottle to pour its contents into the glass",
             True,  "tilt",  "enrichment"),
    TestCase("pick the bottle and tilt it at 45 degrees over the bowl",
             True,  "tilt",  "enrichment"),

    # ── ENRICHMENT: multi-primitive combos ──────────────────────────────────
    TestCase("pick the bottle, pour it into the glass, then stir with the spoon",
             True,  "pour",  "enrichment"),
    TestCase("pour the bottle into the mug and stir the contents with the spoon",
             True,  "pour",  "enrichment"),
    TestCase("cut on the cutting board then place the plate next to it",
             True,  "cut",   "enrichment"),
    TestCase("pour the bottle into the glass, then move the glass onto the tray",
             True,  "pour",  "enrichment"),

    # ── NOVEL: VLM must invent the PDDL action from scratch ─────────────────
    TestCase("pick the bottle and tilt it to check if there is liquid inside",
             True,  "tilt",  "novel"),
    TestCase("shake the bottle gently before pouring it into the glass",
             True,  None,    "novel"),
    TestCase("measure the amount of liquid in the glass",
             True,  None,    "novel"),
    TestCase("press down on the plate to flatten it against the cutting board",
             True,  None,    "novel"),
    TestCase("squeeze the bottle to extract its contents into the bowl",
             True,  None,    "novel"),
    TestCase("flip the plate upside down and place it on the tray",
             True,  None,    "novel"),
    TestCase("inspect the bottom of the glass to check for cracks",
             True,  None,    "novel"),
    TestCase("rotate the bottle 180 degrees and tap it on the cutting board",
             True,  None,    "novel"),
    TestCase("scrape the residue from the cutting board into the bowl",
             True,  None,    "novel"),

    # ── STANDARD: pure pick/place (control group, no enrichment needed) ─────
    TestCase("pick the glass and place it on the tray",
             False, "pick",  "standard"),
    TestCase("pick the can and place it next to the plate",
             False, "pick",  "standard"),
    TestCase("move the glass and the cup onto the tray",
             False, "pick",  "standard"),
    TestCase("pick the knife and place it on the cutting board",
             False, "pick",  "standard"),
    TestCase("move the bottle to the other side of the counter",
             False, "pick",  "standard"),
    TestCase("pick up the mug and place it near the glass",
             False, "pick",  "standard"),
    TestCase("arrange the glass, the cup and the can in a row on the tray",
             False, "pick",  "standard"),
    TestCase("swap the positions of the bottle and the glass",
             False, "pick",  "standard"),
    TestCase("pick all items from the front row and place them on the cutting board",
             False, "pick",  "standard"),
    TestCase("move the spoon from its current position to the tray",
             False, "pick",  "standard"),
]

# Alias kept for backward compatibility
TEST_SUITE = GENERIC_SUITE

# ─────────────────────────────────────────────────────────────────────────────
#  Office scene test suite (real available objects)
#  Objects: tazza (cup), penna (pen), laptop, keyboard, mouse, forbici
#           (scissors), quaderno (notebook), bottiglia (bottle), smartphone
# ─────────────────────────────────────────────────────────────────────────────

GROUP_NAMES: dict[str, str] = {
    "A": "A_core_primitives",
    "B": "B_enrichment",
    "C": "C_grasp_mode",
    "D": "D_multi_object",
    "E": "E_templates",
    "F": "F_edge_cases",
    "":  "ungrouped",
}

OFFICE_SUITE: list[TestCase] = [
    # ── Group A — Core primitives (no enrichment, baseline) ─────────────────
    TestCase("pick the pen and place it next to the keyboard",
             False, "pick", "standard", group="A", expected_grasp_mode="top_down"),
    TestCase("put the smartphone on the notebook",
             False, "pick", "standard", group="A", expected_grasp_mode="top_down"),
    TestCase("move the mouse to the other side of the keyboard",
             False, "pick", "standard", group="A", expected_grasp_mode="top_down"),

    # ── Group B — Enrichment required (non-core verb) ────────────────────────
    TestCase("pick the bottle and pour the water in the cup",
             True,  "pour",  "enrichment", group="B", expected_grasp_mode="side"),
    TestCase("cut the paper on the desk with the scissors",
             True,  "cut",   "enrichment", group="B", expected_grasp_mode="handle"),
    TestCase("write on the notebook with the pen",
             True,  "write", "enrichment", group="B", expected_grasp_mode="handle"),
    TestCase("flip the notebook upside down on the desk",
             True,  "flip",  "enrichment", group="B", expected_grasp_mode="side"),
    TestCase("scan the barcode on the bottle with the smartphone",
             True,  "scan",  "enrichment", group="B"),

    # ── Group C — Grasp mode reasoning (both directions) ────────────────────
    TestCase("tilt the bottle to check if it has water",
             True,  "tilt",  "enrichment", group="C", expected_grasp_mode="side"),
    TestCase("pour from the bottle into the cup",
             True,  "pour",  "enrichment", group="C", expected_grasp_mode="side"),
    TestCase("pick up the cup and place it on the keyboard",
             False, "pick",  "standard",   group="C", expected_grasp_mode="top_down"),
    TestCase("pick up the pen and drop it into the cup",
             False, "pick",  "standard",   group="C", expected_grasp_mode="top_down"),

    # ── Group D — Multi-object / multi-step ─────────────────────────────────
    TestCase("put the pen and the mouse on the notebook",
             False, "pick", "standard", group="D"),
    TestCase("clear the keyboard: move the mouse and the smartphone away from it",
             False, "pick", "standard", group="D"),
    TestCase("pick the pen, place it on the notebook, then put the bottle on the desk",
             False, "pick", "standard", group="D"),

    # ── Group E — Template selection (stacking / containers) ────────────────
    TestCase("stack the notebook on top of the laptop",
             False, None, "standard", group="E"),
    TestCase("open the laptop and place the smartphone inside it",
             False, None, "standard", group="E"),

    # ── Group F — Edge cases ─────────────────────────────────────────────────
    TestCase("hand me the scissors",
             True,  None, "novel", group="F"),
    TestCase("make a phone call with the smartphone",
             False, None, "novel", group="F"),
    TestCase("put everything on the desk in order",
             False, None, "novel", group="F"),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Per-run result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    task: str
    run_idx: int
    task_idx: int
    category: str
    needs_enrichment: bool
    expected_primitive: str | None

    # VLM output
    domain_template: str = ""
    steps_count: int = 0
    steps_primitives: list[str] = field(default_factory=list)
    enrichment_triggered: bool = False
    n_new_predicates: int = 0
    n_new_actions: int = 0
    new_action_names: list[str] = field(default_factory=list)
    raw_domain_additions: dict = field(default_factory=dict)

    # Pipeline
    pddl_valid: bool = False
    enricher_applied: list[str] = field(default_factory=list)
    enricher_skipped: list[str] = field(default_factory=list)
    goal_extracted: bool = False
    goal_string: str = ""
    plan_found: bool = False
    plan_length: int = 0
    primitives_in_plan: list[str] = field(default_factory=list)
    correct_primitive_in_plan: bool = False

    # Timing & errors
    inference_time_s: float = 0.0
    error: str = ""          # pipeline error (NOT FD — FD unavailable is expected on host)
    fd_error: str = ""       # FastDownward specific error (non-fatal)

    # Model identification (set by _run_one_model)
    model_id: str = ""
    model_short: str = ""

    # Group / folder (set during run)
    group: str = ""
    expected_grasp_mode: str | None = None
    actual_grasp_mode: str | None = None
    grasp_mode_correct: bool = False
    task_folder: str = ""   # path relative to model_dir (includes group subdir)


# ─────────────────────────────────────────────────────────────────────────────
#  Run one evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_single(
    tc: TestCase,
    run_idx: int,
    task_idx: int,
    vlm,
    image_path: Path,
    fd_available: bool,
    run_dir: Path,
) -> RunResult:
    from planner.problem_generator import generate_problem
    from planner.domain_enricher import DomainEnricher, DomainAdditions

    result = RunResult(
        task=tc.task, run_idx=run_idx, task_idx=task_idx,
        category=tc.category, needs_enrichment=tc.needs_enrichment,
        expected_primitive=tc.expected_primitive,
        group=tc.group,
        expected_grasp_mode=tc.expected_grasp_mode,
    )

    # Folder for this task+run — organised by group when available
    slug = tc.task[:40].lower().replace(" ", "_").replace(",", "")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    folder_name = f"{task_idx:03d}_{slug}_run{run_idx}"
    group_label  = GROUP_NAMES.get(tc.group, "ungrouped")
    group_dir    = run_dir / group_label
    group_dir.mkdir(parents=True, exist_ok=True)
    task_dir = group_dir / folder_name
    task_dir.mkdir(parents=True, exist_ok=True)
    result.task_folder = f"{group_label}/{folder_name}"

    enriched_domain_text = ""
    problem_text = ""
    fd_plan_data = []

    try:
        # ── 1. VLM ────────────────────────────────────────────────────────
        t0 = time.time()
        plan = vlm.plan_remaining(tc.task, [str(image_path)], [])
        result.inference_time_s = time.time() - t0

        result.domain_template   = plan.domain_template
        result.steps_count       = len(plan.steps)
        result.steps_primitives  = [s.primitive for s in plan.steps]

        da = plan.domain_additions or {}
        result.raw_domain_additions = da
        new_preds = da.get("new_predicates", []) or []
        new_acts  = da.get("new_actions",    []) or []
        result.n_new_predicates      = len(new_preds)
        result.n_new_actions         = len(new_acts)
        result.new_action_names      = [a.get("name","?") for a in new_acts if isinstance(a, dict)]
        result.enrichment_triggered  = bool(new_preds or new_acts)

        # Extract grasp_mode from first pick step
        pick_steps = [s for s in plan.steps if s.primitive == "pick"]
        if pick_steps:
            result.actual_grasp_mode = pick_steps[0].args.get("grasp_mode", "top_down")
            if result.expected_grasp_mode:
                result.grasp_mode_correct = (
                    result.actual_grasp_mode == result.expected_grasp_mode
                )

        # Save VLM plan (structured) + raw output text
        (task_dir / "plan_vlm.json").write_text(
            json.dumps({
                "goal": plan.goal,
                "domain_template": plan.domain_template,
                "domain_additions": da,
                "steps": [{"primitive": s.primitive, "args": s.args} for s in plan.steps],
                "raw_output": plan.raw_output,      # full model text before JSON parsing
            }, indent=2, ensure_ascii=False)
        )
        # Also save raw text as plain file for easy reading/quoting in the thesis
        (task_dir / "raw_output.txt").write_text(plan.raw_output, encoding="utf-8")

        # ── 2. Domain enrichment ──────────────────────────────────────────
        domain_file = _REPO / "pddl" / "domains" / f"{plan.domain_template}.pddl"
        if not domain_file.exists():
            domain_file = _REPO / "pddl" / "domains" / "manipulation_base.pddl"

        domain_text = domain_file.read_text()
        enricher    = DomainEnricher()
        additions   = DomainAdditions(
            new_types             = da.get("new_types", []) or [],
            new_predicates        = new_preds,
            new_actions           = new_acts,
            modified_preconditions= da.get("modified_preconditions", {}) or {},
        )
        er = enricher.enrich(domain_text, additions)
        result.pddl_valid        = er.is_valid
        result.enricher_applied  = er.additions_applied
        result.enricher_skipped  = er.additions_skipped
        enriched_domain_text     = er.domain_text

        (task_dir / "domain_enriched.pddl").write_text(enriched_domain_text)

        # ── 3. Problem generation ─────────────────────────────────────────
        problem_text = generate_problem(plan)
        result.goal_extracted = "no explicit goal inferred" not in problem_text
        # Extract the actual goal fact (skip block headers and structural keywords)
        _SKIP_PREFIXES = ("(:goal", "(:objects", "(:init", "(:domain",
                          "(define", "(and", "(or")
        in_goal = False
        for line in problem_text.splitlines():
            s = line.strip()
            if "(:goal" in s:
                in_goal = True
                continue
            if in_goal and s.startswith("(") and not any(s.startswith(p) for p in _SKIP_PREFIXES):
                result.goal_string = s
                break

        (task_dir / "problem.pddl").write_text(problem_text)

        # ── 4. FastDownward (non-fatal: FD lives in Docker, not on host) ────
        if fd_available and er.is_valid:
            try:
                from planner.fast_downward import FastDownwardPlanner
                fd = FastDownwardPlanner()
                fd_plan = fd.solve(enriched_domain_text, problem_text)
                if fd_plan:
                    result.plan_found  = True
                    result.plan_length = len(fd_plan)
                    fd_plan_data = [{"primitive": s.primitive, "args": s.args} for s in fd_plan]
                    result.primitives_in_plan = [s.primitive for s in fd_plan]
            except Exception as e:
                result.fd_error = str(e)  # non-fatal: don't mark run as errored

        if fd_plan_data:
            (task_dir / "plan_fd.json").write_text(
                json.dumps(fd_plan_data, indent=2))

        # ── 5. Primitive check ────────────────────────────────────────────
        all_prims = result.primitives_in_plan or result.steps_primitives
        result.primitives_in_plan = all_prims
        if tc.expected_primitive:
            result.correct_primitive_in_plan = tc.expected_primitive in all_prims

    except Exception as e:
        result.error = str(e)

    # Save per-task metadata
    (task_dir / "metadata.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False))

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[RunResult]) -> dict:
    def _r(n, d): return round(n / d, 3) if d else None

    # fd_error is non-fatal — only pipeline errors exclude a run from metrics
    enr  = [r for r in results if r.needs_enrichment and not r.error]
    std  = [r for r in results if not r.needs_enrichment and not r.error]
    ok   = [r for r in results if not r.error]
    enr_triggered = [r for r in ok if r.enrichment_triggered]

    return {
        "total_runs":                    len(results),
        "error_rate":                    _r(sum(1 for r in results if r.error), len(results)),
        "fd_available_rate":             _r(sum(1 for r in ok if not r.fd_error), len(ok)) if ok else None,
        "enrichment_recall":             _r(sum(1 for r in enr  if r.enrichment_triggered), len(enr)),
        "enrichment_false_positive_rate":_r(sum(1 for r in std  if r.enrichment_triggered), len(std)),
        "pddl_validity_rate":            _r(sum(1 for r in enr_triggered if r.pddl_valid), len(enr_triggered)),
        "goal_extraction_rate":          _r(sum(1 for r in enr  if r.goal_extracted), len(enr)),
        "plan_found_rate_enrichment":    _r(sum(1 for r in enr  if r.plan_found), len(enr)),
        "plan_found_rate_standard":      _r(sum(1 for r in std  if r.plan_found), len(std)),
        "correct_primitive_rate":        _r(
            sum(1 for r in ok if r.correct_primitive_in_plan),
            sum(1 for r in ok if r.expected_primitive)),
        "avg_inference_time_s":          round(sum(r.inference_time_s for r in ok) / max(len(ok),1), 2),
        "avg_new_actions_per_enrichment":round(
            sum(r.n_new_actions for r in enr_triggered) / max(len(enr_triggered),1), 2),

        # ── Per-category breakdown ────────────────────────────────────────
        "by_category": {
            cat: {
                "n": len(grp := [r for r in ok if r.category == cat]),
                "enrichment_recall":   _r(sum(1 for r in grp if r.enrichment_triggered), len(grp)),
                "correct_primitive":   _r(sum(1 for r in grp if r.correct_primitive_in_plan), len(grp)),
                "goal_extracted":      _r(sum(1 for r in grp if r.goal_extracted), len(grp)),
                "avg_inference_time":  round(sum(r.inference_time_s for r in grp) / max(len(grp),1), 2),
            }
            for cat in ("enrichment", "novel", "standard")
        },

        # ── Per-group breakdown (A-F) ────────────────────────────────────
        "by_group": {
            grp_key: {
                "n": len(g := [r for r in ok if r.group == grp_key]),
                "enrichment_recall":   _r(sum(1 for r in g if r.enrichment_triggered), len(g)),
                "correct_primitive":   _r(
                    sum(1 for r in g if r.correct_primitive_in_plan),
                    sum(1 for r in g if r.expected_primitive)
                ),
                "grasp_mode_accuracy": _r(
                    sum(1 for r in g if r.grasp_mode_correct),
                    sum(1 for r in g if r.expected_grasp_mode is not None)
                ),
                "avg_inference_time":  round(sum(r.inference_time_s for r in g) / max(len(g), 1), 2),
            }
            for grp_key in sorted({r.group for r in ok} - {""})
        },

        # ── Per-expected-primitive breakdown ─────────────────────────────
        "by_primitive": {
            prim: {
                "n": len(grp := [r for r in ok if r.expected_primitive == prim]),
                "enrichment_recall": _r(sum(1 for r in grp if r.enrichment_triggered), len(grp)),
                "correct_primitive": _r(sum(1 for r in grp if r.correct_primitive_in_plan), len(grp)),
            }
            for prim in sorted({r.expected_primitive for r in ok if r.expected_primitive})
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
#  HTML report
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f0f2f5;
       color: #1a1a2e; font-size: 14px; }
.container { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; color: #16213e; }
.subtitle { color: #666; margin-bottom: 28px; font-size: 0.9rem; }
h2 { font-size: 1.15rem; font-weight: 600; margin: 24px 0 12px; color: #16213e; }

/* Metric cards */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px,1fr));
         gap: 14px; margin-bottom: 32px; }
.card { background: #fff; border-radius: 10px; padding: 16px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.card .label { font-size: 0.78rem; color: #888; text-transform: uppercase;
               letter-spacing: .05em; margin-bottom: 6px; }
.card .value { font-size: 1.6rem; font-weight: 700; }
.card .bar-wrap { background: #e8eaf0; border-radius: 4px; height: 6px;
                  margin-top: 8px; overflow: hidden; }
.card .bar { height: 100%; border-radius: 4px; transition: width .5s; }
.green  { color: #2e7d32; } .bar.green  { background: #4caf50; }
.yellow { color: #f57f17; } .bar.yellow { background: #ffc107; }
.red    { color: #c62828; } .bar.red    { background: #ef5350; }
.blue   { color: #1565c0; } .bar.blue   { background: #42a5f5; }
.gray   { color: #555; }

/* Scene image */
.scene-wrap { margin-bottom: 28px; }
.scene-wrap img { max-height: 320px; border-radius: 10px;
                  box-shadow: 0 2px 8px rgba(0,0,0,.15); }

/* Results table */
.results-table { width: 100%; border-collapse: collapse; background: #fff;
                 border-radius: 10px; overflow: hidden;
                 box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 32px; }
.results-table th { background: #16213e; color: #fff; padding: 10px 12px;
                    text-align: left; font-size: 0.78rem; text-transform: uppercase;
                    letter-spacing: .05em; }
.results-table td { padding: 9px 12px; border-bottom: 1px solid #eee;
                    vertical-align: top; }
.results-table tr:last-child td { border-bottom: none; }
.results-table tr:hover td { background: #f5f7ff; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
         font-size: 0.72rem; font-weight: 600; }
.b-enrich  { background: #e3f2fd; color: #1565c0; }
.b-standard{ background: #f3e5f5; color: #6a1b9a; }
.b-novel   { background: #fff3e0; color: #e65100; }
.ok   { color: #2e7d32; font-weight: 700; }
.fail { color: #c62828; font-weight: 700; }
.na   { color: #aaa; }

/* Detail block */
details { border: 1px solid #e0e0e0; border-radius: 8px; margin-bottom: 10px;
          overflow: hidden; }
summary { padding: 10px 14px; cursor: pointer; background: #fafafa;
          font-weight: 600; list-style: none; }
summary::before { content: '▶ '; font-size: .8em; }
details[open] summary::before { content: '▼ '; }
.detail-body { padding: 12px 14px; background: #fff; }
pre { background: #1e1e2e; color: #cdd6f4; padding: 12px; border-radius: 6px;
      font-size: 0.78rem; overflow-x: auto; white-space: pre-wrap; margin-top: 6px; }
.tag-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.tag { background: #e8eaf0; padding: 2px 8px; border-radius: 12px;
       font-size: 0.75rem; }

/* Footer */
footer { text-align: center; color: #aaa; font-size: 0.8rem;
         margin-top: 32px; padding-top: 16px; border-top: 1px solid #ddd; }
"""


def _pct_color(v: float | None, threshold_ok=0.7, threshold_warn=0.4) -> str:
    if v is None: return "gray"
    if v >= threshold_ok: return "green"
    if v >= threshold_warn: return "yellow"
    return "red"


def _metric_card(label: str, value, is_rate: bool = True, inverse: bool = False) -> str:
    if value is None:
        return f"""<div class="card"><div class="label">{label}</div>
        <div class="value gray">N/A</div></div>"""
    if is_rate:
        pct = int(value * 100)
        col = _pct_color(1 - value if inverse else value)
        bar_w = pct
        bar_w_inv = 100 - pct if inverse else pct
        return f"""<div class="card">
  <div class="label">{label}</div>
  <div class="value {col}">{pct}%</div>
  <div class="bar-wrap"><div class="bar {col}" style="width:{bar_w_inv}%"></div></div>
</div>"""
    return f"""<div class="card">
  <div class="label">{label}</div>
  <div class="value blue">{value}</div>
</div>"""


def _sym(val: bool | None, na: bool = False) -> str:
    if na or val is None: return '<span class="na">—</span>'
    return '<span class="ok">✓</span>' if val else '<span class="fail">✗</span>'


def _badge(cat: str) -> str:
    cls = {"enrichment": "b-enrich", "standard": "b-standard", "novel": "b-novel"}.get(cat, "")
    return f'<span class="badge {cls}">{cat}</span>'


def _task_row(r: RunResult, run_dir: Path) -> str:
    short = (r.task[:60] + "…") if len(r.task) > 63 else r.task
    enr_sym  = _sym(r.enrichment_triggered) if r.needs_enrichment else _sym(None, na=True)
    val_sym  = _sym(r.pddl_valid) if r.enrichment_triggered else _sym(None, na=True)
    goal_sym = _sym(r.goal_extracted) if r.needs_enrichment else _sym(None, na=True)
    prim_sym = _sym(r.correct_primitive_in_plan) if r.expected_primitive else _sym(None, na=True)
    plan_sym = _sym(r.plan_found)
    t        = f"{r.inference_time_s:.1f}s"
    err      = f'<br><span style="color:#c62828;font-size:.75rem">ERR: {r.error[:60]}</span>' if r.error else ""
    fd_warn  = f'<br><span style="color:#f57f17;font-size:.72rem">FD: {r.fd_error[:50]}</span>' if r.fd_error else ""
    err      = err + fd_warn
    actions  = ", ".join(r.new_action_names) if r.new_action_names else "—"
    primitives = ", ".join(r.steps_primitives) if r.steps_primitives else "—"

    # Per-task folder link (task_folder already includes group subdir)
    folder_href = r.task_folder
    gm_str = r.actual_grasp_mode or "—"
    gm_cls = ("ok" if r.grasp_mode_correct else ("fail" if r.expected_grasp_mode else "na"))

    grp_badge = f'<span style="font-size:.75rem;font-weight:700;color:#555">{r.group or "—"}</span>'
    return f"""<tr>
  <td><a href="{folder_href}/plan_vlm.json" target="_blank" style="color:#1565c0;text-decoration:none">{short}</a>{err}</td>
  <td>{grp_badge}</td>
  <td>{_badge(r.category)}</td>
  <td>{enr_sym}</td>
  <td>{val_sym}</td>
  <td>{goal_sym}</td>
  <td>{prim_sym}</td>
  <td><span class="{gm_cls}">{gm_str}</span></td>
  <td>{plan_sym}</td>
  <td>{t}</td>
  <td style="font-size:.8rem;color:#555">{actions}</td>
  <td style="font-size:.8rem;color:#555">{primitives}</td>
</tr>"""


def _task_detail(r: RunResult) -> str:
    da_json = json.dumps(r.raw_domain_additions, indent=2, ensure_ascii=False)
    applied = "\n".join(f"  ✓ {a}" for a in r.enricher_applied) or "  (none)"
    skipped = "\n".join(f"  ✗ {s}" for s in r.enricher_skipped) or "  (none)"
    steps   = "\n".join(f"  {i+1}. {s['primitive']}({s['args']})" for i, s in
                        enumerate([{"primitive": p, "args": {}} for p in r.steps_primitives]))
    gm_actual   = r.actual_grasp_mode or "—"
    gm_expected = r.expected_grasp_mode or "—"
    gm_ok = ("✓" if r.grasp_mode_correct else ("✗" if r.expected_grasp_mode else "—"))
    return f"""<details>
  <summary>{r.task[:80]} — run #{r.run_idx} (Group {r.group or "—"} / {r.category})</summary>
  <div class="detail-body">
    <p><strong>Domain template:</strong> {r.domain_template} &nbsp;
       <strong>Steps:</strong> {r.steps_count} &nbsp;
       <strong>Time:</strong> {r.inference_time_s:.2f}s &nbsp;
       <strong>Grasp mode:</strong> {gm_actual} (expected: {gm_expected}) {gm_ok}</p>

    <p style="margin-top:8px"><strong>VLM plan steps:</strong></p>
    <pre>{steps or "(none)"}</pre>

    <p style="margin-top:10px"><strong>Domain additions (raw VLM output):</strong></p>
    <pre>{da_json}</pre>

    <p style="margin-top:10px"><strong>Enricher applied:</strong></p>
    <pre style="background:#1b5e20;color:#a5d6a7">{applied}</pre>
    <p style="margin-top:6px"><strong>Enricher skipped:</strong></p>
    <pre style="background:#b71c1c;color:#ffcdd2">{skipped}</pre>

    <p style="margin-top:10px"><strong>Goal:</strong>
       {r.goal_string or ("<em>fallback (gripper-empty) — enrichment missing</em>" if r.needs_enrichment else "<em>standard goal</em>")}</p>
    {f'<p style="color:#c62828"><strong>Error:</strong> {r.error}</p>' if r.error else ""}
  </div>
</details>"""


def generate_html_report(
    results: list[RunResult],
    metrics: dict,
    run_dir: Path,
    config: dict,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Metric cards
    cards = "".join([
        _metric_card("Enrichment Recall",     metrics["enrichment_recall"]),
        _metric_card("PDDL Validity",         metrics["pddl_validity_rate"]),
        _metric_card("Goal Extraction",       metrics["goal_extraction_rate"]),
        _metric_card("Plan Found (enrich.)",  metrics["plan_found_rate_enrichment"]),
        _metric_card("Correct Primitive",     metrics["correct_primitive_rate"]),
        _metric_card("False Positive Rate",   metrics["enrichment_false_positive_rate"],
                     inverse=True),
        _metric_card("Avg Inference Time",    metrics["avg_inference_time_s"],
                     is_rate=False),
        _metric_card("Error Rate",            metrics["error_rate"], inverse=True),
    ])

    # Scene image (relative path)
    scene_html = ""
    if (run_dir / "scene.png").exists():
        scene_html = """<div class="scene-wrap">
  <h2>Input Scene</h2>
  <img src="scene.png" alt="Scene overview">
</div>"""

    # Results table rows
    rows = "".join(_task_row(r, run_dir) for r in results)

    # Detail blocks — grouped by group (A-F) when available, else by category
    has_groups = any(r.group for r in results)
    if has_groups:
        by_group: dict[str, list[RunResult]] = {}
        for r in results:
            key = r.group or ""
            by_group.setdefault(key, []).append(r)
        details_html = ""
        for grp_key in sorted(by_group.keys()):
            rs = by_group[grp_key]
            grp_label = GROUP_NAMES.get(grp_key, grp_key or "Ungrouped")
            details_html += f'<h2>Group {grp_label.replace("_", " ").title()}</h2>'
            details_html += "".join(_task_detail(r) for r in rs)
    else:
        by_cat: dict[str, list[RunResult]] = {"enrichment": [], "novel": [], "standard": []}
        for r in results:
            by_cat.setdefault(r.category, []).append(r)
        details_html = ""
        for cat, rs in by_cat.items():
            if not rs: continue
            details_html += f"<h2>{cat.title()} Tasks</h2>"
            details_html += "".join(_task_detail(r) for r in rs)

    # Config summary
    cfg_lines = "".join(f"<li><strong>{k}:</strong> {v}</li>"
                        for k, v in config.items())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Enrichment Eval — {config.get('timestamp','')}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">

  <h1>🔬 VLM Domain Enrichment — Evaluation Report</h1>
  <p class="subtitle">Generated {ts} &nbsp;|&nbsp;
     {metrics['total_runs']} runs &nbsp;|&nbsp;
     {len({r.task for r in results})} unique tasks &nbsp;|&nbsp;
     Image: {config.get('image_name','—')}</p>

  <h2>Summary Metrics</h2>
  <div class="cards">{cards}</div>

  {scene_html}

  <h2>Results per Run</h2>
  <table class="results-table">
    <thead><tr>
      <th>Task</th><th>Grp</th><th>Cat.</th>
      <th title="VLM added domain_additions">Enrich</th>
      <th title="DomainEnricher applied additions without errors">PDDL ✓</th>
      <th title="ProblemGenerator extracted non-fallback goal">Goal ✓</th>
      <th title="Expected primitive appeared in plan">Prim ✓</th>
      <th title="grasp_mode of first pick step (green=correct, red=wrong)">Grasp</th>
      <th title="FastDownward found a valid plan">Plan ✓</th>
      <th>Time</th>
      <th>New Actions</th>
      <th>VLM Steps</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <h2>Detailed Results</h2>
  {details_html}

  <h2>Run Configuration</h2>
  <ul style="padding-left:20px;line-height:2">{cfg_lines}</ul>

  <footer>VLM-RobotPlanner — Thesis experiment · eval_enrichment.py</footer>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(metrics: dict, n_tasks: int):
    W = 70
    print("\n" + "=" * W)
    print("  AGGREGATE METRICS")
    print("=" * W)
    pairs = [
        ("Enrichment recall (needs → triggered)",  metrics["enrichment_recall"]),
        ("PDDL validity (triggered → valid PDDL)", metrics["pddl_validity_rate"]),
        ("Goal extraction rate",                   metrics["goal_extraction_rate"]),
        ("Plan found — enrichment tasks",          metrics["plan_found_rate_enrichment"]),
        ("Plan found — standard tasks",            metrics["plan_found_rate_standard"]),
        ("Correct primitive in plan",              metrics["correct_primitive_rate"]),
        ("False positive enrichment rate",         metrics["enrichment_false_positive_rate"]),
        ("Avg inference time (s)",                 metrics["avg_inference_time_s"]),
        ("Error rate",                             metrics["error_rate"]),
    ]
    for label, val in pairs:
        if val is None:
            print(f"  {label:<45}  N/A")
            continue
        if isinstance(val, float) and val <= 1.0 and "time" not in label.lower():
            bar_len = int(val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {label:<45}  [{bar}] {val:.1%}")
        else:
            print(f"  {label:<45}  {val}")
    print("=" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_MODELS = [
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "OpenGVLab/InternVL2_5-8B",
]


def _run_one_model(
    model_id: str,
    suite: list[TestCase],
    n_runs: int,
    image_path: Path,
    fd_ok: bool,
    run_dir: Path,
    api_key: str | None = None,
) -> tuple[list[RunResult], dict]:
    """Load model, run all tasks, return (results, metrics)."""
    from vlm.planner import create_planner, model_short_name
    short = model_short_name(model_id)
    model_dir = run_dir / short
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tasks").mkdir(exist_ok=True)

    print(f"\n{'─'*70}")
    print(f"  MODEL: {short}  ({model_id})")
    print(f"{'─'*70}")
    is_api = "gemini" in model_id.lower()
    print(f"  {'Connecting to API' if is_api else 'Loading weights'}...", end="", flush=True)

    vlm = create_planner(model_id, api_key=api_key)
    vlm.load()
    print(f" done.")

    total = len(suite) * n_runs
    results: list[RunResult] = []
    task_counter = 0
    for tc in suite:
        task_counter += 1
        for run_idx in range(n_runs):
            done = len(results) + 1
            print(f"  [{done:3}/{total}] [{tc.category:10}] {tc.task[:50]}",
                  end="", flush=True)
            r = run_single(tc, run_idx, task_counter, vlm, image_path, fd_ok, model_dir)
            r.model_id    = model_id     # tag result with model
            r.model_short = short
            results.append(r)
            e_sym = "✓" if r.enrichment_triggered else "·"
            v_sym = "✓" if r.pddl_valid else ("·" if not r.enrichment_triggered else "✗")
            err   = " [ERR]" if r.error else ""
            print(f"  [{e_sym}{v_sym}] {r.inference_time_s:.1f}s{err}")

    metrics = compute_metrics(results)

    # Per-model JSON
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (model_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))

    # Per-model HTML (reuse existing generator)
    config_model = {"timestamp": "", "image_name": image_path.name,
                    "model": model_id, "n_runs": n_runs, "n_tasks": len(suite)}
    html = generate_html_report(results, metrics, model_dir, config_model)
    (model_dir / "report.html").write_text(html, encoding="utf-8")

    print_summary(metrics, len(suite))
    return results, metrics


def _generate_comparison_report(
    all_results: dict[str, list[RunResult]],
    all_metrics: dict[str, dict],
    run_dir: Path,
    config: dict,
) -> str:
    """Generate an HTML report comparing all models side by side."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    models = list(all_metrics.keys())

    # Metric keys to compare
    metric_keys = [
        ("Enrichment Recall",     "enrichment_recall"),
        ("PDDL Validity",         "pddl_validity_rate"),
        ("Goal Extraction",       "goal_extraction_rate"),
        ("Correct Primitive",     "correct_primitive_rate"),
        ("False Positive Enrich.","enrichment_false_positive_rate"),
        ("Avg Inference (s)",     "avg_inference_time_s"),
        ("Error Rate",            "error_rate"),
    ]

    def _pct(v, is_time=False, inverse=False):
        if v is None: return "N/A"
        if is_time:   return f"{v:.1f}s"
        return f"{v:.1%}"

    def _cell_color(key, v):
        if v is None: return "#f5f5f5"
        inverse_keys = {"enrichment_false_positive_rate", "error_rate", "avg_inference_time_s"}
        good = v < 0.2 if key in inverse_keys else v >= 0.8
        warn = (0.2 <= v < 0.5) if key in inverse_keys else (0.5 <= v < 0.8)
        if key == "avg_inference_time_s": return "#fff"
        if good: return "#e8f5e9"
        if warn: return "#fff9c4"
        return "#ffebee"

    # Header row
    th_models = "".join(f"<th>{m}</th>" for m in models)
    # Metric rows
    rows = ""
    for label, key in metric_keys:
        is_time = "time" in key
        cells = "".join(
            f'<td style="background:{_cell_color(key, all_metrics[m].get(key))};text-align:center">'
            f'{_pct(all_metrics[m].get(key), is_time=is_time)}</td>'
            for m in models
        )
        rows += f"<tr><td><strong>{label}</strong></td>{cells}</tr>\n"

    # Winner annotation per metric
    winner_row = "<tr><td><em>Best</em></td>"
    for key_label, key in metric_keys:
        inverse_keys = {"enrichment_false_positive_rate", "error_rate", "avg_inference_time_s"}
        vals = {m: all_metrics[m].get(key) for m in models}
        valid = {m: v for m, v in vals.items() if v is not None}
        if valid:
            best = min(valid, key=lambda m: valid[m]) if key in inverse_keys \
                   else max(valid, key=lambda m: valid[m])
            winner_row += f'<td style="text-align:center;font-weight:700;color:#2e7d32" colspan="1">{best}</td>' \
                          if len(models) == 1 else ""
    winner_row += "</tr>"

    # Per-task comparison table
    task_rows = ""
    all_tasks = sorted({r.task for results in all_results.values() for r in results})
    for task in all_tasks:
        short_task = (task[:55] + "…") if len(task) > 58 else task
        cells = ""
        for m in models:
            rs = [r for r in all_results[m] if r.task == task]
            if not rs:
                cells += "<td>—</td>"
                continue
            enr = sum(1 for r in rs if r.enrichment_triggered) / len(rs)
            val = sum(1 for r in rs if r.pddl_valid and r.enrichment_triggered) / max(sum(1 for r in rs if r.enrichment_triggered), 1)
            prim = sum(1 for r in rs if r.correct_primitive_in_plan) / len(rs)
            cat  = rs[0].category
            cat_badge = {"enrichment":"#e3f2fd","standard":"#f3e5f5","novel":"#fff3e0"}.get(cat,"#eee")
            cells += (f'<td style="font-size:.8rem;text-align:center">'
                      f'E:{enr:.0%} V:{val:.0%} P:{prim:.0%}</td>')
        task_rows += f"<tr><td>{short_task}</td>{cells}</tr>\n"

    scene_html = ""
    if (run_dir / "scene.png").exists():
        scene_html = '<img src="scene.png" alt="Scene" style="max-height:260px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.15);margin-bottom:24px">'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Multi-Model Enrichment Comparison</title>
<style>
body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f0f2f5;color:#1a1a2e;font-size:14px;margin:0}}
.wrap{{max-width:1100px;margin:0 auto;padding:28px}}
h1{{font-size:1.7rem;font-weight:700;margin-bottom:4px}}
.sub{{color:#888;font-size:.9rem;margin-bottom:28px}}
h2{{font-size:1.1rem;font-weight:600;margin:24px 0 10px;color:#16213e}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;
       overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:28px}}
th{{background:#16213e;color:#fff;padding:10px 14px;font-size:.8rem;text-transform:uppercase}}
td{{padding:9px 14px;border-bottom:1px solid #eee}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:#f8f9ff}}
footer{{text-align:center;color:#aaa;font-size:.8rem;margin-top:32px;padding-top:16px;border-top:1px solid #ddd}}
</style>
</head>
<body>
<div class="wrap">
<h1>🔬 Multi-Model Enrichment Comparison</h1>
<p class="sub">Generated {ts} &nbsp;|&nbsp; {len(models)} models &nbsp;|&nbsp;
{len(all_tasks)} tasks &nbsp;|&nbsp; {config.get('n_runs',1)} run(s) each</p>

{scene_html}

<h2>Aggregate Metrics</h2>
<table>
<thead><tr><th>Metric</th>{th_models}</tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>Per-Task Breakdown  <small style="font-weight:400;color:#888">(E=Enrichment, V=PDDL Valid, P=Correct Primitive)</small></h2>
<table>
<thead><tr><th>Task</th>{th_models}</tr></thead>
<tbody>{task_rows}</tbody>
</table>

<h2>Individual Reports</h2>
<ul style="padding-left:20px;line-height:2.2">
{"".join(f'<li><a href="{m}/report.html">{m}</a></li>' for m in models)}
</ul>

<footer>VLM-RobotPlanner · eval_enrichment.py multi-model comparison</footer>
</div>
</body>
</html>"""


def main():
    ap = argparse.ArgumentParser(description="Evaluate VLM domain enrichment (single or multi-model)")
    ap.add_argument("--image",    default="data/scene_overview.png")
    ap.add_argument("--n-runs",   type=int, default=1)
    ap.add_argument("--category", choices=["enrichment","standard","novel","all"], default="all")
    ap.add_argument("--tasks",    nargs="+", default=None)
    ap.add_argument("--no-fastdownward", action="store_true")
    ap.add_argument(
        "--output-dir", "-o", default=None, metavar="DIR",
        help=(
            "Directory where the timestamped run folder is created. "
            "Default: data/eval_runs/ inside the repo. "
            "Example: --output-dir /home/user/thesis_experiments"
        ),
    )
    ap.add_argument(
        "--suite", choices=["office", "generic", "all"], default="office",
        help=(
            "Test suite to use. "
            "'office' = real scene objects (tazza, penna, laptop, ...). "
            "'generic' = kitchen objects (glass, bottle, spoon, ...). "
            "'all' = both suites combined. Default: office"
        ),
    )
    ap.add_argument(
        "--group", nargs="+", default=None,
        metavar="GRP",
        help="Run only these groups (e.g. --group A C). Default: all groups.",
    )
    ap.add_argument(
        "--api-key", default=None, metavar="KEY",
        help="API key for cloud models (Gemini). Can also be set via GOOGLE_API_KEY env var."
    )
    ap.add_argument(
        "--models", nargs="+",
        default=["Qwen/Qwen3-VL-8B-Instruct"],
        metavar="MODEL_ID",
        help=(
            "One or more model IDs to evaluate.\n"
            "HuggingFace: Qwen3-VL-*, Qwen2.5-VL-*, InternVL2_5-*\n"
            "Google: gemini-2.0-flash, gemini-1.5-pro, gemini-2.5-pro-preview-06-05\n"
            f"Default: Qwen/Qwen3-VL-8B-Instruct\n"
            f"Full comparison: --models {' '.join(_DEFAULT_MODELS)}"
        ),
    )
    args = ap.parse_args()

    image_path = _REPO / args.image
    if not image_path.exists():
        sys.exit(f"[ERROR] Image not found: {image_path}")

    # Select suite
    if args.suite == "office":
        base_suite = OFFICE_SUITE
    elif args.suite == "generic":
        base_suite = GENERIC_SUITE
    else:
        base_suite = OFFICE_SUITE + GENERIC_SUITE

    # Filter suite
    suite = base_suite
    if args.category != "all":
        suite = [tc for tc in suite if tc.category == args.category]
    if args.group:
        groups = {g.upper() for g in args.group}
        suite = [tc for tc in suite if tc.group.upper() in groups]
    if args.tasks:
        suite = [tc for tc in suite if any(t.lower() in tc.task.lower() for t in args.tasks)]
    if not suite:
        sys.exit("[ERROR] No matching test cases.")

    # FastDownward
    fd_ok = not args.no_fastdownward
    if fd_ok:
        try:
            from planner.fast_downward import FastDownwardPlanner
            FastDownwardPlanner()
        except Exception:
            print("[WARN] FastDownward not available — skipping plan validation.")
            fd_ok = False

    # Run directory
    ts_str   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    multi    = len(args.models) > 1
    run_name = f"{ts_str}_{'multi_model' if multi else args.suite}_eval"
    base_dir = Path(args.output_dir) if args.output_dir else (_REPO / "data" / "eval_runs")
    run_dir  = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, run_dir / "scene.png")

    print(f"[INFO] Run directory: {run_dir.relative_to(_REPO)}")
    print(f"[INFO] Models: {args.models}")
    print(f"[INFO] {len(suite)} tasks × {args.n_runs} run(s) × {len(args.models)} model(s) "
          f"= {len(suite) * args.n_runs * len(args.models)} total evaluations")

    # ── Run each model ───────────────────────────────────────────────────────
    all_results: dict[str, list[RunResult]] = {}
    all_metrics: dict[str, dict]            = {}

    for model_id in args.models:
        from vlm.planner import model_short_name
        short = model_short_name(model_id)
        results, metrics = _run_one_model(
            model_id, suite, args.n_runs, image_path, fd_ok, run_dir,
            api_key=args.api_key,
        )
        all_results[short] = results
        all_metrics[short] = metrics

    # ── Global summary files ─────────────────────────────────────────────────
    config = {
        "timestamp":    ts_str,
        "image_name":   image_path.name,
        "models":       args.models,
        "n_runs":       args.n_runs,
        "n_tasks":      len(suite),
        "suite":        args.suite,
        "groups":       args.group or "all",
        "categories":   args.category,
        "fastdownward": fd_ok,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    # ── Comparison or single report ──────────────────────────────────────────
    if multi:
        html = _generate_comparison_report(all_results, all_metrics, run_dir, config)
        report_path = run_dir / "comparison_report.html"
    else:
        # Single model: use existing per-model report as main report
        short = model_short_name(args.models[0])
        report_path = run_dir / "report.html"
        shutil.copy2(run_dir / short / "report.html", report_path)
        html = report_path.read_text(encoding="utf-8")

    report_path.write_text(html, encoding="utf-8")

    print(f"\n[OK] Run directory : {run_dir.relative_to(_REPO)}")
    print(f"[OK] Report        : {report_path.relative_to(_REPO)}")
    print(f"[OK] Open report   : xdg-open {report_path}\n")


if __name__ == "__main__":
    main()
