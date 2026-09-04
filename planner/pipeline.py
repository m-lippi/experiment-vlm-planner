"""
Pipeline — pure Python, no ROS2 dependency.

Coordinates the full planning sequence:
  VLM → DomainEnricher → ProblemGenerator → FastDownward → plan_parser

Keeping this class ROS2-free makes it testable standalone and reusable
outside the robot (e.g. in a web demo or offline batch evaluation).

The ROS2 Orchestrator node is a thin wrapper that calls Pipeline.run()
and dispatches the resulting PrimitiveCall list to MoveIt / Nav2.

Repair loop (Phase D):
  If FastDownward returns no plan, the pipeline classifies the failure
  (syntax error vs. unsolvable) and retries up to `repair_retries` times.
  Full LLM-based repair is marked TODO — for Phase 1 the loop just retries
  without modification and reports a clear failure reason on exhaustion.
"""

from __future__ import annotations

import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from planner.domain_enricher import DomainEnricher, EnrichmentResult
from planner.fast_downward import FastDownwardPlanner
from planner.plan_parser import parse_plan, normalize_to_primitives, PrimitiveCall
from planner.problem_generator import generate_problem, DOMAIN_TEMPLATE_TO_NAME
from vlm.planner import VLMPlan, VLMPlanner


# ── Domain library ────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).parent.parent
_DOMAINS_DIR = _REPO_ROOT / "pddl" / "domains"

DOMAIN_TEMPLATE_FILES: dict[str, str] = {
    "manipulation_base":       "manipulation_base.pddl",
    "manipulation_stacking":   "manipulation_stacking.pddl",
    "containers_manipulation": "containers_manipulation.pddl",
    "navigation_manipulation": "navigation_manipulation.pddl",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Full output of Pipeline.run(). Intermediate artifacts are included for
    debugging and offline analysis; failure_stage locates where the run stopped.

    failure_stage values: "vlm" | "enrichment" | "planning" | "repair_exhausted"
    """
    success:            bool
    primitives:         list[PrimitiveCall]   # ready for robot dispatch

    # Intermediate artifacts (useful for debugging and thesis analysis)
    vlm_plan:           VLMPlan | None         = None
    enrichment_result:  EnrichmentResult | None = None
    pddl_problem:       str                    = ""
    pddl_actions:       list[str] | None       = None  # raw Fast Downward output

    # Failure info
    error:              str | None             = None
    repair_attempts:    int                    = 0
    failure_stage:      str | None             = None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Stateless planning pipeline. Create once at startup; call run() per task.

    Args:
        vlm:             VLMPlanner instance (lazy-loaded if None).
        enricher:        DomainEnricher instance.
        fd_planner:      FastDownwardPlanner instance.
        domains_dir:     Directory containing the four domain template files.
        repair_retries:  Max retries when Fast Downward fails (default 3).
    """

    def __init__(
        self,
        vlm:            VLMPlanner | None        = None,
        enricher:       DomainEnricher | None    = None,
        fd_planner:     FastDownwardPlanner | None = None,
        domains_dir:    Path | str | None        = None,
        repair_retries: int                      = 3,
    ) -> None:
        self._vlm            = vlm
        self._enricher       = enricher       or DomainEnricher()
        self._fd_planner     = fd_planner     or FastDownwardPlanner()
        self._domains_dir    = Path(domains_dir) if domains_dir else _DOMAINS_DIR
        self._repair_retries = repair_retries

    # ── Public API ────────────────────────────────────────────────────────────

    def load_vlm(self) -> None:
        """Load VLM weights. Call once at startup (expensive — GPU required)."""
        if self._vlm is None:
            self._vlm = VLMPlanner()
        self._vlm.load()

    def run(
        self,
        command:    str,
        images:     list,
        vlm_plan:   VLMPlan | None = None,
    ) -> PipelineResult:
        """
        Execute the full pipeline for a single task.

        Args:
            command:   Natural language task description.
            images:    Scene images (file paths or PIL Images).
            vlm_plan:  Pre-computed VLMPlan (skips VLM inference — useful for
                       dry-run, testing, or when plan is provided externally).

        Returns:
            PipelineResult with success flag, primitives, and all intermediates.
        """
        # ── Stage 1: VLM inference ────────────────────────────────────────────
        if vlm_plan is None:
            if self._vlm is None:
                return PipelineResult(
                    success=False,
                    primitives=[],
                    error="VLM not loaded — call load_vlm() first",
                    failure_stage="vlm",
                )
            try:
                vlm_plan = self._vlm.plan(command, images)
            except Exception as exc:
                return PipelineResult(
                    success=False,
                    primitives=[],
                    error=f"VLM inference failed: {exc}",
                    failure_stage="vlm",
                )

        # ── Stage 2: Domain selection + enrichment ────────────────────────────
        try:
            domain_path = self._select_domain(vlm_plan.domain_template)
            base_domain = domain_path.read_text()
        except FileNotFoundError as exc:
            return PipelineResult(
                success=False,
                primitives=[],
                vlm_plan=vlm_plan,
                error=f"Domain template not found: {exc}",
                failure_stage="enrichment",
            )

        additions = vlm_plan.to_domain_additions()
        enrichment = self._enricher.enrich(base_domain, additions)
        if not enrichment.is_valid:
            return PipelineResult(
                success=False,
                primitives=[],
                vlm_plan=vlm_plan,
                enrichment_result=enrichment,
                error=f"Domain enrichment invalid: {enrichment.errors}",
                failure_stage="enrichment",
            )

        # ── Stage 3: PDDL problem generation ─────────────────────────────────
        pddl_problem = generate_problem(vlm_plan)

        # ── Stage 4: Fast Downward + repair loop ──────────────────────────────
        pddl_actions, repair_attempts, fd_error = self._plan_with_repair(
            enrichment.domain_text, pddl_problem
        )

        if pddl_actions is None:
            stage = "repair_exhausted" if repair_attempts > 0 else "planning"
            return PipelineResult(
                success=False,
                primitives=[],
                vlm_plan=vlm_plan,
                enrichment_result=enrichment,
                pddl_problem=pddl_problem,
                pddl_actions=None,
                error=fd_error or "Fast Downward found no valid plan",
                repair_attempts=repair_attempts,
                failure_stage=stage,
            )

        # ── Stage 5: Parse + normalize ────────────────────────────────────────
        primitives = normalize_to_primitives(parse_plan(pddl_actions))

        return PipelineResult(
            success=True,
            primitives=primitives,
            vlm_plan=vlm_plan,
            enrichment_result=enrichment,
            pddl_problem=pddl_problem,
            pddl_actions=pddl_actions,
            repair_attempts=repair_attempts,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select_domain(self, template_name: str) -> Path:
        """Resolve domain template name → .pddl file path."""
        if template_name not in DOMAIN_TEMPLATE_FILES:
            raise FileNotFoundError(
                f"Unknown domain template '{template_name}'. "
                f"Available templates: {list(DOMAIN_TEMPLATE_FILES)}"
            )
        path = self._domains_dir / DOMAIN_TEMPLATE_FILES[template_name]
        if not path.exists():
            raise FileNotFoundError(
                f"Domain file '{path}' not found. "
                f"Available templates: {list(DOMAIN_TEMPLATE_FILES)}"
            )
        return path

    def _plan_with_repair(
        self,
        domain_text:   str,
        problem_text:  str,
    ) -> tuple[list[str] | None, int, str | None]:
        """
        Try Fast Downward up to repair_retries + 1 times.

        Returns:
            (pddl_actions, attempts_made, last_error)
            pddl_actions is None if all attempts failed.

        TODO (Phase 2): on failure, call VLM repair with error context instead
        of just retrying the same problem unchanged.
        """
        last_error: str | None = None

        for attempt in range(self._repair_retries + 1):
            try:
                pddl_actions = self._fd_planner.solve_from_strings(
                    domain_text, problem_text
                )
                if pddl_actions is not None:
                    return pddl_actions, attempt, None

                last_error = "unsolvable — no plan exists for the given problem"

            except RuntimeError as exc:
                last_error = str(exc)

            # TODO Phase 2: classify error (syntax vs. unsolvable) and call
            # VLM repair with the error message before the next attempt.
            if attempt < self._repair_retries:
                pass  # placeholder for LLM-based repair

        return None, self._repair_retries, last_error
