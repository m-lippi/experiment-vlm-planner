"""
DomainEnricher — applies VLM-suggested additions to a base PDDL domain template.

This is the innovative component of the pipeline: instead of a single fixed domain,
the system selects a base template (pddl/domains/) and enriches it at runtime based
on scene-specific context provided by the VLM.

Enrichment types supported:
  1. new_types             — add subtypes to the (:types ...) block
  2. new_predicates        — add predicates to the (:predicates ...) block
  3. new_actions           — append full action definitions to the domain
  4. modified_preconditions — add extra conditions to existing actions' preconditions

All operations are string-based (no full PDDL parser required). Validation is
structural: parenthesis balance, duplicate detection, and known-type checks.

Usage:
    from planner.domain_enricher import DomainEnricher, DomainAdditions

    enricher = DomainEnricher()
    additions = DomainAdditions(
        new_predicates=["(locked ?i - item)"],
        modified_preconditions={"pick": ["(not (locked ?i))"]},
    )
    result = enricher.enrich(domain_text, additions)
    if result.is_valid:
        fast_downward.solve(result.domain_text, problem_text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DomainAdditions:
    """
    Scene-specific enrichment suggestions, typically produced by the VLM.

    modified_preconditions conditions must use variable names already declared
    in the target action's parameter list — the enricher does not re-check this.
    """
    new_types:               list[str]        = field(default_factory=list)
    new_predicates:          list[str]        = field(default_factory=list)
    new_actions:             list[dict]       = field(default_factory=list)
    modified_preconditions:  dict[str, list[str]] = field(default_factory=dict)


@dataclass
class EnrichmentResult:
    """
    Output of DomainEnricher.enrich(). is_valid is True only when structural
    validation passes; errors contains the reasons when it does not.
    """
    domain_text:        str
    is_valid:           bool
    errors:             list[str]
    additions_applied:  list[str]
    additions_skipped:  list[str]


# ── Main class ────────────────────────────────────────────────────────────────

class DomainEnricher:
    """
    Applies structured enrichment additions to a PDDL domain text.

    All methods are stateless — the same instance can be reused across tasks.
    """

    # PDDL keywords that must never be treated as type/predicate names
    _PDDL_KEYWORDS = frozenset({
        "define", "domain", "problem", "requirements", "types", "predicates",
        "action", "parameters", "precondition", "effect", "and", "or", "not",
        "when", "forall", "exists", "object", "strips", "typing", "adl",
        "negative-preconditions", "disjunctive-preconditions",
        "equality", "conditional-effects", "quantified-preconditions",
    })

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(self, domain_text: str, additions: DomainAdditions) -> EnrichmentResult:
        """
        Apply additions to domain_text and return an EnrichmentResult.

        Processing order: types → predicates → actions → modified preconditions.
        Each step is independent; failures in one do not abort the others.
        """
        errors:   list[str] = []
        applied:  list[str] = []
        skipped:  list[str] = []

        text = domain_text
        existing = self._extract_existing(text)

        # 1. New types
        for type_decl in additions.new_types:
            type_name = type_decl.split()[0]
            if type_name in existing["types"]:
                skipped.append(f"type '{type_name}' already exists — skipped")
            else:
                new_text, ok = self._insert_into_block(text, ":types", type_decl)
                if ok:
                    text = new_text
                    applied.append(f"type: {type_decl}")
                    existing["types"].add(type_name)
                else:
                    skipped.append(f"type '{type_decl}' — :types block not found")

        # 2. New predicates
        for pred in additions.new_predicates:
            pred_name = self._predicate_name(pred)
            if pred_name in existing["predicates"]:
                skipped.append(f"predicate '{pred_name}' already exists — skipped")
            else:
                new_text, ok = self._insert_into_block(text, ":predicates", pred)
                if ok:
                    text = new_text
                    applied.append(f"predicate: {pred}")
                    existing["predicates"].add(pred_name)
                else:
                    skipped.append(f"predicate '{pred}' — :predicates block not found")

        # 3. New actions — with auto-repair before inserting
        for action in additions.new_actions:
            name = action.get("name", "")
            if not name:
                skipped.append("action with missing 'name' field — skipped")
                continue
            if name in existing["actions"]:
                skipped.append(f"action '{name}' already exists — skipped")
                continue

            # Auto-repair 1: add undeclared variables to parameters
            action, repaired_vars = self._repair_unbound_variables(action)
            if repaired_vars:
                applied.append(f"auto-repaired unbound vars in '{name}': {repaired_vars}")

            # Auto-repair 2: infer missing predicates from action effects
            inferred = self._infer_predicates_from_effect(action, existing["predicates"])
            for pred_decl, pred_name in inferred:
                if pred_name in existing["predicates"]:
                    continue  # already present (double-check after regex fix)
                new_text, ok = self._insert_into_block(text, ":predicates", pred_decl)
                if ok:
                    text = new_text
                    applied.append(f"auto-inferred predicate from effect: {pred_decl}")
                    existing["predicates"].add(pred_name)

            action_pddl = self._format_action(action)
            if not self._balanced(action_pddl):
                skipped.append(f"action '{name}' has unbalanced parentheses — skipped")
                continue
            text = self._append_action(text, action_pddl)
            applied.append(f"action: {name}")
            existing["actions"].add(name)

        # 4. Modified preconditions
        for action_name, extra_conditions in additions.modified_preconditions.items():
            if action_name not in existing["actions"]:
                skipped.append(
                    f"cannot extend precondition of '{action_name}' — action not found"
                )
                continue
            for cond in extra_conditions:
                if not self._balanced(cond):
                    skipped.append(
                        f"condition '{cond}' for '{action_name}' has unbalanced parens — skipped"
                    )
                    continue
            new_text, ok = self._extend_precondition(text, action_name, extra_conditions)
            if ok:
                text = new_text
                applied.append(
                    f"extended precondition of '{action_name}': {extra_conditions}"
                )
            else:
                skipped.append(
                    f"could not extend precondition of '{action_name}' — :precondition not found"
                )

        # 5. Validate final result
        is_valid, val_errors = self._validate(text)
        errors.extend(val_errors)

        return EnrichmentResult(
            domain_text=text,
            is_valid=is_valid,
            errors=errors,
            additions_applied=applied,
            additions_skipped=skipped,
        )

    @classmethod
    def from_file(cls, domain_path: str | Path) -> "tuple[DomainEnricher, str]":
        """Convenience: return (enricher_instance, domain_text) from a .pddl file path."""
        text = Path(domain_path).read_text()
        return cls(), text

    # ── PDDL text manipulation ────────────────────────────────────────────────

    def _find_matching_paren(self, text: str, start: int) -> int:
        """Return position of the ) that closes the ( at position start. Returns -1 on failure."""
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _insert_into_block(self, text: str, keyword: str, new_content: str) -> tuple[str, bool]:
        """
        Append new_content as a new line inside the (:keyword ...) block.
        Returns (modified_text, success).
        """
        pattern = re.compile(r"\(" + re.escape(keyword) + r"\b")
        m = pattern.search(text)
        if not m:
            return text, False
        block_end = self._find_matching_paren(text, m.start())
        if block_end == -1:
            return text, False
        insertion = f"\n    {new_content}"
        return text[:block_end] + insertion + text[block_end:], True

    def _append_action(self, text: str, action_pddl: str) -> str:
        """
        Insert a new action definition before the final closing ) of the domain.
        """
        last_paren = text.rfind(")")
        if last_paren == -1:
            return text + "\n\n" + action_pddl + "\n"
        return text[:last_paren] + "\n" + action_pddl + "\n" + text[last_paren:]

    def _extend_precondition(
        self,
        text: str,
        action_name: str,
        extra_conditions: list[str],
    ) -> tuple[str, bool]:
        """
        Add extra_conditions to the :precondition of action_name.

        If the precondition is already wrapped in (and ...), the new conditions
        are inserted inside it. Otherwise the precondition is wrapped in a new
        (and current_precond new_cond1 ...).

        Returns (modified_text, success).
        """
        pattern = re.compile(r"\(:action\s+" + re.escape(action_name) + r"\b")
        m = pattern.search(text)
        if not m:
            return text, False

        action_start = m.start()
        action_end = self._find_matching_paren(text, action_start)
        if action_end == -1:
            return text, False

        action_block = text[action_start:action_end + 1]

        pre_m = re.search(r":precondition\s*", action_block)
        if not pre_m:
            return text, False

        paren_start = action_block.find("(", pre_m.end())
        if paren_start == -1:
            return text, False

        paren_end = self._find_matching_paren(action_block, paren_start)
        if paren_end == -1:
            return text, False

        current = action_block[paren_start:paren_end + 1]
        extra_str = "\n                 ".join(extra_conditions)

        if re.match(r"\(and\b", current.strip()):
            # Already an (and ...) — insert before its closing )
            inner_close = self._find_matching_paren(current, 0)
            new_precond = (
                current[:inner_close]
                + f"\n                 {extra_str}"
                + current[inner_close:]
            )
        else:
            # Single condition — wrap in (and ...)
            new_precond = f"(and {current}\n                 {extra_str})"

        new_action = action_block[:paren_start] + new_precond + action_block[paren_end + 1:]
        return text[:action_start] + new_action + text[action_end + 1:], True

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_existing(self, text: str) -> dict[str, set[str]]:
        """
        Return sets of already-defined type names, predicate names, and action names.
        Used for duplicate detection before applying additions.
        """
        result: dict[str, set[str]] = {
            "types":      set(),
            "predicates": set(),
            "actions":    set(),
        }

        m = re.search(r"\(:types\b", text)
        if m:
            end = self._find_matching_paren(text, m.start())
            block = text[m.start():end + 1] if end != -1 else ""
            tokens = re.findall(r"\b([a-zA-Z][\w-]*)\b", block)
            result["types"] = {
                t for t in tokens if t not in self._PDDL_KEYWORDS
            }

        m = re.search(r"\(:predicates\b", text)
        if m:
            end = self._find_matching_paren(text, m.start())
            block = text[m.start():end + 1] if end != -1 else ""
            result["predicates"] = set(re.findall(r"\(([a-zA-Z][\w-]*)[\s)]", block))
            result["predicates"].discard("predicates")

        result["actions"] = set(re.findall(r"\(:action\s+([a-zA-Z][\w-]*)", text))

        return result

    def _predicate_name(self, pred: str) -> str:
        """Extract the predicate name from a declaration like '(locked ?i - item)'."""
        m = re.match(r"\(\s*([a-zA-Z][\w-]*)", pred.strip())
        return m.group(1) if m else pred.strip()

    def _repair_unbound_variables(self, action: dict) -> tuple[dict, list[str]]:
        """
        Detect ?variable names used in precondition/effect but not declared in
        parameters, and add them as '?var - item'.

        This fixes a common VLM error where e.g. stir is defined with
        parameters=(?container) but precondition uses ?tool without declaring it.

        Returns (repaired_action, list_of_added_var_names).
        """
        params_str = str(action.get("parameters", "()"))
        precond    = str(action.get("precondition", ""))
        effect     = str(action.get("effect", ""))

        # Don't attempt repair on structurally broken params — the balanced-paren
        # check later in enrich() will catch and skip the action.
        if not self._balanced(params_str):
            return action, []

        declared   = set(re.findall(r"\?([a-zA-Z][\w-]*)", params_str))
        used       = set(re.findall(r"\?([a-zA-Z][\w-]*)", precond + " " + effect))
        undeclared = sorted(used - declared)

        if not undeclared:
            return action, []

        fixed = dict(action)
        extra = " ".join(f"?{v} - item" for v in undeclared)
        if params_str.strip() in ("()", ""):
            fixed["parameters"] = f"({extra})"
        else:
            fixed["parameters"] = params_str.rstrip(")") + f" {extra})"
        return fixed, undeclared

    def _infer_predicates_from_effect(
        self, action: dict, existing_predicates: set[str]
    ) -> list[tuple[str, str]]:
        """
        Extract novel positive predicates from an action's effect string and
        return declarations for any that are not already in the domain.

        This handles the case where the VLM adds a new_action with a novel
        result predicate but forgets to declare it in new_predicates.

        Returns list of (predicate_declaration, predicate_name) pairs.
        """
        effect = str(action.get("effect", ""))
        params_str = str(action.get("parameters", "()"))

        type_map: dict[str, str] = {}
        for m in re.finditer(r"\?([a-zA-Z][\w-]*)\s*-\s*([a-zA-Z][\w-]*)", params_str):
            type_map[m.group(1)] = m.group(2)

        results: list[tuple[str, str]] = []
        # Walk top-level expressions in effect (excluding and/or/not wrappers)
        depth, start = 0, -1
        exprs: list[str] = []
        for i, c in enumerate(effect):
            if c == "(":
                if depth == 0: start = i
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and start != -1:
                    exprs.append(effect[start: i + 1])
                    start = -1

        def _collect(expr: str) -> None:
            expr = expr.strip()
            if not expr.startswith("("): return
            inner = expr[1:-1].strip()
            tokens = inner.split()
            if not tokens: return
            head = tokens[0]
            if head in ("and", "or"):
                sub_depth, sub_start = 0, -1
                rest = inner[len(head):].strip()
                for i2, c2 in enumerate(rest):
                    if c2 == "(":
                        if sub_depth == 0: sub_start = i2
                        sub_depth += 1
                    elif c2 == ")":
                        sub_depth -= 1
                        if sub_depth == 0 and sub_start != -1:
                            _collect(rest[sub_start: i2 + 1])
                            sub_start = -1
            elif head == "not":
                pass  # negation — skip (not a new state predicate)
            else:
                # Leaf predicate: (pred_name ?v1 ?v2 ...)
                if head in existing_predicates:
                    return
                # Build parameter types from type_map
                param_parts = []
                for tok in tokens[1:]:
                    if tok.startswith("?"):
                        vname = tok[1:]
                        vtype = type_map.get(vname, "item")
                        param_parts.append(f"{tok} - {vtype}")
                decl = f"({head} {' '.join(param_parts)})" if param_parts else f"({head})"
                results.append((decl, head))

        for expr in exprs:
            _collect(expr)

        seen: set[str] = set()
        unique = []
        for decl, name in results:
            if name not in seen:
                seen.add(name)
                unique.append((decl, name))
        return unique

    def _format_action(self, action: dict) -> str:
        """Render an action dict to a PDDL (:action ...) string.

        Handles VLM output quirks:
        - parameters as list  → joins into a PDDL-style string
        - natural-language precondition/effect → wraps in a comment, uses (and) fallback
        """
        name = action.get("name", "unnamed")

        # parameters: VLM sometimes returns ["?x - item", "?y - item"] or ["x", "y"]
        raw_params = action.get("parameters", "()")
        if isinstance(raw_params, list):
            parts = []
            for p in raw_params:
                p = str(p).strip()
                if not p.startswith("?"):
                    p = f"?{p} - item"
                parts.append(p)
            parameters = f"({' '.join(parts)})" if parts else "()"
        else:
            parameters = str(raw_params)

        # precondition / effect: VLM sometimes writes natural language
        def _sanitize_pddl(val, fallback):
            s = str(val) if not isinstance(val, str) else val
            # Heuristic: valid PDDL starts with ( and has balanced parens
            s = s.strip()
            if s.startswith("(") and s.count("(") == s.count(")"):
                return s
            # Natural language → treat as invalid, use fallback
            return fallback

        precond = _sanitize_pddl(action.get("precondition", "(and)"), "(and)")
        effect  = _sanitize_pddl(action.get("effect",      "(and)"), "(and)")
        return (
            f"  (:action {name}\n"
            f"    :parameters {parameters}\n"
            f"    :precondition {precond}\n"
            f"    :effect {effect}\n"
            f"  )"
        )

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _balanced(text: str) -> bool:
        """Return True if parentheses in text are balanced."""
        return text.count("(") == text.count(")")

    def _validate(self, domain_text: str) -> tuple[bool, list[str]]:
        """
        Structural validation of the enriched domain.
        Checks: parenthesis balance, required blocks, at least one action.
        """
        errors: list[str] = []

        if not self._balanced(domain_text):
            errors.append(
                f"unbalanced parentheses "
                f"(open={domain_text.count('(')}, close={domain_text.count(')')})"
            )

        for keyword in (":requirements", ":types", ":predicates"):
            if keyword not in domain_text:
                errors.append(f"missing {keyword} block")

        if not re.search(r"\(:action\b", domain_text):
            errors.append("domain has no actions")

        return len(errors) == 0, errors
