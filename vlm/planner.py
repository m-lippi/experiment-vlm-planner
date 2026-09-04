"""
VLM planning module.
Input:  user command (str) + scene image(s) (file paths or PIL Images)
Output: VLMPlan — structured list of primitive calls
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from PIL import Image


@dataclass
class PlanStep:
    primitive: str      # e.g. "pick"
    args: dict          # e.g. {"object": "red_cup"}


_EMPTY_ADDITIONS: dict = {
    "new_types": [],
    "new_predicates": [],
    "new_actions": [],
    "modified_preconditions": {},
}

def _load_base_domain_predicates() -> frozenset[str]:
    """Extract predicate names declared in the :predicates block of all domain files."""
    domains_dir = Path(__file__).resolve().parent.parent / "pddl" / "domains"
    names: set[str] = set()
    for path in domains_dir.glob("*.pddl"):
        raw  = path.read_text()
        # Strip PDDL line comments (;...\n) before any paren-counting.
        text = re.sub(r";[^\n]*", "", raw)
        # Isolate the :predicates block (everything between (:predicates and the
        # matching closing paren at the same nesting depth).
        start = text.find("(:predicates")
        if start == -1:
            continue
        depth = 0
        end   = start
        for i, ch in enumerate(text[start:], start):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        block = text[start:end + 1]
        # Each predicate declaration is a top-level s-expr inside the block:
        # (predicate-name ?param - type ...)
        # Skip the outer (:predicates ...) wrapper by stripping it, then find
        # all top-level opening parens.
        inner = block[len("(:predicates"):].strip()
        d = 0
        tok = []
        for ch in inner:
            if ch == "(":
                d += 1
                tok.append(ch)
            elif ch == ")":
                d -= 1
                tok.append(ch)
                if d == 0:
                    declaration = "".join(tok).strip()
                    tok = []
                    m = re.match(r"\(\s*([a-zA-Z][a-zA-Z0-9\-]*)", declaration)
                    if m:
                        names.add(m.group(1))
            elif d > 0:
                tok.append(ch)
    return frozenset(names)


_BASE_DOMAIN_PREDICATES: frozenset[str] = _load_base_domain_predicates()

# PDDL structural keywords — not predicate names.
_PDDL_KEYWORDS: frozenset[str] = frozenset({
    "and", "or", "not", "forall", "exists", "when", "imply",
    "increase", "decrease", "assign",
})


def _fix_enrichment_predicates(domain_additions: dict) -> dict:
    """
    Ensure every predicate used in new_actions is declared in new_predicates.

    The VLM often defines a new action (e.g. pour) that introduces novel
    predicates (e.g. empty, full) in its precondition/effect but forgets to
    list them in new_predicates. Fast Downward rejects such domains.

    This function scans all precondition/effect strings, extracts predicate
    names that are neither in the base domain nor already declared, and
    auto-generates their declaration by inferring the signature from the
    action's parameter list.
    """
    new_actions = domain_additions.get("new_actions", [])
    if not new_actions:
        return domain_additions

    declared: set[str] = set()
    for decl in domain_additions.get("new_predicates", []):
        m = re.match(r"\(\s*([a-zA-Z][a-zA-Z0-9\-]*)", decl)
        if m:
            declared.add(m.group(1))

    auto_added: dict[str, str] = {}  # name -> declaration string

    for action in new_actions:
        # Build var->type map from the action parameters string
        params_str = action.get("parameters", "")
        var_type: dict[str, str] = {}
        for var, typ in re.findall(r"\?(\w+)\s*-\s*(\w+)", params_str):
            var_type[var] = typ

        for field_name in ("precondition", "effect"):
            expr = action.get(field_name, "")
            # Flatten negations so (not (pred ...)) becomes (pred ...)
            expr_flat = re.sub(r"\(\s*not\s+", "(", expr)
            for m in re.finditer(
                r"\(\s*([a-zA-Z][a-zA-Z0-9\-]+)((?:\s+\?\w+)*)\s*\)",
                expr_flat,
            ):
                pred_name = m.group(1)
                args_str  = m.group(2).strip()
                if pred_name in _PDDL_KEYWORDS:
                    continue
                if pred_name in _BASE_DOMAIN_PREDICATES:
                    continue
                if pred_name in declared or pred_name in auto_added:
                    continue
                # Infer signature from variables used in this occurrence
                vars_used = re.findall(r"\?(\w+)", args_str)
                sig_parts = [
                    f"?{v} - {var_type.get(v, 'item')}" for v in vars_used
                ]
                decl = (
                    f"({pred_name} {' '.join(sig_parts)})"
                    if sig_parts else f"({pred_name})"
                )
                auto_added[pred_name] = decl
                declared.add(pred_name)

    if not auto_added:
        return domain_additions

    print(f"[INFO] Auto-declared missing predicates: {list(auto_added.keys())}")
    result = dict(domain_additions)
    result["new_predicates"] = (
        list(domain_additions.get("new_predicates", [])) + list(auto_added.values())
    )
    return result


@dataclass
class VLMPlan:
    goal: str
    steps: list[PlanStep]
    raw_output: str           # raw VLM text (kept for debugging / thesis analysis)
    domain_template: str = "manipulation_base"
    domain_additions: dict = field(default_factory=lambda: {
        "new_types": [],
        "new_predicates": [],
        "new_actions": [],
        "modified_preconditions": {},
    })

    def to_domain_additions(self):
        """Convert domain_additions dict → DomainAdditions for use with DomainEnricher."""
        from planner.domain_enricher import DomainAdditions
        d = self.domain_additions
        return DomainAdditions(
            new_types=d.get("new_types", []),
            new_predicates=d.get("new_predicates", []),
            new_actions=d.get("new_actions", []),
            modified_preconditions=d.get("modified_preconditions", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string (for host→container transport)."""
        return json.dumps({
            "goal": self.goal,
            "steps": [{"primitive": s.primitive, "args": s.args} for s in self.steps],
            "raw_output": self.raw_output,
            "domain_template": self.domain_template,
            "domain_additions": self.domain_additions,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "VLMPlan":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        steps = [PlanStep(primitive=s["primitive"], args=s["args"])
                 for s in data.get("steps", [])]
        return cls(
            goal=data["goal"],
            steps=steps,
            raw_output=data.get("raw_output", ""),
            domain_template=data.get("domain_template", "manipulation_base"),
            domain_additions=data.get("domain_additions", _EMPTY_ADDITIONS.copy()),
        )


ImageInput = Union[str, Path, Image.Image]


# ── Short display names for HTML reports / logging ──────────────────────────
MODEL_SHORT_NAMES: dict[str, str] = {
    "Qwen/Qwen3-VL-8B-Instruct":         "Qwen3-VL-8B",
    "Qwen/Qwen2.5-VL-7B-Instruct":       "Qwen2.5-VL-7B",
    "Qwen/Qwen2.5-VL-14B-Instruct":      "Qwen2.5-VL-14B",
    "OpenGVLab/InternVL2_5-8B":           "InternVL2.5-8B",
    "OpenGVLab/InternVL2_5-14B":          "InternVL2.5-14B",
}


def model_short_name(model_id: str) -> str:
    """Return a short display name for a model ID."""
    return MODEL_SHORT_NAMES.get(model_id, model_id.split("/")[-1])


# ── Base interface ───────────────────────────────────────────────────────────

class BasePlanner:
    """
    Common interface for all VLM backends.
    Subclasses implement load() and plan_remaining().
    """

    SYSTEM_PROMPT_REPLANNING_PATH = (
        Path(__file__).parent / "prompts" / "system_prompt_replanning.txt"
    )

    def __init__(self, model_id: str):
        self.model_id   = model_id
        self.short_name = model_short_name(model_id)

    def load(self) -> None:
        raise NotImplementedError

    def plan_remaining(
        self,
        task: str,
        images: list[ImageInput],
        completed_steps: list[str],
        failed_step: str | None = None,
        scene_objects: list[str] | None = None,
        prior_enrichment: dict | None = None,
    ) -> VLMPlan:
        raise NotImplementedError

    def _to_pil(self, img: ImageInput) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        return Image.open(img).convert("RGB")

    def _parse_output(self, command: str, raw: str) -> VLMPlan:
        """Shared JSON parser — identical for all backends."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        json_str = match.group(1) if match else raw.strip()
        try:
            data   = json.loads(json_str)
            steps  = [PlanStep(**s) for s in data.get("steps", [])]
            goal   = data.get("goal", command)
            dt     = data.get("domain_template", "manipulation_base")
            da     = data.get("domain_additions", _EMPTY_ADDITIONS.copy())
        except (json.JSONDecodeError, TypeError):
            steps, goal, dt, da = [], command, "manipulation_base", _EMPTY_ADDITIONS.copy()
        da = _fix_enrichment_predicates(da)
        return VLMPlan(goal=goal, steps=steps, raw_output=raw,
                       domain_template=dt, domain_additions=da)

    def _build_user_text(
        self,
        task: str,
        completed_steps: list[str],
        failed_step: str | None,
        scene_objects: list[str] | None,
        prior_enrichment: dict | None = None,
    ) -> str:
        completed_str = (
            "\n".join(f"  - {s}" for s in completed_steps)
            if completed_steps else "  (none yet)"
        )
        failed_str = (
            f"\n\nFailed step (just failed — replan needed):\n  {failed_step}"
            if failed_step else ""
        )
        scene_str = ""
        if scene_objects:
            scene_str = (
                "\n\nScene objects (use EXACTLY these names in your plan):\n"
                + "\n".join(f"  - {o}" for o in scene_objects)
            )
        enrichment_str = ""
        if prior_enrichment and prior_enrichment.get("new_actions"):
            lines = ["\n\nPrior domain enrichments (actions you defined in earlier iterations):"]
            for a in prior_enrichment["new_actions"]:
                lines.append(
                    f"  - {a.get('name','?')} {a.get('parameters','')}: "
                    f"pre={a.get('precondition','?')} "
                    f"eff={a.get('effect','?')}"
                )
            lines.append(
                "You may reuse these definitions unchanged, refine them if the previous "
                "attempt failed, or replace them if the task context has changed. "
                "If they are still valid, leave domain_additions empty."
            )
            enrichment_str = "\n".join(lines)
        return (
            f"Task goal: {task}\n\n"
            f"Completed steps:\n{completed_str}"
            f"{failed_str}{scene_str}{enrichment_str}\n\n"
            f"Generate the COMPLETE remaining plan to finish the task."
        )


# ── InternVL2.5 backend ──────────────────────────────────────────────────────

class InternVLPlanner(BasePlanner):
    """
    VLM backend for InternVL2.5 family (OpenGVLab/InternVL2_5-*).

    InternVL uses a different API than Qwen:
      - model.chat(tokenizer, pixel_values, question, generation_config)
      - Image preprocessing via torchvision transforms
      - System prompt prepended to user text (no separate roles dict)
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD  = (0.229, 0.224, 0.225)
    _IMG_SIZE      = 448   # InternVL2.5 standard tile size

    def __init__(self, model_id: str = "OpenGVLab/InternVL2_5-8B"):
        super().__init__(model_id)
        self._model     = None
        self._tokenizer = None

    def load(self) -> None:
        from vlm.model_loader import load_internvl
        self._model, self._tokenizer = load_internvl(self.model_id)

    def plan_remaining(
        self,
        task: str,
        images: list[ImageInput],
        completed_steps: list[str],
        failed_step: str | None = None,
        scene_objects: list[str] | None = None,
        prior_enrichment: dict | None = None,
    ) -> VLMPlan:
        if self._model is None:
            raise RuntimeError("Call load() before plan_remaining()")

        import torch
        system_prompt = self.SYSTEM_PROMPT_REPLANNING_PATH.read_text()
        user_text     = self._build_user_text(task, completed_steps, failed_step, scene_objects, prior_enrichment)

        # Build pixel_values for the first image (primary overview)
        pil_img      = self._to_pil(images[0]) if images else None
        pixel_values = self._preprocess_image(pil_img) if pil_img else None

        # InternVL format: <image> token + combined system+user text
        question = f"<image>\n{system_prompt}\n\n{user_text}"

        gen_cfg = {"max_new_tokens": 768, "do_sample": False}
        with torch.no_grad():
            raw = self._model.chat(
                self._tokenizer,
                pixel_values,
                question,
                gen_cfg,
            )

        plan = self._parse_output(task, raw)
        try:
            data = json.loads(
                raw if raw.strip().startswith("{")
                else re.search(r"\{.*\}", raw, re.DOTALL).group()
            )
            if data.get("complete"):
                plan.steps = []
        except Exception:
            pass
        return plan

    def _preprocess_image(self, pil_img: Image.Image):
        """Convert PIL image to InternVL pixel_values tensor."""
        import torch
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Resize((self._IMG_SIZE, self._IMG_SIZE),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])
        device = next(self._model.parameters()).device
        return transform(pil_img.convert("RGB")).unsqueeze(0).to(
            dtype=torch.bfloat16, device=device
        )


# ── Factory ──────────────────────────────────────────────────────────────────

# ── Gemini backend (Google AI Studio) ───────────────────────────────────────

class GeminiPlanner(BasePlanner):
    """
    VLM backend for Google Gemini models via google-genai SDK.

    API key is read from the GOOGLE_API_KEY environment variable, or passed
    explicitly via the api_key constructor argument.

    Supported model IDs:
      gemini-2.0-flash          (fast, free tier, recommended)
      gemini-1.5-pro            (higher quality, slower)
      gemini-2.5-pro-preview-06-05  (latest preview)
    """

    def __init__(self, model_id: str = "gemini-2.0-flash", api_key: str | None = None):
        super().__init__(model_id)
        self._api_key = api_key
        self._client  = None

    def load(self) -> None:
        """Initialise the Gemini client (no weights to download — API call)."""
        import os
        from google import genai

        key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini API key not found. Set the GOOGLE_API_KEY environment variable "
                "or pass api_key= to create_planner()."
            )
        self._client = genai.Client(api_key=key)

    def plan_remaining(
        self,
        task: str,
        images: list[ImageInput],
        completed_steps: list[str],
        failed_step: str | None = None,
        scene_objects: list[str] | None = None,
        prior_enrichment: dict | None = None,
    ) -> VLMPlan:
        if self._client is None:
            raise RuntimeError("Call load() before plan_remaining()")

        import base64, io
        from google.genai import types as gtypes

        system_prompt = self.SYSTEM_PROMPT_REPLANNING_PATH.read_text()
        user_text     = self._build_user_text(task, completed_steps, failed_step, scene_objects, prior_enrichment)

        parts: list = []
        for img in images:
            pil_img = self._to_pil(img)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            parts.append(gtypes.Part.from_bytes(
                data=buf.getvalue(), mime_type="image/png"
            ))
        parts.append(gtypes.Part.from_text(text=user_text))

        # Ensure model name has the required "models/" prefix
        model_name = self.model_id if self.model_id.startswith("models/") \
                     else f"models/{self.model_id}"

        # Mandatory inter-request delay: keeps usage under free-tier RPM limit
        import time
        time.sleep(getattr(self, "_inter_request_delay", 8.0))

        # Retry with exponential backoff on 429 (rate limit)
        raw = ""
        for attempt in range(3):
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=4096,
                        temperature=0.0,
                    ),
                )
                raw = response.text or ""
                break
            except Exception as exc:
                msg = str(exc)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    import re as _re
                    m_wait = _re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)", msg, _re.I)
                    wait = int(m_wait.group(1)) + 2 if m_wait else 65
                    print(f"\n    [RATE LIMIT] waiting {wait}s (attempt {attempt+1}/3)...", end="", flush=True)
                    time.sleep(wait)
                    if attempt == 2:
                        raise RuntimeError(f"Gemini rate limit not resolved after 3 retries. "
                                           f"Daily quota may be exhausted — retry tomorrow.")
                else:
                    raise

        plan = self._parse_output(task, raw)
        try:
            data = json.loads(
                raw if raw.strip().startswith("{")
                else re.search(r"\{.*\}", raw, re.DOTALL).group()
            )
            if data.get("complete"):
                plan.steps = []
        except Exception:
            pass
        return plan


_GEMINI_NAMES: dict[str, str] = {
    # Without prefix
    "gemini-2.5-flash":                       "Gemini-2.5-Flash",
    "gemini-2.5-pro":                         "Gemini-2.5-Pro",
    "gemini-2.0-flash":                       "Gemini-2.0-Flash",
    "gemini-2.0-flash-lite":                  "Gemini-2.0-Flash-Lite",
    "gemini-robotics-er-1.5-preview":         "Gemini-Robotics-1.5",
    "gemini-robotics-er-1.6-preview":         "Gemini-Robotics-1.6",
    # With models/ prefix (as returned by client.models.list())
    "models/gemini-2.5-flash":               "Gemini-2.5-Flash",
    "models/gemini-2.5-pro":                 "Gemini-2.5-Pro",
    "models/gemini-2.0-flash":               "Gemini-2.0-Flash",
    "models/gemini-2.0-flash-lite":          "Gemini-2.0-Flash-Lite",
    "models/gemini-robotics-er-1.5-preview": "Gemini-Robotics-1.5",
    "models/gemini-robotics-er-1.6-preview": "Gemini-Robotics-1.6",
}
MODEL_SHORT_NAMES.update(_GEMINI_NAMES)


def create_planner(model_id: str, api_key: str | None = None) -> BasePlanner:
    """
    Factory: return the right BasePlanner subclass for the given model ID.

    Supported model families:
      Qwen3-VL-*         → VLMPlanner       (Qwen3VLForConditionalGeneration)
      Qwen2.5-VL-*       → VLMPlanner       (Qwen2VLForConditionalGeneration)
      InternVL2_5-*      → InternVLPlanner
      InternVL2-*        → InternVLPlanner
      gemini-*           → GeminiPlanner     (Google AI Studio API)
    """
    mid = model_id.lower()
    if "internvl" in mid:
        return InternVLPlanner(model_id)
    if "gemini" in mid:
        return GeminiPlanner(model_id, api_key=api_key)
    # Default: Qwen family (handles Qwen3, Qwen2.5, future versions)
    return VLMPlanner(model_id)


class VLMPlanner(BasePlanner):
    """
    VLM backend for Qwen-VL family (Qwen3-VL-* and Qwen2.5-VL-*).
    Both share the same inference API (qwen_vl_utils + AutoProcessor).
    """

    SYSTEM_PROMPT_PATH            = Path(__file__).parent / "prompts" / "system_prompt.txt"
    SYSTEM_PROMPT_LOOP_PATH       = Path(__file__).parent / "prompts" / "system_prompt_loop.txt"

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__(model_id)
        # Legacy alias kept for backward compatibility
        self.model_name = model_id
        self._model     = None
        self._processor = None

    def load(self) -> None:
        """Load model weights. Call once at startup (heavy operation)."""
        from vlm.model_loader import load_qwen_vl
        self._model, self._processor = load_qwen_vl(self.model_id)

    def plan(
        self,
        command: str,
        images: list[ImageInput],
        scene_context: dict | None = None,
    ) -> VLMPlan:
        """
        Args:
            command:       Natural language task (e.g. "pick the red cup …").
            images:        One or more scene images (file paths or PIL Images).
            scene_context: Optional dict with known PDDL names, e.g.:
                           {"items": ["red_cup", "blue_box"],
                            "locations": ["shelf_b"]}
                           When provided, the names are appended to the user
                           message so the VLM uses them verbatim in its output.
                           This prevents name-mismatch failures between VLM
                           output and the PDDL problem / Gazebo oracle.

        Returns:
            VLMPlan with primitive sequence grounded on what the VLM sees.
        """
        if self._model is None:
            raise RuntimeError("Call load() before plan()")

        pil_images = [self._to_pil(img) for img in images]
        system_prompt = self.SYSTEM_PROMPT_PATH.read_text()

        messages = self._build_messages(system_prompt, command, pil_images, scene_context)
        raw = self._run_inference(messages)
        return self._parse_output(command, raw)

    def plan_next_step(
        self,
        task:            str,
        images:          list[ImageInput],
        completed_steps: list[str],
    ) -> VLMPlan:
        """Closed-loop mode: plan the NEXT SINGLE action given current state.

        Args:
            task:            Overall task description (unchanged throughout loop).
            images:          Current scene image(s) from wrist camera.
            completed_steps: List of already-executed primitives (e.g. ["pick(red_cup)"]).

        Returns:
            VLMPlan with 0 steps (task complete) or 1 step (next action).
        """
        if self._model is None:
            raise RuntimeError("Call load() before plan_next_step()")

        pil_images   = [self._to_pil(img) for img in images]
        system_prompt = self.SYSTEM_PROMPT_LOOP_PATH.read_text()

        completed_str = (
            "\n".join(f"  - {s}" for s in completed_steps)
            if completed_steps else "  (none yet)"
        )
        user_text = (
            f"Task goal: {task}\n\n"
            f"Completed steps:\n{completed_str}\n\n"
            f"What is the NEXT single action?"
        )

        messages = self._build_messages(system_prompt, user_text, pil_images)
        raw = self._run_inference(messages)
        plan = self._parse_output(task, raw)

        # Handle complete=true signal
        try:
            data = __import__("json").loads(
                raw if raw.strip().startswith("{") else
                __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL).group()
            )
            if data.get("complete"):
                plan.steps = []   # empty steps = task done
        except Exception:
            pass

        return plan

    def plan_remaining(
        self,
        task:             str,
        images:           list[ImageInput],
        completed_steps:  list[str],
        failed_step:      str | None = None,
        scene_objects:    list[str] | None = None,
        prior_enrichment: dict | None = None,
    ) -> VLMPlan:
        """Replanning mode: generate the COMPLETE remaining plan from current state.

        Called either at task start (completed_steps=[]) or after a failure.
        Returns ALL remaining steps to complete the task.

        Args:
            task:             Overall task description.
            images:           Current scene image(s) — overview camera preferred.
            completed_steps:  Steps already executed successfully.
            failed_step:      The step that just failed (for replan context), or None.
            prior_enrichment: Accumulated domain_additions from previous iterations.
                              Passed to the VLM so it can reuse, refine, or replace
                              previously defined actions.

        Returns:
            VLMPlan with all remaining steps, or empty steps if task is complete.
        """
        if self._model is None:
            raise RuntimeError("Call load() before plan_remaining()")

        pil_images    = [self._to_pil(img) for img in images]
        system_prompt = self.SYSTEM_PROMPT_REPLANNING_PATH.read_text()
        user_text     = self._build_user_text(task, completed_steps, failed_step, scene_objects, prior_enrichment)
        messages      = self._build_messages(system_prompt, user_text, pil_images)
        raw  = self._run_inference(messages)
        plan = self._parse_output(task, raw)

        try:
            data = __import__("json").loads(
                raw if raw.strip().startswith("{") else
                __import__("re").search(r"\{.*\}", raw, __import__("re").DOTALL).group()
            )
            if data.get("complete"):
                plan.steps = []
            # Store replan metadata in plan for debug logging
            if data.get("replanning") and data.get("replan_reason"):
                plan.raw_output = f"[REPLAN: {data['replan_reason']}]\n{plan.raw_output}"
        except Exception:
            pass

        # Second pass (VLM-as-judge): always run when the plan has steps so
        # the VLM can self-review enrichment, predicate declarations, and
        # grasp_mode consistency against all planning rules.
        if plan.steps:
            plan = self._check_and_fix_enrichment(task, plan, pil_images,
                                                   completed_steps, failed_step,
                                                   scene_objects, prior_enrichment)

        return plan

    # ------------------------------------------------------------------
    # Internal helpers (Qwen-specific; _to_pil / _parse_output from BasePlanner)
    # ------------------------------------------------------------------

    def _check_and_fix_enrichment(
        self,
        task:             str,
        plan:             VLMPlan,
        pil_images:       list,
        completed_steps:  list[str],
        failed_step:      str | None,
        scene_objects:    list[str] | None,
        prior_enrichment: dict | None,
    ) -> VLMPlan:
        """Second VLM pass (VLM-as-judge): same system prompt + images as the
        first pass. Feed the first-pass plan back to the VLM so it can self-review
        against all planning rules and correct any violations.
        """
        print("[INFO] Enrichment self-review pass (2nd cycle)...")
        system_prompt = self.SYSTEM_PROMPT_REPLANNING_PATH.read_text()

        plan_summary = json.dumps({
            "goal":             plan.goal,
            "domain_template":  plan.domain_template,
            "domain_additions": plan.domain_additions,
            "steps": [{"primitive": s.primitive, "args": s.args} for s in plan.steps],
        }, indent=2)

        base_user_text = self._build_user_text(
            task, completed_steps, failed_step, scene_objects, prior_enrichment
        )
        review_user_text = (
            f"{base_user_text}\n\n"
            f"---\n"
            f"Your previous attempt produced this plan:\n{plan_summary}\n\n"
            f"Review it against ALL planning rules above. Pay particular attention to:\n"
            f"  - Rule 7: does every non-core action verb in the task appear as a "
            f"primitive name with its definition in domain_additions?\n"
            f"  - Rule 6: for each pick step, does the action that follows require the "
            f"object to be tilted or inverted? If yes, grasp_mode must be 'side'.\n"
            f"If the plan violates any rule, output the corrected plan. "
            f"If it is already correct, output it unchanged."
        )

        messages  = self._build_messages(system_prompt, review_user_text, pil_images)
        raw       = self._run_inference(messages)
        corrected = self._parse_output(task, raw)

        # Sanity guard: never replace a valid plan with an empty one
        if not corrected.steps and plan.steps:
            print("[WARN] Self-review returned empty steps — keeping original plan.")
            return plan

        if corrected.domain_additions.get("new_actions"):
            n = len(corrected.domain_additions["new_actions"])
            print(f"[INFO] Self-review: corrected plan with {n} custom action(s).")
        else:
            print("[INFO] Self-review: plan confirmed correct.")
        return corrected

    def _build_messages(
        self,
        system_prompt: str,
        command: str,
        images: list[Image.Image],
        scene_context: dict | None = None,
    ) -> list[dict]:
        image_content = [{"type": "image", "image": img} for img in images]
        user_text = command
        if scene_context:
            parts = ["\n\nKnown PDDL names — use these EXACTLY in your JSON output:"]
            if scene_context.get("items"):
                parts.append(f"  Items: {', '.join(scene_context['items'])}")
            if scene_context.get("locations"):
                parts.append(f"  Locations: {', '.join(scene_context['locations'])}")
            user_text += "\n".join(parts)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": user_text}],
            },
        ]

    def _run_inference(self, messages: list[dict]) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=512)

        output_ids = generated[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # _parse_output inherited from BasePlanner
