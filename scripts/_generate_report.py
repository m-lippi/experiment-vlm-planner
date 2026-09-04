"""
Generate a self-contained HTML report for a single VLM-RobotPlanner run.

Usage (standalone):
    python scripts/_generate_report.py data/runs/2026-07-02_16-35-13_kitchen_...

Usage (from run_loop_host.py):
    from scripts._generate_report import generate_html_report
    report_path = generate_html_report(_RUN_DIR)
"""

from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_run_info(run_dir: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    p = run_dir / "run_info.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
    return info


def _load_debug(iter_dir: Path) -> dict[str, Any] | None:
    p = iter_dir / "debug.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _collect_iterations(run_dir: Path) -> list[tuple[int, Path, dict[str, Any]]]:
    """Return sorted list of (iter_num, iter_dir, debug_data) for valid iterations."""
    results = []
    for d in sorted(run_dir.iterdir()):
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and d.is_dir():
            data = _load_debug(d)
            if data is not None:
                results.append((int(m.group(1)), d, data))
    results.sort(key=lambda x: x[0])
    return results


def _img_b64(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    enc = base64.b64encode(raw).decode()
    suffix = path.suffix.lstrip(".").lower()
    mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
    return f"data:image/{mime};base64,{enc}"


def _collect_images(run_dir: Path, iter_num: int, iter_dir: Path) -> dict[str, str | None]:
    tag = f"iter_{iter_num:02d}"
    return {
        "overview": _img_b64(iter_dir / "overview.png"),
        "wrist": _img_b64(iter_dir / "wrist.png"),
        "overview_annotated": _img_b64(iter_dir / "overview_annotated.png"),
        "dino": _img_b64(run_dir / f"{tag}_dino.png"),
        "dino_wrist": _img_b64(run_dir / f"{tag}_dino_wrist.png"),
    }


# ---------------------------------------------------------------------------
# Compute run-level statistics
# ---------------------------------------------------------------------------

def _run_stats(iters: list[tuple[int, Path, dict[str, Any]]], run_info: dict[str, str]) -> dict[str, Any]:
    if not iters:
        return {
            "n_iters": 0, "vlm_time_total": 0.0,
            "enrichment_count": 0, "completed_steps": [],
            "task": run_info.get("task", ""), "world": run_info.get("world", ""),
            "timestamp": run_info.get("timestamp", ""),
        }

    last_debug = iters[-1][2]
    completed = last_debug.get("completed_steps", [])
    total_vlm = sum(d.get("vlm_time_s", 0.0) for _, _, d in iters)

    # Count unique novel actions across all iterations (from full_remaining_plan)
    seen_actions: set[str] = set()
    for _, _, d in iters:
        frp = d.get("full_remaining_plan", {})
        for act in frp.get("domain_additions", {}).get("new_actions", []):
            name = act.get("name", "")
            if name:
                seen_actions.add(name)

    return {
        "n_iters": len(iters),
        "vlm_time_total": total_vlm,
        "enrichment_count": len(seen_actions),
        "completed_steps": completed,
        "task": last_debug.get("task", run_info.get("task", "")),
        "world": last_debug.get("world", run_info.get("world", "")),
        "timestamp": run_info.get("timestamp", ""),
    }


def _run_status(stats: dict[str, Any]) -> str:
    """Return 'completed', 'partial', or 'failed'."""
    steps = stats.get("completed_steps", [])
    task = stats.get("task", "").lower()
    if not steps:
        return "failed"
    # Heuristic: if the last completed step involves the final object/action in the task
    # we can't know for sure without oracle; use 'partial' if steps > 0 but ambiguous.
    return "partial" if steps else "failed"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def _css() -> str:
    return """
:root {
  --bg: #0d1117;
  --bg-card: #161b22;
  --bg-card2: #1c2128;
  --border: #30363d;
  --border-active: #58a6ff;
  --text: #e6edf3;
  --text-muted: #8b949e;
  --text-mono: #79c0ff;
  --green: #3fb950;
  --red: #f85149;
  --yellow: #d29922;
  --blue: #58a6ff;
  --purple: #bc8cff;
  --gold-bg: #2a1f00;
  --gold-border: #d4a017;
  --gold-text: #f0c040;
  --radius: 8px;
  --shadow: 0 2px 8px rgba(0,0,0,.45);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  padding: 0 0 60px;
}
a { color: var(--blue); }
code, pre {
  font-family: Consolas, "Cascadia Code", Monaco, "Courier New", monospace;
  font-size: 12.5px;
}
/* ── header ── */
.page-header {
  background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
  border-bottom: 1px solid var(--border);
  padding: 28px 32px 22px;
}
.page-header h1 {
  font-size: 22px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: -0.3px;
  margin-bottom: 6px;
}
.page-header .meta {
  color: var(--text-muted);
  font-size: 13px;
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
}
.page-header .meta span { display: flex; align-items: center; gap: 5px; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: .3px;
}
.badge-success { background: rgba(63,185,80,.15); color: var(--green); border: 1px solid rgba(63,185,80,.4); }
.badge-fail    { background: rgba(248,81,73,.15);  color: var(--red);   border: 1px solid rgba(248,81,73,.4); }
.badge-warn    { background: rgba(210,153,34,.15); color: var(--yellow);border: 1px solid rgba(210,153,34,.4); }
.badge-blue    { background: rgba(88,166,255,.12); color: var(--blue);  border: 1px solid rgba(88,166,255,.3); }
.badge-gold    { background: var(--gold-bg);       color: var(--gold-text); border: 1px solid var(--gold-border); }
/* ── summary bar ── */
.summary-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 16px 32px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
}
.stat-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 18px;
  min-width: 130px;
  flex: 1;
}
.stat-box .stat-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .6px; }
.stat-box .stat-value { font-size: 22px; font-weight: 700; color: var(--text); margin-top: 2px; }
.stat-box .stat-sub   { font-size: 11px; color: var(--text-muted); }
.stat-box.highlight { border-color: var(--gold-border); background: var(--gold-bg); }
.stat-box.highlight .stat-value { color: var(--gold-text); }
/* ── pipeline diagram ── */
.pipeline-section {
  padding: 20px 32px;
  border-bottom: 1px solid var(--border);
}
.pipeline-section h2 { font-size: 14px; font-weight: 600; color: var(--text-muted); margin-bottom: 14px; text-transform: uppercase; letter-spacing: .5px; }
.pipeline {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0;
  overflow-x: auto;
  padding-bottom: 4px;
}
.pipe-node {
  background: var(--bg-card);
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 16px;
  text-align: center;
  min-width: 110px;
}
.pipe-node.active { border-color: var(--green); background: rgba(63,185,80,.07); }
.pipe-node.enriched { border-color: var(--gold-border); background: var(--gold-bg); }
.pipe-node .node-label { font-size: 12px; font-weight: 600; }
.pipe-node .node-sub   { font-size: 10px; color: var(--text-muted); margin-top: 2px; }
.pipe-node .node-detail { font-size: 10px; color: var(--text-muted); margin-top: 4px; border-top: 1px solid var(--border); padding-top: 4px; font-family: monospace; }
.pipe-node.enriched .node-detail { color: var(--gold-text); border-top-color: rgba(212,160,23,.3); }
/* ── meta chips ── */
.meta-chip { display: inline-block; background: rgba(88,166,255,.12); color: var(--blue); border: 1px solid rgba(88,166,255,.25); border-radius: 4px; padding: 0 6px; font-size: 11px; font-weight: 600; letter-spacing: .3px; margin-right: 2px; }
.pipe-arrow { color: var(--text-muted); padding: 0 6px; font-size: 18px; line-height: 1; align-self: center; }
/* ── iterations ── */
.iterations { padding: 20px 32px; }
.iterations h2 { font-size: 14px; font-weight: 600; color: var(--text-muted); margin-bottom: 14px; text-transform: uppercase; letter-spacing: .5px; }
.iter-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 12px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.iter-card.has-enrichment { border-left: 3px solid var(--gold-border); }
.iter-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid transparent;
  transition: background .15s;
}
.iter-header:hover { background: var(--bg-card2); }
.iter-header.open { border-bottom-color: var(--border); }
.iter-num {
  width: 32px; height: 32px;
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-weight: 700; font-size: 13px; flex-shrink: 0;
}
.iter-title { font-weight: 600; font-size: 14px; }
.iter-sub { font-size: 12px; color: var(--text-muted); flex: 1; }
.iter-body { display: none; }
.iter-body.open { display: block; }
/* ── tabs ── */
.tab-bar {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card2);
  padding: 0 16px;
}
.tab-btn {
  padding: 9px 16px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--text-muted);
  font-size: 13px;
  cursor: pointer;
  transition: color .15s, border-color .15s;
  margin-bottom: -1px;
}
.tab-btn:hover { color: var(--text); }
.tab-btn.active { color: var(--text); border-bottom-color: var(--blue); }
.tab-pane { display: none; padding: 16px; }
.tab-pane.active { display: block; }
/* ── image grid ── */
.img-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.img-cell { flex: 1; min-width: 200px; max-width: 380px; }
.img-cell .img-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 6px; }
.img-cell img {
  width: 100%;
  border-radius: 6px;
  border: 1px solid var(--border);
  cursor: zoom-in;
  display: block;
}
/* ── PDDL / code blocks ── */
.code-block {
  background: #010409;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
  overflow-x: auto;
  font-family: Consolas, "Cascadia Code", Monaco, monospace;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre;
  color: #c9d1d9;
}
.kw  { color: #ff7b72; }
.sym { color: #79c0ff; }
.str { color: #a5d6ff; }
.cmt { color: #8b949e; font-style: italic; }
/* ── enrichment box ── */
.enrichment-box {
  background: var(--gold-bg);
  border: 1.5px solid var(--gold-border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 14px;
}
.enrichment-box .enrich-header {
  font-size: 14px; font-weight: 700; color: var(--gold-text);
  margin-bottom: 10px; display: flex; align-items: center; gap: 8px;
}
.enrichment-box .enrich-action { margin-bottom: 10px; }
.enrichment-box .enrich-action:last-child { margin-bottom: 0; }
.enrich-name { font-size: 16px; font-weight: 700; color: var(--gold-text); margin-bottom: 8px; }
.enrich-field { margin-bottom: 6px; }
.enrich-field .field-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 3px; }
.enrich-note {
  background: rgba(212,160,23,.08);
  border-left: 3px solid var(--gold-border);
  padding: 8px 12px;
  font-size: 12px;
  color: #c9a227;
  border-radius: 0 4px 4px 0;
  margin-top: 12px;
  font-style: italic;
}
/* ── plan steps list ── */
.step-list { list-style: none; }
.step-list li {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px solid rgba(48,54,61,.5);
}
.step-list li:last-child { border-bottom: none; }
.step-idx {
  font-size: 11px; color: var(--text-muted);
  min-width: 24px; padding-top: 1px;
}
.step-primitive { font-weight: 600; color: var(--blue); }
.step-args { font-size: 12px; color: var(--text-muted); font-family: monospace; }
/* ── section labels ── */
.section-label {
  font-size: 11px; font-weight: 600; color: var(--text-muted);
  text-transform: uppercase; letter-spacing: .6px;
  margin-bottom: 8px; margin-top: 14px;
}
.section-label:first-child { margin-top: 0; }
/* ── execution primitives ── */
.exec-box {
  background: var(--bg-card2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 12px;
}
.exec-box .prim-name { font-size: 17px; font-weight: 700; color: var(--purple); margin-bottom: 6px; }
.exec-box .prim-args-table { width: 100%; border-collapse: collapse; }
.exec-box .prim-args-table td { padding: 3px 8px; font-size: 12px; }
.exec-box .prim-args-table td:first-child { color: var(--text-muted); width: 110px; font-family: monospace; }
.completed-steps { margin-top: 12px; }
.completed-chip {
  display: inline-block;
  background: rgba(63,185,80,.1);
  border: 1px solid rgba(63,185,80,.3);
  color: var(--green);
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 12px;
  margin: 3px 3px 3px 0;
  font-family: monospace;
}
/* ── lightbox ── */
#lbOverlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.88);
  z-index: 9999;
  align-items: center; justify-content: center;
  cursor: zoom-out;
}
#lbOverlay.open { display: flex; }
#lbOverlay img { max-width: 94vw; max-height: 92vh; border-radius: 6px; border: 1px solid var(--border); }
/* ── plan modal ── */
#planOverlay {
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,.75);
  z-index: 9998;
  align-items: flex-start; justify-content: center;
  padding: 40px 16px;
  overflow-y: auto;
}
#planOverlay.open { display: flex; }
#planModal {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: 0 8px 32px rgba(0,0,0,.6);
  width: 100%; max-width: 680px;
  padding: 0;
  position: relative;
}
#planModal .modal-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}
#planModal .modal-header h3 { font-size: 16px; font-weight: 600; }
#planModal .modal-close {
  background: none; border: none; color: var(--text-muted);
  font-size: 20px; cursor: pointer; padding: 0 4px; line-height: 1;
}
#planModal .modal-close:hover { color: var(--text); }
#planModal .modal-body { padding: 20px; }
.plan-step-row {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 8px 0; border-bottom: 1px solid rgba(48,54,61,.5);
}
.plan-step-row:last-child { border-bottom: none; }
.plan-step-row .ps-num { min-width: 28px; font-size: 12px; color: var(--text-muted); padding-top: 2px; }
.plan-step-row .ps-body { flex: 1; }
.plan-step-row .ps-prim { font-weight: 600; color: var(--blue); font-size: 14px; }
.plan-step-row .ps-args { font-size: 12px; color: var(--text-muted); font-family: monospace; }
.plan-step-row .ps-status { font-size: 18px; line-height: 1; padding-top: 2px; }
.ps-done   { color: var(--green); }
.ps-pending { color: var(--text-muted); }
/* ── open plan button ── */
.btn-plan {
  background: rgba(88,166,255,.1); color: var(--blue);
  border: 1px solid rgba(88,166,255,.35);
  border-radius: var(--radius); padding: 8px 16px;
  font-size: 13px; font-weight: 600; cursor: pointer;
  transition: background .15s;
  white-space: nowrap;
}
.btn-plan:hover { background: rgba(88,166,255,.2); }
/* ── footer ── */
.page-footer {
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
  padding: 24px 32px;
  border-top: 1px solid var(--border);
  margin-top: 20px;
}
/* ── responsive ── */
@media (max-width: 640px) {
  .page-header, .summary-bar, .pipeline-section, .iterations { padding-left: 16px; padding-right: 16px; }
  .summary-bar { flex-direction: column; }
  .pipeline { flex-direction: column; align-items: flex-start; }
  .pipe-arrow { transform: rotate(90deg); }
}
"""


# ---------------------------------------------------------------------------
# JS
# ---------------------------------------------------------------------------

def _js() -> str:
    return """
function toggleIter(id) {
  var hdr = document.getElementById('hdr-' + id);
  var body = document.getElementById('body-' + id);
  var open = body.classList.contains('open');
  body.classList.toggle('open', !open);
  hdr.classList.toggle('open', !open);
}

function switchTab(iterId, tabName) {
  var panes = document.querySelectorAll('#body-' + iterId + ' .tab-pane');
  var btns  = document.querySelectorAll('#body-' + iterId + ' .tab-btn');
  panes.forEach(function(p) { p.classList.remove('active'); });
  btns.forEach(function(b) { b.classList.remove('active'); });
  var pane = document.getElementById('tab-' + iterId + '-' + tabName);
  var btn  = document.getElementById('btn-' + iterId + '-' + tabName);
  if (pane) pane.classList.add('active');
  if (btn)  btn.classList.add('active');
}

(function() {
  var overlay = document.getElementById('lbOverlay');
  var lbImg   = document.getElementById('lbImg');
  overlay.addEventListener('click', function() { overlay.classList.remove('open'); });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      overlay.classList.remove('open');
      document.getElementById('planOverlay').classList.remove('open');
    }
  });
  window.openLightbox = function(src) {
    lbImg.src = src;
    overlay.classList.add('open');
  };
})();

(function() {
  var planOverlay = document.getElementById('planOverlay');
  planOverlay.addEventListener('click', function(e) {
    if (e.target === planOverlay) planOverlay.classList.remove('open');
  });
  window.openPlanModal  = function() { planOverlay.classList.add('open'); };
  window.closePlanModal = function() { planOverlay.classList.remove('open'); };
})();

// Open first iteration by default
window.addEventListener('DOMContentLoaded', function() {
  var first = document.querySelector('.iter-card');
  if (first) {
    var id = first.dataset.iterid;
    if (id) toggleIter(id);
  }
});
"""


# ---------------------------------------------------------------------------
# PDDL syntax highlighting (pure string transforms)
# ---------------------------------------------------------------------------

def _highlight_pddl(pddl: str) -> str:
    import html as html_mod
    lines = []
    kws = {"define", "domain", "problem", "requirements", "types", "predicates",
           "action", "parameters", "precondition", "effect", "objects", "init",
           "goal", "and", "or", "not", "forall", "exists", "when", "increase",
           "decrease", "assign", "typing"}
    for raw_line in pddl.splitlines():
        safe = html_mod.escape(raw_line)
        # comments
        if re.match(r"^\s*;", raw_line):
            lines.append(f'<span class="cmt">{safe}</span>')
            continue
        # highlight keywords: :keyword
        safe = re.sub(
            r":([a-zA-Z][\w-]*)",
            lambda m: f'<span class="kw">:{m.group(1)}</span>',
            safe
        )
        # highlight known PDDL keywords as bare words
        def kw_replace(m: re.Match) -> str:
            w = m.group(0)
            if w.lower() in kws:
                return f'<span class="kw">{w}</span>'
            # variables
            if w.startswith("?"):
                return f'<span class="sym">{w}</span>'
            return w
        safe = re.sub(r"\?[\w-]+|[a-zA-Z][\w-]*", kw_replace, safe)
        lines.append(safe)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def _html_header(stats: dict[str, Any], status: str) -> str:
    import html as h
    ts = h.escape(stats["timestamp"])
    world = h.escape(stats["world"].capitalize())
    task = h.escape(stats["task"])

    if status == "completed":
        badge = '<span class="badge badge-success">&#10003; Completato</span>'
    elif status == "failed":
        badge = '<span class="badge badge-fail">&#10007; Fallito</span>'
    else:
        badge = '<span class="badge badge-warn">&#9651; Parziale</span>'

    return f"""
<header class="page-header">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <h1>VLM Robot Planner &mdash; Execution Report</h1>
    {badge}
  </div>
  <div class="meta" style="margin-top:10px;">
    <span><span class="meta-chip">Data</span> {ts}</span>
    <span><span class="meta-chip">Mondo</span> <strong>{world}</strong></span>
    <span><span class="meta-chip">Task</span> <em>{task}</em></span>
  </div>
</header>
"""


def _html_summary(stats: dict[str, Any]) -> str:
    n = stats["n_iters"]
    vlm_t = stats["vlm_time_total"]
    enrich = stats["enrichment_count"]
    steps = len(stats["completed_steps"])

    enrich_cls = "stat-box highlight" if enrich > 0 else "stat-box"
    enrich_sub = "nuove actions defined by VLM" if enrich > 0 else "no PDDL enrichment"

    return f"""
<section class="summary-bar">
  <div class="stat-box">
    <div class="stat-label">Iterazioni</div>
    <div class="stat-value">{n}</div>
    <div class="stat-sub">cicli VLM &rarr; robot</div>
  </div>
  <div class="{enrich_cls}">
    <div class="stat-label">Domain Enrichment</div>
    <div class="stat-value">{enrich}</div>
    <div class="stat-sub">{enrich_sub}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Tempo VLM totale</div>
    <div class="stat-value">{vlm_t:.1f}s</div>
    <div class="stat-sub">inferenza Qwen3-VL-8B</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Passi completati</div>
    <div class="stat-value">{steps}</div>
    <div class="stat-sub">primitives eseguiti OK</div>
  </div>
  <div style="display:flex;align-items:center;">
    <button class="btn-plan" onclick="openPlanModal()">&#9776; Piano Completo</button>
  </div>
</section>
"""


def _html_pipeline(stats: dict[str, Any]) -> str:
    enrich_count = stats["enrichment_count"]
    enrich = enrich_count > 0
    n_iters = stats["n_iters"]
    vlm_t = stats["vlm_time_total"]
    n_steps = len(stats["completed_steps"])

    def node(cls: str, label: str, sub: str, detail: str = "") -> str:
        detail_html = f'<div class="node-detail">{detail}</div>' if detail else ""
        return (f'<div class="pipe-node {cls}">'
                f'<div class="node-label">{label}</div>'
                f'<div class="node-sub">{sub}</div>'
                f'{detail_html}</div>')

    arrow = '<span class="pipe-arrow">&rarr;</span>'
    enricher_cls = "enriched" if enrich else "active"

    vlm_detail = f"{n_iters} iter. / {vlm_t:.0f}s totali"
    enrich_detail = (f"{enrich_count} azione/i nuove definit{('e' if enrich_count > 1 else 'a')} da zero"
                     if enrich else "dominio base sufficiente")
    fd_detail = f"{n_iters} problem{('i' if n_iters > 1 else '')} validat{('i' if n_iters > 1 else 'o')}"
    prim_detail = f"{n_steps} step{('s' if n_steps != 1 else '')} eseguiti"
    robot_detail = "Franka Panda (sim)"

    nodes = [
        node("active",       "VLM",            "Qwen3-VL-8B-Instruct",       vlm_detail),
        arrow,
        node(enricher_cls,   "Domain Enricher", "estensione PDDL dinamica",   enrich_detail),
        arrow,
        node("active",       "FastDownward",    "validazione simbolica",       fd_detail),
        arrow,
        node("active",       "Primitives",      "pick / place / novel acts",   prim_detail),
        arrow,
        node("active",       "Robot",           "esecuzione fisica",           robot_detail),
    ]
    inner = "\n".join(nodes)
    enrichment_note = ""
    if enrich:
        enrichment_note = ('<p style="margin-top:10px;font-size:12px;color:#d4a017;">'
                           '&#9733; La VLM ha ragionato autonomamente sulla semantica di azioni nuove </p>')

    return f"""
<section class="pipeline-section">
  <h2>Pipeline</h2>
  <div class="pipeline">{inner}</div>
  {enrichment_note}
</section>
"""


def _html_enrichment_box(new_actions: list[dict[str, Any]]) -> str:
    if not new_actions:
        return ""
    import html as h
    actions_html = []
    for act in new_actions:
        name = h.escape(str(act.get("name", "")))
        params = h.escape(str(act.get("parameters", "")))
        pre = h.escape(str(act.get("precondition", "")))
        eff = h.escape(str(act.get("effect", "")))
        actions_html.append(f"""
<div class="enrich-action">
  <div class="enrich-name">{name}</div>
  <div class="enrich-field">
    <div class="field-label">Parametri</div>
    <code>{params}</code>
  </div>
  <div class="enrich-field">
    <div class="field-label">Precondizioni</div>
    <code>{pre}</code>
  </div>
  <div class="enrich-field">
    <div class="field-label">Effetti</div>
    <code>{eff}</code>
  </div>
</div>
""")
    actions_inner = "".join(actions_html)
    return f"""
<div class="enrichment-box">
  <div class="enrich-header">Domain Enrichment</div>
  {actions_inner}
</div>
"""


def _html_images_tab(images: dict[str, str | None], iter_id: str) -> str:
    cells = []
    labels = {
        "overview": "Overview Camera",
        "overview_annotated": "Overview Annotata",
        "wrist": "Wrist Camera",
        "dino": "GroundingDINO (overview)",
        "dino_wrist": "GroundingDINO (wrist)",
    }
    for key, label in labels.items():
        src = images.get(key)
        if src:
            cells.append(f"""
<div class="img-cell">
  <div class="img-label">{label}</div>
  <img src="{src}" alt="{label}" onclick="openLightbox(this.src)" loading="lazy">
</div>
""")
    if not cells:
        return '<p style="color:var(--text-muted);font-size:13px;">Nessuna immagine disponibile per questa iterazione.</p>'
    return f'<div class="img-grid">{"".join(cells)}</div>'


def _format_args(args: Any) -> str:
    if isinstance(args, dict):
        return ", ".join(f"{k}={v}" for k, v in args.items())
    return str(args)


def _html_vlm_tab(debug: dict[str, Any], iter_id: str) -> str:
    import html as h

    vlm_time = debug.get("vlm_time_s", 0.0)
    frp = debug.get("full_remaining_plan", {})
    steps = frp.get("steps", [])
    new_actions = frp.get("domain_additions", {}).get("new_actions", [])

    # plan steps list
    steps_html_items = []
    for i, s in enumerate(steps, 1):
        prim = h.escape(str(s.get("primitive", "")))
        args_str = h.escape(_format_args(s.get("args", {})))
        steps_html_items.append(
            f'<li><span class="step-idx">{i}.</span>'
            f'<span><span class="step-primitive">{prim}</span> '
            f'<span class="step-args">{args_str}</span></span></li>'
        )
    steps_html = f'<ul class="step-list">{"".join(steps_html_items)}</ul>' if steps_html_items else \
        '<p style="color:var(--text-muted);font-size:13px;">Nessun passo rimanente.</p>'

    enrichment_html = _html_enrichment_box(new_actions)

    replan_note = ""
    raw_out = frp.get("raw_output", "")
    if isinstance(raw_out, str) and raw_out.startswith("[REPLAN:"):
        reason_match = re.match(r"\[REPLAN:\s*(.*?)\]", raw_out)
        if reason_match:
            reason = h.escape(reason_match.group(1))
            replan_note = f'<div class="badge badge-warn" style="margin-bottom:12px;">&#9654; Replanning: {reason}</div>'

    return f"""
{replan_note}
<div class="section-label" style="margin-bottom:10px;">
  Output VLM
  <span style="font-weight:400;font-size:11px;color:var(--text-muted);">
    &mdash; piano strutturato (JSON parsato, non testo grezzo) &bull;
    inferenza Qwen3-VL-8B-Instruct in <strong>{vlm_time:.1f}s</strong>
  </span>
</div>
{enrichment_html}
<div class="section-label">Piano rimanente ({len(steps)} passi)</div>
{steps_html}
"""


def _pddl_action_to_str(act: dict) -> str:
    name   = act.get("name", "?")
    params = act.get("parameters", "")
    pre    = act.get("precondition", "")
    eff    = act.get("effect", "")
    return (f"  (:action {name}\n"
            f"    :parameters {params}\n"
            f"    :precondition {pre}\n"
            f"    :effect {eff}\n"
            f"  )")


def _html_pddl_tab(debug: dict[str, Any]) -> str:
    import html as h

    pddl       = debug.get("pddl_problem", "")
    domain     = debug.get("domain_template", "")
    da         = debug.get("domain_additions") or {}
    new_preds  = da.get("new_predicates", [])
    new_acts   = da.get("new_actions", [])
    new_types  = da.get("new_types", [])

    domain_esc = h.escape(str(domain))

    # ── Problem PDDL ──────────────────────────────────────────────────────────
    if not pddl:
        problem_section = '<p style="color:var(--text-muted);">Nessun problema PDDL disponibile.</p>'
    else:
        problem_section = f'<pre class="code-block">{_highlight_pddl(pddl)}</pre>'

    # ── Domain additions (from this iteration's plan) ─────────────────────────
    domain_additions_section = ""
    if new_acts or new_preds or new_types:
        lines = [";;; domain additions generati dalla VLM per questo step"]
        if new_types:
            types_str = " ".join(str(t) for t in new_types)
            lines.append(f"  (:types {types_str})")
        if new_preds:
            lines.append("  (:predicates")
            for p in new_preds:
                lines.append(f"    {p}")
            lines.append("  )")
        for act in new_acts:
            lines.append(_pddl_action_to_str(act))
        additions_pddl = "\n".join(lines)
        domain_additions_section = f"""
<div class="section-label" style="margin-top:16px;color:var(--gold-text);">
  Domain Additions (generati dalla VLM &mdash; non nel template base)
</div>
<pre class="code-block" style="border-color:var(--gold-border);">{_highlight_pddl(additions_pddl)}</pre>
"""

    return f"""
<div class="section-label">
  <span style="font-weight:400;font-size:11px;color:var(--text-muted);">
    (1) il <strong>PDDL problem</strong> generato dinamicamente da questo step &mdash;
    oggetti, init state inferito dalla struttura del piano, goal derivato dagli effetti.
    (2) le <strong>domain additions</strong> generate dalla VLM per le azioni novel, se presenti.
    Il <strong>domain template</strong> di base usato &egrave;: <code>{domain_esc}</code>
  </span>
</div>
{problem_section}
{domain_additions_section}"""


def _html_exec_tab(debug: dict[str, Any]) -> str:
    import html as h

    prim = h.escape(str(debug.get("step_primitive", "")))
    args = debug.get("step_args", {})
    completed = debug.get("completed_steps", [])

    rows = ""
    if isinstance(args, dict):
        rows = "".join(
            f"<tr><td>{h.escape(str(k))}</td><td>{h.escape(str(v))}</td></tr>"
            for k, v in args.items()
        )
    args_table = f'<table class="prim-args-table"><tbody>{rows}</tbody></table>' if rows else \
        '<span style="color:var(--text-muted);">nessun argomento</span>'

    dino = debug.get("dino_estimates", {})
    dino_html = ""
    if dino:
        dino_items = "".join(
            f"<tr><td>{h.escape(str(obj))}</td><td>{h.escape(str(pos))}</td></tr>"
            for obj, pos in dino.items()
        )
        dino_html = f"""
<div class="section-label">Stime GroundingDINO (x, y in m)</div>
<table class="prim-args-table" style="font-family:monospace;">
  <thead><tr><td>Oggetto</td><td>Posizione</td></tr></thead>
  <tbody>{dino_items}</tbody>
</table>
"""

    chips = "".join(f'<span class="completed-chip">{h.escape(str(s))}</span>' for s in completed)
    completed_html = f'<div class="completed-steps">{chips}</div>' if chips else \
        '<span style="color:var(--text-muted);font-size:12px;">Nessun passo completato prima di questa iterazione.</span>'

    overview_note = ""
    if debug.get("using_overview_cam"):
        overview_note = '<span class="badge badge-blue" style="margin-left:8px;">Overview cam</span>'
    return f"""
<div class="exec-box">
  <div class="prim-name">{prim}{overview_note}</div>
  {args_table}
  </div>
{dino_html}
<div class="section-label">Passi completati prima di questa iterazione</div>
{completed_html}
"""


def _html_plan_modal(iters: list[tuple[int, Path, dict[str, Any]]], stats: dict[str, Any]) -> str:
    """Modal with the complete plan from iter_01 + completion status of each step."""
    import html as h

    if not iters:
        return '<div id="planOverlay"><div id="planModal"><div class="modal-body">Nessun dato.</div></div></div>'

    # Full plan = all steps as planned by VLM in iter_01
    first_debug = iters[0][2]
    frp = first_debug.get("full_remaining_plan", {})
    all_steps = frp.get("steps", [])

    # Completed steps set — extract primitive name for fuzzy match
    completed_raw = stats.get("completed_steps", [])
    completed_prims = []
    for cs in completed_raw:
        m = re.match(r"([a-zA-Z_]+)\(", cs)
        if m:
            completed_prims.append(m.group(1).lower())

    # Novel actions from enrichment
    new_actions = frp.get("domain_additions", {}).get("new_actions", [])
    novel_names = {a.get("name", "").lower() for a in new_actions}

    # Build step rows
    rows = []
    done_count = 0
    for i, step in enumerate(all_steps):
        prim = h.escape(str(step.get("primitive", "?")))
        args_str = h.escape(_format_args(step.get("args", {})))
        is_done = i < len(completed_prims) and completed_prims[i] == prim.lower()
        if is_done:
            done_count += 1
        status_icon = '<span class="ps-status ps-done">&#10003;</span>' if is_done else \
                      '<span class="ps-status ps-pending">&#9675;</span>'
        novel_badge = (' <span class="badge badge-gold" style="font-size:10px;padding:1px 6px;">'
                       'novel</span>') if prim.lower() in novel_names else ""
        rows.append(f"""
<div class="plan-step-row">
  <span class="ps-num">{i+1}.</span>
  <div class="ps-body">
    <span class="ps-prim">{prim}</span>{novel_badge}
    <span class="ps-args"> &nbsp;{args_str}</span>
  </div>
  {status_icon}
</div>""")

    rows_html = "".join(rows)
    n_total = len(all_steps)
    progress_pct = int(done_count / n_total * 100) if n_total else 0

    # Enrichment summary inside modal
    enrich_html = ""
    if new_actions:
        names = ", ".join(f"<code>{h.escape(a.get('name',''))}</code>" for a in new_actions)
        enrich_html = f"""
<div style="margin-top:16px;padding:10px 14px;background:var(--gold-bg);border:1px solid var(--gold-border);border-radius:6px;font-size:12px;">
  <strong style="color:var(--gold-text);">Domain Enrichment</strong>
  &mdash; azioni novel generate dalla VLM: {names}
</div>"""

    return f"""
<div id="planOverlay">
  <div id="planModal">
    <div class="modal-header">
      <h3>Piano Completo &mdash; {n_total} step &nbsp;
        <span style="font-size:12px;font-weight:400;color:var(--text-muted);">
          {done_count}/{n_total} completati ({progress_pct}%)
        </span>
      </h3>
      <button class="modal-close" onclick="closePlanModal()">&#10005;</button>
    </div>
    <div class="modal-body">
      {rows_html}
      {enrich_html}
    </div>
  </div>
</div>"""


def _has_enrichment(debug: dict[str, Any]) -> bool:
    frp = debug.get("full_remaining_plan", {})
    return bool(frp.get("domain_additions", {}).get("new_actions"))


def _html_iter_card(iter_num: int, debug: dict[str, Any], images: dict[str, str | None]) -> str:
    import html as h

    iter_id = f"i{iter_num}"
    prim = h.escape(str(debug.get("step_primitive", "?")))
    args_str = h.escape(_format_args(debug.get("step_args", {})))
    enriched = _has_enrichment(debug)
    card_cls = "iter-card has-enrichment" if enriched else "iter-card"
    enrich_badge = ' <span class="badge badge-gold">Domain Enrichment</span>' if enriched else ""

    tabs_def = [
        ("images", "&#9724; Immagini"),
        ("vlm",    "&#9670; VLM Output"),
        ("pddl",   "&#9636; PDDL"),
        ("exec",   "&#9881; Esecuzione"),
    ]

    tab_btns = "".join(
        f'<button class="tab-btn{" active" if i == 0 else ""}" '
        f'id="btn-{iter_id}-{key}" '
        f'onclick="switchTab(\'{iter_id}\', \'{key}\')">{label}</button>'
        for i, (key, label) in enumerate(tabs_def)
    )

    images_html = _html_images_tab(images, iter_id)
    vlm_html = _html_vlm_tab(debug, iter_id)
    pddl_html = _html_pddl_tab(debug)
    exec_html = _html_exec_tab(debug)

    panes = [
        ("images", images_html, True),
        ("vlm",    vlm_html,    False),
        ("pddl",   pddl_html,   False),
        ("exec",   exec_html,   False),
    ]
    panes_html = "".join(
        f'<div class="tab-pane{" active" if active else ""}" id="tab-{iter_id}-{key}">{content}</div>'
        for key, content, active in panes
    )

    return f"""
<div class="{card_cls}" data-iterid="{iter_id}">
  <div class="iter-header" id="hdr-{iter_id}" onclick="toggleIter('{iter_id}')">
    <div class="iter-num">{iter_num}</div>
    <div>
      <div class="iter-title">{prim}({args_str}){enrich_badge}</div>
    </div>
    <div class="iter-sub"></div>
    <span style="color:var(--text-muted);font-size:18px;">&#8964;</span>
  </div>
  <div class="iter-body" id="body-{iter_id}">
    <div class="tab-bar">{tab_btns}</div>
    {panes_html}
  </div>
</div>
"""


def _html_footer(stats: dict[str, Any]) -> str:
    steps = stats.get("completed_steps", [])
    n = len(steps)
    return f"""
<footer class="page-footer">
  Passi totali completati: <strong>{n}</strong> &nbsp;&bull;&nbsp;
</footer>
"""


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def generate_html_report(run_dir: Path) -> Path:
    """
    Read all iter_XX/debug.json from run_dir and write run_dir/report.html.
    Returns the path to the generated report.
    """
    run_dir = Path(run_dir)
    run_info = _load_run_info(run_dir)
    iters = _collect_iterations(run_dir)
    stats = _run_stats(iters, run_info)
    status = _run_status(stats)

    # Simple heuristic: any completed steps → at minimum 'partial'.
    if stats["completed_steps"]:
        status = "partial"
    # Check last iter for plan completion signal
    if iters:
        last = iters[-1][2]
        frp = last.get("full_remaining_plan", {})
        # raw_output may contain "complete": true
        raw = frp.get("raw_output", "")
        if isinstance(raw, str) and '"complete": true' in raw:
            status = "completed"

    cards_html = []
    for iter_num, iter_dir, debug in iters:
        images = _collect_images(run_dir, iter_num, iter_dir)
        cards_html.append(_html_iter_card(iter_num, debug, images))

    if not cards_html:
        cards_html_str = '<p style="color:var(--text-muted);">Nessuna iterazione trovata.</p>'
    else:
        cards_html_str = "\n".join(cards_html)

    plan_modal_html = _html_plan_modal(iters, stats)

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VLM Report &mdash; {stats.get("task", "")}</title>
<style>
{_css()}
</style>
</head>
<body>
{_html_header(stats, status)}
{_html_summary(stats)}
{_html_pipeline(stats)}
<section class="iterations">
  <h2>Iterazioni</h2>
  {cards_html_str}
</section>
{_html_footer(stats)}
<!-- lightbox -->
<div id="lbOverlay"><img id="lbImg" src="" alt="fullsize"></div>
<!-- plan modal -->
{plan_modal_html}
<script>
{_js()}
</script>
</body>
</html>
"""

    out_path = run_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python _generate_report.py <run_dir>", file=sys.stderr)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    report = generate_html_report(run_dir)
    print(f"Report generated: {report}")


if __name__ == "__main__":
    main()
