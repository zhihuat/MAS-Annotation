#!/usr/bin/env python3
"""
Flask backend for progress annotation tool.

Provides APIs for:
- Listing traces (from summaries + raw trace files)
- Extracting task descriptions from raw traces
- Extracting plans via PlanExtractor
- Loading/saving progress annotations (plan, root cause assignment)

Launch:
    python demo/progress_annotator.py
    python demo/progress_annotator.py --port 6060
"""

import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util as _ilu
import types as _types
import yaml
from flask import Flask, jsonify, request, send_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# src/progress_monitor/__init__.py imports FlatlineDetector → value_function → src.llm,
# which pulls in heavy deps. Block it by pre-registering a stub package, then load
# only the two submodules we need directly from their files.
def _load_direct(dotted_name: str, file_path: Path):
    """Load a module from its file, bypassing its package __init__."""
    spec = _ilu.spec_from_file_location(dotted_name, file_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Register stub for src.progress_monitor so its __init__.py never runs
if "src.progress_monitor" not in sys.modules:
    _pm_stub = _types.ModuleType("src.progress_monitor")
    _pm_stub.__path__ = [str(PROJECT_ROOT / "src" / "progress_monitor")]
    _pm_stub.__package__ = "src.progress_monitor"
    sys.modules["src.progress_monitor"] = _pm_stub

# Stub src.graph.constants — exists only in parent mas_error_analysis repo
if "src.graph" not in sys.modules:
    _graph = _types.ModuleType("src.graph")
    _graph.__path__ = []
    sys.modules["src.graph"] = _graph

    class _SpanAttributes:
        INPUT_VALUE = "input.value"

    class _TruncationLimits:
        TASK_DESCRIPTION = 2000

    _gc = _types.ModuleType("src.graph.constants")
    _gc.SpanAttributes = _SpanAttributes
    _gc.TruncationLimits = _TruncationLimits
    sys.modules["src.graph.constants"] = _gc

_src = PROJECT_ROOT / "src"
_load_direct("src.progress_monitor.config",         _src / "progress_monitor" / "config.py")
_load_direct("src.utils.trace_utils",               _src / "utils" / "trace_utils.py")
_load_direct("src.progress_monitor.plan_extractor", _src / "progress_monitor" / "plan_extractor.py")

from src.progress_monitor.config import ProgressMonitorConfig
from src.progress_monitor.plan_extractor import PlanExtractor
from src.utils.trace_utils import build_span_map, extract_task_description, flatten_spans

app = Flask(__name__)

# Load config.yaml
_config_path = PROJECT_ROOT / "config.yaml"
if not _config_path.exists():
    raise FileNotFoundError(
        f"config.yaml not found at {_config_path}. "
        "Create it with at least:\n  annotator_id: your_name"
    )
with open(_config_path) as f:
    _cfg = yaml.safe_load(f)

annotator_id: str = _cfg.get("annotator_id", "default")

# Directories
config = ProgressMonitorConfig()
summary_file = PROJECT_ROOT / _cfg.get("summary_file", config.summary_file)
trace_dir = PROJECT_ROOT / _cfg.get("trace_dir", config.trace_dir)
annotation_dir = PROJECT_ROOT / _cfg.get("annotation_dir", config.annotation_dir)
save_dir = PROJECT_ROOT / "data" / "annotations" / annotator_id
annotations_root = PROJECT_ROOT / "data" / "annotations"

# Singletons
plan_extractor = PlanExtractor(llm_client=None)
_summaries_cache: dict[str, str] | None = None


def _load_summaries() -> dict[str, str]:
    global _summaries_cache
    if _summaries_cache is None:
        with open(summary_file, "r") as f:
            _summaries_cache = json.load(f)
    return _summaries_cache


def _load_annotation(trace_id: str) -> dict | None:
    """Load annotation for a trace from save_dir. Returns None if not found."""
    path = save_dir / f"{trace_id}.json"
    if not path.exists():
        return None
    text = path.read_text().strip()
    if not text:
        return None
    return json.loads(text)


# ── Routes ────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_file(
        Path(__file__).parent / "progress_annotator.html",
        mimetype="text/html",
    )


@app.route("/api/traces")
def list_traces():
    """List all trace IDs that have both a summary and a raw trace file."""
    summaries = _load_summaries()
    traces = []
    for tid in sorted(summaries.keys()):
        trace_file = trace_dir / f"{tid}.json"
        if not trace_file.exists():
            continue

        annotation = _load_annotation(tid)
        saved = annotation is not None and not annotation.get("excluded", False)
        excluded = annotation is not None and annotation.get("excluded", False)
        has_gt = (annotation_dir / f"{tid}.json").exists()

        # Summary preview: first 120 chars
        summary_preview = summaries[tid][:120].replace("\n", " ") if summaries[tid] else ""

        traces.append({
            "trace_id": tid,
            "saved": saved,
            "excluded": excluded,
            "has_gt": has_gt,
            "summary_preview": summary_preview,
        })
    return jsonify(traces)


@app.route("/api/traces/stats")
def trace_stats():
    """Return annotation progress stats."""
    summaries = _load_summaries()
    total = sum(1 for tid in summaries if (trace_dir / f"{tid}.json").exists())
    saved = 0
    excluded = 0
    for tid in summaries:
        ann = _load_annotation(tid)
        if ann is None:
            continue
        if ann.get("excluded", False):
            excluded += 1
        else:
            saved += 1
    return jsonify({"total": total, "saved": saved, "excluded": excluded})


@app.route("/api/trace/<trace_id>")
def get_trace(trace_id: str):
    """Load full trace data: summary, task description, plans, and annotations."""
    summaries = _load_summaries()
    if trace_id not in summaries:
        return jsonify({"error": "Trace not found in summaries"}), 404

    summary = summaries[trace_id]

    # Load raw trace
    trace_file = trace_dir / f"{trace_id}.json"
    task_description = "Task description not available"
    raw_spans = None
    hierarchical_spans = None
    if trace_file.exists():
        with open(trace_file, "r") as f:
            raw_trace = json.load(f)
        raw_spans = flatten_spans(raw_trace)
        hierarchical_spans = raw_trace.get("spans", [])
        span_map = build_span_map(raw_trace)
        task_description = extract_task_description(span_map)

    # Extract plans
    plans = plan_extractor.extract_plans(trace_id, raw_spans, summary)
    merged = plan_extractor.consolidate_plans(task_description, plans, trace_id)

    plans_data = []
    for p in plans:
        plans_data.append({
            "span_id": p.span_id,
            "plan_type": p.plan_type,
            "steps": [
                {"step_number": s.step_number, "description": s.description, "depends_on": s.depends_on}
                for s in p.steps
            ],
            "raw_text": p.raw_text,
        })

    merged_data = [
        {"step_number": s.step_number, "description": s.description, "depends_on": s.depends_on}
        for s in merged
    ]

    # Load root cause annotation (GT) if available
    gt_annotation = None
    anno_path = annotation_dir / f"{trace_id}.json"
    if anno_path.exists():
        with open(anno_path, "r") as f:
            gt_annotation = json.load(f)

    # Load saved annotation
    saved_annotation = _load_annotation(trace_id)
    annotation_status = None
    if saved_annotation is not None:
        annotation_status = "excluded" if saved_annotation.get("excluded", False) else "saved"

    return jsonify({
        "trace_id": trace_id,
        "summary": summary,
        "task_description": task_description,
        "extracted_plans": plans_data,
        "merged_plan": merged_data,
        "gt_annotation": gt_annotation,
        "saved_annotation": saved_annotation,
        "annotation_status": annotation_status,
        "spans": hierarchical_spans,
    })


@app.route("/api/trace/<trace_id>/compare")
def compare_trace(trace_id: str):
    """Get annotations from all annotators for side-by-side comparison."""
    result = {}
    if not annotations_root.exists():
        return jsonify(result)
    for annotator_dir in sorted(annotations_root.iterdir()):
        if not annotator_dir.is_dir():
            continue
        ann_path = annotator_dir / f"{trace_id}.json"
        if ann_path.exists():
            text = ann_path.read_text().strip()
            if text:
                result[annotator_dir.name] = json.loads(text)
    return jsonify(result)


@app.route("/api/trace/<trace_id>", methods=["PUT"])
def save_trace(trace_id: str):
    """Save progress annotation for a trace."""
    data = request.get_json()
    save_dir.mkdir(parents=True, exist_ok=True)

    annotation = {
        "trace_id": trace_id,
        "finalized_plan": data.get("finalized_plan", []),
        "root_cause_step": data.get("root_cause_step", None),
        "root_cause_span_id": data.get("root_cause_span_id", None),
        "root_cause_reasoning": data.get("root_cause_reasoning", ""),
        "step_annotations": data.get("step_annotations", []),
        "notes": data.get("notes", ""),
        "excluded": False,
    }

    (save_dir / f"{trace_id}.json").write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False) + "\n"
    )
    return jsonify({"ok": True})


@app.route("/api/trace/<trace_id>/exclude", methods=["PUT"])
def exclude_trace(trace_id: str):
    """Mark a trace as excluded."""
    data = request.get_json()
    save_dir.mkdir(parents=True, exist_ok=True)

    annotation = {
        "trace_id": trace_id,
        "finalized_plan": data.get("finalized_plan", []),
        "root_cause_step": data.get("root_cause_step", None),
        "root_cause_span_id": data.get("root_cause_span_id", None),
        "root_cause_reasoning": data.get("root_cause_reasoning", ""),
        "step_annotations": data.get("step_annotations", []),
        "notes": data.get("notes", ""),
        "excluded": True,
    }

    (save_dir / f"{trace_id}.json").write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False) + "\n"
    )
    return jsonify({"ok": True})


# ── Main ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Progress Annotator")
    parser.add_argument("--port", type=int, default=6060)
    args = parser.parse_args()

    print(f"Annotator ID:   {annotator_id}")
    print(f"Summary file:   {summary_file}")
    print(f"Trace dir:      {trace_dir}")
    print(f"Save dir:       {save_dir}")
    print(f"Port:           {args.port}")
    print(f"Summaries:      {len(_load_summaries())}")
    app.run(debug=True, port=args.port)
