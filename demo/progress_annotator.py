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
import re
import logging
import threading
import uuid
from datetime import datetime, timezone
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
logger = logging.getLogger(__name__)


def _resolve_path(value: str | Path) -> Path:
    """Resolve path relative to project root unless absolute."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)

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
_summary_file_cfg = str(_cfg.get("summary_file", config.summary_file))
summary_file = _resolve_path(_summary_file_cfg)
trace_root_dir = _resolve_path(_cfg.get("trace_root_dir", "data/traces"))
annotation_dir = _resolve_path(_cfg.get("annotation_dir", config.annotation_dir))
annotations_root = PROJECT_ROOT / "data" / "annotations"
legacy_save_root_dir = annotations_root / annotator_id
_forced_default_task: str | None = None

# Singletons
plan_extractor = PlanExtractor(llm_client=None)
_summaries_cache: dict[str, dict[str, str]] = {}
_summaries_lock = threading.Lock()
_trace_summarizer = None
_summary_jobs: dict[str, dict] = {}
_summary_jobs_lock = threading.Lock()


def _clean_summary_preview(summary: str, max_len: int = 120) -> str:
    """Strip markdown heading prefixes from summary preview."""
    if not summary:
        return ""
    text = summary.replace("\n", " ").strip()
    text = re.sub(r"^#+\s*", "", text)
    return text[:max_len]


def _summary_file_for_task(task_name: str) -> Path:
    """Resolve the summary file path for a specific task."""
    if "{task}" in _summary_file_cfg:
        return _resolve_path(_summary_file_cfg.format(task=task_name))
    # Default: keep configured filename but isolate by task subdirectory.
    return summary_file.parent / task_name / summary_file.name


def _load_summaries(task_name: str) -> dict[str, str]:
    with _summaries_lock:
        if task_name not in _summaries_cache:
            task_summary_file = _summary_file_for_task(task_name)
            if not task_summary_file.exists():
                # Backward compatibility: if old shared summary file exists, load only
                # this task's trace_ids from it (avoid cross-task contamination).
                fallback = {}
                if task_summary_file != summary_file and summary_file.exists():
                    try:
                        legacy_raw = json.loads(summary_file.read_text())
                        if isinstance(legacy_raw, dict):
                            task_ids = set(_trace_ids_for_task(task_name))
                            fallback = {
                                tid: text
                                for tid, text in legacy_raw.items()
                                if tid in task_ids and str(text).strip()
                            }
                            if fallback:
                                logger.info(
                                    "Loaded %d summaries for task '%s' from legacy shared file: %s",
                                    len(fallback),
                                    task_name,
                                    summary_file,
                                )
                    except Exception:
                        logger.exception("Failed reading legacy shared summary file: %s", summary_file)

                if not fallback:
                    logger.warning(
                        "Summary file not found for task '%s': %s; continue with empty summaries",
                        task_name,
                        task_summary_file,
                    )
                _summaries_cache[task_name] = fallback
            else:
                with open(task_summary_file, "r") as f:
                    _summaries_cache[task_name] = json.load(f)
    return _summaries_cache[task_name]


def _save_summaries(task_name: str, summaries: dict[str, str]) -> None:
    """Persist summaries for a task and refresh cache."""
    with _summaries_lock:
        task_summary_file = _summary_file_for_task(task_name)
        task_summary_file.parent.mkdir(parents=True, exist_ok=True)
        task_summary_file.write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
        _summaries_cache[task_name] = summaries


def _get_trace_summarizer():
    """Lazy-init TraceSummarizer to avoid loading LLM deps unless needed."""
    global _trace_summarizer
    if _trace_summarizer is not None:
        return _trace_summarizer

    from src.llm import DEFAULT_MODEL, create_llm_client
    from src.utils.summarizer import TraceSummarizer

    summary_model = (
        (_cfg.get("summary_model") or "").strip()
        or os.environ.get("SUMMARY_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    max_completion_tokens = int(_cfg.get("summary_max_completion_tokens", 8000))
    client = create_llm_client(
        model=summary_model,
        temperature=0.0,
        max_completion_tokens=max_completion_tokens,
    )
    _trace_summarizer = TraceSummarizer(
        client=client,
        cache_path=None,
        max_completion_tokens=max_completion_tokens,
    )
    return _trace_summarizer


def _generate_summary_for_trace(task_name: str, trace_id: str) -> str:
    """Generate summary for one trace_id from raw spans."""
    trace_file = _trace_dir_for_task(task_name) / f"{trace_id}.json"
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace not found: {trace_file}")

    with open(trace_file, "r") as f:
        raw_trace = json.load(f)
    flat_spans = flatten_spans(raw_trace)
    if not flat_spans:
        return ""

    truncation_limit = int(_cfg.get("summary_truncation_limit", 2000))
    summarizer = _get_trace_summarizer()
    return summarizer.summarize(flat_spans, truncation_limit=truncation_limit)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_generate_summaries_job(job_id: str, task_name: str, missing_ids: list[str]) -> None:
    try:
        summaries = dict(_load_summaries(task_name))
        for idx, tid in enumerate(missing_ids, start=1):
            with _summary_jobs_lock:
                job = _summary_jobs.get(job_id)
                if not job:
                    return
                job["current_trace_id"] = tid
                job["processed"] = idx - 1

            try:
                summary = _generate_summary_for_trace(task_name, tid)
                if not str(summary).strip():
                    with _summary_jobs_lock:
                        _summary_jobs[job_id]["failed"].append({
                            "trace_id": tid,
                            "error": "Generated summary is empty",
                        })
                else:
                    summaries[tid] = summary
                    _save_summaries(task_name, summaries)
                    with _summary_jobs_lock:
                        _summary_jobs[job_id]["generated"] += 1
                        _summary_jobs[job_id]["generated_ids"].append(tid)
            except Exception as e:
                logger.exception("Failed to generate summary for %s/%s", task_name, tid)
                with _summary_jobs_lock:
                    _summary_jobs[job_id]["failed"].append({
                        "trace_id": tid,
                        "error": str(e),
                    })

            with _summary_jobs_lock:
                _summary_jobs[job_id]["processed"] = idx

        with _summary_jobs_lock:
            if job_id in _summary_jobs:
                _summary_jobs[job_id]["status"] = "completed"
                _summary_jobs[job_id]["current_trace_id"] = None
                _summary_jobs[job_id]["finished_at"] = _utc_now_iso()
    except Exception as e:
        logger.exception("Summary generation job crashed for task %s", task_name)
        with _summary_jobs_lock:
            if job_id in _summary_jobs:
                _summary_jobs[job_id]["status"] = "failed"
                _summary_jobs[job_id]["error"] = str(e)
                _summary_jobs[job_id]["current_trace_id"] = None
                _summary_jobs[job_id]["finished_at"] = _utc_now_iso()


def _list_tasks() -> list[str]:
    """List available task names under trace_root_dir."""
    if not trace_root_dir.exists():
        return []
    tasks = [
        p.name
        for p in sorted(trace_root_dir.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]
    return tasks


def _default_task_name() -> str | None:
    tasks = _list_tasks()
    if _forced_default_task and _forced_default_task in tasks:
        return _forced_default_task
    configured = (_cfg.get("default_task") or "").strip()
    if configured and configured in tasks:
        return configured
    return tasks[0] if tasks else None


def _resolve_task_name(task_name: str | None) -> str:
    task_name = (task_name or "").strip()
    tasks = _list_tasks()
    if not task_name:
        default_task = _default_task_name()
        if not default_task:
            raise ValueError(
                f"No tasks found under {trace_root_dir}. "
                "Expected directories like data/traces/GAIA"
            )
        return default_task
    if task_name not in tasks:
        raise ValueError(
            f"Unknown task '{task_name}'. Available tasks: "
            f"{', '.join(tasks) if tasks else '(none)'}"
        )
    return task_name


def _task_name_from_request() -> str:
    return _resolve_task_name(request.args.get("task"))


def _trace_dir_for_task(task_name: str) -> Path:
    return trace_root_dir / task_name


def _save_dir_for_task(task_name: str) -> Path:
    # Legacy layout: data/annotations/{annotator_id}/{task}
    return legacy_save_root_dir / task_name


def _saved_dir_for_task(task_name: str) -> Path:
    # New layout requested by user.
    return _trace_dir_for_task(task_name) / "saved"


def _excluded_dir_for_task(task_name: str) -> Path:
    # New layout requested by user.
    return _trace_dir_for_task(task_name) / "excluded"


def _saved_path_for_trace(task_name: str, trace_id: str) -> Path:
    return _saved_dir_for_task(task_name) / f"{trace_id}.json"


def _excluded_path_for_trace(task_name: str, trace_id: str) -> Path:
    return _excluded_dir_for_task(task_name) / f"{trace_id}.json"


def _gt_annotation_path(task_name: str, trace_id: str) -> Path:
    task_scoped = annotation_dir / task_name / f"{trace_id}.json"
    if task_scoped.exists():
        return task_scoped
    return annotation_dir / f"{trace_id}.json"


def _trace_ids_for_task(task_name: str) -> list[str]:
    trace_dir = _trace_dir_for_task(task_name)
    if not trace_dir.exists():
        return []
    return sorted(
        p.stem for p in trace_dir.glob("*.json")
        if p.is_file()
    )


def _load_annotation(task_name: str, trace_id: str) -> dict | None:
    """Load annotation for a trace from task-scoped save dir. Returns None if not found."""
    data, _ = _load_annotation_with_path(task_name, trace_id)
    return data


def _display_path(path: Path | None) -> str:
    """Render a readable path for UI."""
    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_annotation_with_path(task_name: str, trace_id: str) -> tuple[dict | None, Path | None]:
    """Load annotation and return both parsed data and source file path."""
    # Prefer new task-local layout under trace directory.
    candidates: list[Path] = []
    saved_path = _saved_path_for_trace(task_name, trace_id)
    excluded_path = _excluded_path_for_trace(task_name, trace_id)
    if saved_path.exists():
        candidates.append(saved_path)
    if excluded_path.exists():
        candidates.append(excluded_path)

    # Backward-compatible fallback for previous layouts.
    legacy_task_path = _save_dir_for_task(task_name) / f"{trace_id}.json"
    legacy_task_saved_path = _save_dir_for_task(task_name) / "saved" / f"{trace_id}.json"
    legacy_task_excluded_path = _save_dir_for_task(task_name) / "excluded" / f"{trace_id}.json"
    legacy_flat_path = legacy_save_root_dir / f"{trace_id}.json"
    if legacy_task_path.exists():
        candidates.append(legacy_task_path)
    if legacy_task_saved_path.exists():
        candidates.append(legacy_task_saved_path)
    if legacy_task_excluded_path.exists():
        candidates.append(legacy_task_excluded_path)
    if legacy_flat_path.exists():
        candidates.append(legacy_flat_path)

    if not candidates:
        return None, None

    # If multiple files exist, use the most recently modified one.
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    text = path.read_text().strip()
    if not text:
        return None, None
    data = json.loads(text)

    # Ensure excluded status matches new directory semantics.
    if path.parent.name == "excluded":
        data["excluded"] = True
    elif path.parent.name == "saved":
        data["excluded"] = False
    return data, path


# ── Routes ────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_file(
        Path(__file__).parent / "progress_annotator.html",
        mimetype="text/html",
    )


@app.route("/api/tasks")
def list_tasks():
    tasks = _list_tasks()
    return jsonify({
        "tasks": tasks,
        "default_task": _default_task_name(),
        "trace_root_dir": str(trace_root_dir),
    })


@app.route("/api/traces")
def list_traces():
    """List trace IDs for current task with summary preview and annotation status."""
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    summaries = _load_summaries(task_name)
    trace_ids = _trace_ids_for_task(task_name)
    traces = []
    for tid in trace_ids:
        annotation, annotation_path = _load_annotation_with_path(task_name, tid)
        saved = annotation is not None and not annotation.get("excluded", False)
        excluded = annotation is not None and annotation.get("excluded", False)
        has_gt = _gt_annotation_path(task_name, tid).exists()
        summary_text = summaries.get(tid, "")

        # Summary preview: first 120 chars
        summary_preview = _clean_summary_preview(summary_text, max_len=120)

        traces.append({
            "trace_id": tid,
            "task": task_name,
            "saved": saved,
            "excluded": excluded,
            "has_gt": has_gt,
            "has_summary": bool(str(summary_text).strip()),
            "annotation_path": _display_path(annotation_path),
            "summary_preview": summary_preview,
        })
    return jsonify(traces)


@app.route("/api/trace/<trace_id>/ensure_summary", methods=["POST"])
def ensure_trace_summary(trace_id: str):
    """Generate and persist summary for this trace if missing."""
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    summaries = dict(_load_summaries(task_name))
    existing = summaries.get(trace_id, "")
    if str(existing).strip():
        return jsonify({
            "ok": True,
            "generated": False,
            "trace_id": trace_id,
            "summary": existing,
            "summary_preview": _clean_summary_preview(existing, max_len=120),
        })

    try:
        summary = _generate_summary_for_trace(task_name, trace_id)
    except Exception as e:
        logger.exception("Failed to generate summary for %s/%s", task_name, trace_id)
        return jsonify({"ok": False, "error": str(e)}), 500

    if not str(summary).strip():
        return jsonify({"ok": False, "error": "Generated summary is empty"}), 500

    summaries[trace_id] = summary
    _save_summaries(task_name, summaries)
    return jsonify({
        "ok": True,
        "generated": True,
        "trace_id": trace_id,
        "summary": summary,
        "summary_preview": _clean_summary_preview(summary, max_len=120),
    })


@app.route("/api/summaries/generate_missing", methods=["POST"])
def generate_missing_summaries():
    """Start async job to batch-generate missing summaries for current task."""
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    payload = request.get_json(silent=True) or {}
    raw_limit = payload.get("limit", request.args.get("limit"))
    limit = 0
    if raw_limit not in (None, ""):
        try:
            limit = max(0, int(raw_limit))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "limit must be an integer"}), 400

    summaries = _load_summaries(task_name)
    trace_ids = _trace_ids_for_task(task_name)
    missing_ids = [tid for tid in trace_ids if not str(summaries.get(tid, "")).strip()]
    total_missing = len(missing_ids)
    if limit > 0:
        missing_ids = missing_ids[:limit]

    if not missing_ids:
        return jsonify({
            "ok": True,
            "task": task_name,
            "job_id": None,
            "status": "completed",
            "total_missing": total_missing,
            "requested": 0,
            "processed": 0,
            "generated": 0,
            "generated_ids": [],
            "failed": [],
        })

    job_id = uuid.uuid4().hex
    job = {
        "ok": True,
        "job_id": job_id,
        "task": task_name,
        "status": "running",
        "total_missing": total_missing,
        "requested": len(missing_ids),
        "processed": 0,
        "generated": 0,
        "generated_ids": [],
        "failed": [],
        "current_trace_id": None,
        "started_at": _utc_now_iso(),
        "finished_at": None,
    }
    with _summary_jobs_lock:
        _summary_jobs[job_id] = job

    t = threading.Thread(
        target=_run_generate_summaries_job,
        args=(job_id, task_name, missing_ids),
        daemon=True,
    )
    t.start()

    return jsonify(job), 202


@app.route("/api/summaries/jobs/<job_id>")
def get_summary_job(job_id: str):
    with _summary_jobs_lock:
        job = _summary_jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": f"job not found: {job_id}"}), 404
        return jsonify(job)


@app.route("/api/traces/stats")
def trace_stats():
    """Return annotation progress stats."""
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    trace_ids = _trace_ids_for_task(task_name)
    total = len(trace_ids)
    saved = 0
    excluded = 0
    for tid in trace_ids:
        ann = _load_annotation(task_name, tid)
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
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    summaries = _load_summaries(task_name)
    summary = summaries.get(trace_id, "")

    # Load raw trace
    trace_file = _trace_dir_for_task(task_name) / f"{trace_id}.json"
    if not trace_file.exists():
        return jsonify({"error": f"Trace not found for task '{task_name}'"}), 404

    task_description = "Task description not available"
    raw_spans = None
    hierarchical_spans = None
    with open(trace_file, "r") as f:
        raw_trace = json.load(f)
    raw_spans = flatten_spans(raw_trace)
    hierarchical_spans = raw_trace.get("spans", [])
    span_map = build_span_map(raw_trace)
    task_description = extract_task_description(span_map)

    # Extract plans
    plans = plan_extractor.extract_plans(trace_id, raw_spans, summary or "")
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
    anno_path = _gt_annotation_path(task_name, trace_id)
    if anno_path.exists():
        with open(anno_path, "r") as f:
            gt_annotation = json.load(f)

    # Load saved annotation
    saved_annotation, annotation_path = _load_annotation_with_path(task_name, trace_id)
    annotation_status = None
    if saved_annotation is not None:
        annotation_status = "excluded" if saved_annotation.get("excluded", False) else "saved"

    return jsonify({
        "trace_id": trace_id,
        "task": task_name,
        "summary": summary,
        "task_description": task_description,
        "extracted_plans": plans_data,
        "merged_plan": merged_data,
        "gt_annotation": gt_annotation,
        "saved_annotation": saved_annotation,
        "annotation_path": _display_path(annotation_path),
        "annotation_status": annotation_status,
        "spans": hierarchical_spans,
    })


@app.route("/api/trace/<trace_id>/compare")
def compare_trace(trace_id: str):
    """Get annotations from all annotators for side-by-side comparison."""
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = {}
    if not annotations_root.exists():
        return jsonify(result)
    for annotator_dir in sorted(annotations_root.iterdir()):
        if not annotator_dir.is_dir():
            continue
        ann_path = annotator_dir / task_name / f"{trace_id}.json"
        if not ann_path.exists():
            # Backward-compatible fallback for old flat annotation layout.
            ann_path = annotator_dir / f"{trace_id}.json"
        if ann_path.exists():
            text = ann_path.read_text().strip()
            if text:
                result[annotator_dir.name] = json.loads(text)
    # Also expose current task-local trace annotations.
    local_saved = _saved_path_for_trace(task_name, trace_id)
    local_excluded = _excluded_path_for_trace(task_name, trace_id)
    local_path = local_excluded if local_excluded.exists() else local_saved
    if local_path.exists():
        text = local_path.read_text().strip()
        if text:
            data = json.loads(text)
            data["excluded"] = local_path.parent.name == "excluded"
            result["task_local"] = data
    return jsonify(result)


@app.route("/api/trace/<trace_id>", methods=["PUT"])
def save_trace(trace_id: str):
    """Save progress annotation for a trace."""
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    data = request.get_json()
    save_dir = _saved_dir_for_task(task_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    excluded_path = _excluded_path_for_trace(task_name, trace_id)
    if excluded_path.exists():
        excluded_path.unlink()

    annotation = {
        "trace_id": trace_id,
        "task": task_name,
        "root_cause_step": data.get("root_cause_step", None),
        "root_cause_span_id": data.get("root_cause_span_id", None),
        "root_cause_reasoning": data.get("root_cause_reasoning", ""),
        "step_annotations": data.get("step_annotations", []),
        "notes": data.get("notes", ""),
        "excluded": False,
    }

    saved_path = _saved_path_for_trace(task_name, trace_id)
    saved_path.write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False) + "\n"
    )
    return jsonify({"ok": True, "annotation_path": _display_path(saved_path)})


@app.route("/api/trace/<trace_id>/exclude", methods=["PUT"])
def exclude_trace(trace_id: str):
    """Mark a trace as excluded."""
    try:
        task_name = _task_name_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    data = request.get_json()
    save_dir = _excluded_dir_for_task(task_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_path = _saved_path_for_trace(task_name, trace_id)
    if saved_path.exists():
        saved_path.unlink()

    annotation = {
        "trace_id": trace_id,
        "task": task_name,
        "root_cause_step": data.get("root_cause_step", None),
        "root_cause_span_id": data.get("root_cause_span_id", None),
        "root_cause_reasoning": data.get("root_cause_reasoning", ""),
        "step_annotations": data.get("step_annotations", []),
        "notes": data.get("notes", ""),
        "excluded": True,
    }

    excluded_path = _excluded_path_for_trace(task_name, trace_id)
    excluded_path.write_text(
        json.dumps(annotation, indent=2, ensure_ascii=False) + "\n"
    )
    return jsonify({"ok": True, "annotation_path": _display_path(excluded_path)})


# ── Main ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Root Cause Annotator")
    parser.add_argument("--port", type=int, default=6060)
    args = parser.parse_args()

    tasks = _list_tasks()
    default_task = _default_task_name()

    print(f"Annotator ID:   {annotator_id}")
    print(f"Summary base:   {summary_file}")
    if default_task:
        print(f"Summary file:   {_summary_file_for_task(default_task)} (default task)")
    print(f"Trace root:     {trace_root_dir}")
    print(f"Save dirs:      data/traces/{{task}}/saved  |  data/traces/{{task}}/excluded")
    print(f"Legacy save:    {legacy_save_root_dir}")
    print(f"Default task:   {default_task}")
    print(f"Tasks found:    {len(tasks)} ({', '.join(tasks) if tasks else 'none'})")
    print(f"Port:           {args.port}")
    if default_task:
        print(f"Summaries:      {len(_load_summaries(default_task))} (default task)")
    else:
        print("Summaries:      0")
    app.run(debug=True, port=args.port)
