"""Trace and span utilities.

Common functions for extracting information from trace data structures.
Supports both DistilledTrace (pydantic) and raw dict traces.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Union

from src.graph.constants import SpanAttributes, TruncationLimits

if TYPE_CHECKING:
    from src.graph.distilled_trace import DistilledTrace

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Trace truncation — shared by TraceSummarizer and TraceDetector
# -----------------------------------------------------------------------

# Top-level span keys to keep (drop trace_id, timestamp, trace_state,
# service_name, resource_attributes, scope_name, scope_version, links, logs).
_KEEP_SPAN_KEYS = {
    "span_id",
    "span_name",
    "span_kind",
    "parent_span_id",
    "span_attributes",
    "duration",
    "status_code",
    "status_message",
    "events",
}

# span_attributes keys to keep (drop pat.*, telemetry.*,
# llm.input_messages.*, llm.output_messages.* which are
# redundant with input.value / output.value).
_KEEP_ATTR_KEYS = {
    "input.value",
    "output.value",
    "openinference.span.kind",
    "llm.model_name",
    "llm.token_count.prompt",
    "llm.token_count.completion",
    "llm.token_count.total",
}


def smart_truncate(value: Any, limit: int) -> str:
    """Structure-aware truncation for input/output values.

    If *value* is a messages-style list
    (``[{"role": ..., "content": ...}, ...]``), each message's role is
    preserved and its content truncated individually.  System messages
    get the full *limit* budget while user/assistant messages receive
    ``limit // 3``.

    For plain strings or other structures, falls back to simple
    character-level truncation.
    """
    if not value:
        return ""

    # Unwrap {"messages": [...]} wrapper
    if isinstance(value, dict) and "messages" in value:
        value = value["messages"]

    if isinstance(value, list) and value and isinstance(value[0], dict):
        msgs = value
        if all("role" in m for m in msgs):
            system_budget = limit
            other_budget = limit // 3

            truncated_msgs = []
            for m in msgs:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                budget = system_budget if role == "system" else other_budget

                if len(content[0].get('text', '')) > budget:
                    content[0]['text'] = content[0]['text'][:budget] + "..."
                truncated_msgs.append({"role": role, "content": content})

            return json.dumps(truncated_msgs, ensure_ascii=False)

    # Fallback: plain string truncation
    text = str(value)
    if len(text) > limit:
        text = text[:limit] + "..."
    return text


def truncate_spans(
    flat_spans: list[dict[str, Any]],
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Strip metadata and truncate long values from a flat list of spans.

    The function:
    * keeps only the span keys listed in ``_KEEP_SPAN_KEYS``,
    * filters ``span_attributes`` to ``_KEEP_ATTR_KEYS``,
    * parses JSON-encoded ``input.value`` and applies
      :func:`smart_truncate`.

    Args:
        flat_spans: List of raw span dicts.
        limit: Maximum character length for individual attribute values
            before truncation.

    Returns:
        A new list of truncated span dicts (originals are not mutated).
    """
    processed: list[dict[str, Any]] = []
    for span in flat_spans:
        s = {k: v for k, v in span.items() if k in _KEEP_SPAN_KEYS}

        raw_attrs = span.get("span_attributes", {})
        attrs = {k: v for k, v in raw_attrs.items() if k in _KEEP_ATTR_KEYS}
        s["span_attributes"] = attrs

        for key in ["input.value", "output.value"]:
            if key in attrs:
                raw = attrs[key]
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    parsed = raw
                attrs[key] = smart_truncate(parsed, limit)

        processed.append(s)
    return processed


def flatten_spans(raw_trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a raw trace's spans into a flat list.

    Handles both:
    - Flat dict format: ``{"spans": {span_id: span_data, ...}}``
    - Hierarchical list format: ``{"spans": [{..., "child_spans": [...]}, ...]}``
    """
    top = raw_trace.get("spans", [])
    if isinstance(top, dict):
        result = []
        for sid, span in top.items():
            s = dict(span)
            s.setdefault("span_id", sid)
            result.append(s)
        return result

    out: list[dict[str, Any]] = []

    def walk(span: dict[str, Any], parent_id: str | None = None) -> None:
        flat = {k: v for k, v in span.items() if k != "child_spans"}
        if parent_id is not None:
            flat.setdefault("parent_span_id", parent_id)
        out.append(flat)
        sid = flat.get("span_id")
        for child in span.get("child_spans", []):
            walk(child, sid)

    for s in top:
        walk(s)
    return out


def get_trace_id(trace: Union["DistilledTrace", dict[str, Any]]) -> str:
    """Get trace_id from either DistilledTrace or raw dict."""
    if isinstance(trace, dict):
        return trace.get("trace_id", "unknown")
    return getattr(trace, "trace_id", "unknown")


def build_span_map(
    trace: Union["DistilledTrace", dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Build a span_id -> span_data mapping from trace.

    For DistilledTrace: Returns span.to_dict() for each span.
    For raw dict: Uses flatten_spans and indexes by span_id.
    """
    if not isinstance(trace, dict):
        # DistilledTrace-like object path: rely on shape instead of runtime import
        # so this utility works in different execution environments.
        spans_obj = getattr(trace, "spans", {})
        if isinstance(spans_obj, dict):
            out: dict[str, dict[str, Any]] = {}
            for span_id, span in spans_obj.items():
                if hasattr(span, "to_dict"):
                    out[span_id] = span.to_dict()
                elif hasattr(span, "model_dump"):
                    out[span_id] = span.model_dump()
                elif isinstance(span, dict):
                    out[span_id] = span
                else:
                    out[span_id] = {"span_id": span_id}
            return out

    spans = flatten_spans(trace)
    return {s["span_id"]: s for s in spans if s.get("span_id")}


def _spans_as_iterable(
    spans: Union[dict[str, dict[str, Any]], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Normalize spans (dict map or flat list) to a list for iteration."""
    if isinstance(spans, dict):
        return list(spans.values())
    return spans


def extract_task_description(
    spans: Union[dict[str, dict[str, Any]], list[dict[str, Any]]],
    max_length: int = TruncationLimits.TASK_DESCRIPTION
) -> str:
    """Extract task description from trace spans.

    Accepts either a span_map (dict keyed by span_id) or a flat list of spans.

    Tries multiple methods:
    1. Look for get_examples_to_answer span (GAIA benchmark specific)
    2. Look for Agent.run input with task field
    """
    span_list = _spans_as_iterable(spans)

    # Method 1: Look for get_examples_to_answer span (GAIA specific)
    for span in span_list:
        span_name = span.get("span_name", "")
        if span_name == "get_examples_to_answer":
            logs = span.get("logs", [])
            for log in logs:
                body = log.get("body", {})
                func_output = body.get("function.output", [])
                if isinstance(func_output, list) and len(func_output) > 0:
                    question = func_output[0].get("question", "")
                    if question:
                        return question[:max_length]

    # Method 2: Fallback to Agent.run input
    for span in span_list:
        span_name = span.get("span_name", "")
        if "CodeAgent.run" in span_name or "Agent.run" in span_name:
            attrs = span.get("span_attributes", {})
            input_val = attrs.get(SpanAttributes.INPUT_VALUE, "")
            if input_val:    
                input_data = json.loads(input_val) if isinstance(input_val, str) else input_val
                task = input_data.get("task", "")
                idx = task.find("Here is the task:\n")
                if idx != -1:
                    task = task[idx + len("Here is the task:\n"):]
                    return task
