"""Distilled Trace Schema - Contract between normalizer and RCA algorithms.

This module defines the standardized data structure for trace spans that RCA
algorithms operate on. By normalizing raw traces to this schema, we ensure:

1. Consistent field names regardless of input format (GAIA, OpenInference, etc.)
2. Clean separation between trace normalization and RCA analysis
3. Type-safe access to span properties
4. Easy extension for new trace formats

Architecture:
    Raw Trace (GAIA/OpenInference/etc.)
             │
             ▼
    ┌────────────────┐
    │TraceNormalizer │  ← Normalizes to standard schema
    └───────┬────────┘
            │
            ▼
    ┌────────────────┐
    │DistilledTrace  │  ← This schema (contract)
    │  - trace_id    │
    │  - spans: Dict[span_id, DistilledSpan]
    │  - metadata    │
    └───────┬────────┘
            │
            ▼
       ┌──────────┐
       │RCAService│  → Algorithm.analyze(DistilledTrace, graph)
       └──────────┘
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field, PrivateAttr

from src.graph.constants import SpanKind, StatusCode


class SourceFormat(str, Enum):
    """Known trace source formats."""

    GAIA = "gaia"
    OPENINFERENCE = "openinference"
    TRACEGEN = "tracegen"  # Tracegen format with explicit input_data_sources
    UNKNOWN = "unknown"


class DistilledEvent(BaseModel):
    """Normalized event within a span.

    Events are discrete occurrences within a span's lifetime,
    such as exceptions, logs, or checkpoints.
    """

    name: str
    timestamp: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class DistilledSpan(BaseModel):
    """Normalized span with consistent field names.

    This is the core unit of trace data that RCA algorithms operate on.
    All span-level analysis should use this schema's properties rather
    than accessing raw dictionaries.

    Required Fields:
        span_id: Unique identifier for this span
        span_name: Human-readable name describing the operation

    Kept Top-Level Fields:
        span_id, span_name, span_kind, parent_span_id, span_attributes,
        duration, status_code, status_message, events

    Kept span_attributes Fields:
        input.value, output.value, openinference.span.kind, llm.model_name,
        llm.token_count.prompt, llm.token_count.completion, llm.token_count.total
    """

    # Required identifiers
    span_id: str
    span_name: str

    # Classification and status
    span_kind: SpanKind = SpanKind.UNKNOWN
    status_code: StatusCode = StatusCode.UNSET
    status_message: str | None = None

    # Hierarchy
    parent_span_id: str | None = None

    # Keep only whitelisted span attributes from raw trace.
    span_attributes: dict[str, Any] = Field(default_factory=dict)

    # ISO 8601 duration string from raw trace, e.g. "PT1.23S"
    duration: str | None = None

    # Events and logs
    events: list[DistilledEvent] = Field(default_factory=list)
    # Internal ordering hint from raw span timestamp (not exported).
    _timestamp: str | None = PrivateAttr(default=None)

    @property
    def is_error(self) -> bool:
        """Check if this span has an error status."""
        return self.status_code == StatusCode.ERROR

    @property
    def is_agent(self) -> bool:
        """Check if this is an agent span."""
        return self.span_kind == SpanKind.AGENT

    @property
    def is_llm(self) -> bool:
        """Check if this is an LLM span."""
        return self.span_kind == SpanKind.LLM

    @property
    def is_tool(self) -> bool:
        """Check if this is a tool span."""
        return self.span_kind == SpanKind.TOOL

    @property
    def timestamp(self) -> str | None:
        """Best-effort timestamp for ordering compatibility.

        DistilledSpan no longer stores top-level `timestamp` as a public field.
        Prefer internal raw timestamp captured at construction time; fallback to
        the first event timestamp.
        """
        if self._timestamp:
            return self._timestamp
        for event in self.events:
            if event.timestamp:
                return event.timestamp
        return None

    def get_effective_input(self, max_length: int | None = None) -> str:
        """Get the effective input value with optional truncation.

        Reads from span_attributes["input.value"].

        Args:
            max_length: Optional maximum length for truncation

        Returns:
            Input string or empty string if none available
        """
        import json

        raw = self.span_attributes.get("input.value")
        if raw is None:
            value = ""
        else:
            value = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

        if max_length and len(value) > max_length:
            return value[:max_length] + "... [TRUNCATED]"
        return value

    def get_effective_output(self, max_length: int | None = None) -> str:
        """Get the effective output value with optional truncation.

        Reads from span_attributes["output.value"].

        Args:
            max_length: Optional maximum length for truncation

        Returns:
            Output string or empty string if none available
        """
        import json

        raw = self.span_attributes.get("output.value")
        if raw is None:
            value = ""
        else:
            value = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)

        if max_length and len(value) > max_length:
            return value[:max_length] + "... [TRUNCATED]"
        return value

    def has_exception_event(self) -> bool:
        """Check if span has any exception events."""
        for event in self.events:
            if event.name == "exception":
                return True
            if "exception" in event.attributes:
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dict matching the legacy span format expected by some code
        """
        from src.utils.trace_utils import _KEEP_ATTR_KEYS, _KEEP_SPAN_KEYS

        result: dict[str, Any] = {
            "span_id": self.span_id,
            "span_name": self.span_name,
            "span_kind": self.span_kind.value,
            "status_code": self.status_code.value,
            "parent_span_id": self.parent_span_id,
            "span_attributes": dict(self.span_attributes),
        }

        if self.status_message:
            result["status_message"] = self.status_message

        if self.duration:
            result["duration"] = self.duration

        raw_attrs = result.get("span_attributes", {})
        attrs = {key: value for key, value in raw_attrs.items() if key in _KEEP_ATTR_KEYS}
        result["span_attributes"] = attrs

        if self.events:
            result["events"] = [
                {"Name": e.name, "Timestamp": e.timestamp, "Attributes": e.attributes}
                for e in self.events
            ]

        # Keep only whitelisted top-level keys for downstream consistency.
        filtered = {key: value for key, value in result.items() if key in _KEEP_SPAN_KEYS}
        filtered.setdefault("span_attributes", {})
        return filtered


class DistilledTrace(BaseModel):
    """Normalized trace container with metadata.

    This is the top-level schema that RCA algorithms receive.
    It provides:
    - Direct access to spans by ID
    - Computed metrics for quick filtering
    - Metadata about the original trace format
    """

    # Core data
    trace_id: str
    spans: dict[str, DistilledSpan] = Field(default_factory=dict)

    # Source metadata
    source_format: SourceFormat = SourceFormat.UNKNOWN

    # Task-level context (extracted from trace)
    task_description: str | None = None
    final_answer: str | None = None
    task_succeeded: bool | None = None

    # Raw trace reference for edge cases
    raw_trace: dict[str, Any] | None = Field(default=None, exclude=True)

    @computed_field
    @property
    def total_spans(self) -> int:
        """Total number of spans in the trace."""
        return len(self.spans)

    @computed_field
    @property
    def agent_count(self) -> int:
        """Number of AGENT spans."""
        return sum(1 for s in self.spans.values() if s.span_kind == SpanKind.AGENT)

    @computed_field
    @property
    def llm_count(self) -> int:
        """Number of LLM spans."""
        return sum(1 for s in self.spans.values() if s.span_kind == SpanKind.LLM)

    @computed_field
    @property
    def tool_count(self) -> int:
        """Number of TOOL spans."""
        return sum(1 for s in self.spans.values() if s.span_kind == SpanKind.TOOL)

    @computed_field
    @property
    def error_count(self) -> int:
        """Number of spans with ERROR status."""
        return sum(1 for s in self.spans.values() if s.is_error)

    def get_span(self, span_id: str) -> DistilledSpan | None:
        """Get a span by ID.

        Args:
            span_id: The span ID to look up

        Returns:
            DistilledSpan if found, None otherwise
        """
        return self.spans.get(span_id)

    def get_spans_by_kind(self, kind: SpanKind) -> list[DistilledSpan]:
        """Get all spans of a specific kind.

        Args:
            kind: SpanKind to filter by

        Returns:
            List of matching DistilledSpan objects
        """
        return [s for s in self.spans.values() if s.span_kind == kind]

    def get_error_spans(self) -> list[DistilledSpan]:
        """Get all spans with ERROR status.

        Returns:
            List of DistilledSpan objects with error status
        """
        return [s for s in self.spans.values() if s.is_error]

    def get_root_spans(self) -> list[DistilledSpan]:
        """Get spans without parents (root spans).

        Returns:
            List of DistilledSpan objects that are roots
        """
        return [s for s in self.spans.values() if s.parent_span_id is None]

    def get_children(self, parent_id: str) -> list[DistilledSpan]:
        """Get all direct children of a span.

        Args:
            parent_id: ID of the parent span

        Returns:
            List of child DistilledSpan objects
        """
        return [s for s in self.spans.values() if s.parent_span_id == parent_id]

    def iter_spans_by_timestamp(self) -> list[DistilledSpan]:
        """Get spans sorted by timestamp.

        Returns:
            List of DistilledSpan objects sorted by timestamp
        """
        return sorted(
            self.spans.values(), key=lambda s: s.timestamp or "", reverse=False
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        """Convert to legacy trace dict format for backward compatibility.

        This allows gradual migration of code that expects the old format.

        Returns:
            Dict matching the legacy hierarchical trace format
        """
        # Build hierarchical spans from flat map
        root_spans = []

        # Find root spans and build hierarchy
        def build_hierarchy(span_id: str) -> dict[str, Any]:
            span = self.spans[span_id]
            result = span.to_dict()
            children = self.get_children(span_id)
            if children:
                result["child_spans"] = [
                    build_hierarchy(c.span_id) for c in sorted(children, key=lambda s: s.timestamp or "")
                ]
            return result

        for span in self.get_root_spans():
            root_spans.append(build_hierarchy(span.span_id))

        return {
            "trace_id": self.trace_id,
            "spans": root_spans,
        }
