"""Domain constants and enums for the graph package."""

from enum import Enum
from typing import Final


class SpanKind(str, Enum):
    """OpenInference span kinds for trace analysis."""

    AGENT = "AGENT"
    LLM = "LLM"
    TOOL = "TOOL"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, value: str | None) -> "SpanKind":
        """Convert string to SpanKind, defaulting to UNKNOWN."""
        if value is None:
            return cls.UNKNOWN
        try:
            return cls(value.upper())
        except ValueError:
            return cls.UNKNOWN


class EdgeType(str, Enum):
    """Edge types in the trace graph."""

    # Structural edges
    HIERARCHY = "hierarchy"
    CHILD_OF = "child_of"

    # Data flow edges
    DATA = "data"
    SEQUENCE = "sequence"
    TRANSITIVE_BUBBLE_UP = "transitive_bubble_up"
    AGENT_FORWARD_FLOW = "agent_forward_flow"
    DELEGATION_FLOW = "delegation_flow"

    @classmethod
    def data_flow_types(cls) -> frozenset["EdgeType"]:
        """Return the set of edge types that represent data flow."""
        return frozenset({
            cls.DATA,
            cls.TRANSITIVE_BUBBLE_UP,
            cls.AGENT_FORWARD_FLOW,
            cls.DELEGATION_FLOW,
        })


class StatusCode(str, Enum):
    """Span status codes."""

    OK = "Ok"
    ERROR = "Error"
    UNSET = "Unset"

    @classmethod
    def is_error(cls, value: str | None) -> bool:
        """Check if status code indicates an error."""
        return value == cls.ERROR.value


# Span attribute keys (OpenInference standard)
class SpanAttributes:
    """Common span attribute keys."""

    KIND: Final[str] = "openinference.span.kind"
    INPUT_VALUE: Final[str] = "input.value"
    OUTPUT_VALUE: Final[str] = "output.value"
    TOOL_NAME: Final[str] = "tool.name"
    TOOL_DESCRIPTION: Final[str] = "tool.description"
    LLM_MODEL: Final[str] = "llm.model_name"
    LLM_INVOCATION_PARAMS: Final[str] = "llm.invocation_parameters"
    INPUT_MIME_TYPE: Final[str] = "input.mime_type"
    OUTPUT_MIME_TYPE: Final[str] = "output.mime_type"
    # Tracegen-specific: explicit input data sources for deterministic edge creation
    INPUT_DATA_SOURCES: Final[str] = "smolagents.input_data_sources"


class TruncationLimits:
    """Limits for truncating string content to save tokens."""

    VERY_SHORT: Final[int] = 500
    SHORT: Final[int] = 1000
    MEDIUM: Final[int] = 2000
    LONG: Final[int] = 5000
    DEFAULT: Final[int] = 20000
    VERY_LONG: Final[int] = 40000
    TASK_DESCRIPTION: Final[int] = 20000

    # RCA-specific limits
    NODE_INPUT: Final[int] = 20000
    NODE_OUTPUT: Final[int] = 20000
    GOAL: Final[int] = 10000
    ANSWER: Final[int] = 10000
    DESCRIPTION: Final[int] = 5000
