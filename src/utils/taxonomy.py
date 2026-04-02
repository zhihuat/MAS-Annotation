"""Error taxonomy for trace analysis."""

import logging
from enum import Enum, unique
from typing import Optional

logger = logging.getLogger(__name__)


# Valid category names for prompt enforcement
VALID_CATEGORIES = [
    "Language-only",
    "Tool-related",
    "Poor Information Retrieval",
    "Tool Output Misinterpretation",
    "Incorrect Problem Identification",
    "Tool Selection Errors",
    "Formatting Errors",
    "Instruction Non-compliance",
    "Tool Definition Issues",
    "Environment Setup Errors",
    "Rate Limiting",
    "Authentication Errors",
    "Service Errors",
    "Resource Not Found",
    "Resource Exhaustion",
    "Timeout Issues",
    "Context Handling Failures",
    "Resource Abuse",
    "Goal Deviation",
    "Task Orchestration",
]


@unique
class ErrorCategory(str, Enum):
    """Standardized error categories for RCA (Leaf Nodes)."""

    # Reasoning Errors
    HALLUCINATION_LANGUAGE = "Language-only"
    HALLUCINATION_TOOL = "Tool-related"
    POOR_RETRIEVAL = "Poor Information Retrieval"
    TOOL_OUTPUT_MISINTERPRETATION = "Tool Output Misinterpretation"
    PROBLEM_IDENTIFICATION = "Incorrect Problem Identification"
    TOOL_SELECTION = "Tool Selection Errors"
    FORMATTING = "Formatting Errors"
    INSTRUCTION_NON_COMPLIANCE = "Instruction Non-compliance"

    # System Execution Errors
    TOOL_DEFINITION = "Tool Definition Issues"
    ENVIRONMENT_SETUP = "Environment Setup Errors"
    RATE_LIMITING = "Rate Limiting"
    AUTH_ERROR = "Authentication Errors"
    SERVICE_ERROR = "Service Errors"
    RESOURCE_NOT_FOUND = "Resource Not Found"
    RESOURCE_EXHAUSTION = "Resource Exhaustion"
    TIMEOUT = "Timeout Issues"

    # Planning Errors
    CONTEXT_HANDLING = "Context Handling Failures"
    RESOURCE_ABUSE = "Resource Abuse"
    GOAL_DEVIATION = "Goal Deviation"
    TASK_ORCHESTRATION = "Task Orchestration"

    OTHER = "Other"

    @classmethod
    def from_string(cls, value: str) -> "ErrorCategory":
        """Safe factory method to convert string to Enum."""
        # Try exact match first
        try:
            return cls(value)
        except ValueError:
            pass

        # Try case-insensitive lookup
        for member in cls:
            if member.value.lower() == value.lower():
                return member

        # Fallback to OTHER with warning
        logger.warning(f"Unknown category '{value}' mapped to OTHER")
        return cls.OTHER

    @classmethod
    def validate(cls, value: str) -> tuple[bool, Optional["ErrorCategory"]]:
        """Validate a category string.

        Returns:
            Tuple of (is_valid, matched_category or None)
        """
        # Try exact match
        try:
            return True, cls(value)
        except ValueError:
            pass

        # Try case-insensitive
        for member in cls:
            if member.value.lower() == value.lower():
                return True, member

        return False, None

    @classmethod
    def get_valid_categories_str(cls) -> str:
        """Get comma-separated list of valid categories for prompts."""
        return ", ".join([f'"{c}"' for c in VALID_CATEGORIES])
