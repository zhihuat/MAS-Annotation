"""
Plan extraction from trace summaries.

Extracts agent plans (initial + revised) from hierarchical narrative summaries
using broad regex patterns with LLM fallback, then consolidates into a single
finalized plan via LLM.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src.llm.interface import BaseLLMClient, Message
import textwrap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PlanStep:
    """A single step within an extracted plan."""

    step_number: int
    description: str
    depends_on: list[int] = field(default_factory=list)


@dataclass
class ExtractedPlan:
    """A plan extracted from a trace summary."""

    span_id: str
    plan_type: str  # "initial" | "revised"
    steps: list[PlanStep]
    raw_text: str


# ---------------------------------------------------------------------------
# Pydantic models for LLM structured output
# ---------------------------------------------------------------------------


class LLMPlanStep(BaseModel):
    """A single plan step returned by LLM."""

    step_number: int
    description: str
    depends_on: list[int] = Field(default_factory=list)


class LLMConsolidatedPlan(BaseModel):
    """Consolidated plan returned by LLM."""

    steps: list[LLMPlanStep]


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Match span_id in square brackets at start of a summary line
SPAN_ID_PATTERN = re.compile(r"\[([a-f0-9]{16})\]")

# Broad plan detection: matches lines mentioning plan creation/revision
PLAN_ACTION_VERBS = (
    r"(?:develop|creat|generat|formulat|produc|design|structur|"
    r"revis|updat|reform|modif)"
)
PLAN_KEYWORDS = r"(?:plan|strategy|approach)"
PLAN_LINE_PATTERN = re.compile(
    rf"(?:The model|model)\s+.*?{PLAN_ACTION_VERBS}\w*\s+.*?{PLAN_KEYWORDS}",
    re.IGNORECASE,
)

# Also catch "N-step plan", "plan with N steps", etc.
PLAN_WITH_COUNT_PATTERN = re.compile(
    r"(\d+|two|three|four|five|six|seven|eight|nine|ten)[-\s]step\s+.*?plan"
    r"|plan\s+with\s+(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+steps?",
    re.IGNORECASE,
)

# Extract numbered steps: (1) description, (2) description, ...
NUMBERED_STEP_PATTERN = re.compile(
    r"\((\d+)\)\s*([^(]+?)(?=\(\d+\)|Consumed|The model consumed|$)"
)

# Extract numbered steps with dot: 1. description, 2. description, ...
DOT_NUMBERED_STEP_PATTERN = re.compile(
    r"(?:^|\n)\s*(\d+)\.\s+(.+?)(?=\n\s*\d+\.\s|\n*$)", re.DOTALL
)

# Detect comma-separated plan steps after a colon or "including:"
# Matches patterns like:
#   "6-step plan: verb1 X, verb2 Y, verb3 Z"
#   "plan to solve the task, including: reading X, parsing Y"
#   "plan with 5 steps: step1, step2, step3"
PLAN_ITEMS_PATTERN = re.compile(
    r"(?:plan|strategy|approach)[^.]*?(?::|including:)\s*(.+)",
    re.IGNORECASE,
)

# Also match plans where steps follow "plan to X, Y, Z" without a colon
# e.g., "5-step plan to verify context, retrieve details, consult docs"
PLAN_TO_ITEMS_PATTERN = re.compile(
    r"\d+[-\s]step\s+.*?plan\s+(?:to\s+)?([a-z].*)",
    re.IGNORECASE,
)

# Alternative numbered steps: 1) description, 2) description
ALT_NUMBERED_PATTERN = re.compile(
    r"(?:^|\s)(\d+)\)\s*([^)0-9]+?)(?=\d+\)|Consumed|The model|$)"
)

# Word-to-number mapping for plan step counts
WORD_TO_NUM = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Detect revised/updated plans
REVISION_PATTERN = re.compile(
    r"(?:revis|updat|reform|modif|correct|adjust)\w*\s+.*?"
    r"(?:plan|strategy|approach)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# PlanExtractor
# ---------------------------------------------------------------------------


class PlanExtractor:
    """Extracts and consolidates agent plans from trace summaries.

    Uses broad regex patterns to identify plan-related lines in the hierarchical
    narrative summary. For lines with numbered steps, regex extracts them directly.
    For narrative-style plans without numbering, falls back to LLM parsing.

    After extracting all plans (initial + revisions), uses LLM to consolidate
    them into one finalized plan with proper step dependencies.
    """

    def __init__(self, llm_client: BaseLLMClient | None = None) -> None:
        self._llm_client = llm_client

    def extract_plans(self, trace_id: str, spans: list[dict[str, Any]], summary: str) -> list[ExtractedPlan]:
        """Extract all plans (initial + revised) from a single trace summary.

        Uses span_id references in the summary to look up the original span
        and extract the plan from the LLM output content.

        Args:
            trace_id: Unique trace identifier.
            spans: A list of span dictionaries (flattened).
            summary: Hierarchical narrative summary (markdown).

        Returns:
            List of ExtractedPlan objects in chronological order.
        """
        plans: list[ExtractedPlan] = []

        for line in summary.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line mentions a plan
            if not self._is_plan_line(stripped):
                continue

            # Extract span_id
            span_match = SPAN_ID_PATTERN.search(stripped)
            span_id = span_match.group(1) if span_match else ""

            # Determine plan type
            plan_type = (
                "revised" if REVISION_PATTERN.search(stripped) else "initial"
            )

            # Try to extract plan text from original span output
            plan_text = ""
            if span_id:
                span = self._find_span(spans, span_id)
                if span:
                    plan_text = self._extract_plan_from_span(span)

            # Fallback to summary line text if span lookup failed
            if not plan_text:
                plan_text = self._extract_plan_text(stripped)

            # Try regex extraction of numbered steps
            steps = self._regex_extract_steps(plan_text)

            plans.append(
                ExtractedPlan(
                    span_id=span_id,
                    plan_type=plan_type,
                    steps=steps,
                    raw_text=plan_text,
                )
            )

        logger.info(
            f"[{trace_id}] Extracted {len(plans)} plans "
            f"({sum(1 for p in plans if p.plan_type == 'initial')} initial, "
            f"{sum(1 for p in plans if p.plan_type == 'revised')} revised)"
        )
        return plans

    def consolidate_plans(
        self, task: str, plans: list[ExtractedPlan], trace_id: str = ""
    ) -> list[PlanStep]:
        """Consolidate multiple extracted plans into one finalized plan via LLM.

        Given the initial plan and all revisions in chronological order,
        the LLM produces a single consolidated plan that:
        - Incorporates revisions into the initial plan
        - Resolves conflicts (later revisions take priority)
        - Identifies parallelizable steps and sets depends_on
        - Produces a clean numbered step list with dependencies

        If only one plan exists, still passes through LLM to normalize format
        and extract depends_on.

        Args:
            task: The task description.
            plans: List of ExtractedPlan in chronological order.
            trace_id: For logging.

        Returns:
            List of PlanStep representing the finalized plan.
        """
        if not plans:
            logger.warning(f"[{trace_id}] No plans to consolidate")
            return []

        if self._llm_client is None:
            # No LLM available: fall back to using the last plan's steps
            return self._fallback_consolidate(plans)

        # Build prompt with all plans
        plans_text = self._format_plans_for_llm(plans)
        prompt = self._build_consolidation_prompt(task, plans_text)

        try:
            result = self._llm_client.complete_json(
                messages=[
                    Message(role="system", content=CONSOLIDATION_SYSTEM_PROMPT),
                    Message(role="user", content=prompt),
                ],
                response_model=LLMConsolidatedPlan,
                temperature=0.0,
            )

            if isinstance(result, LLMConsolidatedPlan):
                steps = [
                    PlanStep(
                        step_number=s.step_number,
                        description=s.description,
                        depends_on=s.depends_on,
                    )
                    for s in result.steps
                ]
                logger.info(
                    f"[{trace_id}] Consolidated into {len(steps)} steps via LLM"
                )
                return steps
            else:
                logger.warning(
                    f"[{trace_id}] LLM returned unexpected type: {type(result)}"
                )
                return self._fallback_consolidate(plans)

        except Exception as e:
            logger.error(
                f"[{trace_id}] LLM consolidation failed: {e}, using fallback"
            )
            return self._fallback_consolidate(plans)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_plan_line(self, line: str) -> bool:
        """Check if a summary line mentions plan creation or revision."""
        return bool(
            PLAN_LINE_PATTERN.search(line)
            or PLAN_WITH_COUNT_PATTERN.search(line)
        )

    @staticmethod
    def _find_span(spans: list[dict[str, Any]], span_id: str) -> dict[str, Any] | None:
        """Find a span by span_id in the flat spans list."""
        for s in spans:
            if s.get("span_id") == span_id:
                return s
        return None

    def _extract_plan_from_span(self, span: dict[str, Any]) -> str:
        """Extract plan text from the original span's LLM output.

        Looks for the plan in span_attributes, trying:
        1. llm.output_messages.0.message.content (direct LLM output)
        2. output.value (JSON-encoded output with 'content' field)

        Returns:
            The plan text, or empty string if not found.
        """
        attrs = span.get("span_attributes", {})

        # Strategy 1: Direct LLM output content
        content = attrs.get("llm.output_messages.0.message.content", "")
        if content:
            return content.strip()

        # Strategy 2: output.value as JSON with 'content' key
        output_value = attrs.get("output.value", "")
        if output_value:
            try:
                parsed = json.loads(output_value)
                if isinstance(parsed, dict) and parsed.get("content"):
                    return parsed["content"].strip()
            except (json.JSONDecodeError, TypeError):
                pass

        return ""

    def _extract_plan_text(self, line: str) -> str:
        """Extract the plan description text from a summary line.

        Removes the markdown bullet prefix, span_id, span_name, and duration,
        keeping only the descriptive content.
        """
        # Remove leading bullets and whitespace
        text = re.sub(r"^[\s*]+", "", line)
        # Remove [span_id] **SpanName** (duration): prefix
        text = re.sub(
            r"\[[a-f0-9]{16}\]\s+\*\*[^*]+\*\*\s+\([^)]+\):\s*", "", text
        )
        return text.strip()

    def _regex_extract_steps(self, plan_text: str) -> list[PlanStep]:
        """Extract steps from plan text using multiple regex strategies.

        Strategy order:
        1. Parenthesized numbering: (1) first step, (2) second step, ...
        2. Alternative numbering: 1) first step, 2) second step, ...
        3. Comma-separated list after colon: "plan: verb1 X, verb2 Y, verb3 Z"

        Returns empty list if no steps could be extracted.
        """
        # Strategy 1: (1) ... (2) ... format
        matches = NUMBERED_STEP_PATTERN.findall(plan_text)
        if matches:
            steps = []
            for num_str, desc in matches:
                step_num = int(num_str)
                description = desc.strip().rstrip(",. ")
                if description:
                    steps.append(PlanStep(step_number=step_num, description=description))
            if steps:
                return steps

        # Strategy 2: 1. ... 2. ... format (dot-separated, common in LLM output)
        matches = DOT_NUMBERED_STEP_PATTERN.findall(plan_text)
        if matches:
            steps = []
            for num_str, desc in matches:
                step_num = int(num_str)
                description = desc.strip().rstrip(",. ")
                if description:
                    steps.append(PlanStep(step_number=step_num, description=description))
            if steps:
                return steps

        # Strategy 3: 1) ... 2) ... format
        matches = ALT_NUMBERED_PATTERN.findall(plan_text)
        if matches:
            steps = []
            for num_str, desc in matches:
                step_num = int(num_str)
                description = desc.strip().rstrip(",. ")
                if description:
                    steps.append(PlanStep(step_number=step_num, description=description))
            if steps:
                return steps

        # Strategy 4: Comma-separated list after colon or "including:"
        # e.g., "6-step plan: retrieve X, extract Y, compute Z, ..."
        # or "plan to solve, including: read X, parse Y, extract Z"
        colon_match = PLAN_ITEMS_PATTERN.search(plan_text)
        if not colon_match:
            colon_match = PLAN_TO_ITEMS_PATTERN.search(plan_text)
        if colon_match:
            items_text = colon_match.group(1)
            # Split by commas, but be careful with ", and" patterns
            items_text = re.sub(r",?\s+and\s+", ", ", items_text)
            items = [item.strip().rstrip(". ") for item in items_text.split(",")]
            # Filter out very short items (likely artifacts) and trailing metadata
            items = [
                item for item in items
                if len(item) > 3
                and not item.lower().startswith("consumed")
                and not item.lower().startswith("the model")
            ]
            if len(items) >= 2:
                return [
                    PlanStep(step_number=i + 1, description=item)
                    for i, item in enumerate(items)
                ]

        return []

    def _fallback_consolidate(self, plans: list[ExtractedPlan]) -> list[PlanStep]:
        """Fallback consolidation without LLM: use the first plan with steps.

        Scans plans in chronological order and returns the first one
        that has extracted steps. If none have steps, constructs a single-step
        plan from the first plan's raw text.
        """
        for plan in plans:
            if plan.steps:
                return plan.steps

        # No plan had extracted steps; create single-step plan from first plan's raw text
        first_plan = plans[0]
        return [PlanStep(step_number=1, description=first_plan.raw_text)]

    def _format_plans_for_llm(self, plans: list[ExtractedPlan]) -> str:
        """Format plans for the LLM consolidation prompt."""
        parts = []
        for i, plan in enumerate(plans):
            label = f"Plan {i + 1} ({plan.plan_type})"
            if plan.steps:
                step_lines = "\n".join(
                    f"  ({s.step_number}) {s.description}" for s in plan.steps
                )
                parts.append(f"{label}:\n{step_lines}")
            else:
                parts.append(f"{label}:\n  {plan.raw_text}")
        return "\n\n".join(parts)

    def _build_consolidation_prompt(self, task: str, plans_text: str) -> str:
        """Build the user prompt for plan consolidation."""
        return textwrap.dedent(f"""\
            ## Target Task
            {task}

            ## Agent Plans (Chronological)
            <plans>
            {plans_text}
            </plans>

            ## Instructions
            1. **Synthesize & Resolve**: Merge the plans into one logical sequence. Later revisions take priority in case of conflicts, but retain valid independent steps from earlier plans.
            2. **Relations of Steps**: Build the dependency graph of the steps by identifying which steps depend on others. Do not force a strictly linear sequence if tasks are independent.
            3. **Define Dependencies**: For each step, explicitly list the `step_number`s of any prerequisite steps. Use an empty list `[]` ONLY if the step can be executed immediately at the start.

            ## Expected JSON Output Format
            You must output ONLY valid JSON matching the following schema. Do not include markdown formatting blocks (like ```json), and do not include any conversational text.

            {{
            "steps": [
                {{
                "step_number": 1,
                "description": "<concise action>",
                "depends_on": []
                }}
            ]
            }}

            ## Example
            Task: Find the Wikipedia page for the 2019 game that won the British Academy Games Awards. How many revisions did that page have before the month listed as the game's release date on that Wikipedia page (as of the most recent entry from 2022)?

            Consolidated plan:
            {{
            "steps": [
                {{"step_number": 1, "description": "Determine video game won the British Academy Games Awards of 2019.", "depends_on": []}},
                {{"step_number": 2, "description": "Locate and confirm the official Wikipedia page for that identified game (up to and including the most recent entry from 2022).", "depends_on": [1]}},
                {{"step_number": 3, "description": "Extract the release date from the Wikipedia infobox.", "depends_on": [2]}},
                {{"step_number": 4, "description": "Locate the official Wikipedia history revision page for that game.", "depends_on": [1]}},
                {{"step_number": 5, "description": "Read and filter the revision list to include only those revisions made strictly before the extracted release month.", "depends_on": [3, 4]}},
                {{"step_number": 6, "description": "Count the number of revisions in that filtered list.", "depends_on": [5]}},
                {{"step_number": 7, "description": "Provide this count as the final answer.", "depends_on": [6]}}
            ]
            }}

            Now, generate the consolidated plan for the target task above:"""
        )


# ---------------------------------------------------------------------------
# System prompt for LLM consolidation
# ---------------------------------------------------------------------------

CONSOLIDATION_SYSTEM_PROMPT = (
    """You are a Plan Consolidation Assistant. Your task is to analyze one or more agent execution plans (an initial plan and its subsequent revisions) and generate a single, consolidated ground-truth plan in JSON format.

### PURPOSE
This plan serves as the strict reference for progress monitoring. Each step represents a verifiable milestone. The system will track whether the agent completes each step; if progress stops, that exact step is flagged as the root cause of failure. Therefore, the plan must be accurate, complete, and at the perfect level of granularity.

### CONSOLIDATION RULES
1. **Conflict Resolution:** Later revisions always supersede earlier plans. Preserve the exact intent and methodology of the most recent revision.
2. **Optimal Granularity:** Merge overlapping, redundant, or trivially derivable subtasks into a single logical step.
   - Combine implicit prerequisites with the main action. *Example:* "retrieve the Wikipedia URL" + "visit the Wikipedia page to extract data" → single step: "Visit the Wikipedia page and extract data."
   - **Search-as-means rule:** When a web search is performed solely to locate a target resource (a URL, page, or document), the search query and the identification of the correct result from the search output are a single atomic step. A search that returns no actionable result means the step is incomplete — the search query itself has no standalone value. Never split "search for X" and "select the right result" into separate steps.
3. **Verifiability:** Each step must represent a distinct, non-overlapping unit of work that can be independently verified as pass/fail.
4. **Strict Precision:** Do not generalize descriptions. Retain exact targets, methods, and intents. 
   - *Example:* "Visit Wikipedia revision history page" must NOT be simplified to "Visit Wikipedia page".
5. **Absolute Completeness:** The plan must cover ALL subtasks required to complete the overall task. Do not omit implied but strictly essential milestones.

### DEPENDENCIES & EXECUTION FLOW
- **Parallel Steps:** Identify steps that can be executed concurrently and assign them the same prerequisite dependencies.
- **Sequential Steps:** If a step logically follows another, it MUST include the preceding step's ID in its `depends_on` array.
- **Starting Steps:** Only steps that can begin immediately without any prerequisites should have an empty `depends_on` array `[]`.

### OUTPUT FORMAT
You must return ONLY valid JSON using the following structure, with no markdown formatting or conversational text outside the JSON block:

{
  "steps": [
    {
      "step_number": 1,
      "description": "Precise description of the step.",
      "depends_on": []
    },
    {
      "step_number": 2,
      "description": "Precise description of the next step.",
      "depends_on": [1]
    }
  ]
}"""
)
