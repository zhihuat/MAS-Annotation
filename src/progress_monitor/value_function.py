"""
Progress value function with hierarchical step-by-step analysis.

Evaluates each execution step against a plan DAG to compute a cumulative
progress value V(t) in [0.0, 1.0]. Processes steps in execution order with
context carried forward between steps.

Supports hierarchical drill-down: if a top-level step shows zero progress,
its substeps are analyzed to pinpoint the exact failure location.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
import textwrap

from src.llm.interface import BaseLLMClient, Message
from src.progress_monitor.plan_dag import PlanDAG
from src.utils.trace_utils import smart_truncate
from src.graph.constants import TruncationLimits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalStep:
    """A step in the execution trace with its substeps.

    Parsed from the markdown summary. Preserves parent-child relationships
    based on indentation level.
    """

    span_id: str
    step_name: str
    description: str
    level: int  # depth in hierarchy (0 = top-level under CodeAgent.run)
    children: list["HierarchicalStep"] = field(default_factory=list)
    parent_span_id: str | None = None
    span_kind: str = ""  # "CHAIN", "LLM", "TOOL", etc. from openinference.span.kind
    span_input: str = ""  # raw input from original span attributes
    span_output: str = ""  # raw output from original span attributes


@dataclass
class StepProgress:
    """Progress assessment for a single execution step."""

    span_id: str
    step_name: str
    timestamp_order: int  # chronological position in the execution
    matched_plan_steps: list[int]  # which plan DAG steps this advances
    progress_delta: float  # [0.0, 1.0] - incremental progress from this step
    cumulative_value: float  # running total [0.0, 1.0]
    reasoning: str  # why this score was assigned
    status: str  # "success" | "failed" | "partial" | "hallucination"
    plan_completion_snapshot: dict[int, float] = field(default_factory=dict)
    # snapshot of all plan subtask completion degrees after this step
    # e.g. {1: 1.0, 2: 0.6, 3: 0.0, ...}
    parent_chain_name: str = ""  # name of the enclosing CHAIN/AGENT span
    level: int = 0  # hierarchy level (0 = top-level)
    is_drilldown: bool = False  # True if this came from hierarchical drill-down
    is_hallucination: bool = False  # True if output was fabricated without tool evidence
    history_context: str = ""  # execution history context sent to LLM for scoring


@dataclass
class ProgressContext:
    """Tracks cumulative progress state between step evaluations.

    Carried forward as each step is analyzed, so the LLM has context
    about what work has already been done.

    Dependency backpropagation: when a step is completed, all its
    transitive dependencies (ancestors in the DAG) are also marked
    as completed. E.g., if step 3 (visit URL) depends on step 2
    (get URL) and step 3 is completed, step 2 is automatically
    marked complete — you can't visit a URL without having it.
    """

    completed_subtasks: dict[int, float] = field(default_factory=dict)
    # step_number -> completion degree [0.0, 1.0]
    cumulative_value: float = 0.0
    step_history: list[str] = field(default_factory=list)
    # brief descriptions of what prior steps accomplished
    hallucinated_steps: list[str] = field(default_factory=list)
    # descriptions of steps whose output was hallucinated (no tool evidence)
    dag: Any = field(default=None, repr=False)
    # PlanDAG reference for dependency backpropagation
    preparatory_span_ids: set = field(default_factory=set)
    # Span IDs of initial fact summarization and plan generation steps

    def update(self, step_completions: dict[int, float]) -> float:
        """Update context with newly completed work. Returns the progress delta.

        Args:
            step_completions: Map of plan step number -> completion degree [0.0, 1.0].

        Returns:
            The actual progress delta added (accounting for prior completion).
        """
        # Expand with ancestor dependencies: if step X is completed,
        # all steps that X depends on are implicitly completed too.
        expanded = dict(step_completions)
        if self.dag is not None:
            for step_num, completion in step_completions.items():
                if completion >= 1.0:
                    for ancestor in self.dag.get_ancestors(step_num):
                        if ancestor not in expanded:
                            expanded[ancestor] = 1.0

        delta = 0.0
        for step_num, completion in expanded.items():
            prev = self.completed_subtasks.get(step_num, 0.0)
            new_completion = min(completion, 1.0)
            additional = max(0.0, new_completion - prev)
            if additional > 0:
                self.completed_subtasks[step_num] = min(prev + additional, 1.0)
                delta += additional
        return delta


# ---------------------------------------------------------------------------
# Pydantic response model for LLM scoring
# ---------------------------------------------------------------------------


class StepScoreResponse(BaseModel):
    """LLM response for scoring a single execution step against the plan.

    The LLM analyzes what an execution step accomplished and maps it to
    plan subtasks.
    """

    reasoning: str = Field(
        default="",
        description="Step-by-step explanation of the assessment",
    )
    step_completions: dict[str, float] = Field(
        default_factory=dict,
        description="Per-subtask completion degree. Keys are plan step numbers "
        "(as strings), values are floats 0.0-1.0. Only include steps that this "
        "execution step addresses. E.g., {\"2\": 1.0, \"3\": 0.0}",
    )
    is_hallucination: bool = Field(
        default=False,
        description="True if the output was fabricated without grounding in "
        "actual tool execution",
    )
    step_status: str = Field(
        default="success",
        description="Overall status: success, failed, partial, or hallucination",
    )


# ---------------------------------------------------------------------------
# Step parser
# ---------------------------------------------------------------------------


# Regex for parsing summary lines: indentation, [span_id] **name** (duration): description
SUMMARY_LINE_PATTERN = re.compile(
    r"^(\s*)\*\s+\[([a-f0-9]{16})\]\s+\*\*([^*]+)\*\*\s+\([^)]+\):\s*(.*)"
)


def parse_hierarchical_steps(
    summary: str,
    skip_names: list[str],
    spans: list[dict[str, Any]] | None = None,
) -> list[HierarchicalStep]:
    """Parse the markdown summary into a hierarchy of execution steps.

    Filters out spans in skip_names. Preserves parent-child relationships
    based on markdown indentation level. Enriches each step with span_kind
    (CHAIN, LLM, TOOL) from the original span data when available.

    Args:
        summary: Hierarchical narrative summary (markdown with bullet points).
        skip_names: Span names to exclude (boilerplate like "main", etc.).
        spans: Optional flat list of span dicts for enriching with span_kind.

    Returns:
        List of top-level HierarchicalStep objects (with nested children).
    """
    # Build span_id -> span metadata lookup
    span_meta: dict[str, dict[str, str]] = {}
    if spans:
        for s in spans:
            sid = s.get("span_id", "")
            if not sid:
                continue
            attrs = s.get("span_attributes", {})
            span_meta[sid] = {
                "kind": attrs.get("openinference.span.kind", ""),
                "input": attrs.get("input.value", ""),
                "output": attrs.get("output.value", ""),
            }

    all_steps: list[tuple[int, HierarchicalStep]] = []  # (indent_level, step)

    for line in summary.split("\n"):
        match = SUMMARY_LINE_PATTERN.match(line)
        if not match:
            continue

        indent = len(match.group(1))
        span_id = match.group(2)
        step_name = match.group(3).strip()
        description = match.group(4).strip()

        if step_name in skip_names:
            continue

        meta = span_meta.get(span_id, {})
        step = HierarchicalStep(
            span_id=span_id,
            step_name=step_name,
            description=description,
            level=0,  # will be recomputed below
            span_kind=meta.get("kind", ""),
            span_input=meta.get("input", ""),
            span_output=meta.get("output", ""),
        )
        all_steps.append((indent, step))

    # Build hierarchy from indentation
    if not all_steps:
        return []

    root_steps: list[HierarchicalStep] = []
    stack: list[tuple[int, HierarchicalStep]] = []  # (indent, step)

    for indent, step in all_steps:
        # Pop stack until we find a parent with smaller indent
        while stack and stack[-1][0] >= indent:
            stack.pop()

        if stack:
            parent = stack[-1][1]
            step.parent_span_id = parent.span_id
            step.level = parent.level + 1
            parent.children.append(step)
        else:
            step.level = 0
            root_steps.append(step)

        stack.append((indent, step))

    return root_steps


def flatten_hierarchical_steps(
    steps: list[HierarchicalStep],
) -> list[HierarchicalStep]:
    """Flatten a hierarchy of steps into a single list (depth-first order)."""
    result: list[HierarchicalStep] = []
    for step in steps:
        result.append(step)
        if step.children:
            result.extend(flatten_hierarchical_steps(step.children))
    return result


# ---------------------------------------------------------------------------
# ProgressValueFunction
# ---------------------------------------------------------------------------


class ProgressValueFunction:
    """Evaluates execution steps against a plan DAG using hierarchical analysis.

    Scoring Logic
    ─────────────
    V(t) ∈ [0.0, 1.0] represents cumulative task progress after execution step t.
    Each plan subtask carries equal weight: 1/N where N is the total number of
    plan steps.

    For each execution step (processed in agent execution order):
      1. The LLM receives the current progress context (completed subtasks,
         pending subtasks) and the step's description.
      2. The LLM determines which plan subtask(s) this step addresses and
         the completion degree [0.0, 1.0].
      3. credit = sum(subtask_weight * new_completion) for newly completed work.
         No double credit: if a subtask was already at X% completion, only the
         increment beyond X% is credited.
      4. V(t) = V(t-1) + credit, capped at 1.0.
      5. The context is updated and carried forward to the next step.

    Hierarchical Detection
    ──────────────────────
    Steps are organized in a hierarchy (matching the trace span tree).
      1. First, analyze all top-level steps (direct children of CodeAgent.run).
      2. If a top-level step has zero progress or is flagged as problematic,
         drill into its substeps to identify which specific substep failed.
      3. Continue drilling recursively until the leaf span responsible is found.
    """

    def __init__(
        self,
        plan_dag: PlanDAG,
        llm_client: BaseLLMClient,
        cache_dir: str | Path | None = None,
        force_restart: bool = False,
    ) -> None:
        self._dag = plan_dag
        self._llm = llm_client
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_dir: Path | None = None
        self._force_restart = force_restart

        if cache_dir:
            self._cache_dir = Path(cache_dir)
            os.makedirs(self._cache_dir, exist_ok=True)

    def evaluate_trace(
        self,
        trace_id: str,
        hierarchical_steps: list[HierarchicalStep],
        preparatory_span_ids: set[str] | None = None,
    ) -> list[StepProgress]:
        """Score all execution steps in a trace against the plan DAG.

        Performs depth-first traversal of the full span hierarchy:
        - CHAIN spans: skip scoring, recurse into children (their summaries
          may be misleading and hide hallucinations or missing tool calls).
        - LLM/TOOL spans: score using raw span input/output as ground truth.

        Args:
            trace_id: Unique trace identifier.
            hierarchical_steps: Parsed hierarchy of execution steps.

        Returns:
            List of StepProgress in depth-first execution order.
        """
        if not hierarchical_steps or self._dag.total_steps == 0:
            return []

        self._load_trace_cache(trace_id)

        plan_descriptions = self._dag.get_step_descriptions()
        context = ProgressContext(
            dag=self._dag,
            preparatory_span_ids=preparatory_span_ids or set(),
        )
        results: list[StepProgress] = []
        order_counter = [0]  # mutable counter for recursive DFS

        self._dfs_score(
            trace_id, hierarchical_steps, plan_descriptions,
            context, results, order_counter,
            chain_context="", chain_name="",
        )

        self._save_trace_cache(trace_id)
        return results

    def _dfs_score(
        self,
        trace_id: str,
        steps: list[HierarchicalStep],
        plan_descriptions: dict[int, str],
        context: ProgressContext,
        results: list[StepProgress],
        order_counter: list[int],
        chain_context: str,
        chain_name: str,
    ) -> None:
        """Depth-first traversal scoring of hierarchical steps.

        CHAIN/AGENT spans are transparent wrappers — they are not scored,
        and their children are visited directly. The CHAIN span's description
        is passed down as ``chain_context`` and its name as ``chain_name``.

        Args:
            trace_id: For cache key.
            steps: Steps at the current hierarchy level.
            plan_descriptions: Plan step descriptions for LLM context.
            context: Current progress context (mutated in place).
            results: Accumulator list (mutated in place).
            order_counter: Single-element list used as mutable int counter.
            chain_context: Description from the parent CHAIN span indicating
                the intended purpose of this phase.
            chain_name: Name of the enclosing CHAIN/AGENT span for display.
        """
        for step in steps:
            # CHAIN/AGENT spans: never score directly
            if step.span_kind in ("CHAIN", "AGENT"):
                if step.children:
                    self._dfs_score(
                        trace_id, step.children, plan_descriptions,
                        context, results, order_counter,
                        chain_context=step.description,
                        chain_name=step.step_name,
                    )
                continue

            # Score this span (LLM, TOOL, or other leaf spans)
            score = self._score_step(
                trace_id, step, plan_descriptions, context,
                order_counter[0], is_drilldown=step.level > 0,
                chain_context=chain_context,
                chain_name=chain_name,
            )
            results.append(score)
            order_counter[0] += 1

            # If this span has children, continue DFS into them
            if step.children:
                self._dfs_score(
                    trace_id, step.children, plan_descriptions,
                    context, results, order_counter,
                    chain_context=chain_context,
                    chain_name=chain_name,
                )

    def _score_step(
        self,
        trace_id: str,
        step: HierarchicalStep,
        plan_descriptions: dict[int, str],
        context: ProgressContext,
        order: int,
        is_drilldown: bool = False,
        chain_context: str = "",
        chain_name: str = "",
    ) -> StepProgress:
        """Score a single execution step against the plan DAG.

        Uses LLM to determine which plan subtask(s) this step addresses
        and the completion degree. Updates the context with the result.

        Args:
            trace_id: For caching.
            step: The execution step to score.
            plan_descriptions: Plan step descriptions.
            context: Current progress context (will be mutated).
            order: Chronological position.
            is_drilldown: Whether this is from hierarchical drill-down.
            chain_context: Description from parent CHAIN span indicating
                the intended purpose of this phase.

        Returns:
            StepProgress with the score and metadata.
        """
        # Check cache
        cache_key = step.span_id
        cached = self._get_cached(cache_key)

        if cached is not None:
            score_resp = StepScoreResponse(**cached)
        else:
            score_resp = self._llm_score_step(
                step, plan_descriptions, context, chain_context
            )
            self._set_cached(cache_key, score_resp.model_dump())

        # Mark initial fact/plan steps as preparatory based on structural detection
        if step.span_id in context.preparatory_span_ids:
            score_resp.step_status = "preparatory"
            score_resp.step_completions = {}

        # Convert step_completions keys from str to int
        step_completions: dict[int, float] = {
            int(k): v for k, v in score_resp.step_completions.items()
        }
        matched_plan_steps = sorted(step_completions.keys())

        # If hallucination detected, force zero progress and record it.
        # TOOL spans have real execution evidence — they cannot be hallucinations.
        is_hallucination = score_resp.is_hallucination and step.span_kind != "TOOL"
        if is_hallucination:
            step_completions = {}
            matched_plan_steps = []
            score_resp.step_status = "hallucination"
            context.hallucinated_steps.append(
                f"'{step.step_name}': {score_resp.reasoning[:150]}"
            )

        # Compute actual delta (accounting for prior completion)
        total_steps = self._dag.total_steps
        raw_delta = context.update(step_completions)
        # Scale by per-step weight
        progress_delta = raw_delta * (1.0 / total_steps) if total_steps > 0 else 0.0
        context.cumulative_value = min(context.cumulative_value + progress_delta, 1.0)

        # Snapshot all plan subtask completion degrees after this step
        plan_descriptions = self._dag.get_step_descriptions()
        snapshot = {
            num: context.completed_subtasks.get(num, 0.0)
            for num in sorted(plan_descriptions.keys())
        }

        # Capture history context that was (or would have been) sent to LLM
        if context.step_history:
            recent = "\n".join(f"  - {h}" for h in context.step_history[-5:])
            history_context = f"Recent execution history:\n{recent}"
        else:
            history_context = "No prior steps executed yet."

        # Update history for next step's context — include what this step
        # did and produced, so downstream scoring can judge data dependencies.
        status_tag = f" [{score_resp.step_status.upper()}]"
        if is_hallucination:
            status_tag = " [HALLUCINATED]"
        output_summary = ""
        if step.span_output:
            output_summary = f" → output: {step.span_output[:150]}"
        matched_str = f", plan steps: {matched_plan_steps}" if matched_plan_steps else ""
        brief = (
            f"[{step.span_kind}] '{step.step_name}'{status_tag}{matched_str}"
            f": {score_resp.reasoning[:120]}{output_summary}"
        )
        context.step_history.append(brief)

        return StepProgress(
            span_id=step.span_id,
            step_name=step.step_name,
            timestamp_order=order,
            matched_plan_steps=matched_plan_steps,
            progress_delta=progress_delta,
            cumulative_value=context.cumulative_value,
            reasoning=score_resp.reasoning,
            status=score_resp.step_status,
            plan_completion_snapshot=snapshot,
            parent_chain_name=chain_name,
            level=step.level,
            is_drilldown=is_drilldown,
            is_hallucination=is_hallucination,
            history_context=history_context,
        )

    def _llm_score_step(
        self,
        step: HierarchicalStep,
        plan_descriptions: dict[int, str],
        context: ProgressContext,
        chain_context: str = "",
    ) -> StepScoreResponse:
        """Use LLM to match an execution step to plan subtask(s) and score progress.

        For LLM/TOOL spans, provides the raw span input/output as evidence and
        the parent CHAIN span's description as the intended purpose. The scorer
        must verify that the output genuinely accomplishes the goal — not just
        claims to (hallucination detection).

        Args:
            step: The execution step to score.
            plan_descriptions: All plan steps with descriptions.
            context: Current progress state for LLM context.
            chain_context: Description from parent CHAIN span indicating
                the intended purpose of this phase.

        Returns:
            StepScoreResponse with matched steps and completion degree.
        """
        # Build plan steps summary
        plan_lines = []
        for num in sorted(plan_descriptions.keys()):
            completion = context.completed_subtasks.get(num, 0.0)
            status_str = (
                "COMPLETED" if completion >= 1.0
                else f"{completion:.0%} done" if completion > 0
                else "PENDING"
            )
            plan_lines.append(f"  Step {num} [{status_str}]: {plan_descriptions[num]}")
        plan_summary = "\n".join(plan_lines)

        # Build context summary
        if context.step_history:
            recent_history = "\n".join(
                f"  - {h}" for h in context.step_history[-5:]
            )
            history_section = f"\nRecent execution history:\n{recent_history}"
        else:
            history_section = "\nNo prior steps executed yet."

        # Build hallucination warnings for downstream steps
        hallucination_section = ""
        if context.hallucinated_steps:
            hal_lines = "\n".join(
                f"  - {h}" for h in context.hallucinated_steps
            )
            hallucination_section = (
                f"\n\nWARNING — Prior steps produced hallucinated outputs "
                f"(fabricated without tool evidence):\n{hal_lines}\n"
                f"Any step that consumes these hallucinated outputs "
                f"(e.g., final_answer, FinalOutputTool) should NOT be "
                f"credited as progress."
            )

        # Build intended purpose from chain context
        purpose_section = ""
        if chain_context:
            truncated_ctx = smart_truncate(chain_context, TruncationLimits.MEDIUM)
            purpose_section = f"\n  Intended purpose (from parent phase): {truncated_ctx}"

        # For LLM/TOOL spans, use raw span input/output for ground-truth evidence.
        # For other spans, fall back to the summary description.
        if step.span_kind in ("LLM", "TOOL") and (step.span_input or step.span_output):
            input_section = ""
            if step.span_input:
                truncated_input = smart_truncate(step.span_input, TruncationLimits.VERY_LONG)
                input_section = f"\n  Input: {truncated_input}"
            output_section = ""
            if step.span_output:
                truncated_output = smart_truncate(step.span_output, TruncationLimits.LONG)
                output_section = f"\n  Output: {truncated_output}"

            step_detail = (
                f"  Name: {step.step_name}\n"
                f"  Span kind: {step.span_kind}"
                f"{purpose_section}"
                f"{input_section}"
                f"{output_section}"
            )
        else:
            step_desc = smart_truncate(step.description, TruncationLimits.DEFAULT)
            step_detail = (
                f"  Name: {step.step_name}\n"
                f"  Description: {step_desc}"
                f"{purpose_section}"
            )

        user_content = user_prompt.format(
                        plan_summary=plan_summary,
                        history_section=history_section,
                        hallucination_section=hallucination_section,
                        step_detail=step_detail,
                    )

        try:
            result = self._llm.complete_json(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_content),
                ],
                response_model=StepScoreResponse,
                temperature=0.0,
            )
            if isinstance(result, StepScoreResponse):
                return result
            # If dict returned, validate manually
            return StepScoreResponse(**result)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"LLM scoring failed for step {step.span_id}: {e}")
            return StepScoreResponse(
                step_completions={},
                reasoning=f"LLM scoring failed: {e}",
                step_status="failed",
            )

    # ------------------------------------------------------------------
    # Cache management (follows TraceSummarizer pattern)
    # ------------------------------------------------------------------

    def _load_trace_cache(self, trace_id: str) -> None:
        """Load cached step scores for a single trace."""
        self._cache.clear()
        if self._cache_dir and not self._force_restart:
            cache_path = self._cache_dir / f"{trace_id}.json"
            if cache_path.exists():
                try:
                    cached = json.loads(cache_path.read_text())
                    self._cache.update(cached)
                    logger.info(
                        f"[{trace_id}] Loaded {len(cached)} cached step scores"
                    )
                except (json.JSONDecodeError, IOError):
                    pass

    def _save_trace_cache(self, trace_id: str) -> None:
        """Save step scores cache for a single trace."""
        if self._cache_dir and self._cache:
            cache_path = self._cache_dir / f"{trace_id}.json"
            cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False)
            )

    def _get_cached(self, key: str) -> dict[str, Any] | None:
        """Get a cached score by key."""
        return self._cache.get(key)

    def _set_cached(self, key: str, value: dict[str, Any]) -> None:
        """Set a cached score."""
        self._cache[key] = value



system_prompt = textwrap.dedent("""\
        You are an expert Execution Analysis Engine. Evaluate a single step in an AI agent's execution against its plan.

        ### 1. Progress Rules
        - Progress ONLY counts when a subtask is genuinely completed via successful tool execution.
        - LLM reasoning without tool execution (e.g., internal analysis, query formulation) does NOT count as progress. Only actual tool results count.
        - Navigation tools (e.g., VisitTool) that successfully load a page contribute partial progress (0.2–0.4) toward subtasks that involve that page. Full credit requires actually extracting the needed information (e.g., via FinderTool).
        - If an output fully satisfies a subtask, mark it 1.0. E.g., extracting '20 September 2019' fully satisfies 'find the release month'.
        - Counting / enumeration subtasks: finding some items does NOT complete the subtask if the goal requires ALL items or a total count. Only mark 1.0 when the agent has explicitly confirmed the complete set (e.g., exhaustive search finished, or a definitive total is stated by the source). If completeness is uncertain, cap at 0.5.
        - External failures (HTTP 403/404, anti-scraping blocks, timeouts) mean the step did NOT make progress. Even if the agent attempted the right action, a failed attempt produces no usable result and must be scored 0.0 for that subtask.
        - Web search as means: when a plan step's goal is to locate a specific resource (e.g., "search for and identify the correct Wikipedia page"), merely executing the search query does NOT count as progress. The step is only complete when the agent has identified an actionable result (e.g., a correct URL) from the search output. If the search returns no relevant results or the agent fails to identify the right one (e.g., due to a bad query), score 0.0 — the search attempt alone has no standalone value.
        - Python code execution: if a step invokes Python code, the generated code appears in the current step's output, but its execution result is in the NEXT step's input value under "tool-response". You must check the next step's input to determine whether the code ran successfully and produced valid results before giving credit.

        ### 2. Hallucination & Taint Rules
        - If the agent claims results without a successful TOOL execution, mark is_hallucination=true.
        - Taint propagation: if the step history shows a prior step marked [HALLUCINATED], check whether the CURRENT step's input/output actually uses or depends on data produced by that hallucinated step. If it does, the current step is also a hallucination. If the current step operates on independently obtained data (e.g., fresh tool results unrelated to the hallucinated output), it is NOT tainted — judge it on its own merits.

        ### 3. Output Schema (JSON Only)
        Return ONLY a single valid JSON object. Do not include markdown code blocks (like ```json) or any comments inside the JSON.
        
        {
          "reasoning": "1) intent, 2) evidence, 3) per-subtask assessment",
          "step_completions": {"2": 1.0, "3": 0.0},
          "is_hallucination": false,
          "step_status": "success|failed|partial|hallucination"
        }

        **Field Definitions:**
        - `step_completions`: Per-subtask completion. Keys = plan step numbers (as strings). Values = 0.0 (no progress) to 1.0 (fully complete). Only include subtasks this step is relevant to. Use empty {} if no plan step is addressed.
    """)



user_prompt= textwrap.dedent("""\
    <plan_status>
    {plan_summary}
    </plan_status>

    <execution_history>
    {history_section}
    </execution_history>

    <hallucination_warnings>
    {hallucination_section}
    </hallucination_warnings>

    <step_to_analyze>
    {step_detail}
    </step_to_analyze>

    Based on the context above, analyze the <step_to_analyze>.
    Determine:
    1. What was the intended purpose of this execution step?
    2. Does the actual output genuinely accomplish that purpose, or does it merely claim to without real evidence (hallucination)?
    3. Which plan subtask(s) does this step make real, verified progress on?

    Return ONLY a JSON object matching the schema defined in the system instructions, starting with the 'reasoning' key. Example:
    {{
      "reasoning": "...",
      "step_completions": {{"2": 1.0, "3": 0.0}},
      "is_hallucination": true,
      "step_status": "success"
    }}
""")