"""
Token estimation for failure analysis evaluation (dry-run mode).

Scans traces, generates prompts, counts tokens, and estimates cost
without making any LLM API calls.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.graph.graph_builder import GraphBuilder
from src.utils.trace_utils import build_span_map, extract_task_description, get_trace_id

from src.failure_analysis.config import FailureAnalysisConfig
from src.graph.graph_preprocessor import GraphPreprocessor
from src.schemas.detection import BaseDetectionResponse, AdvancedDetectionResponse, TraceDetectionResponse
from src.failure_analysis import prompts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing per 1M tokens (input_price, output_price) in USD
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "openai/gpt-4.1":            (2.00, 8.00),
    "openai/gpt-4.1-mini":       (0.40, 1.60),
    "openai/gpt-4.1-nano":       (0.10, 0.40),
    "openai/gpt-4o":             (2.50, 10.00),
    "openai/gpt-4o-mini":        (0.15, 0.60),
    "openai/gpt-4-turbo":        (10.00, 30.00),
    # OpenAI-compatible (ARC)
    "openai/gpt-oss-120b":       (0.00, 0.00),
    # Anthropic
    "anthropic/claude-sonnet-4-20250514": (3.00, 15.00),
    "anthropic/claude-haiku-3.5": (0.80, 4.00),
    # Google
    "google/gemini-2.5-pro":     (1.25, 10.00),
    "google/gemini-2.5-flash":   (0.15, 0.60),
    "google/gemini-2.0-flash":   (0.10, 0.40),
    "google/gemini-3-pro-preview": (1.25, 10.00),
}

DEFAULT_PRICING: tuple[float, float] = (2.00, 8.00)

# Completion tokens are typically 30-50% of max_completion_tokens for structured JSON
COMPLETION_RATIO = 0.4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TraceEstimate:
    """Token estimation result for a single trace."""
    trace_id: str
    trace_file: str
    num_spans: int
    num_leaf_nodes: int
    num_llm_calls_min: int
    num_llm_calls_max: int
    prompt_tokens_min: int
    prompt_tokens_max: int
    completion_tokens_min: int
    completion_tokens_max: int


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str, model: str) -> int:
    """Count tokens in text for the given model.

    Uses tiktoken for OpenAI models, character-based estimation for others.
    """
    provider = model.split("/")[0].lower() if "/" in model else ""

    if provider == "openai":
        try:
            import tiktoken
            model_name = model.split("/", 1)[1]
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            logger.debug("tiktoken not installed, using character-based estimation")
            return len(text) // 4

    # Anthropic / Google: ~4 characters per token
    return len(text) // 4


def get_pricing(model: str) -> tuple[float, float]:
    """Get (input_price_per_1M, output_price_per_1M) for a model."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    logger.warning(f"No pricing data for '{model}', using default ${DEFAULT_PRICING[0]}/{DEFAULT_PRICING[1]} per 1M tokens")
    return DEFAULT_PRICING


def _estimate_completion_tokens(max_completion_tokens: int) -> int:
    """Estimate expected completion tokens per call."""
    return int(max_completion_tokens * COMPLETION_RATIO)


def _build_json_suffix(response_model_cls: type) -> str:
    """Build the JSON schema suffix that complete_json() appends to prompts."""
    schema = response_model_cls.model_json_schema()
    return f"\n\nRespond with valid JSON only.\n\nJSON Schema:\n{json.dumps(schema, indent=2)}"


# ---------------------------------------------------------------------------
# Per-trace estimation
# ---------------------------------------------------------------------------

def _build_filtered_graph(trace: dict, config: FailureAnalysisConfig):
    """Build and filter graph from trace (shared pipeline for both algorithms).

    Returns:
        (G_filtered, span_map, preprocessor)
    """
    span_map = build_span_map(trace)

    graph_builder = GraphBuilder()
    G, _ = graph_builder.build_from_trace(
        trace, include_hierarchy=False, include_data_flow=True
    )

    # Remove isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    # Remove excluded nodes
    preprocessor = GraphPreprocessor(config.excluded_node_names)
    G, span_map = preprocessor.remove_excluded_nodes(G, span_map)

    return G, span_map, preprocessor


def estimate_for_trace_base(trace: dict, config: FailureAnalysisConfig) -> TraceEstimate:
    """Estimate tokens for a single trace using the base algorithm.

    Replicates SpanDetector.analyze() steps 1-5, then generates
    all prompts and counts tokens without calling the LLM.
    """
    trace_id = get_trace_id(trace)
    get_span_prompt_fn = prompts.get_span_prompt(config.prompt_strategy)

    G, span_map, preprocessor = _build_filtered_graph(trace, config)
    nodes_and_spans = preprocessor.extract_nodes_and_spans(G, span_map)
    task_desc = extract_task_description(span_map)

    json_suffix = _build_json_suffix(BaseDetectionResponse)
    total_prompt_tokens = 0

    for _, span_data in nodes_and_spans:
        prompt = get_span_prompt_fn(span_data, task_desc, llm_client=None)
        full_prompt = prompt + json_suffix
        total_prompt_tokens += count_tokens(full_prompt, config.llm_model)

    num_calls = len(nodes_and_spans)
    est_completion_per_call = _estimate_completion_tokens(config.max_completion_tokens)

    return TraceEstimate(
        trace_id=trace_id,
        trace_file="",
        num_spans=num_calls,
        num_leaf_nodes=0,
        num_llm_calls_min=num_calls,
        num_llm_calls_max=num_calls,
        prompt_tokens_min=total_prompt_tokens,
        prompt_tokens_max=total_prompt_tokens,
        completion_tokens_min=est_completion_per_call * num_calls,
        completion_tokens_max=est_completion_per_call * num_calls,
    )


def estimate_for_trace_advanced(trace: dict, config: FailureAnalysisConfig) -> TraceEstimate:
    """Estimate tokens for a single trace using the advanced algorithm.

    Since backtracking depth depends on LLM responses, reports min/max ranges:
    - Min: 1 LLM call per leaf node
    - Max: full backtrack to root per leaf node
    """
    trace_id = get_trace_id(trace)
    from src.prompts.advanced import build_backtrack_verify_prompt

    G, span_map, preprocessor = _build_filtered_graph(trace, config)
    leaf_nodes = preprocessor.detect_leaf_nodes(G)

    if not leaf_nodes:
        return TraceEstimate(
            trace_id=trace_id, trace_file="",
            num_spans=G.number_of_nodes(), num_leaf_nodes=0,
            num_llm_calls_min=0, num_llm_calls_max=0,
            prompt_tokens_min=0, prompt_tokens_max=0,
            completion_tokens_min=0, completion_tokens_max=0,
        )

    # Generate prompts for leaf nodes as representative samples
    leaf_token_counts = []
    for leaf_id in leaf_nodes:
        if leaf_id not in span_map:
            continue
        prompt = build_backtrack_verify_prompt(
            span_data=span_map[leaf_id],
            trace_summary="",
            chain_memory=[],
        )
        leaf_token_counts.append(count_tokens(prompt, config.llm_model))

    if not leaf_token_counts:
        return TraceEstimate(
            trace_id=trace_id, trace_file="",
            num_spans=G.number_of_nodes(), num_leaf_nodes=len(leaf_nodes),
            num_llm_calls_min=0, num_llm_calls_max=0,
            prompt_tokens_min=0, prompt_tokens_max=0,
            completion_tokens_min=0, completion_tokens_max=0,
        )

    avg_leaf_tokens = sum(leaf_token_counts) // len(leaf_token_counts)
    # Non-leaf prompts include downstream context (~30% more tokens)
    avg_nonleaf_tokens = int(avg_leaf_tokens * 1.3)

    num_leaves = len(leaf_token_counts)
    max_depth = G.number_of_nodes()  # upper bound: backtrack traverses all nodes

    min_calls = num_leaves
    max_calls = num_leaves * max_depth

    min_prompt_tokens = sum(leaf_token_counts)
    max_prompt_tokens = sum(leaf_token_counts) + num_leaves * (max_depth - 1) * avg_nonleaf_tokens

    est_completion_per_call = _estimate_completion_tokens(config.max_completion_tokens)

    return TraceEstimate(
        trace_id=trace_id,
        trace_file="",
        num_spans=G.number_of_nodes(),
        num_leaf_nodes=num_leaves,
        prompt_tokens_min=min_prompt_tokens,
        prompt_tokens_max=max_prompt_tokens,
        num_llm_calls_min=min_calls,
        num_llm_calls_max=max_calls,
        completion_tokens_min=est_completion_per_call * min_calls,
        completion_tokens_max=est_completion_per_call * max_calls,
    )


def estimate_for_trace_baseline(trace: dict, config: FailureAnalysisConfig) -> TraceEstimate:
    """Estimate tokens for a single trace using the baseline (whole-trace) mode.

    1 LLM call per trace: formats all spans and sends as a single prompt.
    """
    trace_id = get_trace_id(trace)
    get_trace_prompt_fn = prompts.get_trace_prompt(config.prompt_strategy)

    span_map = build_span_map(trace)
    task_desc = extract_task_description(span_map)
    formatted_trace = prompts.format_trace_for_prompt(span_map)

    prompt = get_trace_prompt_fn(formatted_trace, task_desc)
    json_suffix = _build_json_suffix(TraceDetectionResponse)
    full_prompt = prompt + json_suffix

    prompt_tokens = count_tokens(full_prompt, config.llm_model)
    est_completion = _estimate_completion_tokens(config.max_completion_tokens)

    return TraceEstimate(
        trace_id=trace_id,
        trace_file="",
        num_spans=len(span_map),
        num_leaf_nodes=0,
        num_llm_calls_min=1,
        num_llm_calls_max=1,
        prompt_tokens_min=prompt_tokens,
        prompt_tokens_max=prompt_tokens,
        completion_tokens_min=est_completion,
        completion_tokens_max=est_completion,
    )


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def _fmt_range(lo: int, hi: int) -> str:
    """Format a min/max range as a string."""
    if lo == hi:
        return f"{lo:,}"
    return f"{lo:,} - {hi:,}"


def _fmt_cost_range(lo: float, hi: float) -> str:
    if lo == hi:
        return f"${lo:.4f}"
    return f"${lo:.4f} - ${hi:.4f}"


def print_estimation_summary(
    estimates: list[TraceEstimate],
    config: FailureAnalysisConfig,
) -> dict[str, Any]:
    """Print formatted estimation summary and return totals dict."""
    input_price, output_price = get_pricing(config.llm_model)
    is_range = config.algorithm == "advanced"

    # Totals
    total_spans = sum(e.num_spans for e in estimates)
    total_calls_min = sum(e.num_llm_calls_min for e in estimates)
    total_calls_max = sum(e.num_llm_calls_max for e in estimates)
    total_prompt_min = sum(e.prompt_tokens_min for e in estimates)
    total_prompt_max = sum(e.prompt_tokens_max for e in estimates)
    total_completion_min = sum(e.completion_tokens_min for e in estimates)
    total_completion_max = sum(e.completion_tokens_max for e in estimates)

    input_cost_min = total_prompt_min / 1_000_000 * input_price
    input_cost_max = total_prompt_max / 1_000_000 * input_price
    output_cost_min = total_completion_min / 1_000_000 * output_price
    output_cost_max = total_completion_max / 1_000_000 * output_price
    total_cost_min = input_cost_min + output_cost_min
    total_cost_max = input_cost_max + output_cost_max

    sep = "=" * 64
    dash = "-" * 64

    lines = [
        "",
        sep,
        "TOKEN ESTIMATION SUMMARY (DRY RUN)",
        sep,
        f"  Model:           {config.llm_model}",
        f"  Algorithm:       {config.algorithm}",
        f"  Prompt Strategy: {config.prompt_strategy}",
        f"  Max tokens:      {config.max_completion_tokens}",
        f"  Traces:          {len(estimates)}",
        "",
    ]

    # Per-trace table
    if is_range:
        header = f"  {'Trace':<20s}  {'Spans':>5s}  {'Leaves':>6s}  {'Calls':>12s}  {'Prompt Tokens':>20s}"
    else:
        header = f"  {'Trace':<20s}  {'Spans':>5s}  {'Calls':>6s}  {'Prompt Tokens':>14s}"

    lines.append("  Per-Trace Breakdown:")
    lines.append(header)
    lines.append(f"  {dash[:len(header)-2]}")

    for e in estimates:
        trace_label = e.trace_id[:20] if len(e.trace_id) > 20 else e.trace_id
        if is_range:
            lines.append(
                f"  {trace_label:<20s}  {e.num_spans:>5d}  {e.num_leaf_nodes:>6d}"
                f"  {_fmt_range(e.num_llm_calls_min, e.num_llm_calls_max):>12s}"
                f"  {_fmt_range(e.prompt_tokens_min, e.prompt_tokens_max):>20s}"
            )
        else:
            lines.append(
                f"  {trace_label:<20s}  {e.num_spans:>5d}  {e.num_llm_calls_min:>6d}"
                f"  {e.prompt_tokens_min:>14,d}"
            )

    # Totals
    lines.append("")
    lines.append(f"  {'TOTALS':}")
    lines.append(f"  {dash[:50]}")
    lines.append(f"  Total spans:              {total_spans:,}")
    lines.append(f"  Total LLM calls:          {_fmt_range(total_calls_min, total_calls_max)}")
    lines.append(f"  Total prompt tokens:      {_fmt_range(total_prompt_min, total_prompt_max)}")
    lines.append(f"  Est. completion tokens:   {_fmt_range(total_completion_min, total_completion_max)}")

    # Cost
    lines.append("")
    lines.append(f"  ESTIMATED COST (pricing: ${input_price}/1M input, ${output_price}/1M output):")
    lines.append(f"  {dash[:50]}")
    lines.append(f"  Input cost:   {_fmt_cost_range(input_cost_min, input_cost_max)}")
    lines.append(f"  Output cost:  {_fmt_cost_range(output_cost_min, output_cost_max)}")
    lines.append(f"  Total cost:   {_fmt_cost_range(total_cost_min, total_cost_max)}")

    if config.enable_summarize:
        lines.append("")
        lines.append("  Note: Summarization is enabled. Actual prompts may be shorter")
        lines.append("  (this estimate is conservative, without LLM-based summarization).")

    if is_range:
        lines.append("")
        lines.append("  Note: Advanced algorithm shows min-max ranges because backtracking")
        lines.append("  depth depends on LLM responses at runtime.")

    lines.append(sep)

    print("\n".join(lines))

    return {
        "model": config.llm_model,
        "algorithm": config.algorithm,
        "traces": len(estimates),
        "total_spans": total_spans,
        "total_llm_calls_min": total_calls_min,
        "total_llm_calls_max": total_calls_max,
        "total_prompt_tokens_min": total_prompt_min,
        "total_prompt_tokens_max": total_prompt_max,
        "total_completion_tokens_min": total_completion_min,
        "total_completion_tokens_max": total_completion_max,
        "estimated_cost_min": round(total_cost_min, 4),
        "estimated_cost_max": round(total_cost_max, 4),
        "per_trace": [
            {
                "trace_id": e.trace_id,
                "trace_file": e.trace_file,
                "num_spans": e.num_spans,
                "num_llm_calls_min": e.num_llm_calls_min,
                "num_llm_calls_max": e.num_llm_calls_max,
                "prompt_tokens_min": e.prompt_tokens_min,
                "prompt_tokens_max": e.prompt_tokens_max,
            }
            for e in estimates
        ],
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def run_estimation(config: FailureAnalysisConfig, limit: int | None = None) -> dict[str, Any]:
    """Run token estimation across all traces without making LLM calls.

    Args:
        config: Configuration (algorithm, model, paths, etc.)
        limit: Optional limit on number of traces

    Returns:
        Dictionary with estimation results
    """
    trace_dir = Path(config.trace_dir)
    if not trace_dir.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    trace_files = sorted(trace_dir.glob("*.json"))
    if limit:
        trace_files = trace_files[:limit]

    logger.info(f"Estimating token usage for {len(trace_files)} traces "
                f"(algorithm={config.algorithm}, strategy={config.prompt_strategy}, "
                f"model={config.llm_model})")

    estimates: list[TraceEstimate] = []

    for idx, trace_file in enumerate(trace_files, 1):
        logger.debug(f"[{idx}/{len(trace_files)}] Estimating {trace_file.name}")

        try:
            with open(trace_file, 'r') as f:
                trace = json.load(f)

            if config.algorithm == "trace":
                est = estimate_for_trace_baseline(trace, config)
            elif config.algorithm == "base":
                est = estimate_for_trace_base(trace, config)
            else:
                est = estimate_for_trace_advanced(trace, config)

            est.trace_file = trace_file.name
            estimates.append(est)

        except Exception as e:
            logger.warning(f"Failed to estimate for {trace_file.name}: {e}")
            continue

    return print_estimation_summary(estimates, config)
