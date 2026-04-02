"""
TraceSummarizer — generates a global Hierarchical Narrative Summary from
a flat list of trace spans.

Shared utility used by both the APO pipeline and the failure analysis detectors.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from collections import defaultdict

from src.llm import call_llm
from src.utils.trace_utils import truncate_spans


def _build_span_skeleton(flat_spans: list[dict[str, Any]]) -> str:
    """Build an indented span_id + span_name checklist mirroring the parent/child hierarchy."""
    children: dict[str, list[str]] = defaultdict(list)
    span_info: dict[str, str] = {}
    roots: list[str] = []

    for span in flat_spans:
        sid = span.get("span_id", "")
        name = span.get("span_name", "unknown")
        parent = span.get("parent_span_id", "")
        span_info[sid] = name
        if parent and parent in {s.get("span_id") for s in flat_spans}:
            children[parent].append(sid)
        else:
            roots.append(sid)

    lines: list[str] = []

    def _walk(sid: str, depth: int) -> None:
        indent = "    " * depth
        lines.append(f"{indent}* [{sid}] **{span_info.get(sid, 'unknown')}**")
        for child in children.get(sid, []):
            _walk(child, depth + 1)

    for root in roots:
        _walk(root, 0)

    return "\n".join(lines)

# Summarizer prompt templates (inlined to avoid circular imports)
_SUMMARIZER_SYSTEM = (
    "You are a Technical Trace Analyst. You specialize in converting "
    "OpenInference/OpenTelemetry JSON traces into clear, objective narrative "
    "logs for developers. Your output (summary) is used for automated Root "
    "Cause Analysis, so you must prioritize structural accuracy, precise error "
    "reporting, and absolute completeness."
)

_SUMMARIZER_USER = """\
# Objective
Analyze the provided distributed trace and generate a **Hierarchical Narrative Summary**.
Your goal is to describe **what actually happened** in the execution flow.
- Prioritize factual description over speculation.
- Do not critique the logging configuration (e.g., do not complain about empty attributes) unless it causes a visible crash.

# Formatting Rules
1. **Tree Structure:** Use markdown bullet points with indentation to strictly mirror the `span_id` / `parent_span_id` hierarchy.
2. **Header:** `* [span_id] **SpanName** (Duration):` — copy the `span_id` field value **verbatim** from the input JSON.
3. **Body:** A concise summary of the operation.

# Core Instructions

### 1. Length & Density (Crucial)
- **Do not use a fixed word count.** Use proportional detail:
    - **Wrapper/Orchestrator Spans (e.g., main, run):** 1 sentence. State what started.
    - **Simple Tools/Helpers:** 1-2 sentences. Input -> Output.
    - **Complex Logic (LLM Calls/Code Execution):** 3-5 sentences. Summarize the reasoning, the code generated, or the specific plan created.

### 2. Handling Data Fields
- **Attributes:** Look primarily at `input.value`, `output.value`, and `tool_calls`.
- **JSON Strings:** If an input/output is a JSON string, parse it internally and summarize the *content* (e.g., "The model received a prompt about family reunions...").
- **Truncation:** If the log says "...", note that the log is truncated, but assume the execution continued unless a status error exists.

### 3. Error & Anomaly Reporting
- **Explicit Errors:** Only bold **FAILED** if `status_code` is "Error" or if there is an exception stack trace.
- **Logical Deviations:** If an Agent creates a plan (e.g., 7 steps) but executes it differently (e.g., solves it all in Step 2), describe this factually: *"The Agent skipped the remaining planned steps and solved the problem immediately in this step."* Do not call it a "system failure" unless the code actually crashed.

### 4. Handling "CHAIN" Spans
- **Definition:** Spans with `openinference.span.kind: CHAIN` are logical wrappers. They define the **intent** of a step but perform no actual computation.
- **Instruction:** Do not attribute actions to the CHAIN span itself. Instead, state the CHAIN span's goal (e.g., "Step 2 aimed to count adults"), and then describe the specific **Child Span** (Tool or LLM) that performed the execution.

# Input Data
<trace_json>
{trace_json}
</trace_json>

# Output Format Example
* [a1b2c3d4e5f60718] **main** (445ms): Initiated the trace execution.
    * [b2c3d4e5f6071829] **answer_single_question** (200ms): Orchestrated the agent execution.
        * [c3d4e5f607182930] **Step 1** (150ms): Intended to parse the problem.
            * [d4e5f60718293041] **LiteLLMModel** (100ms): The model analyzed the family tree, correctly identified 11 adults and 3 kids, and calculated 18 potatoes needed.

# IMPORTANT:
- **You should summarize each span in order, don't skip any span.**
- **MOST IMPORTANT**: **IF THERE IS A CLEAR ERROR, that makes agent crashes/fails. DESCRIBE IT CLEARLY WITHIN THAT SPAN SUMMARY. (Example: in this step the agent is supposed to take this and do this but it failed because the inputs passed to it are not as expected, it expects ... but got ... This can be ...)**

# Span Checklist
You MUST produce exactly one summary bullet for each span listed below, in this order and hierarchy. Do NOT skip or merge any entry.
{span_skeleton}

Now begin summarising:
"""

# Model context window limits (input tokens).
_MODEL_INPUT_LIMITS: dict[str, int] = {
    "gpt-5-mini": 240_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-4-turbo": 128_000,
    "gpt-oss-120b": 75_000,
    "claude-sonnet-4-20250514": 160_000,
    "claude-haiku-4-5-20251001": 160_000,
    "claude-haiku-3.5": 160_000,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-3-pro-preview": 1_048_576,
    "Kimi-K2.5": 210_000,
}
_DEFAULT_INPUT_LIMIT = 128_000


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


class TraceSummarizer:
    """
    Generates a narrative summary of an entire trace.

    Given the flat list of spans that make up a trace, produces a hierarchical
    markdown summary describing execution flow, errors, and anomalies.

    Manages a trace_id → summary cache with disk persistence so that summaries
    are never regenerated unnecessarily.

    Used by both the APO pipeline and failure analysis detectors.
    """

    def __init__(self, client, cache_path: Optional[str | Path] = None, max_completion_tokens: int = 8000):
        self.client = client
        self.max_completion_tokens = max_completion_tokens
        self._cache: dict[str, str] = {}
        self._cache_path: Optional[Path] = None

        if cache_path:
            cache_dir = Path(cache_path)
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_path = cache_dir / "trace_summaries.json"
            self.load_cache()

    def load_cache(self) -> None:
        if self._cache_path and self._cache_path.exists():
            try:
                cached = json.loads(self._cache_path.read_text())
                self._cache.update(cached)
                print(f"  [Summarizer] Loaded {len(cached)} cached summaries from {self._cache_path}")
            except (json.JSONDecodeError, IOError):
                pass

    def save_cache(self) -> None:
        if self._cache_path:
            self._cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False)
            )

    def get_summary(self, trace_id: str) -> str:
        return self._cache.get(trace_id, "")

    def generate_summaries(self, traces: list[dict], truncation_limit: int = 2000, max_workers: int = 10) -> None:
        """
        Batch-generate summaries for traces not already cached.

        Each trace dict must have ``trace_id`` and ``all_spans`` keys.
        """
        to_summarize: dict[str, list[dict]] = {}
        for t in traces:
            tid = t["trace_id"]
            if tid not in self._cache:
                to_summarize[tid] = t["all_spans"]

        if not to_summarize:
            print(f"  [Summarizer] All {len(traces)} trace(s) already cached, skipping.")
            return

        print(f"\n  [Summarizer] Generating summaries for {len(to_summarize)} trace(s) "
              f"({len(traces) - len(to_summarize)} cached)...")

        def _summarize_one(trace_id: str, spans: list[dict]) -> tuple[str, str]:
            summary = self.summarize(spans, truncation_limit=truncation_limit)
            return trace_id, summary

        with ThreadPoolExecutor(max_workers=min(len(to_summarize), max_workers)) as pool:
            futures = {
                pool.submit(_summarize_one, tid, spans): tid
                for tid, spans in to_summarize.items()
            }
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="  Summarizing", leave=False):
                tid, summary = fut.result()
                self._cache[tid] = summary
                print(f"  [Summarizer] Trace {tid}: {len(summary)} chars")

        self.save_cache()

    def summarize(self, flat_spans: list[dict[str, Any]], truncation_limit: int = 2000) -> str:
        """Generate a Hierarchical Narrative Summary from a flat list of spans."""
        if not flat_spans:
            return ""

        model_name = getattr(self.client, "model", "")
        input_limit = _MODEL_INPUT_LIMITS.get(model_name, _DEFAULT_INPUT_LIMIT)
        input_budget = input_limit - self.max_completion_tokens

        span_skeleton = _build_span_skeleton(flat_spans)

        while truncation_limit > 0:
            preprocessed = truncate_spans(flat_spans, truncation_limit)
            trace_json_str = json.dumps(preprocessed, indent=2)
            user_prompt = _SUMMARIZER_USER.format(
                trace_json=trace_json_str,
                span_skeleton=span_skeleton,
            )
            full_prompt = _SUMMARIZER_SYSTEM + user_prompt

            estimated_tokens = _estimate_tokens(full_prompt)
            if estimated_tokens <= input_budget:
                break

            new_limit = int(truncation_limit * 1 / 2)
            print(f"  [Summarizer] Estimated {estimated_tokens} tokens exceeds "
                  f"input limit {input_budget}, reducing truncation "
                  f"{truncation_limit} -> {new_limit}")
            truncation_limit = new_limit

        response = call_llm(
            self.client,
            [{"role": "user", "content": user_prompt}],
            max_completion_tokens=self.max_completion_tokens,
            system=_SUMMARIZER_SYSTEM,
        )

        return response
