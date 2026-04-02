"""
End-to-end pipeline for progress monitoring.

Orchestrates plan extraction, DAG construction, value function computation,
flatline detection, and visualization across all traces.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.utils.trace_utils import flatten_spans, extract_task_description
from src.llm import create_llm_client
from src.progress_monitor.config import ProgressMonitorConfig
from src.progress_monitor.flatline_detector import FlatlineDetector, FlatlineResult
from src.progress_monitor.plan_dag import PlanDAG
from src.progress_monitor.plan_extractor import PlanExtractor
from src.progress_monitor.value_function import (
    HierarchicalStep,
    ProgressValueFunction,
    StepProgress,
    parse_hierarchical_steps,
)
from src.progress_monitor.visualizer import ProgressVisualizer

logger = logging.getLogger(__name__)


class ProgressMonitorPipeline:
    """End-to-end pipeline: extract plans, build DAGs, compute progress, detect flatlines.

    Processes traces that have both a trace file in trace_dir and an entry
    in the trace summaries file.
    """

    def __init__(self, config: ProgressMonitorConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM client
        logger.info(f"Initializing LLM client: {config.llm_model}")
        self.llm_client = create_llm_client(
            model=config.llm_model,
            temperature=config.temperature,
            max_completion_tokens=config.max_completion_tokens,
            reasoning_effort=config.reasoning_effort,
        )

        # Initialize components
        self.plan_extractor = PlanExtractor(self.llm_client)
        self.flatline_detector = FlatlineDetector(config)

    def run(self, limit: int | None = None) -> dict[str, Any]:
        """Process all traces and return aggregated results.

        Args:
            limit: Maximum number of traces to process (None = all).

        Returns:
            Dict with per-trace results and aggregate statistics.
        """
        limit = limit or self.config.limit

        # 1. Load data
        summaries = self._load_summaries()
        annotations = self._load_annotations()
        trace_ids = self._get_valid_trace_ids(summaries)

        if limit:
            trace_ids = trace_ids[:limit]

        logger.info(
            f"Processing {len(trace_ids)} traces "
            f"({len(annotations)} with annotations)"
        )

        # 2. Process each trace (load spans per trace)
        results: list[dict[str, Any]] = []

        if self.config.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_single_trace_safe,
                        tid,
                        self._load_spans_for_trace(tid),
                        summaries[tid],
                        annotations.get(tid),
                    ): tid
                    for tid in trace_ids
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures), desc="Processing traces"
                ):
                    tid = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        logger.error(f"[{tid}] Failed: {e}")
        else:
            for tid in tqdm(trace_ids, desc="Processing traces"):
                result = self._process_single_trace_safe(
                    tid,
                    self._load_spans_for_trace(tid),
                    summaries[tid],
                    annotations.get(tid),
                )
                if result:
                    results.append(result)

        # 3. Aggregate and save
        aggregate = self._compute_aggregate(results)
        self._save_results(results, aggregate)

        # 4. Generate batch summary plot
        plot_data = self._prepare_plot_data(results, annotations)
        ProgressVisualizer.plot_batch_summary(
            plot_data,
            output_path=self.output_dir / "batch_summary.png",
        )

        logger.info(
            f"Pipeline complete. {len(results)} traces processed. "
            f"Results saved to {self.output_dir}"
        )
        return {"results": results, "aggregate": aggregate}

    def process_single_trace(
        self,
        trace_id: str,
        spans: list[dict[str, Any]],
        summary: str,
        annotation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Full pipeline for one trace.

        Args:
            trace_id: Unique trace identifier.
            spans: Flattened list of span dicts (already processed by flatten_spans).
            summary: Hierarchical narrative summary.
            annotation: Optional root cause annotation for evaluation.

        Returns:
            Dict with plans, DAG, progress, flatline, and optional metrics.
        """

        span_map = {s.get("span_id", ""): s for s in spans if s.get("span_id")}
        task = extract_task_description(span_map)

        # 1. Extract all plans from summary
        plans = self.plan_extractor.extract_plans(trace_id, spans, summary)

        # 2. Identify preparatory span IDs (initial fact + plan steps)
        preparatory_span_ids = self._find_preparatory_spans(plans, summary)

        # 3. Consolidate plans via LLM → finalized plan
        merged_steps = self.plan_extractor.consolidate_plans(task, plans, trace_id)

        # 4. Build & validate PlanDAG
        dag = PlanDAG(merged_steps)
        validation_issues = dag.validate()
        if validation_issues:
            logger.warning(f"[{trace_id}] DAG validation issues: {validation_issues}")

        # 5. Parse hierarchical steps (filter skip_span_names, enrich with span_kind)
        hierarchical_steps = parse_hierarchical_steps(
            summary, self.config.skip_span_names, spans
        )

        # 6. Evaluate value function (top-level first, then drill-down)
        vf = ProgressValueFunction(
            dag, self.llm_client,
            cache_dir=self.config.cache_dir,
            force_restart=self.config.force_restart,
        )
        progress = vf.evaluate_trace(
            trace_id, hierarchical_steps, preparatory_span_ids
        )

        # 6. Detect flatline → identify root cause span
        flatline = self.flatline_detector.detect(progress, trace_id)

        # 7. Generate single-trace plot
        plot_path = self.output_dir / "plots" / f"{trace_id}.png"
        ProgressVisualizer.plot_single_trace(
            progress,
            flatline=flatline,
            title=f"Trace {trace_id[:12]}... (V={flatline.final_value:.2f})",
            output_path=plot_path,
        )

        # 8. Compare with annotation if available
        comparison = None
        if annotation:
            comparison = self._compare_with_annotation(
                flatline, annotation, [asdict(s) for s in progress]
            )

        result = {
            "trace_id": trace_id,
            "plans_extracted": len(plans),
            "plan_types": [p.plan_type for p in plans],
            "finalized_plan": [asdict(s) for s in merged_steps],
            "dag_validation": validation_issues,
            "dag": dag.to_dict(),
            "progress": [asdict(s) for s in progress],
            "flatline": flatline.to_dict(),
            "comparison": comparison,
        }

        return result

    @staticmethod
    def _find_preparatory_spans(
        plans: list, summary: str
    ) -> set[str]:
        """Identify ALL fact summarization and plan generation span IDs.

        Plan span IDs come directly from extract_plans. For each plan span,
        the preceding sibling span at the same indentation level in the
        summary is assumed to be the fact summarization step.

        All occurrences of fact+plan are treated as preparatory, so that
        the flatline root cause always points to an actual execution step.
        """
        import re

        if not plans:
            return set()

        # Collect all plan span IDs
        plan_span_ids = {p.span_id for p in plans if p.span_id}
        if not plan_span_ids:
            return set()

        preparatory: set[str] = set(plan_span_ids)

        # Parse summary lines to find preceding siblings of each plan span
        span_pattern = re.compile(r"^(\s*)\*\s+\[([a-f0-9]{16})\]")
        lines_with_spans: list[tuple[int, str]] = []  # (indent, span_id)

        for line in summary.split("\n"):
            m = span_pattern.match(line)
            if m:
                indent = len(m.group(1))
                span_id = m.group(2)
                lines_with_spans.append((indent, span_id))

        # For each plan span, the preceding sibling at the same indent is the fact span
        for i, (indent, span_id) in enumerate(lines_with_spans):
            if span_id in plan_span_ids and i > 0:
                prev_indent, prev_span_id = lines_with_spans[i - 1]
                if prev_indent == indent:
                    preparatory.add(prev_span_id)

        return preparatory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_single_trace_safe(
        self,
        trace_id: str,
        spans: list[dict[str, Any]],
        summary: str,
        annotation: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Process a single trace with error handling."""
        try:
            return self.process_single_trace(trace_id, spans, summary, annotation)
        except Exception as e:
            logger.error(f"[{trace_id}] Error processing trace: {e}", exc_info=True)
            return None
        
    def _load_spans_for_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """Load and flatten spans for a single trace."""
        trace_path = Path(self.config.trace_dir) / f"{trace_id}.json"
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")

        with open(trace_path, "r") as f:
            trace = json.load(f)

        return flatten_spans(trace)

    def _load_summaries(self) -> dict[str, str]:
        """Load trace summaries from the configured file."""
        summary_path = Path(self.config.summary_file)
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")

        with open(summary_path, "r") as f:
            summaries = json.load(f)

        logger.info(f"Loaded {len(summaries)} trace summaries from {summary_path}")
        return summaries

    def _load_annotations(self) -> dict[str, dict[str, Any]]:
        """Load root cause annotations from the configured directory."""
        anno_dir = Path(self.config.annotation_dir)
        annotations: dict[str, dict[str, Any]] = {}

        if not anno_dir.exists():
            logger.warning(f"Annotation directory not found: {anno_dir}")
            return annotations

        for file_path in anno_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                trace_id = data.get("trace_id", file_path.stem)
                annotations[trace_id] = data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load annotation {file_path}: {e}")

        logger.info(f"Loaded {len(annotations)} annotations from {anno_dir}")
        return annotations

    def _get_valid_trace_ids(self, summaries: dict[str, str]) -> list[str]:
        """Get trace IDs that exist in both summaries and trace files."""
        trace_dir = Path(self.config.trace_dir)
        valid_ids: list[str] = []

        for trace_id in sorted(summaries.keys()):
            trace_file = trace_dir / f"{trace_id}.json"
            if trace_file.exists():
                valid_ids.append(trace_id)
            else:
                logger.debug(f"Trace file missing for summary {trace_id}, skipping")

        logger.info(
            f"{len(valid_ids)} traces have both summary and trace file "
            f"(out of {len(summaries)} summaries)"
        )
        return valid_ids

    def _compare_with_annotation(
        self,
        flatline: FlatlineResult,
        annotation: dict[str, Any],
        progress: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare flatline detection result with ground truth annotation.

        Uses root_cause_span_id from annotation. Computes span distance
        between the detected flatline span and the ground truth span in
        the progress step list.
        """
        gt_span_id = annotation.get("root_cause_span_id", "") or ""
        gt_step = annotation.get("root_cause_step")
        detected_span = flatline.flatline_start_span_id or ""

        # If annotation has no root cause (empty step + empty span_id),
        # the agent succeeded. A correct detection should have no flatline.
        agent_succeeded = not gt_span_id and not gt_step

        if agent_succeeded:
            # Success case: correct if no flatline detected
            exact_match = not flatline.has_flatline
            return {
                "gt_root_cause_span_id": "",
                "detected_span": detected_span,
                "agent_succeeded": True,
                "exact_match": exact_match,
                "span_distance": None,
                "false_positive": flatline.has_flatline,
            }

        # Check exact match
        exact_match = bool(gt_span_id and detected_span == gt_span_id)

        # Compute span distance: how many steps apart in the progress list
        span_distance: int | None = None
        if gt_span_id and detected_span and progress:
            gt_idx: int | None = None
            detected_idx: int | None = None
            for i, step in enumerate(progress):
                sid = step.get("span_id") if isinstance(step, dict) else step.span_id
                if sid == gt_span_id:
                    gt_idx = i
                if sid == detected_span:
                    detected_idx = i
            if gt_idx is not None and detected_idx is not None:
                span_distance = detected_idx - gt_idx

        return {
            "gt_root_cause_span_id": gt_span_id,
            "detected_span": detected_span,
            "agent_succeeded": False,
            "exact_match": exact_match,
            "span_distance": span_distance,
            "false_positive": False,
        }

    def _compute_aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate statistics across all processed traces."""
        if not results:
            return {}

        total = len(results)
        flatlines = [r for r in results if r["flatline"]["has_flatline"]]
        comparisons = [r for r in results if r.get("comparison")]

        exact_matches = sum(
            1 for r in comparisons if r["comparison"]["exact_match"]
        )

        # Span distances (only for traces where both gt and detected exist)
        distances = [
            r["comparison"]["span_distance"]
            for r in comparisons
            if r["comparison"]["span_distance"] is not None
        ]
        avg_span_distance = (
            round(sum(abs(d) for d in distances) / len(distances), 2)
            if distances else None
        )

        final_values = [r["flatline"]["final_value"] for r in results]
        avg_final_value = sum(final_values) / total if total else 0.0

        # False positives: agent succeeded but flatline was detected
        false_positives = sum(
            1 for r in comparisons if r["comparison"].get("false_positive")
        )

        return {
            "total_traces": total,
            "flatlines_detected": len(flatlines),
            "avg_final_value": round(avg_final_value, 4),
            "traces_with_annotations": len(comparisons),
            "exact_span_matches": exact_matches,
            "exact_match_rate": (
                round(exact_matches / len(comparisons), 4) if comparisons else None
            ),
            "avg_span_distance": avg_span_distance,
            "false_positives": false_positives,
        }

    def _save_results(
        self, results: list[dict[str, Any]], aggregate: dict[str, Any]
    ) -> None:
        """Save per-trace results and aggregate stats to output directory."""
        # Save aggregate
        agg_path = self.output_dir / "aggregate_metrics.json"
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2, ensure_ascii=False)

        # Save per-trace results
        detections_dir = self.output_dir / "detections"
        detections_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            trace_id = result["trace_id"]
            result_path = detections_dir / f"{trace_id}.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved {len(results)} results to {detections_dir} "
            f"and aggregate to {agg_path}"
        )

    def _prepare_plot_data(
        self,
        results: list[dict[str, Any]],
        annotations: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prepare data for batch summary plot."""
        plot_data: list[dict[str, Any]] = []
        for r in results:
            fl = r["flatline"]
            trace_outcome = annotations.get(r["trace_id"], {}).get(
                "trace_outcome", "Unknown"
            )
            plot_data.append(
                {
                    "trace_id": r["trace_id"],
                    "final_value": fl["final_value"],
                    "has_flatline": fl["has_flatline"],
                    "flatline_start_index": fl.get("flatline_start_index"),
                    "total_steps": fl["total_steps"],
                    "trace_outcome": trace_outcome,
                }
            )
        return plot_data
