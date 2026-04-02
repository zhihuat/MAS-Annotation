"""Data loading and processing for trace analysis.

Provides two main capabilities:
1. SpanDataProcessor: Loads trace + annotation files into span-level samples
   for the training/evaluation loop.
2. Build utilities: Converts raw GAIA traces into the sample format expected
   by SpanDataProcessor, applying the same graph filtering pipeline as the
   failure analysis detectors.

Usage (CLI):
    python -m src.utils.data_processor \\
        --train path/to/train_traces/ \\
        --test  path/to/test_traces/
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any

from src.metrics.location import prediction_is_correct, compute_span_accuracy
from src.graph.graph_preprocessor import GraphPreprocessor

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Defaults mirror FailureAnalysisConfig
DEFAULT_TRACE_DIR = ROOT_DIR / "data" / "GAIA"
DEFAULT_ANNOTATION_DIR = ROOT_DIR / "data" / "GAIA_ANNOTATIONS"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "apo_samples"
DEFAULT_EXCLUDED_NODES = [
    "main",
    "get_examples_to_answer",
    "answer_single_question",
    "create_agent_hierarchy",
]


# ---------------------------------------------------------------------------
# Trace filtering pipeline (mirrors span_detector.py)
# ---------------------------------------------------------------------------

def extract_filtered_spans(
    trace: dict[str, Any],
    excluded_node_names: list[str],
) -> tuple[dict[str, Any], str]:
    """
    Run the graph filtering pipeline and return only the surviving spans.

    Returns:
        filtered_spans: {span_id: span_data} for spans that survive data-flow filtering
        task_description: extracted from span_map
    """
    preprocessor = GraphPreprocessor(excluded_node_names)
    _, _, nodes_and_spans, task_desc = preprocessor.build_and_filter(trace)
    filtered_spans = {node_id: span_data for node_id, span_data in nodes_and_spans}
    return filtered_spans, task_desc


# ---------------------------------------------------------------------------
# Ground-truth builders
# ---------------------------------------------------------------------------

def build_trace_json(
    trace_id: str,
    filtered_spans: dict[str, Any],
    task_description: str,
) -> dict[str, Any]:
    """Build trace JSON in the format expected by SpanDataProcessor."""
    return {
        "trace_id": trace_id,
        "task_description": task_description,
        "spans": filtered_spans,
    }


def build_gt_json(
    annotation: dict[str, Any],
    filtered_span_ids: set[str],
    task_description: str,
) -> dict[str, Any]:
    """
    Build ground-truth JSON from a causal-chains annotation file.

    Extracts root_cause and symptoms from each causal_chain entry.
    benign_errors are excluded. Only errors whose span_id survives the
    data-flow filter are kept. All error fields are preserved.
    """
    span_errors = []
    for chain in annotation.get("causal_chains", []):
        candidates: list[tuple[str, dict]] = []
        if "root_cause" in chain:
            candidates.append(("root_cause", chain["root_cause"]))
        for sym in chain.get("symptoms", []):
            candidates.append(("symptom", sym))
        for role, err in candidates:
            span_id = err.get("span_id", "")
            if span_id and span_id in filtered_span_ids:
                entry = dict(err)
                entry["role"] = role
                span_errors.append(entry)

    return {
        "task_description": task_description,
        "span_errors": span_errors,
    }


# ---------------------------------------------------------------------------
# SpanDataProcessor
# ---------------------------------------------------------------------------

class SpanDataProcessor:
    """
    Loads trace + annotation files into span-level samples for training/evaluation.

    Trace format (trace_NNN.json):
    {
        "trace_id": "trace_001",
        "task_description": "...",
        "spans": {
            "span_id": {
                "span_name": "...",
                "status_code": "OK",
                "span_attributes": { ... }
            }, ...
        }
    }

    Annotation format (trace_NNN_gt.json):
    {
        "task_description": "...",
        "span_errors": [
            {
                "span_id": "span_001",
                "category": "Language-only",
                "description": "...",
                "evidence": "...",
                "impact": "HIGH"
            }, ...
        ]
    }
    """

    def __init__(self, default_task_description: str = ""):
        self.default_task_description = default_task_description

    # ------------------------------------------------------------------
    # ACE DataProcessor interface
    # ------------------------------------------------------------------

    def process_task_data(self, raw_data: list[dict]) -> list[dict]:
        """Identity transform for API compatibility."""
        return raw_data

    def answer_is_correct(self, prediction: dict, ground_truth: dict) -> bool:
        """Check whether a prediction matches the ground truth."""
        return prediction_is_correct(prediction, ground_truth)

    def evaluate_accuracy(
        self, predictions: list[dict], ground_truths: list[dict]
    ) -> dict:
        """Compute accuracy, precision, recall, and F1 across a batch of predictions."""
        return compute_span_accuracy(predictions, ground_truths)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_samples_from_dirs(
        self,
        trace_dir: str | Path,
        annotation_dir: str | Path,
        limit: int | None = None,
    ) -> list[dict]:
        """Load all span samples from matching trace + annotation files."""
        trace_dir = Path(trace_dir)
        annotation_dir = Path(annotation_dir)

        trace_files = sorted(trace_dir.glob("*.json"))
        trace_files = [f for f in trace_files if not f.stem.endswith("_gt")]
        if limit:
            trace_files = trace_files[:limit]

        samples: list[dict] = []
        for tf in trace_files:
            trace_id = tf.stem
            ann_file = annotation_dir / f"{trace_id}_gt.json"
            if not ann_file.exists():
                ann_file = annotation_dir / f"{trace_id}.json"
            if not ann_file.exists():
                print(f"  [DataProcessor] No annotation found for {trace_id}, skipping")
                continue

            with open(tf) as f:
                trace = json.load(f)
            with open(ann_file) as f:
                annotation = json.load(f)

            samples.extend(self.extract_span_samples(trace, annotation))

        return samples

    def extract_span_samples(self, trace: dict, annotation: dict) -> list[dict]:
        """Extract individual span samples from a single trace + annotation pair."""
        trace_id = trace.get("trace_id", "unknown")
        task_description = (
            annotation.get("task_description")
            or trace.get("task_description")
            or self.default_task_description
        )
        raw_spans = trace.get("spans", {})
        if isinstance(raw_spans, list):
            spans: dict = {s["span_id"]: s for s in raw_spans if "span_id" in s}
        else:
            spans = raw_spans

        gt_per_span: dict[str, dict] = {
            sid: {"has_error": False, "errors": []} for sid in spans
        }
        for error in annotation.get("span_errors", []):
            sid = error.get("span_id")
            if sid and sid in gt_per_span:
                gt_per_span[sid]["has_error"] = True
                gt_per_span[sid]["errors"].append({
                    "category": error.get("category", ""),
                    "description": error.get("description", ""),
                    "evidence": error.get("evidence", ""),
                    "impact": error.get("impact", "MEDIUM"),
                })

        return [
            {
                "span_id": sid,
                "trace_id": trace_id,
                "span_info": span_data,
                "task_description": task_description,
                "ground_truth": gt_per_span.get(
                    sid, {"has_error": False, "errors": []}
                ),
            }
            for sid, span_data in spans.items()
        ]

    # ------------------------------------------------------------------
    # Trace-level loading (raw traces with child_spans hierarchy)
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_trace_spans(spans: list[dict], parent_id: str | None = None) -> list[dict]:
        """Recursively flatten hierarchical child_spans into a flat list."""
        result: list[dict] = []
        for span in spans:
            flat = {k: v for k, v in span.items() if k != "child_spans"}
            flat["parent_span_id"] = parent_id
            result.append(flat)
            result.extend(
                SpanDataProcessor._flatten_trace_spans(
                    span.get("child_spans", []),
                    parent_id=flat.get("span_id"),
                )
            )
        return result

    def load_trace_samples(
        self,
        trace_dir: str | Path,
        annotation_dir: str | Path,
        excluded_node_names: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Load raw traces and return trace-level dicts.

        Each trace dict contains both the full flattened span list (for the
        summarizer) and filtered per-span samples with ground truth (for the
        analyzer).
        """
        if excluded_node_names is None:
            excluded_node_names = DEFAULT_EXCLUDED_NODES

        trace_dir = Path(trace_dir)
        annotation_dir = Path(annotation_dir)

        trace_files = sorted(trace_dir.glob("*.json"))
        trace_files = [f for f in trace_files if not f.stem.endswith("_gt")]
        if limit:
            trace_files = trace_files[:limit]

        traces: list[dict] = []
        for tf in trace_files:
            trace_id = tf.stem

            ann_file = annotation_dir / f"{trace_id}.json"
            if not ann_file.exists():
                ann_file = annotation_dir / f"{trace_id}_gt.json"
            if not ann_file.exists():
                logger.warning(f"No annotation for {trace_id}, skipping")
                continue

            with open(tf) as f:
                raw_trace = json.load(f)
            with open(ann_file) as f:
                annotation = json.load(f)

            # 1. Flatten ALL spans for the summarizer
            top_spans = raw_trace.get("spans", [])
            if isinstance(top_spans, dict):
                all_spans = list(top_spans.values())
            else:
                all_spans = self._flatten_trace_spans(top_spans)

            # 2. Run the filtering pipeline for the analyzer
            try:
                filtered_spans, task_desc = extract_filtered_spans(
                    raw_trace, excluded_node_names
                )
            except Exception as e:
                logger.error(f"[{trace_id}] Filtering failed: {e}")
                continue

            if not filtered_spans:
                logger.warning(f"[{trace_id}] No spans survived filtering")
                continue

            # 3. Build ground-truth from GAIA annotation format
            gt_data = build_gt_json(
                annotation, set(filtered_spans.keys()), task_desc
            )

            # 4. Build per-span samples
            gt_per_span: dict[str, dict] = {
                sid: {"root_cause": None, "symptoms": []} for sid in filtered_spans
            }
            for err in gt_data.get("span_errors", []):
                sid = err.get("span_id")
                role = err.get("role", "symptom")
                if sid and sid in gt_per_span:
                    entry = {k: v for k, v in err.items() if k not in ("span_id", "role")}
                    if role == "root_cause":
                        gt_per_span[sid]["root_cause"] = entry
                    else:
                        gt_per_span[sid]["symptoms"].append(entry)

            span_samples = [
                {
                    "span_id": sid,
                    "trace_id": trace_id,
                    "span_info": span_data,
                    "task_description": task_desc,
                    "ground_truth": gt_per_span.get(
                        sid, {"has_error": False, "errors": []}
                    ),
                }
                for sid, span_data in filtered_spans.items()
            ]

            traces.append({
                "trace_id": trace_id,
                "task_description": task_desc,
                "all_spans": all_spans,
                "span_samples": span_samples,
            })

        return traces


# ---------------------------------------------------------------------------
# Per-trace file processing (for CLI)
# ---------------------------------------------------------------------------

def process_trace(
    trace_id: str,
    trace_dir: Path,
    annotation_dir: Path,
    output_dir: Path,
    excluded_node_names: list[str],
    split: str,
) -> bool:
    """Process one trace and write <trace_id>.json + <trace_id>_gt.json."""
    trace_file = trace_dir / f"{trace_id}.json"
    annotation_file = annotation_dir / f"{trace_id}.json"

    if not trace_file.exists():
        logger.error(f"Trace not found: {trace_file}")
        return False
    if not annotation_file.exists():
        logger.error(f"Annotation not found: {annotation_file}")
        return False

    with open(trace_file) as f:
        trace = json.load(f)
    with open(annotation_file) as f:
        annotation = json.load(f)

    try:
        filtered_spans, task_desc = extract_filtered_spans(trace, excluded_node_names)
    except Exception as e:
        logger.error(f"[{trace_id}] Pipeline failed: {e}", exc_info=True)
        return False

    if not filtered_spans:
        logger.warning(f"[{trace_id}] No spans survived filtering — skipping")
        return False

    errors_total = len(annotation.get("errors", []))
    trace_out_data = build_trace_json(trace_id, filtered_spans, task_desc)
    gt_out_data = build_gt_json(annotation, set(filtered_spans.keys()), task_desc)
    errors_kept = len(gt_out_data["span_errors"])

    logger.info(
        f"[{split}] {trace_id[:16]}…  "
        f"{len(filtered_spans)} spans, "
        f"{errors_kept}/{errors_total} errors retained"
    )

    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{trace_id}.json", "w") as f:
        json.dump(trace_out_data, f, indent=2)
    with open(out_dir / f"{trace_id}_gt.json", "w") as f:
        json.dump(gt_out_data, f, indent=2)

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sample data from real GAIA traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", type=Path, default=None, metavar="DIR")
    parser.add_argument("--test", type=Path, default=None, metavar="DIR")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--excluded-nodes", nargs="+", default=DEFAULT_EXCLUDED_NODES, metavar="NAME")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s  %(message)s",
    )

    if not args.train and not args.test:
        parser.error("Provide at least one directory via --train or --test")

    def scan_trace_ids(directory: Path) -> list[str]:
        if not directory.is_dir():
            logger.error(f"Not a directory: {directory}")
            return []
        ids = sorted(p.stem for p in directory.glob("*.json"))
        logger.info(f"Found {len(ids)} trace files in {directory}")
        return ids

    counts: dict[str, dict[str, int]] = {
        "train": {"ok": 0, "fail": 0},
        "test":  {"ok": 0, "fail": 0},
    }

    for split, split_dir in (("train", args.train), ("test", args.test)):
        if split_dir is None:
            continue
        trace_ids = scan_trace_ids(split_dir)
        for trace_id in trace_ids:
            ok = process_trace(
                trace_id, args.trace_dir, args.annotation_dir,
                args.output_dir, args.excluded_nodes, split,
            )
            counts[split]["ok" if ok else "fail"] += 1

    print("\n=== Summary ===")
    for split in ("train", "test"):
        c = counts[split]
        if c["ok"] + c["fail"]:
            print(f"  {split:5s}: {c['ok']} written, {c['fail']} failed")
    print(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
