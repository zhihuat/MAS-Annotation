#!/usr/bin/env python3
"""
Inter-Annotator Agreement (IAA) script for mas-annotation.

Usage:
    python compute_iaa.py
    python compute_iaa.py --annotator1 zhihuat --annotator2 collaborator
    python compute_iaa.py --out my_report.md

Computes:
    - Percent exact agreement on root_cause_step
    - Cohen's kappa (unweighted, sklearn)
    - Mean absolute step distance
    - Confusion matrix (step indices 1–8, bucket "8+" for higher)

Output: iaa_report.md (or --out path)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    print("Error: scikit-learn is required. Run: pip install scikit-learn")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
ANNOTATIONS_ROOT = ROOT / "data" / "annotations"

BUCKET_MAX = 8   # step indices > BUCKET_MAX go into "8+" bucket
MIN_TRACES = 2   # minimum overlapping traces needed
WARN_TRACES = 20 # warn if fewer than this many overlapping traces


# ── Load ──────────────────────────────────────────────────────────

def load_annotations(annotator: str) -> dict[str, dict]:
    """Load all non-excluded annotations for an annotator. Returns {trace_id: annotation}."""
    ann_dir = ANNOTATIONS_ROOT / annotator
    if not ann_dir.exists():
        print(f"Error: annotation directory not found: {ann_dir}")
        sys.exit(1)

    result = {}
    for f in ann_dir.glob("*.json"):
        text = f.read_text().strip()
        if not text:
            continue
        data = json.loads(text)
        if data.get("excluded", False):
            continue
        tid = data.get("trace_id", f.stem)
        result[tid] = data

    return result


def discover_annotators() -> list[str]:
    """Return sorted list of annotator names (subdirectory names under data/annotations/)."""
    if not ANNOTATIONS_ROOT.exists():
        return []
    return sorted(d.name for d in ANNOTATIONS_ROOT.iterdir() if d.is_dir())


# ── Metrics ───────────────────────────────────────────────────────

def bucket(step: int | None) -> str:
    """Convert step number to confusion matrix bucket."""
    if step is None:
        return "?"
    if step > BUCKET_MAX:
        return f"{BUCKET_MAX}+"
    return str(step)


def compute_metrics(pairs: list[tuple[int | None, int | None]]) -> dict:
    """Compute agreement metrics from a list of (step_a, step_b) pairs."""
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    n_valid = len(valid)
    n_total = len(pairs)
    n_missing = n_total - n_valid

    if n_valid == 0:
        return {
            "n_total": n_total, "n_valid": n_valid, "n_missing": n_missing,
            "pct_agreement": None, "kappa": None, "mad": None,
        }

    # Percent agreement
    agree = sum(1 for a, b in valid if a == b)
    pct_agreement = agree / n_valid * 100

    # Mean absolute distance
    mad = sum(abs(a - b) for a, b in valid) / n_valid

    # Cohen's kappa (labels as raw integers)
    labels_a = [a for a, _ in valid]
    labels_b = [b for _, b in valid]
    try:
        kappa = cohen_kappa_score(labels_a, labels_b)
    except Exception:
        kappa = None

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "pct_agreement": pct_agreement,
        "kappa": kappa,
        "mad": mad,
        "pairs": valid,
    }


def build_confusion_matrix(pairs: list[tuple[int, int]]) -> tuple[list[str], list[list[int]]]:
    """Build confusion matrix with buckets 1..BUCKET_MAX and '8+'."""
    labels_present = set()
    for a, b in pairs:
        labels_present.add(bucket(a))
        labels_present.add(bucket(b))

    # Ordered labels: 1, 2, ..., BUCKET_MAX, "8+"
    ordered = [str(i) for i in range(1, BUCKET_MAX + 1) if str(i) in labels_present]
    plus_label = f"{BUCKET_MAX}+"
    if plus_label in labels_present:
        ordered.append(plus_label)

    label_idx = {l: i for i, l in enumerate(ordered)}
    n = len(ordered)
    matrix = [[0] * n for _ in range(n)]
    for a, b in pairs:
        ba, bb = bucket(a), bucket(b)
        if ba in label_idx and bb in label_idx:
            matrix[label_idx[ba]][label_idx[bb]] += 1

    return ordered, matrix


def kappa_interpretation(k: float | None) -> str:
    if k is None:
        return "N/A"
    if k < 0:
        return "poor (worse than chance)"
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


# ── Report ────────────────────────────────────────────────────────

def render_report(
    annotator1: str,
    annotator2: str,
    ann1: dict[str, dict],
    ann2: dict[str, dict],
    common_ids: list[str],
    metrics: dict,
    cm_labels: list[str],
    cm_matrix: list[list[int]],
    per_trace: list[dict],
) -> str:
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines += [
        "# Inter-Annotator Agreement Report",
        f"Generated: {now}",
        f"Annotators: {annotator1}, {annotator2}",
        f"Traces compared: {len(common_ids)} "
        f"({annotator1}: {len(ann1)} completed, {annotator2}: {len(ann2)} completed)",
        "",
    ]

    if metrics["n_missing"] > 0:
        lines.append(
            f"> **Note:** {metrics['n_missing']} trace(s) had `root_cause_step = null` "
            "in at least one annotator and were excluded from kappa/MAD computation "
            "(still shown in per-trace table)."
        )
        lines.append("")

    if metrics["n_valid"] < WARN_TRACES:
        lines.append(
            f"> **Warning:** Only {metrics['n_valid']} traces have `root_cause_step` set "
            "by both annotators. Cohen's kappa is unstable at N < 20. "
            "Interpret with caution."
        )
        lines.append("")

    # Metrics
    lines += [
        "## Root Cause Step Attribution",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]
    if metrics["pct_agreement"] is not None:
        lines.append(f"| Percent exact agreement | {metrics['pct_agreement']:.1f}% |")
    else:
        lines.append("| Percent exact agreement | N/A |")

    if metrics["kappa"] is not None:
        interp = kappa_interpretation(metrics["kappa"])
        lines.append(f"| Cohen's kappa | {metrics['kappa']:.3f} ({interp}) |")
    else:
        lines.append("| Cohen's kappa | N/A |")

    if metrics["mad"] is not None:
        lines.append(f"| Mean absolute step distance | {metrics['mad']:.2f} steps |")
    else:
        lines.append("| Mean absolute step distance | N/A |")

    lines += [
        "",
        "> **Methods note:** Cohen's kappa computed using `sklearn.metrics.cohen_kappa_score` "
        "(unweighted). Each step index is treated as a nominal category. "
        "No 95% CI reported — bootstrapping adds complexity without proportional value at this N. "
        "Landis-Koch scale: <0.00 poor, 0.00–0.20 slight, 0.21–0.40 fair, "
        "0.41–0.60 moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect.",
        "",
    ]

    # Confusion matrix
    if cm_labels and any(any(row) for row in cm_matrix):
        lines += [
            "## Confusion Matrix (root_cause_step)",
            "",
            f"Rows = {annotator1}, Columns = {annotator2}",
            "",
        ]
        header = "| (A1 \\ A2) | " + " | ".join(cm_labels) + " |"
        sep = "|-----------|" + "|---------|" * len(cm_labels)
        lines += [header, sep]
        for i, label in enumerate(cm_labels):
            row_vals = " | ".join(str(cm_matrix[i][j]) for j in range(len(cm_labels)))
            lines.append(f"| **{label}** | {row_vals} |")
        lines.append("")

    # Per-trace table
    lines += [
        "## Per-Trace Summary",
        "",
        f"| trace_id | {annotator1}_step | {annotator2}_step | diff | flag |",
        "|----------|----------|----------|------|------|",
    ]
    for t in per_trace:
        a = str(t["step_a"]) if t["step_a"] is not None else "—"
        b = str(t["step_b"]) if t["step_b"] is not None else "—"
        diff = t["diff"]
        diff_str = str(diff) if diff is not None else "—"
        flag = t["flag"]
        tid_short = t["trace_id"][:16] + "..."
        lines.append(f"| `{tid_short}` | {a} | {b} | {diff_str} | {flag} |")
    lines.append("")

    # High-disagreement traces
    disagree = [t for t in per_trace if t["flag"] == "⚠ disagree"]
    if disagree:
        lines += [
            "## High-Disagreement Traces (|diff| > 1)",
            "",
        ]
        for t in disagree:
            lines.append(f"### `{t['trace_id']}`")
            lines.append(f"- {annotator1}: Step {t['step_a']}")
            lines.append(f"- {annotator2}: Step {t['step_b']}")
            if t.get("reasoning_a"):
                lines.append(f"- {annotator1} reasoning: \"{t['reasoning_a']}\"")
            if t.get("reasoning_b"):
                lines.append(f"- {annotator2} reasoning: \"{t['reasoning_b']}\"")
            lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement")
    parser.add_argument("--annotator1", help="First annotator name")
    parser.add_argument("--annotator2", help="Second annotator name")
    parser.add_argument("--out", default="iaa_report.md", help="Output markdown file")
    args = parser.parse_args()

    # Discover annotators
    all_annotators = discover_annotators()
    if not all_annotators:
        print(f"Error: no annotator directories found under {ANNOTATIONS_ROOT}")
        sys.exit(1)

    if args.annotator1 and args.annotator2:
        annotator1, annotator2 = args.annotator1, args.annotator2
    elif len(all_annotators) >= 2:
        annotator1, annotator2 = all_annotators[0], all_annotators[1]
        print(f"Auto-selected annotators: {annotator1} and {annotator2}")
    else:
        print(f"Error: only one annotator found ({all_annotators[0]}). "
              "Need at least two. Use --annotator1 and --annotator2.")
        sys.exit(1)

    print(f"Loading {annotator1}...")
    ann1 = load_annotations(annotator1)
    print(f"  {len(ann1)} non-excluded annotations")

    print(f"Loading {annotator2}...")
    ann2 = load_annotations(annotator2)
    print(f"  {len(ann2)} non-excluded annotations")

    common_ids = sorted(set(ann1) & set(ann2))
    if len(common_ids) < MIN_TRACES:
        print(
            f"Error: only {len(common_ids)} overlapping trace(s) found. "
            f"Need at least {MIN_TRACES} to compute agreement."
        )
        sys.exit(1)

    print(f"Overlapping traces: {len(common_ids)}")

    # Build per-trace data
    per_trace = []
    pairs = []
    for tid in common_ids:
        step_a = ann1[tid].get("root_cause_step")
        step_b = ann2[tid].get("root_cause_step")
        reasoning_a = ann1[tid].get("root_cause_reasoning", "")
        reasoning_b = ann2[tid].get("root_cause_reasoning", "")

        if step_a is not None and step_b is not None:
            diff = abs(step_a - step_b)
            flag = "agree" if diff == 0 else ("1-off" if diff == 1 else "⚠ disagree")
        else:
            diff = None
            flag = "missing"

        pairs.append((step_a, step_b))
        per_trace.append({
            "trace_id": tid,
            "step_a": step_a,
            "step_b": step_b,
            "diff": diff,
            "flag": flag,
            "reasoning_a": reasoning_a,
            "reasoning_b": reasoning_b,
        })

    metrics = compute_metrics(pairs)

    valid_pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    cm_labels, cm_matrix = build_confusion_matrix(valid_pairs) if valid_pairs else ([], [])

    report = render_report(
        annotator1, annotator2, ann1, ann2,
        common_ids, metrics, cm_labels, cm_matrix, per_trace,
    )

    out_path = ROOT / args.out
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")

    # Summary to stdout
    print()
    if metrics["pct_agreement"] is not None:
        print(f"  Percent agreement:  {metrics['pct_agreement']:.1f}%")
    if metrics["kappa"] is not None:
        print(f"  Cohen's kappa:      {metrics['kappa']:.3f} ({kappa_interpretation(metrics['kappa'])})")
    if metrics["mad"] is not None:
        print(f"  Mean |step diff|:   {metrics['mad']:.2f}")
    disagree_n = sum(1 for t in per_trace if t["flag"] == "⚠ disagree")
    print(f"  High-disagreement:  {disagree_n} trace(s)")


if __name__ == "__main__":
    main()
