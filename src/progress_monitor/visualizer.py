"""
Visualization utilities for progress monitoring.

Plots value function curves for individual traces and batch summaries.
Uses matplotlib with Agg backend for headless environments.
"""

import logging
from pathlib import Path
from typing import Any

from src.progress_monitor.flatline_detector import FlatlineResult
from src.progress_monitor.value_function import StepProgress

logger = logging.getLogger(__name__)

# Headless backend — must be set before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class ProgressVisualizer:
    """Plots progress value function curves for trace analysis."""

    # Colors for cycling through stage bars
    _STAGE_COLORS = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
        "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
        "#9C755F", "#BAB0AC",
    ]

    @staticmethod
    def _build_stage_spans(
        progress: list[StepProgress],
    ) -> list[tuple[str, int, int]]:
        """Group consecutive steps by parent_chain_name into stage spans.

        Returns:
            List of (stage_name, start_index, end_index) tuples.
        """
        if not progress:
            return []

        spans: list[tuple[str, int, int]] = []
        current_name = progress[0].parent_chain_name
        start = 0

        for i in range(1, len(progress)):
            if progress[i].parent_chain_name != current_name:
                if current_name:  # only add named stages
                    spans.append((current_name, start, i - 1))
                current_name = progress[i].parent_chain_name
                start = i

        # Final group
        if current_name:
            spans.append((current_name, start, len(progress) - 1))

        return spans

    @staticmethod
    def plot_single_trace(
        progress: list[StepProgress],
        flatline: FlatlineResult | None = None,
        title: str = "",
        output_path: str | Path | None = None,
    ) -> None:
        """Plot V(t) for a single trace with stage lines and flatline marker.

        X-axis: execution steps (labeled by step_name).
        Y-axis: cumulative progress V(t) in [0.0, 1.0].
        Below x-axis: colored stage bars showing CHAIN/AGENT span groupings.
        Green dots for steps with progress; red dots for zero-progress;
        orange triangles for hallucinated steps.
        Vertical dashed red line and shaded zone at flatline onset.

        Args:
            progress: List of StepProgress in execution order.
            flatline: Optional FlatlineResult for annotation.
            title: Plot title.
            output_path: If provided, save the figure to this path.
        """
        if not progress:
            logger.warning("No progress data to plot")
            return

        stage_spans = ProgressVisualizer._build_stage_spans(progress)

        # Extra bottom margin for stage bars
        stage_rows = 1 if stage_spans else 0
        fig_height = 5 + stage_rows * 0.6
        fig, ax = plt.subplots(figsize=(max(10, len(progress) * 0.8), fig_height))

        x = list(range(len(progress)))
        y = [s.cumulative_value for s in progress]
        deltas = [s.progress_delta for s in progress]

        # Plot the value function line
        ax.plot(x, y, "b-", linewidth=1.5, alpha=0.7, label="V(t)")

        # Color dots by status
        for i in range(len(x)):
            if getattr(progress[i], "is_hallucination", False):
                color, marker = "orange", "^"
            elif deltas[i] > 1e-9:
                color, marker = "green", "o"
            else:
                color, marker = "red", "o"
            ax.plot(x[i], y[i], marker, color=color, markersize=6, zorder=5)

        # Annotate flatline
        if flatline and flatline.has_flatline and flatline.flatline_start_index is not None:
            fl_idx = flatline.flatline_start_index
            if fl_idx < len(x):
                ax.axvline(
                    x=fl_idx, color="red", linestyle="--", linewidth=1.5,
                    alpha=0.7, label=f"Flatline onset (step {fl_idx})"
                )
                ax.axvspan(
                    fl_idx, len(x) - 1, alpha=0.1, color="red",
                    label="Flatline zone"
                )

        # Step labels on x-axis
        step_labels = []
        for s in progress:
            label = s.step_name
            if len(label) > 25:
                label = label[:22] + "..."
            step_labels.append(label)

        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Cumulative Progress V(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title or "Progress Value Function")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Draw stage bars below x-axis
        if stage_spans:
            # Use axes transform for y so bars sit just below the plot
            stage_y = -0.30  # in axes coords
            bar_height = 0.04
            colors = ProgressVisualizer._STAGE_COLORS

            for idx, (name, start, end) in enumerate(stage_spans):
                color = colors[idx % len(colors)]
                # Draw bar in data x coords, axes y coords
                bar_left = start - 0.4
                bar_width = (end - start) + 0.8
                ax.add_patch(plt.Rectangle(
                    (bar_left, stage_y), bar_width, bar_height,
                    transform=ax.get_xaxis_transform(),
                    color=color, alpha=0.6, clip_on=False,
                ))
                # Stage label centered below bar
                mid_x = (start + end) / 2
                short_name = name if len(name) <= 30 else name[:27] + "..."
                ax.text(
                    mid_x, stage_y - 0.02, short_name,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=6,
                    color=color, fontweight="bold",
                )

            ax.set_xlabel("Execution Steps", labelpad=25)
        else:
            ax.set_xlabel("Execution Steps")

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {output_path}")

        plt.close(fig)

    @staticmethod
    def plot_batch_summary(
        results: list[dict[str, Any]],
        output_path: str | Path | None = None,
    ) -> None:
        """Plot summary statistics across multiple traces.

        Generates a 2x2 figure with:
        - Top-left: Histogram of final V(T) values
        - Top-right: Distribution of flatline onset positions (normalized)
        - Bottom-left: V(T) for success vs failure traces
        - Bottom-right: Number of steps vs final V(T) scatter

        Args:
            results: List of dicts with keys: trace_id, final_value,
                has_flatline, flatline_start_index, total_steps, trace_outcome.
            output_path: If provided, save the figure to this path.
        """
        if not results:
            logger.warning("No results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        final_values = [r["final_value"] for r in results]
        flatline_positions = [
            r["flatline_start_index"] / max(r["total_steps"], 1)
            for r in results
            if r.get("has_flatline") and r.get("flatline_start_index") is not None
        ]
        success_values = [
            r["final_value"] for r in results
            if r.get("trace_outcome") == "Success"
        ]
        failure_values = [
            r["final_value"] for r in results
            if r.get("trace_outcome") == "Failure"
        ]
        total_steps = [r["total_steps"] for r in results]

        # Top-left: Histogram of final V(T)
        ax = axes[0, 0]
        ax.hist(final_values, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Final V(T)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Final Progress Values")
        ax.axvline(x=1.0, color="green", linestyle="--", alpha=0.5, label="Complete")

        # Top-right: Flatline onset positions
        ax = axes[0, 1]
        if flatline_positions:
            ax.hist(
                flatline_positions, bins=15, edgecolor="black",
                alpha=0.7, color="salmon"
            )
        ax.set_xlabel("Flatline Onset (normalized position)")
        ax.set_ylabel("Count")
        ax.set_title("When Does Progress Stall?")

        # Bottom-left: Success vs Failure
        ax = axes[1, 0]
        labels, data, colors = [], [], []
        if success_values:
            labels.append(f"Success (n={len(success_values)})")
            data.append(success_values)
            colors.append("green")
        if failure_values:
            labels.append(f"Failure (n={len(failure_values)})")
            data.append(failure_values)
            colors.append("red")
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.3)
        ax.set_ylabel("Final V(T)")
        ax.set_title("Progress by Trace Outcome")

        # Bottom-right: Steps vs V(T) scatter
        ax = axes[1, 1]
        ax.scatter(total_steps, final_values, alpha=0.5, c="steelblue", edgecolors="black", s=30)
        ax.set_xlabel("Number of Execution Steps")
        ax.set_ylabel("Final V(T)")
        ax.set_title("Steps vs Progress")

        plt.suptitle("Progress Monitor — Batch Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            logger.info(f"Saved batch summary plot to {output_path}")

        plt.close(fig)
