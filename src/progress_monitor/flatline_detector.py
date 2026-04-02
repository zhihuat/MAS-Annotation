"""
Flatline detection for root cause analysis.

Identifies the first execution step after which the value function permanently
stalls (all subsequent steps yield zero progress).
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any

from src.progress_monitor.config import ProgressMonitorConfig
from src.progress_monitor.value_function import StepProgress

logger = logging.getLogger(__name__)


@dataclass
class FlatlineResult:
    """Result of flatline detection for root cause analysis.

    Attributes:
        trace_id: The trace being analyzed.
        has_flatline: Whether a flatline was detected.
        flatline_start_index: Index into the StepProgress list where flatline begins.
            This is the first step from which all subsequent deltas are zero.
        flatline_start_span_id: Span ID of the step where flatline begins.
        flatline_start_step_name: Name of the step where flatline begins.
        final_value: V(T) at the end of the trace.
        total_steps: Total number of execution steps analyzed.
        progress_history: Cumulative values for all steps.
    """

    trace_id: str
    has_flatline: bool
    flatline_start_index: int | None = None
    flatline_start_span_id: str | None = None
    flatline_start_step_name: str | None = None
    final_value: float = 0.0
    total_steps: int = 0
    progress_history: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)


class FlatlineDetector:
    """Detects the first step after which progress permanently stalls.

    Detection Algorithm (Strict)
    ────────────────────────────
    A "flatline" is declared at step index i if ALL of the following hold:

    1. V(T) < 1.0 (the task was not completed)
    2. For all j in [i, i+1, ..., T]: delta(j) == 0
       (from step i onward, no step ever makes progress again)
    3. The root cause candidate is step i — the first step in the
       zero-progress suffix.

    Edge Cases:
    - All zeros from start (no progress ever made): flatline_start_index = 0
    - V(T) == 1.0 (task completed): has_flatline = False
    - Last step makes progress but V(T) < 1.0: has_flatline = False
      (progress didn't stall, it just didn't finish)
    - Empty trace: has_flatline = False

    Optional Noise Tolerance:
    If flatline_noise_tolerance > 0, up to that many steps with tiny delta
    (< 1e-6) within the suffix are tolerated. Default is 0 (strict mode).
    """

    def __init__(self, config: ProgressMonitorConfig) -> None:
        self._noise_tolerance = config.flatline_noise_tolerance

    def detect(self, progress: list[StepProgress], trace_id: str = "") -> FlatlineResult:
        """Scan progress history to find the flatline onset.

        Args:
            progress: List of StepProgress in execution order.
            trace_id: For logging and the result object.

        Returns:
            FlatlineResult describing whether and where a flatline was detected.
        """
        if not progress:
            return FlatlineResult(
                trace_id=trace_id,
                has_flatline=False,
                final_value=0.0,
                total_steps=0,
                progress_history=[],
            )

        n = len(progress)
        history = [s.cumulative_value for s in progress]
        final_value = progress[-1].cumulative_value

        # Task completed — no flatline
        if final_value >= 1.0:
            logger.info(f"[{trace_id}] Task completed (V={final_value:.3f}), no flatline")
            return FlatlineResult(
                trace_id=trace_id,
                has_flatline=False,
                final_value=final_value,
                total_steps=n,
                progress_history=history,
            )

        # Find the last step that made nonzero progress
        last_progress_idx = -1
        noise_count = 0

        for i in range(n - 1, -1, -1):
            if progress[i].progress_delta > 1e-9:
                if noise_count <= self._noise_tolerance:
                    last_progress_idx = i
                    break
                else:
                    # Beyond noise tolerance, treat this as genuine progress
                    last_progress_idx = i
                    break
            else:
                noise_count += 1

        if last_progress_idx == -1:
            # All steps had zero delta — flatline from the very start
            flatline_start = 0
        else:
            flatline_start = last_progress_idx + 1

        # Check if there actually is a flatline suffix
        if flatline_start >= n:
            # Last step made progress but didn't complete — no flatline
            logger.info(
                f"[{trace_id}] Last step made progress but V={final_value:.3f} < 1.0, "
                f"no flatline (task incomplete but not stalled)"
            )
            return FlatlineResult(
                trace_id=trace_id,
                has_flatline=False,
                final_value=final_value,
                total_steps=n,
                progress_history=history,
            )

        # Skip preparatory steps (fact summarization, plan generation)
        # to find the real root cause — the first non-preparatory step
        # in the zero-progress suffix.
        while (
            flatline_start < n
            and progress[flatline_start].status == "preparatory"
        ):
            flatline_start += 1

        if flatline_start >= n:
            # All remaining steps were preparatory — no actionable flatline
            logger.info(
                f"[{trace_id}] All zero-progress steps are preparatory, "
                f"no actionable flatline"
            )
            return FlatlineResult(
                trace_id=trace_id,
                has_flatline=False,
                final_value=final_value,
                total_steps=n,
                progress_history=history,
            )

        flatline_step = progress[flatline_start]
        logger.info(
            f"[{trace_id}] Flatline detected at step {flatline_start} "
            f"(span={flatline_step.span_id}, name='{flatline_step.step_name}'), "
            f"V stalled at {final_value:.3f}"
        )

        return FlatlineResult(
            trace_id=trace_id,
            has_flatline=True,
            flatline_start_index=flatline_start,
            flatline_start_span_id=flatline_step.span_id,
            flatline_start_step_name=flatline_step.step_name,
            final_value=final_value,
            total_steps=n,
            progress_history=history,
        )
