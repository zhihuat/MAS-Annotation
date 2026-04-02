"""
Telegram notification helper for failure analysis results.

Fill in BOT_TOKEN and CHAT_ID before use:
  - BOT_TOKEN: from @BotFather  (e.g. "123456:ABC-...")
  - CHAT_ID:   your chat/group ID (e.g. "123456789")
"""
import os
import json
import logging
import urllib.request
import urllib.error
from typing import Any

logger = logging.getLogger(__name__)

# ── Fill these in ────────────────────────────────────────────────────────────
BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID:   str = os.getenv("TELEGRAM_CHAT_ID")
# ─────────────────────────────────────────────────────────────────────────────


def send_evaluation_results(results_dict: dict[str, Any]) -> bool:
    """
    Send evaluation summary to Telegram.

    Args:
        results_dict: The dict returned by EvaluationRunner.run_evaluation()

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram BOT_TOKEN / CHAT_ID not set — skipping notification")
        return False

    try:
        message = _format_message(results_dict)
        return _send(message)
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def send_pipeline_results(results: dict[str, Any]) -> bool:
    """
    Send prompt-optimization pipeline summary to Telegram.

    Args:
        results: The dict returned by PromptOptimizationPipeline.run()

    Returns:
        True if the message was sent successfully, False otherwise.
    """
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram BOT_TOKEN / CHAT_ID not set — skipping notification")
        return False

    try:
        return _send(_format_pipeline_message(results))
    except Exception as e:
        logger.warning(f"Telegram notification failed: {e}")
        return False


def _format_pipeline_message(results: dict[str, Any]) -> str:
    def _fmt(metrics: dict) -> str:
        acc = metrics.get("accuracy", 0)
        f1  = metrics.get("f1", {})
        return (
            f"Acc `{acc:.3f}` | "
            f"F1 `{f1.get('f1', 0):.3f}` | "
            f"P `{f1.get('precision', 0):.3f}` | "
            f"R `{f1.get('recall', 0):.3f}`"
        )

    lines = ["🤖 *Prompt Optimization Done*", ""]

    if "initial_metrics" in results and "final_metrics" in results:
        init  = results["initial_metrics"]
        final = results["final_metrics"]
        best  = results.get("training_results", {}).get("best_accuracy", 0)
        delta = final.get("accuracy", 0) - init.get("accuracy", 0)
        sign  = "+" if delta >= 0 else ""

        lines += [
            f"Initial  : {_fmt(init)}",
            f"Best acc : `{best:.3f}`",
            f"Final    : {_fmt(final)}",
            f"Δ accuracy: `{sign}{delta:.3f}`",
        ]
    else:
        m = results.get("metrics", {})
        lines.append(_fmt(m))

    return "\n".join(lines)


def _format_message(results_dict: dict[str, Any]) -> str:
    cfg = results_dict.get("config", {})
    agg = results_dict.get("aggregated_metrics", {})
    n   = results_dict.get("traces_processed", 0)
    ts  = results_dict.get("timestamp", "")[:19].replace("T", " ")

    lines = [
        "📊 *Failure Analysis Done*",
        "",
        f"🕐 {ts}",
        f"Algo: `{cfg.get('algorithm', '?')}` | "
        f"Strategy: `{cfg.get('prompt_strategy', '?')}`",
        f"Traces: *{n}*",
        "",
        "*Joint F1* (location + category)",
        f"  F1 `{agg.get('joint_f1', 0):.4f}` | "
        f"P `{agg.get('joint_precision', 0):.4f}` | "
        f"R `{agg.get('joint_recall', 0):.4f}`",
        f"  TP/FP/FN: `{agg.get('joint_tp',0)}`/`{agg.get('joint_fp',0)}`/`{agg.get('joint_fn',0)}`",
        "",
        "*Location F1* (span only)",
        f"  F1 `{agg.get('location_f1', 0):.4f}` | "
        f"P `{agg.get('location_precision', 0):.4f}` | "
        f"R `{agg.get('location_recall', 0):.4f}`",
        f"  TP/FP/FN: `{agg.get('location_tp',0)}`/`{agg.get('location_fp',0)}`/`{agg.get('location_fn',0)}`",
    ]
    return "\n".join(lines)


def _send(text: str) -> bool:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
        if not body.get("ok"):
            logger.warning(f"Telegram API error: {body}")
            return False
    logger.info("Telegram notification sent")
    return True
