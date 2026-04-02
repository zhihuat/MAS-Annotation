"""
Shared utilities for formatting APO insights before prompt injection.
"""

import re

_INSIGHT_LINE_PATTERN = re.compile(
    r"(\[(?:det|mis|cat|exa)-\d{5}\]) helpful=(\d+) harmful=(\d+) :: (.*)"
)


def extract_insights_for_prompt(insights: str) -> str:
    """
    Format insights as a bullet list and prune clearly harmful bullets.

    Keeps section headers and bullet IDs so analyzers can reference insight IDs.
    Returns an empty string when no bullet lines survive pruning.
    """
    output: list[str] = []
    has_bullets = False

    for line in insights.split("\n"):
        m = _INSIGHT_LINE_PATTERN.match(line)
        if m:
            bullet_id = m.group(1)
            helpful = int(m.group(2))
            harmful = int(m.group(3))
            content = m.group(4)
            if harmful > helpful and harmful >= 2:
                continue
            output.append(f"- {bullet_id} {content}")
            has_bullets = True
        elif line.startswith("## "):
            output.append(line)
        else:
            output.append(line)

    return "\n".join(output).strip() if has_bullets else ""
