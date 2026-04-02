"""
Configuration management for the progress monitor module.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProgressMonitorConfig:
    """Configuration for progress monitoring pipeline.

    All path fields are explicit:
    - summary_file: file path to trace_summaries.json
    - trace_dir, annotation_dir, output_dir, cache_dir: directory paths
    No implicit path joining is performed.
    """

    # Data paths (explicit file vs directory)
    summary_file: str = "data/summary/claude-haiku-4-5-20251001/trace_summaries.json"
    trace_dir: str = "data/GAIA"
    annotation_dir: str = "data/progress_annotations"
    output_dir: str = "results/progress_monitor"
    cache_dir: str = "results/progress_monitor/cache"

    # LLM parameters
    llm_model: str = "openai/gpt-oss-120b" # "openai/Kimi-K2.5"# "google/gemini-2.5-flash"
    temperature: float = 0.0
    max_completion_tokens: int = 8192
    reasoning_effort: str = "low"

    # Flatline detection (strict by default: 0 means any zero-delta suffix is flatline)
    flatline_noise_tolerance: int = 0

    # Spans to skip (boilerplate, not real execution steps)
    skip_span_names: list[str] = field(default_factory=lambda: [
        "main",
        "get_examples_to_answer",
        "answer_single_question",
        "create_agent_hierarchy",
    ])

    # Execution
    max_workers: int = 2
    limit: int | None = None
    verbose: bool = False
    force_restart: bool = False  # ignore existing cache, re-run all LLM evals, save new results

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.max_completion_tokens < 1:
            raise ValueError(
                f"max_completion_tokens must be >= 1, got {self.max_completion_tokens}"
            )
        if self.reasoning_effort not in ("low", "medium", "high"):
            raise ValueError(
                f"reasoning_effort must be 'low', 'medium', or 'high', got '{self.reasoning_effort}'"
            )
        if self.flatline_noise_tolerance < 0:
            raise ValueError(
                f"flatline_noise_tolerance must be >= 0, got {self.flatline_noise_tolerance}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "ProgressMonitorConfig":
        """Load configuration from a dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProgressMonitorConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Export configuration to dictionary."""
        return {
            "summary_file": self.summary_file,
            "trace_dir": self.trace_dir,
            "annotation_dir": self.annotation_dir,
            "output_dir": self.output_dir,
            "cache_dir": self.cache_dir,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
            "flatline_noise_tolerance": self.flatline_noise_tolerance,
            "skip_span_names": self.skip_span_names,
            "max_workers": self.max_workers,
            "limit": self.limit,
            "verbose": self.verbose,
            "force_restart": self.force_restart,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
