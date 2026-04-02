"""LLM client factory, call wrapper, and response parsing.

Shared by both failure_analysis and APO pipeline modules.
"""

import json
import time
from typing import Optional

from src.llm.interface import (
    BaseLLMClient, OpenAIClient, AnthropicClient, GoogleClient
)

DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def create_llm_client(
    model: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 2000,
    reasoning_effort: str | None = None,
) -> BaseLLMClient:
    """
    Create appropriate LLM client based on model name.

    Args:
        model: "provider/model_name" (e.g., "openai/gpt-4o-mini", "anthropic/claude-haiku-4-5-20251001")
        reasoning_effort: Optional reasoning effort level ("low", "medium", "high").
    """
    if "/" not in model:
        raise ValueError(f"Model must include provider prefix (e.g., 'openai/gpt-4o-mini'): {model}")

    provider, model_name = model.split("/", 1)
    provider = provider.lower()

    if provider == "openai":
        return OpenAIClient(model_name, temperature, max_completion_tokens, reasoning_effort=reasoning_effort)
    elif provider == "anthropic":
        return AnthropicClient(model_name, temperature, max_completion_tokens)
    elif provider == "google":
        return GoogleClient(model_name, temperature, max_completion_tokens)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported: openai, anthropic, google")


# Alias for backward compatibility
create_pipeline_client = create_llm_client


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------

def call_llm(
    client: BaseLLMClient,
    messages: list[dict],
    max_completion_tokens: int = 8000,
    system: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    Make an LLM call with retry logic.

    Args:
        client: Any BaseLLMClient instance (Anthropic, OpenAI, Google)
        messages: List of {role, content} dicts
        max_completion_tokens: Maximum tokens to generate
        system: Optional system prompt (prepended as role=system message)
        max_retries: Number of retry attempts on transient errors
        retry_delay: Initial delay between retries (doubles on each retry)

    Returns:
        Response text (empty string on unrecoverable failure)
    """
    all_messages: list[dict] = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    delay = retry_delay
    for attempt in range(max_retries):
        try:
            response = client.complete(all_messages, temperature=0.0, max_completion_tokens=max_completion_tokens)
            return response.content
        except Exception as e:
            if attempt < max_retries - 1:
                err_lower = str(e).lower()
                if "rate" in err_lower or "429" in err_lower:
                    print(f"  [LLM] Rate limit hit, retrying in {delay:.1f}s...")
                else:
                    print(f"  [LLM] API error ({e}), retrying...")
                time.sleep(delay)
                delay *= 2
            else:
                raise

    return ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_json_response(response: str) -> Optional[dict | list]:
    """
    Parse JSON from an LLM response, stripping markdown code fences if present.

    Returns:
        Parsed dict/list, or None if parsing fails.
    """
    text = response.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])

    # Attempt direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fall back to finding the outermost JSON array
    arr_start = text.find("[")
    arr_end = text.rfind("]") + 1
    if arr_start != -1 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start:arr_end])
        except json.JSONDecodeError:
            pass

    # Fall back to finding the outermost JSON object
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx])
        except json.JSONDecodeError:
            pass

    return None
