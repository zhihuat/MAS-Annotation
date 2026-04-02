"""LLM interface and provider implementations.

Provides a unified LLMInterface ABC and concrete implementations for
OpenAI, Anthropic, and Google (Gemini) APIs.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from LLM completion."""
    content: str
    model: str
    usage: dict[str, int]  # {"prompt_tokens": X, "completion_tokens": Y}
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LLMInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from messages.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0.0 to 2.0)
            **kwargs: Provider-specific arguments

        Returns:
            LLMResponse with generated content and metadata
        """
        pass

    @abstractmethod
    def complete_json(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        **kwargs
    ) -> dict[str, Any] | BaseModel:
        """Generate JSON completion with optional Pydantic model validation.

        Args:
            messages: List of conversation messages
            response_model: Optional Pydantic model class for structured output
            **kwargs: Provider-specific arguments

        Returns:
            Parsed JSON dict, or validated Pydantic model instance if response_model is provided
        """
        pass

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_completion_tokens: int = 500,
    ) -> dict[str, Any]:
        """Convenience wrapper: build messages from prompt/system, then call complete_json."""
        messages: list[Message] = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))
        return self.complete_json(
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
        )



def _parse_json(response: str) -> dict | list | None:
    """Parse JSON from an LLM response, stripping markdown code fences if present."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "```":
                end = i
                break
        text = "\n".join(lines[start:end])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    arr_start = text.find("[")
    arr_end = text.rfind("]") + 1
    if arr_start != -1 and arr_end > arr_start:
        try:
            return json.loads(text[arr_start:arr_end])
        except json.JSONDecodeError:
            pass

    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx])
        except json.JSONDecodeError:
            pass

    return None


def _to_dict_messages(messages: list) -> list[dict[str, str]]:
    """Convert a list of Message objects or dicts to dict format."""
    result = []
    for m in messages:
        if isinstance(m, Message):
            result.append({"role": m.role, "content": m.content})
        else:
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# Base client (implements LLMInterface via _call_api pattern)
# ---------------------------------------------------------------------------

class BaseLLMClient(LLMInterface):
    """Abstract base class for provider-specific LLM clients.

    Subclasses only need to implement ``_call_api`` which maps provider SDK
    calls to a simple ``(messages, temperature) -> str`` contract.  All
    higher-level methods (``complete``, ``complete_json``) are provided
    by this base class. ``generate_json`` is inherited from LLMInterface.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 2000
    ):
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def _call_api(self, messages: list[dict[str, str]], temperature: float) -> str:
        """Call the provider-specific API. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _call_api()")

    # -- LLMInterface implementation ----------------------------------------

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from messages."""
        dict_messages = _to_dict_messages(messages)
        content = self._call_api(dict_messages, temperature)
        return LLMResponse(
            content=content or "",
            model=self.model,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
            metadata={}
        )

    def complete_json(
        self,
        messages: list,
        response_model: type[BaseModel] | None = None,
        temperature: float | None = None,
        **kwargs
    ) -> dict[str, Any] | BaseModel:
        """Generate JSON completion. Accepts both Message objects and dicts."""
        dict_messages = _to_dict_messages(messages)

        temp = temperature if temperature is not None else self.temperature

        try:
            response_text = self._call_api(dict_messages, temp)
        except Exception as e:
            raise RuntimeError(f"API call failed for {self.model}: {e}") from e

        if response_text is None:
            raise RuntimeError(f"API returned None for {self.model}")

        if response_model is not None:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                end = next((i for i, l in enumerate(lines[1:], 1) if l.strip() == "```"), len(lines))
                text = "\n".join(lines[1:end])
            try:
                return response_model.model_validate_json(text)
            except Exception:
                parsed = _parse_json(text)
                if parsed is not None:
                    return response_model.model_validate(parsed)
                raise ValueError(f"Could not parse response as valid JSON: {text[:300]}")

        parsed = _parse_json(response_text)
        if parsed is None:
            raise ValueError(f"Could not extract JSON from response: {response_text}")
        return parsed

    # -- Utility ------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            response_text = self._call_api(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Reply in one short sentence."},
                    {"role": "user", "content": "Say hello and confirm this is a test."}
                ],
                temperature=0.0
            )
            logger.info(f"Connection test successful for {self.model}: {response_text}")
            return True
        except Exception as e:
            logger.error(f"Connection test failed for {self.model}: {e}")
            return False


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """OpenAI API client (also supports OpenAI-compatible endpoints)."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 2000,
        reasoning_effort: str | None = None,
    ):
        super().__init__(model, temperature, max_completion_tokens)
        self.reasoning_effort = reasoning_effort

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        if "gpt-oss" in model.lower() or "kimi-k2.5" in model.lower():
            base_url = os.getenv("ARC_API_BASE", "https://llm-api.arc.vt.edu/api/v1")
            api_key = os.getenv("ARC_API_KEY", "sk-anything")
        else:
            base_url = None
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _call_api(self, messages: list[dict[str, str]], temperature: float) -> str:
        params = {
            "model": self.model,
            "messages": messages,
        }

        is_reasoning_model = self.model.startswith("o1") or self.model.startswith("o3") or self.model.startswith("gpt-5")
        if temperature is not None and not is_reasoning_model:
            params["temperature"] = temperature

        if self.max_completion_tokens:
            params["max_completion_tokens"] = self.max_completion_tokens

        if hasattr(self, 'reasoning_effort') and self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort

        try:
            response = self.client.chat.completions.create(**params)
            content = response.choices[0].message.content
            if not content:
                finish_reason = response.choices[0].finish_reason
                raise RuntimeError(f"Empty response from {self.model} (finish_reason={finish_reason}). Try increasing max_completion_tokens.")
            return content
        except RuntimeError:
            raise
        except Exception as e:
            raise e


class AnthropicClient(BaseLLMClient):
    """Anthropic (Claude) API client."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 2000
    ):
        super().__init__(model, temperature, max_completion_tokens)

        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)

    def _call_api(self, messages: list[dict[str, str]], temperature: float) -> str:
        system_msg = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_completion_tokens,
            temperature=temperature,
            system=system_msg if system_msg else "",
            messages=user_messages
        )
        return response.content[0].text


class GoogleClient(BaseLLMClient):
    """Google (Gemini) API client using the new google-genai SDK."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_completion_tokens: int = 2000
    ):
        super().__init__(model, temperature, max_completion_tokens)

        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai package required. Install with: pip install google-genai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key, vertexai=False)

    def _call_api(self, messages: list[dict[str, str]], temperature: float) -> str:
        from google.genai import types

        # Extract system instruction and build contents
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
            elif msg["role"] == "assistant":
                contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=self.max_completion_tokens,
            system_instruction=system_instruction,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        return response.text
