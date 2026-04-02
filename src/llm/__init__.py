"""LLM interface, providers, and client utilities.

Submodules:
- interface: LLMInterface ABC, BaseLLMClient, provider implementations (OpenAI, Anthropic, Google)
- client: Factory, retry wrapper, JSON response parsing
"""

from src.llm.interface import (
    Message,
    LLMResponse,
    LLMInterface,
    BaseLLMClient,
    OpenAIClient,
    AnthropicClient,
    GoogleClient,
)
from src.llm.client import (
    DEFAULT_MODEL,
    create_llm_client,
    create_pipeline_client,
    call_llm,
    parse_json_response,
)

__all__ = [
    # interface
    "Message",
    "LLMResponse",
    "LLMInterface",
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
    # client
    "DEFAULT_MODEL",
    "create_llm_client",
    "create_pipeline_client",
    "call_llm",
    "parse_json_response",
]
