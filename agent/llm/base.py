#!/usr/bin/env python3
"""
LLM client interface for the agent.

Provides a unified interface so the rest of the agent can use any supported
provider (OpenAI, Cerebras, etc.) by configuration without code changes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

# When stream=True, providers return their native stream object (OpenAI/Cerebras)
# Callers iterate over it; we don't force a common stream type.
StreamType = Any


class LLMClient(ABC):
    """
    Abstract base for LLM providers.

    Implementations: OpenAI, Cerebras, etc. All use the same complete() signature
    so the agent can swap providers via config (e.g. RAG_LLM_PROVIDER=openai).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g. 'OpenAI', 'Cerebras')."""
        pass

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, StreamType]:
        """
        Run a chat completion.

        Args:
            messages: List of {"role": "user"|"system"|"assistant", "content": "..."}.
            model: Override default model for this call (optional).
            temperature: Sampling temperature (optional).
            max_tokens: Max completion tokens (optional).
            stream: If True, return the provider's stream object; otherwise return content string.

        Returns:
            If stream=False: content string.
            If stream=True: provider-specific stream object (e.g. openai.Stream).
        """
        pass

    def is_available(self) -> bool:
        """Whether this client can make calls (e.g. API key present)."""
        return True
