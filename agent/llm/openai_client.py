#!/usr/bin/env python3
"""OpenAI-backed LLM client for the agent."""

import logging
from typing import List, Dict, Any, Optional, Union

import openai

from .base import LLMClient, StreamType

logger = logging.getLogger(__name__)


class OpenAILLMClient(LLMClient):
    """LLM client using OpenAI API (OpenAI SDK)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-5-nano-2025-08-07",
        default_temperature: float = 1,
        default_max_tokens: int = 8000,
    ):
        self._api_key = api_key
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client: Optional[openai.OpenAI] = None

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OpenAI API key not set; set OPENAI_API_KEY or pass api_key")
            self._client = openai.OpenAI(api_key=self._api_key)
            logger.info("✅ OpenAI LLM client initialized (lazy)")
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key)

    @staticmethod
    def _extract_content(response: Any) -> Optional[str]:
        """Extract text from completion message. Handles content as string or list of parts (reasoning models)."""
        if not response.choices:
            return None
        msg = response.choices[0].message
        raw = getattr(msg, "content", None)
        if isinstance(raw, str):
            return raw if raw else None
        if isinstance(raw, list):
            # Content parts: e.g. [{"type": "output_text", "text": "..."}] or [{"type": "text", "text": "..."}]
            parts = []
            for part in raw:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if text:
                        parts.append(str(text))
                elif hasattr(part, "text"):
                    parts.append(part.text)
            return "".join(parts) if parts else None
        return None

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, StreamType]:
        model = model or self._default_model
        temperature = temperature if temperature is not None else self._default_temperature
        max_tokens = max_tokens or self._default_max_tokens

        # gpt-5-nano (and some reasoning models) only support temperature=1
        if "gpt-5-nano" in (model or "").lower() and temperature != 1:
            temperature = 1

        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=stream,
        )
        response = self.client.chat.completions.create(**kwargs)

        if stream:
            return response
        content = self._extract_content(response)
        if not content or not content.strip():
            choice = response.choices[0] if response.choices else None
            finish_reason = getattr(choice, "finish_reason", None)
            msg = getattr(choice, "message", None)
            refusal = getattr(msg, "refusal", None)
            logger.warning(
                "OpenAI returned empty content. finish_reason=%s, refusal=%s, content_type=%s",
                finish_reason,
                refusal,
                type(getattr(msg, "content", None)).__name__ if msg else None,
            )
            raise ValueError("OpenAI returned empty content")
        return content.strip()
