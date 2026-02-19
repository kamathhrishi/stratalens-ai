#!/usr/bin/env python3
"""Cerebras-backed LLM client for the agent."""

import logging
from typing import List, Dict, Any, Optional, Union

from .base import LLMClient, StreamType

logger = logging.getLogger(__name__)


class CerebrasLLMClient(LLMClient):
    """LLM client using Cerebras Cloud SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "qwen-3-235b-a22b-instruct-2507",
        default_temperature: float = 0.1,
        default_max_tokens: int = 16000,
    ):
        self._api_key = api_key
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("Cerebras API key not set; set CEREBRAS_API_KEY or pass api_key")
            try:
                from cerebras.cloud.sdk import Cerebras
                self._client = Cerebras(api_key=self._api_key)
                logger.info(f"✅ Cerebras LLM client initialized (model: {self._default_model})")
            except ImportError:
                raise RuntimeError("Cerebras SDK not installed. Run: pip install cerebras-cloud-sdk")
        return self._client

    @property
    def provider_name(self) -> str:
        return "Cerebras"

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, StreamType]:
        client = self._get_client()
        model = model or self._default_model
        temperature = temperature if temperature is not None else self._default_temperature
        max_tokens = max_tokens or self._default_max_tokens

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            return completion
        if not completion or not getattr(completion, "choices", None) or len(completion.choices) == 0:
            raise ValueError("Cerebras returned invalid or empty response")
        content = completion.choices[0].message.content
        if content is None or (isinstance(content, str) and not content.strip()):
            raise ValueError("Cerebras returned empty content")
        return content.strip()
