#!/usr/bin/env python3
"""
Router LLM client: picks a provider by config (e.g. auto = Cerebras if available, else OpenAI).
"""

import logging
from typing import List, Dict, Optional, Union

from .base import LLMClient, StreamType

logger = logging.getLogger(__name__)


class RouterLLMClient(LLMClient):
    """
    Routes each call to a primary or fallback client.

    Used when RAG_LLM_PROVIDER=auto: try Cerebras first, fall back to OpenAI.
    You can also use it with a single primary client (no fallback).
    """

    def __init__(self, primary: LLMClient, fallback: Optional[LLMClient] = None):
        self._primary = primary
        self._fallback = fallback

    @property
    def provider_name(self) -> str:
        return f"Router({self._primary.provider_name}" + (
            f"|{self._fallback.provider_name}" if self._fallback else ""
        ) + ")"

    def is_available(self) -> bool:
        return self._primary.is_available() or (self._fallback and self._fallback.is_available())

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[str, StreamType]:
        for client in (self._primary, self._fallback):
            if client is None:
                continue
            if not client.is_available():
                continue
            try:
                return client.complete(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
            except Exception as e:
                logger.warning(
                    "LLM call failed with %s (%s), trying fallback: %s",
                    client.provider_name,
                    type(e).__name__,
                    str(e)[:80],
                )
                if self._fallback is None or client is self._fallback:
                    raise
        raise RuntimeError("No LLM provider available (primary and fallback failed or unavailable)")
