#!/usr/bin/env python3
"""
Factory to create the configured LLM client for the agent.

Usage:
    from agent.llm import get_llm

    config = ...  # RAG Config or dict with get()
    llm = get_llm(config, openai_api_key=os.getenv("OPENAI_API_KEY"))
    text = llm.complete([{"role": "user", "content": "Hello"}])

To switch provider: set env RAG_LLM_PROVIDER=openai | cerebras | auto (default: cerebras).
To override models: set in config or env (e.g. openai_model, cerebras_model, evaluation_model).
"""

import os
import logging
from typing import Optional, Any

from .base import LLMClient
from .openai_client import OpenAILLMClient
from .cerebras_client import CerebrasLLMClient
from .router import RouterLLMClient

logger = logging.getLogger(__name__)


def _config_get(config: Any, key: str, default: Optional[str] = None) -> Optional[str]:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def get_llm(
    config: Any = None,
    *,
    openai_api_key: Optional[str] = None,
    cerebras_api_key: Optional[str] = None,
    provider: Optional[str] = None,
    role: Optional[str] = None,
) -> LLMClient:
    """
    Build the LLM client from config and env.

    Args:
        config: RAG Config (or dict) with keys: llm_provider, openai_model, cerebras_model,
                evaluation_model, openai_temperature, cerebras_temperature, etc.
        openai_api_key: OpenAI API key (default: OPENAI_API_KEY env).
        cerebras_api_key: Cerebras API key (default: CEREBRAS_API_KEY env).
        provider: Override provider for this call (openai | cerebras | auto). Otherwise uses config/env.
        role: Optional role hint for model selection: "default" | "planning" | "evaluation".
              Some configs use evaluation_model for evaluation role.

    Returns:
        LLMClient to use for complete().
    """
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    cerebras_api_key = cerebras_api_key or os.getenv("CEREBRAS_API_KEY")

    resolved = provider or _config_get(config, "llm_provider") or os.getenv("RAG_LLM_PROVIDER", "cerebras")
    resolved = (resolved or "auto").strip().lower()

    # Model overrides by role
    openai_model = _config_get(config, "openai_model") or os.getenv("RAG_OPENAI_MODEL", "gpt-5-nano-2025-08-07")
    cerebras_model = _config_get(config, "cerebras_model") or os.getenv("RAG_CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507")
    if role == "evaluation":
        openai_model = _config_get(config, "evaluation_model") or openai_model
        cerebras_model = _config_get(config, "evaluation_model") or cerebras_model

    openai_temp = 1
    cerebras_temp = 0.1
    openai_max = 8000   # max_completion_tokens; config may override via openai_max_tokens (from RAG_OPENAI_MAX_TOKENS)
    cerebras_max = 16000
    if config and hasattr(config, "get"):
        openai_temp = config.get("openai_temperature", openai_temp)
        cerebras_temp = config.get("cerebras_temperature", cerebras_temp)
        openai_max = config.get("openai_max_tokens", openai_max)
        cerebras_max = config.get("cerebras_max_tokens", cerebras_max)

    def make_openai() -> OpenAILLMClient:
        return OpenAILLMClient(
            api_key=openai_api_key,
            default_model=openai_model,
            default_temperature=openai_temp,
            default_max_tokens=openai_max,
        )

    def make_cerebras() -> CerebrasLLMClient:
        return CerebrasLLMClient(
            api_key=cerebras_api_key,
            default_model=cerebras_model,
            default_temperature=cerebras_temp,
            default_max_tokens=cerebras_max,
        )

    if resolved == "openai":
        client = make_openai()
        if not client.is_available():
            logger.warning("RAG_LLM_PROVIDER=openai but OpenAI not available (missing key?)")
        return client

    if resolved == "cerebras":
        client = make_cerebras()
        if not client.is_available():
            logger.warning("RAG_LLM_PROVIDER=cerebras but Cerebras not available (missing key or SDK?)")
        return client

    # auto: Cerebras first, then OpenAI
    primary = make_cerebras()
    fallback = make_openai() if openai_api_key else None
    if primary.is_available():
        logger.info("🤖 LLM: using Cerebras (primary) with optional OpenAI fallback")
        return RouterLLMClient(primary, fallback)
    if fallback and fallback.is_available():
        logger.info("🤖 LLM: using OpenAI only (Cerebras not available)")
        return fallback
    # Prefer returning primary so error message is about Cerebras if both missing
    if fallback:
        return RouterLLMClient(primary, fallback)
    return primary
