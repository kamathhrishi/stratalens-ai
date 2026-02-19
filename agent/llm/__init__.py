#!/usr/bin/env python3
"""
Agent LLM layer: unified interface to switch between OpenAI, Cerebras, etc.

Usage:
    from agent.llm import get_llm, LLMClient

    llm = get_llm(config, openai_api_key=os.getenv("OPENAI_API_KEY"))
    answer = llm.complete([{"role": "user", "content": "What is revenue?"}])

Configuration (env or config dict):
    RAG_LLM_PROVIDER: "openai" | "cerebras" | "auto" (default: auto)
    RAG_OPENAI_MODEL: default OpenAI model
    RAG_CEREBRAS_MODEL: default Cerebras model
    OPENAI_API_KEY, CEREBRAS_API_KEY: API keys
"""

from .base import LLMClient
from .openai_client import OpenAILLMClient
from .cerebras_client import CerebrasLLMClient
from .router import RouterLLMClient
from .factory import get_llm

__all__ = [
    "LLMClient",
    "OpenAILLMClient",
    "CerebrasLLMClient",
    "RouterLLMClient",
    "get_llm",
]
