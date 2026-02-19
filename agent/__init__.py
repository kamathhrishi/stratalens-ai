#!/usr/bin/env python3
"""
Agent - Unified agent for financial Q&A

This module provides a unified agent that handles:
- RAG-based question answering (earnings transcripts, 10-K filings, news)
- Stock screening queries (fundamental data filtering)
- Iterative self-improvement with quality evaluation

Simplified architecture with no circular dependencies.
"""

from .rag.rag_agent import RAGAgent

# Public API: RAGAgent is the main class, aliased as Agent for backward compat
Agent = RAGAgent
AgentSystem = RAGAgent

def create_agent():
    return RAGAgent()

# Keep prompts
from . import prompts

__all__ = ['Agent', 'RAGAgent', 'AgentSystem', 'create_agent', 'prompts']
