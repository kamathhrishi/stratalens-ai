#!/usr/bin/env python3
"""
Reasoning Planner - Combined Question Analysis + Research Reasoning

This module replaces the separate QuestionAnalyzer and Reasoning stages with
a single LLM call that both analyzes the question AND explains the research approach.

Key improvements:
- Single LLM call instead of two (faster)
- Reasoning drives the analysis (more coherent)
- Preserves exact temporal references (fixes "last 3 quarters" bug)
- Outputs both natural language reasoning + structured metadata
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from agent.llm import get_llm, LLMClient

try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


@dataclass
class ReasoningResult:
    """
    Combined result from reasoning + analysis.

    Contains both human-readable reasoning and structured metadata
    for downstream processing.
    """
    # Natural language reasoning (for user transparency)
    reasoning: str

    # Structured metadata (for search planning)
    tickers: list[str]
    time_refs: list[str]  # Exact temporal phrases from question
    topic: str
    question_type: str
    data_sources: list[str]
    answer_mode: str

    # Validation
    is_valid: bool
    validation_message: str = ""

    # Metadata
    confidence: float = 0.95
    processing_time: float = 0.0


class ReasoningPlanner:
    """
    Combined question analysis and research reasoning.

    Single LLM call that:
    1. Analyzes the question (extracts entities, intent)
    2. Explains the research approach (reasoning)
    3. Outputs structured metadata for search planning
    """

    def __init__(self, config, database_manager=None, conversation_memory=None, llm: Optional[LLMClient] = None):
        """Initialize the reasoning planner. Uses shared LLM from config if llm not provided."""
        self.config = config
        self.database_manager = database_manager
        self.conversation_memory = conversation_memory
        self.llm = llm if llm is not None else get_llm(config)
        logger.info(f"✅ ReasoningPlanner initialized with LLM ({self.llm.provider_name})")

    async def create_reasoning_plan(
        self,
        question: str,
        conversation_id: Optional[str] = None
    ) -> ReasoningResult:
        """
        Analyze question and create research reasoning in a single LLM call.

        Args:
            question: User's raw question
            conversation_id: Optional conversation ID for context

        Returns:
            ReasoningResult with reasoning + structured metadata
        """
        start_time = time.time()
        rag_logger.info(f"🧠 Starting reasoning & analysis for: '{question}'")

        # Get conversation context if available
        conversation_context = ""
        if conversation_id and self.conversation_memory:
            try:
                conversation_context = await self.conversation_memory.format_context(conversation_id)
                if conversation_context:
                    rag_logger.info(f"📜 Using conversation history ({len(conversation_context)} chars)")
                    conversation_context = self._format_conversation_context(conversation_context)
            except Exception as e:
                rag_logger.warning(f"⚠️ Failed to load conversation context: {e}")

        # Get available quarters context
        quarter_context = self._get_quarter_context()

        # Build the prompt
        prompt = self._build_reasoning_prompt(question, quarter_context, conversation_context)

        # Call LLM with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rag_logger.info(f"🤖 Calling LLM for reasoning (attempt {attempt + 1}/{max_retries})")

                # GPT-5-nano / reasoning models: use single user message (no system role) for reliable content
                system_instruction = "You are a financial research assistant. Analyze the question, explain your research approach, and output structured metadata. Respond with valid JSON only—no markdown, no extra text, no emojis.\n\n"
                if self.llm.provider_name == "OpenAI":
                    messages = [{"role": "user", "content": system_instruction + prompt}]
                else:
                    messages = [
                        {"role": "system", "content": system_instruction.strip()},
                        {"role": "user", "content": prompt}
                    ]
                response_text = self.llm.complete(
                    messages,
                    temperature=0.3,
                    max_tokens=4000,  # gpt-5-nano uses many tokens for reasoning before output; 1k was too low (finish_reason=length)
                    stream=False,
                )
                response_text = response_text.strip()

                # Clean up response (remove markdown if present)
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                # ✅ CRITICAL FIX: Check for empty response after cleanup
                if not response_text:
                    rag_logger.warning(f"⚠️ Empty response after markdown cleanup (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return self._create_fallback_result(question, time.time() - start_time)
                    continue

                # Parse JSON
                result_json = json.loads(response_text)

                # Convert to ReasoningResult
                result = self._parse_reasoning_result(result_json, question)
                result.processing_time = time.time() - start_time

                rag_logger.info(f"✅ Reasoning completed in {result.processing_time:.3f}s")
                rag_logger.info(f"📊 Tickers: {result.tickers}, Time: {result.time_refs}, Mode: {result.answer_mode}")

                if LOGFIRE_AVAILABLE and logfire:
                    logfire.info(
                        "rag.reasoning.complete",
                        question=question,
                        tickers=result.tickers,
                        time_refs=result.time_refs,
                        answer_mode=result.answer_mode,
                        data_sources=result.data_sources,
                        processing_time_ms=int(result.processing_time * 1000)
                    )

                return result

            except json.JSONDecodeError as e:
                rag_logger.warning(f"⚠️ JSON parsing failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Return fallback
                    return self._create_fallback_result(question, time.time() - start_time)
                continue

            except Exception as e:
                rag_logger.error(f"❌ Reasoning failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return self._create_fallback_result(question, time.time() - start_time)
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

        # Should not reach here, but return fallback just in case
        return self._create_fallback_result(question, time.time() - start_time)

    def _build_reasoning_prompt(
        self,
        question: str,
        quarter_context: str,
        conversation_context: str
    ) -> str:
        """Build the combined reasoning + analysis prompt."""

        return f"""Analyze this financial question and create a research plan.

QUESTION: "{question}"

{quarter_context}

{conversation_context}

YOUR TASK:
1. Analyze what the user is asking
2. Explain your research approach (2-3 sentences)
3. Extract structured information for search planning

OUTPUT JSON FORMAT:
{{
  "reasoning": "Your research approach explanation (2-3 sentences). Be specific about which data sources, time periods, and what information you'll look for.",

  "tickers": ["TICKER1", "TICKER2"],
  "time_refs": ["PRESERVE EXACT temporal phrases from question"],
  "topic": "High-level topic (e.g., 'AI capital expenditures', 'revenue growth', 'risk factors')",
  "question_type": "specific_company|multiple_companies|financial_metrics|guidance|challenges|outlook|industry_analysis|executive_leadership|business_strategy|latest_news",
  "data_sources": ["earnings_transcripts", "10k", "news"],
  "answer_mode": "direct|standard|detailed",

  "is_valid": true,
  "validation_message": "",
  "confidence": 0.95
}}

CRITICAL INSTRUCTIONS:

**TEMPORAL REFERENCES - STANDARDIZED FORMAT:**
⚠️ Extract temporal references using EXACT standardized keywords:

**If question specifies time period:**
- "last 3 quarters" → time_refs: ["last 3 quarters"]  ✅
- "past 2 years" → time_refs: ["past 2 years"]  ✅
- "Q4 2024" → time_refs: ["Q4 2024"]  ✅
- "last quarter" → time_refs: ["last 1 quarters"]  ✅ (normalize to "last N quarters" format)
- **10-K / annual report for a specific year:** "10-K from 2020", "2020 10-K", "analyze 10k from 2019" → time_refs: ["2020"] or ["2019"] (the 4-digit year only). We have 10-K data from 2019 onward.
- **10-K multi-year or year range:** "performance from 2020 to 2024", "10-K 2020-2024", "compile ABNB 2020 to 2024 based on 10k" → time_refs: ["2020 to 2024"] (one phrase with start and end year so we can search each year's 10-K). Same for "2020-2024", "between 2020 and 2024".

**If question does NOT specify time period:**
- Use: time_refs: ["latest"]  ✅ (EXACTLY the word "latest", nothing else!)
- NOT "latest quarter" ❌
- NOT "most recent quarter" ❌
- NOT "recent" ❌
- ONLY: ["latest"] ✅

DO NOT add extra words! Use the exact standardized keywords!

**DATA SOURCES:**
- "earnings_transcripts" - Quarterly earnings calls (for guidance, commentary, Q&A)
- "10k" - Annual SEC filings (for balance sheets, risk factors, compensation, detailed financials)
- "news" - Latest news articles (for recent developments, announcements)

Choose based on:
- Capex, revenue, guidance, commentary → "earnings_transcripts"
- Balance sheet, debt, risks, compensation → "10k"
- Latest news, recent events → "news"

**ANSWER MODE:**
- "direct" - Simple single-metric lookups ("What was AAPL revenue?")
- "standard" - Moderate questions with some context ("Tell me about AAPL performance")
- "detailed" - Complex analytical questions requiring comprehensive research ("Analyze", "Comment on", "Explain", multi-company comparisons, financial statement analysis)

**VALIDATION:**
Mark is_valid=false for:
- Gibberish, greetings, non-finance questions
- Questions about private companies or data we don't have
- Too vague to answer

EXAMPLES:

QUESTION: "$META AI capex commentary in last 3 quarters"
OUTPUT: {{
  "reasoning": "The user is asking about META's AI-related capital expenditure commentary across the last 3 quarters. Capex guidance is typically discussed in quarterly earnings calls, so I'll search earnings transcripts for management's statements on AI infrastructure investments, datacenter buildouts, GPU purchases, and forward-looking capex guidance.",
  "tickers": ["META"],
  "time_refs": ["last 3 quarters"],
  "topic": "AI capital expenditures commentary",
  "question_type": "specific_company",
  "data_sources": ["earnings_transcripts"],
  "answer_mode": "standard",
  "is_valid": true,
  "validation_message": "",
  "confidence": 0.95
}}

QUESTION: "Comment on Oracle's balance sheet and debt usage"
OUTPUT: {{
  "reasoning": "The user wants a comprehensive analysis of Oracle's balance sheet and debt strategy. This requires detailed financial data from the 10-K filing, including total assets, liabilities, debt structure, maturity schedules, and management's discussion of capital allocation. I'll search the latest 10-K for balance sheet data, debt footnotes, and MD&A sections on capital structure.",
  "tickers": ["ORCL"],
  "time_refs": ["latest"],
  "topic": "balance sheet and debt usage analysis",
  "question_type": "specific_company",
  "data_sources": ["10k"],
  "answer_mode": "detailed",
  "is_valid": true,
  "validation_message": "",
  "confidence": 0.95
}}

QUESTION: "What is NVDA's competitive moat in AI chips?"
OUTPUT: {{
  "reasoning": "The user is asking about NVIDIA's competitive advantages in AI chips. This requires analyzing strategic differentiators like technology leadership, CUDA ecosystem, and market positioning typically discussed in earnings calls and 10-Ks. I'll search recent earnings transcripts and the latest 10-K for management commentary on competitive moat, R&D investments, and market share.",
  "tickers": ["NVDA"],
  "time_refs": ["latest"],
  "topic": "competitive moat in AI chips",
  "question_type": "specific_company",
  "data_sources": ["earnings_transcripts", "10k"],
  "answer_mode": "detailed",
  "is_valid": true,
  "validation_message": "",
  "confidence": 0.95
}}

QUESTION: "Compare AAPL and MSFT revenue growth over last 2 years"
OUTPUT: {{
  "reasoning": "The user wants to compare Apple and Microsoft's revenue growth trends over the last 2 years. I'll search earnings transcripts for both companies across the last 8 quarters to extract revenue figures, growth rates, and segment breakdowns for a comprehensive comparison.",
  "tickers": ["AAPL", "MSFT"],
  "time_refs": ["last 2 years"],
  "topic": "revenue growth comparison",
  "question_type": "multiple_companies",
  "data_sources": ["earnings_transcripts"],
  "answer_mode": "detailed",
  "is_valid": true,
  "validation_message": "",
  "confidence": 0.95
}}

QUESTION: "Analyze $ABNB 10k from 2020"
OUTPUT: {{
  "reasoning": "The user wants analysis of Airbnb's 10-K SEC filing for 2020. I'll search the FY2020 10-K filing for ABNB for business overview, risk factors, financials, and management discussion.",
  "tickers": ["ABNB"],
  "time_refs": ["2020"],
  "topic": "10-K analysis",
  "question_type": "specific_company",
  "data_sources": ["10k"],
  "answer_mode": "detailed",
  "is_valid": true,
  "validation_message": "",
  "confidence": 0.95
}}

QUESTION: "Compile $ABNB performance from 2020 to 2024 based on its 10k"
OUTPUT: {{
  "reasoning": "The user wants a compilation of Airbnb's performance from 2020 to 2024 using 10-K filings. I'll search each year's 10-K (FY2020 through FY2024) for revenue, income, key metrics, and management discussion to present a multi-year view.",
  "tickers": ["ABNB"],
  "time_refs": ["2020 to 2024"],
  "topic": "performance compilation",
  "question_type": "specific_company",
  "data_sources": ["10k"],
  "answer_mode": "detailed",
  "is_valid": true,
  "validation_message": "",
  "confidence": 0.95
}}

Now analyze the question above and output valid JSON only.
"""

    def _get_quarter_context(self) -> str:
        """Get available quarters context for the LLM."""
        if self.config:
            return self.config.get_quarter_context_for_llm()
        return "Limited quarterly data available."

    def _format_conversation_context(self, context: str) -> str:
        """Format conversation history for the prompt."""
        return f"""
═══════════════════════════════════════════════════════════════════════════════
CONVERSATION HISTORY
═══════════════════════════════════════════════════════════════════════════════
The current question might refer to prior context (pronouns, "those companies", etc.)
or might be asking about something completely different. Decide based on the question.

{context}

═══════════════════════════════════════════════════════════════════════════════
END CONVERSATION HISTORY
═══════════════════════════════════════════════════════════════════════════════
"""

    def _parse_reasoning_result(self, result_json: Dict[str, Any], question: str) -> ReasoningResult:
        """Parse LLM JSON response into ReasoningResult."""
        return ReasoningResult(
            reasoning=result_json.get("reasoning", ""),
            tickers=result_json.get("tickers", []),
            time_refs=result_json.get("time_refs", []),
            topic=result_json.get("topic", ""),
            question_type=result_json.get("question_type", "specific_company"),
            data_sources=result_json.get("data_sources", ["earnings_transcripts"]),
            answer_mode=result_json.get("answer_mode", "standard"),
            is_valid=result_json.get("is_valid", True),
            validation_message=result_json.get("validation_message", ""),
            confidence=result_json.get("confidence", 0.95)
        )

    def _create_fallback_result(self, question: str, processing_time: float) -> ReasoningResult:
        """Create a fallback result when LLM fails."""
        rag_logger.warning("⚠️ Using fallback reasoning result")

        # Try to extract ticker from question
        import re
        ticker_match = re.search(r'\$([A-Z]{1,5})\b', question)
        tickers = [ticker_match.group(1)] if ticker_match else []

        return ReasoningResult(
            reasoning=f"Analyzing the question about {', '.join(tickers) if tickers else 'financial data'}. Will search available data sources for relevant information.",
            tickers=tickers,
            time_refs=["latest"],
            topic="general financial question",
            question_type="specific_company" if tickers else "general_market",
            data_sources=["earnings_transcripts"],
            answer_mode="standard",
            is_valid=True,
            validation_message="",
            confidence=0.5,
            processing_time=processing_time
        )
