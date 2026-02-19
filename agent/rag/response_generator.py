#!/usr/bin/env python3
"""
Response Generator for the RAG system.

This module handles response generation, evaluation, and follow-up question
generation for the RAG system.
"""

import json
import logging
import time
import openai
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Import local modules
from .config import Config
from .llm_utils import LLMError, is_retryable_error, get_user_friendly_message, format_error_for_user
from agent.llm import get_llm, LLMClient
from agent.prompts import (
    QUARTER_SYNTHESIS_SYSTEM_PROMPT,
    get_quarter_synthesis_prompt,
    QUESTION_PLANNING_SYSTEM_PROMPT,
    get_question_planning_prompt
)

# Evaluator system prompt: strict, JSON-only, no extra text
EVALUATION_SYSTEM_PROMPT = (
    "You are a strict financial analyst evaluator. Rate answer quality critically; only comprehensive, "
    "well-sourced answers with specific metrics deserve high scores. Respond with valid JSON only—no "
    "markdown, no explanation, no emojis."
)

# Import Logfire for observability (optional)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


# ═════════════════════════════════════════════════════════════════════
# STAGE 1: INITIALIZATION & SETUP
# ═════════════════════════════════════════════════════════════════════

class ResponseGenerator:
    """Handles response generation and evaluation for the RAG system."""

    def __init__(self, config: Config, openai_api_key: Optional[str] = None, llm: Optional[LLMClient] = None):
        """Initialize the response generator.

        Args:
            config: RAG config (model names, temperatures, etc.).
            openai_api_key: OpenAI API key (default: OPENAI_API_KEY env).
            llm: Optional pre-built LLM client. If None, one is created from config and env
                 (RAG_LLM_PROVIDER=openai|cerebras|auto).
        """
        self.config = config
        self.openai_api_key = openai_api_key
        self.llm = llm if llm is not None else get_llm(config, openai_api_key=openai_api_key)

        # Legacy compatibility: some code may check these
        self.openai_available = bool(openai_api_key or __import__("os").getenv("OPENAI_API_KEY"))
        self.cerebras_available = self.llm.is_available()  # Router may use Cerebras
        self.cerebras_client = None  # No longer used; LLM layer handles provider
        self._client = None
        self.groq_client = None
        self.groq_available = False

        logger.info(f"🤖 ResponseGenerator: LLM provider via {self.llm.provider_name}")

        # Cache system prompt templates (built once, reused with substitution)
        self._system_prompt_cache = {}
        self._init_system_prompt_templates()

        logger.info("ResponseGenerator initialized successfully")

    def _init_system_prompt_templates(self):
        """Initialize cached system prompt templates for reuse."""
        # Shared core: citations, format, limitations (used by all templates)
        _core = (
            "**CRITICAL - ANTI-HALLUCINATION RULES:** "
            "1. ONLY use information explicitly stated in the provided {sources}. "
            "2. DO NOT invent, guess, or use external knowledge — not even well-known public facts. "
            "3. If information is not in the sources, you MUST say: \"I don't have information about [topic] in the available data.\" "
            "4. NEVER cite sources that don't exist in the provided context. Every [1], [2], [N1], [10K-1] must be real. "
            "5. NUMBERS AND FIGURES: Every specific number (dollar amount, percentage, growth rate, headcount, etc.) you write MUST appear verbatim in the source you cite for it. If you cite [3] for \"revenue was $280B\", that exact figure must be in source [3]'s text. Do NOT use numbers from your training knowledge. "
            "6. CITATION ACCURACY: Only cite a source for a claim if that source's text directly supports that specific claim. Do not cite a source about acquisitions for a revenue figure. "
            "7. If unsure whether a fact is in the sources, omit it or say it is not available. "
            "\n"
            "**Role:** Answer only from the provided {sources}. No emojis. No knowledge beyond the data. "
            "**Citations - CRITICAL FORMAT:** Each source has a marker like \"SOURCE [13]\", \"SOURCE [55]\", etc. You MUST copy the EXACT marker WITH BRACKETS into your answer: [13], [55], [65]. "
            "NEVER write just the number without brackets (55 is WRONG, [55] is CORRECT). "
            "DO NOT create your own citation labels like [Q2 2023], [FY2024], or [2025]. Use ONLY the exact bracketed markers from the sources. "
            "Place the marker immediately after the claim (e.g., \"Revenue grew 45% [13].\"). ONLY cite sources that actually exist in the provided context. "
            "When multiple source types are provided, cite from ALL that contain relevant information; do not favor one. "
            "**Attribution:** Use human-readable attribution like {attribution} with the citation marker (e.g., \"According to Q1 2025 earnings [1]...\"). "
            "**Format:** Markdown with **bold**, bullets, and tables for multi-figure comparisons. "
            "Use human-friendly periods (Q1 2025, FY 2024) in your text, but use numeric markers for citations. "
            "**Limitations:** Present what you found first. If information is incomplete, say so after your answer (e.g. 'Note: A complete list is not in the filings...'). "
            "If limited, you may end with 'Want me to search thoroughly?' Only if there is no relevant information say \"I don't have information about [topic] in the available data.\" "
            "Never lead with limitations. "
            "**Style:** Do not label your answer ('here is a report', 'this is a summary'). Answer naturally but factually. "
        )
        # Detail level: elaborate (default for detailed mode)
        _detail = (
            "IMPORTANT: Provide ELABORATE and COMPREHENSIVE responses with MAXIMUM DETAIL. ALWAYS MENTION ALL "
            "FINANCIAL FIGURES AND PROJECTIONS PRESENT in the provided data when relevant—exact numbers, percentages, "
            "dollar amounts, growth rates, margins, guidance ranges. Never omit important figures. Include full "
            "context (YoY, sequential, guidance). Be thorough and detailed in your analysis. "
        )

        # Base template for single-ticker
        self._system_prompt_cache['base'] = (
            "You are a financial analyst assistant. "
            + _core.format(sources="{sources}", attribution="{attribution}")
            + _detail
        )

        # Multi-ticker with news
        _multi_news = (
            "You analyze multiple companies using earnings transcripts, 10-K filings, and news. "
            "Reference each company by name. Use tables when comparing across companies or periods. "
            "Acknowledge all source types used (calls, 10-K, news [N1], [N2]). "
        )
        self._system_prompt_cache['multi_ticker_news'] = (
            "You are a financial analyst assistant for evidence-based multi-company analysis. "
            + _multi_news
            + _core.format(sources="earnings transcripts, 10-K filings, and news", attribution="\"According to [Company]'s Q1 2025 earnings call...\" or \"Per [Company]'s FY 2024 10-K...\" and [N1], [N2] for news")
            + "IMPORTANT: Provide ELABORATE and COMPREHENSIVE responses with MAXIMUM DETAIL. ALWAYS MENTION ALL "
            "FINANCIAL FIGURES AND PROJECTIONS FOR EACH COMPANY when relevant—exact numbers, percentages, margins, "
            "guidance, cross-company comparisons. Never omit figures. Be thorough for every company. "
        )

        # Multi-ticker without news
        _multi = (
            "You analyze multiple companies using earnings transcripts and 10-K filings. "
            "Reference each company by name. Use tables when comparing across companies or periods. "
            "When both transcripts and 10-K are provided, cite from both when relevant. "
        )
        self._system_prompt_cache['multi_ticker'] = (
            "You are a financial analyst assistant for evidence-based multi-company analysis. "
            + _multi
            + _core.format(sources="earnings transcripts and 10-K filings", attribution="\"According to [Company]'s Q1 2025 earnings call...\" or \"Per [Company]'s FY 2024 10-K filing...\"")
            + "IMPORTANT: Provide ELABORATE and COMPREHENSIVE responses with MAXIMUM DETAIL. ALWAYS MENTION ALL "
            "FINANCIAL FIGURES AND PROJECTIONS FOR ANY COMPANY when relevant. Include exact numbers, margins, "
            "guidance, cross-company comparisons. If any company mentioned a number, include it. Be thorough and "
            "detailed for every company. "
        )

        logger.info("✅ System prompt templates cached")

    def _get_system_prompt(self, template_key: str, answer_mode: str = None, **kwargs) -> str:
        """Get a system prompt from cache with optional substitutions and mode adjustments."""
        template = self._system_prompt_cache.get(template_key, self._system_prompt_cache['base'])
        if kwargs:
            template = template.format(**kwargs)
        # Adjust verbosity based on answer_mode
        if answer_mode == "direct":
            template = template.replace(
                "IMPORTANT: Provide ELABORATE and COMPREHENSIVE responses with MAXIMUM DETAIL. ALWAYS MENTION ALL",
                "Provide CONCISE, DIRECT answers with key figure(s). Include the most important"
            ).replace(
                "Be thorough and detailed in your analysis. ",
                "Be concise and focused. "
            ).replace(
                "Be thorough and detailed for every company. ",
                "Be concise and focused. "
            ).replace(
                "Never omit figures. Be thorough for every company. ",
                "Be concise and focused. "
            )
        elif answer_mode == "standard":
            template = template.replace(
                "IMPORTANT: Provide ELABORATE and COMPREHENSIVE responses with MAXIMUM DETAIL. ALWAYS MENTION ALL",
                "Provide STRUCTURED, FOCUSED responses with the most relevant metrics. Include key"
            ).replace(
                "Be thorough and detailed in your analysis. ",
                "Stay focused on what matters most. "
            ).replace(
                "Be thorough and detailed for every company. ",
                "Stay focused on what matters most. "
            ).replace(
                "Never omit figures. Be thorough for every company. ",
                "Stay focused on what matters most. "
            )
        return template

    def _get_iteration_stance(self, answer_mode: str = None) -> str:
        """Get the iteration stance text based on answer mode."""
        _mode = answer_mode or "detailed"
        if _mode == "direct":
            return (
                "**DEFAULT STANCE: DO NOT ITERATE unless key information is clearly missing.**\n\n"
                "A 2-4 sentence answer with the correct number(s) is SUFFICIENT. "
                "Only iterate if the specific number or fact requested is not in the answer."
            )
        elif _mode == "standard":
            return (
                "**DEFAULT STANCE: ITERATE ONLY IF KEY INFORMATION IS MISSING.**\n\n"
                "A focused answer with key metrics and a brief summary is SUFFICIENT. "
                "Iterate if important metrics or context are clearly absent, but do not over-elaborate."
            )
        else:  # detailed
            return (
                "**DEFAULT STANCE: ITERATE TO USE ALL AVAILABLE ITERATIONS**\n\n"
                "The agent has multiple iterations available to build a comprehensive answer. "
                "Use them ALL unless the answer is near-PERFECT."
            )

    def _build_citation_instructions(self, has_news: bool = False, has_10k: bool = False) -> str:
        """Build unified citation instructions based on available data sources.

        Consolidates news and 10-K citation templates into a single reusable method.
        """
        if not has_news and not has_10k:
            return ""

        # Build source descriptions dynamically
        sources = []
        markers = []

        if has_news:
            sources.append("news sources ([N1], [N2], etc.) for current context and recent developments")
            markers.append("[N1], [N2] for news")

        if has_10k:
            sources.append("10-K filings ([10K-1], [10K-2], etc.) for annual financials, risk factors, and audited data")
            markers.append("[10K-1], [10K-2] for 10-K filings")

        sources_list = "\n   - ".join(sources)
        markers_text = " and ".join(markers)

        return f"""
4. **Additional Data Sources Available**: You have access to earnings transcripts plus:
   - {sources_list}
   - Earnings transcripts provide quarterly updates, management commentary, and Q&A discussions
   Use whichever sources best answer the question. When multiple are relevant, integrate them naturally.
5. **Citation Markers**: Use {markers_text}. Attribute clearly (e.g., "According to the FY2024 10-K filing ([10K-1])").
6. **Source Attribution**: Reflect the sources you actually used. Mention all sources consulted."""

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 1.5: QUESTION PLANNING/REASONING
    # ═════════════════════════════════════════════════════════════════════

    async def plan_question_approach(self, question: str, question_analysis: Dict[str, Any], available_quarters: list = None, conversation_context: str = "") -> str:
        """
        Generate a freeform reasoning statement about how to approach the question.

        This is the REASONING step that happens before the RAG loop.
        Returns a natural, verbose statement like:
        "The user is asking about Apple's AI strategy, so I need to find..."

        Args:
            question: Original user question
            question_analysis: Analysis from question analyzer
            available_quarters: List of quarters available in database
            conversation_context: Optional formatted recent conversation (for follow-up questions)

        Returns:
            str: Freeform reasoning statement explaining the approach
        """
        plan_start = time.time()
        rag_logger.info(f"🧠 Starting question planning/reasoning...")

        # Get available quarters from config if not provided
        if available_quarters is None:
            available_quarters = self.config.get('available_quarters', [])

        # Log planning start to Logfire
        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "llm.planning.start",
                question=question,
                tickers=question_analysis.get('extracted_tickers', []),
                data_source=question_analysis.get('data_source', 'earnings_transcripts'),
                available_quarters_count=len(available_quarters) if available_quarters else 0
            )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get the planning prompt with data context and optional conversation history
                planning_prompt = get_question_planning_prompt(question, question_analysis, available_quarters, conversation_context=conversation_context or None)

                rag_logger.info(f"🤖 ===== QUESTION PLANNING LLM CALL ===== (attempt {attempt + 1}/{max_retries})")

                rag_logger.info(f"🔍 Using LLM for planning ({self.llm.provider_name})")
                start_time = time.time()
                reasoning = self.llm.complete(
                    [
                        {"role": "system", "content": QUESTION_PLANNING_SYSTEM_PROMPT},
                        {"role": "user", "content": planning_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    stream=False,
                )
                call_time = time.time() - start_time
                rag_logger.info(f"✅ Planning LLM call completed in {call_time:.3f}s")

                # reasoning is already a string from complete()
                reasoning = reasoning.strip()

                # Clean up any quotes if the model wrapped it
                if reasoning.startswith('"') and reasoning.endswith('"'):
                    reasoning = reasoning[1:-1]

                plan_time = time.time() - plan_start
                rag_logger.info(f"🧠 Planning completed in {plan_time:.3f}s")
                rag_logger.info(f"📋 Reasoning: {reasoning[:200]}...")

                # Log to Logfire
                if LOGFIRE_AVAILABLE and logfire:
                    logfire.info(
                        "llm.planning.complete",
                        reasoning=reasoning,
                        plan_time_ms=int(plan_time * 1000)
                    )

                return reasoning

            except Exception as e:
                rag_logger.error(f"❌ Planning error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Return a basic reasoning statement on failure
                    tickers = question_analysis.get('extracted_tickers', [])
                    ticker_text = f" for {', '.join(tickers)}" if tickers else ""
                    return f"The user is asking about{ticker_text}: {question}. I need to search the available financial data to find relevant information to answer this question."

        # Fallback (should not reach here)
        return f"Analyzing the question: {question}"

    @property
    def client(self):
        """Legacy: lazy OpenAI client for code that still uses it (e.g. rag_agent follow-up). Prefer using self.llm.complete()."""
        if self._client is None and self.openai_api_key:
            self._client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info("✅ OpenAI client initialized (lazy)")
        return self._client

    def _llm_complete_with_retry(self, messages: List[Dict], model: Optional[str] = None,
                                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                                 stream: bool = False, max_retries: int = 3):
        """Call LLM with retries for empty responses. Returns content str or stream object."""
        for attempt in range(max_retries):
            try:
                result = self.llm.complete(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
                if stream:
                    return result
                if result and isinstance(result, str) and result.strip():
                    return result
                rag_logger.warning(f"   ⚠️ Empty response on attempt {attempt + 1}/{max_retries}")
            except Exception as e:
                if attempt < max_retries - 1:
                    rag_logger.warning(f"   ⚠️ API call failed on attempt {attempt + 1}: {e}")
                    continue
                raise
        raise Exception(f"API returned empty response after {max_retries} attempts")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: SINGLE-TICKER RESPONSE GENERATION
    # ═════════════════════════════════════════════════════════════════════

    def generate_openai_response(self, question: str, context_chunks: List[str], chunk_objects: List[Dict[str, Any]] = None, return_details: bool = False, ticker: str = None, year: int = None, quarter: int = None, stream_callback=None, news_context: str = None, ten_k_context: str = None, previous_answer: str = None, conversation_context: str = None, retry_callback=None, answer_mode: str = None) -> str:
        """Generate response using OpenAI API based only on retrieved chunks with citations.

        If multiple quarters are detected, automatically uses parallel quarter processing
        for better structured responses.
        conversation_context: Optional formatted string of recent conversation (last N exchanges) for stateful follow-up questions.
        """
        generation_start = time.time()
        rag_logger.info(f"🤖 Starting OpenAI response generation")
        rag_logger.info(f"📊 Input parameters: question_length={len(question)}, chunks={len(context_chunks)}, chunk_objects={len(chunk_objects) if chunk_objects else 0}")

        # Log to Logfire with question for tracing
        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "llm.generation.start",
                provider="openai",
                ticker=ticker,
                question=question,
                chunks_count=len(context_chunks),
                has_news_context=news_context is not None,
                has_10k_context=ten_k_context is not None,
                has_previous_answer=previous_answer is not None
            )
        
        if not self.llm.is_available():
            rag_logger.error(f"❌ LLM not available")
            raise Exception("LLM not available for response generation")
        
        # Check if we have multiple quarters - if so, use parallel processing
        if chunk_objects:
            quarters_mentioned = set()
            for chunk_obj in chunk_objects:
                if chunk_obj.get('year') and chunk_obj.get('quarter'):
                    quarters_mentioned.add(f"{chunk_obj['year']}_q{chunk_obj['quarter']}")
            
            if len(quarters_mentioned) > 1:
                rag_logger.info(f"🚀 Detected {len(quarters_mentioned)} quarters: {sorted(quarters_mentioned)}")
                rag_logger.info(f"⚡ Using parallel quarter processing for better structured response")
                
                # Run async parallel processing in a way that works with existing event loops
                def run_async_in_thread():
                    """Run async code in a new thread with its own event loop."""
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(
                            self.generate_openai_response_parallel_quarters(
                                question, chunk_objects, ticker, stream_callback, retry_callback, conversation_context
                            )
                        )
                    finally:
                        new_loop.close()
                
                # Check if we're already in an async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context - use thread to avoid "loop already running" error
                    rag_logger.info(f"🔄 Detected running event loop, using thread executor for parallel processing")
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async_in_thread)
                        return future.result()
                except RuntimeError:
                    # No running loop - safe to run directly
                    rag_logger.info(f"🔄 No running event loop, executing parallel processing directly")
                    return run_async_in_thread()
        
        # Single quarter or no quarter metadata - proceed with regular processing
        rag_logger.info(f"📝 Using regular single-quarter processing")
        
        # Prepare context from retrieved chunks with citation markers and quarter/year metadata
        rag_logger.info(f"📝 Preparing context from {len(context_chunks)} chunks...")
        rag_logger.info(f"🎯 Using ALL {len(context_chunks)} selected chunks for answer generation")

        if not context_chunks and not news_context and not ten_k_context:
            # Explicitly show no data when there are no chunks
            context = "[NO DATA FOUND - No earnings transcripts, 10-K filings, or news articles were found for this query]"
            rag_logger.warning(f"⚠️ No context available - setting explicit no-data message")
        else:
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                # Use the ACTUAL citation marker from the chunk object, not the array index
                if chunk_objects and i < len(chunk_objects) and isinstance(chunk_objects[i], dict):
                    # Get the real citation marker (e.g., "13", "65", "11")
                    actual_citation = chunk_objects[i].get('citation', i+1)
                    citation_marker = f"[{actual_citation}]"

                    # Get metadata for this chunk
                    chunk_year = chunk_objects[i].get('year')
                    chunk_quarter = chunk_objects[i].get('quarter')
                    chunk_ticker = chunk_objects[i].get('ticker')
                    if chunk_year and chunk_quarter:
                        chunk_metadata = f" (Q{chunk_quarter} {chunk_year})"
                    elif chunk_ticker:
                        chunk_metadata = f" ({chunk_ticker})"
                    else:
                        chunk_metadata = ""
                else:
                    citation_marker = f"[{i+1}]"
                    chunk_metadata = ""

                # Format: "SOURCE [55] (Q2 2025): chunk text"
                # Make citation marker VERY prominent so LLM always uses it WITH brackets
                context_parts.append(f"SOURCE {citation_marker}{chunk_metadata}:\n{chunk}")
                rag_logger.info(f"📄 Source {citation_marker}: length={len(chunk)}, metadata={chunk_metadata}")

            context = "\n\n".join(context_parts)

        rag_logger.info(f"✅ Context prepared: total_length={len(context)}")
        rag_logger.info(f"📊 All {len(context_chunks)} chunks will be sent to LLM for answer generation")
        
        # Create focused prompt with source attribution
        company_name = ticker if ticker else "the company"
        
        # Determine if this is a multi-quarter query
        is_multi_quarter = False
        quarters_mentioned = set()
        if chunk_objects:
            for chunk_obj in chunk_objects:
                if chunk_obj.get('year') and chunk_obj.get('quarter'):
                    quarters_mentioned.add(f"{chunk_obj['year']}_q{chunk_obj['quarter']}")
            is_multi_quarter = len(quarters_mentioned) > 1
        
        # Get comprehensive quarter context for LLM
        quarters_info = self.config.get_quarter_context_for_llm()
        
        if is_multi_quarter:
            quarter_info = "financial data"
            quarters_list = ", ".join(sorted(quarters_mentioned))
            source_description = f"{company_name}'s financial data ({quarters_list})"
            available_quarters = self.config.get('available_quarters', [])
            data_limitation_note = f"Note: Our database contains financial data for {len(available_quarters)} quarters: {', '.join(available_quarters)}. This includes earnings transcripts (quarterly) and 10-K filings (annual). The analysis covers the available periods that match your request."
        else:
            if year and quarter:
                quarter_info = f"{year}_q{quarter}"
                source_description = f"{company_name}'s {quarter_info} financial data"
                base_note = f"the {quarter_info} financial data"
            else:
                quarter_info = "financial data"
                source_description = f"{company_name}'s financial data"
                base_note = "available financial data"
            available_quarters = self.config.get('available_quarters', [])
            data_limitation_note = (
                f"Note: This analysis is based on {base_note}. Our database contains data for "
                f"{len(available_quarters)} quarters, including earnings transcripts and 10-K filings. "
                f"Please determine the latest available period based on the available quarters listed above."
            )

        # Add news context if available
        news_section = ""
        if news_context:
            news_section = f"\n\n{news_context}\n\nNote: The above news sources provide recent developments and current information. Use them alongside other available data sources as appropriate for answering the question."

        # Add 10-K SEC filings context if available
        ten_k_section = ""
        if ten_k_context:
            ten_k_section = f"\n\n{ten_k_context}\n\nNote: The above 10-K SEC filing data provides comprehensive annual financial information, including balance sheets, income statements, cash flow statements, and detailed business disclosures. Use this data as appropriate for answering the question, combining with other available sources when relevant."

        # Add previous answer if available (for iterative improvement)
        previous_answer_section = ""
        if previous_answer:
            previous_answer_section = f"""

PREVIOUS ANSWER (build upon this):
{previous_answer}

IMPORTANT: You are improving the previous answer. Build upon it by:
1. Keeping all accurate information from the previous answer
2. Adding new details, numbers, quotes, or context from the additional information below
3. Correcting any inaccuracies if found
4. Expanding on areas that were incomplete
5. Maintaining the same structure and flow while enhancing it
Do NOT start from scratch - improve and expand the previous answer."""

        # Add conversation history for stateful follow-up questions (sliding window of last N exchanges)
        conversation_context_section = ""
        if conversation_context:
            conversation_context_section = f"""

CONVERSATION HISTORY (recent messages for context; the current question may refer to "those companies", "their", "above", etc.):
{conversation_context}

Answer the current question with full context from this conversation. Resolve pronouns and references using the history above.
If the user asks "would you say the same about [X]?" or "what about [company]?", apply the same conclusion or framework from your previous answer to the new company (e.g. whether the same risk applies, or the same metric trend), then answer for that company."""

        # Only include news-specific citation instructions when news_context is present
        news_sources_instruction = ""
        if news_context:
            news_sources_instruction = """
7. **News Sources Available**: You have access to both earnings transcripts and recent news sources. Use whichever sources best answer the question:
   - News sources ([N1], [N2], etc.) provide current context, recent developments, and market reactions
   - Earnings transcripts provide official company statements, financial metrics, and management commentary
   - When both are relevant, naturally integrate them; when only one is relevant, use only that source
   - Neither source type is more authoritative - choose based on what the question asks for
8. **When referencing news sources**: Use the citation markers (e.g., [N1], [N2]) and attribute them clearly (e.g., "According to recent reports ([N1])").
9. **Source Attribution**: Reflect the sources you actually used. If you used both, mention both; if you only used one type, that's fine too."""

        # Only include 10-K-specific citation instructions when ten_k_context is present
        ten_k_sources_instruction = ""
        if ten_k_context:
            ten_k_sources_instruction = """
7. **10-K SEC Filing Data Available**: You have access to 10-K annual report data alongside earnings call transcripts. Use whichever sources best answer the question:
   - 10-K filings ([10K-1], [10K-2], etc.) provide comprehensive annual financial statements, detailed business disclosures, risk factors, and audited financial data
   - Earnings transcripts provide quarterly updates, management commentary, and Q&A discussions
   - When both are relevant, naturally integrate them; when only one is relevant, use only that source
   - For balance sheet, comprehensive financial statements, or annual data questions, 10-K filings are typically most appropriate
8. **When referencing 10-K sources**: Use the EXACT citation markers from the context (e.g., [10K-1], [10K-2]) and attribute them clearly (e.g., "According to the FY2024 10-K filing ([10K-1])").
9. **Multi-year 10-K data**: When the context includes data from multiple fiscal years (e.g. FY2020, FY2021, FY2022), structure your answer by year where appropriate (e.g. "In FY2020 ... In FY2021 ...") and cite the relevant [10K-N] for each period.
10. **Source Attribution**: Reflect the sources you actually used. If you used both, mention both; if you only used one type, that's fine too."""

        # Determine what data sources are available for prompt text
        available_sources = []
        if news_context:
            available_sources.append("news sources")
        if ten_k_context:
            available_sources.append("10-K SEC filings")
        if available_sources:
            data_sources_text = f" {', '.join(available_sources)} are available - use whichever sources are most relevant for answering the question."
        else:
            data_sources_text = " Do not use any external knowledge."

        # When there is no transcript/10-K/news context, we do not auto-search news; we ask the LLM
        # to state that no data was found and to nudge the user: "Would you like me to search the news instead?"
        no_structured_context = (not context_chunks) and not news_context and not ten_k_context

        # Mode-aware prompt construction
        _mode = answer_mode or "detailed"
        if no_structured_context:
            no_context_note = '''

⚠️ CRITICAL: NO DATA AVAILABLE - SPECIAL INSTRUCTIONS:
- NO source documents were found for this query
- You MUST NOT generate any citations like [1], [2], [N1], [10K-1] etc.
- Do NOT invent or hallucinate any data, facts, or figures
- Respond with: "I don't have information about [topic] in the available earnings transcripts or SEC filings."
- Then ask: "Would you like me to search recent news instead?"
- Keep the response to 2-3 sentences maximum
'''
        else:
            no_context_note = ''

        if _mode == "direct":
            prompt = f"""CRITICAL RULES - READ FIRST:
1. ONLY use information from the "Source information" section below
2. NEVER cite sources that don't exist in the provided data
3. If the answer isn't in the provided sources, respond EXACTLY: "I don't have information about [topic] in the available data."
4. DO NOT hallucinate, guess, invent, or use external knowledge
5. CITATION FORMAT IS CRITICAL:
   - Each source has a marker like "SOURCE [13]", "SOURCE [55]", "SOURCE [65]"
   - You MUST use the EXACT marker WITH BRACKETS: [13], [55], [65]
   - NEVER write just the number without brackets: write [55] NOT 55
   - Copy the exact marker from the source line
   - DO NOT create your own labels like [Q2 2023] or [FY2024]
6. If unsure or data is missing, SAY SO - do not make up answers

---

Answer from {source_description} only. No emojis.{data_sources_text}{previous_answer_section}{conversation_context_section}

DATA CONTEXT: {quarters_info}
{data_limitation_note}
{news_section}{ten_k_section}

Source information:
{context}

Question: {question}

Instructions: Answer in 2-4 sentences with the specific number(s) or fact(s) requested. Use **bold** for key figures. CITATION EXAMPLES: If you see "SOURCE [55]" in the data, write [55] in your answer (WITH BRACKETS). If you see "SOURCE [13]", write [13]. ALWAYS include the brackets. DO NOT write 55 or 13 alone.{news_sources_instruction}{ten_k_sources_instruction} Attribute naturally (e.g. "According to {company_name}'s Q1 2025 earnings call [13]..."). Present findings first; if incomplete, add a short note and optionally "Want me to search thoroughly?" Never say "chunks"—refer to company documents. Use a markdown table for multiple numbers. Do not label the format ("here is a direct answer").
{no_context_note}

Answer in markdown. Be factual and grounded in provided sources only."""

        elif _mode == "standard":
            prompt = f"""CRITICAL RULES - READ FIRST:
1. ONLY use information from the "Source information" section below
2. NEVER cite sources that don't exist in the provided data
3. If the answer isn't in the provided sources, respond EXACTLY: "I don't have information about [topic] in the available data."
4. DO NOT hallucinate, guess, invent, or use external knowledge
5. CITATION FORMAT IS CRITICAL:
   - Each source has a marker like "SOURCE [13]", "SOURCE [55]", "SOURCE [65]"
   - You MUST use the EXACT marker WITH BRACKETS: [13], [55], [65]
   - NEVER write just the number without brackets: write [55] NOT 55
   - Copy the exact marker from the source line
   - DO NOT create your own labels like [Q2 2023] or [FY2024]
6. If unsure or data is missing, SAY SO - do not make up answers

---

Answer concisely from {source_description}.{data_sources_text}{previous_answer_section}{conversation_context_section}

DATA CONTEXT: {quarters_info}
{data_limitation_note}
{news_section}{ten_k_section}

Source information:
{context}

Question: {question}

Instructions: 3-5 sentences with key facts and figures. **Bold** key metrics. CITATION EXAMPLES: If you see "SOURCE [55]" in the data, write [55] in your answer (WITH BRACKETS). If you see "SOURCE [13]", write [13]. ALWAYS include the brackets. DO NOT write 55 or 13 alone.{news_sources_instruction}{ten_k_sources_instruction} Present findings first; if incomplete, note briefly and optionally nudge ("Want me to search thoroughly?"). Use tables only for 3+ data points. End with a short follow-up nudge ("Want more detail on X?").
{no_context_note}

Answer in markdown. Be conversational but factually grounded in provided sources only."""

        else:  # detailed
            prompt = f"""CRITICAL RULES - READ FIRST:
1. ONLY use information from the "Source information" section below
2. NEVER cite sources that don't exist in the provided data
3. If the answer isn't in the provided sources, respond EXACTLY: "I don't have information about [topic] in the available data."
4. DO NOT hallucinate, guess, invent, or use external knowledge
5. CITATION FORMAT IS CRITICAL:
   - Each source has a marker like "SOURCE [13]", "SOURCE [55]", "SOURCE [65]"
   - You MUST use the EXACT marker WITH BRACKETS: [13], [55], [65]
   - NEVER write just the number without brackets: write [55] NOT 55
   - Copy the exact marker from the source line
   - DO NOT create your own labels like [Q2 2023] or [FY2024]
   - EVERY fact must have a citation with brackets
6. If unsure or data is missing, SAY SO - do not make up answers
7. Check every fact, number, and claim against the provided sources before including it

---

Provide a comprehensive answer from {source_description}.{data_sources_text}{previous_answer_section}{conversation_context_section}

DATA CONTEXT: {quarters_info}
{data_limitation_note}
{news_section}{ten_k_section}

Source information:
{context}

Question: {question}

Instructions: Start with a 2-3 sentence summary; use ## headers by topic; use all relevant data and tables for comparisons. Be direct and conversational—avoid jargon and clichés ("Analyst Assessment", "Bottom Line"). CITATION EXAMPLES: If you see "SOURCE [55]" in the data, write [55] in your answer (WITH BRACKETS). If you see "SOURCE [13]", write [13]. ALWAYS include the brackets. DO NOT write 55 or 13 alone. EVERY fact needs a citation.{news_sources_instruction}{ten_k_sources_instruction} Reference {company_name} and period (e.g. "In Q1 2025 [13]..."). Present findings first; if incomplete, acknowledge briefly and optionally nudge. Never say "chunks." End with 2-3 takeaways and a follow-up ("Want to explore X?").
{no_context_note}

Answer in markdown. Thorough but approachable. Factually grounded in provided sources only."""

        rag_logger.info(f"📝 Prompt created: length={len(prompt)}")
        rag_logger.info(f"🔍 Prompt preview: {prompt[:200]}...")

        try:
            max_tokens = self.config.get("cerebras_max_tokens", self.config.get("openai_max_tokens", 8000))
            if self.llm.provider_name == "OpenAI" or self.config.get("llm_provider") == "openai":
                temperature = self.config.get("openai_temperature", 1)
            else:
                temperature = self.config.get("cerebras_temperature", 0.1)
            if answer_mode:
                from .config import AnswerMode, ANSWER_MODE_CONFIG
                try:
                    mode_config = ANSWER_MODE_CONFIG[AnswerMode(answer_mode)]
                    max_tokens = min(max_tokens, mode_config["max_tokens"])
                except (ValueError, KeyError):
                    pass
            rag_logger.info(f"🤖 Sending request via {self.llm.provider_name}")
            rag_logger.info(f"📊 Request parameters: max_tokens={max_tokens}, temperature={temperature}, answer_mode={_mode}")

            # Define system prompt using cached template with dynamic substitutions
            # Build list of available sources for the prompt
            available_sources = []
            if context_chunks:
                available_sources.append("financial filings")
            if ten_k_context:
                available_sources.append("10-K SEC filings")
            if news_context:
                available_sources.append("news sources")

            sources_text = " and ".join(available_sources) if available_sources else "the provided financial data"

            # Build source attribution instructions based on what's available
            attribution_instructions = []
            if context_chunks:
                attribution_instructions.append("\"According to [Company]'s Q1 2025 earnings call...\" or \"Per [Company]'s FY 2024 10-K filing...\"")
            if ten_k_context:
                attribution_instructions.append("[10K-1], [10K-2] markers for 10-K data")
            if news_context:
                attribution_instructions.append("[N1], [N2] for news sources")

            attribution_text = ", ".join(attribution_instructions) if attribution_instructions else "appropriate source citations"

            # Use cached template with substitutions
            system_prompt = self._get_system_prompt('base', answer_mode=_mode, sources=sources_text, attribution=attribution_text)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Check if streaming is requested
            if stream_callback:
                # Detailed LLM stage logging for streaming response generation
                rag_logger.info(f"🤖 ===== RESPONSE GENERATION LLM CALL (STREAMING) =====")
                rag_logger.info(f"🔍 Provider: {self.llm.provider_name}")
                rag_logger.info(f"📊 Max tokens: {max_tokens}")
                rag_logger.info(f"🌡️ Temperature: {temperature}")
                rag_logger.info(f"🎯 Ticker context: {ticker or 'General'}")
                rag_logger.info(f"📅 Quarter context: {f'{year} Q{quarter}' if year and quarter else 'Multiple/General'}")
                rag_logger.info(f"📋 Prompt length: {len(prompt)} characters")
                rag_logger.info(f"📊 Context chunks: {len(context_chunks)}")
                rag_logger.info(f"📋 Prompt preview: {prompt[:400]}...")

                rag_logger.info(f"🌊 Using streaming mode with callback")
                answer = ""
                total_tokens = 0
                response = None  # Initialize response for streaming path
                
                start_time = time.time()
                
                max_attempts = 3
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        stream = self.llm.complete(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                        )
                        break
                    except Exception as api_error:
                        error_text = str(api_error)
                        # Retry only on obvious transient/rate-limit style errors
                        is_transient = is_retryable_error(api_error)
                        if attempt >= max_attempts or not is_transient:
                            rag_logger.error(f"❌ Streaming API error (final): {api_error}")
                            # Raise user-friendly error instead of exposing internals
                            raise LLMError(
                                user_message=get_user_friendly_message(api_error),
                                technical_message=str(api_error),
                                retryable=is_transient
                            )
                        # Longer backoff: 5s, 10s, 15s instead of 2s, 4s, 6s
                        wait_seconds = 5 + (attempt * 5)
                        rag_logger.warning(
                            f"⚠️ Streaming API rate limit / queue error, retrying attempt {attempt}/{max_attempts} "
                            f"in {wait_seconds}s: {api_error}"
                        )
                        # Notify frontend about retry if callback provided
                        if retry_callback:
                            try:
                                retry_callback({
                                    'type': 'api_retry',
                                    'message': f'AI service busy, retrying in {wait_seconds}s...',
                                    'step': 'generation',
                                    'data': {
                                        'attempt': attempt,
                                        'max_attempts': max_attempts,
                                        'wait_seconds': wait_seconds,
                                        'provider': self.llm.provider_name
                                    }
                                })
                            except Exception as cb_error:
                                rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                        time.sleep(wait_seconds)

                chunk_count = 0
                content_chunk_count = 0
                try:
                    for chunk in stream:
                        chunk_count += 1
                        try:
                            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                content_chunk_count += 1
                                
                                # Log first few chunks with content for debugging
                                if content_chunk_count <= 3:
                                    rag_logger.info(f"🌊 RAG Single-ticker: Chunk {content_chunk_count} content: '{content[:100]}...' (length: {len(content)})")
                                
                                answer += content
                                # Call the streaming callback with the new content
                                try:
                                    stream_callback(content)
                                except Exception as e:
                                    rag_logger.warning(f"⚠️ Stream callback error: {e}")
                            
                            # Track token usage if available
                            if chunk.usage:
                                total_tokens = chunk.usage.total_tokens
                        except Exception as chunk_error:
                            rag_logger.error(f"❌ Error processing chunk {chunk_count}: {chunk_error}")
                            rag_logger.error(f"Chunk structure: {chunk}")
                except Exception as stream_error:
                    rag_logger.error(f"❌ Stream iteration error: {stream_error}")
                    if len(answer) == 0:
                        raise Exception(f"Stream failed with no content: {stream_error}")
                
                rag_logger.info(f"📊 Stream statistics: total_chunks={chunk_count}, content_chunks={content_chunk_count}, empty_chunks={chunk_count - content_chunk_count}")
                
                stream_time = time.time() - start_time
                rag_logger.info(f"✅ ===== RESPONSE GENERATION STREAMING COMPLETED ===== (stream time: {stream_time:.3f}s)")
                rag_logger.info(f"📊 Total response length: {len(answer)} characters")
                rag_logger.info(f"📊 Total tokens used: {total_tokens}")
                
                # Check if streaming failed to produce content
                if len(answer) == 0:
                    rag_logger.warning(f"⚠️ Streaming returned 0 characters. Falling back to non-streaming mode...")
                    # Retry with non-streaming (also with basic retry handling)
                    start_time = time.time()
                    max_attempts = 3
                    attempt = 0
                    response = None
                    while True:
                        attempt += 1
                        try:
                            answer = self.llm.complete(
                                messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=False,
                            )
                            response = None  # no response object; answer is the string
                            break
                        except Exception as api_error:
                            is_transient = is_retryable_error(api_error)
                            if attempt >= max_attempts or not is_transient:
                                rag_logger.error(f"❌ Non-streaming fallback API error (final): {api_error}")
                                raise LLMError(
                                    user_message=get_user_friendly_message(api_error),
                                    technical_message=str(api_error),
                                    retryable=is_transient
                                )
                            # Longer backoff: 5s, 10s, 15s instead of 2s, 4s, 6s
                            wait_seconds = 5 + (attempt * 5)
                            rag_logger.warning(
                                f"⚠️ Non-streaming fallback rate limit / queue error, retrying attempt "
                                f"{attempt}/{max_attempts} in {wait_seconds}s: {api_error}"
                            )
                            # Notify frontend about retry if callback provided
                            if retry_callback:
                                try:
                                    retry_callback({
                                        'type': 'api_retry',
                                        'message': f'AI service busy, retrying in {wait_seconds}s...',
                                        'step': 'generation',
                                        'data': {
                                            'attempt': attempt,
                                            'max_attempts': max_attempts,
                                            'wait_seconds': wait_seconds,
                                            'provider': self.llm.provider_name
                                        }
                                    })
                                except Exception as cb_error:
                                    rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                            time.sleep(wait_seconds)
                    call_time = time.time() - start_time
                    if response:
                        answer = response.choices[0].message.content.strip()
                    total_tokens = response.usage.total_tokens if response and getattr(response, 'usage', None) else 0
                    rag_logger.info(f"✅ Non-streaming fallback completed (call time: {call_time:.3f}s)")
                    rag_logger.info(f"📊 Fallback response length: {len(answer)} characters")
                    rag_logger.info(f"📊 Fallback tokens used: {total_tokens}")
                    
                    # Stream the complete answer in chunks to simulate streaming
                    if stream_callback and len(answer) > 0:
                        chunk_size = 50  # Stream in chunks of 50 characters
                        for i in range(0, len(answer), chunk_size):
                            chunk_content = answer[i:i+chunk_size]
                            try:
                                stream_callback(chunk_content)
                            except Exception as e:
                                rag_logger.warning(f"⚠️ Fallback stream callback error: {e}")
                else:
                    rag_logger.info(f"⚡ Average streaming speed: {len(answer)/stream_time:.1f} chars/sec")
                
                rag_logger.info(f"📝 Response preview: {answer[:300]}...")
            else:
                # Non-streaming response generation
                rag_logger.info(f"🤖 ===== RESPONSE GENERATION LLM CALL (NON-STREAMING) =====")
                rag_logger.info(f"🔍 Provider: {self.llm.provider_name}")
                rag_logger.info(f"📊 Max tokens: {max_tokens}")
                rag_logger.info(f"🌡️ Temperature: {temperature}")
                rag_logger.info(f"🎯 Ticker context: {ticker or 'General'}")
                rag_logger.info(f"📅 Quarter context: {f'{year} Q{quarter}' if year and quarter else 'Multiple/General'}")
                rag_logger.info(f"📋 Prompt length: {len(prompt)} characters")
                rag_logger.info(f"📊 Context chunks: {len(context_chunks)}")
                rag_logger.info(f"📋 Prompt preview: {prompt[:400]}...")
                
                # Non-streaming mode (original behavior) with basic retry for transient errors
                start_time = time.time()
                max_attempts = 3
                attempt = 0
                response = None
                while True:
                    attempt += 1
                    try:
                        answer = self.llm.complete(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=False,
                        )
                        response = None  # no response object when using llm.complete
                        break
                    except Exception as api_error:
                        is_transient = is_retryable_error(api_error)
                        if attempt >= max_attempts or not is_transient:
                            rag_logger.error(f"❌ Non-streaming API error (final): {api_error}")
                            raise LLMError(
                                user_message=get_user_friendly_message(api_error),
                                technical_message=str(api_error),
                                retryable=is_transient
                            )
                        wait_seconds = 5 + (attempt * 5)
                        rag_logger.warning(
                            f"⚠️ Non-streaming rate limit / queue error, retrying attempt "
                            f"{attempt}/{max_attempts} in {wait_seconds}s: {api_error}"
                        )
                        # Notify frontend about retry if callback provided
                        if retry_callback:
                            try:
                                retry_callback({
                                    'type': 'api_retry',
                                    'message': f'AI service busy, retrying in {wait_seconds}s...',
                                    'step': 'generation',
                                    'data': {
                                        'attempt': attempt,
                                        'max_attempts': max_attempts,
                                        'wait_seconds': wait_seconds,
                                        'provider': self.llm.provider_name
                                    }
                                })
                            except Exception as cb_error:
                                rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                        time.sleep(wait_seconds)
                call_time = time.time() - start_time
                rag_logger.info(f"✅ Non-streaming fallback completed (call time: {call_time:.3f}s)")
                if response:
                    rag_logger.info(f"📊 Response tokens used: {response.usage.total_tokens if response.usage else 'unknown'}")
                if response and hasattr(response, 'model'):
                    rag_logger.info(f"🤖 Model used: {response.model}")
                if response:
                    answer = response.choices[0].message.content.strip()
                rag_logger.info(f"📝 Fallback response length: {len(answer)} characters")
                rag_logger.info(f"📝 Answer preview: {answer[:300]}...")
            
            # Log generation completion to Logfire with full prompt and answer
            generation_time = time.time() - generation_start
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info(
                    "llm.generation.complete",
                    provider=self.llm.provider_name,
                    ticker=ticker,
                    answer_length=len(answer),
                    generation_time_ms=int(generation_time * 1000),
                    tokens_used=response.usage.total_tokens if response and getattr(response, 'usage', None) else None,
                    prompt=prompt if prompt else None,
                    answer=answer if answer else None,
                    context=context if context else None
                )

            if return_details:
                rag_logger.info(f"📊 Returning detailed response with metadata")
                return {
                    'answer': answer,
                    'prompt': prompt,
                    'context': context,
                    'model': self.config.get("openai_model"),
                    'tokens_used': response.usage.total_tokens if response and getattr(response, 'usage', None) else None
                }
            else:
                rag_logger.info(f"📝 Returning simple answer")
                return answer

        except Exception as e:
            # Log error to Logfire
            if LOGFIRE_AVAILABLE and logfire:
                logfire.error(
                    "llm.generation.error",
                    error=str(e),
                    ticker=ticker
                )
            logger.error(f"Error generating OpenAI response: {e}")
            raise Exception(f"Failed to generate response: {e}")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 3: MULTI-TICKER RESPONSE GENERATION
    # ═════════════════════════════════════════════════════════════════════

    def generate_multi_ticker_response(self, question: str, all_chunks: List[Dict[str, Any]], individual_results: List[Dict[str, Any]], show_details: bool = False, comprehensive: bool = True, stream_callback=None, news_context: str = None, ten_k_context: str = None, previous_answer: str = None, conversation_context: str = None, retry_callback=None, answer_mode: str = None) -> str:
        """Generate response using all chunks with company labels for multi-ticker questions.
        conversation_context: Optional formatted string of recent conversation for stateful follow-up questions.
        """
        rag_logger.info(f"🤖 Generating multi-ticker response using {len(all_chunks)} chunks with company labels")
        
        # Create a mapping of chunk citations to tickers
        citation_to_ticker = {}
        for result in individual_results:
            ticker = result['ticker']
            for citation in result['citations']:
                # Ensure citation is hashable for use as dict key
                if isinstance(citation, dict):
                    hashable_citation = citation.get('citation') or citation.get('id') or citation.get('chunk_index') or str(citation)
                else:
                    hashable_citation = citation
                citation_to_ticker[hashable_citation] = ticker
        
        # Prepare context with SOURCE [N] markers (matches single-ticker format for consistent citation handling)
        rag_logger.info(f"📝 Preparing context with SOURCE markers for {len(all_chunks)} chunks")
        context_parts = []
        for i, chunk in enumerate(all_chunks):
            citation_marker = chunk.get('citation', i + 1)
            ticker = chunk.get('ticker') or citation_to_ticker.get(citation_marker, 'Unknown')
            year = chunk.get('year', 'Unknown')
            quarter = chunk.get('quarter', 'Unknown')
            quarter_info = f"Q{quarter} {year}" if year != 'Unknown' and quarter != 'Unknown' else "financial data"
            context_parts.append(f"SOURCE [{citation_marker}] [{ticker}] {quarter_info}:\n{chunk['chunk_text']}")
            rag_logger.info(f"📄 SOURCE [{citation_marker}] {ticker} {quarter_info}: length={len(chunk['chunk_text'])}")
        
        context = "\n\n".join(context_parts)
        rag_logger.info(f"✅ Context prepared: total_length={len(context)}")

        # Add news context if available
        news_section = ""
        if news_context:
            news_section = f"\n\n{news_context}\n\nNote: The above news sources provide recent developments and current information. Use them alongside other available data sources as appropriate for answering the question."

        # Add 10-K SEC filings context if available
        ten_k_section = ""
        if ten_k_context:
            ten_k_section = f"\n\n{ten_k_context}\n\nNote: The above 10-K SEC filing data provides comprehensive annual financial information, including balance sheets, income statements, cash flow statements, and detailed business disclosures. Use this data as appropriate for answering the question, combining with other available sources when relevant."

        # Add previous answer if available (for iterative improvement)
        previous_answer_section = ""
        if previous_answer:
            previous_answer_section = f"""

PREVIOUS ANSWER (build upon this):
{previous_answer}

IMPORTANT: You are improving the previous answer. Build upon it by:
1. Keeping all accurate information from the previous answer
2. Adding new details, numbers, quotes, or context from the additional information below
3. Correcting any inaccuracies if found
4. Expanding on areas that were incomplete
5. Maintaining the same structure and flow while enhancing it
Do NOT start from scratch - improve and expand the previous answer."""

        conversation_context_section_multi = ""
        if conversation_context:
            conversation_context_section_multi = f"""

CONVERSATION HISTORY (recent messages; the current question may refer to "those companies", "their", "above", etc.):
{conversation_context}

Answer the current question with full context from this conversation. Resolve pronouns and references using the history above.
If the user asks "would you say the same about [X]?" or "what about [company]?", apply the same conclusion or framework from your previous answer to the new company (e.g. whether the same risk applies, or the same metric trend), then answer for that company."""

        # Create prompt based on comprehensive setting
        # Build unified citation instructions based on available sources
        additional_sources_instruction = self._build_citation_instructions(
            has_news=bool(news_context),
            has_10k=bool(ten_k_context)
        )

        # Determine what data sources are available for prompt text
        available_sources = []
        if news_context:
            available_sources.append("news sources")
        if ten_k_context:
            available_sources.append("10-K SEC filings")
        if available_sources:
            data_sources_text = f" {', '.join(available_sources)} are available - use whichever sources are most relevant for answering the question."
        else:
            data_sources_text = ""

        no_structured_context_multi = (not all_chunks) and not news_context and not ten_k_context
        no_context_instruction = '\n19. **No structured data available**: Clearly state that the information is not available in the company filings. End with a short nudge: "Would you like me to search the news instead?"' if no_structured_context_multi else ''

        # Use answer_mode if provided, fall back to comprehensive flag
        _mode = answer_mode or ("detailed" if comprehensive else "standard")

        if _mode == "direct":
            prompt = f"""You are a financial analyst. Do not use emojis in your responses. Answer the following question based on the provided company data.{data_sources_text}{previous_answer_section}{conversation_context_section_multi}

{news_section}{ten_k_section}
Company Data (each source labeled as SOURCE [N] [TICKER] Quarter):
{context}

Question: {question}

INSTRUCTIONS:
1. Answer in 2-4 sentences with the key figure(s) for each company.
2. Use **bold** for key figures. Reference companies by name.
3. When comparing numbers across companies, use a markdown table.
4. Use the exact SOURCE markers from the context (e.g., [1], [2], [3]) for transcript citations, [10K-1] for 10-K data, [N1] for news
{additional_sources_instruction}
5. NEVER use the word "chunks" - reference official documents naturally.
6. NEVER label or describe the format of your answer. Just answer naturally.
{no_context_instruction}

Answer in **markdown format**."""
        elif _mode == "detailed":
            prompt = f"""You are a senior equity research analyst. Analyze the following company data and answer the question comprehensively.{data_sources_text}{previous_answer_section}{conversation_context_section_multi}

{news_section}{ten_k_section}
Company Data (each source labeled as SOURCE [N] [TICKER] Quarter):
{context}

Question: {question}

INSTRUCTIONS:

**FORMAT & STRUCTURE:**
1. Write a comprehensive, well-structured comparative answer - NOT just a list of quotes
2. Start with an **Executive Summary** (3-4 sentences capturing the key findings across all companies)
3. Use clear sections with headers (##) - organize by theme OR by company depending on what's clearer
4. Include a **Comparative Analysis** section highlighting key differences and similarities
5. End with a **Conclusion** section (3-5 sentences) that synthesizes the key insights and provides actionable takeaways

**CONTENT DEPTH & ANALYTICAL VOICE:**
6. **Provide analysis and context throughout your response**, not just raw quotes. Explain implications and significance
7. **Weave analytical perspective throughout** - assess which company is better positioned, identify relative strengths/weaknesses, and provide your view on competitive dynamics within the main body of your analysis
8. **Use markdown tables to compare companies on key metrics** (revenue, margins, growth rates, etc.)
9. Include specific numbers, percentages, and metrics with proper attribution
10. Highlight industry trends or themes that emerge across multiple companies
11. **Integrate your analytical insights naturally** - discuss which company's approach seems more promising and what risks seem most material as part of your comparative analysis, not in a separate section
12. Use quotes sparingly and strategically - integrate them into your analysis

**ATTRIBUTION & SOURCES:**
13. Reference companies by name with proper attribution (e.g., "Apple disclosed in its FY2024 10-K...", "According to Microsoft's Q1 2025 earnings call...")
14. Use the exact SOURCE markers from the context (e.g., [1], [2], [3]) for transcript citations, [10K-1] for 10-K data, [N1] for news
{additional_sources_instruction}
15. Always mention the specific period (e.g., "In FY2024...", "During Q1 2025...")

**QUALITY STANDARDS:**
16. **Stay focused on the question** - ignore irrelevant information
17. If some companies have no relevant data, explicitly state this
18. NEVER use the word "chunks" - reference official company documents naturally
19. Use rich markdown: headers, **bold**, bullet points, **tables for numerical comparisons**
20. NEVER label or describe the format of your answer (e.g., do NOT say "here is a research report" or "this is a comparative analysis"). Just answer naturally.
{no_context_instruction}

Answer in **markdown format** with actionable insights across all companies."""
        else:  # standard
            prompt = f"""You are a financial analyst. Do not use emojis in your responses. Answer the question based on the following company data.{data_sources_text}{previous_answer_section}{conversation_context_section_multi}

{news_section}{ten_k_section}
Company Data (each source labeled as SOURCE [N] [TICKER] Quarter):
{context}

Question: {question}

INSTRUCTIONS:
1. Start with a 2-3 sentence overview of the key findings
2. Include specific numbers, metrics, and properly attributed quotes
3. **Use markdown tables when comparing numbers across companies** (revenue, margins, growth, etc.)
4. Reference companies by name with period attribution (e.g., "Apple's FY2024 10-K shows...", "In Microsoft's Q1 2025 call...")
{additional_sources_instruction}
5. Highlight key similarities and differences between companies
6. If some companies lack relevant data, note this briefly
7. **Stay focused on the question** - ignore irrelevant information
8. NEVER use the word "chunks" - reference official documents naturally
9. Use markdown: **bold** for emphasis, bullet points for lists, tables for comparisons
10. NEVER label or describe the format of your answer. Just answer naturally.
{no_context_instruction}

Answer in **markdown format**."""

        rag_logger.info(f"📝 Multi-ticker prompt created: length={len(prompt)}")

        # Use cached system prompt for multi-ticker (with or without news), mode-aware
        multi_ticker_system_prompt = self._get_system_prompt(
            'multi_ticker_news' if news_context else 'multi_ticker',
            answer_mode=_mode
        )
        
        multi_ticker_messages = [
            {"role": "system", "content": multi_ticker_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            max_tokens = self.config.get("cerebras_max_tokens", self.config.get("openai_max_tokens", 8000))
            if self.llm.provider_name == "OpenAI" or self.config.get("llm_provider") == "openai":
                temperature = self.config.get("openai_temperature", 1)
            else:
                temperature = self.config.get("cerebras_temperature", 0.1)
            if answer_mode:
                from .config import AnswerMode, ANSWER_MODE_CONFIG
                try:
                    mode_config = ANSWER_MODE_CONFIG[AnswerMode(answer_mode)]
                    max_tokens = min(max_tokens, mode_config["max_tokens"])
                except (ValueError, KeyError):
                    pass
            rag_logger.info(f"🤖 Sending multi-ticker request via {self.llm.provider_name}")

            # Check if streaming is requested
            if stream_callback:
                # Detailed LLM stage logging for multi-ticker streaming
                rag_logger.info(f"🤖 ===== MULTI-TICKER RESPONSE GENERATION LLM CALL (STREAMING) =====")
                rag_logger.info(f"🔍 Provider: {self.llm.provider_name}")
                rag_logger.info(f"📊 Max tokens: {max_tokens}")
                rag_logger.info(f"🌡️ Temperature: {temperature}")
                rag_logger.info(f"📊 Total chunks count: {len(all_chunks)}")
                rag_logger.info(f"📊 Individual results count: {len(individual_results)}")
                rag_logger.info(f"📋 Multi-ticker prompt length: {len(prompt)} characters")
                rag_logger.info(f"📋 Multi-ticker prompt preview: {prompt[:400]}...")
                
                rag_logger.info(f"🌊 Using streaming mode for multi-ticker response")
                answer = ""
                
                start_time = time.time()
                
                max_attempts = 3
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        stream = self.llm.complete(
                            multi_ticker_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=True,
                        )
                        break
                    except Exception as api_error:
                        is_transient = is_retryable_error(api_error)
                        if attempt >= max_attempts or not is_transient:
                            rag_logger.error(f"❌ Multi-ticker streaming API error (final): {api_error}")
                            raise LLMError(
                                user_message=get_user_friendly_message(api_error),
                                technical_message=str(api_error),
                                retryable=is_transient
                            )
                        # Longer backoff: 5s, 10s, 15s instead of 2s, 4s, 6s
                        wait_seconds = 5 + (attempt * 5)
                        rag_logger.warning(
                            f"⚠️ Multi-ticker streaming rate limit / queue error, retrying attempt "
                            f"{attempt}/{max_attempts} in {wait_seconds}s: {api_error}"
                        )
                        # Notify frontend about retry if callback provided
                        if retry_callback:
                            try:
                                retry_callback({
                                    'type': 'api_retry',
                                    'message': f'AI service busy, retrying in {wait_seconds}s...',
                                    'step': 'generation',
                                    'data': {
                                        'attempt': attempt,
                                        'max_attempts': max_attempts,
                                        'wait_seconds': wait_seconds,
                                        'provider': self.llm.provider_name
                                    }
                                })
                            except Exception as cb_error:
                                rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                        time.sleep(wait_seconds)

                chunk_count = 0
                content_chunk_count = 0
                try:
                    for chunk in stream:
                        chunk_count += 1
                        try:
                            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                                content = chunk.choices[0].delta.content
                                content_chunk_count += 1
                                
                                # Log first few chunks with content for debugging
                                if content_chunk_count <= 3:
                                    rag_logger.info(f"🌊 RAG Multi-ticker: Chunk {content_chunk_count} content: '{content[:100]}...' (length: {len(content)})")
                                
                                answer += content
                                # Call the streaming callback with the new content
                                try:
                                    stream_callback(content)
                                except Exception as e:
                                    rag_logger.warning(f"⚠️ Stream callback error: {e}")
                        except Exception as chunk_error:
                            rag_logger.error(f"❌ Error processing chunk {chunk_count}: {chunk_error}")
                            rag_logger.error(f"Chunk structure: {chunk}")
                except Exception as stream_error:
                    rag_logger.error(f"❌ Stream iteration error: {stream_error}")
                    if len(answer) == 0:
                        raise Exception(f"Stream failed with no content: {stream_error}")
                
                rag_logger.info(f"📊 Stream statistics: total_chunks={chunk_count}, content_chunks={content_chunk_count}, empty_chunks={chunk_count - content_chunk_count}")
                
                stream_time = time.time() - start_time
                rag_logger.info(f"✅ ===== MULTI-TICKER RESPONSE GENERATION STREAMING COMPLETED ===== (stream time: {stream_time:.3f}s)")
                rag_logger.info(f"📊 Multi-ticker response length: {len(answer)} characters")
                
                # Check if streaming failed to produce content
                if len(answer) == 0:
                    rag_logger.warning(f"⚠️ Streaming returned 0 characters. Falling back to non-streaming mode...")
                    # Retry with non-streaming (with basic retry handling)
                    start_time = time.time()
                    max_attempts = 3
                    attempt = 0
                    response = None
                    while True:
                        attempt += 1
                        try:
                            answer = self.llm.complete(
                                multi_ticker_messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                stream=False,
                            )
                            response = None
                            break
                        except Exception as api_error:
                            is_transient = is_retryable_error(api_error)
                            if attempt >= max_attempts or not is_transient:
                                rag_logger.error(f"❌ Multi-ticker non-streaming fallback API error (final): {api_error}")
                                raise LLMError(
                                    user_message=get_user_friendly_message(api_error),
                                    technical_message=str(api_error),
                                    retryable=is_transient
                                )
                            # Longer backoff: 5s, 10s, 15s instead of 2s, 4s, 6s
                            wait_seconds = 5 + (attempt * 5)
                            rag_logger.warning(
                                f"⚠️ Multi-ticker non-streaming fallback rate limit / queue error, retrying attempt "
                                f"{attempt}/{max_attempts} in {wait_seconds}s: {api_error}"
                            )
                            # Notify frontend about retry if callback provided
                            if retry_callback:
                                try:
                                    retry_callback({
                                        'type': 'api_retry',
                                        'message': f'AI service busy, retrying in {wait_seconds}s...',
                                        'step': 'generation',
                                        'data': {
                                            'attempt': attempt,
                                            'max_attempts': max_attempts,
                                            'wait_seconds': wait_seconds,
                                            'provider': self.llm.provider_name
                                        }
                                    })
                                except Exception as cb_error:
                                    rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                            time.sleep(wait_seconds)
                    call_time = time.time() - start_time
                    if response:
                        answer = response.choices[0].message.content.strip()
                    rag_logger.info(f"✅ Non-streaming fallback completed (call time: {call_time:.3f}s)")
                    rag_logger.info(f"📊 Fallback response length: {len(answer)} characters")
                    
                    # Stream the complete answer in chunks to simulate streaming
                    if stream_callback and len(answer) > 0:
                        chunk_size = 50  # Stream in chunks of 50 characters
                        for i in range(0, len(answer), chunk_size):
                            chunk_content = answer[i:i+chunk_size]
                            try:
                                stream_callback(chunk_content)
                            except Exception as e:
                                rag_logger.warning(f"⚠️ Fallback stream callback error: {e}")
                else:
                    rag_logger.info(f"⚡ Average streaming speed: {len(answer)/stream_time:.1f} chars/sec")
                
                rag_logger.info(f"📝 Multi-ticker response preview: {answer[:300]}...")
            else:
                rag_logger.info(f"🤖 ===== MULTI-TICKER RESPONSE GENERATION LLM CALL (NON-STREAMING) =====")
                rag_logger.info(f"🔍 Provider: {self.llm.provider_name}")
                rag_logger.info(f"📊 Max tokens: {max_tokens}")
                rag_logger.info(f"🌡️ Temperature: {temperature}")
                rag_logger.info(f"📊 Total chunks count: {len(all_chunks)}")
                rag_logger.info(f"📊 Individual results count: {len(individual_results)}")
                rag_logger.info(f"📋 Multi-ticker prompt length: {len(prompt)} characters")
                rag_logger.info(f"📋 Multi-ticker prompt preview: {prompt[:400]}...")
                
                start_time = time.time()
                max_attempts = 3
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        answer = self.llm.complete(
                            multi_ticker_messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            stream=False,
                        )
                        break
                    except Exception as api_error:
                        is_transient = is_retryable_error(api_error)
                        if attempt >= max_attempts or not is_transient:
                            rag_logger.error(f"❌ Multi-ticker non-streaming API error (final): {api_error}")
                            raise LLMError(
                                user_message=get_user_friendly_message(api_error),
                                technical_message=str(api_error),
                                retryable=is_transient
                            )
                        wait_seconds = 5 + (attempt * 5)
                        rag_logger.warning(
                            f"⚠️ Multi-ticker non-streaming rate limit / queue error, retrying attempt "
                            f"{attempt}/{max_attempts} in {wait_seconds}s: {api_error}"
                        )
                        if retry_callback:
                            try:
                                retry_callback({
                                    'type': 'api_retry',
                                    'message': f'AI service busy, retrying in {wait_seconds}s...',
                                    'step': 'generation',
                                    'data': {
                                        'attempt': attempt,
                                        'max_attempts': max_attempts,
                                        'wait_seconds': wait_seconds,
                                        'provider': self.llm.provider_name
                                    }
                                })
                            except Exception as cb_error:
                                rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                        time.sleep(wait_seconds)

                call_time = time.time() - start_time
                rag_logger.info(f"✅ ===== MULTI-TICKER RESPONSE GENERATION LLM RESPONSE ===== (call time: {call_time:.3f}s)")
                rag_logger.info(f"🏢 Provider: {self.llm.provider_name}")
                rag_logger.info(f"📝 Multi-ticker answer: length={len(answer)} characters")
                rag_logger.info(f"📝 Multi-ticker answer preview: {answer[:300]}...")
            
            return answer
            
        except Exception as e:
            # Log full error details server-side but return a generic, user-safe error
            logger.error(f"Error generating multi-ticker response: {e}", exc_info=True)
            raise Exception("The service is temporarily busy and could not generate a multi-company answer. Please try again in a moment.")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 4: PARALLEL QUARTER PROCESSING & SYNTHESIS
    # ═════════════════════════════════════════════════════════════════════

    def _group_chunks_by_quarter(self, chunk_objects: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by quarter for parallel processing.
        
        Args:
            chunk_objects: List of chunk dictionaries with year and quarter metadata
            
        Returns:
            Dictionary mapping quarter_id (e.g., '2025_q1') to list of chunks
        """
        quarters_map = defaultdict(list)
        
        for chunk in chunk_objects:
            year = chunk.get('year')
            quarter = chunk.get('quarter')
            
            if year and quarter:
                quarter_id = f"{year}_q{quarter}"
                quarters_map[quarter_id].append(chunk)
        
        rag_logger.info(f"📊 Grouped {len(chunk_objects)} chunks into {len(quarters_map)} quarters: {list(quarters_map.keys())}")
        return dict(quarters_map)
    
    async def _generate_quarter_response(self, question: str, quarter_id: str, quarter_chunks: List[Dict[str, Any]],
                                        ticker: str = None, retry_callback=None, conversation_context: str = None) -> Dict[str, Any]:
        """Generate response for a single quarter (runs in parallel with other quarters).

        Args:
            question: User's question
            quarter_id: Quarter identifier (e.g., '2025_q1')
            quarter_chunks: Chunks for this specific quarter
            ticker: Company ticker symbol
            retry_callback: Optional callback for retry notifications

        Returns:
            Dict with quarter_id, answer, and metadata
        """
        try:
            # Extract year and quarter from quarter_id
            year, quarter = quarter_id.split('_q')
            year = int(year)
            quarter = int(quarter)

            # Prepare context for this quarter
            context_chunks = [chunk['chunk_text'] for chunk in quarter_chunks]

            rag_logger.info(f"🔄 Generating response for {quarter_id} with {len(context_chunks)} chunks")

            # Run synchronous response generation in thread pool
            # Use functools.partial to pass keyword arguments through run_in_executor
            from functools import partial
            loop = asyncio.get_event_loop()
            gen_func = partial(
                self.generate_openai_response,
                question=question,
                context_chunks=context_chunks,
                chunk_objects=quarter_chunks,
                return_details=False,
                ticker=ticker,
                year=year,
                quarter=quarter,
                stream_callback=None,  # No streaming for parallel processing
                conversation_context=conversation_context,
                retry_callback=retry_callback
            )
            answer = await loop.run_in_executor(None, gen_func)
            
            rag_logger.info(f"✅ Generated response for {quarter_id}: {len(answer)} characters")
            
            return {
                'quarter_id': quarter_id,
                'year': year,
                'quarter': quarter,
                'answer': answer,
                'chunks': quarter_chunks,
                'chunk_count': len(quarter_chunks)
            }
            
        except Exception as e:
            rag_logger.error(f"❌ Error generating response for {quarter_id}: {e}")
            # Parse year and quarter even for error responses to ensure consistent structure
            try:
                year, quarter = quarter_id.split('_q')
                year = int(year)
                quarter = int(quarter)
            except:
                year = 0
                quarter = 0
            return {
                'quarter_id': quarter_id,
                'year': year,
                'quarter': quarter,
                'answer': f"Error processing {quarter_id}: {str(e)}",
                'chunks': quarter_chunks,
                'chunk_count': len(quarter_chunks),
                'error': str(e)
            }
    
    async def generate_openai_response_parallel_quarters(self, question: str, chunk_objects: List[Dict[str, Any]],
                                                         ticker: str = None, stream_callback=None, retry_callback=None,
                                                         conversation_context: str = None) -> str:
        """Generate responses for each quarter in parallel, then combine them.

        This method is used when multiple quarters are detected in the query.
        It processes each quarter independently in parallel, then intelligently
        combines all quarter responses into a comprehensive final answer.

        Args:
            question: User's question
            chunk_objects: List of chunk dictionaries with metadata
            ticker: Company ticker symbol (optional)
            stream_callback: Callback for streaming the final combined response
            
        Returns:
            Combined comprehensive answer from all quarters
        """
        rag_logger.info(f"🚀 Starting parallel quarter processing")
        
        # Group chunks by quarter
        quarters_map = self._group_chunks_by_quarter(chunk_objects)
        
        if len(quarters_map) <= 1:
            # Single quarter or no quarters - use regular processing
            rag_logger.info(f"⚠️ Only {len(quarters_map)} quarter(s) detected, using regular processing")
            context_chunks = [chunk['chunk_text'] for chunk in chunk_objects]
            year = chunk_objects[0].get('year') if chunk_objects else None
            quarter = chunk_objects[0].get('quarter') if chunk_objects else None
            return self.generate_openai_response(question, context_chunks, chunk_objects,
                                                False, ticker, year, quarter, stream_callback,
                                                conversation_context=conversation_context)
        
        # Send initial progress message
        if stream_callback:
            try:
                company_name = ticker if ticker else "the company"
                quarters_list = sorted(quarters_map.keys())
                quarters_human = [f"Q{q.split('_q')[1]} {q.split('_q')[0]}" for q in quarters_list]
                stream_callback(f"\n\n🔍 **Analyzing {', '.join(quarters_human)} for {company_name}...**\n\n")
            except Exception as e:
                rag_logger.warning(f"⚠️ Stream callback error for initial message: {e}")
        
        # Process each quarter in parallel
        rag_logger.info(f"⚡ Processing {len(quarters_map)} quarters in parallel")

        tasks = []
        for quarter_id, quarter_chunks in quarters_map.items():
            task = self._generate_quarter_response(question, quarter_id, quarter_chunks, ticker, retry_callback, conversation_context)
            tasks.append(task)
        
        # Wait for all quarters to complete
        quarter_responses = await asyncio.gather(*tasks)
        
        # Sort responses by quarter (chronological order)
        quarter_responses.sort(key=lambda x: x['quarter_id'])
        
        rag_logger.info(f"✅ Completed parallel processing for {len(quarter_responses)} quarters")
        
        # Send progress message before synthesis
        if stream_callback:
            try:
                stream_callback(f"\n\n✅ **Preparing comprehensive response...**\n\n")
            except Exception as e:
                rag_logger.warning(f"⚠️ Stream callback error for synthesis message: {e}")
        
        # Combine quarter responses into final answer
        final_answer = await self._combine_quarter_responses(
            question, quarter_responses, ticker, stream_callback
        )
        
        return final_answer
    
    async def _combine_quarter_responses(self, question: str, quarter_responses: List[Dict[str, Any]], 
                                        ticker: str = None, stream_callback=None) -> str:
        """Combine multiple quarter responses into a single comprehensive answer.
        
        Args:
            question: Original user question
            quarter_responses: List of responses from each quarter
            ticker: Company ticker symbol
            stream_callback: Callback for streaming the combined response
            
        Returns:
            Final combined comprehensive answer
        """
        rag_logger.info(f"🔄 Combining {len(quarter_responses)} quarter responses into final answer")

        # Filter out failed responses and log them
        successful_responses = []
        failed_responses = []
        for qr in quarter_responses:
            if qr.get('error'):
                failed_responses.append(qr)
                rag_logger.warning(f"⚠️ Skipping failed quarter {qr.get('quarter_id', 'unknown')}: {qr.get('error')}")
            else:
                successful_responses.append(qr)

        # If all responses failed, raise an error
        if not successful_responses:
            error_msg = "All quarter API calls failed. The AI service may be experiencing high traffic."
            rag_logger.error(f"❌ {error_msg}")
            raise Exception(error_msg)

        # Log if some responses failed
        if failed_responses:
            rag_logger.warning(f"⚠️ {len(failed_responses)} of {len(quarter_responses)} quarter(s) failed, proceeding with {len(successful_responses)} successful")

        # Use only successful responses for synthesis
        quarter_responses = successful_responses

        # Prepare context with all quarter responses
        company_name = ticker if ticker else "the company"
        quarters_human = [f"Q{qr.get('quarter', '?')} {qr.get('year', '?')}" for qr in quarter_responses]

        # Get synthesis prompt from centralized prompts
        prompt = get_quarter_synthesis_prompt(question, quarter_responses, company_name, quarters_human)

        rag_logger.info(f"📝 Synthesis prompt created: length={len(prompt)}")
        
        synthesis_messages = [
            {"role": "system", "content": QUARTER_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        try:
            max_tokens = self.config.get("cerebras_max_tokens", self.config.get("openai_max_tokens", 8000))
            if self.llm.provider_name == "OpenAI" or self.config.get("llm_provider") == "openai":
                temperature = self.config.get("openai_temperature", 1)
            else:
                temperature = self.config.get("cerebras_temperature", 0.1)
            rag_logger.info(f"🤖 Using {self.llm.provider_name} for synthesis")
            
            if stream_callback:
                rag_logger.info(f"🌊 Streaming synthesis response")
                answer = ""
                start_time = time.time()
                stream = self.llm.complete(
                    synthesis_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        answer += content
                        try:
                            stream_callback(content)
                        except Exception as e:
                            rag_logger.warning(f"⚠️ Stream callback error: {e}")
                stream_time = time.time() - start_time
                rag_logger.info(f"✅ Synthesis streaming completed: {len(answer)} chars in {stream_time:.2f}s")
            else:
                start_time = time.time()
                answer = self.llm.complete(
                    synthesis_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                call_time = time.time() - start_time
                rag_logger.info(f"✅ Synthesis completed: {len(answer)} chars in {call_time:.2f}s")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error combining quarter responses: {e}")
            # Fallback: concatenate quarter responses with headers and metadata
            fallback = f"# Multi-Quarter Analysis for {company_name}\n\n"
            fallback += f"**Quarters Analyzed**: {', '.join(quarters_human)}\n\n"
            fallback += f"**Question**: {question}\n\n"
            fallback += "---\n\n"
            for qr in quarter_responses:
                quarter_label = f"Q{qr['quarter']} {qr['year']}"
                fallback += f"## {quarter_label} - {company_name}\n\n"
                fallback += f"{qr['answer']}\n\n"
                fallback += "---\n\n"
            return fallback

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 5: ANSWER QUALITY EVALUATION
    # ═════════════════════════════════════════════════════════════════════

    async def evaluate_answer_quality(self, original_question: str, answer: str, context_chunks: List[str], available_chunks: List[Dict[str, Any]] = None, conversation_memory=None, conversation_id: str = None, follow_up_questions_asked: List[str] = None, evaluation_context: List[Dict[str, Any]] = None, reasoning_context: str = None, data_source: str = None, answer_mode: str = None) -> Dict[str, Any]:
        """Evaluate the quality of the generated answer and let agent decide if iteration is needed.

        Considers conversation history and the agent's initial reasoning to understand context.

        Args:
            follow_up_questions_asked: List of questions already asked in previous iterations to avoid duplicates
            evaluation_context: List of previous iteration evaluations with full context
                [{"iteration": int, "evaluation": Dict, "confidence": float}, ...]
            reasoning_context: The agent's initial reasoning about what info to find (guides evaluation)
            data_source: The data source routing from question analysis ('10k', 'earnings_transcripts', 'latest_news', 'hybrid')

        Returns dict with: completeness_score, accuracy_score, clarity_score, specificity_score,
        overall_confidence, should_iterate, iteration_decision_reasoning, follow_up_questions, evaluation_reasoning
        """
        eval_start = time.time()

        # Log evaluation start to Logfire with full question and answer
        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "llm.evaluation.start",
                question=original_question,
                answer=answer if answer else None,
                question_length=len(original_question),
                answer_length=len(answer),
                context_chunks_count=len(context_chunks),
                previous_iterations=len(evaluation_context) if evaluation_context else 0
            )

        if not self.openai_available:
            raise Exception("OpenAI not available for answer evaluation")
        
        max_retries = 3  # Keep it small
        retry_delay = 0  # No delays for immediate retries
        
        for attempt in range(max_retries):
            try:
                rag_logger.info(f"🔄 Evaluation attempt {attempt + 1}/{max_retries}")
                
                # Get conversation history if available
                conversation_context = ""
                if conversation_id and conversation_memory:
                    try:
                        conversation_context = await conversation_memory.format_context(conversation_id)
                        if conversation_context:
                            rag_logger.info(f"📜 Retrieved conversation history for evaluation")
                            conversation_context = f"\n\nCONVERSATION HISTORY:\n{conversation_context}\n\nIMPORTANT: The user's current question may be a follow-up referencing previous context. Consider whether the answer addresses the FULL intent given the conversation history."
                    except Exception as e:
                        rag_logger.warning(f"⚠️ Failed to get conversation context: {e}")
                        conversation_context = ""
                
                # Prepare context summary for evaluation
                context_summary = ""
                if available_chunks:
                    context_summary = "\n\nAvailable Context Chunks:\n"
                    for i, chunk in enumerate(available_chunks[:5], 1):  # Show first 5 chunks
                        context_summary += f"\nChunk {i} (similarity: {chunk.get('distance', chunk.get('similarity', 'N/A'))}):\n"
                        context_summary += f"Content: {chunk.get('chunk_text', '')[:200]}...\n"
                        if chunk.get('year') and chunk.get('quarter'):
                            context_summary += f"Quarter: {chunk['year']}_q{chunk['quarter']}\n"
                        if chunk.get('ticker'):
                            context_summary += f"Ticker: {chunk['ticker']}\n"

                # Prepare iteration memory section with full evaluation history
                iteration_memory_section = ""
                if evaluation_context and len(evaluation_context) > 0:
                    iteration_memory_section = "\n\n" + "="*80 + "\n"
                    iteration_memory_section += "ITERATION MEMORY - WHAT HAS ALREADY BEEN DONE:\n"
                    iteration_memory_section += "="*80 + "\n"

                    for iter_info in evaluation_context:
                        iteration_num = iter_info.get('iteration', 'Unknown')
                        eval_data = iter_info.get('evaluation', {})
                        searches = iter_info.get('searches_performed', {})

                        iteration_memory_section += f"\n--- ITERATION {iteration_num} ---\n"
                        iteration_memory_section += f"Confidence Score: {iter_info.get('confidence', 'N/A')}\n"
                        iteration_memory_section += f"Decision: {'Continued iterating' if eval_data.get('should_iterate') else 'Stopped'}\n"
                        iteration_memory_section += f"Reasoning: {eval_data.get('iteration_decision_reasoning', 'N/A')}\n"

                        # Show what searches were actually performed and their results
                        if searches:
                            iteration_memory_section += f"\nSearches Performed:\n"
                            if searches.get('transcript_search_performed'):
                                results = searches.get('transcript_search_results', 0)
                                iteration_memory_section += f"  - Transcript search: {results} chunks found\n"
                            if searches.get('news_search_performed'):
                                results = searches.get('news_search_results', 0)
                                iteration_memory_section += f"  - News search: {results} articles found\n"
                            if searches.get('ten_k_search_performed'):
                                results = searches.get('ten_k_search_results', 0)
                                iteration_memory_section += f"  - 10-K search: {results} results found\n"

                        if eval_data.get('follow_up_questions'):
                            iteration_memory_section += f"\nQuestions Asked:\n"
                            for q in eval_data.get('follow_up_questions', []):
                                iteration_memory_section += f"  - {q}\n"

                        iteration_memory_section += "\n"

                    iteration_memory_section += "="*80 + "\n"
                    iteration_memory_section += "CRITICAL INSTRUCTIONS FOR THIS ITERATION:\n"
                    iteration_memory_section += "="*80 + "\n"
                    iteration_memory_section += "1. Review ALL previous iterations above - including what searches were performed\n"
                    iteration_memory_section += "2. DO NOT ask questions that are semantically similar to already-asked questions\n"
                    iteration_memory_section += "3. DO NOT re-investigate gaps that were already addressed in previous iterations\n"
                    iteration_memory_section += "4. If a data source was searched and returned 0 results, do NOT keep retrying the same source\n"
                    iteration_memory_section += "5. If previous iterations already gathered specific data types (e.g., revenue metrics),\n"
                    iteration_memory_section += "   do NOT ask for the same data again - focus on different aspects\n"
                    iteration_memory_section += "6. If the user's specified data source has been exhaustively searched with limited results,\n"
                    iteration_memory_section += "   consider setting should_iterate=false and providing the best answer with available data\n"
                    iteration_memory_section += "="*80 + "\n"

                # Include agent's initial reasoning if available
                reasoning_section = ""
                if reasoning_context:
                    reasoning_section = f"""
AGENT'S RESEARCH PLAN (what we intended to find):
{reasoning_context}

Evaluate whether the answer addresses what the agent planned to find above.
"""

                # Data source routing - controls which search types are allowed
                data_source_routing = f"""
**MANDATORY DATA SOURCE CONSTRAINT (from user's question):**
The user's question was analyzed and routed to data_source="{data_source}".

**STRICT RULES - YOU MUST FOLLOW THESE:**

1. **If user explicitly mentioned a data source in their question (e.g., "from the 10k", "in the earnings call", "latest news"), you MUST respect that choice throughout ALL iterations.**

2. **For data_source="{data_source}":**"""

                if data_source == '10k':
                    data_source_routing += """
   - The user explicitly wants 10-K SEC filing data
   - NEVER set needs_transcript_search=true - this violates the user's explicit request
   - Only set needs_news_search=true if user ALSO explicitly asked for news in the question
   - If 10-K data has been exhaustively searched:
     * If you found relevant information (even partial), present it with should_iterate=false
     * ONLY if you found absolutely nothing relevant after exhaustive search, set should_iterate=false and state: "The requested information is not available in the 10-K filings"
     * Do NOT suggest switching to earnings transcripts - respect the user's data source choice"""
                elif data_source == 'latest_news':
                    data_source_routing += """
   - The user explicitly wants latest news/web data
   - NEVER set needs_transcript_search=true - this violates the user's explicit request
   - Set needs_news_search=true if more news would help answer the question
   - If news has been exhaustively searched:
     * If you found relevant information (even partial), present it with should_iterate=false
     * ONLY if you found absolutely nothing relevant after exhaustive search, set should_iterate=false and state: "The requested information is not available in recent news"
     * Do NOT suggest switching to transcripts/10-K - respect the user's data source choice"""
                elif data_source == 'earnings_transcripts':
                    data_source_routing += """
   - The user is asking about earnings transcripts/calls
   - Set needs_transcript_search=true if more transcript data would help
   - Only set needs_news_search=true if user ALSO explicitly asked for news
   - If transcripts have been exhaustively searched:
     * If you found relevant information (even partial), present it with should_iterate=false
     * ONLY if you found absolutely nothing relevant after exhaustive search, set should_iterate=false and state: "The requested information is not available in earnings transcripts"
     * Do NOT automatically switch to other sources unless user asked for them"""
                else:  # hybrid or unspecified
                    data_source_routing += """
   - No specific data source restriction - user's question is broad
   - Set needs_transcript_search=true if transcript data would help
   - Set needs_news_search=true if news data would help
   - You may use multiple data sources to build a comprehensive answer"""

                data_source_routing += """

3. **EXHAUSTION RULE:** If you've already searched the user's specified data source in previous iterations and found limited/no additional information, DO NOT keep retrying the same source. Instead:
   - Set should_iterate=false
   - Provide the best answer possible with available data
   - Be transparent about data limitations
   - Do NOT switch to unauthorized data sources just to have "something" to show"""

                evaluation_prompt = f"""
Evaluate this AI-generated answer against the user's question. Be strict; only excellent, comprehensive answers should pass.
{conversation_context}
Original question: {original_question}
{reasoning_section}
**News requests:** If the question asks for "latest news", "recent news", "what's happening", "search the web", etc., set needs_news_search=true regardless of other sources.
- If the original question contains: "latest news", "recent news", "current news", "breaking news", "what's happening", "latest updates", "recent developments", "what's new", "any news", "recent events", "current events", "latest information", "search the web", "check online", "look up news"
- Then you MUST set needs_news_search=true regardless of whether 10-K or transcripts have information
- User explicitly asking for news means they want news sources included

Generated Answer: {answer}

Context Available: {len(context_chunks)} chunks used in answer generation{context_summary}{iteration_memory_section}

TASK 1 - STRICT Evaluation (rate each 1-10, BE HARSH):

1. **Completeness (1-10)**: Does the answer FULLY address EVERY part of the original question?
   - Score 8-10: Addresses ALL aspects with comprehensive detail
   - Score 5-7: Addresses main points but missing some aspects or depth
   - Score 1-4: Incomplete, vague, or misses key parts of the question
   
2. **Accuracy (1-10)**: Is EVERY statement factually correct and properly sourced?
   - Score 8-10: All facts verified from context, no speculation
   - Score 5-7: Mostly accurate but some unsupported claims
   - Score 1-4: Contains errors or unsupported statements
   
3. **Clarity (1-10)**: Is the answer crystal clear and well-organized?
   - Score 8-10: Perfectly structured, easy to understand
   - Score 5-7: Generally clear but could be better organized
   - Score 1-4: Confusing, poorly structured
   
4. **Specificity (1-10)**: Does it provide CONCRETE numbers, quotes, and details?
   - Score 8-10: Rich with specific data, quotes, metrics
   - Score 5-7: Some specifics but mostly general statements
   - Score 1-4: Vague, lacks concrete details

TASK 2 - CRITICAL Analysis for Iteration Decision:

**CONFIDENCE CALIBRATION GUIDELINES:**

Set `overall_confidence` using these calibrated guidelines:

- **0.90-1.0**: Answer is factually correct with specific numbers/evidence from sources
  - All key facts present and cited
  - May have minor formatting issues but content is complete
  - DO NOT penalize for verbosity if facts are correct
  - **If answer contains specific cited numbers/dates that directly answer the question → confidence ≥ 0.85**

- **0.75-0.89**: Answer is mostly correct but missing minor details
  - Core facts present but could add supporting context
  - All numbers/dates cited but explanation could be clearer

- **0.50-0.74**: Answer has correct direction but incomplete evidence
  - Has the right idea but vague on specifics
  - Missing important numbers or dates
  - Needs another iteration to solidify

- **0.25-0.49**: Answer is partially wrong or highly speculative
  - Makes claims without source support
  - Contains factual errors
  - **If answer says "cannot determine" due to missing data → confidence should be 0.20-0.40**

- **<0.25**: Answer is fundamentally wrong or completely lacks evidence
  - No relevant information retrieved
  - Cannot answer the question

**Important calibration rules:**
1. If answer contains specific cited numbers/dates that directly answer the question → confidence ≥ 0.85
2. If answer says "cannot determine" due to missing data → confidence should be 0.20-0.40 (low confidence is correct)
3. If answer is hedging ("might be", "possibly") but has correct facts → confidence 0.70-0.80
4. DO NOT give 0.50 as a default - choose a calibrated score based on actual answer quality

{self._get_iteration_stance(answer_mode)}

**ITERATE (should_iterate=true) if ANY of these apply:**
- Missing ANY quantitative data (revenue, growth %, margins, EPS, guidance numbers)
- Missing direct executive quotes on strategy, outlook, or key topics
- Lacks time comparisons (QoQ, YoY, multi-quarter trends)
- Missing product/segment/geographic breakdowns
- No forward guidance or outlook mentioned
- Incomplete competitive comparisons
- Any vague or general statements that could be more specific
- Any aspect of question addressed partially or superficially
- Could benefit from additional context or elaboration
- Answer is good but NOT exceptional
- **If there's ANY room for improvement, iterate**

**SEARCH TYPE DECISION (needs_transcript_search and needs_news_search):**
The agent must decide what type of information to search for in the next iteration:

**DATA SOURCE ROUTING - CRITICAL OVERRIDE:**
{data_source_routing}

**TRANSCRIPT SEARCH (needs_transcript_search=true if ANY apply):**
- Answer lacks specific financial metrics, revenue numbers, growth percentages, margins, EPS
- Missing executive quotes, guidance, or strategic commentary
- Question asks about earnings, financial performance, quarterly results, or company strategy
- Need historical context, trends, or comparisons across quarters
- Answer would benefit from detailed financial data (earnings transcripts, 10-K filings)
- **Set needs_transcript_search=true if historical financial data would significantly improve the answer**
- **⚠️ CRITICAL: Check the DATA SOURCE ROUTING section above FIRST. If the user explicitly requested a different data source (10k, news), you MUST NOT set needs_transcript_search=true - this would violate their explicit request.**

**NEWS/WEB SEARCH (needs_news_search=true if ANY condition applies):**
- **ALWAYS use news if user EXPLICITLY asks for it:**
  1. User asks for "latest news", "recent news", "current news", "breaking news", "what's happening", "latest updates", "recent developments"
  2. User asks "what's new", "any news", "recent events", "current events", "latest information"
  3. User explicitly requests web/news sources: "search the web", "check online", "look up news"
  4. **If user explicitly asks for news, set needs_news_search=true regardless of 10-K/transcript availability**

- **Use news as FALLBACK when:**
  1. 10-K filings have been searched and don't contain the required information, AND
  2. Earnings transcripts have been searched and don't contain the required information, AND
  3. The question asks about very recent events (within days/weeks) that may not be in 10-K or earnings transcripts, OR
  4. The answer lacks information about current market conditions, recent stock movements, or breaking news that isn't in 10-K/transcripts, OR
  5. Question involves topics that change frequently (e.g., stock price, recent partnerships, current events) not covered in filings/transcripts

- **DO NOT use news as fallback if:**
  - 10-K filings contain the information (even if not perfectly formatted)
  - Earnings transcripts contain the information
  - The question can be answered from 10-K or transcript data
  - You haven't exhausted 10-K and transcript searches first (unless user explicitly asked for news)

- **Summary: Set needs_news_search=true if user explicitly asks for news OR if 10-K/transcripts are insufficient**

**RESTRICTION - DO NOT USE WEB SEARCH IF:**
- **User explicitly mentions "earnings calls", "earnings transcripts", "conference calls", "quarterly calls", or similar terms indicating they want ONLY earnings transcript information**
- **User's question specifically asks about what was said/discussed in earnings calls - stick to transcript data only**
- **Question is clearly about earnings call content (e.g., "What did they say in the earnings call?", "What was discussed in the transcript?")**
- **If user explicitly wants earnings call information, set needs_news_search=false even if transcript results are limited - respect user's intent**

**IMPORTANT:** The agent can decide to search BOTH transcripts AND news if both would help. Set both flags to true if needed.

**STOP iterating (should_iterate=false) if ANY of these are true:**

**Condition A - Answer is excellent:**
- Answer is EXCEPTIONAL and COMPREHENSIVE (90%+ complete)
- EVERY part of question thoroughly addressed with rich detail
- Abundant specific numbers, percentages, and concrete metrics
- Answer is so complete that another iteration would add minimal value

**Condition B - Data source exhausted (CRITICAL):**
- User explicitly requested a specific data source (10k, earnings transcripts, or news)
- That data source has been searched in previous iterations
- No significant new information was found in recent iterations
- **In this case: STOP and provide the best answer with available data. Do NOT switch to unauthorized data sources.**
- State clearly in your reasoning: "The user's specified data source has been exhaustively searched."

**AUTOMATIC STOP CONDITION:**
- If overall_confidence reaches 0.9 (90%) or higher, the system will AUTOMATICALLY stop iterating
- When confidence is 90%+, set should_iterate=false to align with the auto-stop

**IMPORTANT:** Err on the side of iterating for broad questions. But if user specified a data source and it's exhausted, respect that and stop gracefully.

TASK 3 - DETAILED Reasoning (ELABORATE and SPECIFIC):
In "iteration_decision_reasoning", provide a COMPREHENSIVE 3-5 sentence explanation:

**If iterating (should_iterate=true)** - MOST COMMON - include:
1. What specific data/details are MISSING (e.g., "Missing specific revenue growth percentages")
2. What aspects of the question are INCOMPLETE (e.g., "Doesn't compare margins across quarters")  
3. What would IMPROVE the answer (e.g., "Need executive quotes on strategy")
4. Why another iteration would add VALUE
Example: "While the answer mentions revenue, it lacks specific growth percentages and year-over-year comparisons. The question asks about profitability but no margin data is provided. Executive commentary on future guidance is missing. Another iteration could gather these specific metrics and quotes to provide a complete financial picture."

**If stopping (should_iterate=false)** - RARE - only when answer is EXCEPTIONAL:
1. Explicitly confirm answer is COMPREHENSIVE and EXCEPTIONAL (90%+ quality)
2. List ALL aspects thoroughly addressed with specific examples
3. Confirm all metrics, quotes, comparisons present
4. State why another iteration would add minimal value
Example: "This answer is exceptionally comprehensive, addressing all aspects with specific revenue growth of 15% YoY, detailed margin expansion of 200bps with segment breakdowns, multiple direct executive quotes on strategy and guidance, complete QoQ and YoY comparisons with proper context, and forward outlook. The answer is 95%+ complete with rich quantitative detail and executive insights. Another iteration would add minimal incremental value."

TASK 4 - Search-Optimized Follow-up Queries:
If iterating, provide 1-3 SEARCH-OPTIMIZED KEYWORD PHRASES (NOT questions) for semantic/vector search.

**CRITICAL - Format for RAG Search:**
- Use DECLARATIVE phrases with key concepts, NOT full questions
- Extract core entities, metrics, and topics
- Remove question framing ("What", "How", "Did")
- Keep it focused and concise (5-10 words max)
- Think: "What would find this in a vector database?"

**Examples:**
❌ BAD: "What specific revenue growth percentage was reported and how does it compare to the previous quarter?"
✅ GOOD: "revenue growth percentage quarter comparison"

❌ BAD: "Did executives provide updated capex guidance for 2025 and what portion was tied to AI?"
✅ GOOD: "capex guidance 2025 AI allocation"

❌ BAD: "What were the operating margins and gross margins, and how do they trend over recent quarters?"
✅ GOOD: "operating margins gross margins quarterly trend"

These keyword phrases will be used for semantic search, so optimize for embedding similarity.

REQUIRED JSON FIELDS:
- completeness_score, accuracy_score, clarity_score, specificity_score (1-10)
- overall_confidence (0.0-1.0)
- should_iterate (boolean)
- needs_transcript_search (boolean) - REQUIRED: decide if transcript search needed
- needs_news_search (boolean) - REQUIRED: decide if news search needed
- transcript_search_query (string or null) - REQUIRED if needs_transcript_search=true
- news_search_query (string or null) - REQUIRED if needs_news_search=true
- iteration_decision_reasoning (string)
- follow_up_questions (array of strings)
- evaluation_reasoning (string)

IMPORTANT: Respond ONLY with valid JSON. No markdown or additional text.

EXAMPLE (transcript_search_query and follow_up_questions must be keyword phrases, not full questions):
{{
    "completeness_score": 6,
    "accuracy_score": 8,
    "clarity_score": 7,
    "specificity_score": 5,
    "overall_confidence": 0.65,
    "should_iterate": true,
    "needs_transcript_search": true,
    "needs_news_search": false,
    "transcript_search_query": "revenue growth percentage quarter comparison",
    "news_search_query": null,
    "iteration_decision_reasoning": "While the answer correctly identifies revenue trends, it lacks specific growth percentages which are central to the question. The user asked about profitability but no operating margin data or comparisons are provided. Executive commentary on forward guidance and strategic priorities is completely absent. The answer would benefit from quarter-over-quarter comparisons to show the progression mentioned. These gaps prevent the answer from fully addressing the user's information needs.",
    "follow_up_questions": ["revenue growth percentage quarter comparison", "operating margins gross margins quarterly trend", "executive forward guidance strategic priorities market outlook"],
    "evaluation_reasoning": "The answer provides directional accuracy on revenue trends but lacks the quantitative precision (specific %, margin data) and executive insights (quotes, guidance) needed to fully satisfy the question. The structure is clear but the content depth is insufficient for a comprehensive financial analysis. Specificity score is low due to absence of concrete metrics and time-based comparisons."
}}
"""

                # Detailed LLM stage logging for evaluation
                rag_logger.info(f"🤖 ===== ANSWER EVALUATION LLM CALL ===== (attempt {attempt + 1}/{max_retries})")
                rag_logger.info(f"🔍 Model: {self.config.get('evaluation_model', 'gpt-4.1-mini-2025-04-14')}")
                rag_logger.info(f"📊 Max tokens: 500")
                rag_logger.info(f"🌡️ Temperature: {self.config.get('evaluation_temperature', 0.1)}")
                rag_logger.info(f"📋 Evaluation prompt length: {len(evaluation_prompt)} characters")
                rag_logger.info(f"📝 Original question: {original_question}")
                rag_logger.info(f"📊 Answer length: {len(answer)} characters")
                rag_logger.info(f"📊 Context chunks count: {len(context_chunks)}")
                if evaluation_context and len(evaluation_context) > 0:
                    rag_logger.info(f"🔄 Iteration memory: {len(evaluation_context)} previous iterations")
                    for iter_info in evaluation_context:
                        iter_num = iter_info.get('iteration', '?')
                        iter_questions = iter_info.get('evaluation', {}).get('follow_up_questions', [])
                        rag_logger.info(f"   - Iteration {iter_num}: {len(iter_questions)} questions asked")
                else:
                    rag_logger.info(f"🔄 No iteration memory (first iteration)")
                rag_logger.info(f"📋 Evaluation prompt preview: {evaluation_prompt[:300]}...")
                
                evaluation_model = self.config.get("evaluation_model", self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507"))
                rag_logger.info(f"🤖 Using {self.llm.provider_name} for evaluation: {evaluation_model}")
                
                start_time = time.time()
                evaluation_text = self.llm.complete(
                    [
                        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    model=evaluation_model,
                    temperature=self.config.get("evaluation_temperature", 0.05),
                    max_tokens=3000,
                    stream=False,
                )
                call_time = time.time() - start_time
                
                rag_logger.info(f"✅ ===== ANSWER EVALUATION LLM RESPONSE ===== (call time: {call_time:.3f}s)")
                
                if not evaluation_text or not evaluation_text.strip():
                    raise ValueError(f"Empty response from LLM on attempt {attempt + 1}")
                
                evaluation_text = evaluation_text.strip()
                rag_logger.info(f"📝 Raw evaluation response length: {len(evaluation_text)} characters")
                rag_logger.info(f"📝 Raw evaluation response (first 300 chars): {evaluation_text[:300]}...")
                
                # Clean up the response (remove any markdown formatting)
                if evaluation_text.startswith("```json"):
                    evaluation_text = evaluation_text[7:]
                    rag_logger.info("🧹 Removed ```json prefix from evaluation response")
                if evaluation_text.endswith("```"):
                    evaluation_text = evaluation_text[:-3]
                    rag_logger.info("🧹 Removed ``` suffix from evaluation response")
                evaluation_text = evaluation_text.strip()
                
                # Try to parse JSON response
                try:
                    evaluation = json.loads(evaluation_text)
                    rag_logger.info(f"✅ ===== EVALUATION PARSING SUCCESSFUL =====")
                    rag_logger.info(f"📊 Completeness score: {evaluation.get('completeness_score', 'unknown')}/10")
                    rag_logger.info(f"📊 Accuracy score: {evaluation.get('accuracy_score', 'unknown')}/10")
                    rag_logger.info(f"📊 Clarity score: {evaluation.get('clarity_score', 'unknown')}/10")
                    rag_logger.info(f"📊 Specificity score: {evaluation.get('specificity_score', 'unknown')}/10")
                    rag_logger.info(f"📊 Overall confidence: {evaluation.get('overall_confidence', 'unknown')}")
                    rag_logger.info(f"🤖 Should iterate: {evaluation.get('should_iterate', 'unknown')}")
                    rag_logger.info(f"💭 Iteration reasoning: {evaluation.get('iteration_decision_reasoning', 'unknown')}")
                    if evaluation.get('follow_up_questions'):
                        rag_logger.info(f"📊 Follow-up questions count: {len(evaluation['follow_up_questions'])}")
                        rag_logger.info(f"📊 Follow-up questions: {evaluation['follow_up_questions'][:3]}...")  # First 3 questions
                    
                    # Validate required fields
                    required_fields = ["completeness_score", "accuracy_score", "clarity_score", "specificity_score", "overall_confidence", "should_iterate", "iteration_decision_reasoning", "follow_up_questions"]
                    for field in required_fields:
                        if field not in evaluation:
                            raise ValueError(f"Missing required field: {field}")
                    
                    rag_logger.info(f"✅ Successfully parsed evaluation JSON on attempt {attempt + 1}")

                    # Validate confidence score makes sense
                    confidence = evaluation.get('overall_confidence', 0.5)
                    should_iterate = evaluation.get('should_iterate', True)

                    # Consistency check: high confidence shouldn't iterate
                    if confidence >= 0.85 and should_iterate:
                        rag_logger.warning(f"⚠️  Confidence {confidence:.2f} but should_iterate=True - overriding to False")
                        evaluation['should_iterate'] = False

                    # Consistency check: low confidence should iterate (unless last iteration)
                    if confidence < 0.60 and not should_iterate:
                        rag_logger.warning(f"⚠️  Confidence {confidence:.2f} but should_iterate=False - may need more iteration")

                    # Log evaluation completion to Logfire with full evaluation details
                    eval_time = time.time() - eval_start
                    if LOGFIRE_AVAILABLE and logfire:
                        logfire.info(
                            "llm.evaluation.complete",
                            overall_confidence=evaluation.get('overall_confidence', 0),
                            should_iterate=evaluation.get('should_iterate', False),
                            completeness_score=evaluation.get('completeness_score', 0),
                            specificity_score=evaluation.get('specificity_score', 0),
                            follow_up_questions_count=len(evaluation.get('follow_up_questions', [])),
                            follow_up_questions=evaluation.get('follow_up_questions', []),
                            evaluation_reasoning=evaluation.get('evaluation_reasoning', ''),
                            iteration_decision_reasoning=evaluation.get('iteration_decision_reasoning', ''),
                            eval_time_ms=int(eval_time * 1000)
                        )

                    return evaluation
                    
                except (json.JSONDecodeError, ValueError) as e:
                    rag_logger.warning(f"⚠️ JSON parsing failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        # Return a default evaluation if all retries fail
                        rag_logger.error(f"❌ All {max_retries} JSON parsing attempts failed, returning default evaluation")
                        return {
                            "completeness_score": 7,
                            "accuracy_score": 7,
                            "clarity_score": 7,
                            "specificity_score": 7,
                            "overall_confidence": 0.7,
                            "should_iterate": False,
                            "iteration_decision_reasoning": "Unable to evaluate - using default scores",
                            "follow_up_questions": ["What additional details can you provide?", "Are there any specific metrics mentioned?"],
                            "evaluation_reasoning": "Default evaluation due to JSON parsing failure after all retries"
                        }
                    else:
                        # Immediate retry for JSON parsing issues
                        rag_logger.info(f"🔄 Immediate retry for JSON parsing (attempt {attempt + 2}/{max_retries})")
                        continue
                        
            except Exception as e:
                rag_logger.error(f"❌ Error in answer evaluation attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Return a default evaluation if all retries fail
                    rag_logger.error(f"❌ All {max_retries} evaluation attempts failed, returning default evaluation")
                    return {
                        "completeness_score": 7,
                        "accuracy_score": 7,
                        "clarity_score": 7,
                        "specificity_score": 7,
                        "overall_confidence": 0.7,
                        "should_iterate": False,
                        "iteration_decision_reasoning": "Unable to evaluate - using default scores",
                        "follow_up_questions": ["What additional details can you provide?", "Are there any specific metrics mentioned?"],
                        "evaluation_reasoning": "Default evaluation due to API failure after all retries"
                    }
                else:
                    # Immediate retry for API errors
                    rag_logger.info(f"🔄 Immediate retry for API error (attempt {attempt + 2}/{max_retries})")
                    continue
