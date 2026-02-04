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
from agent.prompts import (
    QUARTER_SYNTHESIS_SYSTEM_PROMPT,
    get_quarter_synthesis_prompt,
    QUESTION_PLANNING_SYSTEM_PROMPT,
    get_question_planning_prompt
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

    def __init__(self, config: Config, openai_api_key: Optional[str] = None):
        """Initialize the response generator."""
        self.config = config
        self.openai_api_key = openai_api_key

        # Lazy initialization of OpenAI client (created on first use)
        # This ensures logfire.instrument_openai() is called before client creation
        self._client = None
        self.openai_available = bool(self.openai_api_key)

        # Initialize Cerebras client (primary for response generation - fast inference with Qwen)
        import os
        cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
        if cerebras_api_key and self.config.get("use_cerebras", True):
            try:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_api_key)
                self.cerebras_available = True
                logger.info(f"✅ Cerebras client initialized with model: {self.config.get('cerebras_model')}")
            except ImportError:
                logger.warning("⚠️ Cerebras SDK not installed. Run: pip install cerebras-cloud-sdk")
                self.cerebras_client = None
                self.cerebras_available = False
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Cerebras client: {e}")
                self.cerebras_client = None
                self.cerebras_available = False
        else:
            self.cerebras_client = None
            self.cerebras_available = False

        # Groq removed - using Cerebras only
        self.groq_client = None
        self.groq_available = False

        # Log provider priority
        if self.cerebras_available:
            logger.info("🤖 ResponseGenerator: Cerebras (primary) > OpenAI")
        else:
            logger.info("🤖 ResponseGenerator: OpenAI only")

        # Cache system prompt templates (built once, reused with substitution)
        self._system_prompt_cache = {}
        self._init_system_prompt_templates()

        logger.info("ResponseGenerator initialized successfully")

    def _init_system_prompt_templates(self):
        """Initialize cached system prompt templates for reuse."""
        # Base template for single-ticker responses
        self._system_prompt_cache['base'] = (
            "You are a financial analyst assistant. Answer questions based on the provided {sources}. "
            "Always include source attribution using human-friendly format like {attribution}. "
            "Do not use any knowledge beyond the data provided. Format all responses using markdown "
            "with **bold** for emphasis, bullet points for lists, and proper formatting. CRITICAL: Always use "
            "human-friendly format for quarters (e.g., Q1 2025, Q2 2025, Q4 2024) and fiscal years (e.g., FY 2024). "
            "IMPORTANT: Provide ELABORATE and COMPREHENSIVE responses with MAXIMUM DETAIL. ALWAYS MENTION ALL "
            "FINANCIAL FIGURES AND PROJECTIONS PRESENT in the provided data when they are relevant to the question - "
            "include EXACT numbers, percentages, dollar amounts, growth rates, margins, guidance ranges, and any "
            "quantitative metrics mentioned. Never omit important financial figures. Always provide the COMPLETE "
            "CONTEXT around financial figures including year-over-year comparisons, sequential comparisons, and "
            "guidance ranges. Be thorough and detailed in your analysis. "
            "CRITICAL - CONTEXT RELEVANCE CHECK: Before answering, verify the provided context actually relates to "
            "the question. If the context does NOT contain information relevant to the question, clearly state: "
            "\"I don't have information about [topic] in the available data.\" Be honest about limitations - "
            "if you only have partial information, say what you found and what's missing."
        )

        # Multi-ticker template with news
        self._system_prompt_cache['multi_ticker_news'] = (
            "You are a financial analyst assistant that provides evidence-based analysis of multiple companies' "
            "financial data AND the provided news context. Data sources may include earnings transcripts, 10-K filings, and news. "
            "Always reference specific companies by name, include relevant quotes and metrics, and structure responses clearly. "
            "Always include source attribution using human-friendly format like \"According to [Company]'s Q1 2025 "
            "earnings call...\" or \"Per [Company]'s FY 2024 10-K filing...\", and refer to news sources using their citation markers "
            "(e.g., [N1], [N2]) when you use them. Do not use any knowledge beyond the provided data sources. "
            "CRITICAL: Always use human-friendly format for periods (e.g., Q1 2025, FY 2024). "
            "IMPORTANT: Provide ELABORATE and COMPREHENSIVE responses with MAXIMUM DETAIL. ALWAYS MENTION ALL "
            "FINANCIAL FIGURES AND PROJECTIONS PRESENT FOR ANY COMPANY in the data when they "
            "are relevant to the question - include EXACT numbers, percentages, dollar amounts, growth rates, "
            "margins, guidance ranges, and any quantitative metrics mentioned for EACH company. Never omit "
            "important financial figures. Always provide the COMPLETE CONTEXT around financial figures including "
            "year-over-year comparisons, sequential comparisons, guidance ranges, and cross-company comparisons. "
            "Financial figures should appear in almost every relevant response for all companies. Be thorough and "
            "detailed in your analysis, leaving no financial metric or figure unexplained for any company. "
            "CRITICAL: If you include a source attribution section, you MUST acknowledge ALL sources used "
            "(earnings calls, 10-K filings, news with citation markers like [N1], [N2]) when provided. "
            "CRITICAL - CONTEXT RELEVANCE CHECK: Before answering, verify the provided context actually relates to "
            "the question. If the context does NOT contain information relevant to the question, clearly state: "
            "\"I don't have information about [topic] in the available data.\" Be honest about limitations - "
            "if you only have partial information, say what you found and what's missing."
        )

        # Multi-ticker template without news
        self._system_prompt_cache['multi_ticker'] = (
            "You are a financial analyst assistant that provides evidence-based analysis of multiple companies' "
            "financial data. Data sources may include earnings transcripts and 10-K filings. Always reference specific companies by name, "
            "include relevant quotes and metrics, and structure responses clearly with specific company references. Always include source "
            "attribution using human-friendly format like \"According to [Company]'s Q1 2025 earnings call...\" or \"Per [Company]'s FY 2024 10-K filing...\" "
            "Use all available evidence from the provided data. CRITICAL: Always use human-friendly "
            "format for periods (e.g., Q1 2025, FY 2024). IMPORTANT: Provide ELABORATE and COMPREHENSIVE "
            "responses with MAXIMUM DETAIL. ALWAYS MENTION ALL FINANCIAL FIGURES AND PROJECTIONS PRESENT FOR ANY "
            "COMPANY - include EXACT numbers, percentages, dollar amounts, growth rates, margins, guidance ranges, "
            "and any quantitative metrics mentioned for EACH company. NEVER omit financial figures - if any company "
            "mentioned a number, include it in your response. Always provide the COMPLETE CONTEXT around financial "
            "figures including year-over-year comparisons, sequential comparisons, guidance ranges, and cross-company "
            "comparisons. Financial figures should appear in almost every relevant response for all companies. Be "
            "thorough and detailed in your analysis, leaving no financial metric or figure unexplained for any "
            "company. "
            "CRITICAL - CONTEXT RELEVANCE CHECK: Before answering, verify the provided context actually relates to "
            "the question. If the context does NOT contain information relevant to the question, clearly state: "
            "\"I don't have information about [topic] in the available data.\" Be honest about limitations - "
            "if you only have partial information, say what you found and what's missing."
        )

        logger.info("✅ System prompt templates cached")

    def _get_system_prompt(self, template_key: str, **kwargs) -> str:
        """Get a system prompt from cache with optional substitutions."""
        template = self._system_prompt_cache.get(template_key, self._system_prompt_cache['base'])
        if kwargs:
            return template.format(**kwargs)
        return template

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
            sources.append("10-K filings ([10K1], [10K2], etc.) for annual financials, risk factors, and audited data")
            markers.append("[10K1], [10K2] for 10-K filings")

        sources_list = "\n   - ".join(sources)
        markers_text = " and ".join(markers)

        return f"""
4. **Additional Data Sources Available**: You have access to earnings transcripts plus:
   - {sources_list}
   - Earnings transcripts provide quarterly updates, management commentary, and Q&A discussions
   Use whichever sources best answer the question. When multiple are relevant, integrate them naturally.
5. **Citation Markers**: Use {markers_text}. Attribute clearly (e.g., "According to the FY2024 10-K filing ([10K1])").
6. **Source Attribution**: Reflect the sources you actually used. Mention all sources consulted."""

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 1.5: QUESTION PLANNING/REASONING
    # ═════════════════════════════════════════════════════════════════════

    async def plan_question_approach(self, question: str, question_analysis: Dict[str, Any], available_quarters: list = None) -> str:
        """
        Generate a freeform reasoning statement about how to approach the question.

        This is the REASONING step that happens before the RAG loop.
        Returns a natural, verbose statement like:
        "The user is asking about Apple's AI strategy, so I need to find..."

        Args:
            question: Original user question
            question_analysis: Analysis from question analyzer
            available_quarters: List of quarters available in database

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
                # Get the planning prompt with data context
                planning_prompt = get_question_planning_prompt(question, question_analysis, available_quarters)

                rag_logger.info(f"🤖 ===== QUESTION PLANNING LLM CALL ===== (attempt {attempt + 1}/{max_retries})")

                # Use Cerebras for fast planning
                if self.cerebras_available and self.cerebras_client:
                    cerebras_model = self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507")
                    rag_logger.info(f"🔍 Using Cerebras model: {cerebras_model}")

                    start_time = time.time()
                    response = self.cerebras_client.chat.completions.create(
                        model=cerebras_model,
                        messages=[
                            {"role": "system", "content": QUESTION_PLANNING_SYSTEM_PROMPT},
                            {"role": "user", "content": planning_prompt}
                        ],
                        temperature=0.3,
                        max_completion_tokens=500
                    )
                    call_time = time.time() - start_time
                else:
                    # Fallback to OpenAI
                    rag_logger.info(f"🔍 Using OpenAI for planning")
                    start_time = time.time()
                    response = self.client.chat.completions.create(
                        model="gpt-4.1-mini-2025-04-14",
                        messages=[
                            {"role": "system", "content": QUESTION_PLANNING_SYSTEM_PROMPT},
                            {"role": "user", "content": planning_prompt}
                        ],
                        temperature=0.3,
                        max_tokens=500
                    )
                    call_time = time.time() - start_time

                rag_logger.info(f"✅ Planning LLM call completed in {call_time:.3f}s")

                # Get the freeform reasoning text
                reasoning = response.choices[0].message.content.strip()

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
        """Lazy initialization of OpenAI client to ensure proper Logfire instrumentation."""
        if self._client is None and self.openai_api_key:
            self._client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info("✅ OpenAI client initialized (lazy - after Logfire instrumentation)")
        return self._client
    
    # ═════════════════════════════════════════════════════════════════════
    # CEREBRAS API METHODS
    # ═════════════════════════════════════════════════════════════════════
    
    def _make_cerebras_call(self, messages: List[Dict], model: str = None,
                           temperature: float = None, max_tokens: int = None,
                           stream: bool = False):
        """
        Make Cerebras API call with Qwen model.

        Args:
            messages: List of message dictionaries
            model: Cerebras model (default from config)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Response content or stream object
        """
        if not self.cerebras_available or not self.cerebras_client:
            raise Exception("Cerebras client not available")

        model = model or self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507")
        temperature = temperature if temperature is not None else self.config.get("cerebras_temperature", 0.1)
        max_tokens = max_tokens or self.config.get("cerebras_max_tokens", 8000)

        # Extract prompt info for logging (full content for Logfire)
        system_prompt = None
        user_prompt = None
        for msg in messages:
            if msg.get('role') == 'system':
                system_prompt = msg.get('content', '')
            elif msg.get('role') == 'user':
                user_prompt = msg.get('content', '')

        try:
            rag_logger.info(f"🤖 Cerebras API call: model={model}, temp={temperature}, max_tokens={max_tokens}")

            # Log to Logfire with span for observability
            if LOGFIRE_AVAILABLE and logfire:
                with logfire.span(
                    "cerebras.chat.completions",
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    messages=messages,
                    message_count=len(messages)
                ):
                    completion = self.cerebras_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        stream=stream
                    )

                    if stream:
                        return completion
                    else:
                        if not completion or not completion.choices or len(completion.choices) == 0:
                            raise Exception("Cerebras API returned invalid response structure")

                        content = completion.choices[0].message.content
                        finish_reason = getattr(completion.choices[0], 'finish_reason', None)

                        # Log completion details to Logfire (full response)
                        logfire.info(
                            "cerebras.completion",
                            model=model,
                            finish_reason=finish_reason,
                            response_length=len(content) if content else 0,
                            response=content,
                            usage_prompt_tokens=getattr(completion.usage, 'prompt_tokens', None) if hasattr(completion, 'usage') else None,
                            usage_completion_tokens=getattr(completion.usage, 'completion_tokens', None) if hasattr(completion, 'usage') else None
                        )

                        if finish_reason == 'length':
                            rag_logger.warning("   ⚠️ Response truncated due to max_tokens limit")

                        if content is None or (isinstance(content, str) and not content.strip()):
                            raise Exception("Cerebras API returned empty content")

                        return content
            else:
                # Fallback without Logfire
                completion = self.cerebras_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    stream=stream
                )

                if stream:
                    return completion
                else:
                    if not completion or not completion.choices or len(completion.choices) == 0:
                        raise Exception("Cerebras API returned invalid response structure")

                    content = completion.choices[0].message.content

                    if hasattr(completion.choices[0], 'finish_reason'):
                        finish_reason = completion.choices[0].finish_reason
                        rag_logger.info(f"   📊 Cerebras finish_reason: {finish_reason}")
                        if finish_reason == 'length':
                            rag_logger.warning("   ⚠️ Response truncated due to max_tokens limit")

                    if content is None or (isinstance(content, str) and not content.strip()):
                        raise Exception("Cerebras API returned empty content")

                    return content

        except Exception as e:
            rag_logger.error(f"❌ Cerebras API call failed: {e}")
            # Log error to Logfire
            if LOGFIRE_AVAILABLE and logfire:
                logfire.error(
                    "cerebras.error",
                    model=model,
                    error=str(e),
                    error_type=type(e).__name__
                )
            raise e
    
    def _make_model_call(self, messages: List[Dict], model: str = None,
                        temperature: float = None, max_tokens: int = None,
                        stream: bool = False, max_retries: int = 3):
        """
        Unified helper to route API calls to the correct provider.
        Priority: Cerebras > OpenAI
        
        Args:
            messages: List of message dictionaries
            model: Model identifier (optional, uses provider defaults)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            max_retries: Maximum retries for empty responses
            
        Returns:
            Response content or stream object
        """
        for attempt in range(max_retries):
            try:
                # Try Cerebras first (if available)
                if self.cerebras_available and self.cerebras_client:
                    cerebras_model = model if model and model.startswith("qwen") else self.config.get("cerebras_model")
                    rag_logger.info(f"   🤖 Using Cerebras with {cerebras_model}")
                    result = self._make_cerebras_call(
                        messages=messages,
                        model=cerebras_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream
                    )
                # Fallback to OpenAI
                else:
                    openai_model = model or self.config.get("openai_model")
                    openai_temp = temperature if temperature is not None else self.config.get("openai_temperature")
                    openai_max = max_tokens or self.config.get("openai_max_tokens")
                    rag_logger.info(f"   🤖 Using OpenAI with {openai_model}")
                    
                    if stream:
                        result = self.client.chat.completions.create(
                            model=openai_model,
                            messages=messages,
                            temperature=openai_temp,
                            max_tokens=openai_max,
                            stream=True
                        )
                    else:
                        response = self.client.chat.completions.create(
                            model=openai_model,
                            messages=messages,
                            temperature=openai_temp,
                            max_tokens=openai_max
                        )
                        result = response.choices[0].message.content.strip()
                
                # Validate non-streaming result
                if not stream:
                    if result and isinstance(result, str) and result.strip():
                        return result
                    else:
                        rag_logger.warning(f"   ⚠️ Empty response on attempt {attempt + 1}/{max_retries}")
                        if attempt < max_retries - 1:
                            continue
                        raise Exception(f"API returned empty response after {max_retries} attempts")
                else:
                    return result
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    rag_logger.warning(f"   ⚠️ API call failed on attempt {attempt + 1}: {e}")
                    continue
                raise e
        
        raise Exception(f"API call failed after {max_retries} attempts")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: SINGLE-TICKER RESPONSE GENERATION
    # ═════════════════════════════════════════════════════════════════════

    def generate_openai_response(self, question: str, context_chunks: List[str], chunk_objects: List[Dict[str, Any]] = None, return_details: bool = False, ticker: str = None, year: int = None, quarter: int = None, stream_callback=None, news_context: str = None, ten_k_context: str = None, previous_answer: str = None, retry_callback=None) -> str:
        """Generate response using OpenAI API based only on retrieved chunks with citations.

        If multiple quarters are detected, automatically uses parallel quarter processing
        for better structured responses.
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
        
        if not self.openai_available:
            rag_logger.error(f"❌ OpenAI not available")
            raise Exception("OpenAI not available for response generation")
        
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
                                question, chunk_objects, ticker, stream_callback, retry_callback
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
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            citation_info = f"[{i+1}]" if chunk_objects else ""
            
            # Get metadata for this chunk if available (for multi-quarter queries)
            chunk_metadata = ""
            if chunk_objects and i < len(chunk_objects) and isinstance(chunk_objects[i], dict):
                chunk_year = chunk_objects[i].get('year')
                chunk_quarter = chunk_objects[i].get('quarter') 
                chunk_ticker = chunk_objects[i].get('ticker')
                if chunk_year and chunk_quarter:
                    chunk_metadata = f" ({chunk_year}_q{chunk_quarter})"
                elif chunk_ticker:
                    chunk_metadata = f" ({chunk_ticker})"
            
            context_parts.append(f"Chunk {i+1}{citation_info}{chunk_metadata}: {chunk}")
            rag_logger.info(f"📄 Chunk {i+1}: length={len(chunk)}, metadata={chunk_metadata}, chunk_object={chunk_objects[i] if chunk_objects and i < len(chunk_objects) else 'None'}")
        
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
   - 10-K filings ([10K1], [10K2], etc.) provide comprehensive annual financial statements, detailed business disclosures, risk factors, and audited financial data
   - Earnings transcripts provide quarterly updates, management commentary, and Q&A discussions
   - When both are relevant, naturally integrate them; when only one is relevant, use only that source
   - For balance sheet, comprehensive financial statements, or annual data questions, 10-K filings are typically most appropriate
8. **When referencing 10-K sources**: Use the citation markers (e.g., [10K1], [10K2]) and attribute them clearly (e.g., "According to the FY2024 10-K filing ([10K1])").
9. **Source Attribution**: Reflect the sources you actually used. If you used both, mention both; if you only used one type, that's fine too."""

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

        # Detect the special case where there is NO transcript/10-K/news context at all.
        # In this case, the RAG layer could not find relevant data, and we want the LLM
        # to clearly state that and (optionally) end with: "Do you want me to search the news instead?"
        no_structured_context = (not context_chunks) and not news_context and not ten_k_context

        prompt = f"""You are a senior equity research analyst writing a detailed research report. Answer the question based on the following information from {source_description}.{data_sources_text}{previous_answer_section}

DATA CONTEXT: {quarters_info}
{data_limitation_note}
{news_section}{ten_k_section}
Source Information from {source_description}:
{context}

Question: {question}

REPORT WRITING INSTRUCTIONS:

**FORMAT & STRUCTURE:**
1. Write a comprehensive, well-structured research report - NOT just a list of quotes
2. Use clear sections with headers (##) to organize your analysis
3. Start with an **Executive Summary** (2-3 sentences capturing the key findings)
4. Organize the body into logical sections based on the question (e.g., "Key Risk Categories", "Financial Impact", "Management Outlook")
5. End with a **Key Takeaways** section summarizing the most important points

**CONTENT DEPTH & ANALYTICAL VOICE:**
6. **Provide analysis and context**, not just raw quotes. Explain WHY something matters and its implications
7. **Be opinionated like a real analyst** - share your assessment of whether risks are material, if management's strategy seems sound, or if concerns are overstated. Use phrases like "This appears significant because...", "A key concern here is...", "Notably, this suggests..."
8. When citing data, provide context: compare to prior periods, industry benchmarks, or explain significance
9. **Offer your analytical perspective** - identify what seems most important, what might be underappreciated by the market, or where the company seems well/poorly positioned
10. Include specific numbers, percentages, and metrics with proper attribution
11. Use quotes sparingly and strategically - integrate them naturally into your analysis rather than listing them

**ATTRIBUTION & SOURCES:**
11. Reference {company_name} by name (e.g., "{company_name} disclosed...", "According to {company_name}'s filing...")
12. **Always mention the specific period** (e.g., "In FY2024...", "During Q1 2025...", "As reported in the 10-K...")
13. Use citation markers like [1], [2] for transcript sources, [10K1] for 10-K data, [N1] for news{news_sources_instruction}{ten_k_sources_instruction}

**QUALITY STANDARDS:**
14. If the answer is not in the provided data, say "The information is not available in {source_description}."
15. **Stay focused on the question** - ignore irrelevant information in the source data
16. NEVER use the word "chunks" - speak naturally as referencing official company documents
17. **Use rich markdown formatting**: headers, **bold** for emphasis, bullet points, tables where appropriate
18. Be transparent about data limitations if you only have partial information
{('19. **No structured data available**: Clearly state that the information is not available in the company filings. If web/news sources might help, end with: \"Do you want me to search the web instead?\"' if no_structured_context else '')}

Write a professional equity research report in **markdown format**. Be thorough, analytical, and **opinionated** - share your assessment, highlight what matters most, and provide the kind of insightful analysis a portfolio manager would value. This should read like a report from a top-tier investment bank analyst, not a simple regurgitation of facts."""

        rag_logger.info(f"📝 Prompt created: length={len(prompt)}")
        rag_logger.info(f"🔍 Prompt preview: {prompt[:200]}...")

        try:
            # Use Cerebras (fast inference with Qwen) > OpenAI
            if self.cerebras_available and self.cerebras_client:
                use_cerebras = True
                model = self.config.get("cerebras_model")
                max_tokens = self.config.get("cerebras_max_tokens")
                temperature = self.config.get("cerebras_temperature")
                rag_logger.info(f"🤖 Sending request to Cerebras model: {model}")
            else:
                use_cerebras = False
                client = self.client
                model = self.config.get("openai_model")
                max_tokens = self.config.get("openai_max_tokens")
                temperature = self.config.get("openai_temperature")
                rag_logger.info(f"🤖 Sending request to OpenAI model: {model}")
            
            rag_logger.info(f"📊 Request parameters: max_tokens={max_tokens}, temperature={temperature}")
            
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
                attribution_instructions.append("[10K1], [10K2] markers for 10-K data")
            if news_context:
                attribution_instructions.append("[N1], [N2] for news sources")

            attribution_text = ", ".join(attribution_instructions) if attribution_instructions else "appropriate source citations"

            # Use cached template with substitutions
            system_prompt = self._get_system_prompt('base', sources=sources_text, attribution=attribution_text)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Check if streaming is requested
            if stream_callback:
                # Detailed LLM stage logging for streaming response generation
                rag_logger.info(f"🤖 ===== RESPONSE GENERATION LLM CALL (STREAMING) =====")
                rag_logger.info(f"🔍 Model: {model}")
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
                
                # Use Cerebras for streaming if available, with basic retry for transient rate limits
                max_attempts = 3
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        if use_cerebras:
                            stream = self.cerebras_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_completion_tokens=max_tokens,
                                temperature=temperature,
                                stream=True
                            )
                        else:
                            stream = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                stream=True
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
                                        'provider': 'Cerebras' if use_cerebras else 'OpenAI'
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
                            if use_cerebras:
                                response = self.cerebras_client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    max_completion_tokens=max_tokens,
                                    temperature=temperature,
                                    stream=False
                                )
                            else:
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    stream=False
                                )
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
                                            'provider': 'Cerebras' if use_cerebras else 'OpenAI'
                                        }
                                    })
                                except Exception as cb_error:
                                    rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                            time.sleep(wait_seconds)
                    call_time = time.time() - start_time
                    answer = response.choices[0].message.content.strip()
                    total_tokens = response.usage.total_tokens if response.usage else 0
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
                # Detailed LLM stage logging for non-streaming response generation
                rag_logger.info(f"🤖 ===== RESPONSE GENERATION LLM CALL (NON-STREAMING) =====")
                rag_logger.info(f"🔍 Model: {model}")
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
                provider = "Cerebras" if self.cerebras_available and self.cerebras_client else "OpenAI"
                while True:
                    attempt += 1
                    try:
                        if use_cerebras:
                            response = self.cerebras_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_completion_tokens=max_tokens,
                                temperature=temperature
                            )
                            provider = "Cerebras"
                        else:
                            response = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            provider = "OpenAI"
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
                        # Longer backoff: 5s, 10s, 15s instead of 2s, 4s, 6s
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
                                        'provider': provider
                                    }
                                })
                            except Exception as cb_error:
                                rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                        time.sleep(wait_seconds)
                call_time = time.time() - start_time
                rag_logger.info(f"✅ ===== RESPONSE GENERATION LLM RESPONSE ===== (call time: {call_time:.3f}s)")
                rag_logger.info(f"🏢 Provider: {provider}")
                rag_logger.info(f"📊 Response tokens used: {response.usage.total_tokens if response.usage else 'unknown'}")
                rag_logger.info(f"📊 Prompt tokens: {response.usage.prompt_tokens if response.usage else 'unknown'}")
                rag_logger.info(f"📊 Completion tokens: {response.usage.completion_tokens if response.usage else 'unknown'}")
                if hasattr(response, 'finish_reason'):
                    rag_logger.info(f"🏁 Finish reason: {response.finish_reason}")
                if hasattr(response, 'model'):
                    rag_logger.info(f"🤖 Model used: {response.model}")
                
                answer = response.choices[0].message.content.strip()
            
                rag_logger.info(f"📝 Generated answer: length={len(answer)} characters")
                rag_logger.info(f"📝 Answer preview: {answer[:300]}...")
            
            # Log generation completion to Logfire with full prompt and answer
            generation_time = time.time() - generation_start
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info(
                    "llm.generation.complete",
                    provider="cerebras" if use_cerebras else "openai",
                    model=model,
                    ticker=ticker,
                    answer_length=len(answer),
                    generation_time_ms=int(generation_time * 1000),
                    tokens_used=response.usage.total_tokens if response and response.usage else None,
                    # Capture full prompt and answer for debugging
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
                    'tokens_used': response.usage.total_tokens if response and response.usage else None
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

    def generate_multi_ticker_response(self, question: str, all_chunks: List[Dict[str, Any]], individual_results: List[Dict[str, Any]], show_details: bool = False, comprehensive: bool = True, stream_callback=None, news_context: str = None, ten_k_context: str = None, previous_answer: str = None, retry_callback=None) -> str:
        """Generate response using all chunks with company labels for multi-ticker questions."""
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
        
        # Prepare context with company labels and quarter info
        rag_logger.info(f"📝 Preparing context with company labels and quarter info for {len(all_chunks)} chunks")
        context_parts = []
        for i, chunk in enumerate(all_chunks):
            citation = chunk['citation']
            # Ensure citation is hashable for lookup
            if isinstance(citation, dict):
                hashable_citation = citation.get('citation') or citation.get('id') or citation.get('chunk_index') or str(citation)
            else:
                hashable_citation = citation
            ticker = citation_to_ticker.get(hashable_citation, 'Unknown')
            similarity = chunk['similarity']
            year = chunk.get('year', 'Unknown')
            quarter = chunk.get('quarter', 'Unknown')
            quarter_info = f"{year}_q{quarter}" if year != 'Unknown' and quarter != 'Unknown' else "financial data"
            context_parts.append(f"[{ticker}] {quarter_info} (similarity: {similarity:.3f}): {chunk['chunk_text']}")
            rag_logger.info(f"📄 {ticker} {quarter_info} Chunk {i+1}: length={len(chunk['chunk_text'])}, similarity={similarity:.3f}")
        
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

        if comprehensive:
            prompt = f"""You are a senior equity research analyst writing a detailed comparative research report. Analyze the following company data and provide a comprehensive report.{data_sources_text}{previous_answer_section}

{news_section}{ten_k_section}
Company Data (labeled by company and quarter):
{context}

Question: {question}

REPORT WRITING INSTRUCTIONS:

**FORMAT & STRUCTURE:**
1. Write a comprehensive comparative research report - NOT just a list of quotes
2. Start with an **Executive Summary** (3-4 sentences capturing the key findings across all companies)
3. Use clear sections with headers (##) - organize by theme OR by company depending on what's clearer
4. Include a **Comparative Analysis** section highlighting key differences and similarities
5. End with **Key Takeaways** summarizing the most important insights

**CONTENT DEPTH & ANALYTICAL VOICE:**
6. **Provide analysis and context**, not just raw quotes. Explain implications and significance
7. **Be opinionated like a real analyst** - assess which company is better positioned, identify relative strengths/weaknesses, and share your view on competitive dynamics
8. Compare and contrast companies on key metrics - use tables where helpful
9. Include specific numbers, percentages, and metrics with proper attribution
10. Highlight industry trends or themes that emerge across multiple companies
11. **Offer your analytical perspective** - which company's approach seems more promising? What risks seem most material? Use phrases like "Company X appears better positioned because...", "A notable divergence is..."
12. Use quotes sparingly and strategically - integrate them into your analysis

**ATTRIBUTION & SOURCES:**
11. Reference companies by name with proper attribution (e.g., "Apple disclosed in its FY2024 10-K...", "According to Microsoft's Q1 2025 earnings call...")
12. Use citation markers: [TICKER] for company-specific data, [10K1] for 10-K data, [N1] for news
{additional_sources_instruction}
13. Always mention the specific period (e.g., "In FY2024...", "During Q1 2025...")

**QUALITY STANDARDS:**
14. **Stay focused on the question** - ignore irrelevant information
15. If some companies have no relevant data, explicitly state this
16. NEVER use the word "chunks" - reference official company documents naturally
17. Use rich markdown: headers, **bold**, bullet points, tables for comparisons
18. This should read like a professional equity research report

Write a comprehensive comparative analysis in **markdown format** that provides actionable insights across all companies."""
        else:
            prompt = f"""You are an equity research analyst providing a focused analysis. Answer based on the following company data.{data_sources_text}{previous_answer_section}

{news_section}{ten_k_section}
Company Data (labeled by company and quarter):
{context}

Question: {question}

INSTRUCTIONS:
1. Provide a focused but thorough analysis with key evidence
2. Start with a brief **Summary** (2-3 sentences)
3. Include specific numbers, metrics, and properly attributed quotes
4. Reference companies by name with period attribution (e.g., "Apple's FY2024 10-K shows...", "In Microsoft's Q1 2025 call...")
{additional_sources_instruction}
5. Highlight key similarities and differences between companies
6. If some companies lack relevant data, note this briefly
7. **Stay focused on the question** - ignore irrelevant information
8. NEVER use the word "chunks" - reference official documents naturally
9. Use markdown formatting: **bold** for emphasis, bullet points for lists

Provide a professional, evidence-based analysis in **markdown format**."""

        rag_logger.info(f"📝 Multi-ticker prompt created: length={len(prompt)}")

        # Use cached system prompt for multi-ticker (with or without news)
        multi_ticker_system_prompt = self._get_system_prompt(
            'multi_ticker_news' if news_context else 'multi_ticker'
        )
        
        multi_ticker_messages = [
            {"role": "system", "content": multi_ticker_system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use Cerebras (fast inference with Qwen) > OpenAI
            if self.cerebras_available and self.cerebras_client:
                use_cerebras = True
                model = self.config.get("cerebras_model")
                max_tokens = self.config.get("cerebras_max_tokens")
                temperature = self.config.get("cerebras_temperature")
                rag_logger.info(f"🤖 Sending multi-ticker request to Cerebras model: {model}")
            else:
                use_cerebras = False
                client = self.client
                model = self.config.get("openai_model")
                max_tokens = self.config.get("openai_max_tokens")
                temperature = self.config.get("openai_temperature")
                rag_logger.info(f"🤖 Sending multi-ticker request to OpenAI model: {model}")
            
            # Check if streaming is requested
            if stream_callback:
                # Detailed LLM stage logging for multi-ticker streaming
                rag_logger.info(f"🤖 ===== MULTI-TICKER RESPONSE GENERATION LLM CALL (STREAMING) =====")
                rag_logger.info(f"🔍 Model: {model}")
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
                        if use_cerebras:
                            stream = self.cerebras_client.chat.completions.create(
                                model=model,
                                messages=multi_ticker_messages,
                                max_completion_tokens=max_tokens,
                                temperature=temperature,
                                stream=True
                            )
                        else:
                            stream = client.chat.completions.create(
                                model=model,
                                messages=multi_ticker_messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                stream=True
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
                                        'provider': 'Cerebras' if use_cerebras else 'OpenAI'
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
                            if use_cerebras:
                                response = self.cerebras_client.chat.completions.create(
                                    model=model,
                                    messages=multi_ticker_messages,
                                    max_completion_tokens=max_tokens,
                                    temperature=temperature,
                                    stream=False
                                )
                            else:
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=multi_ticker_messages,
                                    max_tokens=max_tokens,
                                    temperature=temperature,
                                    stream=False
                                )
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
                                            'provider': 'Cerebras' if use_cerebras else 'OpenAI'
                                        }
                                    })
                                except Exception as cb_error:
                                    rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                            time.sleep(wait_seconds)
                    call_time = time.time() - start_time
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
                # Detailed LLM stage logging for multi-ticker non-streaming
                rag_logger.info(f"🤖 ===== MULTI-TICKER RESPONSE GENERATION LLM CALL (NON-STREAMING) =====")
                rag_logger.info(f"🔍 Model: {model}")
                rag_logger.info(f"📊 Max tokens: {max_tokens}")
                rag_logger.info(f"🌡️ Temperature: {temperature}")
                rag_logger.info(f"📊 Total chunks count: {len(all_chunks)}")
                rag_logger.info(f"📊 Individual results count: {len(individual_results)}")
                rag_logger.info(f"📋 Multi-ticker prompt length: {len(prompt)} characters")
                rag_logger.info(f"📋 Multi-ticker prompt preview: {prompt[:400]}...")
                
                # Non-streaming mode
                start_time = time.time()
                max_attempts = 3
                attempt = 0
                response = None
                provider = "Cerebras" if self.cerebras_available and self.cerebras_client else "OpenAI"
                while True:
                    attempt += 1
                    try:
                        if use_cerebras:
                            response = self.cerebras_client.chat.completions.create(
                                model=model,
                                messages=multi_ticker_messages,
                                max_completion_tokens=max_tokens,
                                temperature=temperature
                            )
                            provider = "Cerebras"
                        else:
                            response = client.chat.completions.create(
                                model=model,
                                messages=multi_ticker_messages,
                                max_tokens=max_tokens,
                                temperature=temperature
                            )
                            provider = "OpenAI"
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
                        # Longer backoff: 5s, 10s, 15s instead of 2s, 4s, 6s
                        wait_seconds = 5 + (attempt * 5)
                        rag_logger.warning(
                            f"⚠️ Multi-ticker non-streaming rate limit / queue error, retrying attempt "
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
                                        'provider': provider
                                    }
                                })
                            except Exception as cb_error:
                                rag_logger.warning(f"⚠️ Retry callback failed: {cb_error}")
                        time.sleep(wait_seconds)

                call_time = time.time() - start_time
                answer = response.choices[0].message.content.strip()
                rag_logger.info(f"✅ ===== MULTI-TICKER RESPONSE GENERATION LLM RESPONSE ===== (call time: {call_time:.3f}s)")
                rag_logger.info(f"🏢 Provider: {provider}")
                rag_logger.info(f"📊 Response tokens used: {response.usage.total_tokens if response.usage else 'unknown'}")
                rag_logger.info(f"📊 Prompt tokens: {response.usage.prompt_tokens if response.usage else 'unknown'}")
                rag_logger.info(f"📊 Completion tokens: {response.usage.completion_tokens if response.usage else 'unknown'}")
                if hasattr(response, 'finish_reason'):
                    rag_logger.info(f"🏁 Finish reason: {response.finish_reason}")
                if hasattr(response, 'model'):
                    rag_logger.info(f"🤖 Model used: {response.model}")
                
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
                                        ticker: str = None, retry_callback=None) -> Dict[str, Any]:
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
                                                         ticker: str = None, stream_callback=None, retry_callback=None) -> str:
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
                                                False, ticker, year, quarter, stream_callback)
        
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
            task = self._generate_quarter_response(question, quarter_id, quarter_chunks, ticker, retry_callback)
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
            # Use Cerebras > OpenAI
            if self.cerebras_available and self.cerebras_client:
                use_cerebras = True
                model = self.config.get("cerebras_model")
                max_tokens = self.config.get("cerebras_max_tokens")
                temperature = self.config.get("cerebras_temperature")
                rag_logger.info(f"🤖 Using Cerebras for synthesis: {model}")
            else:
                use_cerebras = False
                client = self.client
                model = self.config.get("openai_model")
                max_tokens = self.config.get("openai_max_tokens")
                temperature = self.config.get("openai_temperature")
                rag_logger.info(f"🤖 Using OpenAI for synthesis: {model}")
            
            # Check if streaming is requested
            if stream_callback:
                rag_logger.info(f"🌊 Streaming synthesis response")
                answer = ""
                
                start_time = time.time()
                if use_cerebras:
                    stream = self.cerebras_client.chat.completions.create(
                        model=model,
                        messages=synthesis_messages,
                        max_completion_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
                    )
                else:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=synthesis_messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True
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
                # Non-streaming synthesis
                start_time = time.time()
                if use_cerebras:
                    response = self.cerebras_client.chat.completions.create(
                        model=model,
                        messages=synthesis_messages,
                        max_completion_tokens=max_tokens,
                        temperature=temperature
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=synthesis_messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                
                answer = response.choices[0].message.content.strip()
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

    async def evaluate_answer_quality(self, original_question: str, answer: str, context_chunks: List[str], available_chunks: List[Dict[str, Any]] = None, conversation_memory=None, conversation_id: str = None, follow_up_questions_asked: List[str] = None, evaluation_context: List[Dict[str, Any]] = None, reasoning_context: str = None, data_source: str = None) -> Dict[str, Any]:
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
**🚨 MANDATORY DATA SOURCE CONSTRAINT (from user's question):**
The user's question was analyzed and routed to data_source="{data_source}".

**STRICT RULES - YOU MUST FOLLOW THESE:**

1. **If user explicitly mentioned a data source in their question (e.g., "from the 10k", "in the earnings call", "latest news"), you MUST respect that choice throughout ALL iterations.**

2. **For data_source="{data_source}":**"""

                if data_source == '10k':
                    data_source_routing += """
   - The user explicitly wants 10-K SEC filing data
   - NEVER set needs_transcript_search=true - this violates the user's explicit request
   - Only set needs_news_search=true if user ALSO explicitly asked for news in the question
   - If 10-K data has been exhaustively searched and doesn't contain the answer:
     * Set should_iterate=false
     * State clearly: "The requested information is not available in the 10-K filings"
     * Do NOT suggest switching to earnings transcripts - respect the user's data source choice"""
                elif data_source == 'latest_news':
                    data_source_routing += """
   - The user explicitly wants latest news/web data
   - NEVER set needs_transcript_search=true - this violates the user's explicit request
   - Set needs_news_search=true if more news would help answer the question
   - If news has been exhaustively searched and doesn't contain the answer:
     * Set should_iterate=false
     * State clearly: "The requested information is not available in recent news"
     * Do NOT suggest switching to transcripts/10-K - respect the user's data source choice"""
                elif data_source == 'earnings_transcripts':
                    data_source_routing += """
   - The user is asking about earnings transcripts/calls
   - Set needs_transcript_search=true if more transcript data would help
   - Only set needs_news_search=true if user ALSO explicitly asked for news
   - If transcripts have been exhaustively searched and don't contain the answer:
     * Set should_iterate=false
     * State clearly: "The requested information is not available in earnings transcripts"
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
You are a STRICT expert financial analyst evaluating the quality of an AI-generated answer. Be critical and demanding - only excellent, comprehensive answers should pass.
{conversation_context}
Original User Question: {original_question}
{reasoning_section}
**CHECK ORIGINAL QUESTION FOR EXPLICIT NEWS REQUESTS:**
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

**DEFAULT STANCE: ITERATE TO USE ALL AVAILABLE ITERATIONS**

The agent has multiple iterations available to build a comprehensive answer. Use them ALL unless the answer is near-PERFECT.

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

**🚨 DATA SOURCE ROUTING - CRITICAL OVERRIDE:**
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

TASK 4 - Targeted Follow-up Questions:
If iterating, provide 1-3 VERY SPECIFIC follow-up questions targeting the exact gaps identified.

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

EXAMPLE RESPONSE (demonstrating proper level of detail in reasoning):
{{
    "completeness_score": 6,
    "accuracy_score": 8,
    "clarity_score": 7,
    "specificity_score": 5,
    "overall_confidence": 0.65,
    "should_iterate": true,
    "needs_transcript_search": true,
    "needs_news_search": false,
    "transcript_search_query": "What specific revenue growth percentage was reported and how does it compare to the previous quarter?",
    "news_search_query": null,
    "iteration_decision_reasoning": "While the answer correctly identifies revenue trends, it lacks specific growth percentages which are central to the question. The user asked about profitability but no operating margin data or comparisons are provided. Executive commentary on forward guidance and strategic priorities is completely absent. The answer would benefit from quarter-over-quarter comparisons to show the progression mentioned. These gaps prevent the answer from fully addressing the user's information needs.",
    "follow_up_questions": ["What specific revenue growth percentage was reported and how does it compare to the previous quarter?", "What were the operating margins and gross margins, and how do they trend over recent quarters?", "What did executives say about forward guidance, strategic priorities, and market outlook?"],
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
                
                # Use Cerebras client for evaluation (fast, cost-effective)
                evaluation_model = self.config.get("evaluation_model", self.config.get("cerebras_model", "qwen-3-235b-a22b-instruct-2507"))
                
                # Determine which client to use based on model
                if self.cerebras_available and self.cerebras_client:
                    client = self.cerebras_client
                    rag_logger.info(f"🤖 Using Cerebras client for evaluation: {evaluation_model}")
                else:
                    client = self.client
                    rag_logger.info(f"🤖 Using OpenAI client for evaluation: {evaluation_model}")
                
                start_time = time.time()
                if self.cerebras_available and self.cerebras_client:
                    response = client.chat.completions.create(
                        model=evaluation_model,
                        messages=[
                            {"role": "system", "content": "You are a STRICT financial analyst evaluator. Be critical and demanding - only truly comprehensive, detailed answers with specific metrics should receive high scores. Always respond with valid JSON only. No additional text or formatting."},
                            {"role": "user", "content": evaluation_prompt}
                        ],
                        temperature=self.config.get("evaluation_temperature", 0.05),
                        max_completion_tokens=3000  # Increased to allow more detailed evaluation reasoning
                    )
                else:
                    response = client.chat.completions.create(
                        model=evaluation_model,
                        messages=[
                            {"role": "system", "content": "You are a STRICT financial analyst evaluator. Be critical and demanding - only truly comprehensive, detailed answers with specific metrics should receive high scores. Always respond with valid JSON only. No additional text or formatting."},
                            {"role": "user", "content": evaluation_prompt}
                        ],
                        temperature=self.config.get("evaluation_temperature", 0.05),
                        max_tokens=3000  # Increased to allow more detailed evaluation reasoning
                    )
                call_time = time.time() - start_time
                
                # Detailed evaluation response logging
                rag_logger.info(f"✅ ===== ANSWER EVALUATION LLM RESPONSE ===== (call time: {call_time:.3f}s)")
                rag_logger.info(f"📊 Response tokens used: {response.usage.total_tokens if response.usage else 'unknown'}")
                rag_logger.info(f"📊 Prompt tokens: {response.usage.prompt_tokens if response.usage else 'unknown'}")
                rag_logger.info(f"📊 Completion tokens: {response.usage.completion_tokens if response.usage else 'unknown'}")
                if hasattr(response, 'finish_reason'):
                    rag_logger.info(f"🏁 Finish reason: {response.finish_reason}")
                if hasattr(response, 'model'):
                    rag_logger.info(f"🤖 Model used: {response.model}")
                
                # Check if we got a valid response
                if not response or not response.choices or not response.choices[0].message.content:
                    raise ValueError(f"Empty response from OpenAI API on attempt {attempt + 1}")
                
                evaluation_text = response.choices[0].message.content.strip()
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
