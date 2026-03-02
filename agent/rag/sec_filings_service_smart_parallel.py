#!/usr/bin/env python3
"""
SEC Filings Service for RAG System

Features:
- Planning phase generates targeted sub-questions and search plan
- Parallel search execution (table + text simultaneously)
- Dynamic replanning based on evaluation feedback
- 91% accuracy on FinanceBench, ~10s per question
"""

import asyncio
import logging
import os
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import time

# Import LLM utilities for retry and error handling
from .llm_utils import LLMError, is_retryable_error, get_user_friendly_message

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')

# SEC 10-K Section Definitions (same as iterative service)
SEC_10K_SECTIONS = {
    "item_1": {"title": "Business", "keywords": ["business", "operations", "products", "services"]},
    "item_1a": {"title": "Risk Factors", "keywords": ["risk", "risks", "risk factors", "uncertainty"]},
    "item_7": {"title": "MD&A", "keywords": ["md&a", "management discussion", "analysis", "performance"]},
    "item_8": {"title": "Financial Statements", "keywords": ["financial statements", "balance sheet", "income statement", "cash flow"]},
}

# Financial keywords for table-first detection
FINANCIAL_KEYWORDS = [
    'revenue', 'income', 'profit', 'loss', 'earnings', 'sales', 'expenses',
    'assets', 'liabilities', 'equity', 'cash flow', 'ratio', 'margin',
    'million', 'billion', 'percent', '%', 'dollar', 'amount', 'total', 'eps',
    'balance sheet', 'income statement', 'cash flow statement'
]


class SmartParallelSECFilingsService:
    """
    SEC Filings Service with planning-driven parallel retrieval.

    91% accuracy on FinanceBench, ~10s per question.
    """

    def __init__(self, database_manager, config):
        """
        Initialize Smart Parallel SEC Filings Service.

        Args:
            database_manager: DatabaseManager instance for database access
            config: Config instance with settings
        """
        self.database_manager = database_manager
        self.config = config
        self.max_iterations = 7

        # Initialize cross-encoder for reranking
        try:
            rag_logger.info("🔧 Loading cross-encoder model for reranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.cross_encoder_available = True
            rag_logger.info("✅ Cross-encoder loaded successfully")
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load cross-encoder: {e}")
            self.cross_encoder = None
            self.cross_encoder_available = False

        # Initialize LLM clients
        self._init_llm_clients()

        # Session state
        self.current_session = None

        logger.info("✅ SEC Filings Service initialized")

    def _init_llm_clients(self):
        """Initialize LLM clients for planning and generation."""
        # Cerebras client (fast, for planning and evaluation)
        try:
            cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
            if cerebras_api_key:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_api_key)
                self.cerebras_available = True
                self.cerebras_model = "qwen-3-235b-a22b-instruct-2507"
                rag_logger.info("✅ Cerebras client initialized (Qwen 235B)")
            else:
                self.cerebras_client = None
                self.cerebras_available = False
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load Cerebras: {e}")
            self.cerebras_client = None
            self.cerebras_available = False

        # OpenAI fallback (used on Cerebras 429)
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                from agent.llm.openai_client import OpenAILLMClient
                self.openai_client = OpenAILLMClient(
                    api_key=openai_api_key,
                    default_model="gpt-5-nano-2025-08-07",
                )
                self.openai_available = True
                rag_logger.info("✅ OpenAI fallback client initialized (gpt-5-nano)")
            else:
                self.openai_client = None
                self.openai_available = False
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load OpenAI fallback: {e}")
            self.openai_client = None
            self.openai_available = False

        # Gemini client (fallback)
        try:
            import google.generativeai as genai
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                genai.configure(api_key=google_api_key)
                self.gemini_available = True
                self.gemini_model = "gemini-2.0-flash"
                rag_logger.info("✅ Gemini client initialized")
            else:
                self.gemini_available = False
        except Exception as e:
            rag_logger.warning(f"⚠️ Failed to load Gemini: {e}")
            self.gemini_available = False

    def _reset_session(self, question: str):
        """Reset session state for a new question."""
        self.current_session = {
            'question': question,
            'planning_phase': None,
            'sub_questions': [],
            'search_plan': [],
            'iteration_history': [],
            'accumulated_chunks': [],
            'current_answer': None,
            'evaluation_history': [],
            'session_start': datetime.now().isoformat(),
            'api_calls': 0
        }

    def _make_llm_call(self, messages: List[Dict], temperature: float = 0.1,
                       max_tokens: int = 2000, expect_json: bool = False) -> str:
        """Make LLM API call: Cerebras with retries → OpenAI gpt-5-nano → Gemini."""
        from cerebras.cloud.sdk import RateLimitError as CerebrasRateLimitError
        self.current_session['api_calls'] = self.current_session.get('api_calls', 0) + 1

        # --- Cerebras with retries ---
        last_error = None
        if self.cerebras_available:
            for attempt in range(3):
                try:
                    response = self.cerebras_client.chat.completions.create(
                        model=self.cerebras_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
                except CerebrasRateLimitError as e:
                    last_error = e
                    if attempt < 2:
                        wait_time = (attempt + 1) * 5
                        rag_logger.warning(f"Cerebras 429 (attempt {attempt + 1}/3). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        rag_logger.warning("Cerebras 429 after max retries — falling back to OpenAI gpt-5-nano")
                except Exception as e:
                    last_error = e
                    if is_retryable_error(e) and attempt < 2:
                        wait_time = (attempt + 1) * 2
                        rag_logger.warning(f"Cerebras failed (attempt {attempt + 1}/3): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        rag_logger.warning(f"Cerebras failed: {e}")
                        break

        # --- OpenAI fallback ---
        if self.openai_available:
            try:
                result = self.openai_client.complete(
                    messages=messages,
                    max_tokens=max_tokens,
                    reasoning_effort="medium",
                )
                rag_logger.info("OpenAI gpt-5-nano fallback succeeded")
                return result
            except Exception as e:
                last_error = e
                rag_logger.warning(f"OpenAI fallback failed: {e}, trying Gemini")

        # --- Gemini fallback ---
        if self.gemini_available:
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel(self.gemini_model)
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = e
                rag_logger.error(f"Gemini fallback failed: {e}")

        if last_error:
            raise LLMError(
                user_message=get_user_friendly_message(last_error),
                technical_message=str(last_error),
                retryable=is_retryable_error(last_error)
            )
        raise LLMError(
            user_message="Unable to process your request. Please try again.",
            technical_message="No LLM client available",
            retryable=False
        )

    async def _make_llm_call_async(self, messages: List[Dict], temperature: float = 0.1,
                                   max_tokens: int = 2000, expect_json: bool = False) -> str:
        """Async LLM call: Cerebras with retries → OpenAI gpt-5-nano → Gemini."""
        from cerebras.cloud.sdk import RateLimitError as CerebrasRateLimitError
        self.current_session['api_calls'] = self.current_session.get('api_calls', 0) + 1

        # --- Cerebras with retries ---
        last_error = None
        if self.cerebras_available:
            for attempt in range(3):
                try:
                    response = self.cerebras_client.chat.completions.create(
                        model=self.cerebras_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    return response.choices[0].message.content
                except CerebrasRateLimitError as e:
                    last_error = e
                    if attempt < 2:
                        wait_time = (attempt + 1) * 5
                        rag_logger.warning(f"Cerebras 429 (attempt {attempt + 1}/3). Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        rag_logger.warning("Cerebras 429 after max retries — falling back to OpenAI gpt-5-nano")
                except Exception as e:
                    last_error = e
                    if is_retryable_error(e) and attempt < 2:
                        wait_time = (attempt + 1) * 2
                        rag_logger.warning(f"Cerebras failed (attempt {attempt + 1}/3): {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        rag_logger.warning(f"Cerebras failed: {e}")
                        break

        # --- OpenAI fallback ---
        if self.openai_available:
            try:
                result = self.openai_client.complete(
                    messages=messages,
                    max_tokens=max_tokens,
                    reasoning_effort="medium",
                )
                rag_logger.info("OpenAI gpt-5-nano fallback succeeded")
                return result
            except Exception as e:
                last_error = e
                rag_logger.warning(f"OpenAI fallback failed: {e}, trying Gemini")

        # --- Gemini fallback ---
        if self.gemini_available:
            try:
                import google.generativeai as genai
                model = genai.GenerativeModel(self.gemini_model)
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                last_error = e
                rag_logger.error(f"Gemini fallback failed: {e}")

        if last_error:
            raise LLMError(
                user_message=get_user_friendly_message(last_error),
                technical_message=str(last_error),
                retryable=is_retryable_error(last_error)
            )
        raise LLMError(
            user_message="Unable to process your request. Please try again.",
            technical_message="No LLM client available",
            retryable=False
        )

    def _parse_json_with_retry(self, response_text: str, max_retries: int = 5,
                                default_result: Dict = None) -> Dict:
        """Parse JSON from LLM response with retry and cleanup."""
        if default_result is None:
            default_result = {}

        for attempt in range(max_retries):
            try:
                # Clean markdown code blocks
                text = response_text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                # Find JSON object
                start_idx = text.find('{')
                end_idx = text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = text[start_idx:end_idx]
                    return json.loads(json_str)

            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    rag_logger.warning(f"JSON parsing failed after {max_retries} attempts: {e}")
                    return default_result

        return default_result

    async def execute_smart_parallel_search_async(
        self,
        query: str,
        query_embedding: np.ndarray,
        ticker: str,
        fiscal_year: int = None,
        max_iterations: int = 5,
        confidence_threshold: float = 0.8,
        event_yielder=None,
        embedding_function=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute smart parallel 10-K search with planning-driven retrieval.

        This is the main entry point that implements:
        1. Smart Planning - generate sub-questions AND search_plan
        2. Parallel Multi-Query Retrieval - execute ALL searches in parallel
        3. Answer Generation - generate answer with ALL chunks
        4. Evaluation & Dynamic Replanning - replan if quality < 90%

        Args:
            query: User's question
            query_embedding: Query embedding vector (fallback if embedding_function not provided)
            ticker: Company ticker symbol
            fiscal_year: Optional fiscal year filter
            max_iterations: Maximum iterations (default 5)
            confidence_threshold: Quality score to stop early (default 0.9)
            event_yielder: Optional callback for streaming events
            embedding_function: Optional function to generate embeddings for sub-questions
                               (signature: embedding_function(text) -> np.ndarray)

        Yields:
            Events with iteration progress and final results
        """
        max_iterations = min(max_iterations, self.max_iterations)

        rag_logger.info(f"🚀 Starting SMART PARALLEL 10-K search for {ticker}")
        rag_logger.info(f"   Max iterations: {max_iterations}, Threshold: {confidence_threshold}")

        # Reset session
        self._reset_session(query)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 0: SMART PLANNING
        # ═══════════════════════════════════════════════════════════════════
        if event_yielder:
            yield {
                'type': 'planning_start',
                'message': '🧠 Generating smart investigation plan...',
                'data': {'question': query}
            }

        plan = await self._plan_investigation(query)
        sub_questions = plan.get('sub_questions', [query])
        search_plan = plan.get('search_plan', [])
        analysis = plan.get('analysis', {})

        self.current_session['planning_phase'] = plan
        self.current_session['sub_questions'] = sub_questions
        self.current_session['search_plan'] = search_plan

        rag_logger.info(f"📋 Planning complete: {len(sub_questions)} sub-questions, {len(search_plan)} searches")

        if event_yielder:
            yield {
                'type': 'planning_complete',
                'message': f'📋 Plan ready: {len(sub_questions)} sub-questions, {len(search_plan)} searches',
                'data': {
                    'sub_questions': sub_questions,
                    'search_plan': search_plan,
                    'complexity': analysis.get('complexity_assessment', 'medium')
                }
            }

        # Get all tables upfront
        all_tables = await self._get_all_tables_for_ticker(ticker, fiscal_year)
        rag_logger.info(f"📊 {len(all_tables)} tables available")

        # Initialize tracking
        accumulated_chunks = []
        current_answer = None
        seen_chunk_ids = set()
        next_chunk_idx = 1  # Global [10K-i] counter across iterations

        # ═══════════════════════════════════════════════════════════════════
        # ITERATIVE LOOP
        # ═══════════════════════════════════════════════════════════════════
        for iteration in range(max_iterations):
            iteration_num = iteration + 1
            rag_logger.info(f"\n🔄 ITERATION {iteration_num}/{max_iterations}")

            if event_yielder:
                yield {
                    'type': 'iteration_start',
                    'message': f'🔄 Iteration {iteration_num}: Parallel retrieval ({len(search_plan)} queries)',
                    'data': {'iteration': iteration_num, 'num_queries': len(search_plan)}
                }

            # ═══════════════════════════════════════════════════════════════
            # PHASE 1: PARALLEL MULTI-QUERY RETRIEVAL
            # ═══════════════════════════════════════════════════════════════
            retrieved_chunks = await self._parallel_multi_query_retrieval(
                search_plan=search_plan,
                query_embedding=query_embedding,
                ticker=ticker,
                fiscal_year=fiscal_year,
                all_tables=all_tables,
                embedding_function=embedding_function
            )

            # Deduplicate and accumulate
            new_chunks = []
            for chunk in retrieved_chunks:
                chunk_id = chunk.get('id', str(hash(chunk.get('content', '')[:100])))
                if chunk_id not in seen_chunk_ids:
                    accumulated_chunks.append(chunk)
                    new_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

            rag_logger.info(f"   ✅ Retrieved {len(new_chunks)} new chunks, total: {len(accumulated_chunks)}")

            if event_yielder:
                yield {
                    'type': 'retrieval_complete',
                    'message': f'📥 Retrieved {len(new_chunks)} chunks',
                    'data': {'new_chunks': len(new_chunks), 'total_chunks': len(accumulated_chunks)}
                }

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: ANSWER GENERATION
            # Pass only new_chunks but start numbering from next_chunk_idx
            # so [10K-i] markers are globally sequential across iterations.
            # Previous answer's citations (e.g. [10K-2]) remain valid.
            # ═══════════════════════════════════════════════════════════════
            chunks_to_show = new_chunks[:20]
            current_answer = await self._generate_smart_answer(
                question=query,
                sub_questions=sub_questions,
                chunks=chunks_to_show,
                previous_answer=current_answer,
                iteration=iteration_num,
                chunk_start_idx=next_chunk_idx
            )
            next_chunk_idx += len(chunks_to_show)

            rag_logger.info(f"   ✅ Answer generated ({len(current_answer)} chars)")

            # ═══════════════════════════════════════════════════════════════
            # PHASE 3: EVALUATION
            # Evaluate against all accumulated chunks so references to
            # earlier-iteration chunks are not penalised.
            # ═══════════════════════════════════════════════════════════════
            evaluation = await self._evaluate_answer_quality(
                question=query,
                answer=current_answer,
                chunks=accumulated_chunks
            )

            quality_score = evaluation.get('quality_score', 0.0)
            missing_info = evaluation.get('missing_info', [])

            self.current_session['evaluation_history'].append({
                'iteration': iteration_num,
                'quality_score': quality_score,
                'missing_info': missing_info
            })

            rag_logger.info(f"   📊 Quality: {quality_score:.2f}")

            if event_yielder:
                yield {
                    'type': 'evaluation_complete',
                    'message': f'📊 Quality: {quality_score:.0%}',
                    'data': {
                        'quality_score': quality_score,
                        'missing_info': missing_info[:3]
                    }
                }

            # ═══════════════════════════════════════════════════════════════
            # EARLY TERMINATION CHECK
            # ═══════════════════════════════════════════════════════════════
            if quality_score >= confidence_threshold and not missing_info:
                rag_logger.info(f"   🎉 Early termination: quality {quality_score:.2f} >= {confidence_threshold}, no missing info")
                break

            # ═══════════════════════════════════════════════════════════════
            # PHASE 4: DYNAMIC REPLANNING
            # ═══════════════════════════════════════════════════════════════
            if iteration < max_iterations - 1:
                new_search_plan = await self._replan_based_on_evaluation(
                    evaluation=evaluation,
                    current_search_plan=search_plan,
                    question=query
                )

                if new_search_plan:
                    search_plan = new_search_plan
                    rag_logger.info(f"   🔄 Replanned: {len(new_search_plan)} new queries")
                else:
                    rag_logger.info("   ⚠️ No replan possible, stopping")
                    break

        # ═══════════════════════════════════════════════════════════════════
        # FINAL RESULTS
        # ═══════════════════════════════════════════════════════════════════
        iterations_used = len(self.current_session['evaluation_history'])
        final_quality = self.current_session['evaluation_history'][-1]['quality_score'] if self.current_session['evaluation_history'] else 0.0

        rag_logger.info(f"\n✅ Smart Parallel Flow Complete")
        rag_logger.info(f"   Iterations: {iterations_used}")
        rag_logger.info(f"   Final Quality: {final_quality:.2f}")
        rag_logger.info(f"   Total Chunks: {len(accumulated_chunks)}")

        # Post-process: normalize any malformed [10K_TICKER_FY_CHUNKIDX] citations
        # that Qwen may generate despite [10K-N] instructions
        if current_answer:
            current_answer = self._fix_10k_citation_format(current_answer, accumulated_chunks)

        # Yield final results
        yield {
            'type': 'search_complete',
            'message': f'✅ Search complete ({iterations_used} iterations, {final_quality:.0%} quality)',
            'data': {
                'answer': current_answer,
                'chunks': accumulated_chunks,
                'iterations': iterations_used,
                'quality_score': final_quality,
                'sub_questions': sub_questions,
                'api_calls': self.current_session.get('api_calls', 0)
            }
        }

    async def _plan_investigation(self, question: str) -> Dict:
        """
        Generate smart investigation plan with sub-questions and search strategy.

        Returns:
            Dict with sub_questions and search_plan
        """
        prompt = f"""You are a financial analyst creating a SEARCH STRATEGY for a RAG system that retrieves data from 10-K filings. Do not use emojis in your responses.

QUESTION: {question}

Create a strategic search plan to retrieve SPECIFIC DATA from the 10-K filing.

SEARCH QUERIES are used as embedding search queries against a vector database of 10-K chunks.
Each query is independently embedded and searched — they run in parallel, so use MULTIPLE SHORT FOCUSED queries rather than one long query.

QUERY RULES:
- Each query must be SHORT (2–5 words). A shorter, focused phrase produces a better embedding match than a long noisy one.
- Do NOT cram synonyms or alternatives into a single query — put each angle as a SEPARATE query in the plan.
- Do NOT use question form. Do NOT include company names, tickers, or years.
- GOOD: "total revenue"  |  "annual revenue subscription"  |  "cost of revenue"
- BAD: "total revenue cost of revenue gross profit income statement operating expenses" (too many concepts, dilutes embedding)

DERIVED METRICS AND RATIOS — CRITICAL:
If the question asks for a metric computed from two or more values, search for each component separately. Never search for the derived metric directly.
- Per-unit figure → one query for numerator, one query for denominator
- Financial ratio → one query per line item
- Margin % → numerator query + denominator query
For each component, generate 2–3 short alternative-phrasing queries to approach from different angles.

SECTION TARGETING — GENERAL PRINCIPLE:
Think about where in a 10-K the data physically lives and use vocabulary from that section.
- Narrative/operational data (workforce, strategy, business overview): item_1 vocabulary
- Risk factors: item_1a vocabulary
- Management commentary and trends: item_7 MD&A vocabulary
- Financial line items and tables: item_8 Financial Statements vocabulary

SEARCH TYPES:
- "table": For quantitative data (revenue, COGS, assets, any number/line item)
- "text": For narrative data (workforce disclosures, strategy, explanations)

Return ONLY valid JSON:
{{
    "analysis": {{
        "question_type": "Description",
        "complexity_assessment": "simple|medium|complex"
    }},
    "sub_questions": [
        "Specific data-retrieval sub-question 1",
        "Specific data-retrieval sub-question 2"
    ],
    "search_plan": [
        {{"query": "short focused phrase", "type": "table|text", "priority": 1}}
    ]
}}

EXAMPLES:

Question: "What is revenue per employee?"
- {{"query": "total revenue", "type": "table", "priority": 1}}
- {{"query": "subscription professional services revenue", "type": "table", "priority": 2}}
- {{"query": "number of employees", "type": "text", "priority": 1}}
- {{"query": "human capital workforce", "type": "text", "priority": 2}}

Question: "What is the R&D expense as a percentage of total operating expenses?"
- {{"query": "research development expense", "type": "table", "priority": 1}}
- {{"query": "total operating expenses", "type": "table", "priority": 1}}
- {{"query": "sales marketing general administrative", "type": "table", "priority": 2}}

Question: "What is the debt-to-equity ratio?"
- {{"query": "long-term debt borrowings", "type": "table", "priority": 1}}
- {{"query": "short-term debt current liabilities", "type": "table", "priority": 2}}
- {{"query": "stockholders equity", "type": "table", "priority": 1}}
- {{"query": "total equity retained earnings", "type": "table", "priority": 2}}

Question: "What is the gross margin trend?"
- {{"query": "total revenue", "type": "table", "priority": 1}}
- {{"query": "cost of revenue gross profit", "type": "table", "priority": 1}}
- {{"query": "gross margin percentage", "type": "table", "priority": 2}}

Now analyze the original question and create the search plan."""

        try:
            messages = [
                {"role": "system", "content": "Financial analyst. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]

            response = await self._make_llm_call_async(messages, temperature=0.2, max_tokens=2000)

            # Derive a keyword fallback by stripping question words
            import re as _re
            keyword_fallback = _re.sub(r'\b(what|how|why|when|where|who|is|are|was|were|did|do|does|the|a|an)\b', '', question, flags=_re.IGNORECASE).strip()
            result = self._parse_json_with_retry(response, default_result={
                'analysis': {'complexity_assessment': 'medium'},
                'sub_questions': [question],
                'search_plan': [
                    {'query': keyword_fallback, 'type': 'table', 'priority': 1},
                    {'query': keyword_fallback, 'type': 'text', 'priority': 2}
                ]
            })

            sub_questions = result.get('sub_questions', [question])
            search_plan = result.get('search_plan', [])
            analysis = result.get('analysis', {})

            rag_logger.info(f"🧠 INVESTIGATION PLAN:")
            rag_logger.info(f"   Complexity: {analysis.get('complexity_assessment', 'unknown')}")
            rag_logger.info(f"   Sub-questions ({len(sub_questions)}):")
            for i, sq in enumerate(sub_questions, 1):
                rag_logger.info(f"     {i}. {sq}")
            rag_logger.info(f"   Search queries ({len(search_plan)}):")
            for s in search_plan:
                rag_logger.info(f"     [{s.get('type','?')}] p{s.get('priority','?')}: {s.get('query','?')}")

            return {
                'success': True,
                'sub_questions': sub_questions,
                'search_plan': search_plan,
                'analysis': analysis
            }

        except Exception as e:
            rag_logger.error(f"Planning failed: {e}")
            return {
                'success': False,
                'sub_questions': [question],
                'search_plan': [
                    {'query': question, 'type': 'table', 'priority': 1},
                    {'query': question, 'type': 'text', 'priority': 2}
                ],
                'analysis': {}
            }

    async def _parallel_multi_query_retrieval(
        self,
        search_plan: List[Dict],
        query_embedding: np.ndarray,
        ticker: str,
        fiscal_year: int,
        all_tables: List[Dict],
        top_k: int = 20,
        embedding_function=None
    ) -> List[Dict]:
        """
        Execute multiple searches in PARALLEL.

        Returns combined, deduplicated list of chunks.

        Args:
            search_plan: List of search items with 'query' and 'type'
            query_embedding: Fallback embedding if embedding_function not provided
            ticker: Company ticker
            fiscal_year: Fiscal year filter
            all_tables: Pre-fetched tables for table searches
            top_k: Number of chunks per search
            embedding_function: Function to generate embeddings for sub-questions
        """
        if not search_plan:
            return []

        all_chunks = []
        seen_ids = set()

        # Execute searches in parallel using ThreadPoolExecutor
        def execute_single_search(search_item):
            query = search_item.get('query', '')
            search_type = search_item.get('type', 'table')

            try:
                if search_type == 'table':
                    # Table retrieval via LLM selection
                    chunks = self._retrieve_tables_sync(query, all_tables, ticker=ticker, fiscal_year=fiscal_year, top_k=5)
                else:
                    # Text retrieval with reranking
                    if embedding_function is not None:
                        try:
                            sub_question_embedding = embedding_function([query])[0]
                        except Exception as e:
                            rag_logger.warning(f"⚠️ Failed to generate embedding for sub-question, using fallback: {e}")
                            sub_question_embedding = query_embedding
                    else:
                        sub_question_embedding = query_embedding

                    chunks = self._retrieve_text_sync(query, sub_question_embedding, ticker, fiscal_year, top_k)

                # Tag source and ensure ticker/fiscal_year are set
                for chunk in chunks:
                    chunk['source_query'] = query
                    chunk['source_type'] = search_type
                    # Ensure ticker and fiscal_year are always set from search context
                    if not chunk.get('ticker'):
                        chunk['ticker'] = ticker
                    if not chunk.get('fiscal_year'):
                        chunk['fiscal_year'] = fiscal_year

                return chunks

            except Exception as e:
                rag_logger.error(f"Search failed for '{query[:40]}': {e}")
                return []

        # Execute in parallel using run_in_executor so the event loop isn't blocked
        # (ThreadPoolExecutor + as_completed is synchronous and blocks SSE streaming)
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, execute_single_search, item) for item in search_plan]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                rag_logger.error(f"Search task failed: {result}")
                continue
            for chunk in result:
                chunk_id = chunk.get('id', str(hash(chunk.get('content', '')[:100])))
                if chunk_id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk_id)

        # Sort: tables first, then by relevance
        all_chunks.sort(key=lambda x: (
            0 if x.get('type') == 'table' else 1,
            -x.get('similarity', 0)
        ))

        return all_chunks

    def _retrieve_tables_sync(self, query: str, all_tables: List[Dict], ticker: str = None,
                              fiscal_year: int = None, top_k: int = 5,
                              query_embedding=None) -> List[Dict]:
        """Synchronous table retrieval with LLM selection."""
        if not all_tables:
            return []

        # Create table summary for LLM with enhanced metadata
        table_summaries = []
        for i, table in enumerate(all_tables[:50]):  # Limit for prompt size
            path = table.get('path_string', 'Unknown')
            sec_title = table.get('sec_section_title', '')
            content = table.get('content', '')

            # Create content preview (first 150 chars, prioritize headers/first row)
            content_preview = content[:150].replace('\n', ' ').strip()

            # Build rich summary: path | section | preview
            summary_parts = [f"{i+1}. {path}"]
            if sec_title and sec_title != 'Unknown':
                summary_parts.append(f"[Section: {sec_title}]")
            if content_preview:
                summary_parts.append(f"[Preview: {content_preview}...]")

            table_summaries.append(" | ".join(summary_parts))

        prompt = f"""Do not use emojis. Select the most relevant tables for this query: "{query}"

Available tables (showing: path | section | content preview):
{chr(10).join(table_summaries)}

Analyze the section titles and content previews to identify relevant tables.
Return JSON with table indices (1-indexed), selecting up to {top_k} tables:
{{"selected_tables": [1, 5, 12], "reasoning": "Brief explanation of why these tables match the query"}}"""

        try:
            messages = [
                {"role": "system", "content": "Select tables. Return JSON only."},
                {"role": "user", "content": prompt}
            ]

            response = self._make_llm_call(messages, temperature=0.1, max_tokens=500)
            result = self._parse_json_with_retry(response, default_result={'selected_tables': [1, 2]})

            selected_indices = result.get('selected_tables', [1, 2])
            chunks = []
            for idx in selected_indices[:top_k]:
                if 1 <= idx <= len(all_tables):
                    table = all_tables[idx - 1]
                    chunks.append({
                        'id': table.get('id', f'table_{idx}'),
                        'content': table.get('content', ''),
                        'chunk_text': table.get('chunk_text', table.get('content', '')),
                        'type': 'table',
                        'chunk_type': 'table',
                        'path_string': table.get('path_string', ''),
                        'sec_section': table.get('sec_section', ''),
                        'sec_section_title': table.get('sec_section_title', 'Financial Data'),
                        'ticker': table.get('ticker') or ticker,
                        'fiscal_year': table.get('fiscal_year') or fiscal_year,
                        'char_offset': table.get('char_offset'),
                        'chunk_length': table.get('chunk_length'),
                        'source_type': '10-K',
                        'similarity': 1.0
                    })

            return chunks

        except Exception as e:
            rag_logger.error(f"Table selection failed: {e}")
            return all_tables[:2] if all_tables else []

    def _retrieve_text_sync(self, query: str, query_embedding: np.ndarray,
                            ticker: str, fiscal_year: int, top_k: int = 20) -> List[Dict]:
        """Synchronous text retrieval with section selection and reranking."""
        try:
            # STEP 1: Get available sections for this ticker/fiscal_year
            available_sections = self._get_available_sections(ticker, fiscal_year)

            # STEP 2: Use LLM to select relevant sections (if sections available)
            selected_sections = []
            if available_sections:
                selected_sections = self._select_relevant_sections(query, available_sections)

            rag_logger.info(f"   🔍 TEXT query='{query}' → sections={selected_sections or 'ALL'}")

            # STEP 3: Get text chunks with optional section filtering
            chunks = self.database_manager.search_10k_filings(
                query_embedding=query_embedding,
                ticker=ticker,
                fiscal_year=fiscal_year,
                selected_sections=selected_sections if selected_sections else None
            )

            if not chunks:
                return []

            # Filter to text only (no tables)
            text_chunks = [c for c in chunks if c.get('chunk_type') != 'table']
            text_chunks = text_chunks[:top_k * 3]  # Limit candidates

            if not text_chunks:
                return []

            # Rerank with cross-encoder if available
            if self.cross_encoder_available and self.cross_encoder:
                pairs = [(query, chunk.get('chunk_text', '')) for chunk in text_chunks]
                scores = self.cross_encoder.predict(pairs)

                for chunk, score in zip(text_chunks, scores):
                    chunk['similarity'] = float(score)
                    chunk['content'] = chunk.get('chunk_text', '')  # Normalize field name
                    chunk['type'] = chunk.get('chunk_type', 'text')  # Normalize type field

                text_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            else:
                for chunk in text_chunks:
                    chunk['content'] = chunk.get('chunk_text', '')  # Normalize field name
                    chunk['type'] = chunk.get('chunk_type', 'text')  # Normalize type field

            return text_chunks[:top_k]

        except Exception as e:
            rag_logger.error(f"Text retrieval failed: {e}")
            return []

    def _get_available_sections(self, ticker: str, fiscal_year: int) -> List[Dict]:
        """
        Get all unique SEC sections available for a ticker/fiscal_year.

        Returns list like:
        [
            {'sec_section': 'item_1a', 'sec_section_title': 'Risk Factors', 'chunk_count': 42},
            {'sec_section': 'item_7', 'sec_section_title': "MD&A", 'chunk_count': 156},
        ]
        """
        try:
            from psycopg2.extras import RealDictCursor
            conn = self.database_manager._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            query = """
            SELECT DISTINCT
                sec_section,
                sec_section_title,
                COUNT(*) as chunk_count
            FROM ten_k_chunks
            WHERE UPPER(ticker) = %s AND fiscal_year = %s AND sec_section IS NOT NULL
            GROUP BY sec_section, sec_section_title
            ORDER BY chunk_count DESC
            """

            cursor.execute(query, (ticker.upper(), fiscal_year))
            sections = cursor.fetchall()
            self.database_manager._return_db_connection(conn)

            result = [dict(row) for row in sections] if sections else []
            rag_logger.info(f"📚 Found {len(result)} SEC sections for {ticker} FY{fiscal_year}")
            return result

        except Exception as e:
            rag_logger.error(f"Failed to get sections: {e}")
            return []

    def _select_relevant_sections(self, query: str, available_sections: List[Dict]) -> List[str]:
        """
        Use LLM to select relevant SEC sections for the query.

        Args:
            query: User's question
            available_sections: List of available sections with metadata

        Returns:
            List of selected sec_section identifiers (e.g., ['item_1a', 'item_7'])
        """
        if not available_sections:
            return []

        # Create section summary for LLM
        section_summaries = []
        for i, section in enumerate(available_sections, 1):
            sec_title = section.get('sec_section_title', 'Unknown')
            sec_id = section.get('sec_section', 'unknown')
            chunk_count = section.get('chunk_count', 0)
            section_summaries.append(f"{i}. {sec_id} - {sec_title} ({chunk_count} chunks)")

        prompt = f"""Do not use emojis. Select the most relevant SEC 10-K sections for this question: {query}

Available sections:
{chr(10).join(section_summaries)}

SEC 10-K Section Guide:
- item_1 (Business): Company description, products, services, operations; also contains Human Capital subsection (employee headcount, workforce size, full-time employees, talent strategy) — use for any workforce or operational narrative data
- item_1a (Risk Factors): Risks to business and financials - USE FOR RISK QUESTIONS
- item_1b (Unresolved Staff Comments): SEC review comments
- item_2 (Properties): Physical locations, facilities
- item_3 (Legal Proceedings): Lawsuits, litigation, legal matters
- item_5 (Market for Common Equity): Stock info, shareholders
- item_7 (MD&A): Management analysis, financial performance discussion
- item_7a (Market Risk Disclosures): Interest rate, currency risks
- item_8 (Financial Statements): Balance sheet, income statement, cash flow
- item_9a (Controls and Procedures): Internal controls, compliance
- item_10 (Directors and Officers): Board members, executives
- item_11 (Executive Compensation): Salaries, bonuses, stock options

Return JSON with section numbers (1-indexed):
{{"selected_sections": [1, 3], "reasoning": "Brief explanation"}}

IMPORTANT: Select 1-3 sections maximum. Be selective."""

        try:
            messages = [
                {"role": "system", "content": "Select relevant SEC sections. Return JSON only."},
                {"role": "user", "content": prompt}
            ]

            response = self._make_llm_call(messages, temperature=0.1, max_tokens=500)
            result = self._parse_json_with_retry(response, default_result={'selected_sections': []})

            selected_indices = result.get('selected_sections', [])
            selected_sections = []

            for idx in selected_indices:
                if 1 <= idx <= len(available_sections):
                    sec_section = available_sections[idx - 1].get('sec_section')
                    if sec_section:
                        selected_sections.append(sec_section)

            if selected_sections:
                rag_logger.info(f"🎯 Selected {len(selected_sections)} sections: {selected_sections}")
            else:
                rag_logger.info("⚠️ No specific sections selected, will search all")

            return selected_sections

        except Exception as e:
            rag_logger.error(f"Section selection failed: {e}")
            return []  # Return empty to search all sections

    async def _get_all_tables_for_ticker(self, ticker: str, fiscal_year: int = None) -> List[Dict]:
        """Get all available tables for a ticker.

        Prefers ten_k_chunks table rows (structured ingestion, has char_offset for highlighting).
        Falls back to ten_k_tables (old format, no char_offset) only when ten_k_chunks has none.
        """
        try:
            # Prefer ten_k_chunks table chunks — they have char_offset/chunk_length for precise highlighting
            structured_chunks = self.database_manager.get_all_table_chunks(ticker=ticker, fiscal_year=fiscal_year)
            if structured_chunks:
                rag_logger.info(f"📊 Using {len(structured_chunks)} structured table chunks from ten_k_chunks for {ticker} FY{fiscal_year}")
                return structured_chunks
            # Fallback: ten_k_tables (old ingestion, no char_offset)
            rag_logger.info(f"📊 No structured chunks for {ticker} FY{fiscal_year} — falling back to ten_k_tables")
            return await self.database_manager.get_all_tables_for_ticker_async(ticker=ticker, fiscal_year=fiscal_year)
        except Exception as e:
            rag_logger.error(f"Failed to get tables: {e}")
            return []

    async def _generate_smart_answer(
        self,
        question: str,
        sub_questions: List[str],
        chunks: List[Dict],
        previous_answer: Optional[str],
        iteration: int,
        chunk_start_idx: int = 1
    ) -> str:
        """Generate answer using retrieved chunks."""
        # Build context using globally sequential [10K-i] markers
        context_parts = []
        for i, chunk in enumerate(chunks, chunk_start_idx):
            chunk_type = (chunk.get('type') or chunk.get('chunk_type') or 'text').upper()
            section = chunk.get('sec_section_title', chunk.get('path_string', 'SEC Filing'))
            fy = chunk.get('fiscal_year', '')
            fy_label = f" FY{fy}" if fy else ""
            # Tables are never truncated — financial statements need full data
            raw_content = chunk.get('content', '')
            content = raw_content if chunk_type == 'TABLE' else raw_content[:2000]
            context_parts.append(f"SOURCE [10K-{i}] [{chunk_type}]{fy_label} {section}\n{content}")

        context = "\n---\n".join(context_parts)

        if previous_answer:
            prompt = f"""Do not use emojis. Refine the answer using newly retrieved sources.

QUESTION: {question}

SUB-QUESTIONS:
{chr(10).join(f'- {sq}' for sq in sub_questions)}

PREVIOUS ANSWER (citations [10K-1]...[10K-{chunk_start_idx - 1}] remain valid):
{previous_answer}

NEW SOURCES ({f'[10K-{chunk_start_idx}]' if context_parts else 'none'}...[10K-{chunk_start_idx + len(context_parts) - 1}]):
{context}

Integrate new data. Use exact markers [10K-N]. Provide precise numbers. If the question asks for a derived metric and all components are now available, compute and show it explicitly."""
        else:
            prompt = f"""NO EMOJIS. Answer the question using the retrieved data.

QUESTION: {question}

SUB-QUESTIONS:
{chr(10).join(f'- {sq}' for sq in sub_questions)}

RETRIEVED DATA:
{context}

Cite sources using exact markers [10K-1], [10K-2] etc. Provide precise numbers with units (e.g., "$2.5 billion", "$245 million", "42.1%") — never state a bare number without its unit. **Bold every financial figure** (e.g. **$2.5 billion**, **+15%**, **42,000 employees**). Use a markdown table whenever presenting 2+ values across periods or metrics. Note any missing info.

DERIVED METRICS — CRITICAL: If the question asks for a ratio, per-unit figure, or any value computed from multiple inputs, and the source data contains the required components, compute the result directly. Do not say you cannot calculate it. Show your work (numerator, denominator, result) and cite each component source."""

        try:
            messages = [
                {"role": "system", "content": "Expert financial analyst. ABSOLUTELY NO EMOJIS OR SPECIAL SYMBOLS — use plain text and markdown only. Cite sources with [10K-N] markers. Be precise. Bold ALL financial figures (dollar amounts, percentages, ratios, counts). Use markdown tables for multi-period or multi-metric comparisons. CRITICAL: If retrieved data is from a different fiscal year than requested, state this at the top. Never silently answer from a wrong period."},
                {"role": "user", "content": prompt}
            ]

            return await self._make_llm_call_async(messages, temperature=0.1, max_tokens=4000)

        except Exception as e:
            rag_logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}"

    async def _evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        chunks: List[Dict]
    ) -> Dict:
        """Evaluate answer quality and identify gaps."""
        prompt = f"""Do not use emojis. Evaluate this financial analysis answer. Be strict about scope.

QUESTION: {question}

ANSWER:
{answer[:3000]}

Evaluate strictly:
1. Does it answer the question completely with actual numbers?
2. Are all required data points present and cited?
3. If the question requires computing a metric from multiple inputs, are ALL input components present in the answer?

CRITICAL: `missing_info` must list every specific data point that is absent from the answer. If the answer says any value "could not be found", "is not disclosed", or "cannot be computed", that value MUST appear in missing_info. An answer that reports missing data is NOT complete.

Return JSON:
{{
    "quality_score": 0.0-1.0,
    "issues": ["issue1"],
    "missing_info": ["specific data point 1 that is absent", "specific data point 2"],
    "suggestions": ["what to search for next"]
}}"""

        try:
            messages = [
                {"role": "system", "content": "Evaluate answers. Return JSON only."},
                {"role": "user", "content": prompt}
            ]

            response = await self._make_llm_call_async(messages, temperature=0.1, max_tokens=1000)
            result = self._parse_json_with_retry(response, default_result={
                'quality_score': 0.7,
                'issues': [],
                'missing_info': [],
                'suggestions': []
            })

            return result

        except Exception as e:
            rag_logger.error(f"Evaluation failed: {e}")
            return {'quality_score': 0.7, 'issues': [], 'missing_info': []}

    async def _replan_based_on_evaluation(
        self,
        evaluation: Dict,
        current_search_plan: List[Dict],
        question: str
    ) -> List[Dict]:
        """Generate new searches based on evaluation feedback."""
        quality_score = evaluation.get('quality_score', 0.0)
        missing_info = evaluation.get('missing_info', [])

        if quality_score >= 0.80 and not missing_info:
            return []

        executed_queries = [item.get('query', '') for item in current_search_plan]

        prompt = f"""Do not use emojis. Generate NEW short search queries to fill gaps in a 10-K retrieval.

QUESTION: {question}

ALREADY SEARCHED (these did NOT retrieve the needed data):
{chr(10).join(f'- {q}' for q in executed_queries)}

MISSING INFO:
{chr(10).join(f'- {m}' for m in missing_info[:5])}

RULES:
- Each query must be SHORT (2–5 words). Short focused phrases produce better embedding matches than long noisy ones.
- Use completely different vocabulary from the queries already tried — synonyms, alternative terminology, different section vocabulary.
- Do NOT repeat or rephrase existing queries.
- Think about what vocabulary the 10-K actually uses in the section containing this data.
- Generate multiple short queries (each a different angle) rather than one long query.

Return JSON with 2-4 NEW short searches:
{{
    "new_searches": [
        {{"query": "short alternative phrase", "type": "table|text", "priority": 1}}
    ]
}}"""

        try:
            messages = [
                {"role": "system", "content": "Generate searches. Return JSON only."},
                {"role": "user", "content": prompt}
            ]

            response = await self._make_llm_call_async(messages, temperature=0.2, max_tokens=500)
            result = self._parse_json_with_retry(response, default_result={'new_searches': []})

            return result.get('new_searches', [])

        except Exception as e:
            rag_logger.error(f"Replanning failed: {e}")
            return []

    def format_10k_context(self, chunks: List[Dict]) -> str:
        """Format chunks for context display. Uses [10K-1], [10K-2] markers to match
        get_10k_citations() so in-text citations are clickable in the frontend.
        When multiple fiscal years are present, labels each chunk with FY so the model
        can structure multi-year answers (e.g. 'FY2020 ... FY2021 ...')."""
        if not chunks:
            return "No relevant 10-K data found."

        parts = []
        for i, chunk in enumerate(chunks[:15], 1):
            chunk_type = chunk.get('type', 'unknown').upper()
            section = chunk.get('sec_section_title', 'SEC Filing')
            raw_content = chunk.get('content', '')
            content = raw_content if chunk_type == 'TABLE' else raw_content[:1500]
            fy = chunk.get('fiscal_year', '')
            fy_label = f" FY{fy}" if fy else ""
            # Use [10K-1], [10K-2] to match citation markers sent to frontend
            parts.append(f"SOURCE [10K-{i}] [{chunk_type}]{fy_label} {section}\n{content}")

        return "\n\n---\n\n".join(parts)

    def get_10k_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Get citations from chunks."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            # chunk_text is the canonical field; 'content' is an alias used by table chunks
            full_content = chunk.get('chunk_text') or chunk.get('content', '')
            # Build a stable chunk_id so the frontend can scroll to the exact mark
            chunk_id = chunk.get('id') or chunk.get('citation') or f"chunk-{i}"
            citations.append({
                'source_number': i,
                'type': '10-K',  # Explicitly mark as 10-K citation
                'ticker': chunk.get('ticker', ''),
                'fiscal_year': chunk.get('fiscal_year', ''),
                'section': chunk.get('sec_section_title', chunk.get('section', 'SEC Filing')),
                'path': chunk.get('path_string', 'Document'),
                'chunk_text': full_content[:500],  # Include chunk text for frontend (truncated)
                'chunk_type': chunk.get('type', 'text'),  # table or text
                'marker': f"[10K-{i}]",  # Citation marker
                'preview': full_content[:200],
                'char_offset': chunk.get('char_offset'),  # Character offset for precise highlighting
                'chunk_length': chunk.get('chunk_length') or len(full_content),
                'chunk_id': chunk_id,  # Stable ID for per-chunk scroll targeting in the viewer
            })
        return citations

    @staticmethod
    def _fix_10k_citation_format(text: str, chunks: List[Dict], chunk_start_idx: int = 1) -> str:
        """
        Post-process SEC answer to replace malformed [10K_TICKER_FY_CHUNKIDX] markers
        that Qwen generates despite [10K-N] instructions.

        Builds a mapping from chunk_index → sequential [10K-N] position using the
        accumulated chunks list, then replaces any bad-format citations in the text.
        Unresolvable citations are stripped.
        """
        # Build mapping: DB chunk_index → [10K-N] sequential marker
        chunk_idx_to_n: dict = {}
        for i, chunk in enumerate(chunks, chunk_start_idx):
            raw_ci = chunk.get('chunk_index')
            if raw_ci is not None:
                try:
                    chunk_idx_to_n[int(raw_ci)] = i
                except (ValueError, TypeError):
                    pass
            # Also parse from the citation field: "10K_ORCL_FY2025_630" → 630
            citation_str = chunk.get('citation', '')
            if citation_str:
                parts = citation_str.rsplit('_', 1)
                if len(parts) == 2:
                    try:
                        chunk_idx_to_n[int(parts[1])] = i
                    except (ValueError, TypeError):
                        pass

        def replace_bad(m):
            try:
                db_idx = int(m.group(3))
                if db_idx in chunk_idx_to_n:
                    return f"[10K-{chunk_idx_to_n[db_idx]}]"
            except (ValueError, IndexError):
                pass
            return ''  # strip unresolvable marker

        return re.sub(r'\[10K_([A-Za-z]+)_FY(\d+)_(\d+)\]', replace_bad, text)

    @staticmethod
    def _remap_sec_citations(
        answer: str,
        citations: List[Dict],
        offset: int,
    ) -> tuple:
        """
        Shift [10K-N] markers in answer text and citation dicts by `offset`.

        Returns (remapped_answer, remapped_citations).
        offset=0 → no-op (first ticker keeps its numbers).
        """
        if offset == 0:
            return answer, citations

        def _shift(m):
            return f"[10K-{int(m.group(1)) + offset}]"

        remapped_answer = re.sub(r'\[10K-(\d+)\]', _shift, answer)

        remapped_citations = []
        for c in citations:
            nc = c.copy()
            nc['source_number'] = c['source_number'] + offset
            nc['marker'] = f"[10K-{c['source_number'] + offset}]"
            remapped_citations.append(nc)

        return remapped_answer, remapped_citations
