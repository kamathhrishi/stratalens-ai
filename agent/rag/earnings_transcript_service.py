#!/usr/bin/env python3
"""
Earnings Transcript Service for RAG System.

Features:
- Planning phase generates dense keyword phrases (not raw question)
- Parallel hybrid search across tickers and quarters
- Answer generation with [TC-N] citation markers
- 2-iteration evaluate → replan loop (same pattern as SEC filings service)
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from .llm_utils import LLMError, is_retryable_error, get_user_friendly_message

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


class EarningsTranscriptService:
    """
    Earnings transcript service with planning-driven parallel retrieval.

    Follows the same pattern as SmartParallelSECFilingsService:
    plan → parallel search → generate → evaluate → replan (up to 2 iterations).
    """

    MAX_ITERATIONS = 2

    def __init__(self, search_engine, config):
        """
        Args:
            search_engine: SearchEngine instance (has search_similar_chunks)
            config: Config instance
        """
        self.search_engine = search_engine
        self.config = config
        self._init_llm_clients()
        logger.info("✅ EarningsTranscriptService initialized")

    # ═══════════════════════════════════════════════════════════════════════
    # LLM SETUP (same pattern as SEC service: Cerebras + Gemini fallback)
    # ═══════════════════════════════════════════════════════════════════════

    def _init_llm_clients(self):
        """Initialize Cerebras (fast), OpenAI gpt-5-nano (429 fallback), and Gemini (fallback) LLM clients."""
        import os

        # Cerebras
        try:
            cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
            if cerebras_api_key:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_api_key)
                self.cerebras_available = True
                self.cerebras_model = "qwen-3-235b-a22b-instruct-2507"
                rag_logger.info("✅ [Transcript] Cerebras client initialized")
            else:
                self.cerebras_client = None
                self.cerebras_available = False
        except Exception as e:
            rag_logger.warning(f"⚠️ [Transcript] Cerebras init failed: {e}")
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
                rag_logger.info("✅ [Transcript] OpenAI fallback client initialized (gpt-5-nano)")
            else:
                self.openai_client = None
                self.openai_available = False
        except Exception as e:
            rag_logger.warning(f"⚠️ [Transcript] OpenAI fallback init failed: {e}")
            self.openai_client = None
            self.openai_available = False

        # Gemini fallback
        try:
            import google.generativeai as genai
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                genai.configure(api_key=google_api_key)
                self.gemini_available = True
                self.gemini_model = "gemini-2.0-flash"
                rag_logger.info("✅ [Transcript] Gemini client initialized")
            else:
                self.gemini_available = False
        except Exception as e:
            rag_logger.warning(f"⚠️ [Transcript] Gemini init failed: {e}")
            self.gemini_available = False

    async def _make_llm_call_async(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """Async LLM call: Cerebras with retries → OpenAI gpt-5-nano → Gemini."""
        from cerebras.cloud.sdk import RateLimitError as CerebrasRateLimitError
        last_error = None

        # --- Cerebras with retries ---
        if self.cerebras_available:
            for attempt in range(3):
                try:
                    response = self.cerebras_client.chat.completions.create(
                        model=self.cerebras_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                except CerebrasRateLimitError as e:
                    last_error = e
                    if attempt < 2:
                        wait_time = (attempt + 1) * 5
                        rag_logger.warning(f"[Transcript] Cerebras 429 (attempt {attempt + 1}/3). Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        rag_logger.warning("[Transcript] Cerebras 429 after max retries — falling back to OpenAI gpt-5-nano")
                except Exception as e:
                    last_error = e
                    if is_retryable_error(e) and attempt < 2:
                        await asyncio.sleep((attempt + 1) * 2)
                    else:
                        rag_logger.warning(f"[Transcript] Cerebras failed: {e}")
                        break

        # --- OpenAI fallback ---
        if self.openai_available:
            try:
                result = self.openai_client.complete(
                    messages=messages,
                    max_tokens=max_tokens,
                    reasoning_effort="medium",
                )
                rag_logger.info("[Transcript] OpenAI gpt-5-nano fallback succeeded")
                return result
            except Exception as e:
                last_error = e
                rag_logger.warning(f"[Transcript] OpenAI fallback failed: {e}, trying Gemini")

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
                rag_logger.error(f"[Transcript] Gemini fallback failed: {e}")

        if last_error:
            raise LLMError(
                user_message=get_user_friendly_message(last_error),
                technical_message=str(last_error),
                retryable=is_retryable_error(last_error),
            )
        raise LLMError(
            user_message="Unable to process your request. Please try again.",
            technical_message="No LLM client available",
            retryable=False,
        )

    def _parse_json(self, text: str, default: Dict = None) -> Dict:
        """Strip markdown fences and parse JSON."""
        if default is None:
            default = {}
        try:
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            start = text.find('{')
            end = text.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            rag_logger.warning(f"[Transcript] JSON parse failed: {e}")
        return default

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: PLANNING
    # ═══════════════════════════════════════════════════════════════════════

    async def _plan_queries(self, question: str) -> Dict:
        """
        Generate dense keyword phrases for transcript retrieval.

        Rules (same philosophy as SEC service _plan_investigation):
        - 2–4 dense keyword phrases, NOT question form
        - No company names, tickers, or quarter references (already scoped)
        - Sub-questions are human-readable descriptions of what we need
        """
        prompt = f"""You are a financial analyst creating a SEARCH STRATEGY for earnings call transcripts. Do not use emojis.

QUESTION: {question}

Generate search queries for a hybrid vector + keyword search over earnings transcript chunks.

CRITICAL RULES for search_queries:
- Short dense keyword phrases only — NOT full sentences or questions
- Do NOT include company names, ticker symbols, or quarter/year references
  (the search is already scoped to the correct company and time period)
- 2 to 4 queries covering different aspects of the question

GOOD examples:
- "billings deferred revenue non-GAAP metrics"
- "RPO backlog remaining performance obligations"
- "guidance outlook future quarters revenue"
- "operating expenses headcount hiring"

BAD examples:
- "What were DDOG's billings?" (question form — forbidden)
- "Datadog Q3 2024 billings" (company name + quarter — forbidden)

sub_questions: human-readable descriptions of what data you need to find
search_queries: the actual dense keyword phrases for retrieval

Return ONLY valid JSON:
{{
    "sub_questions": [
        "Descriptive sub-question 1",
        "Descriptive sub-question 2"
    ],
    "search_queries": [
        "dense keyword phrase 1",
        "dense keyword phrase 2"
    ]
}}"""

        messages = [
            {"role": "system", "content": "Financial analyst. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        keyword_fallback = re.sub(
            r'\b(what|how|why|when|where|who|is|are|was|were|did|do|does|the|a|an)\b',
            '', question, flags=re.IGNORECASE
        ).strip()

        try:
            raw = await self._make_llm_call_async(messages, temperature=0.2, max_tokens=1000)
            result = self._parse_json(raw, default={
                "sub_questions": [question],
                "search_queries": [keyword_fallback],
            })
        except Exception as e:
            rag_logger.warning(f"[Transcript] Planning LLM call failed: {e}")
            result = {
                "sub_questions": [question],
                "search_queries": [keyword_fallback],
            }

        rag_logger.info("🧠 TRANSCRIPT PLAN:")
        for sq in result.get("sub_questions", []):
            rag_logger.info(f"   sub_q: {sq}")
        for q in result.get("search_queries", []):
            rag_logger.info(f"   query: {q}")

        return result

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: PARALLEL HYBRID SEARCH
    # ═══════════════════════════════════════════════════════════════════════

    async def _parallel_hybrid_search(
        self,
        search_queries: List[str],
        transcript_searches,
        chunks_per_query: int = 10,
    ) -> List[Dict]:
        """
        Run all (query × ticker × quarter) combinations in parallel.

        Deduplicates by citation index (keeps highest similarity).
        """
        loop = asyncio.get_running_loop()

        tasks = []
        for query in search_queries:
            for ts in transcript_searches:
                ticker = ts.ticker.upper()
                # Prefix $TICKER so search_engine extracts ticker for DB filtering
                scoped_query = f"${ticker} {query}"
                for quarter_str in ts.quarters:
                    tasks.append((scoped_query, quarter_str))

        if not tasks:
            return []

        # Run all searches in a single executor thread (sequential within thread) to avoid
        # Python logging lock contention/deadlock when many threads log simultaneously.
        def run_all_searches() -> List[List[Dict]]:
            all_results = []
            for scoped_query, quarter_str in tasks:
                try:
                    chunks = self.search_engine.search_similar_chunks(
                        scoped_query, max_results=chunks_per_query, target_quarter=quarter_str
                    )
                    all_results.append(chunks)
                except Exception as e:
                    rag_logger.warning(f"[Transcript] search_similar_chunks failed ({scoped_query}, {quarter_str}): {e}")
                    all_results.append([])
            return all_results

        results = await loop.run_in_executor(None, run_all_searches)

        # Merge: deduplicate by citation index, keep highest similarity
        best: Dict[Any, Dict] = {}
        for chunk_list in results:
            for chunk in chunk_list:
                key = chunk.get('citation')
                if key is None:
                    key = id(chunk)
                existing = best.get(key)
                if existing is None or chunk.get('similarity', 0) > existing.get('similarity', 0):
                    best[key] = chunk

        merged = sorted(best.values(), key=lambda c: c.get('similarity', 0), reverse=True)
        rag_logger.info(f"[Transcript] Parallel search returned {len(merged)} unique chunks")
        return merged

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: ANSWER GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    async def _generate_answer(
        self,
        question: str,
        sub_questions: List[str],
        chunks: List[Dict],
        previous_answer: Optional[str] = None,
        chunk_start_idx: int = 1,
    ) -> str:
        """Generate answer with [TC-N] citation markers."""
        context_parts = []
        for i, chunk in enumerate(chunks, chunk_start_idx):
            ticker = chunk.get("ticker", "")
            year = chunk.get("year", "")
            quarter = chunk.get("quarter", "")
            text = (chunk.get("chunk_text") or "")[:2000]
            context_parts.append(f"SOURCE [TC-{i}] [{ticker} Q{quarter} {year}]\n{text}")

        context = "\n\n".join(context_parts) if context_parts else "[NO TRANSCRIPT DATA FOUND]"

        sub_q_lines = "\n".join([f"- {sq}" for sq in sub_questions]) if sub_questions else f"- {question}"

        previous_section = ""
        if previous_answer:
            previous_section = f"""
PREVIOUS ANSWER (improve and expand — do not start from scratch):
{previous_answer}

"""

        prompt = f"""{previous_section}Answer the question using the retrieved earnings transcript passages below.

QUESTION: {question}

WHAT TO FIND:
{sub_q_lines}

RETRIEVED PASSAGES:
{context}

CITATION RULES — CRITICAL:
- ALWAYS cite with EXACT bracket markers: [TC-1], [TC-2], [TC-3], etc.
- NEVER write a bare number like 1. or 6. or 10. without the [TC-] prefix and brackets
- Every fact, number, or metric MUST have a [TC-N] citation
- Do not invent citation numbers — only use numbers that appear in the RETRIEVED PASSAGES above

YEAR FORMATTING — CRITICAL:
- ALWAYS write full 4-digit years: 2023, 2024, 2025 — NEVER abbreviate as 23, 24, 25, 223, 224, 225
- In tables, write "Q3 2023", "Q4 2024" — never "Q3 23" or "Q3 223"

OTHER RULES:
- Provide precise numbers and metrics where available
- **Bold every financial figure** (e.g. **$748M**, **+21%**, **$908M**)
- MANDATORY: If the answer contains data across 2 or more periods or metrics, present it as a markdown table — no bullet lists for multi-period data
- In prose, refer to the company by name or ticker WITHOUT a dollar sign (write "Datadog" or "DDOG", never "$DDOG")
- Note if data is missing or unavailable for specific quarters (include as a row in the table with "N/A")
- If the user's question requests a specific format, follow that format exactly
- Do not use emojis
- No external knowledge — only use the provided passages

End your answer with:

**You might also ask:**
- [Write a specific follow-up question using the actual company name and ticker (no $ prefix, e.g. "Datadog" or "DDOG") — related metric or trend]
- [Write a specific follow-up question using the actual company name and ticker (no $ prefix) — different analytical angle]
- [Write a specific follow-up question using the actual company name and ticker (no $ prefix) — deeper on a key finding]"""

        messages = [
            {"role": "system", "content": "You are a precise financial analyst. Answer only from the provided sources. No emojis. CRITICAL: Always cite facts with [TC-N] bracket markers — never use bare numbers alone as citations."},
            {"role": "user", "content": prompt},
        ]

        try:
            return await self._make_llm_call_async(messages, temperature=0.1, max_tokens=2000)
        except Exception as e:
            rag_logger.error(f"[Transcript] Answer generation failed: {e}")
            return f"Unable to generate answer from transcript data: {e}"

    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: EVALUATION
    # ═══════════════════════════════════════════════════════════════════════

    async def _evaluate_answer(
        self,
        question: str,
        answer: str,
        chunks: List[Dict],
    ) -> Dict:
        """Evaluate answer quality and identify gaps for replanning."""
        chunk_summary = "\n".join([
            f"[TC-{i}] {chunk.get('ticker','')} Q{chunk.get('quarter','')} {chunk.get('year','')}: {(chunk.get('chunk_text') or '')[:200]}"
            for i, chunk in enumerate(chunks[:15], 1)
        ])

        prompt = f"""Evaluate this earnings transcript answer for completeness and accuracy.

QUESTION: {question}

ANSWER:
{answer}

AVAILABLE SOURCES (sample):
{chunk_summary}

Rate from 0.0 to 1.0 and identify what key data is still missing.

Return ONLY valid JSON:
{{
    "quality_score": 0.0,
    "missing_info": ["specific missing data point 1", "specific missing data point 2"],
    "additional_queries": ["dense keyword phrase for missing data 1", "dense keyword phrase for missing data 2"]
}}"""

        messages = [
            {"role": "system", "content": "Financial evaluator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._make_llm_call_async(messages, temperature=0.1, max_tokens=500)
            return self._parse_json(raw, default={"quality_score": 0.7, "missing_info": [], "additional_queries": []})
        except Exception as e:
            rag_logger.warning(f"[Transcript] Evaluation LLM call failed: {e}")
            return {"quality_score": 0.7, "missing_info": [], "additional_queries": []}

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════════

    async def execute_search_async(
        self,
        query: str,
        question_analysis: Dict,
        transcript_searches,
        event_yielder=None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main entry point: plan → search → generate → evaluate → replan (≤2 iterations).

        Yields streaming events; final event type='search_complete'.
        """
        confidence_threshold = 0.85

        yield {'type': 'planning_start', 'message': 'Planning transcript search...', 'data': {'question': query}}

        plan = await self._plan_queries(query)
        sub_questions = plan.get('sub_questions', [query])
        search_queries = plan.get('search_queries', [query])

        yield {
            'type': 'planning_complete',
            'data': {'sub_questions': sub_questions, 'search_queries': search_queries},
        }

        # ── Iterative loop (max 2 iterations) ─────────────────────────────
        accumulated_chunks: List[Dict] = []
        seen_citation_keys = set()
        current_answer: Optional[str] = None
        next_chunk_idx = 1  # Global [TC-i] counter (like SEC service)
        current_queries = search_queries

        for iteration in range(self.MAX_ITERATIONS):
            iteration_num = iteration + 1
            rag_logger.info(f"[Transcript] Iteration {iteration_num}/{self.MAX_ITERATIONS}")

            # Search
            new_chunks_raw = await self._parallel_hybrid_search(current_queries, transcript_searches)

            # Deduplicate against ALL retrieved chunks (seen_citation_keys tracks everything
            # so future iterations don't re-retrieve the same chunks).
            # accumulated_chunks ONLY contains chunks shown to the LLM so that
            # get_citations(accumulated_chunks) produces TC-N numbers that match the answer.
            new_chunks = []
            for chunk in new_chunks_raw:
                key = chunk.get('citation', id(chunk))
                if key not in seen_citation_keys:
                    seen_citation_keys.add(key)
                    new_chunks.append(chunk)

            # Limit to 20 shown to the LLM; only these are added to accumulated_chunks
            chunks_to_show = new_chunks[:20]
            accumulated_chunks.extend(chunks_to_show)

            yield {
                'type': 'retrieval_complete',
                'data': {
                    'iteration': iteration_num,
                    'new_chunks': len(chunks_to_show),
                    'chunks_found': len(accumulated_chunks),
                },
            }

            if not accumulated_chunks:
                rag_logger.warning("[Transcript] No chunks found — stopping early")
                break

            # Skip answer generation if no new chunks were added this iteration
            if not chunks_to_show:
                rag_logger.info("[Transcript] No new chunks this iteration — keeping previous answer")
                break

            # Generate answer using new chunks; previous answer carries earlier [TC-N] refs
            current_answer = await self._generate_answer(
                question=query,
                sub_questions=sub_questions,
                chunks=chunks_to_show,
                previous_answer=current_answer,
                chunk_start_idx=next_chunk_idx,
            )
            next_chunk_idx += len(chunks_to_show)

            rag_logger.info(f"[Transcript] Answer generated ({len(current_answer)} chars)")

            # Evaluate (only replan if more iterations remain)
            if iteration < self.MAX_ITERATIONS - 1:
                evaluation = await self._evaluate_answer(query, current_answer, accumulated_chunks)
                quality_score = evaluation.get('quality_score', 0.7)
                missing_info = evaluation.get('missing_info', [])
                additional_queries = evaluation.get('additional_queries', [])

                rag_logger.info(f"[Transcript] Quality: {quality_score:.2f}, missing: {missing_info}")

                yield {
                    'type': 'evaluation_complete',
                    'data': {
                        'iteration': iteration_num,
                        'quality_score': quality_score,
                        'missing_info': missing_info[:3],
                    },
                }

                if quality_score >= confidence_threshold:
                    rag_logger.info(f"[Transcript] Early stop: quality {quality_score:.2f} >= {confidence_threshold}")
                    break

                # Replan with additional queries from evaluator
                if additional_queries:
                    current_queries = additional_queries
                    rag_logger.info(f"[Transcript] Replanning with {len(current_queries)} new queries: {current_queries}")
                else:
                    rag_logger.info("[Transcript] No replan queries — stopping")
                    break

        yield {
            'type': 'search_complete',
            'data': {
                'answer': current_answer or '',
                'chunks': accumulated_chunks,
                'sub_questions': sub_questions,
            },
        }

    # ═══════════════════════════════════════════════════════════════════════
    # CITATION FORMATTING
    # ═══════════════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════════
    # CITATION REMAPPING  (used by _stage_transcript_search to merge tickers)
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _remap_citations(
        answer: str,
        citations: List[Dict],
        offset: int,
    ) -> tuple:
        """
        Shift [TC-N] markers in answer text and citation dicts by `offset`.

        Returns (remapped_answer, remapped_citations).
        offset=0 → no-op (first ticker keeps its numbers).
        """
        if offset == 0:
            return answer, citations

        def _shift(m):
            return f"[TC-{int(m.group(1)) + offset}]"

        remapped_answer = re.sub(r'\[TC-(\d+)\]', _shift, answer)

        remapped_citations = []
        for c in citations:
            nc = c.copy()
            nc['source_number'] = c['source_number'] + offset
            nc['marker'] = f"[TC-{c['source_number'] + offset}]"
            nc['chunk_id'] = re.sub(
                r'tc_(.+?)_(\d+)$',
                lambda m: f"tc_{m.group(1)}_{int(m.group(2)) + offset}",
                c.get('chunk_id', ''),
            )
            remapped_citations.append(nc)

        return remapped_answer, remapped_citations

    @staticmethod
    def _fix_bare_tc_citations(text: str, max_tc: int) -> str:
        """
        Post-process synthesis output to fix bare citation numbers the LLM emitted
        despite instructions.

        Two patterns fixed:
        1. Concatenated: "1112" → "[TC-11][TC-12]" (split into two valid TC indices)
        2. Standalone at end of clause: "grew 27% 4." → "grew 27% [TC-4]."
        """
        # Step 1: fix concatenated 3-4 digit numbers that split into two valid TC indices
        def expand_concat(m):
            num_str = m.group(1)
            num = int(num_str)
            # Calendar years (1900–2100) are never citation pairs — skip them
            if 1900 <= num <= 2100:
                return m.group(0)
            for split in range(1, len(num_str)):
                left, right = int(num_str[:split]), int(num_str[split:])
                if 1 <= left <= max_tc and 1 <= right <= max_tc:
                    return f"[TC-{left}][TC-{right}]"
            return m.group(0)

        # Only match 3-4 digit numbers not already inside a bracket or preceded by digit, comma, or $
        # Lookbehind for comma and $ prevents matching numbers inside dollar amounts like $1,295,920
        text = re.sub(r'(?<!\[)(?<!TC-)(?<!\d)(?<!,)(?<!\$)(\d{3,4})(?!\d)(?![%,])', expand_concat, text)

        # Step 2: fix standalone 1-2 digit numbers that appear between whitespace and
        # sentence-ending punctuation — very likely bare citation numbers
        def fix_standalone(m):
            n = int(m.group(2))
            if 1 <= n <= max_tc:
                return f"{m.group(1)}[TC-{n}]"
            return m.group(0)

        # Pattern: space + 1-2 digits + (period/newline/end) not preceded by % or $ or word char
        # Comma removed from lookahead: "6,838" must not match (6 before , is part of a number)
        # \w in lookbehind: "January 31" must not match (31 after letter is a date not a citation)
        text = re.sub(r'(?<![%$\d\w])( )(\d{1,2})(?=\s*[.\n]|$)', fix_standalone, text)

        return text

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-TICKER SYNTHESIS
    # ═══════════════════════════════════════════════════════════════════════

    async def synthesize_subagents(
        self,
        question: str,
        subagent_results: List[Dict],
        news_context: Optional[str] = None,
    ) -> str:
        """
        Synthesize answers from multiple subagents (transcript and/or SEC) into one
        coherent answer, preserving all [TC-N] and [10K-N] citation markers exactly.

        subagent_results: [{'type': 'transcript'|'10k', 'ticker': str, 'answer': str, 'citations': [...]}, ...]
        news_context: Optional Tavily news context string (additive, no citation renaming needed)
        """
        from collections import Counter
        ticker_type_counts = Counter((r['ticker'], r.get('type')) for r in subagent_results)

        sections = []
        for r in subagent_results:
            if r.get('type') == 'transcript':
                label = f"{r['ticker']} (Earnings Transcripts)"
            else:
                year = r.get('fiscal_year')
                if year and ticker_type_counts[(r['ticker'], r.get('type'))] > 1:
                    label = f"{r['ticker']} FY{year} (10-K Filing)"
                else:
                    label = f"{r['ticker']} (10-K Filing)"
            sections.append(f"=== {label} ===\n{r['answer']}")

        sources_text = "\n\n".join(sections)

        news_section = ""
        if news_context:
            news_section = f"\n\nADDITIONAL NEWS CONTEXT:\n{news_context}\n"

        # Collect valid TC citation numbers — only transcript citations, NOT 10-K citations.
        # 10-K citation source_numbers must not inflate max_tc or bare 10K numbers get mangled.
        all_tc_indices = set()
        for r in subagent_results:
            for c in r.get('citations', []):
                if c.get('type', '').lower() in ('transcript', 'tc'):
                    all_tc_indices.add(c.get('source_number', 0))
        max_tc = max(all_tc_indices) if all_tc_indices else 0

        prompt = f"""Synthesize the following financial analyses into one comprehensive answer.

QUESTION: {question}

PER-SOURCE ANALYSES:
{sources_text}{news_section}

SYNTHESIS REQUIREMENTS:
- Write a single coherent answer — do NOT reproduce the `=== SOURCE ===` section headers from the input
- Use ## markdown headings to organize your answer into logical sections (by metric, time period, theme, or company comparison)
- Use markdown tables to compare periods or companies where numbers are involved
- If the user's question requests a specific format (e.g. bullet points, table, brief summary, detailed breakdown, numbered list), follow that format exactly

CITATION FORMAT — ABSOLUTE REQUIREMENT:
The input analyses use citation markers like [TC-11], [TC-12], [TC-28], [10K-3].
You MUST copy these EXACTLY — with square brackets, TC- prefix, and the number.

❌ WRONG — NEVER do this:
  - "Azure grew 39% year-over-year 1112"      ← bare concatenated numbers, no brackets
  - "Cloud revenue grew 27% 4"                 ← bare number, no brackets
  - "grew 23% year-over-year 1624"             ← bare numbers, no brackets
  - "revenue was $168B 16 24"                  ← numbers with spaces, no brackets

✅ CORRECT — always do this:
  - "Azure grew 39% year-over-year [TC-11][TC-12]"
  - "Cloud revenue grew 27% [TC-4]"
  - "grew 23% year-over-year [TC-16][TC-24]"
  - "revenue was $168B [TC-16][TC-24]"

Every single fact you write MUST have a [TC-N] or [10K-N] marker with brackets.

OTHER RULES:
- Incorporate any news context naturally where relevant
- Do not use emojis
- Do not cite sources not present in the analyses above

End your answer with:

**You might also ask:**
- [Question specific to the same company/ticker(s) and a related metric or trend — include the $TICKER(s)]
- [Question specific to the same company/ticker(s) from a different analytical angle — include the $TICKER(s)]
- [Question specific to the same company/ticker(s) that goes deeper on a key finding — include the $TICKER(s)]"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial analyst synthesizing multiple data-source analyses. "
                    "CRITICAL: Citation markers like [TC-11], [TC-12], [10K-3] MUST appear with their full brackets and prefix. "
                    "NEVER output bare numbers as citations (e.g. '1112' or '4' or '1624'). "
                    "Always write [TC-11][TC-12] — never 1112."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self._make_llm_call_async(messages, temperature=0.1, max_tokens=5000)
            # Post-process: fix any bare citation numbers the LLM emitted despite instructions
            result = self._fix_bare_tc_citations(result, max_tc)
            return result
        except Exception as e:
            rag_logger.error(f"[Transcript] Subagent synthesis failed: {e}")
            # Fallback: concatenate answers
            return "\n\n".join(
                f"**{r['ticker']}**\n{r['answer']}" for r in subagent_results
            )

    async def synthesize_multi_ticker(
        self,
        question: str,
        ticker_results: List[Dict],
    ) -> str:
        """
        Synthesize per-ticker answers (already citation-remapped) into one
        coherent comparative answer, preserving all [TC-N] markers exactly.

        ticker_results: [{'ticker': str, 'answer': str, 'citations': [...]}, ...]
        """
        sections = "\n\n".join(
            f"=== {r['ticker']} Analysis ===\n{r['answer']}"
            for r in ticker_results
        )

        prompt = f"""Synthesize the following per-company earnings transcript analyses into one comprehensive comparative answer.

QUESTION: {question}

PER-COMPANY ANALYSES:
{sections}

CITATION RULES — CRITICAL:
- Preserve ALL [TC-N] citation markers EXACTLY as they appear — do not renumber or drop any
- NEVER write a bare number like 1. or 6. without the [TC-] prefix and brackets
- Every fact or metric you include MUST carry a [TC-N] citation marker

OTHER RULES:
- Always write a unified narrative — never present the per-company analyses as separate labelled blocks
- Use markdown tables for side-by-side metric comparisons
- If the user's question requests a specific format (e.g. bullet points, brief summary, table, numbered list), follow that format exactly
- Do not use emojis
- Do not cite sources that are not present in the analyses above

End your answer with:

**You might also ask:**
- [Question comparing the same companies on a related metric — include all $TICKERs]
- [Question about one of the companies from a different analytical angle — include the $TICKER]
- [Question that goes deeper on a key finding from the comparison — include the relevant $TICKER(s)]"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial analyst synthesizing comparative earnings call analyses. "
                    "Preserve every [TC-N] citation marker exactly as written."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        max_tc = max(
            (c.get('source_number', 0) for r in ticker_results for c in r.get('citations', [])
             if c.get('type', '').lower() in ('transcript', 'tc')),
            default=0,
        )

        try:
            result = await self._make_llm_call_async(messages, temperature=0.1, max_tokens=2000)
            return self._fix_bare_tc_citations(result, max_tc)
        except Exception as e:
            rag_logger.error(f"[Transcript] Multi-ticker synthesis failed: {e}")
            # Fallback: concatenate answers
            return "\n\n".join(
                f"**{r['ticker']}**\n{r['answer']}" for r in ticker_results
            )

    def get_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Build structured citation list from accumulated chunks."""
        citations = []
        for i, chunk in enumerate(chunks, 1):
            ticker = chunk.get("ticker", "")
            year = chunk.get("year", "")
            quarter = chunk.get("quarter", "")
            chunk_index = chunk.get("citation", i)
            chunk_text = chunk.get("chunk_text") or ""
            citations.append({
                "source_number": i,
                "type": "transcript",
                "marker": f"[TC-{i}]",
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "chunk_text": chunk_text[:500],
                "preview": chunk_text[:200],
                "chunk_index": chunk_index,
                "chunk_id": f"tc_{ticker}_{year}_q{quarter}_{chunk_index}",
                "char_offset": chunk.get("char_offset"),
                "chunk_length": chunk.get("chunk_length"),
            })
        return citations
