"""
Qualitative Company Screener

Two-stage pipeline for screening companies on qualitative criteria
extracted from earnings transcripts and 10-K filings:

1. Optional financial SQL filter -> narrows ticker universe
2. Bulk cross-company RAG search -> finds qualitative matches with LLM-synthesized evidence
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import openai
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class QualitativeScreener:
    """Orchestrates qualitative screening across earnings transcripts and 10-K filings."""

    # Minimum company count for a quarter to be considered "well-populated"
    MIN_COMPANIES_FOR_DEFAULT = 200

    def __init__(self, financial_analyzer, rag_system):
        self.financial_analyzer = financial_analyzer
        self.search_engine = rag_system.search_engine
        self.database_manager = rag_system.database_manager
        self.config = rag_system.config
        self.openai_client = openai.OpenAI()
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def _get_best_default_quarter(self) -> tuple[int, int]:
        """Return (year, quarter) for the latest well-populated quarter."""
        quarter_details = self.config.get("quarter_details", {})
        available = self.config.get("available_quarters", [])
        for qid in available:
            details = quarter_details.get(qid, {})
            if details.get("company_count", 0) >= self.MIN_COMPANIES_FOR_DEFAULT:
                return details["year"], details["quarter"]
        # Fallback: latest quarter regardless
        if available and quarter_details:
            d = quarter_details[available[0]]
            return d["year"], d["quarter"]
        return datetime.now().year - 1, 4

    def _make_reasoning_event(self, event_type: str, message: str, step: str = "reasoning") -> Dict:
        return {
            'type': 'reasoning',
            'event': {
                'event_type': event_type,
                'message': message,
                'details': {'step': step},
                'timestamp': datetime.now().isoformat(),
            }
        }

    # ------------------------------------------------------------------
    # Stage 1: Intent Split (sync, CPU-bound LLM call)
    # ------------------------------------------------------------------
    def _split_intent(self, question: str) -> Dict[str, Any]:
        default_year, default_quarter = self._get_best_default_quarter()

        # For GPT-5 reasoning models: no temperature, no system role, use reasoning_effort
        system_context = (
            "You classify stock-screening queries. Separate them into:\n"
            "- financial: metrics, ratios, sectors, market cap, revenue, P/E, etc. "
            "(answerable with SQL on structured financial data)\n"
            "- qualitative: themes, sentiment, strategy, management commentary "
            "(requires searching text documents like earnings transcripts or 10-K filings)\n\n"
            "Return JSON with these keys:\n"
            '  "mode": "financial_only" | "qualitative_only" | "mixed",\n'
            '  "financial": string or null (natural language description of financial criteria, NOT SQL),\n'
            '  "qualitative": string or null (text-search part),\n'
            '  "source": "transcript" or "10k" (only needed if qualitative is not null),\n'
            '  "time_scope": {"year": int, "quarter": int or null}\n\n'
            "IMPORTANT:\n"
            "- The 'financial' field should be a natural language query, NOT SQL.\n"
            "- The 'qualitative' field should stay CLOSE to the original user phrasing - do NOT over-elaborate or add extra details.\n"
            "- Keep queries concise and preserve the user's intent.\n\n"
            "Examples:\n"
            '- "tech stocks with P/E < 20" → mode: "financial_only", financial: "tech stocks with P/E under 20"\n'
            '- "companies discussing AI capex" → mode: "qualitative_only", qualitative: "discussing AI capex"\n'
            '- "companies building their own foundational models" → mode: "qualitative_only", qualitative: "building their own foundational models"\n'
            '- "tech stocks with revenue > $50B investing in AI" → mode: "mixed", '
            'financial: "tech stocks with revenue greater than 50 billion", qualitative: "investing in AI"\n'
            '- "top 10 stocks by market cap" → mode: "financial_only"\n\n'
            f"Default year to {default_year} and quarter to {default_quarter} "
            "if not specified. Use quarter=null for 10-K filings."
        )

        response = self.openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {
                    "role": "user",
                    "content": f"{system_context}\n\nQuery: {question}\n\nProvide your response as JSON only."
                },
            ],
        )
        return json.loads(response.choices[0].message.content)

    # ------------------------------------------------------------------
    # Stage 5: LLM Evidence Summary (sync, called via asyncio.to_thread)
    # ------------------------------------------------------------------
    def _summarize_evidence(self, ticker: str, source: str, qualitative_query: str, chunks: List[Dict]) -> Dict[str, Any]:
        excerpts = "\n---\n".join(c['chunk_text'][:600] for c in chunks[:3])

        # For GPT-5 reasoning models: no temperature, no system role, use reasoning_effort and max_completion_tokens
        prompt = (
            f"Given excerpts from {ticker}'s {source}, evaluate how well this company matches the query: '{qualitative_query}'.\n\n"
            f"Excerpts:\n{excerpts}\n\n"
            "Return JSON with:\n"
            '1. "relevance_score": 0-100 (how well it matches - be strict, only 70+ for strong matches)\n'
            '2. "evidence": 1-2 sentence explanation with specific quotes/numbers\n\n'
            "If it doesn't match well, score it low (20-40) and explain why."
        )

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                max_completion_tokens=4000,  # Increased for complex reasoning
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            # Log the response structure for debugging
            logger.info(f"Response for {ticker}: choices={len(response.choices)}, finish_reason={response.choices[0].finish_reason if response.choices else 'N/A'}")

            content = response.choices[0].message.content
            if not content:
                logger.warning(f"Empty content from gpt-5-nano for {ticker}. Full response: {response.model_dump()}")
                return {'relevance_score': 0, 'evidence': 'Unable to generate summary'}

            # Parse JSON response
            try:
                result = json.loads(content.strip())
                return {
                    'relevance_score': result.get('relevance_score', 0),
                    'evidence': result.get('evidence', content.strip())
                }
            except json.JSONDecodeError:
                # Fallback if not valid JSON
                logger.warning(f"Non-JSON response for {ticker}, using raw content")
                return {'relevance_score': 50, 'evidence': content.strip()}

        except Exception as e:
            logger.error(f"Error summarizing evidence for {ticker}: {e}", exc_info=True)
            return {'relevance_score': 0, 'evidence': f"Error: {str(e)}"}

    # ------------------------------------------------------------------
    # Main streaming pipeline (async generator)
    # ------------------------------------------------------------------
    async def screen_with_streaming(self, question: str, top_n: int = 20, page: int = 1, page_size: Optional[int] = None) -> AsyncGenerator[Dict, None]:
        """Async generator yielding SSE events. Auto-detects financial vs qualitative vs mixed."""
        start_time = time.time()

        # --- Stage 1: Intent Split ---
        yield self._make_reasoning_event("step_start", "Understanding your question...", "intent_split")
        try:
            intent = await asyncio.to_thread(self._split_intent, question)
        except Exception as e:
            logger.error(f"Intent split failed: {e}")
            yield self._make_reasoning_event("step_error", f"Failed to analyze query: {e}", "intent_split")
            yield {'type': 'error', 'message': f"Failed to analyze query: {e}"}
            return

        mode = intent.get("mode", "financial_only")
        financial_part = intent.get("financial")
        qualitative_part = intent.get("qualitative")
        source = intent.get("source", "transcript")
        default_year, default_quarter = self._get_best_default_quarter()
        time_scope = intent.get("time_scope", {})
        year = time_scope.get("year", default_year)
        quarter = time_scope.get("quarter", default_quarter)

        # --- Pure financial: delegate to existing financial streamer ---
        if mode == "financial_only" or not qualitative_part:
            yield self._make_reasoning_event(
                "step_complete",
                "Detected pure financial query \u2014 using financial screener",
                "intent_split",
            )
            for event in self.financial_analyzer.query_with_streaming(
                question=question,
                page=page,
                page_size=page_size,
            ):
                yield event
            return

        source_label = "earnings transcripts" if source == "transcript" else "10-K filings"
        time_label = f"FY{year}" if quarter is None else f"Q{quarter} {year}"
        # Keep it simple - show original query, not rephrased version
        if financial_part and qualitative_part:
            search_desc = f"Filtering by financials, then searching {source_label.lower()}"
        elif qualitative_part:
            search_desc = f"Searching {source_label.lower()}"
        else:
            search_desc = "Processing query"
        yield self._make_reasoning_event(
            "step_complete",
            search_desc,
            "intent_split",
        )

        # --- Stage 2: Financial Filter (optional, for mixed queries) ---
        ticker_list: List[str] = []
        if financial_part:
            yield self._make_reasoning_event("step_start", f"Filtering companies by {financial_part.lower()}...", "financial_filter")
            try:
                fin_result = await asyncio.to_thread(
                    self.financial_analyzer.query, financial_part, 1, None
                )
                if fin_result and not fin_result.get("error") and fin_result.get("data_rows"):
                    for row in fin_result["data_rows"]:
                        sym = row.get("symbol") or row.get("Symbol") or row.get("ticker") or row.get("Ticker")
                        if sym:
                            ticker_list.append(str(sym).upper())

                if len(ticker_list) >= 3:
                    yield self._make_reasoning_event(
                        "step_complete",
                        f"Found {len(ticker_list)} companies matching financial criteria",
                        "financial_filter",
                    )
                else:
                    # Too few results — likely a bad filter. Search all companies instead.
                    n_found = len(ticker_list)
                    ticker_list = []
                    yield self._make_reasoning_event(
                        "step_complete",
                        f"Expanding search to include all companies for better results",
                        "financial_filter",
                    )
            except Exception as e:
                logger.warning(f"Financial filter failed, proceeding without: {e}")
                yield self._make_reasoning_event(
                    "step_error",
                    f"Searching all companies for your criteria",
                    "financial_filter",
                )

        # --- Stage 3: Bulk RAG Search ---
        company_scope = f"{len(ticker_list)} companies" if ticker_list else "all companies"
        yield self._make_reasoning_event(
            "step_start",
            f"Searching {source_label.lower()} across {company_scope}...",
            "bulk_search",
        )

        # Encode query embedding (CPU-bound, run in thread)
        query_embedding = await asyncio.to_thread(
            self.search_engine.embedding_model.encode, [qualitative_part]
        )
        query_embedding = query_embedding[0]

        # Async bulk search — fetch 100 chunks per company for cross-encoder re-ranking
        if source == "transcript":
            search_quarter = quarter if quarter else 4
            company_chunks = await self.database_manager.bulk_search_transcripts_async(
                query_embedding=query_embedding,
                tickers=ticker_list,
                year=year,
                quarter=search_quarter,
                chunks_per_company=100,  # Fetch 100 for cross-encoder
            )
        else:
            company_chunks = await self.database_manager.bulk_search_10k_async(
                query_embedding=query_embedding,
                tickers=ticker_list,
                fiscal_year=year,
                chunks_per_company=100,  # Fetch 100 for cross-encoder
            )

        yield self._make_reasoning_event(
            "step_complete",
            f"Found {len(company_chunks)} companies with relevant information",
            "bulk_search",
        )

        if not company_chunks:
            yield self._make_reasoning_event("step_error", "No companies found matching your criteria", "aggregate")
            yield {
                'type': 'result',
                'data': {
                    'success': True,
                    'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                    'friendly_columns': {
                        'symbol': 'Symbol',
                        'relevance_score': 'Relevance',
                        'evidence_summary': 'Evidence',
                        'citations': 'Citations',
                    },
                    'data_rows': [],
                    'message': 'No companies matched the qualitative criteria.',
                },
            }
            return

        # --- Stage 3.5: Cross-Encoder Re-ranking ---
        yield self._make_reasoning_event(
            "step_start",
            f"Refining results for better accuracy...",
            "cross_encoder",
        )

        # Re-rank chunks within each company using cross-encoder
        reranked_chunks = {}
        for ticker, chunks in company_chunks.items():
            if not chunks:
                continue

            # Prepare pairs: (query, chunk_text)
            pairs = [[qualitative_part, chunk['chunk_text']] for chunk in chunks]

            # Get cross-encoder scores (CPU-bound, run in thread)
            ce_scores = await asyncio.to_thread(self.cross_encoder.predict, pairs)

            # Add scores and sort
            for i, chunk in enumerate(chunks):
                chunk['cross_encoder_score'] = float(ce_scores[i])

            # Keep top 20 chunks per company
            chunks_sorted = sorted(chunks, key=lambda x: x['cross_encoder_score'], reverse=True)
            reranked_chunks[ticker] = chunks_sorted[:20]

        company_chunks = reranked_chunks

        yield self._make_reasoning_event(
            "step_complete",
            f"Results refined successfully",
            "cross_encoder",
        )

        # --- Stream Partial Results #1: Initial companies with cross-encoder scores ---
        initial_rows = []
        for ticker, chunks in company_chunks.items():
            if chunks:
                # Don't show cross-encoder scores - they'll be replaced by LLM scores
                initial_rows.append({
                    'symbol': ticker,
                    'relevance_score': None,  # Will be filled by LLM
                    'evidence_summary': '...',  # Placeholder
                    'citations': [],
                })

        # Sort and yield initial partial result (None scores last; higher score first)
        initial_rows.sort(key=lambda x: (x['relevance_score'] is None, -(x['relevance_score'] or 0)))
        yield {
            'type': 'partial_result',
            'data': {
                'success': True,
                'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                'friendly_columns': {
                    'symbol': 'Symbol',
                    'relevance_score': 'Relevance',
                    'evidence_summary': 'Evidence',
                    'citations': 'Citations',
                },
                'data_rows': initial_rows[:top_n],  # Show top N immediately
                'stage': 'cross_encoder_complete',
            },
        }

        # --- Stage 4: Aggregate & Rank ---
        yield self._make_reasoning_event("step_start", "Analyzing and ranking results...", "aggregate")

        scored: List[Dict] = []
        for ticker, chunks in company_chunks.items():
            # Use cross-encoder scores for better ranking
            avg_sim = sum(c.get('cross_encoder_score', c.get('similarity', 0)) for c in chunks) / len(chunks)
            scored.append({
                'ticker': ticker,
                'avg_similarity': avg_sim,
                'top_excerpt': chunks[0]['chunk_text'][:300] if chunks else '',
                'chunks': chunks,
            })

        scored.sort(key=lambda x: x['avg_similarity'], reverse=True)
        top_companies = scored[:top_n]

        yield self._make_reasoning_event(
            "step_complete",
            f"Identified top {len(top_companies)} most relevant companies",
            "aggregate",
        )

        # --- Stage 4b: Deep Hybrid Search (Pass 2) ---
        yield self._make_reasoning_event(
            "step_start",
            f"Performing detailed analysis on top {len(top_companies)} companies...",
            "deep_search",
        )

        target_quarter = f"{year}_q{quarter}" if quarter else f"{year}_q4"

        async def _deep_search_one(entry: Dict) -> Dict:
            """Run hybrid search for one company, return updated entry."""
            ticker = entry['ticker']
            try:
                if source == "transcript":
                    deep_chunks = await self.search_engine.follow_up_search_async(
                        question=qualitative_part,
                        has_tickers=True,
                        is_general_question=False,
                        is_multi_ticker=False,
                        tickers_to_process=[ticker],
                        target_quarter=target_quarter,
                        target_quarters=[target_quarter],
                    )
                else:
                    # For 10-K, use the async 10-K search
                    deep_chunks = await self.database_manager.search_10k_filings_async(
                        query_embedding=query_embedding,
                        ticker=ticker,
                        fiscal_year=year,
                    )

                if deep_chunks:
                    # Recompute similarity from the richer hybrid results
                    avg_sim = sum(c.get('similarity', 1 - c.get('distance', 0.5)) for c in deep_chunks) / len(deep_chunks)
                    best_text = deep_chunks[0].get('chunk_text', '')[:300]
                    return {
                        **entry,
                        'avg_similarity': avg_sim,
                        'top_excerpt': best_text,
                        'chunks': deep_chunks[:5],  # Keep top 5 for summarization
                    }
            except Exception as e:
                logger.warning(f"Deep search failed for {ticker}: {e}")
            return entry  # Fallback to Pass 1 results

        deep_results = await asyncio.gather(
            *[_deep_search_one(entry) for entry in top_companies],
            return_exceptions=True,
        )

        for i, result in enumerate(deep_results):
            if isinstance(result, Exception):
                logger.warning(f"Deep search exception for {top_companies[i]['ticker']}: {result}")
            else:
                top_companies[i] = result

        # Re-sort after deep search (scores may have changed)
        top_companies.sort(key=lambda x: x['avg_similarity'], reverse=True)

        yield self._make_reasoning_event(
            "step_complete",
            f"Detailed analysis complete",
            "deep_search",
        )

        # --- Stream Partial Results #2: Updated scores after deep search ---
        partial_rows_refined = []
        for entry in top_companies:
            partial_rows_refined.append({
                'symbol': entry['ticker'],
                'relevance_score': None,  # Will be filled by LLM scoring
                'evidence_summary': '...',  # Still generating
                'citations': [],
            })

        yield {
            'type': 'partial_result',
            'data': {
                'success': True,
                'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                'friendly_columns': {
                    'symbol': 'Symbol',
                    'relevance_score': 'Relevance',
                    'evidence_summary': 'Evidence',
                    'citations': 'Citations',
                },
                'data_rows': partial_rows_refined,
                'stage': 'deep_search_complete',
            },
        }

        # --- Stage 5: LLM Evidence Summary (parallel via threads) ---
        yield self._make_reasoning_event(
            "step_start",
            f"Preparing results with supporting evidence...",
            "summarize",
        )

        tasks = [
            asyncio.to_thread(
                self._summarize_evidence,
                entry['ticker'],
                source_label,
                qualitative_part,
                entry['chunks'],
            )
            for entry in top_companies
        ]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        for i, summary in enumerate(summaries):
            if isinstance(summary, Exception):
                logger.error(f"Failed to generate summary for {top_companies[i]['ticker']}: {summary}")
                top_companies[i]['evidence_summary'] = "Unable to generate summary - please try again"
                top_companies[i]['llm_relevance_score'] = 0
            elif isinstance(summary, dict):
                top_companies[i]['evidence_summary'] = summary.get('evidence', 'No summary available')
                top_companies[i]['llm_relevance_score'] = summary.get('relevance_score', 0)
            else:
                logger.warning(f"Unexpected summary format for {top_companies[i]['ticker']}")
                top_companies[i]['evidence_summary'] = str(summary) if summary else "No summary available"
                top_companies[i]['llm_relevance_score'] = 0

        # Re-sort by LLM relevance score (more accurate than cross-encoder)
        top_companies.sort(key=lambda x: x.get('llm_relevance_score', 0), reverse=True)

        yield self._make_reasoning_event("step_complete", "Results ready", "summarize")

        # --- Stage 6: Yield Result ---
        elapsed = time.time() - start_time
        data_rows = []
        for entry in top_companies:
            # Format sources for citation display
            sources = []
            for idx, chunk in enumerate(entry.get('chunks', [])[:5], 1):  # Top 5 chunks
                sources.append({
                    'type': source,  # 'transcript' or '10k'
                    'ticker': entry['ticker'],
                    'chunk_text': chunk.get('chunk_text', ''),
                    'similarity': chunk.get('similarity', chunk.get('distance', 0)),
                    'marker': f"[{idx}]",
                    'year': year,
                    'quarter': quarter if source == 'transcript' else None,
                    'fiscal_year': year if source == '10k' else None,
                    'section': chunk.get('section'),
                })

            data_rows.append({
                'symbol': entry['ticker'],
                'relevance_score': entry.get('llm_relevance_score', 0) / 100,  # Convert 0-100 to 0-1
                'evidence_summary': entry.get('evidence_summary', ''),
                'citations': sources,  # Citations in separate column
            })

        yield {
            'type': 'result',
            'data': {
                'success': True,
                'columns': ['symbol', 'relevance_score', 'evidence_summary', 'citations'],
                'friendly_columns': {
                    'symbol': 'Symbol',
                    'relevance_score': 'Relevance',
                    'evidence_summary': 'Evidence',
                    'citations': 'Citations',
                },
                'data_rows': data_rows,
                'message': f'Found {len(data_rows)} companies matching qualitative criteria ({elapsed:.1f}s)',
                'execution_time': elapsed,
            },
        }
