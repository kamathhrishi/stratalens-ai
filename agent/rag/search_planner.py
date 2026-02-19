#!/usr/bin/env python3
"""
Search Planner for the RAG System

This module handles strategic planning for search execution:
- Decides which data sources to use (transcripts, 10-K, news)
- Resolves temporal references to actual quarters/years
- Generates search queries per data source
- Creates execution plan with reasoning

Responsibilities:
- WHAT to search (data sources, tickers, quarters, queries)
- NOT HOW to search (that's SearchEngine and SECFilingsService)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from agent.llm import get_llm, LLMClient
from agent.rag.database_manager import DatabaseConnectionError

try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


# ============================================================================
# DATA CLASSES - Search Plan Structure
# ============================================================================

@dataclass
class TranscriptSearch:
    """Search specification for earnings transcripts"""
    ticker: str
    quarters: List[str]  # e.g., ["2024_q4", "2024_q3"]
    query: str  # Natural language search query

    def __repr__(self):
        return f"TranscriptSearch(ticker={self.ticker}, quarters={self.quarters}, query='{self.query[:50]}...')"


@dataclass
class TenKSearch:
    """Search specification for 10-K SEC filings"""
    ticker: str
    year: int  # e.g., 2024
    query: str  # Search query (SECFilingsService handles section routing)

    def __repr__(self):
        return f"TenKSearch(ticker={self.ticker}, year={self.year}, query='{self.query[:50]}...')"


@dataclass
class NewsSearch:
    """Search specification for latest news"""
    query: str  # Keywords for news search

    def __repr__(self):
        return f"NewsSearch(query='{self.query[:50]}...')"


@dataclass
class SearchPlan:
    """
    Complete search plan specifying WHAT to search.

    This is a declarative plan - just lists the searches to perform.
    The executor (RAG Agent) decides HOW to run them (parallel, sequential, etc.)
    """
    earnings_transcripts: List[TranscriptSearch] = field(default_factory=list)
    ten_k: List[TenKSearch] = field(default_factory=list)
    news: List[NewsSearch] = field(default_factory=list)

    reasoning: str = ""  # Human-readable explanation of the plan

    def has_transcripts(self) -> bool:
        """Check if plan includes transcript searches"""
        return len(self.earnings_transcripts) > 0

    def has_10k(self) -> bool:
        """Check if plan includes 10-K searches"""
        return len(self.ten_k) > 0

    def has_news(self) -> bool:
        """Check if plan includes news searches"""
        return len(self.news) > 0

    def is_empty(self) -> bool:
        """Check if plan has no searches"""
        return self.total_searches() == 0

    def total_searches(self) -> int:
        """Total number of searches across all data sources"""
        return len(self.earnings_transcripts) + len(self.ten_k) + len(self.news)

    def __repr__(self):
        return (
            f"SearchPlan(\n"
            f"  transcripts={len(self.earnings_transcripts)}, "
            f"  10k={len(self.ten_k)}, "
            f"  news={len(self.news)}\n"
            f"  reasoning='{self.reasoning[:100]}...'\n"
            f")"
        )


# ============================================================================
# SEARCH PLANNER - Strategic Decision Maker
# ============================================================================

class SearchPlanner:
    """
    Creates strategic search plans based on question analysis.

    Responsibilities:
    1. Decide which data sources to use (transcripts, 10-K, news, hybrid)
    2. Resolve temporal references (Q4 2024 → 2024_q4)
    3. Generate appropriate search queries per data source
    4. Create declarative search plan

    Does NOT:
    - Decide search engine parameters (chunks, weights, etc.)
    - Decide execution strategy (parallel, sequential, etc.)
    - Execute searches (that's SearchEngine/SECFilingsService)
    """

    def __init__(self, database_manager, config=None, llm: Optional[LLMClient] = None):
        """
        Initialize SearchPlanner.

        Args:
            database_manager: DatabaseManager for checking data availability
            config: Optional Config object (for quarter info, etc.)
            llm: Optional LLM client. If None, uses get_llm(config).
        """
        self.database_manager = database_manager
        self.config = config
        self.llm = llm if llm is not None else get_llm(config)

        rag_logger.info("✅ SearchPlanner initialized")

    def create_plan(
        self,
        question_analysis: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> SearchPlan:
        """
        Create search plan given question analysis.

        Args:
            question_analysis: Analysis from QuestionAnalyzer with:
                - tickers: List[str]
                - time_refs: List[str] (unresolved, e.g., ["Q4 2024", "latest"])
                - topic: str
                - question_type: str
                - user_hints: Dict (optional explicit user requests)
                - confidence: float
            user_preferences: Optional user preferences (not used yet)

        Returns:
            SearchPlan: Declarative plan with searches per data source
        """
        rag_logger.info("🎯 ===== SEARCH PLANNING STARTING =====")
        rag_logger.info(f"📋 Question Analysis: tickers={question_analysis.get('tickers')}, "
                       f"type={question_analysis.get('question_type')}, "
                       f"topic={question_analysis.get('topic')}, "
                       f"time_refs={question_analysis.get('time_refs')}")

        # Extract from analysis
        tickers = question_analysis.get('tickers', question_analysis.get('extracted_tickers', []))
        time_refs = question_analysis.get('time_refs', [])
        topic = question_analysis.get('topic', '')
        question_type = question_analysis.get('question_type', '')
        user_hints = question_analysis.get('user_hints', {})
        original_question = question_analysis.get('original_question', '')

        rag_logger.info(f"🔍 QUARTER SELECTION DEBUG - Input time_refs: {time_refs}, tickers: {tickers}")

        # 1. Check what data is available
        available_data = self._check_availability(tickers, time_refs)

        # 2. Decide which data sources to use (LLM-based, no keywords)
        data_sources = self._decide_data_sources(
            topic,
            question_type,
            user_hints,
            user_preferences,
            original_question
        )

        rag_logger.info(f"📊 Data sources selected: {data_sources}")

        # 3. Resolve temporal references to actual quarters/years
        resolved_time = self._resolve_temporal_references(
            time_refs,
            tickers,
            available_data
        )

        rag_logger.info(f"📅 FINAL TEMPORAL RESOLUTION: {resolved_time}")
        rag_logger.info(f"📅 FINAL QUARTERS TO SEARCH: {resolved_time.get('quarters', [])}")

        # 4. Generate searches per data source
        transcript_searches = []
        ten_k_searches = []
        news_searches = []

        if 'earnings_transcripts' in data_sources:
            transcript_searches = self._generate_transcript_searches(
                tickers,
                resolved_time.get('quarters', []),
                topic,
                question_type
            )

        if '10k' in data_sources:
            ten_k_searches = self._generate_10k_searches(
                tickers,
                resolved_time.get('years', []),
                topic,
                question_type,
                available_data=available_data
            )

        if 'news' in data_sources:
            news_searches = self._generate_news_searches(
                tickers,
                topic,
                question_type
            )

        # 5. Generate reasoning
        reasoning = self._generate_reasoning(
            question_type,
            data_sources,
            tickers,
            resolved_time,
            transcript_searches,
            ten_k_searches,
            news_searches
        )

        # 6. Create search plan
        plan = SearchPlan(
            earnings_transcripts=transcript_searches,
            ten_k=ten_k_searches,
            news=news_searches,
            reasoning=reasoning
        )

        rag_logger.info("🎯 ===== SEARCH PLAN CREATED =====")
        rag_logger.info(f"📊 Total searches: {plan.total_searches()} "
                       f"(transcripts={len(transcript_searches)}, "
                       f"10k={len(ten_k_searches)}, "
                       f"news={len(news_searches)})")
        rag_logger.info(f"💭 Reasoning: {reasoning}")

        return plan

    # ========================================================================
    # HELPER METHODS - Data Availability
    # ========================================================================

    def _check_availability(
        self,
        tickers: List[str],
        time_refs: List[str]
    ) -> Dict[str, Any]:
        """
        Check what data is available in the database.

        Args:
            tickers: List of ticker symbols
            time_refs: List of temporal references (unresolved)

        Returns:
            Dict with:
                - quarters: Dict[ticker, List[quarter_ids]]
                - ten_k_years: Dict[ticker, List[years]]
        """
        available_quarters = {}
        available_10k_years = {}

        for ticker in tickers:
            # Get available quarters from database
            if self.database_manager:
                try:
                    quarters = self.database_manager.get_last_n_quarters_for_company(ticker, 50)
                    available_quarters[ticker] = quarters or []

                    # Extract years from quarters for 10-K availability
                    years = list(set(int(q.split('_')[0]) for q in quarters if '_' in q))
                    years.sort(reverse=True)
                    available_10k_years[ticker] = years
                except DatabaseConnectionError:
                    # Re-raise database connection errors - don't silently fail
                    raise
            else:
                available_quarters[ticker] = []
                available_10k_years[ticker] = []

        return {
            'quarters': available_quarters,
            'ten_k_years': available_10k_years
        }

    # ========================================================================
    # HELPER METHODS - Data Source Selection
    # ========================================================================

    def _decide_data_sources(
        self,
        topic: str,
        question_type: str,
        user_hints: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]],
        original_question: str = ""
    ) -> List[str]:
        """
        LLM-based decision on which data sources to use.
        NO keyword matching - pure LLM reasoning.

        Args:
            topic: Question topic/subject
            question_type: Type of question
            user_hints: Explicit user requests
            user_preferences: Optional user preferences
            original_question: The original user question

        Returns:
            List of data sources: ['earnings_transcripts', '10k', 'news']
        """
        # Check for explicit user hints first (highest priority)
        if user_hints.get('data_source'):
            hint = user_hints['data_source']
            if hint == 'hybrid':
                sources = ['earnings_transcripts', '10k']
            elif hint in ['10k', 'news', 'earnings_transcripts']:
                sources = [hint]
            rag_logger.info(f"📌 Using user-specified data source: {hint}")
            return sources

        if not self.llm.is_available():
            rag_logger.warning("⚠️ LLM not available, falling back to earnings_transcripts")
            return ['earnings_transcripts']

        prompt = f"""You are a data source router for a financial RAG system. Determine which data sources to use for this question.

QUESTION: {original_question}
TOPIC: {topic}
QUESTION_TYPE: {question_type}

AVAILABLE DATA SOURCES:
1. earnings_transcripts - Quarterly earnings call transcripts with management commentary, Q&A, guidance
2. 10k - Annual SEC 10-K filings with audited financials, risk factors, business descriptions, detailed financial statements
3. news - Latest news articles and market updates

INSTRUCTIONS:
- If question asks about quarterly results, guidance, earnings calls, management commentary → earnings_transcripts
- If question asks about annual financials, 10-K, risk factors, audited statements, balance sheets, full year data → 10k
- If question asks about latest news, recent developments, breaking news → news
- If question needs BOTH quarterly commentary AND annual filings → both earnings_transcripts and 10k
- When in doubt about needing 10-K data, include it along with transcripts

Return ONLY valid JSON with this structure:
{{
  "data_sources": ["earnings_transcripts"],  // or ["10k"], ["news"], or ["earnings_transcripts", "10k"], etc.
  "reasoning": "Brief explanation of why these sources"
}}"""

        try:
            content = self.llm.complete(
                [
                    {"role": "system", "content": "You are a data source router. Respond with valid JSON only. Do not use emojis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300,
                stream=False,
            )
            result = json.loads(content)
            sources = result.get("data_sources", ["earnings_transcripts"])
            reasoning = result.get("reasoning", "")

            rag_logger.info(f"🤖 LLM selected data sources: {sources}")
            rag_logger.info(f"💭 Reasoning: {reasoning}")

            return sources

        except Exception as e:
            rag_logger.error(f"❌ LLM data source routing failed: {e}, defaulting to earnings_transcripts")
            return ['earnings_transcripts']

    # ========================================================================
    # HELPER METHODS - Temporal Resolution
    # ========================================================================

    def _resolve_temporal_references(
        self,
        time_refs: List[str],
        tickers: List[str],
        available_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve temporal references to actual quarter IDs and years.

        Args:
            time_refs: List of temporal references (e.g., ["Q4 2024", "latest"])
            tickers: List of tickers
            available_data: Available quarters and years from _check_availability

        Returns:
            Dict with:
                - quarters: List[str] - resolved quarter IDs
                - years: List[int] - resolved years for 10-K
                - context: str - "specific", "multiple", "latest"
        """
        # If no time refs, default to latest
        if not time_refs:
            return self._resolve_latest(tickers, available_data)

        # Try to parse each time reference
        quarters = []
        years = []
        context = "specific"

        for ref in time_refs:
            ref_lower = ref.lower() if isinstance(ref, str) else ""

            # Latest/recent
            if ref_lower in ['latest', 'most recent', 'current']:
                latest = self._resolve_latest(tickers, available_data)
                quarters.extend(latest.get('quarters', []))
                years.extend(latest.get('years', []))
                context = "latest"

            # Multiple quarters (e.g., "last 3 quarters") - CHECK BEFORE 'q' match!
            elif 'last' in ref_lower or 'past' in ref_lower:
                count = self._extract_number(ref_lower)
                logger.info(f"🔍 Detected 'last/past' in time_ref: '{ref}', extracted count: {count}")

                if not count:
                    logger.error(f"❌ Failed to extract count from: '{ref}'")
                    # Don't fail hard - try to fallback to latest
                    continue

                if not tickers or not self.database_manager:
                    logger.error(f"❌ Cannot resolve 'last {count}' - missing tickers or database_manager")
                    # Don't fail hard - will fallback to latest at end of loop
                    continue

                logger.info(f"🔍 QUARTER SELECTION: Calling get_last_n_quarters_for_company('{tickers[0]}', {count})")
                ticker_quarters = self.database_manager.get_last_n_quarters_for_company(
                    tickers[0], count
                )
                logger.info(f"✅ QUARTER SELECTION: Returned quarters for {tickers[0]}: {ticker_quarters}")
                rag_logger.info(f"✅ QUARTER SELECTION: Database returned {len(ticker_quarters) if ticker_quarters else 0} quarters: {ticker_quarters}")

                # ✅ CRITICAL FIX: Validate that we got quarters
                if not ticker_quarters:
                    logger.error(f"❌ No quarters found for {tickers[0]} in database - check data ingestion")
                    # Don't fail hard - fallback to latest
                    continue

                quarters.extend(ticker_quarters)
                context = "multiple"

            # Specific quarter (e.g., "Q4 2024", "2024 Q4")
            elif 'q' in ref_lower and any(char.isdigit() for char in ref_lower):
                parsed_quarter = self._parse_quarter_reference(ref)
                if parsed_quarter:
                    quarters.append(parsed_quarter)

            # Year range (e.g., "2020 to 2024", "2020-2024", "from 2020 to 2024") for multi-year 10-K
            elif self._is_year_range_ref(ref_lower):
                range_years = self._parse_year_range(ref_lower)
                if range_years:
                    years.extend(range_years)
                    for y in range_years:
                        quarters.extend([f"{y}_q{q}" for q in [4, 3, 2, 1]])
                    context = "multiple"
                    rag_logger.info(f"📅 Resolved year range '{ref}' → years {range_years}")

            # Year only (e.g., "2024")
            elif ref_lower.isdigit() and len(ref_lower) == 4:
                year = int(ref_lower)
                years.append(year)
                # Add all quarters of that year
                year_quarters = [f"{year}_q{q}" for q in [4, 3, 2, 1]]
                quarters.extend(year_quarters)
                context = "multiple"

            # Catch remaining cases
            elif False:
                count = self._extract_number(ref_lower)
                logger.info(f"🔍 Detected 'last/past' in time_ref: '{ref}', extracted count: {count}")

                if not count:
                    logger.error(f"❌ Failed to extract count from: '{ref}'")
                    # Don't fail hard - try to fallback to latest
                    continue

                if not tickers or not self.database_manager:
                    logger.error(f"❌ Cannot resolve 'last {count}' - missing tickers or database_manager")
                    # Don't fail hard - will fallback to latest at end of loop
                    continue

                logger.info(f"🔍 QUARTER SELECTION: Calling get_last_n_quarters_for_company('{tickers[0]}', {count})")
                ticker_quarters = self.database_manager.get_last_n_quarters_for_company(
                    tickers[0], count
                )
                logger.info(f"✅ QUARTER SELECTION: Returned quarters for {tickers[0]}: {ticker_quarters}")
                rag_logger.info(f"✅ QUARTER SELECTION: Database returned {len(ticker_quarters) if ticker_quarters else 0} quarters: {ticker_quarters}")

                # ✅ CRITICAL FIX: Validate that we got quarters
                if not ticker_quarters:
                    logger.error(f"❌ No quarters found for {tickers[0]} in database - check data ingestion")
                    # Don't fail hard - fallback to latest
                    continue

                quarters.extend(ticker_quarters)
                context = "multiple"

        # Deduplicate and sort
        quarters = sorted(set(quarters), reverse=True)
        years = sorted(set(years), reverse=True)

        # Extract years from quarters if not already set
        if quarters and not years:
            years = sorted(set(int(q.split('_')[0]) for q in quarters if '_' in q), reverse=True)

        return {
            'quarters': quarters,
            'years': years,
            'context': context
        }

    def _resolve_latest(
        self,
        tickers: List[str],
        available_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve 'latest' to actual latest quarter/year"""
        quarters = []
        years = []

        if tickers:
            for ticker in tickers:
                ticker_quarters = available_data['quarters'].get(ticker, [])
                if ticker_quarters:
                    quarters.append(ticker_quarters[0])  # Latest quarter
                    year = int(ticker_quarters[0].split('_')[0])
                    years.append(year)

        return {
            'quarters': sorted(set(quarters), reverse=True),
            'years': sorted(set(years), reverse=True),
            'context': 'latest'
        }

    def _parse_quarter_reference(self, ref: str) -> Optional[str]:
        """Parse quarter reference like 'Q4 2024' to '2024_q4'"""
        import re

        # Try pattern: Q4 2024 or 2024 Q4
        match = re.search(r'q(\d)[,\s]*(\d{4})|(\d{4})[,\s]*q(\d)', ref.lower())
        if match:
            if match.group(1) and match.group(2):
                quarter = match.group(1)
                year = match.group(2)
            else:
                year = match.group(3)
                quarter = match.group(4)
            return f"{year}_q{quarter}"

        return None

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from text like 'last 3 quarters' → 3"""
        import re
        match = re.search(r'\d+', text)
        return int(match.group()) if match else None

    def _is_year_range_ref(self, ref: str) -> bool:
        """True if ref looks like a year range: '2020 to 2024', '2020-2024', 'from 2020 to 2024'."""
        import re
        if not ref or len(ref) < 9:
            return False
        # Two 4-digit years with 'to', '-', or 'through' between
        if re.search(r'\d{4}\s*(?:to|-|through)\s*\d{4}', ref):
            return True
        # Multiple 4-digit years in one string
        years = re.findall(r'\b(19|20)\d{2}\b', ref)
        return len(years) >= 2

    def _parse_year_range(self, ref: str) -> Optional[List[int]]:
        """Parse year range from phrases like '2020 to 2024', '2020-2024', 'from 2020 to 2024'.
        Returns inclusive list of years [2020, 2021, 2022, 2023, 2024] or None if not a range."""
        import re
        ref_clean = (ref or "").strip().lower()
        # Match: 2020 to 2024, 2020 - 2024, from 2020 to 2024, 2020–2024 (en-dash), 2020—2024 (em-dash)
        match = re.search(r'(?:from\s+)?(\d{4})\s*(?:to|-|–|—)\s*(\d{4})', ref_clean)
        if not match:
            return None
        y1, y2 = int(match.group(1)), int(match.group(2))
        if y1 > y2:
            y1, y2 = y2, y1
        # Cap range to avoid runaway (e.g. 1990-2024 = 35 years)
        max_span = 10
        if y2 - y1 + 1 > max_span:
            y2 = y1 + max_span - 1
        return list(range(y1, y2 + 1))

    # ========================================================================
    # HELPER METHODS - Search Query Generation
    # ========================================================================

    def _generate_transcript_searches(
        self,
        tickers: List[str],
        quarters: List[str],
        topic: str,
        question_type: str
    ) -> List[TranscriptSearch]:
        """Generate transcript search specifications"""
        rag_logger.info(f"🔎 _generate_transcript_searches called: tickers={tickers}, quarters={quarters}")
        searches = []

        if not tickers or not quarters:
            return searches

        # Generate search query from topic
        query = self._create_search_query(topic, question_type, 'transcripts')

        # Create search per ticker
        for ticker in tickers:
            searches.append(TranscriptSearch(
                ticker=ticker,
                quarters=quarters,
                query=query
            ))

        return searches

    def _generate_10k_searches(
        self,
        tickers: List[str],
        years: List[int],
        topic: str,
        question_type: str,
        available_data: Optional[Dict[str, Any]] = None
    ) -> List[TenKSearch]:
        """Generate 10-K search specifications.

        - Multi-year (e.g. "2020 to 2024"): one search per (ticker, year) for each requested year.
        - Single year or latest: one search per ticker for that year.
        Supports multiple companies and multiple years (cross-company, multi-year).
        """
        searches = []
        if not tickers:
            return searches

        query = self._create_search_query(topic, question_type, '10k')
        available_10k_years = (available_data or {}).get('ten_k_years', {})
        # Cap total 10-K searches to avoid runaway (e.g. 5 tickers x 6 years = 30 → cap at 20)
        max_10k_searches = 20

        # Multiple years requested → one search per (ticker, year); search all requested years
        # (do not filter to available_10k_years so we attempt 2020/2021 etc.; missing years return no chunks)
        if len(years) > 1:
            for ticker in tickers:
                year_list = sorted(years, reverse=True)
                for year in year_list:
                    if len(searches) >= max_10k_searches:
                        rag_logger.info(f"📄 10-K search cap ({max_10k_searches}) reached, skipping remaining")
                        break
                    searches.append(TenKSearch(ticker=ticker, year=year, query=query))
                    rag_logger.info(f"📄 10-K search: {ticker} FY{year} (multi-year)")
            return searches

        # Single year or no specific years
        requested_year = int(years[0]) if years else None
        for ticker in tickers:
            if requested_year is not None:
                year_to_use = requested_year
                rag_logger.info(f"📄 10-K search: {ticker} FY{year_to_use} (user-requested year)")
            else:
                year_to_use = self._get_latest_10k_fiscal_year(ticker)
                if year_to_use:
                    rag_logger.info(f"📄 10-K search: {ticker} FY{year_to_use} (latest available)")
                elif years:
                    year_to_use = years[0]
                    rag_logger.warning(f"⚠️ No 10-K data found for {ticker}, using transcript year {year_to_use}")

            if year_to_use:
                searches.append(TenKSearch(ticker=ticker, year=year_to_use, query=query))
            else:
                rag_logger.warning(f"⚠️ No fiscal year available for {ticker} 10-K search")

        return searches

    def _get_latest_10k_fiscal_year(self, ticker: str) -> Optional[int]:
        """
        Get the latest available 10-K fiscal year for a ticker from the database.

        Returns the most recent fiscal year that actually has 10-K data,
        not just the current calendar year.
        """
        try:
            conn = self.database_manager._get_db_connection()
            cursor = conn.cursor()

            # Query ten_k_chunks for latest fiscal year
            query = """
                SELECT MAX(fiscal_year)
                FROM ten_k_chunks
                WHERE UPPER(ticker) = %s
            """

            cursor.execute(query, (ticker.upper(),))
            result = cursor.fetchone()
            self.database_manager._return_db_connection(conn)

            if result and result[0]:
                latest_year = result[0]
                rag_logger.debug(f"📊 Latest 10-K fiscal year for {ticker}: {latest_year}")
                return latest_year
            else:
                rag_logger.debug(f"📊 No 10-K data found for {ticker}")
                return None

        except Exception as e:
            rag_logger.error(f"❌ Error getting 10-K fiscal year for {ticker}: {e}")
            return None

    def _generate_news_searches(
        self,
        tickers: List[str],
        topic: str,
        question_type: str
    ) -> List[NewsSearch]:
        """Generate news search specifications"""
        # Generate news query
        if tickers:
            ticker_str = " ".join(tickers)
            query = f"{ticker_str} {topic}" if topic else ticker_str
        else:
            query = topic

        return [NewsSearch(query=query)]

    def _create_search_query(
        self,
        topic: str,
        question_type: str,
        data_source: str
    ) -> str:
        """
        Create search query from topic.

        For transcripts: semantic natural language
        For 10-K: more keyword-focused
        """
        if not topic:
            topic = "financial performance and results"

        # For now, just return the topic
        # Future: Could use LLM to refine query based on data source
        return topic

    # ========================================================================
    # HELPER METHODS - Reasoning Generation
    # ========================================================================

    def _generate_reasoning(
        self,
        question_type: str,
        data_sources: List[str],
        tickers: List[str],
        resolved_time: Dict[str, Any],
        transcript_searches: List[TranscriptSearch],
        ten_k_searches: List[TenKSearch],
        news_searches: List[NewsSearch]
    ) -> str:
        """Generate human-readable reasoning for the search plan"""

        # Build company description
        if len(tickers) == 1:
            company_desc = f"{tickers[0]}"
        elif len(tickers) == 2:
            company_desc = f"{tickers[0]} and {tickers[1]}"
        else:
            company_desc = f"{len(tickers)} companies"

        # Build time description (prefer 10-K years when plan is 10-K-only to avoid "20 quarters")
        if ten_k_searches and not transcript_searches and len(ten_k_searches) > 0:
            years_10k = sorted(set(s.year for s in ten_k_searches), reverse=True)
            if len(years_10k) == 1:
                time_desc = f"FY{years_10k[0]}"
            elif len(years_10k) > 1:
                time_desc = f"FY{years_10k[-1]}–FY{years_10k[0]} ({len(years_10k)} years)"
            else:
                time_desc = "recent data"
        else:
            quarters = resolved_time.get('quarters', [])
            if len(quarters) == 1:
                time_desc = f"Q{quarters[0].split('_q')[1]} {quarters[0].split('_')[0]}"
            elif len(quarters) > 1:
                time_desc = f"{len(quarters)} quarters"
            else:
                time_desc = "recent data"

        # Build source description
        source_parts = []
        if transcript_searches:
            source_parts.append("earnings transcripts")
        if ten_k_searches:
            source_parts.append("10-K filings")
        if news_searches:
            source_parts.append("latest news")

        source_desc = ", ".join(source_parts)

        # Generate reasoning
        reasoning = f"Searching {source_desc} for {company_desc} ({time_desc})."

        # Add specifics
        if transcript_searches and ten_k_searches:
            reasoning += " Using both transcripts for recent commentary and 10-K for comprehensive annual data."
        elif ten_k_searches:
            reasoning += " Using 10-K filings for official annual disclosures."
        elif news_searches:
            reasoning += " Using news search for current developments."

        return reasoning
