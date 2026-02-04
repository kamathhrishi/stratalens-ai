#!/usr/bin/env python3
"""
RAG Agent - Orchestration and response generation for RAG system

This module contains the orchestration logic including:
- Response generation (LLM integration)
- Quality evaluation and iterative improvement
- Follow-up question generation
- Main RAG flow execution
"""

import os
import json
import logging
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Import required dependencies
import openai

# Import Logfire for observability (optional)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

# Import analytics utilities
from analytics.analytics_utils import RAGAnalyticsLogger

# Import extracted modules
from .config import Config
from .question_analyzer import QuestionAnalyzer
from .database_manager import DatabaseManager
from .search_engine import SearchEngine
from .response_generator import ResponseGenerator
from .tavily_service import TavilyService
# SEC 10-K Filing Service (planning + parallel retrieval)
from .sec_filings_service_smart_parallel import SmartParallelSECFilingsService as SECFilingsService

# Import RAG utilities
from .rag_utils import (
    generate_user_friendly_limit_message,
    assess_answer_quality,
    normalize_ticker
)

# Import centralized prompts
from .. import prompts
from agent.prompts import (
    CONTEXT_AWARE_FOLLOWUP_SYSTEM_PROMPT,
    get_context_aware_followup_prompt
)

# Load .env file from the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create a more detailed logger for RAG operations
rag_logger = logging.getLogger('rag_agent')
rag_logger.setLevel(logging.INFO)


class RAGAgent:
    """
    RAG Agent - Orchestrates the complete RAG flow including response generation
    and iterative improvement.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the RAG Agent.

        Sets up the RAG Agent by initializing all necessary components for orchestration.

        Args:
            openai_api_key (Optional[str]): OpenAI API key for LLM operations.
                If None, will attempt to load from environment variables.
        """
        self.instance_id = f"RAGAgent_{int(time.time() * 1000)}"
        logger.info(f"🚀 Creating RAG Agent instance: {self.instance_id}")

        self.config = Config()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")

        # Fetch available quarters from database
        try:
            self.config.fetch_available_quarters_from_db()
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch quarters from database during initialization: {e}")
            logger.warning("⚠️ RAG Agent will continue with default configuration")
            logger.warning("⚠️ You may need to run the quarter formatting script first")

        # Initialize components
        self.database_manager = DatabaseManager(self.config)
        self.question_analyzer = QuestionAnalyzer(None, self.config, self.database_manager)
        self.search_engine = SearchEngine(self.config, self.database_manager)
        self.response_generator = ResponseGenerator(self.config, self.openai_api_key)
        self.tavily_service = TavilyService()
        self.sec_service = SECFilingsService(self.database_manager, self.config)

        # Initialize analytics logger
        self.analytics_logger = RAGAnalyticsLogger(self.config.get_connection_string())

        # Thread pool for CPU-bound operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Lazy initialization of OpenAI client (created on first use)
        # This ensures logfire.instrument_openai() is called before client creation
        self._client = None

        # Initialize Cerebras client (primary for response generation - fast inference with Qwen)
        cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
        if cerebras_api_key:
            try:
                from cerebras.cloud.sdk import Cerebras
                self.cerebras_client = Cerebras(api_key=cerebras_api_key)
                self.cerebras_available = True
                rag_logger.info(f"✅ Cerebras client initialized with model: {self.config.get('cerebras_model')}")
            except ImportError:
                rag_logger.warning("⚠️ Cerebras SDK not installed. Run: pip install cerebras-cloud-sdk")
                self.cerebras_client = None
                self.cerebras_available = False
            except Exception as e:
                rag_logger.warning(f"⚠️ Failed to initialize Cerebras client: {e}")
                self.cerebras_client = None
                self.cerebras_available = False
        else:
            self.cerebras_client = None
            self.cerebras_available = False
            rag_logger.warning("⚠️ CEREBRAS_API_KEY not found")

        # Groq removed - using Cerebras only
            self.groq_client = None
            self.groq_available = False

        self.openai_available = True
        
        # Log LLM provider priority
        if self.cerebras_available:
            rag_logger.info("🤖 LLM Priority: Cerebras (primary) > OpenAI (fallback)")
        else:
            rag_logger.info("🤖 LLM Priority: OpenAI only")

        logger.info(f"🚀 RAG Agent initialized successfully (instance: {self.instance_id})")

        # Log hybrid search configuration
        if self.config.get("hybrid_search_enabled", True):
            logger.info(f"🔀 Hybrid search enabled - Vector weight: {self.config.get('vector_weight', 0.7)}, Keyword weight: {self.config.get('keyword_weight', 0.3)}")
        else:
            logger.info("⚠️ Hybrid search disabled - using vector-only search")

    @property
    def client(self):
        """Lazy initialization of OpenAI client to ensure proper Logfire instrumentation."""
        if self._client is None and self.openai_api_key:
            self._client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info("✅ OpenAI client initialized (lazy - after Logfire instrumentation)")
        return self._client

    def set_database_connection(self, db_connection):
        """Set the database connection for retrieving conversation history."""
        self.question_analyzer.conversation_memory.set_database_connection(db_connection)

    def __del__(self):
        """Cleanup when instance is destroyed."""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
                rag_logger.info("✅ Thread pool shut down")
        except Exception as e:
            rag_logger.warning(f"⚠️ Error during RAG Agent cleanup: {e}")

    def _get_db_connection(self):
        """Get database connection from pool."""
        return self.database_manager._get_db_connection()

    def _return_db_connection(self, conn):
        """Return connection to pool."""
        self.database_manager._return_db_connection(conn)

    def _process_single_ticker_sync(self, ticker: str, question: str, processed_question: str,
                                   is_multi_ticker: bool, target_quarters: List[str]) -> Dict[str, Any]:
        """Process a single ticker synchronously with parallel retrieval optimization."""
        ticker_start_time = time.time()

        try:
            rag_logger.info("=" * 80)
            rag_logger.info(f"🎯 STARTING TICKER PROCESSING: {ticker}")
            rag_logger.info(f"📅 Target quarters: {target_quarters}")
            rag_logger.info(f"📝 Processed question: {processed_question[:100]}...")
            rag_logger.info("=" * 80)
            print(f"   🔍 Processing {ticker} synchronously...")

            # Create ticker-specific question for better search results
            if is_multi_ticker:
                print(f"   🎯 Creating ticker-specific question for {ticker}...")
                ticker_specific_question = self.question_analyzer.create_ticker_specific_question(processed_question, ticker)
                print(f"   ✅ Ticker-specific question: '{ticker_specific_question}'")
            else:
                ticker_specific_question = processed_question

            # For multi-ticker queries, get company-specific quarters
            # Different companies may have different latest quarters available
            ticker_quarters = target_quarters
            if is_multi_ticker:
                if len(target_quarters) == 1:
                    # Single quarter - get company-specific latest quarter
                    company_latest = self.database_manager.get_latest_quarter_for_company(ticker)
                    if company_latest:
                        ticker_quarters = [company_latest]
                        rag_logger.info(f"   📅 Using company-specific latest quarter for {ticker}: {company_latest}")
                        print(f"   📅 Using {ticker}'s latest quarter: {company_latest}")
                    else:
                        rag_logger.warning(f"   ⚠️ No company-specific quarter found for {ticker}, using provided quarter: {target_quarters[0]}")
                else:
                    # Multiple quarters - get company-specific last N quarters
                    # This handles "last N quarters" queries where each company may have different latest quarters
                    quarter_count = len(target_quarters)
                    company_quarters = self.database_manager.get_last_n_quarters_for_company(ticker, quarter_count)
                    if company_quarters:
                        ticker_quarters = company_quarters
                        rag_logger.info(f"   📅 Using company-specific last {quarter_count} quarters for {ticker}: {company_quarters}")
                        print(f"   📅 Using {ticker}'s last {quarter_count} quarters: {', '.join(company_quarters)}")
                    else:
                        rag_logger.warning(f"   ⚠️ No company-specific quarters found for {ticker}, using provided quarters")

            # Search for this specific ticker with quarter filtering
            print(f"   🔍 Searching {ticker} transcripts...")
            search_start = time.time()

            query_embedding = self.search_engine.embedding_model.encode([ticker_specific_question])

            # Search across multiple quarters in parallel if specified
            if len(ticker_quarters) == 1:
                ticker_chunks = self.database_manager._search_postgres_with_ticker(
                    query_embedding, ticker, ticker_quarters[0]
                )
            else:
                # Multiple quarters - parallel search
                def search_quarter_sync(quarter):
                    return self.database_manager._search_postgres_with_ticker(
                        query_embedding, ticker, quarter
                    )

                quarter_results = {}
                optimal_workers = min(len(ticker_quarters), os.cpu_count() * 2, 8)

                with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                    future_to_quarter = {
                        executor.submit(search_quarter_sync, quarter): quarter
                        for quarter in ticker_quarters
                    }

                    for future in as_completed(future_to_quarter):
                        quarter = future_to_quarter[future]
                        try:
                            quarter_chunks = future.result()
                            quarter_results[quarter] = quarter_chunks
                        except Exception as e:
                            rag_logger.error(f"❌ Error searching {ticker} in quarter {quarter}: {e}")
                            quarter_results[quarter] = []

                # Take top chunks from each quarter
                chunks_per_quarter = self.config.get("chunks_per_quarter", 15)
                ticker_chunks = []
                for quarter in ticker_quarters:
                    quarter_chunks = quarter_results.get(quarter, [])
                    quarter_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    top_quarter_chunks = quarter_chunks[:chunks_per_quarter]
                    for chunk in top_quarter_chunks:
                        chunk['source_quarter'] = quarter
                    ticker_chunks.extend(top_quarter_chunks)

                # Final sort by similarity
                ticker_chunks.sort(key=lambda x: x.get('similarity', 0), reverse=True)

            search_time = time.time() - search_start
            print(f"   ⏱️  Search completed in {search_time:.3f}s")

            if ticker_chunks:
                print(f"   ✅ Found {len(ticker_chunks)} chunks for {ticker}")

                # Generate individual answer for this ticker
                print(f"   🤖 Generating answer for {ticker}...")
                ticker_context = [chunk['chunk_text'] for chunk in ticker_chunks]
                ticker_citations = ticker_chunks

                year = ticker_chunks[0].get('year') if ticker_chunks else None
                quarter = ticker_chunks[0].get('quarter') if ticker_chunks else None

                ticker_answer = self.response_generator.generate_openai_response(
                    ticker_specific_question,
                    ticker_context,
                    ticker_citations,
                    ticker=ticker,
                    year=year,
                    quarter=quarter
                )

                ticker_total_time = time.time() - ticker_start_time
                print(f"   ✅ {ticker} completed in {ticker_total_time:.3f}s")

                return {
                    'ticker': ticker,
                    'answer': ticker_answer,
                    'chunks': ticker_chunks,
                    'context_chunks': ticker_context,
                    'citations': ticker_citations
                }
            else:
                print(f"   ⚠️ No chunks found for {ticker}")
                return {
                    'ticker': ticker,
                    'answer': f"No relevant information found for {ticker}",
                    'chunks': [],
                    'context_chunks': [],
                    'citations': []
                }

        except Exception as e:
            print(f"   ❌ Error processing {ticker}: {e}")
            rag_logger.error(f"❌ Error processing ticker {ticker}: {e}")
            return {
                'ticker': ticker,
                'answer': f"Error processing {ticker}: {str(e)}",
                'chunks': [],
                'context_chunks': [],
                'citations': []
            }

    def _process_tickers_parallel_sync(self, tickers: List[str], question: str, processed_question: str,
                                     is_multi_ticker: bool, target_quarters: List[str]) -> List[Dict[str, Any]]:
        """Process multiple tickers in parallel using ThreadPoolExecutor."""
        try:
            rag_logger.info(f"🚀 Starting parallel processing of {len(tickers)} tickers")

            results = []
            cpu_cores = os.cpu_count() or 4

            # Worker allocation strategy
            if len(tickers) <= 4:
                optimal_workers = min(len(tickers), cpu_cores * 2, 12)
            else:
                optimal_workers = min(len(tickers), cpu_cores * 3, 16)

            rag_logger.info(f"🚀 Using {optimal_workers} workers for {len(tickers)} tickers")

            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._process_single_ticker_sync, ticker, question, processed_question, is_multi_ticker, target_quarters): ticker
                    for ticker in tickers
                }

                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        result = future.result()
                        results.append(result)
                        rag_logger.info(f"✅ Completed processing for {ticker}")
                    except Exception as e:
                        rag_logger.error(f"❌ Exception processing ticker {ticker}: {e}")
                        results.append({
                            'ticker': ticker,
                            'answer': f"Error processing {ticker}: {str(e)}",
                            'chunks': [],
                            'context_chunks': [],
                            'citations': []
                        })

            rag_logger.info(f"✅ Parallel processing completed: {len(results)} tickers processed")
            return results

        except Exception as e:
            rag_logger.error(f"❌ Error in parallel ticker processing: {e}")
            # Fallback to sequential processing
            rag_logger.info("🔄 Falling back to sequential processing")
            return [self._process_single_ticker_sync(ticker, question, processed_question, is_multi_ticker, target_quarters) for ticker in tickers]

    async def _search_with_multiple_questions(self, rephrased_questions: List[str], target_quarters: List[str], target_quarter: str, ticker: str = None) -> List[Dict[str, Any]]:
        """Search for chunks using multiple rephrased questions in parallel.
        
        Executes searches with all rephrased questions simultaneously using asyncio,
        then combines and deduplicates results to get a richer set of relevant chunks.
        
        REFACTORED: Now properly uses asyncio without creating nested event loops.
        This prevents blocking and ensures better concurrency for multiple users.
        
        Args:
            rephrased_questions (List[str]): List of 3 rephrased questions to search with.
            target_quarters (List[str]): Quarters to search (e.g., ['2025_q1', '2025_q2']).
            target_quarter (str): Primary quarter or 'multiple'.
            ticker (str, optional): Specific ticker to search for.
        
        Returns:
            List[Dict[str, Any]]: Combined and deduplicated chunks with structure:
                [
                    {
                        "chunk_text": str,
                        "ticker": str,
                        "year": int,
                        "quarter": int,
                        "citation": str,
                        "distance": float,
                        "query_source": str  # Which rephrased question found this
                    },
                    ...
                ]
        """
        rag_logger.info(f"🔍 Searching with {len(rephrased_questions)} rephrased questions in parallel")
        
        async def search_with_question_async(question_text: str, question_idx: int):
            """Helper function to search with a single question using asyncio."""
            try:
                rag_logger.info(f"   🔍 Query {question_idx + 1}: '{question_text}'")
                
                if ticker:
                    # Ticker-specific search
                    if len(target_quarters) > 1:
                        chunks = []
                        # Encode query asynchronously to avoid blocking
                        loop = asyncio.get_event_loop()
                        query_embedding = await loop.run_in_executor(
                            None,
                            self.search_engine.embedding_model.encode,
                            [question_text]
                        )
                        
                        # Search in parallel across quarters
                        for quarter in target_quarters:
                            # Use thread pool executor for DB operations
                            q_chunks = await loop.run_in_executor(
                                None,
                                self.database_manager._search_postgres_with_ticker,
                                query_embedding,
                                ticker,
                                quarter
                            )
                            chunks.extend(q_chunks)
                    else:
                        # Single quarter
                        loop = asyncio.get_event_loop()
                        query_embedding = await loop.run_in_executor(
                            None,
                            self.search_engine.embedding_model.encode,
                            [question_text]
                        )
                        chunks = await loop.run_in_executor(
                            None,
                            self.database_manager._search_postgres_with_ticker,
                            query_embedding,
                            ticker,
                            target_quarter
                        )
                else:
                    # General search
                    if len(target_quarters) > 1:
                        # Use async multi-quarter search
                        chunks = await self.search_engine._search_multiple_quarters_async(
                            question_text, target_quarters, chunks_per_quarter=self.config.get("chunks_per_quarter", 15)
                        )
                    else:
                        # Single quarter general search
                        loop = asyncio.get_event_loop()
                        chunks = await loop.run_in_executor(
                            None,
                            self.search_engine.search_similar_chunks,
                            question_text,
                            self.config.get("chunks_per_quarter", 15),
                            target_quarter
                        )
                
                # Add query source to each chunk
                for chunk in chunks:
                    chunk['query_source'] = f"Query {question_idx + 1}"
                    chunk['query_text'] = question_text
                
                rag_logger.info(f"      ✅ Query {question_idx + 1}: Found {len(chunks)} chunks")
                return chunks
                
            except Exception as e:
                rag_logger.error(f"      ❌ Query {question_idx + 1}: Search failed: {e}")
                return []
        
        # Execute all searches in parallel using asyncio.gather
        # This is better than ThreadPoolExecutor for async code
        tasks = [
            search_with_question_async(q, i)
            for i, q in enumerate(rephrased_questions)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results, filtering out exceptions
        all_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                rag_logger.error(f"Query {i + 1} raised exception: {result}")
            else:
                all_results.extend(result)
        
        # Deduplicate chunks by citation (keep highest similarity score)
        seen_citations = {}
        for chunk in all_results:
            citation = chunk.get('citation')
            if citation not in seen_citations:
                seen_citations[citation] = chunk
            else:
                # Keep chunk with better similarity score
                existing_distance = seen_citations[citation].get('distance', 1.0)
                new_distance = chunk.get('distance', 1.0)
                if new_distance < existing_distance:  # Lower distance = better similarity
                    seen_citations[citation] = chunk
        
        deduplicated_chunks = list(seen_citations.values())
        
        # Sort by similarity (distance)
        deduplicated_chunks.sort(key=lambda x: x.get('distance', 1.0))
        
        # Limit total chunks based on number of quarters being searched
        chunks_per_quarter = self.config.get("chunks_per_quarter", 15)
        num_quarters = len(target_quarters) if target_quarters else 1
        max_total_chunks = chunks_per_quarter * num_quarters
        final_chunks = deduplicated_chunks[:max_total_chunks]
        
        rag_logger.info(f"✅ Parallel search complete:")
        rag_logger.info(f"   📊 Total chunks from all queries: {len(all_results)}")
        rag_logger.info(f"   📊 After deduplication: {len(deduplicated_chunks)}")
        rag_logger.info(f"   📊 Final chunks (after limit): {len(final_chunks)}")
        
        return final_chunks

    async def _perform_question_analysis(self, question: str, conversation_id: str) -> tuple:
        """Perform question analysis and determine target quarters.

        Analyzes the user's question to extract tickers, determine question type,
        identify target quarters, and validate data availability. Handles quarter
        limits and provides user-friendly error messages for unavailable data.

        Args:
            question (str): The user's question to analyze.
            conversation_id (str): Unique conversation identifier.
        
        Returns:
            tuple: (question_analysis, target_quarters, error_dict)
                - question_analysis (Dict or None): Analysis results with structure:
                    {
                        "status": str,  # 'success', 'rejected', or 'error'
                        "extracted_ticker": str,  # Primary ticker symbol
                        "extracted_tickers": List[str],  # All ticker symbols
                        "question_type": str,  # Question category
                        "processed_question": str,  # Cleaned question
                        "quarter_context": str,  # Quarter information
                        "quarter_reference": str,  # Specific quarter ref
                        "limits_exceeded": Dict,  # Any limit violations
                        "message": str  # Status message (for rejected)
                    }
                - target_quarters (List[str] or None): Quarters to search (e.g., ['2025_q1', '2025_q2'])
                - error_dict (Dict or None): Error information if validation fails:
                    {
                        "status": str,  # 'error'
                        "message": str,  # User-friendly error message
                        "available_quarters": List[str],  # Available data
                        "requested_quarter": str  # What was requested
                    }
        """
        analysis_start = time.time()
        print(f"\n📋 STEP 1: QUESTION ANALYSIS")
        print(f"{'─'*50}")
        rag_logger.info(f"📋 Step 1: Starting question analysis...")
        
        question_analysis = await self.question_analyzer.process_question(question, conversation_id)
        analysis_time = time.time() - analysis_start
        
        # Add conversation_id to question_analysis for evaluation context
        question_analysis['conversation_id'] = conversation_id
        
        # Log question analyzer results to analytics
        await self.analytics_logger.log_question_analyzer_result(
            success=True,
            analysis_result=question_analysis,
            retry_count=0,
            response_time_ms=int(analysis_time * 1000)
        )
        
        # Normalize tickers in question_analysis BEFORE quarter lookup
        # This ensures company-specific quarter resolution works for aliases (e.g., TSMC → TSM)
        raw_tickers = question_analysis.get('extracted_tickers', [])
        if raw_tickers:
            question_analysis['extracted_tickers'] = [normalize_ticker(t) for t in raw_tickers if t]
        raw_ticker = question_analysis.get('extracted_ticker')
        if raw_ticker:
            question_analysis['extracted_ticker'] = normalize_ticker(raw_ticker)

        # Debug output
        rag_logger.info(f"🔍 Question analysis debug:")
        rag_logger.info(f"   quarter_context: {question_analysis.get('quarter_context')}")
        rag_logger.info(f"   quarter_reference: {question_analysis.get('quarter_reference')}")
        rag_logger.info(f"   extracted_ticker: {question_analysis.get('extracted_ticker')}")
        rag_logger.info(f"   extracted_tickers: {question_analysis.get('extracted_tickers')}")

        # Determine target quarters
        target_quarters = self.question_analyzer.get_quarters_to_search(question_analysis)
        
        # Apply quarter limit
        max_quarters = self.config.get("max_quarters", 12)
        if len(target_quarters) > max_quarters:
            rag_logger.warning(f"⚠️ Quarter limit exceeded: {len(target_quarters)} quarters requested, limiting to {max_quarters}")
            original_count = len(target_quarters)
            skipped_quarters = target_quarters[max_quarters:]
            target_quarters = target_quarters[:max_quarters]
            print(f"⚠️ Too many quarters requested ({original_count}). Limiting to most recent {max_quarters} quarters (3 years).")
            print(f"📅 Processing quarters: {', '.join(target_quarters)}")
            
            question_analysis['limits_exceeded'] = question_analysis.get('limits_exceeded', {})
            question_analysis['limits_exceeded']['quarters'] = {
                'requested': original_count,
                'processed': max_quarters,
                'skipped': skipped_quarters
            }
        
        # Validate quarters availability
        if not target_quarters:
            quarter_reference = question_analysis.get('quarter_reference')
            if quarter_reference:
                available_quarters_human = []
                for q in self.config.get('available_quarters', []):
                    if '_q' in q:
                        year, quarter = q.split('_q')
                        available_quarters_human.append(f"{year} Q{quarter}")
                    else:
                        available_quarters_human.append(q)
                
                requested_quarter_human = quarter_reference
                if '_q' in quarter_reference:
                    year, quarter = quarter_reference.split('_q')
                    requested_quarter_human = f"{year} Q{quarter}"
                
                return None, None, {
                    "status": "error",
                    "message": f"Sorry, I don't have earnings transcript data for {requested_quarter_human} available. The available quarters are: {', '.join(available_quarters_human)}.",
                    "available_quarters": self.config.get('available_quarters', []),
                    "requested_quarter": quarter_reference
                }
            else:
                return None, None, {
                    "status": "error",
                    "message": "Sorry, I don't have any earnings transcript data available at the moment. Please try again later.",
                    "available_quarters": []
                }
        
        target_quarter = target_quarters[0] if len(target_quarters) == 1 else 'multiple'
        rag_logger.info(f"📅 Target quarters determined: {target_quarters}")
        
        print(f"⏱️  Analysis completed in {analysis_time:.3f}s")
        print(f"📊 Analysis Results:")
        print(f"   ✅ Status: {question_analysis['status']}")
        print(f"   🎯 Primary ticker: {question_analysis.get('extracted_ticker', 'None')}")
        print(f"   📝 Question type: {question_analysis.get('question_type', 'Unknown')}")
        print(f"   📅 Target quarters: {target_quarters}")
        
        rag_logger.info(f"✅ Question analysis completed in {analysis_time:.3f}s")
        
        # Handle rejected questions
        if question_analysis['status'] == 'rejected':
            rag_logger.warning(f"❌ Question rejected: {question_analysis.get('message', 'Unknown reason')}")
            return None, None, {
                'success': False,
                'errors': [question_analysis['message']],
                'analysis': question_analysis,
                'timing': {
                    'analysis': analysis_time,
                    'total': analysis_time
                }
            }
        
        return question_analysis, target_quarters, None
    
    async def _execute_search(self, question: str, question_analysis: Dict, target_quarters: List[str]) -> tuple:
        """Execute search based on question type (general vs ticker-specific).

        Determines the search strategy (general/single-ticker/multi-ticker) and executes
        appropriate search operations. Handles ticker limits, parallel processing for
        multiple tickers, and aggregates results with citations.

        Args:
            question (str): The user's original question.
            question_analysis (Dict): Analysis results from question analyzer (see _perform_question_analysis).
            target_quarters (List[str]): Quarters to search (e.g., ['2025_q1', '2025_q2']).
        
        Returns:
            tuple: (individual_results, all_chunks, all_citations, search_time, 
                    is_general_question, is_multi_ticker, tickers_to_process, target_quarter)
                - individual_results (List[Dict]): Results per ticker with structure:
                    [
                        {
                            "ticker": str,  # Company ticker
                            "answer": str,  # Generated answer for this ticker
                            "chunks": List[Dict],  # Retrieved chunks
                            "context_chunks": List[str],  # Text chunks
                            "citations": List  # Citation references
                        },
                        ...
                    ]
                - all_chunks (List[Dict]): All retrieved chunks with metadata
                - all_citations (List): All citation references
                - search_time (float): Total search time in seconds
                - is_general_question (bool): Whether question is general (no ticker)
                - is_multi_ticker (bool): Whether question involves multiple tickers
                - tickers_to_process (List[str]): Tickers that were searched
                - target_quarter (str): Primary quarter or 'multiple'
        """
        search_start = time.time()
        print(f"\n🔍 STEP 2: SEARCH STRATEGY DETERMINATION")
        print(f"{'─'*50}")
        rag_logger.info(f"🔍 Step 2: Starting vector search...")

        processed_question = question_analysis.get('processed_question', question)
        extracted_tickers = [normalize_ticker(t) for t in question_analysis.get('extracted_tickers', []) if t]
        extracted_ticker = normalize_ticker(question_analysis.get('extracted_ticker') or '')
        if not extracted_ticker:
            extracted_ticker = None
        target_quarter = target_quarters[0] if len(target_quarters) == 1 else 'multiple'

        print(f"🔍 Search Strategy Analysis:")
        print(f"   📝 Processed question: '{processed_question}'")
        print(f"   🎯 Extracted ticker: {extracted_ticker}")
        print(f"   🎯 All tickers: {extracted_tickers}")
        
        # Determine search strategy
        has_tickers = bool(extracted_ticker or extracted_tickers)
        is_multi_ticker = len(extracted_tickers) > 1
        is_general_question = not has_tickers
        
        if has_tickers:
            tickers_to_process = extracted_tickers if is_multi_ticker else [extracted_ticker]
            
            # Apply ticker limit
            max_tickers = self.config.get("max_tickers", 4)
            if len(tickers_to_process) > max_tickers:
                rag_logger.warning(f"⚠️ Ticker limit exceeded: {len(tickers_to_process)} tickers requested, limiting to {max_tickers}")
                tickers_to_process = tickers_to_process[:max_tickers]
                print(f"⚠️ Too many tickers requested. Limiting to first {max_tickers} tickers.")
                
                question_analysis['limits_exceeded'] = question_analysis.get('limits_exceeded', {})
                question_analysis['limits_exceeded']['tickers'] = {
                    'requested': len(extracted_tickers),
                    'processed': max_tickers,
                    'skipped': extracted_tickers[max_tickers:]
                }
        else:
            tickers_to_process = []
        
        print(f"📊 Search Strategy Decision:")
        print(f"   🎯 Has tickers: {has_tickers}")
        print(f"   🔄 Is multi-ticker: {is_multi_ticker}")
        print(f"   🌐 Is general question: {is_general_question}")
        print(f"   📋 Tickers to process: {tickers_to_process}")
        
        # DISABLED: Question rephrasing for initial search
        # Using original/processed question directly for better precision
        print(f"\n🔄 STEP 2.5: QUERY PREPARATION")
        print(f"{'─'*50}")
        # rephrased_questions = self._generate_rephrased_questions(question, question_analysis)
        rephrased_questions = [processed_question]  # Use only the processed question
        print(f"✅ Using direct question (rephrasing disabled): {processed_question}")
        
        # Execute search based on type
        print(f"\n🔍 STEP 3: SEARCH EXECUTION")
        print(f"{'─'*50}")
        individual_results = []
        all_chunks = []
        all_citations = []
        
        if is_general_question:
            # General search across all companies
            print(f"🌐 Processing general question")

            search_start_time = time.time()
            # Use parallel multi-query search
            general_chunks = await self._search_with_multiple_questions(
                rephrased_questions, target_quarters, target_quarter, ticker=None
            )
            search_time_inner = time.time() - search_start_time
            
            await self.analytics_logger.log_chunk_retrieval_result(
                chunks_retrieved=len(general_chunks),
                chunks_used=len(general_chunks),
                chunk_details=general_chunks[:5] if general_chunks else None,
                similarity_threshold=self.config.get('similarity_threshold'),
                chunks_per_quarter=self.config.get("chunks_per_quarter"),
                retrieval_time_ms=int(search_time_inner * 1000)
            )

            print(f"⏱️  General search completed in {search_time_inner:.3f}s")
            print(f"📊 Search results: {len(general_chunks)} chunks found")
            
            if general_chunks:
                # Group chunks by ticker and generate answers
                ticker_groups = {}
                for chunk in general_chunks:
                    ticker = chunk.get('ticker', 'Unknown')
                    if ticker not in ticker_groups:
                        ticker_groups[ticker] = []
                    ticker_groups[ticker].append(chunk)
                
                for ticker, ticker_chunks in ticker_groups.items():
                    ticker_context = [chunk['chunk_text'] for chunk in ticker_chunks]
                    ticker_citations = ticker_chunks
                    year = ticker_chunks[0].get('year') if ticker_chunks else None
                    quarter = ticker_chunks[0].get('quarter') if ticker_chunks else None
                    
                    ticker_answer = self.response_generator.generate_openai_response(
                        processed_question, ticker_context, ticker_citations, 
                        ticker=ticker, year=year, quarter=quarter
                    )
                    
                    individual_results.append({
                        'ticker': ticker,
                        'answer': ticker_answer,
                        'chunks': ticker_chunks,
                        'context_chunks': ticker_context,
                        'citations': ticker_citations
                    })
                
                all_chunks.extend(general_chunks)
                all_citations.extend([chunk['citation'] for chunk in general_chunks])
        else:
            # Ticker-specific search with query expansion
            print(f"🎯 Processing ticker-specific question with query expansion")
            
            if len(tickers_to_process) > 1:
                print(f"🚀 Using parallel processing for {len(tickers_to_process)} tickers")
                individual_results = self._process_tickers_parallel_sync(
                    tickers_to_process, question, processed_question, is_multi_ticker, target_quarters
                )
            else:
                print(f"🎯 Processing single ticker: {tickers_to_process[0]}")
                ticker = tickers_to_process[0]
                
                # Use direct search for single ticker (rephrasing disabled)
                search_start_time = time.time()
                ticker_chunks = await self._search_with_multiple_questions(
                    rephrased_questions, target_quarters, target_quarter, ticker=ticker
                )
                search_time_inner = time.time() - search_start_time
                
                print(f"   ✅ Found {len(ticker_chunks)} chunks for {ticker}")
                
                # Create individual result
                ticker_context = [chunk['chunk_text'] for chunk in ticker_chunks]
                ticker_citations = ticker_chunks
                year = ticker_chunks[0].get('year') if ticker_chunks else None
                quarter = ticker_chunks[0].get('quarter') if ticker_chunks else None
                
                ticker_answer = self.response_generator.generate_openai_response(
                    question, ticker_context, ticker_citations, 
                    ticker=ticker, year=year, quarter=quarter
                )
                
                individual_results = [{
                    'ticker': ticker,
                    'answer': ticker_answer,
                    'chunks': ticker_chunks,
                    'context_chunks': ticker_context,
                    'citations': ticker_citations
                }]
            
            # Collect all chunks and citations
            for result in individual_results:
                all_chunks.extend(result.get('chunks', []))
                all_citations.extend(result.get('citations', []))
        
        search_time = time.time() - search_start
        print(f"\n⏱️  SEARCH PHASE COMPLETED")
        print(f"   ✅ Search completed in {search_time:.3f}s")
        print(f"   📊 Found {len(all_chunks)} total chunks")
        
        return individual_results, all_chunks, all_citations, search_time, is_general_question, is_multi_ticker, tickers_to_process, target_quarter
    
    async def _perform_parallel_follow_up_search(self, follow_up_questions: List[str], has_tickers: bool, 
                                                 is_general_question: bool, is_multi_ticker: bool, 
                                                 tickers_to_process: List[str], target_quarter, 
                                                 target_quarters: List[str]) -> List[Dict[str, Any]]:
        """Perform parallel follow-up searches using multiple questions simultaneously.
        
        Searches with ALL follow-up questions in parallel to maximize chunk retrieval
        in a single iteration. If 3 questions generate ~5 chunks each, we get ~15 chunks
        total instead of needing 3 iterations.
        
        Args:
            follow_up_questions (List[str]): List of follow-up questions to search with.
            has_tickers (bool): Whether the query has specific tickers.
            is_general_question (bool): Whether this is a general question.
            is_multi_ticker (bool): Whether multiple tickers are involved.
            tickers_to_process (List[str]): List of tickers to search.
            target_quarter (str): Primary quarter or 'multiple'.
            target_quarters (List[str]): All quarters to search.
        
        Returns:
            List[Dict[str, Any]]: Combined and deduplicated chunks from all searches.
        """
        rag_logger.info(f"🔄 Parallel follow-up search with {len(follow_up_questions)} questions")
        
        # Log all questions
        for i, q in enumerate(follow_up_questions, 1):
            rag_logger.info(f"   🔍 Question {i}/{len(follow_up_questions)}: {q[:80]}...")
        
        # Execute all searches in parallel using asyncio.gather
        search_tasks = [
            self._perform_follow_up_search(
                follow_up_q, has_tickers, is_general_question, is_multi_ticker,
                tickers_to_process, target_quarter, target_quarters
            )
            for follow_up_q in follow_up_questions
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Collect all chunks from successful searches
        all_refined_chunks = []
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                rag_logger.error(f"   ❌ Question {i} failed: {result}")
            elif result:
                rag_logger.info(f"   ✅ Question {i} returned {len(result)} chunks")
                all_refined_chunks.extend(result)
            else:
                rag_logger.info(f"   ⚠️ Question {i} returned no chunks")
        
        # Deduplicate by citation (keep best scoring version of each)
        if all_refined_chunks:
            unique_chunks = {}
            for chunk in all_refined_chunks:
                citation = chunk.get('citation', '')
                if citation:
                    # Keep chunk with best (lowest) distance score
                    if citation not in unique_chunks:
                        unique_chunks[citation] = chunk
                    else:
                        existing_distance = unique_chunks[citation].get('distance', 1.0)
                        new_distance = chunk.get('distance', 1.0)
                        if new_distance < existing_distance:
                            unique_chunks[citation] = chunk
            
            deduplicated_chunks = list(unique_chunks.values())
            rag_logger.info(f"📊 Parallel search: {len(all_refined_chunks)} chunks → {len(deduplicated_chunks)} unique chunks")
            
            return deduplicated_chunks
        
        rag_logger.warning(f"⚠️ No chunks found from any of the {len(follow_up_questions)} follow-up questions")
        return []
    
    async def _perform_follow_up_search(self, follow_up_question: str, has_tickers: bool, is_general_question: bool, 
                                       is_multi_ticker: bool, tickers_to_process: List[str], target_quarter, 
                                       target_quarters: List[str]) -> List[Dict[str, Any]]:
        """Perform follow-up search based on question type (single question).
        
        REFACTORED: Now properly uses asyncio with run_in_executor to prevent event loop blocking.
        This ensures better concurrency for multiple users.
        
        Executes additional searches using refined follow-up questions to gather
        more relevant context. Adapts search strategy based on whether the question
        is general, single-ticker, or multi-ticker. Uses parallel processing for
        multi-ticker searches.
        
        Args:
            follow_up_question (str): Refined follow-up question for search.
            has_tickers (bool): Whether the query has specific tickers.
            is_general_question (bool): Whether this is a general question.
            is_multi_ticker (bool): Whether multiple tickers are involved.
            tickers_to_process (List[str]): List of tickers to search.
            target_quarter (str): Primary quarter or 'multiple'.
            target_quarters (List[str]): All quarters to search.
        
        Returns:
            List[Dict[str, Any]]: Refined chunks from follow-up search with structure:
                [
                    {
                        "chunk_text": str,  # Text content
                        "ticker": str,  # Company ticker
                        "year": int,  # Year of data
                        "quarter": int,  # Quarter number (1-4)
                        "citation": str,  # Citation reference
                        "distance": float,  # Similarity score
                        "search_type": str  # 'vector' or 'keyword'
                    },
                    ...
                ]
        """
        loop = asyncio.get_event_loop()
        print(f"🔍 Follow-up search strategy: has_tickers={has_tickers}, is_general={is_general_question}, is_multi_ticker={is_multi_ticker}")
        
        # Convert 'multiple' string to None for search functions
        search_quarter = None if target_quarter == 'multiple' else target_quarter
        
        if has_tickers and not is_general_question:
            # Ticker-specific follow-up search
            if is_multi_ticker:
                # Multi-ticker parallel search
                print(f"🔄 Multi-ticker follow-up search...")
                refined_chunks = []
                
                # Encode query asynchronously to avoid blocking
                query_embedding = await loop.run_in_executor(
                    None,
                    self.search_engine.embedding_model.encode,
                    [follow_up_question]
                )
                
                # Create async search tasks for each ticker (vector + keyword)
                async def search_ticker(ticker: str):
                    """Search for a single ticker with both vector and keyword."""
                    try:
                        # Vector search
                        vector_chunks = await loop.run_in_executor(
                            None,
                            self.database_manager._search_postgres_with_ticker,
                            query_embedding,
                            ticker,
                            search_quarter
                        )
                        
                        # Keyword search
                        keyword_chunks = await loop.run_in_executor(
                            None,
                            self.search_engine._search_keywords_with_ticker,
                            follow_up_question,
                            ticker,
                            search_quarter
                        )
                        
                        return vector_chunks, keyword_chunks
                    except Exception as e:
                        print(f"   ❌ {ticker} search failed: {e}")
                        return [], []
                
                # Execute all ticker searches in parallel
                ticker_tasks = [search_ticker(ticker) for ticker in tickers_to_process]
                ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
                
                # Collect results
                for i, result in enumerate(ticker_results):
                    if isinstance(result, Exception):
                        print(f"   ❌ {tickers_to_process[i]} search raised exception: {result}")
                    else:
                        vector_chunks, keyword_chunks = result
                        refined_chunks.extend(vector_chunks)
                        refined_chunks.extend(keyword_chunks)
                
                if refined_chunks:
                    from .rag_utils import combine_search_results
                    refined_chunks = combine_search_results(
                        [chunk for chunk in refined_chunks if chunk.get('search_type') != 'keyword'],
                        [chunk for chunk in refined_chunks if chunk.get('search_type') == 'keyword'],
                        self.config.get("vector_weight", 0.7),
                        self.config.get("keyword_weight", 0.3),
                        self.config.get("similarity_threshold", 0.3)
                    )
            else:
                # Single ticker search
                ticker = tickers_to_process[0]
                print(f"🎯 Single ticker follow-up search for {ticker}...")
                
                # Encode query asynchronously
                query_embedding = await loop.run_in_executor(
                    None,
                    self.search_engine.embedding_model.encode,
                    [follow_up_question]
                )
                
                # Run vector and keyword searches in parallel
                vector_task = loop.run_in_executor(
                    None,
                    self.database_manager._search_postgres_with_ticker,
                    query_embedding,
                    ticker,
                    search_quarter
                )
                keyword_task = loop.run_in_executor(
                    None,
                    self.search_engine._search_keywords_with_ticker,
                    follow_up_question,
                    ticker,
                    search_quarter
                )
                
                vector_chunks, keyword_chunks = await asyncio.gather(vector_task, keyword_task)
                
                from .rag_utils import combine_search_results
                refined_chunks = combine_search_results(
                    vector_chunks, keyword_chunks,
                    self.config.get("vector_weight", 0.7),
                    self.config.get("keyword_weight", 0.3),
                    self.config.get("similarity_threshold", 0.3)
                )
        else:
            # General question follow-up search
            print(f"🌐 Follow-up search across all companies...")
            if len(target_quarters) > 1:
                refined_chunks = await self.search_engine._search_multiple_quarters_async(follow_up_question, target_quarters, chunks_per_quarter=self.config.get("chunks_per_quarter", 15))
            else:
                # Single quarter general search - use executor to avoid blocking
                refined_chunks = await loop.run_in_executor(
                    None,
                    self.search_engine.search_similar_chunks,
                    follow_up_question,
                    self.config.get("chunks_per_quarter", 15),
                    search_quarter
                )
        
        return refined_chunks
    
    async def _run_iterative_improvement(self, question: str, individual_results: List[Dict], all_chunks: List[Dict],
                                        all_citations: List[str], is_general_question: bool, is_multi_ticker: bool,
                                        tickers_to_process: List[str], max_iterations: int, show_details: bool,
                                        comprehensive: bool, stream_callback, target_quarter,
                                        target_quarters: List[str], question_analysis: Dict, event_yielder=None, news_context: str = None, ten_k_context: str = None) -> tuple:
        """Run iterative improvement loop to refine the answer.

        The agent generates an initial answer, then iteratively evaluates and improves it
        until the confidence threshold is met or max iterations are reached.

        Flow:
        1. Generate initial answer for evaluation
        2. Iteratively evaluate and improve:
           - Evaluate answer quality with LLM
           - Generate follow-up questions for missing information
           - Search for additional relevant chunks
           - Regenerate answer with enriched context
        3. Generate final streaming answer
        
        Args:
            question (str): The user's original question.
            individual_results (List[Dict]): Search results per ticker (see _execute_search).
            all_chunks (List[Dict]): All retrieved chunks with metadata.
            all_citations (List[str]): All citation references.
            is_general_question (bool): Whether question is general.
            is_multi_ticker (bool): Whether multiple tickers involved.
            tickers_to_process (List[str]): Tickers being processed.
            max_iterations (int): Maximum improvement iterations.
            show_details (bool): Whether to print debug information.
            comprehensive (bool): Whether to use comprehensive mode.
            stream_callback (callable): Callback for streaming responses.
            target_quarter (str): Primary quarter or 'multiple'.
            target_quarters (List[str]): All quarters to search.
            question_analysis (Dict): Question analysis results (see _perform_question_analysis).
        
        Returns:
            tuple: (best_answer, best_confidence, best_citations, best_context_chunks,
                    best_chunks, evaluation_context, follow_up_questions_asked,
                    accumulated_chunks, generation_time) where:
                - best_answer (str): The best generated answer
                - best_confidence (float): Confidence score 0-1
                - best_citations (List): Citation references
                - best_context_chunks (List[str]): Text chunks used
                - best_chunks (List[Dict]): Full chunk objects
                - evaluation_context (List[Dict]): Evaluation history per iteration:
                    [{"iteration": int, "evaluation": Dict, "confidence": float}, ...]
                - follow_up_questions_asked (List[str]): Follow-up questions used
                - accumulated_chunks (List[Dict]): All chunks gathered
                - generation_time (float): Time spent in seconds
        """
        generation_start = time.time()
        print(f"\n🤖 STEP 4: ITERATIVE RESPONSE GENERATION")
        print(f"{'─'*50}")
        rag_logger.info(f"🤖 Step 3: Starting iterative response generation...")

        # If we have neither transcript results nor news context nor 10-K context, there is no structured data to answer from.
        # Instead of immediately suggesting news ourselves, ask the LLM to explain the limitation and (optionally) end with
        # "Do you want me to search the news instead?" so the user can decide.
        # For pure-news or pure-10K flows (news_context or ten_k_context provided), we still run iterative improvement.
        if not individual_results and not news_context and not ten_k_context:
            rag_logger.warning(
                "⚠️ No transcript chunks found for any ticker and no 10-K or news context available - "
                "delegating to response generator with empty context"
            )

            # Let the response generator craft a transparent answer about the lack of data.
            # It will see that there is no transcript / 10-K / news context and follow special
            # instructions in its prompt (including optionally ending with the news-search question).
            empty_answer = self.response_generator.generate_openai_response(
                question=question,
                context_chunks=[],
                chunk_objects=[],
                ticker=tickers_to_process[0] if tickers_to_process else None,
                stream_callback=None,
                news_context=None,
                ten_k_context=None,
                previous_answer=None,
            )

            generation_time = time.time() - generation_start
            return (
                empty_answer,               # best_answer
                0.0,                        # best_confidence
                [],                         # best_citations
                [],                         # best_context_chunks
                [],                         # best_chunks
                [],                         # evaluation_context
                [],                         # follow_up_questions_asked
                [],                         # accumulated_chunks
                generation_time             # generation_time
            )
        
        # Initialize variables
        accumulated_chunks = all_chunks.copy()
        accumulated_citations = all_citations.copy()
        evaluation_context = []
        follow_up_questions_asked = []

        # Create sync retry callback that queues events for the async event_yielder
        # This allows the response generator to notify the frontend about retries
        retry_event_queue = []

        def sync_retry_callback(event):
            """Sync callback that collects retry events for later async yielding"""
            retry_event_queue.append(event)
            # Also log for visibility
            rag_logger.info(f"🔄 Retry event queued: {event.get('message', 'unknown')}")

        async def flush_retry_events():
            """Yield any queued retry events to the frontend"""
            if event_yielder and retry_event_queue:
                for event in retry_event_queue:
                    await event_yielder(event)
                retry_event_queue.clear()

        # Generate initial answer (NO streaming for initial - we'll stream the final answer only)
        print(f"🤖 Generating initial answer for evaluation")
        if is_general_question:
            initial_answer = self.response_generator.generate_multi_ticker_response(
                question, accumulated_chunks, individual_results, show_details, comprehensive, stream_callback=None, news_context=news_context, ten_k_context=ten_k_context, retry_callback=sync_retry_callback
            )
        elif is_multi_ticker and len(individual_results) > 1:
            initial_answer = self.response_generator.generate_multi_ticker_response(
                question, accumulated_chunks, individual_results, show_details, comprehensive, stream_callback=None, news_context=news_context, ten_k_context=ten_k_context, retry_callback=sync_retry_callback
            )
        else:
            initial_answer = self.response_generator.generate_openai_response(
                question, [chunk['chunk_text'] for chunk in accumulated_chunks], accumulated_chunks,
                ticker=tickers_to_process[0] if tickers_to_process else None, stream_callback=None, news_context=news_context, ten_k_context=ten_k_context, retry_callback=sync_retry_callback
            )
        # Flush any retry events that occurred during initial answer generation
        await flush_retry_events()
        
        best_answer = initial_answer
        best_confidence = 0.0
        best_citations = accumulated_citations.copy()
        best_context_chunks = [chunk['chunk_text'] for chunk in accumulated_chunks]
        best_chunks = accumulated_chunks.copy()
        
        # Debug: Log initial citations
        initial_news = [c for c in accumulated_citations if isinstance(c, dict) and c.get('type') == 'news']
        initial_10k = [c for c in accumulated_citations if isinstance(c, dict) and c.get('type') == '10-K']
        rag_logger.info(f"🔍 DEBUG: Initial accumulated_citations contains {len(initial_news)} news, {len(initial_10k)} 10-K, {len(accumulated_citations)} total")
        
        # Add user-friendly limit messages
        if question_analysis.get('limits_exceeded'):
            from .rag_utils import generate_user_friendly_limit_message
            limit_message = generate_user_friendly_limit_message(question_analysis['limits_exceeded'])
            if limit_message:
                best_answer = limit_message + "\n\n" + best_answer
        
        # Iterative improvement loop
        print(f"🔄 Starting iterative improvement loop (max {max_iterations} iterations)")
        print(f"🔍 DEBUG: max_iterations type: {type(max_iterations)}, value: {max_iterations}")
        print(f"🔍 DEBUG: range(max_iterations) = {list(range(max_iterations))}")
        
        for iteration in range(max_iterations):
            print(f"\n🔄 ITERATION {iteration + 1}/{max_iterations}")
            rag_logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")

            # Log iteration start to Logfire
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info(
                    "rag.iteration.start",
                    iteration=iteration + 1,
                    max_iterations=max_iterations,
                    current_chunks=len(accumulated_chunks),
                    current_confidence=best_confidence
                )
            
            # Stream iteration start event - SEND IMMEDIATELY IN REAL-TIME
            if event_yielder:
                await event_yielder({
                    'type': 'iteration_start',
                    'message': 'Refining answer with additional context',
                    'step': 'iteration',
                    'data': {
                        'iteration': iteration + 1,
                        'total_iterations': max_iterations,
                        'current_chunks': len(accumulated_chunks),
                        'current_confidence': best_confidence
                    }
                })
                # Yield control to allow the event to be sent before evaluation starts
                await asyncio.sleep(0.01)
            
            # Perform strict evaluation to decide if we should iterate
            
            # Evaluate answer quality
            from .rag_utils import assess_answer_quality
            answer_quality = assess_answer_quality(best_answer, len(accumulated_chunks))
            
            # Pass conversation memory and ID for context-aware evaluation
            conversation_memory = self.question_analyzer.conversation_memory if hasattr(self, 'question_analyzer') else None
            conversation_id_for_eval = question_analysis.get('conversation_id') if question_analysis else None
            reasoning_context = question_analysis.get('reasoning_statement')  # Get the agent's initial reasoning

            # Evaluate answer quality with full context (including reasoning)
            # Pass data_source to respect routing (don't search transcripts if user asked for 10k only)
            current_data_source = question_analysis.get('data_source', 'earnings_transcripts') if question_analysis else 'earnings_transcripts'

            # Wrap evaluation in try-except to gracefully handle LLM API errors
            # If evaluation fails but we have an answer, return what we have instead of crashing
            try:
                if answer_quality['is_insufficient']:
                    evaluation = await self.response_generator.evaluate_answer_quality(
                        question, best_answer, [chunk['chunk_text'] for chunk in accumulated_chunks], accumulated_chunks,
                        conversation_memory=conversation_memory, conversation_id=conversation_id_for_eval,
                        follow_up_questions_asked=follow_up_questions_asked,
                        evaluation_context=evaluation_context,
                        reasoning_context=reasoning_context,
                        data_source=current_data_source
                    )
                else:
                    evaluation = await self.response_generator.evaluate_answer_quality(
                        question, best_answer, [chunk['chunk_text'] for chunk in accumulated_chunks],
                        conversation_memory=conversation_memory, conversation_id=conversation_id_for_eval,
                        follow_up_questions_asked=follow_up_questions_asked,
                        evaluation_context=evaluation_context,
                        reasoning_context=reasoning_context,
                        data_source=current_data_source
                    )
            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
                # LLM API error during evaluation - if we have an answer, return it gracefully
                rag_logger.warning(f"⚠️ LLM API error during evaluation at iteration {iteration + 1}: {e}")
                if best_answer:
                    rag_logger.info(f"✅ Returning existing answer (generated before API error)")
                    if event_yielder:
                        await event_yielder({
                            'type': 'api_retry',
                            'message': f'API error during evaluation, returning best answer so far',
                            'step': 'iteration',
                            'data': {'error': str(e), 'iteration': iteration + 1, 'graceful_fallback': True}
                        })
                    break  # Exit iteration loop and return what we have
                else:
                    raise  # No answer yet, propagate the error

            evaluation_confidence = evaluation.get('overall_confidence', 0.5)
            print(f"📊 Evaluation: confidence={evaluation_confidence:.3f}")

            # Log evaluation to Logfire
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info(
                    "rag.iteration.evaluation",
                    iteration=iteration + 1,
                    confidence=evaluation_confidence,
                    should_iterate=evaluation.get('should_iterate', False),
                    needs_news_search=evaluation.get('needs_news_search', False),
                    needs_transcript_search=evaluation.get('needs_transcript_search', False),
                    completeness_score=evaluation.get('completeness_score', 0),
                    specificity_score=evaluation.get('specificity_score', 0)
                )

            # Track searches performed in this iteration (will be updated after searches complete)
            iteration_searches = {
                'transcript_search_performed': False,
                'transcript_search_results': 0,
                'news_search_performed': False,
                'news_search_results': 0,
                'ten_k_search_performed': False,
                'ten_k_search_results': 0
            }

            evaluation_context.append({
                'iteration': iteration + 1,
                'evaluation': evaluation,
                'confidence': evaluation_confidence,
                'searches_performed': iteration_searches  # Will be updated after searches
            })
            
            # Note: Evaluation happens internally - frontend only sees action traces

            if evaluation_confidence > best_confidence:
                best_confidence = evaluation_confidence
            
            # Extract evaluation details - let the agent decide
            follow_up_questions = evaluation.get('follow_up_questions', [])
            should_iterate = evaluation.get('should_iterate', False)
            needs_news_search = evaluation.get('needs_news_search', False)
            news_search_query = evaluation.get('news_search_query', None)
            needs_transcript_search = evaluation.get('needs_transcript_search', False)
            transcript_search_query = evaluation.get('transcript_search_query', None)
            iteration_reasoning = evaluation.get('iteration_decision_reasoning', '')
            completeness_score = evaluation.get('completeness_score', 10)
            specificity_score = evaluation.get('specificity_score', 10)

            # Track all new chunks found in THIS iteration (from all search types)
            # These will be used for answer generation along with previous_answer
            iteration_new_chunks = []
            
            # Log the agent's decision (but don't print evaluation reasoning to avoid it appearing in frontend)
            print(f"\n🤖 Agent Decision: {'ITERATE' if should_iterate else 'STOP'}")
            if iteration_reasoning:
                # Only log to rag_logger, don't print (print statements might get captured)
                rag_logger.info(f"💭 Agent reasoning: {iteration_reasoning[:100]}...")  # Truncate for logs
            else:
                rag_logger.info(f"🤖 Agent iteration decision: {should_iterate} (no explicit reasoning)")
            
            # Stream the agent's decision to frontend
            if event_yielder and should_iterate:
                await event_yielder({
                    'type': 'agent_decision',
                    'message': 'Analyzing answer quality',
                    'step': 'iteration',
                    'data': {
                        'iteration': iteration + 1,
                        'should_iterate': should_iterate,
                        'confidence': evaluation_confidence,
                        'completeness_score': completeness_score,
                        'specificity_score': specificity_score,
                        'accuracy_score': evaluation.get('accuracy_score', 0),
                        'clarity_score': evaluation.get('clarity_score', 0),
                        'has_follow_up_questions': len(follow_up_questions) > 0,
                        'follow_up_count': len(follow_up_questions)
                    }
                })
                await asyncio.sleep(0.01)
            
            # Respect the agent's decision, but cap at max iterations
            should_stop = False
            stop_reason = None

            if iteration == max_iterations - 1:
                print(f"🏁 Stopping: reached max iterations ({max_iterations})")
                should_stop = True
                stop_reason = f"Reached max iterations ({max_iterations})"
            elif evaluation_confidence >= 0.9:
                print(f"🏁 Stopping: confidence score {evaluation_confidence:.1%} exceeds 90% threshold")
                rag_logger.info(f"🎯 Auto-stop triggered: confidence {evaluation_confidence:.3f} >= 0.9")
                should_stop = True
                stop_reason = f"High confidence score ({evaluation_confidence:.1%})"
            elif not should_iterate:
                print(f"🏁 Stopping: agent determined answer is sufficient")
                should_stop = True
                stop_reason = "Agent determined answer is sufficient"
            elif not follow_up_questions:
                print(f"🏁 Stopping: agent wants to iterate but provided no follow-up questions")
                should_stop = True
                stop_reason = "No follow-up questions generated"
            else:
                print(f"🔄 Continuing: agent identified opportunities for improvement")
                should_stop = False
            
            if should_stop:
                # Stream final status
                if event_yielder and iteration < max_iterations - 1:
                    await event_yielder({
                        'type': 'iteration_complete',
                        'message': 'Research complete',
                        'step': 'iteration',
                        'data': {
                            'iteration': iteration + 1,
                            'total_iterations': max_iterations,
                            'final_confidence': evaluation_confidence,
                            'completeness_score': completeness_score,
                            'specificity_score': specificity_score,
                            'reason': stop_reason or 'Answer quality sufficient'
                        }
                    })
                    await asyncio.sleep(0.01)
                break
            
            # Check if agent decided to search for transcripts
            # SAFETY CHECK: Respect data_source routing - don't search transcripts if user asked for 10k/news only
            if needs_transcript_search and current_data_source in ['10k', 'latest_news']:
                rag_logger.info(f"🚫 Skipping transcript search - data_source='{current_data_source}' (user explicitly asked for {current_data_source} only)")
                needs_transcript_search = False  # Override the evaluation's decision

            if needs_transcript_search and transcript_search_query:
                rag_logger.info(f"📄 Agent decided to search transcripts: '{transcript_search_query}'")
                print(f"📄 Agent decided to search earnings transcripts...")
                
                # Stream transcript search event
                if event_yielder:
                    await event_yielder({
                        'type': 'iteration_transcript_search',
                        'message': 'Searching earnings transcripts to enhance answer',
                        'step': 'iteration',
                        'data': {
                            'iteration': iteration + 1,
                            'query': transcript_search_query
                        }
                    })
                    await asyncio.sleep(0.01)
                
                # Perform transcript search using the follow-up question
                has_tickers = bool(tickers_to_process)
                transcript_chunks = await self._perform_parallel_follow_up_search(
                    [transcript_search_query], has_tickers, is_general_question, is_multi_ticker,
                    tickers_to_process, target_quarter, target_quarters
                )
                
                # Track that transcript search was performed
                iteration_searches['transcript_search_performed'] = True
                iteration_searches['transcript_search_results'] = len(transcript_chunks) if transcript_chunks else 0

                if transcript_chunks:
                    rag_logger.info(f"✅ Found {len(transcript_chunks)} transcript chunks from agent decision")
                    print(f"   ✅ Found {len(transcript_chunks)} transcript chunks")

                    # Merge with existing chunks and track for this iteration
                    existing_citations = {chunk['citation'] for chunk in accumulated_chunks}
                    for chunk in transcript_chunks:
                        if chunk['citation'] not in existing_citations:
                            accumulated_chunks.append(chunk)
                            iteration_new_chunks.append(chunk)  # Track for this iteration's answer
                            existing_citations.add(chunk['citation'])

                            # Add citation
                            citation_entry = {
                                "type": "transcript",
                                "marker": f"[{chunk['citation']}]",
                                "ticker": chunk.get('ticker', 'Unknown'),
                                "quarter": f"Q{chunk.get('quarter', '?')} {chunk.get('year', '?')}",
                                "chunk_index": chunk.get('citation', '')
                            }
                            accumulated_citations.append(citation_entry)
                else:
                    rag_logger.warning("⚠️ No transcript chunks found from agent-initiated search")
            
            # Check if agent decided to search for news
            if needs_news_search and self.tavily_service.is_available() and news_search_query:
                rag_logger.info(f"📰 Agent decided to search for news: '{news_search_query}'")
                print(f"📰 Agent decided to search for latest news...")
                
                # Stream news search event
                if event_yielder:
                    await event_yielder({
                        'type': 'iteration_news_search',
                        'message': 'Searching for latest news to enhance answer',
                        'step': 'iteration',
                        'data': {
                            'iteration': iteration + 1,
                            'query': news_search_query
                        }
                    })
                    await asyncio.sleep(0.01)
                
                # Perform news search
                iteration_news_results = self.tavily_service.search_news(news_search_query, max_results=5, include_answer="advanced")

                # Track that news search was performed
                iteration_searches['news_search_performed'] = True
                iteration_searches['news_search_results'] = len(iteration_news_results.get('results', [])) if iteration_news_results else 0

                if iteration_news_results.get("results"):
                    rag_logger.info(f"✅ Found {len(iteration_news_results['results'])} news articles from agent decision")
                    print(f"   ✅ Found {len(iteration_news_results['results'])} news articles")
                    
                    # Update news_context with new results
                    if iteration_news_results:
                        # Format news context and merge with existing if any
                        new_news_context = self.tavily_service.format_news_context(iteration_news_results)
                        if news_context:
                            # Merge news contexts
                            news_context = f"{news_context}\n\n=== ADDITIONAL NEWS (from iteration) ===\n{new_news_context}"
                        else:
                            news_context = new_news_context
                        
                        # Add news citations
                        new_news_citations = self.tavily_service.get_news_citations(iteration_news_results)
                        for news_citation in new_news_citations:
                            citation_entry = {
                                "type": "news",
                                "marker": f"[N{news_citation['index']}]",
                                "title": news_citation["title"],
                                "url": news_citation["url"],
                                "published_date": news_citation.get("published_date", "")
                            }
                            accumulated_citations.append(citation_entry)
                            rag_logger.info(f"🌐 Added iteration news citation: marker={citation_entry['marker']}, title={citation_entry['title'][:60]}, url={citation_entry['url'][:50]}")
                        rag_logger.info(f"📎 After adding iteration news: {len(accumulated_citations)} total citations")
                else:
                    rag_logger.warning("⚠️ No news results found from agent-initiated search")
            
            # Generate context-aware questions if needed
            if answer_quality['is_insufficient']:
                context_aware_questions = self._generate_context_aware_follow_up_questions(question, best_answer, accumulated_chunks)
                all_follow_up_questions = list(set(follow_up_questions + context_aware_questions))
            else:
                all_follow_up_questions = follow_up_questions
            
            # SAFETY CHECK: Skip transcript-based follow-up search for 10k/news only queries
            # The follow-up search uses the transcript database, so it should be skipped
            # when the user explicitly asked for a different data source
            skip_followup_transcript_search = current_data_source in ['10k', 'latest_news']

            if skip_followup_transcript_search and all_follow_up_questions:
                rag_logger.info(f"🚫 Skipping follow-up transcript search - data_source='{current_data_source}' (user explicitly asked for {current_data_source} only)")
                print(f"🚫 Skipping transcript search - using {current_data_source} data only")
                all_follow_up_questions = []  # Clear follow-up questions to prevent transcript search

            if all_follow_up_questions:
                # Use ALL follow-up questions (not just the first one)
                # This retrieves ~5 chunks per question, so 3 questions = ~15 chunks total
                follow_up_questions_asked.extend(all_follow_up_questions)

                print(f"🔄 Searching with {len(all_follow_up_questions)} follow-up questions in parallel:")
                for i, q in enumerate(all_follow_up_questions, 1):
                    print(f"   {i}. {q}")

                # Stream follow-up question event
                if event_yielder:
                    questions_list = '\n'.join([f'Searching: "{q}"' for q in all_follow_up_questions])
                    await event_yielder({
                        'type': 'iteration_followup',
                        'message': questions_list,
                        'step': 'iteration',
                        'data': {
                            'iteration': iteration + 1,
                            'followup_question': all_follow_up_questions[0],
                            'all_questions': all_follow_up_questions
                        }
                    })
                    await asyncio.sleep(0.01)
                
                # Search for additional chunks using ALL follow-up questions in parallel
                has_tickers = bool(tickers_to_process)
                refined_chunks = await self._perform_parallel_follow_up_search(
                    all_follow_up_questions, has_tickers, is_general_question, is_multi_ticker,
                    tickers_to_process, target_quarter, target_quarters
                )
                
                if refined_chunks:
                    # Merge with existing chunks
                    existing_citations = {chunk['citation'] for chunk in accumulated_chunks}
                    new_chunks = [chunk for chunk in refined_chunks if chunk['citation'] not in existing_citations]
                    accumulated_chunks.extend(new_chunks)
                    iteration_new_chunks.extend(new_chunks)  # Track for this iteration's answer
                    accumulated_citations.extend([chunk['citation'] for chunk in new_chunks])

                    print(f"   🔍 Added {len(new_chunks)} new chunks from follow-up search")

                    # Stream search results event
                    if event_yielder:
                        await event_yielder({
                            'type': 'iteration_search',
                            'message': 'Incorporating additional relevant sources',
                            'step': 'iteration',
                            'data': {
                                'iteration': iteration + 1,
                                'new_chunks_count': len(new_chunks),
                                'total_chunks': len(accumulated_chunks),
                                'sources': list(set([chunk.get('ticker', 'Unknown') for chunk in new_chunks]))
                            }
                        })
                        await asyncio.sleep(0.01)

            # Regenerate answer if we found ANY new chunks in this iteration (from any search type)
            if iteration_new_chunks:
                rag_logger.info(f"📝 Regenerating answer with {len(iteration_new_chunks)} new chunks from this iteration + previous answer")

                # Regenerate answer using only NEW chunks from this iteration + previous answer
                # The previous answer is a "compression" of all prior chunks, so we don't need to re-send them
                # This is more token-efficient and ensures the LLM focuses on new information
                # Wrap in try-except to handle LLM API errors gracefully
                try:
                    if is_general_question or (is_multi_ticker and len(individual_results) > 1):
                        refined_answer = self.response_generator.generate_multi_ticker_response(
                            question, iteration_new_chunks, individual_results, show_details, comprehensive, stream_callback=None, news_context=news_context, ten_k_context=ten_k_context, previous_answer=best_answer, retry_callback=sync_retry_callback
                        )
                    else:
                        refined_answer = self.response_generator.generate_openai_response(
                            question, [chunk['chunk_text'] for chunk in iteration_new_chunks], iteration_new_chunks,
                            ticker=tickers_to_process[0] if tickers_to_process else None, stream_callback=None, news_context=news_context, ten_k_context=ten_k_context, previous_answer=best_answer, retry_callback=sync_retry_callback
                        )
                    # Flush any retry events that occurred during refinement
                    await flush_retry_events()

                    best_answer = refined_answer
                    best_citations = accumulated_citations.copy()
                    best_context_chunks = [chunk['chunk_text'] for chunk in accumulated_chunks]
                    best_chunks = accumulated_chunks.copy()
                except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
                    # LLM API error during refinement - keep the existing answer and stop iterating
                    rag_logger.warning(f"⚠️ LLM API error during answer refinement at iteration {iteration + 1}: {e}")
                    rag_logger.info(f"✅ Keeping previous answer (before API error)")
                    if event_yielder:
                        await event_yielder({
                            'type': 'api_retry',
                            'message': f'API error during refinement, keeping previous answer',
                            'step': 'iteration',
                            'data': {'error': str(e), 'iteration': iteration + 1, 'graceful_fallback': True}
                        })
                    await flush_retry_events()
                    break  # Exit iteration loop and return what we have
            else:
                rag_logger.info(f"📝 No new chunks found in iteration {iteration + 1}, keeping previous answer")
        
        # After all iterations complete, generate/regenerate the final answer WITH streaming
        if stream_callback:
            if best_answer is None or stream_callback:
                print(f"🎬 Generating final answer with streaming...")
                if event_yielder:
                    await event_yielder({
                        'type': 'iteration_final',
                        'message': 'Preparing comprehensive response',
                        'step': 'iteration',
                        'data': {
                            'total_iterations': max_iterations,
                            'final_chunks': len(accumulated_chunks),
                            'final_confidence': best_confidence
                        }
                    })
                    # Yield control to allow the final event to be sent before answer streaming starts
                    await asyncio.sleep(0.01)

                # Generate/regenerate with streaming enabled
                # Wrap in try-except to handle LLM API errors gracefully
                try:
                    if is_general_question or (is_multi_ticker and len(individual_results) > 1):
                        final_answer = self.response_generator.generate_multi_ticker_response(
                            question, accumulated_chunks, individual_results, show_details, comprehensive, stream_callback=stream_callback, news_context=news_context, ten_k_context=ten_k_context, retry_callback=sync_retry_callback
                        )
                    else:
                        final_answer = self.response_generator.generate_openai_response(
                            question, [chunk['chunk_text'] for chunk in accumulated_chunks], accumulated_chunks,
                            ticker=tickers_to_process[0] if tickers_to_process else None, stream_callback=stream_callback, news_context=news_context, ten_k_context=ten_k_context, retry_callback=sync_retry_callback
                        )
                    # Flush any retry events that occurred during final generation
                    await flush_retry_events()

                    # Update with the generated/streamed version
                    best_answer = final_answer
                except (openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.APIStatusError) as e:
                    # LLM API error during final streaming - return existing answer if we have one
                    rag_logger.warning(f"⚠️ LLM API error during final answer generation: {e}")
                    await flush_retry_events()
                    if best_answer:
                        rag_logger.info(f"✅ Returning existing answer (generated before API error)")
                        if event_yielder:
                            await event_yielder({
                                'type': 'api_retry',
                                'message': f'API error during final generation, returning best answer',
                                'step': 'final',
                                'data': {'error': str(e), 'graceful_fallback': True}
                            })
                        # Stream the existing answer to the callback so the frontend receives it
                        if stream_callback:
                            stream_callback(best_answer)
                    else:
                        raise  # No answer at all, propagate the error
        
        generation_time = time.time() - generation_start
        print(f"\n⏱️  GENERATION PHASE COMPLETED in {generation_time:.3f}s")
        
        # Log detailed generation breakdown
        logger.info("=" * 80)
        logger.info("⏱️  RAG PIPELINE TIMING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"📊 Total Generation Time: {generation_time:.3f}s")
        logger.info(f"   - Max iterations: {max_iterations}")
        logger.info(f"   - Actual iterations: {len(evaluation_context)}")
        logger.info(f"   - Follow-up questions asked: {len(follow_up_questions_asked)}")
        logger.info(f"   - Total chunks accumulated: {len(accumulated_chunks)}")
        logger.info(f"   - Final confidence: {best_confidence:.3f}")
        logger.info("=" * 80)
        
        return (best_answer, best_confidence, best_citations, best_context_chunks, best_chunks,
                evaluation_context, follow_up_questions_asked, accumulated_chunks, generation_time)

    def _generate_context_aware_follow_up_questions(self, original_question: str, current_answer: str, available_chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate follow-up questions based on available context and gaps in the answer.
        
        Analyzes the available chunks and current answer to suggest intelligent follow-up
        questions that would help find missing or additional relevant information.
        Considers what data is already available to avoid redundant searches.
        
        Args:
            original_question (str): The user's original question.
            current_answer (str): The currently generated answer.
            available_chunks (List[Dict[str, Any]]): Chunks available in context with structure:
                [
                    {
                        "chunk_text": str,  # The text content
                        "ticker": str,  # Company ticker
                        "distance": float,  # Similarity score
                        "metadata": {
                            "date": str,  # Date of the data
                            ...
                        }
                    },
                    ...
                ]
        
        Returns:
            List[str]: List of 2-3 context-aware follow-up questions, e.g.:
                ["What specific metrics were mentioned?", "Are there any risks discussed?"]
        
        Raises:
            Exception: If OpenAI is not available.
        """
        if not self.openai_available:
            raise Exception("OpenAI not available for context-aware follow-up questions")
        if not available_chunks:
            return []
        
        max_retries = 2  # Keep it small
        retry_delay = 0  # No delays
        
        for attempt in range(max_retries):
            try:
                # Get context-aware follow-up prompt from centralized prompts
                analysis_prompt = get_context_aware_followup_prompt(original_question, current_answer, available_chunks)

                # Detailed LLM stage logging for context-aware follow-up questions
                rag_logger.info(f"🤖 ===== CONTEXT-AWARE FOLLOW-UP QUESTIONS LLM CALL ===== (attempt {attempt + 1}/{max_retries})")
                rag_logger.info(f"🔍 Model: {self.config.get('evaluation_model', 'gpt-4.1-mini-2025-04-14')}")
                rag_logger.info(f"📊 Max tokens: 300")
                rag_logger.info(f"🌡️ Temperature: 0.3")
                rag_logger.info(f"📝 Original question: {original_question}")
                rag_logger.info(f"📊 Current answer length: {len(current_answer)} characters")
                rag_logger.info(f"📊 Available chunks count: {len(available_chunks)}")
                rag_logger.info(f"📋 Analysis prompt length: {len(analysis_prompt)} characters")
                rag_logger.info(f"📋 Analysis prompt preview: {analysis_prompt[:300]}...")
                
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.config.get("evaluation_model", "gpt-4.1-mini-2025-04-14"),
                    messages=[
                        {"role": "system", "content": CONTEXT_AWARE_FOLLOWUP_SYSTEM_PROMPT},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                call_time = time.time() - start_time
                
                # Detailed response logging
                rag_logger.info(f"✅ ===== CONTEXT-AWARE FOLLOW-UP QUESTIONS LLM RESPONSE ===== (call time: {call_time:.3f}s)")
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
                
                response_text = response.choices[0].message.content.strip()
                rag_logger.info(f"📝 Raw context-aware response: {response_text[:100]}...")
                
                # Clean up the response (remove any markdown formatting)
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                

                try:
                    follow_up_questions = json.loads(response_text)
                    if isinstance(follow_up_questions, list) and len(follow_up_questions) > 0:
                        rag_logger.info(f"✅ ===== CONTEXT-AWARE QUESTIONS PARSING SUCCESSFUL =====")
                        rag_logger.info(f"📊 Questions count: {len(follow_up_questions)}")
                        rag_logger.info(f"📝 Generated questions: {follow_up_questions}")
                        rag_logger.info(f"✅ Successfully parsed context-aware questions on attempt {attempt + 1}")
                        return follow_up_questions
                    else:
                        raise ValueError("Invalid or empty list returned")
                except (json.JSONDecodeError, ValueError) as e:
                    rag_logger.warning(f"⚠️ JSON parsing failed on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        # Return default questions if all retries fail
                        rag_logger.error(f"❌ All {max_retries} JSON parsing attempts failed, returning default questions")
                        return ["What additional details can you provide?", "Are there any specific metrics mentioned?", "What risks or challenges were discussed?"]
                    else:
                        # Immediate retry for JSON parsing issues
                        rag_logger.info(f"🔄 Immediate retry for JSON parsing (attempt {attempt + 2}/{max_retries})")
                        continue
                        
            except Exception as e:
                rag_logger.warning(f"⚠️ Context-aware question generation failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    # Return default questions if all retries fail
                    rag_logger.error(f"❌ All {max_retries} attempts failed, returning default questions")
                    return ["What additional details can you provide?", "Are there any specific metrics mentioned?", "What risks or challenges were discussed?"]
                else:
                    # Immediate retry for API errors
                    rag_logger.info(f"🔄 Immediate retry for API error (attempt {attempt + 2}/{max_retries})")
                    continue

    async def execute_rag_flow(self, question: str, show_details: bool = False, comprehensive: bool = True, stream_callback=None, max_iterations: int = None, conversation_id: str = None, stream: bool = True):
        """
        ╔═══════════════════════════════════════════════════════════════════════╗
        ║  MAIN RAG FLOW EXECUTION - Earnings Transcript Q&A Pipeline          ║
        ╚═══════════════════════════════════════════════════════════════════════╝

        Execute the complete RAG flow with iterative improvement:

        Pipeline Stages:
        ────────────────
        1. Setup & Initialization
        2. Question Analysis (ticker extraction, intent detection)
        2.1. Question Reasoning (planning approach)
        3. Search Execution (vector + keyword search)
        4. Initial Answer Generation
        5. Iterative Improvement
        6. Final Response Assembly

        Args:
            question: User's question to answer
            show_details: Print debug information (default: False)
            comprehensive: Use comprehensive mode for multi-ticker (default: True)
            stream_callback: Callback for streaming responses
            max_iterations: Max improvement iterations (default: 3)
            conversation_id: Unique conversation ID for memory tracking
            stream: Whether to yield progress events (default: True)

        Yields (if stream=True):
            Event dictionaries with type, message, step, and data

        Returns (if stream=False):
            Complete response object with answer, citations, and metadata
        """

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 1: SETUP & INITIALIZATION
        # ═════════════════════════════════════════════════════════════════════

        if stream:
            yield {'type': 'progress', 'message': 'Starting analysis...', 'step': 'init', 'data': {}}

        # Setup and initialization
        start_time = time.time()

        # Use configuration default if max_iterations not provided
        if max_iterations is None:
            max_iterations = self.config.get("max_iterations", 3)

        rag_logger.info(f"🚀 Starting complete RAG flow")
        rag_logger.info(f"📝 Question: '{question}'")
        rag_logger.info(f"🔄 Max iterations: {max_iterations}")

        # Start Logfire span for the entire RAG flow
        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "rag.flow.start",
                question=question,
                max_iterations=max_iterations,
                conversation_id=conversation_id,
                comprehensive=comprehensive,
                stream=stream
            )

        # Start analytics tracking
        await self.analytics_logger.start_pipeline(
            original_question=question,
            user_id=None,
            conversation_id=conversation_id
        )

        # Debug output
        print(f"\n{'='*80}")
        print(f"🔍 RAG FLOW DEBUG - DETAILED ANALYSIS")
        print(f"{'='*80}")
        print(f"📝 Question: '{question}'")
        print(f"🔄 Max iterations: {max_iterations}")
        print(f"⏰ Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        print(f"{'='*80}")


        # ═════════════════════════════════════════════════════════════════════
        # STAGE 2: QUESTION ANALYSIS & VALIDATION
        # ═════════════════════════════════════════════════════════════════════

        analysis_start = time.time()
        if stream:
            yield {'type': 'progress', 'message': 'Analyzing question...', 'step': 'analysis', 'data': {}}

        question_analysis, target_quarters, early_return = await self._perform_question_analysis(question, conversation_id)
        print("\n\n\n")
        print("QUESTION ANALYSIS: ", question_analysis)
        print("\n\n\n")
        analysis_time = time.time() - analysis_start

        # Handle early returns (errors or rejected questions) BEFORE logging
        # This handles invalid/rejected questions where question_analysis is None
        if early_return:
            analysis = early_return.get('analysis', {})

            # Check if this is a rejected question (not an error - just can't help with this topic)
            if analysis.get('status') == 'rejected':
                message = analysis.get('message', 'I can only help with public company financial data.')
                suggestions = analysis.get('suggestions', [])

                # Yield as 'rejected' type - frontend should display this nicely, not as an error
                yield {
                    'type': 'rejected',
                    'message': message,
                    'step': 'complete',
                    'data': {
                        'suggestions': suggestions,
                        'original_question': early_return.get('original_question', question)
                    }
                }
                return  # Stop generator

            # For actual errors, extract error message
            error_msg = early_return.get('error')
            if not error_msg and 'errors' in early_return and early_return['errors']:
                error_msg = early_return['errors'][0]
            if not error_msg:
                error_msg = 'Unknown error'

            yield {'type': 'error', 'message': error_msg, 'step': 'analysis', 'data': early_return}
            return  # Stop generator

        # Log question analysis to Logfire (only for valid questions)
        if LOGFIRE_AVAILABLE and logfire and question_analysis:
            logfire.info(
                "rag.question_analysis",
                tickers=question_analysis.get('extracted_tickers', []),
                data_source=question_analysis.get('data_source', 'earnings_transcripts'),
                needs_10k=question_analysis.get('needs_10k', False),
                needs_latest_news=question_analysis.get('needs_latest_news', False),
                quarter_context=question_analysis.get('quarter_context', 'latest'),
                target_quarters=target_quarters,
                confidence=question_analysis.get('confidence', 0),
                analysis_time_ms=int(analysis_time * 1000)
            )

        # Emit analysis complete
        if stream:
            tickers = question_analysis.get('extracted_tickers', [])
            # Question analyzer returns "reason" field
            reasoning = question_analysis.get('reason', '')
            
            # Use reasoning as the main message if available
            message = reasoning if reasoning else (
                f"Analyzing {'and '.join(tickers) if len(tickers) <= 2 else f'{len(tickers)} companies'}"
                if tickers else "General financial question detected"
            )
            
            yield {
                'type': 'analysis',
                'message': message,
                'step': 'analysis',
                'data': {
                    'tickers': tickers,
                    'target_quarters': target_quarters,  # Send all target quarters
                    'quarter_context': question_analysis.get('quarter_context', 'latest'),
                    'reasoning': reasoning,  # Send the actual reasoning from question analyzer
                    'question_type': question_analysis.get('question_type', ''),  # Keep for debugging
                    'confidence': question_analysis.get('confidence', 0)
                }
            }


        # ═════════════════════════════════════════════════════════════════════
        # STAGE 2.1: QUESTION PLANNING/REASONING
        # ═════════════════════════════════════════════════════════════════════

        if stream:
            yield {'type': 'progress', 'message': 'Planning research approach...', 'step': 'planning', 'data': {}}

        rag_logger.info("🧠 Generating research reasoning...")
        print(f"\n🧠 STAGE 2.1: QUESTION PLANNING/REASONING")
        print(f"{'─'*60}")

        reasoning_statement = None
        try:
            reasoning_statement = await self.response_generator.plan_question_approach(question, question_analysis)

            print(f"📋 Reasoning: {reasoning_statement}")
            print(f"{'─'*60}")

            # Stream the reasoning to frontend
            if stream:
                yield {
                    'type': 'reasoning',
                    'message': reasoning_statement,
                    'step': 'planning',
                    'data': {
                        'reasoning': reasoning_statement
                    }
                }

            # Log to Logfire
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info(
                    "rag.reasoning.complete",
                    reasoning=reasoning_statement
                )

        except Exception as e:
            rag_logger.error(f"❌ Reasoning failed: {e}")
            print(f"⚠️ Reasoning failed: {e}, continuing with basic approach")
            tickers = [normalize_ticker(t) for t in question_analysis.get('extracted_tickers', []) if t]
            ticker_text = f" for {', '.join(tickers)}" if tickers else ""
            reasoning_statement = f"The user is asking about{ticker_text}: {question}. I will search the available financial data to answer this."

        # Store reasoning in question_analysis for use in RAG loop
        question_analysis['reasoning_statement'] = reasoning_statement


        # ═════════════════════════════════════════════════════════════════════
        # STAGE 2.5: NEWS SEARCH (if needed)
        # ═════════════════════════════════════════════════════════════════════

        news_results = None
        needs_latest_news = question_analysis.get('needs_latest_news', False)
        
        if needs_latest_news:
            if self.tavily_service.is_available():
                if stream:
                    yield {'type': 'progress', 'message': 'Searching latest news...', 'step': 'news_search', 'data': {}}
                
                rag_logger.info("📰 Question requires latest news - searching Tavily...")
                
                # Build news search query
                extracted_tickers = [normalize_ticker(t) for t in question_analysis.get('extracted_tickers', []) if t]
                if extracted_tickers:
                    # Include ticker in search query
                    news_query = f"{question} {' '.join(extracted_tickers)}"
                else:
                    news_query = question
                
                # Perform news search
                news_results = self.tavily_service.search_news(news_query, max_results=5, include_answer="advanced")
                
                if news_results.get("results"):
                    rag_logger.info(f"✅ Found {len(news_results['results'])} news articles")

                    # Log news search to Logfire
                    if LOGFIRE_AVAILABLE and logfire:
                        logfire.info(
                            "rag.news_search",
                            query=news_query,
                            articles_found=len(news_results['results']),
                            tickers=extracted_tickers
                        )

                    if stream:
                        yield {
                            'type': 'news_search',
                            'message': f'Found {len(news_results["results"])} recent news articles',
                            'step': 'news_search',
                            'data': {
                                'articles_count': len(news_results["results"]),
                                'query': news_query
                            }
                        }
                else:
                    rag_logger.warning("⚠️ No news results found")
                    news_results = None
            else:
                rag_logger.warning("⚠️ Question requires latest news, but Tavily service is not available. Install tavily-python and set TAVILY_API_KEY to enable news search.")
                if stream:
                    yield {
                        'type': 'news_search',
                        'message': 'News search unavailable - Tavily not configured',
                        'step': 'news_search',
                        'data': {
                            'error': 'Tavily service not available. Install tavily-python and set TAVILY_API_KEY environment variable.'
                        }
                    }

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 2.6: 10-K SEC FILINGS SEARCH (if needed)
        # ═════════════════════════════════════════════════════════════════════

        ten_k_results = []
        needs_10k = question_analysis.get('needs_10k', False)
        data_source = question_analysis.get('data_source', 'earnings_transcripts')
        
        # Check if 10k is needed: either via needs_10k flag OR data_source field
        should_search_10k = needs_10k or data_source in ['10k', 'hybrid']

        if should_search_10k:
            if stream:
                yield {'type': 'progress', 'message': 'Searching 10-K filings...', 'step': '10k_search', 'data': {}}

            rag_logger.info(f"📄 Question requires 10-K data (needs_10k={needs_10k}, data_source={data_source}) - searching SEC filings...")

            # Get tickers from question analysis
            extracted_tickers = [normalize_ticker(t) for t in question_analysis.get('extracted_tickers', []) if t]

            # Extract fiscal year from quarter reference for 10-K searches
            # 10-K filings are annual reports, so Q4 of a year corresponds to that year's fiscal year
            fiscal_year = None
            quarter_reference = question_analysis.get('quarter_reference')
            if quarter_reference:
                # Extract year from quarter reference (e.g., "2025_q4" -> 2025, "2024_all" -> 2024)
                import re
                year_match = re.search(r'(\d{4})', str(quarter_reference))
                if year_match:
                    fiscal_year = int(year_match.group(1))
                    rag_logger.info(f"📅 Extracted fiscal year {fiscal_year} from quarter reference: {quarter_reference}")
                # For "latest" or other non-year references, use None to get most recent

            if extracted_tickers:
                # Generate query embedding for vector search
                query_embedding = await self.search_engine.encode_query_async(question)

                # Search 10-K for each ticker using planning-driven parallel retrieval
                # - Smart planning generates sub-questions and search strategy
                # - Parallel execution for speed
                # - Dynamic replanning based on evaluation feedback
                for ticker in extracted_tickers[:3]:  # Limit to 3 tickers to avoid overload
                    try:
                        # 10-K search with planning + parallel retrieval:
                        # - Phase 0: Generate sub-questions and search plan
                        # - Phase 1: Execute ALL searches in parallel
                        # - Phase 2: Generate answer with ALL chunks
                        # - Phase 3: Evaluate and replan if needed
                        # - Max 5 iterations
                        async for event in self.sec_service.execute_smart_parallel_search_async(
                            query=question,
                            query_embedding=query_embedding,
                            ticker=ticker,
                            fiscal_year=fiscal_year,
                            max_iterations=5,  # Hard limit on iterations
                            confidence_threshold=0.9,
                            event_yielder=stream,  # Pass stream flag to enable event yielding
                            embedding_function=self.search_engine.embedding_model.encode  # Generate embeddings for sub-questions
                        ):
                            # Handle SEC agent events
                            event_type = event.get('type', '')
                            event_data = event.get('data', {})

                            if event_type == 'search_complete':
                                # Final result with chunks
                                chunks = event_data.get('chunks', [])
                                if chunks:
                                    ten_k_results.extend(chunks)
                                    rag_logger.info(f"✅ Found {len(chunks)} 10-K chunks for {ticker} via parallel search")

                            elif stream:
                                # Forward SEC agent events as user-friendly reasoning traces
                                if event_type == 'planning_start':
                                    yield {
                                        'type': 'reasoning',
                                        'message': f"Looking at {ticker}'s annual report...",
                                        'step': '10k_planning',
                                        'data': {'ticker': ticker, 'phase': 'planning'}
                                    }
                                elif event_type == 'planning_complete':
                                    sub_questions = event_data.get('sub_questions', [])
                                    if sub_questions:
                                        # Format sub-questions as a thinking list with dashes
                                        questions_text = "\n".join([f"- {q}" for q in sub_questions[:4]])
                                        yield {
                                            'type': 'reasoning',
                                            'message': f"To answer this, I need to find:\n{questions_text}",
                                            'step': '10k_planning',
                                            'data': {'sub_questions': sub_questions}
                                        }
                                elif event_type == 'retrieval_complete':
                                    new_chunks = event_data.get('new_chunks', 0)
                                    if new_chunks > 0:
                                        yield {
                                            'type': 'reasoning',
                                            'message': f"Found {new_chunks} relevant sections in the filing",
                                            'step': '10k_retrieval',
                                            'data': event_data
                                        }
                                elif event_type == 'evaluation_complete':
                                    quality = event_data.get('quality_score', 0)
                                    missing = event_data.get('missing_info', [])
                                    if quality >= 0.9:
                                        yield {
                                            'type': 'reasoning',
                                            'message': "I have enough information to answer this question",
                                            'step': '10k_evaluation',
                                            'data': event_data
                                        }
                                    elif missing:
                                        yield {
                                            'type': 'reasoning',
                                            'message': f"Still looking for: {missing[0] if missing else 'more details'}...",
                                            'step': '10k_evaluation',
                                            'data': event_data
                                        }
                    except Exception as e:
                        rag_logger.warning(f"⚠️ Failed to search 10-K for {ticker}: {e}")

                if ten_k_results:
                    rag_logger.info(f"✅ Total 10-K results: {len(ten_k_results)} chunks from {len(set(chunk.get('ticker', '') for chunk in ten_k_results))} companies")

                    # Log 10-K search to Logfire
                    if LOGFIRE_AVAILABLE and logfire:
                        logfire.info(
                            "rag.10k_search",
                            tickers=extracted_tickers[:3],
                            fiscal_year=fiscal_year,
                            chunks_found=len(ten_k_results),
                            companies_found=len(set(chunk.get('ticker', '') for chunk in ten_k_results))
                        )

                    if stream:
                        yield {
                            'type': '10k_search',
                            'message': f'Found {len(ten_k_results)} relevant passages from {len(set(chunk.get("ticker", "") for chunk in ten_k_results))} companies',
                            'step': '10k_search',
                            'data': {
                                'chunks_found': len(ten_k_results),
                                'tickers_processed': len(extracted_tickers),
                                'companies_found': len(set(chunk.get('ticker', '') for chunk in ten_k_results))
                            }
                        }
                else:
                    rag_logger.warning("⚠️ No 10-K results found")
                    if stream:
                        yield {
                            'type': '10k_search',
                            'message': 'Found 0 relevant passages from 0 companies',
                            'step': '10k_search',
                            'data': {
                                'chunks_found': 0,
                                'tickers_processed': len(extracted_tickers),
                                'companies_found': 0
                            }
                        }
            else:
                rag_logger.warning("⚠️ Question requires 10-K but no tickers extracted")
                if stream:
                    yield {
                        'type': '10k_search',
                        'message': 'Cannot search 10-K - no company ticker identified',
                        'step': '10k_search',
                        'data': {
                            'error': 'No company ticker found in question'
                        }
                    }

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 3: SEARCH EXECUTION (Vector + Keyword Search)
        # ═════════════════════════════════════════════════════════════════════

        # Decide search strategy based on data_source from question analyzer
        # This is the NEW approach - explicit routing via data_source field
        data_source = question_analysis.get('data_source', 'earnings_transcripts')

        # DEBUG: Log what RAG agent received
        rag_logger.info(f"🔍 DEBUG [RAG AGENT RECEIVED]: data_source={data_source}, needs_10k={question_analysis.get('needs_10k')}")

        # Adjust max_iterations for SEC/10-K queries (use sec_max_iterations config)
        # Only apply to explicit 10k queries, not hybrid (to avoid wasting tokens)
        if data_source == '10k' or question_analysis.get('needs_10k', False):
            sec_max_iterations = self.config.get("sec_max_iterations", 5)
            if max_iterations < sec_max_iterations:
                rag_logger.info(f"📈 SEC/10-K query detected - increasing max_iterations from {max_iterations} to {sec_max_iterations}")
                max_iterations = sec_max_iterations

        # Skip earnings transcript search ONLY if data source is exclusively '10k' or 'latest_news'
        # For 'hybrid', we want to search transcripts too
        # Also, even for '10k' queries, we might want to search transcripts if it's a general question
        skip_initial_transcript_search = data_source in ['10k', 'latest_news'] and data_source != 'hybrid'

        if skip_initial_transcript_search:
            rag_logger.info(f"📋 Data source '{data_source}' detected - skipping initial transcript search. Will use alternative data source.")
            search_results = (None, [], [], 0.0, False, False, [], None)
            individual_results = []
            all_chunks = []
            all_citations = []
            search_time = 0.0
            is_general_question = False
            is_multi_ticker = False
            tickers_to_process = [normalize_ticker(t) for t in question_analysis.get('extracted_tickers', []) if t]
            target_quarter = None
        else:
            if stream:
                yield {'type': 'progress', 'message': 'Searching documents...', 'step': 'search', 'data': {}}
            
            # Log detailed timing for search phase
            logger.info("=" * 80)
            logger.info("🔍 STARTING SEARCH PHASE - DETAILED TIMING")
            logger.info("=" * 80)
            search_phase_start = time.time()

            search_results = await self._execute_search(question, question_analysis, target_quarters)

            search_phase_end = time.time()
            logger.info(f"🔍 SEARCH PHASE COMPLETED in {search_phase_end - search_phase_start:.3f}s")
            logger.info("=" * 80)
            individual_results, all_chunks, all_citations, search_time, is_general_question, is_multi_ticker, tickers_to_process, target_quarter = search_results

            # Log transcript search to Logfire
            if LOGFIRE_AVAILABLE and logfire:
                logfire.info(
                    "rag.transcript_search",
                    chunks_found=len(all_chunks),
                    tickers=tickers_to_process,
                    target_quarters=target_quarters,
                    is_multi_ticker=is_multi_ticker,
                    search_time_ms=int(search_time * 1000)
                )
        
        if stream:
            # Only show transcript search message if we actually performed transcript search
            # (skip if data_source is '10k' or 'latest_news' only)
            if not skip_initial_transcript_search:
                # Build transcript details showing which companies and quarters were found
                transcripts = {}
                if all_chunks:
                    for chunk in all_chunks:
                        ticker = chunk.get('ticker', 'Unknown')
                        year = chunk.get('year', '')
                        quarter = chunk.get('quarter', '')
                        
                        if ticker not in transcripts:
                            transcripts[ticker] = set()
                        
                        if year and quarter:
                            transcripts[ticker].add(f"Q{quarter} {year}")
                
                # Convert sets to sorted lists
                transcripts_sorted = {
                    ticker: sorted(list(quarters), reverse=True)[:3]  # Show up to 3 most recent
                    for ticker, quarters in transcripts.items()
                }
                
                yield {
                    'type': 'search',
                    'message': f"Found {len(all_chunks)} relevant passages from {len(transcripts)} {'company' if len(transcripts) == 1 else 'companies'}",
                    'step': 'search',
                    'data': {
                        'chunks_found': len(all_chunks),
                        'tickers_processed': len(tickers_to_process) if tickers_to_process else 0,
                        'transcripts': transcripts_sorted
                    }
                }

        # ═════════════════════════════════════════════════════════════════════
        # STAGE 3.5: AUTO NEWS FALLBACK (when transcript search returns 0 chunks)
        # ═════════════════════════════════════════════════════════════════════
        # If transcript search returned 0 chunks and we have tickers but no news/10-K context yet,
        # automatically trigger a news search so the user gets useful information instead of
        # "information not available."  This prevents the broken flow where the LLM suggests
        # "Do you want me to search the news?" but the user's "yes" is treated as a new query.
        if (not all_chunks and not news_results and not ten_k_results
                and tickers_to_process and not skip_initial_transcript_search
                and self.tavily_service.is_available()):
            ticker_list = ', '.join(tickers_to_process)
            rag_logger.info(
                f"📰 AUTO-FALLBACK: Transcript search returned 0 chunks for {tickers_to_process}. "
                f"Automatically searching news as fallback..."
            )
            if stream:
                yield {'type': 'progress', 'message': f'No earnings transcripts found for {ticker_list} in our database — searching latest news instead...', 'step': 'news_fallback', 'data': {}}

            fallback_news_query = f"{question} {' '.join(tickers_to_process)}"
            news_results = self.tavily_service.search_news(fallback_news_query, max_results=5, include_answer="advanced")

            if news_results and news_results.get("results"):
                rag_logger.info(f"✅ Auto-fallback news search found {len(news_results['results'])} articles")
                if stream:
                    yield {
                        'type': 'news_search',
                        'message': f'Found {len(news_results["results"])} news articles as fallback',
                        'step': 'news_fallback',
                        'data': {'articles_found': len(news_results['results'])}
                    }
            else:
                rag_logger.warning("⚠️ Auto-fallback news search also returned no results")

        # Step 4: Prepare news context and citations (for both streaming and non-streaming)
        # Format news context if available and get news citations
        news_context_str = None
        news_citations = []
        if news_results and news_results.get("results"):
            news_context_str = self.tavily_service.format_news_context(news_results)
            news_citations = self.tavily_service.get_news_citations(news_results)
            rag_logger.info(f"📰 Initial news search returned {len(news_citations)} citations")

        # Format 10-K context and citations
        ten_k_context_str = None
        ten_k_citations = []
        if ten_k_results:
            ten_k_context_str = self.sec_service.format_10k_context(ten_k_results)
            ten_k_citations = self.sec_service.get_10k_citations(ten_k_results)
            rag_logger.info(f"📄 Initial 10-K search returned {len(ten_k_citations)} citations")

        # Add news and 10-K citations to all_citations
        combined_citations = all_citations.copy()
        rag_logger.info(f"📎 Starting with {len(all_citations)} transcript citations")

        if news_citations:
            # Format news citations for inclusion
            for news_citation in news_citations:
                citation_entry = {
                    "type": "news",
                    "marker": f"[N{news_citation['index']}]",
                    "title": news_citation["title"],
                    "url": news_citation["url"],
                    "published_date": news_citation.get("published_date", "")
                }
                combined_citations.append(citation_entry)
                rag_logger.info(f"🌐 Added initial news citation: marker={citation_entry['marker']}, title={citation_entry['title'][:60]}, url={citation_entry['url'][:50]}")
            rag_logger.info(f"📎 After adding initial news: {len(combined_citations)} total citations ({len(all_citations)} transcript + {len(news_citations)} news + {len(ten_k_citations)} 10-K)")
        else:
            rag_logger.info(f"📎 No initial news citations to add")

        if ten_k_citations:
            # Add 10-K citations
            for ten_k_citation in ten_k_citations:
                combined_citations.append(ten_k_citation)
                rag_logger.info(f"📄 Added 10-K citation: {ten_k_citation.get('ticker', 'N/A')} FY{ten_k_citation.get('fiscal_year', 'N/A')} - {ten_k_citation.get('section', 'SEC Filing')}")
            rag_logger.info(f"📎 After adding 10-K: {len(combined_citations)} total citations ({len(all_citations)} transcript + {len(news_citations)} news + {len(ten_k_citations)} 10-K)")
        else:
            rag_logger.info(f"📎 No 10-K citations to add")

        rag_logger.info(f"📎 Final combined citations: {len(combined_citations)} total")

        # Step 5: Run iterative improvement with token streaming
        if stream:
            yield {'type': 'progress', 'message': 'Generating response...', 'step': 'generation', 'data': {}}

            # Set up token streaming using async queue
            token_queue = asyncio.Queue()
            generation_complete = asyncio.Event()
            generation_result = None
            loop = asyncio.get_running_loop()

            def token_callback(content: str):
                """Callback for streaming tokens - puts them in queue"""
                loop.call_soon_threadsafe(token_queue.put_nowait, content)

            # Create event yielder for iteration events
            iteration_event_queue = asyncio.Queue()

            async def iteration_event_yielder(event):
                """Helper to queue iteration events for streaming"""
                await iteration_event_queue.put(event)

            # Run generation in background task
            async def run_generation():
                nonlocal generation_result
                try:
                    
                    generation_result = await self._run_iterative_improvement(
                        question, individual_results, all_chunks, combined_citations, is_general_question, is_multi_ticker,
                        tickers_to_process, max_iterations, show_details, comprehensive, token_callback,
                        target_quarter, target_quarters, question_analysis, event_yielder=iteration_event_yielder,
                        news_context=news_context_str,
                        ten_k_context=ten_k_context_str
                    )
                except Exception as e:
                    logger.error(f"Error in token streaming generation: {e}", exc_info=True)
                    generation_result = {'success': False, 'error': str(e)}
                finally:
                    generation_complete.set()
            
            # Start generation task
            gen_task = asyncio.create_task(run_generation())
            
            # Stream both tokens and iteration events as they arrive
            # Prioritize iteration events to ensure they're sent immediately in real-time
            try:
                while not generation_complete.is_set() or not token_queue.empty() or not iteration_event_queue.empty():
                    # Check for iteration events first (higher priority) - use immediate check
                    try:
                        iteration_event = await asyncio.wait_for(iteration_event_queue.get(), timeout=0.001)
                        rag_logger.info(f"🌊 AGENT: Yielding iteration event immediately: type={iteration_event.get('type')}")
                        yield iteration_event
                        # After yielding an iteration event, immediately check for more iteration events
                        # This ensures all queued iteration events are sent in real-time
                        continue
                    except asyncio.TimeoutError:
                        pass
                    
                    # Only check for tokens if there are no immediate iteration events
                    try:
                        token = await asyncio.wait_for(token_queue.get(), timeout=0.01)
                        yield {'type': 'token', 'content': token, 'step': 'generation', 'data': {}}
                    except asyncio.TimeoutError:
                        # Small sleep to prevent tight loop from consuming CPU
                        await asyncio.sleep(0.001)
                        continue
            except Exception as e:
                logger.error(f"Error streaming tokens and iteration events: {e}", exc_info=True)
            
            # Wait for generation to complete
            await gen_task
            improvement_results = generation_result
        else:
            # Non-streaming path
            improvement_results = await self._run_iterative_improvement(
                question, individual_results, all_chunks, combined_citations, is_general_question, is_multi_ticker,
                tickers_to_process, max_iterations, show_details, comprehensive, None,
                target_quarter, target_quarters, question_analysis, event_yielder=None, news_context=news_context_str,
                ten_k_context=ten_k_context_str
            )
        
        # Handle early return from improvement (no results found)
        if isinstance(improvement_results, dict) and not improvement_results.get('success'):
            improvement_results['timing'] = {
                'analysis': analysis_time,
                'search': search_time,
                'total': time.time() - start_time
            }
            yield {'type': 'error', 'message': improvement_results.get('error', 'No results found'), 'step': 'generation', 'data': improvement_results}
            return  # Stop generator
        
        (best_answer, best_confidence, best_citations, best_context_chunks, best_chunks,
         evaluation_context, follow_up_questions_asked, accumulated_chunks, generation_time) = improvement_results
        
        # Step 5: Finalize response
        total_time = time.time() - start_time
        
        # Log comprehensive timing summary
        logger.info("=" * 80)
        logger.info("⏱️  COMPLETE RAG PIPELINE TIMING BREAKDOWN")
        logger.info("=" * 80)
        logger.info(f"📝 Question: {question[:80]}...")
        logger.info(f"")
        logger.info(f"⏱️  ANALYSIS PHASE")
        logger.info(f"   Time: {analysis_time:.3f}s")
        logger.info(f"")
        logger.info(f"🔍 SEARCH PHASE")
        logger.info(f"   Time: {search_time:.3f}s")
        logger.info(f"")
        logger.info(f"🤖 GENERATION PHASE")
        logger.info(f"   Time: {generation_time:.3f}s")
        logger.info(f"   Max iterations: {max_iterations}")
        logger.info(f"   Actual iterations: {len(evaluation_context)}")
        logger.info(f"   Follow-up questions: {len(follow_up_questions_asked)}")
        logger.info(f"   Chunks found: {len(all_chunks)}")
        logger.info(f"")
        logger.info(f"⏱️  TOTAL TIME: {total_time:.3f}s")
        logger.info(f"")
        logger.info(f"📊 BREAKDOWN:")
        logger.info(f"   - Analysis: {analysis_time:.3f}s ({analysis_time/total_time*100:.1f}%)")
        logger.info(f"   - Search: {search_time:.3f}s ({search_time/total_time*100:.1f}%)")
        logger.info(f"   - Generation: {generation_time:.3f}s ({generation_time/total_time*100:.1f}%)")
        logger.info("=" * 80)

        # Log RAG flow completion to Logfire
        if LOGFIRE_AVAILABLE and logfire:
            logfire.info(
                "rag.flow.complete",
                total_time_ms=int(total_time * 1000),
                analysis_time_ms=int(analysis_time * 1000),
                search_time_ms=int(search_time * 1000),
                generation_time_ms=int(generation_time * 1000),
                max_iterations=max_iterations,
                actual_iterations=len(evaluation_context),
                chunks_found=len(all_chunks),
                confidence=best_confidence,
                tickers=tickers_to_process,
                data_source=question_analysis.get('data_source', 'earnings_transcripts'),
                answer_length=len(best_answer) if best_answer else 0
            )

        # Finalize response
        print(f"\n{'='*80}")
        print(f"🎯 RAG FLOW COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"⏱️  Total Time: {total_time:.3f}s")
        print(f"{'='*80}")

        # Ensure limit messages are included
        if question_analysis.get('limits_exceeded') and not best_answer.startswith("⚠️"):
            from .rag_utils import generate_user_friendly_limit_message
            limit_message = generate_user_friendly_limit_message(question_analysis['limits_exceeded'])
            if limit_message:
                best_answer = limit_message + best_answer

        # Deduplicate citations and chunks from all iterations
        unique_citations = []
        seen_citations = set()
        
        # Debug: Log what's in best_citations before deduplication
        news_in_best = [c for c in best_citations if isinstance(c, dict) and c.get('type') == 'news']
        tenk_in_best = [c for c in best_citations if isinstance(c, dict) and c.get('type') == '10-K']
        rag_logger.info(f"🔍 DEBUG: best_citations contains {len(news_in_best)} news citations, {len(tenk_in_best)} 10-K citations, {len(best_citations)} total")

        for citation in best_citations:
            # Handle both string citations and dict citations
            citation_key = citation
            if isinstance(citation, dict):
                # For news citations, use marker as unique key (more reliable than URL)
                if citation.get('type') == 'news':
                    citation_key = citation.get('marker', citation.get('url', str(citation)))
                    rag_logger.debug(f"🔍 Processing news citation: marker={citation.get('marker')}, url={citation.get('url')[:50] if citation.get('url') else 'None'}")
                # For 10-K citations, use marker as unique key
                elif citation.get('type') == '10-K':
                    citation_key = citation.get('marker', str(citation))
                else:
                    # For transcript citations, use existing logic
                    citation_key = citation.get('citation') or citation.get('id') or citation.get('chunk_index') or str(citation)

            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                unique_citations.append(citation)
                if isinstance(citation, dict) and citation.get('type') == 'news':
                    rag_logger.info(f"✅ Added news citation to unique_citations: {citation.get('marker')}")
            elif isinstance(citation, dict) and citation.get('type') == 'news':
                rag_logger.warning(f"⚠️ Duplicate news citation skipped: {citation.get('marker')}, url={citation.get('url')[:50] if citation.get('url') else 'None'}")

        # Also deduplicate chunks - use citation as the unique key
        unique_chunks = []
        seen_chunk_citations = set()

        for chunk in best_chunks:
            chunk_citation = chunk.get('citation')
            if isinstance(chunk_citation, dict):
                chunk_citation = chunk_citation.get('citation') or chunk_citation.get('id') or chunk_citation.get('chunk_index') or str(chunk_citation)

            if chunk_citation not in seen_chunk_citations:
                seen_chunk_citations.add(chunk_citation)
                unique_chunks.append(chunk)

        rag_logger.info(f"📎 Deduplicated citations: {len(best_citations)} -> {len(unique_citations)} unique citations from all iterations")
        rag_logger.info(f"📄 Deduplicated chunks: {len(best_chunks)} -> {len(unique_chunks)} unique chunks from all iterations")

        # Include ALL citations — the frontend organizes them by type for display.
        # Previously we filtered to only marker-referenced citations, but this caused citations
        # to be silently dropped whenever the LLM forgot to include [N1]/[10K1]/[1] markers.
        filtered_citations = list(unique_citations)

        news_citations_count = len([c for c in filtered_citations if isinstance(c, dict) and c.get('type') == 'news'])
        ten_k_citations_count = len([c for c in filtered_citations if isinstance(c, dict) and c.get('type') == '10-K'])
        transcript_citations_count = len(filtered_citations) - news_citations_count - ten_k_citations_count

        rag_logger.info(f"📎 Including all {len(filtered_citations)} citations: "
                        f"{transcript_citations_count} transcript, {news_citations_count} news, {ten_k_citations_count} 10-K")
        if news_citations_count > 0:
            rag_logger.info(f"🌐 News citations: {[c.get('title', 'No title')[:50] for c in filtered_citations if isinstance(c, dict) and c.get('type') == 'news']}")
        if ten_k_citations_count > 0:
            ten_k_summary = [f"{c.get('ticker')} FY{c.get('fiscal_year')} - {c.get('section', 'Unknown')[:40]}" for c in filtered_citations if isinstance(c, dict) and c.get('type') == '10-K']
            rag_logger.info(f"📄 10-K citations: {ten_k_summary}")

        response_data = {
            'answer': best_answer,
            'confidence': best_confidence,
            'citations': filtered_citations,  # Only include citations actually used
            'context_chunks': best_context_chunks,
            'iterations': evaluation_context,
            'total_iterations': len(evaluation_context),
            'follow_up_questions_asked': follow_up_questions_asked,
            'accumulated_chunks_count': len(accumulated_chunks)
        }

        if is_multi_ticker or is_general_question:
            response_data['individual_results'] = individual_results

        if question_analysis.get('limits_exceeded'):
            response_data['limits_exceeded'] = question_analysis['limits_exceeded']

        # Store conversation in memory
        if conversation_id:
            self.question_analyzer.add_to_conversation_memory(conversation_id, question, "user")
            answer_summary = best_answer[:100] + "..." if len(best_answer) > 100 else best_answer
            self.question_analyzer.add_to_conversation_memory(conversation_id, answer_summary, "assistant")

        # Log LLM generation results
        await self.analytics_logger.log_llm_generation_result(
            success=True,
            final_answer=best_answer,
            response_time_ms=int(generation_time * 1000),
            model_used="cerebras" if self.cerebras_available else "openai"
        )

        # Finish analytics tracking
        await self.analytics_logger.finish_pipeline(overall_success=True)

        final_result = {
            'success': True,
            'response': response_data,
            'chunks': unique_chunks,
            'analysis': question_analysis,
            'timing': {
                'analysis': analysis_time,
                'search': search_time,
                'generation': generation_time,
                'total': total_time
            }
        }

        # Always yield final result (for both streaming and non-streaming)
        yield {
            'type': 'result',
            'message': 'Response generated successfully',
            'step': 'complete',
            'data': final_result  # Return complete result
        }

    async def execute_rag_flow_async(self, question: str, **kwargs) -> Dict[str, Any]:
        """Non-streaming wrapper - returns final result only."""
        kwargs['stream'] = False
        final_result = None
        async for event in self.execute_rag_flow(question, **kwargs):
            if event.get('type') == 'result':
                final_result = event.get('data')
        return final_result
    


# Convenience function for easy import
def create_rag_agent(openai_api_key: Optional[str] = None) -> RAGAgent:
    """Create a new RAG Agent instance.
    
    Factory function for creating RAGAgent instances. Provides a convenient
    way to instantiate the agent without directly calling the class constructor.
    
    Args:
        openai_api_key (Optional[str]): OpenAI API key. If None, loads from environment.
    
    Returns:
        RAGAgent: A new RAGAgent instance ready for use.
    
    Example:
        >>> agent = create_rag_agent()
        >>> result = await agent.execute_rag_flow("What is Apple's revenue?")
    """
    return RAGAgent(openai_api_key)


# Backward compatibility: RAGSystem is now RAGAgent
RAGSystem = RAGAgent
create_rag_system = create_rag_agent
