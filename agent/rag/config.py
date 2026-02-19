#!/usr/bin/env python3
"""
Configuration management for the RAG system.

This module handles all configuration settings for the RAG system including
database connections, model settings, processing limits, and quarter information.
"""

import os
import logging
import psycopg2
from enum import Enum
from typing import List


class AnswerMode(str, Enum):
    """Determines response depth based on question complexity."""
    DIRECT = "direct"          # Simple factual lookup
    STANDARD = "standard"      # Moderate analysis
    DETAILED = "detailed"      # Full research report
    DEEP_SEARCH = "deep_search"  # Exhaustive search with 10 iterations


ANSWER_MODE_CONFIG = {
    AnswerMode.DIRECT:      {"max_iterations": 2, "max_tokens": 2000, "confidence_threshold": 0.7},
    AnswerMode.STANDARD:    {"max_iterations": 3, "max_tokens": 6000, "confidence_threshold": 0.8},
    AnswerMode.DETAILED:    {"max_iterations": 4, "max_tokens": 16000, "confidence_threshold": 0.9},
    AnswerMode.DEEP_SEARCH: {"max_iterations": 10, "max_tokens": 20000, "confidence_threshold": 0.95},
}

# Configure logging
logger = logging.getLogger(__name__)


class Config:
    """Configuration management for the RAG system."""
    
    def __init__(self):
        self.config = {
            # Core RAG settings
            "transcripts_folder": "earnings_transcripts_2025_q1",  # Default to latest quarter
            "embeddings_file": "embeddings.npz",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "similarity_threshold": 0,
            "chunks_per_quarter": 15,  # Number of chunks to retrieve per quarter
            
            # Quarter settings - will be populated dynamically from database
            "available_quarters": [],  # Will be fetched from database
            "default_quarter": "",  # Will be set dynamically from database
            "quarter_folders": {},  # Legacy - not used with database
            
            # Quarter information for LLMs
            "quarter_descriptions": {
                "q1": "Q1 (January-March)",
                "q2": "Q2 (April-June)", 
                "q3": "Q3 (July-September)",
                "q4": "Q4 (October-December)"
            },
            
            # OpenAI settings (response generation)
            "openai_model": "gpt-5-nano-2025-08-07",  # OpenAI for response generation
            "openai_max_tokens": int(os.getenv("RAG_OPENAI_MAX_TOKENS", "8000")),  # max_completion_tokens; lower = faster
            "openai_temperature": 1,
            
            # Groq settings (question analysis, evaluation, rephrasing)
            "groq_model": "openai/gpt-oss-20b",
            "groq_max_tokens": 16000,  # Increased for more detailed responses
            "groq_temperature": 0.3,
            
            # Cerebras settings (primary for response generation - fast inference)
            "cerebras_model": "qwen-3-235b-a22b-instruct-2507",
            "cerebras_max_tokens": 16000,  # Increased for more detailed responses
            "cerebras_temperature": 0.1,
            "use_cerebras": True,  # Enable Cerebras as primary LLM (legacy; prefer llm_provider)
            # LLM provider: "openai" | "cerebras" | "auto" (auto = Cerebras if key set, else OpenAI)
            "llm_provider": os.getenv("RAG_LLM_PROVIDER", "cerebras"),
            
            # Embedding settings
            "embedding_model": "all-MiniLM-L6-v2",
            
            # Database settings (Railway PostgreSQL with pgvector)
            "database_url": os.getenv("DATABASE_URL", ""),  # Main database URL
            "pgvector_url": os.getenv("PG_VECTOR", ""),  # Use PG_VECTOR for transcript chunks and vectors
            
            
            # Iterative RAG settings
            "max_iterations": int(os.getenv("RAG_MAX_ITERATIONS", "3")),
            "sec_max_iterations": int(os.getenv("SEC_MAX_ITERATIONS", "5")),  # More iterations for 10-K/SEC queries
            # Use Cerebras model for evaluation as well (no Groq/OpenAI OSS)
            "evaluation_model": os.getenv("RAG_EVALUATION_MODEL", "qwen-3-235b-a22b-instruct-2507"),
            "evaluation_temperature": 0.1,
            
            # Hybrid search settings
            "hybrid_search_enabled": True,
            "keyword_weight": 0.3,  # Weight for keyword search results
            "vector_weight": 0.7,   # Weight for vector search results
            "keyword_max_results": 10,  # Max results from keyword search
            
            # Processing limits for performance and resource management
            "max_tickers": 8,  # Maximum number of tickers to process in a single query (increased with parallel optimization)
            "max_quarters": 12,  # Maximum number of quarters to process in a single query (increased to allow 3 years of quarterly data)
            
            # Debug mode - enable to log EXPLAIN ANALYZE for query optimization
            "debug_mode": os.getenv("RAG_DEBUG_MODE", "false").lower() == "true",
        }
    
    def get(self, key: str, default=None):
        if key == 'available_quarters':
            result = self.config.get(key, default)
            logger.info(f"🔍 Config.get() - available_quarters: {result}")
            return result
        return self.config.get(key, default)
    
    def __getitem__(self, key):
        """Enable dictionary-style access to config values."""
        if key == 'available_quarters':
            logger.info(f"🔍 Config accessed - available_quarters: {self.config[key]}")
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Enable dictionary-style assignment to config values."""
        if key == 'available_quarters':
            logger.info(f"🔍 Config modified - available_quarters: {value}")
        self.config[key] = value
    
    def get_connection_string(self):
        """Get PostgreSQL connection string from DATABASE_URL."""
        return self.config['database_url']
    
    def get_pgvector_connection_string(self):
        """Get PostgreSQL connection string for pgvector database."""
        return self.config['pgvector_url']
    
    def _sort_quarters_chronologically(self, quarters: List[str]) -> List[str]:
        """
        Sort quarters in reverse chronological order (latest first).
        
        Quarters from the database are already sorted by year DESC, quarter DESC,
        so this method primarily validates and ensures proper ordering.
        
        Args:
            quarters: List of quarters in format ['2025_q2', '2025_q1', '2024_q4', ...]
            
        Returns:
            Sorted list with latest quarter first
        """
        if not quarters:
            return quarters
        
        # Parse and sort quarters by (year, quarter) in descending order
        def parse_quarter(q: str) -> tuple:
            try:
                year, quarter = q.split('_q')
                return (int(year), int(quarter))
            except:
                return (0, 0)
        
        sorted_quarters = sorted(quarters, key=parse_quarter, reverse=True)
        logger.info(f"🔄 Sorted quarters (latest first): {sorted_quarters}")
        return sorted_quarters
    
    def get_latest_quarter(self) -> str:
        """
        Get the latest quarter from available quarters.
        
        Returns:
            The latest quarter (first in the sorted list, which is already in DESC order)
        """
        available_quarters = self.config.get("available_quarters", [])
        if available_quarters:
            return available_quarters[0]
        return self.config.get("default_quarter", "")
    
    
    def fetch_available_quarters_from_db(self):
        """Fetch available quarters and transcript information from database."""
        logger.info(f"🚀 Starting fetch_available_quarters_from_db...")
        
        # If we've already fetched quarters for this process, reuse them to avoid
        # creating a new database connection on every request.
        existing_quarters = self.config.get("available_quarters")
        existing_details = self.config.get("quarter_details")
        if existing_quarters and existing_details:
            logger.info("ℹ️ available_quarters already loaded in config; reusing cached values")
            return existing_quarters, existing_details
        try:
            connection_string = self.get_pgvector_connection_string()
            logger.info(f"🔍 Attempting to connect to database...")
            logger.info(f"🔍 Connection string (masked): {connection_string[:50]}...")
            conn = psycopg2.connect(connection_string)
            cursor = conn.cursor()
            logger.info(f"✅ Database connection successful")
            
            # Get unique quarters and years from database, format as YYYY_qN
            query = """
            SELECT DISTINCT year, quarter, COUNT(*) as transcript_count,
                   COUNT(DISTINCT ticker) as company_count
            FROM transcript_chunks 
            GROUP BY year, quarter 
            ORDER BY year DESC, quarter DESC
            """
            
            logger.info(f"🔍 Executing query to fetch quarters...")
            cursor.execute(query)
            results = cursor.fetchall()
            logger.info(f"✅ Query executed successfully, found {len(results)} quarter records")
            
            available_quarters = []
            quarter_details = {}
            
            for year, quarter, transcript_count, company_count in results:
                quarter_id = f"{year}_q{quarter}"
                available_quarters.append(quarter_id)
                
                quarter_details[quarter_id] = {
                    "year": year,
                    "quarter": quarter,
                    "quarter_name": f"Q{quarter} {year}",
                    "transcript_count": transcript_count,
                    "company_count": company_count,
                    "description": f"Q{quarter} {year} ({self.config.get('quarter_descriptions', {}).get(f'q{quarter}', f'Quarter {quarter}')})"
                }
            
            logger.info(f"🔍 Processed quarters: {available_quarters}")
            logger.info(f"🔍 Quarter details: {quarter_details}")
            
            conn.close()
            
            # Sort quarters in reverse chronological order (latest first)
            # Database query already returns them sorted, but we ensure proper ordering
            sorted_quarters = self._sort_quarters_chronologically(available_quarters)
            
            # Update config with fetched data
            self.config["available_quarters"] = sorted_quarters
            self.config["quarter_details"] = quarter_details
            
            # Set default quarter to the first quarter (latest available)
            if sorted_quarters:
                self.config["default_quarter"] = sorted_quarters[0]
                logger.info(f"✅ Set default quarter to {sorted_quarters[0]} (latest available)")
            
            logger.info(f"✅ Fetched {len(sorted_quarters)} available quarters from database: {sorted_quarters}")
            logger.info(f"🔍 Config updated - available_quarters: {self.config['available_quarters']}")
            return sorted_quarters, quarter_details
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch quarters from database: {e}")
            # No fallback - let the system fail if database is unavailable
            self.config["available_quarters"] = []
            self.config["quarter_details"] = {}
            raise Exception(f"Database connection failed: {e}")
    
    def get_quarter_context_for_llm(self) -> str:
        """Generate comprehensive quarter context information for LLM prompts."""
        available_quarters = self.config.get("available_quarters", [])
        quarter_details = self.config.get("quarter_details", {})
        
        if not available_quarters:
            return "Limited quarterly data available in our database."
        
        context_parts = [
            "AVAILABLE FINANCIAL DATA:",
            f"Our database contains financial data for {len(available_quarters)} quarters (earnings transcripts and 10-K filings):",
        ]
        
        for quarter_id in available_quarters:
            details = quarter_details.get(quarter_id, {})
            quarter_name = details.get("quarter_name", quarter_id)
            description = details.get("description", quarter_id)
            company_count = details.get("company_count", "unknown")
            transcript_count = details.get("transcript_count", "unknown")
            
            context_parts.append(f"- {description}: {company_count} companies, {transcript_count} transcript sections")
        
        # Convert quarters to human-friendly format for display
        latest_quarter = self.get_latest_quarter()
        if latest_quarter and '_q' in latest_quarter:
            year, quarter = latest_quarter.split('_q')
            latest_quarter_human = f"{year} Q{quarter}"
        else:
            latest_quarter_human = latest_quarter if latest_quarter else 'Unknown'
        
        oldest_quarter_human = available_quarters[-1] if available_quarters else 'Unknown'
        if oldest_quarter_human != 'Unknown' and '_q' in oldest_quarter_human:
            year, quarter = oldest_quarter_human.split('_q')
            oldest_quarter_human = f"{year} Q{quarter}"
        
        context_parts.extend([
            "",
            "PERIOD FORMAT: Use database format for quarters (e.g., 2025_q1) or fiscal years (FY 2024)",
            "FISCAL QUARTERS: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec",
            f"LATEST QUARTER: {latest_quarter} (most recent quarterly data available)",
            f"OLDEST QUARTER: {available_quarters[-1] if available_quarters else 'Unknown'}",
            "",
            "DATA AVAILABILITY:",
            f"Our database contains financial data ({len(available_quarters)} quarters total).",
            "Data types: Earnings transcripts (quarterly), 10-K SEC filings (annual), and news sources.",
            "**IMPORTANT**: The latest available data may differ by company and data type.",
            "When users ask about 'latest' or 'most recent' data, use the most appropriate source for their question.",
            "Data includes:",
            "- Earnings transcripts: Quarterly calls (quarters listed above; coverage may start around 2023).",
            "- 10-K filings: Annual SEC filings with audited financials, risk factors, compensation data. **10-K filings are available from fiscal year 2019 onward** (e.g. FY2019, FY2020, FY2021, ...). When a user asks for a specific year's 10-K (e.g. '10-K from 2020'), that year is available if ingested.",
            "- News: Recent developments, announcements, and market updates"
        ])
        
        return "\n".join(context_parts)
