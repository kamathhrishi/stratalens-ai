import re
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import math
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# from currency_converter import CurrencyConverter
import tempfile
import requests
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase

# Fix for Pydantic 2.11+ compatibility with LangChain
try:
    from langchain_core.caches import BaseCache
except ImportError:
    # Define BaseCache if not available to fix model rebuild issues
    from typing import Any, Optional
    from abc import ABC, abstractmethod
    
    class BaseCache(ABC):
        """Abstract base cache class for LangChain compatibility."""
        
        @abstractmethod
        def lookup(self, prompt: str, llm_string: str) -> Optional[Any]:
            """Look up cached result."""
            pass
        
        @abstractmethod
        def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
            """Update cache with new result."""
            pass

# Import logging utilities
from app.utils.logging_utils import log_message, log_error, log_warning, log_info

# Import metadata and reference data
from .metadata import get_enhanced_table_metadata, AVAILABLE_SECTORS, AVAILABLE_INDUSTRIES

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Helper function to validate required environment variables
# Environment variable helper imported from config
from config import get_required_env_var

# Import PostgreSQL connection utilities
from db.db_connection_utils import get_postgres_connection
from db.data_transformation_utils import convert_postgres_to_camelcase


class SchemaValidationError(Exception):
    """Raised when schema validation fails"""
    pass


class SystemInitializationError(Exception):
    """Raised when the system fails to initialize properly"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class QueryProcessingError(Exception):
    """Raised when query processing fails at any stage"""
    def __init__(self, message: str, stage: str = "unknown", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class IntentParsingError(Exception):
    """Raised when intent parsing fails"""
    pass

class ReasoningEvent:
    """Represents a single step or event in the AI's reasoning process."""
    def __init__(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.event_type = event_type  # e.g., 'step_start', 'step_complete', 'step_error', 'info'
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()


class FinancialDataAnalyzer:
    """
    Enhanced financial data text-to-SQL system with intent parsing,
    efficient two-level caching, sector/industry context, and PostgreSQL integration.
    Now using PostgreSQL for improved data consistency and real-time insights.
    
    ENHANCED: Always includes key financial metrics (sector, market cap, industry, price, PE/PB ratios)
    by default and ensures user-friendly column names.
    """

    def __init__(self,
                 intent_parser_model: str = "gpt-4.1-2025-04-14",  # Back to OpenAI
                 table_selection_model: str = "llama-3.1-8b-instant",  # Changed to Groq model
                 sql_generation_model: str = "gpt-4.1-2025-04-14",  # Changed to OpenAI model
                 intent_parser_temp: float = 0.1,
                 table_selection_temp: float = 0.0,
                 sql_generation_temp: float = 1.0,
                 api_key: Optional[str] = None,
                 groq_api_key: Optional[str] = None,  # NEW: Added Groq API key parameter
                 default_page_size: int = 20,
                 llm_timeout: int = 120,
                 sql_timeout: int = 30,
                 total_timeout: int = 180,
                 max_parallel_workers: int = 2):

        self.api_key = api_key or get_required_env_var("OPENAI_API_KEY", "OpenAI API key for LLM operations")
        self.groq_api_key = groq_api_key or get_required_env_var("GROQ_API_KEY", "Groq API key for LLM operations")  # NEW: Store Groq API key
        self.default_page_size = default_page_size
        self.llm_timeout = llm_timeout
        self.sql_timeout = sql_timeout
        self.total_timeout = total_timeout
        self.max_parallel_workers = max_parallel_workers

        self.stored_results = {}
        self.query_pipeline_cache = {}
        
        # Thread pool for parallel agent execution
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_workers)

        self.agent_configs = {
            'intent_parser': {'model': 'gpt-5-nano-2025-08-07', 'temperature': 0, 'llm': None, 'provider': 'openai'},
            'table_selection': {'model': 'gpt-5-nano-2025-08-07', 'temperature': 0, 'llm': None, 'provider': 'openai'},
            'sql_generation': {'model': 'gpt-5-nano-2025-08-07', 'temperature': 0, 'llm': None, 'provider': 'openai'},
            'semantic_post_processing': {'model': 'gpt-5-nano-2025-08-07', 'temperature': 0, 'llm': None, 'provider': 'openai'}
        }

        self.available_industries = AVAILABLE_INDUSTRIES
        self.available_sectors = AVAILABLE_SECTORS

        self.postgres_connection = None
        self.table_schemas = {}
        self._initialize_system()

    def _initialize_system(self):
        log_message("Initializing Financial Data Text-to-SQL System with PostgreSQL...", is_milestone=True)
        try:
            self._initialize_agent_llms()
            self._initialize_postgres_connection()
            self._load_and_validate_schemas()
            log_message("✅ System initialization completed successfully", is_milestone=True)
        except Exception as e:
            error_msg = f"System initialization failed: {str(e)}"
            log_message(f"❌ {error_msg}", is_milestone=True)
            raise SystemInitializationError(error_msg, {"original_error": str(e)})

    def _initialize_agent_llms(self):
        # Print analyzer environment variables
        print("\n")
        print("ANALYZER ENV VARIABLES")
        print("=" * 40)
        
        analyzer_vars = [
            ('OPENAI_API_KEY', 'OpenAI API key for LLM operations'),
            ('GROQ_API_KEY', 'Groq API key for LLM operations')
        ]
        
        for i, (var_name, description) in enumerate(analyzer_vars, 1):
            value = os.getenv(var_name)
            if value:
                # Mask API keys
                masked_value = value[:8] + '*' * (len(value) - 12) + value[-4:] if len(value) > 12 else '*' * len(value)
                print(f"  {i}. {var_name}: {masked_value} ✓")
            else:
                print(f"  {i}. {var_name}: NOT SET ❌")
        
        print("=" * 40)
        print()
        
        # Check for required API keys
        openai_needed = any(config['provider'] == 'openai' for config in self.agent_configs.values())
        groq_needed = any(config['provider'] == 'groq' for config in self.agent_configs.values())
        
        if openai_needed and not self.api_key:
            raise SystemInitializationError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        if groq_needed and not self.groq_api_key:
            raise SystemInitializationError("Groq API key not found. Please set GROQ_API_KEY environment variable.")
            
        log_message("🤖 Initializing Agent-Specific Models:", is_milestone=False)
        
        # Fix Pydantic model rebuild issue before initializing ChatOpenAI instances
        try:
            ChatOpenAI.model_rebuild()
            log_message("✅ ChatOpenAI model rebuilt successfully")
        except Exception as e:
            log_message(f"⚠️ Warning: Could not rebuild ChatOpenAI model: {e}")
            # Continue anyway - the model might still work
        
        for agent_name, config in self.agent_configs.items():
            try:
                provider_info = f"via {config['provider'].upper()}"
                log_message(f"  {agent_name.replace('_', ' ').title()}: {config['model']} (temp: {config['temperature']}) {provider_info}")
                
                if config['provider'] == 'openai':
                    config['llm'] = ChatOpenAI(
                        model=config['model'],
                        temperature=config['temperature'],
                        openai_api_key=self.api_key,
                        max_completion_tokens=2000,
                        request_timeout=self.llm_timeout
                    )
                elif config['provider'] == 'groq':
                    # Handle Groq's temperature requirements (minimum 1e-8)
                    groq_temperature = max(config['temperature'], 1e-8)
                    config['llm'] = ChatOpenAI(
                        model=config['model'],
                        temperature=groq_temperature,
                        openai_api_key=self.groq_api_key,
                        openai_api_base="https://api.groq.com/openai/v1",  # Groq API base URL
                        max_completion_tokens=2000,
                        request_timeout=self.llm_timeout
                    )
            except Exception as e:
                raise SystemInitializationError(f"Failed to initialize {agent_name}: {e}")
        log_message("✅ All agent models initialized successfully", is_milestone=False)

    def _initialize_postgres_connection(self):
        try:
            log_message("🔍 Initializing PostgreSQL connection...")
            
            # Test the connection to ensure it works
            test_conn = get_postgres_connection()
            test_conn.close()
            log_message("✅ PostgreSQL connection test successful")
            
            log_message("✅ Connected to PostgreSQL database successfully")
            
        except Exception as e:
            log_message(f"❌ Failed to initialize PostgreSQL connection: {e}")
            log_message(f"🔍 Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise SystemInitializationError(f"Failed to initialize PostgreSQL connection: {e}")


    def _get_enhanced_table_metadata(self, table_name: str) -> Dict[str, Any]:
            """Enhanced table metadata with ACTUAL column names verified from data"""
            enhanced_metadata = get_enhanced_table_metadata()
              
            # Return the metadata for the requested table, or a default structure
            return enhanced_metadata.get(table_name, {
                'description': f'Data for table: {table_name}', 
                'primary_sort_column': None, 
                'sort_direction': 'DESC',
                'date_columns': [], 
                'key_columns': [], 
                'filter_hints': f'Check actual schema with DESCRIBE {table_name}', 
                'is_time_series': False,
                'time_column': None,
                'period_column': None,
                'special_notes': f'⚠️ Column structure not verified for {table_name}'
            })

    def sort_cached_dataframe(self, query_hash: str, column: str, direction: str = 'asc') -> Optional[pd.DataFrame]:
        """
        Sort cached dataframe by specified column and direction
        
        Args:
            query_hash: Hash of the original query to find cached data
            column: Column name to sort by
            direction: 'asc' or 'desc'
        
        Returns:
            Sorted DataFrame or None if not found
        """
        try:
            cached_pipeline = self._get_cached_pipeline_by_hash(query_hash)
            if cached_pipeline and 'raw_results_df' in cached_pipeline:
                df = cached_pipeline['raw_results_df'].copy()
                
                # Validate column exists
                if column not in df.columns:
                    log_message(f"Column '{column}' not found in dataframe. Available: {list(df.columns)}")
                    return None
                
                # Sort the dataframe
                ascending = direction == 'asc'
                
                # Handle different data types for sorting
                if df[column].dtype == 'object':
                    # For object/string columns, handle NaN/None values
                    df_sorted = df.sort_values(
                        by=column, 
                        ascending=ascending, 
                        na_position='last',
                        key=lambda x: x.astype(str).str.lower() if x.dtype == 'object' else x
                    )
                else:
                    # For numeric columns
                    df_sorted = df.sort_values(by=column, ascending=ascending, na_position='last')
                
                log_message(f"✅ Sorted dataframe by {column} ({direction}): {len(df_sorted)} rows")
                return df_sorted
            
            return None
            
        except Exception as e:
            log_message(f"❌ Error sorting cached dataframe: {e}")
            return None

    def _load_and_validate_schemas(self):
        log_message("Loading and validating table schemas with enhanced metadata...", is_milestone=True)
        try:
            # Get tables from PostgreSQL
            log_message("🔍 Querying PostgreSQL for available tables...")
            conn = get_postgres_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'financial_data'"
                    )
                    tables_result = cursor.fetchall()
                    actual_db_tables = [row['table_name'] for row in tables_result]
                    log_message(f"✅ Tables found in PostgreSQL: {actual_db_tables}")
            finally:
                conn.close()
            
            # Check if we have any tables
            if not actual_db_tables:
                log_message("⚠️ No tables found in PostgreSQL financial_data schema")
        except Exception as e:
            log_message(f"❌ Could not dynamically get table names from PostgreSQL: {e}")
            log_message("🔧 Using predefined list as fallback...")
            actual_db_tables = [
                "company_profiles", "income_statements", "balance_sheets", "cash_flow_statements",
                "key_metrics", "financial_ratios", "historical_market_cap",
                "income_statements_ttm", "balance_sheets_ttm", "cash_flow_statements_ttm", "financial_ratios_ttm", "key_metrics_ttm",
                "income_statements_growth_rates", "balance_sheets_growth_rates", "cash_flow_statements_growth_rates"
            ]

        log_message(f"🔍 Processing {len(actual_db_tables)} tables for schema loading...")
        for table_name in actual_db_tables:
            try:
                log_message(f"📋 Processing table: {table_name}")
                columns = self._get_table_columns(table_name)
                if columns:
                    log_message(f"📊 Getting metadata for {table_name}...")
                    metadata = self._get_enhanced_table_metadata(table_name)
                    self.table_schemas[table_name] = {
                        'columns': columns,
                        'description': metadata['description'],
                        'metadata': metadata
                    }
                    log_message(f"✅ Loaded schema for {table_name}: {len(columns)} columns with enhanced metadata")
                else:
                    log_message(f"⚠️ No columns found for table {table_name} during schema load.")
            except Exception as e:
                log_message(f"❌ Failed to load schema for {table_name}: {e}")
                log_message(f"🔍 Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()

        log_message(f"📊 Schema loading complete. Loaded {len(self.table_schemas)} tables: {list(self.table_schemas.keys())}")
        
        if not self.table_schemas:
            log_message("❌ CRITICAL: No table schemas loaded. Database might be empty or tables are unreadable.")
            raise SystemInitializationError("CRITICAL: No table schemas loaded. Database might be empty or tables are unreadable.")

        essential_tables = ["company_profiles", "income_statements"]
        missing_tables = [t for t in essential_tables if t not in self.table_schemas]
        if missing_tables:
            log_message(f"❌ CRITICAL: Essential tables missing from schema: {missing_tables}")
            log_message(f"📋 Available tables: {list(self.table_schemas.keys())}")
            raise SystemInitializationError(f"CRITICAL: Essential tables missing from schema: {missing_tables}")

        cp_columns = self.table_schemas.get("company_profiles", {}).get('columns', [])
        essential_cp_cols = ["symbol", "companyname", "sector", "industry"]
        missing_cp_cols = [col for col in essential_cp_cols if col not in cp_columns]
        if missing_cp_cols:
            log_message(f"❌ company_profiles table missing essential columns: {missing_cp_cols}")
            log_message(f"📋 Found columns: {cp_columns}")
            raise SystemInitializationError(f"company_profiles table missing essential columns: {missing_cp_cols}. Found: {cp_columns}")

        log_message(f"✅ Schema validation passed: {len(self.table_schemas)} tables loaded with metadata", is_milestone=True)

    def _get_table_columns(self, table_name: str) -> List[str]:
        try:
            # Use PostgreSQL's information_schema to get column information
            log_message(f"🔍 Getting columns for table: {table_name}")
            conn = get_postgres_connection()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_name = %s AND table_schema = 'financial_data' ORDER BY ordinal_position",
                        [table_name]
                    )
                    columns_result = cursor.fetchall()
                    columns = [row['column_name'] for row in columns_result]
                    
                log_message(f"✅ Found {len(columns)} columns for {table_name}: {columns[:5]}{'...' if len(columns) > 5 else ''}")
                return columns
            finally:
                conn.close()
        except Exception as e:
            log_message(f"❌ Error getting columns for {table_name}: {e}")
            log_message(f"🔍 Error type: {type(e).__name__}")
            return []
        
    def _get_cached_pipeline_by_hash(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached pipeline by hash directly"""
        # Add debugging
        log_message(f"🔍 Looking for cache with hash: {query_hash}")
        log_message(f"🗂️ Available cache hashes: {list(self.query_pipeline_cache.keys())}")
        
        result = self.query_pipeline_cache.get(query_hash)
        if result:
            log_message(f"✅ Found cached pipeline for hash {query_hash}")
        else:
            log_message(f"❌ No cached pipeline found for hash {query_hash}")
        
        return result

    def get_query_hash_from_question(self, question: str) -> str:
        """Get the hash for a given question to access cached data"""
        return self._get_question_hash(question)

    def paginate_cached_data(self, question: str, page: int = 1, page_size: Optional[int] = 20) -> Dict[str, Any]:
        """
        Paginate cached data without sorting - using exact matching
        
        Args:
            question: Original question to find cached data
            page: Page number for pagination
            page_size: Number of rows per page
        
        Returns:
            Dictionary with paginated data
        """
        try:
            # Try exact cache match first
            cached_pipeline = self._get_cached_pipeline(question)
            
            if not cached_pipeline:
                return {
                    'success': False,
                    'error': 'No cached data found for this query',
                    'data_rows': [],
                    'columns': [],
                    'friendly_columns': {},
                    'total_rows': 0
                }
            
            # Get the raw dataframe
            if 'raw_results_df' not in cached_pipeline:
                return {
                    'success': False,
                    'error': 'No raw results data found in cache',
                    'data_rows': [],
                    'columns': [],
                    'friendly_columns': {},
                    'total_rows': 0
                }
            
            df = cached_pipeline['raw_results_df'].copy()
            
            # Paginate the data (no sorting)
            should_paginate = self._detect_pagination_intent(question)
            paginated_result = self._paginate_results(df, page, page_size, should_paginate)
            page_df = paginated_result['page_data']
            
            # Get columns and friendly names
            columns = list(df.columns)
            
            # Format data for JSON
            data_rows = self._handle_dataframe_for_json(page_df, question, columns)
            
            # Initialize friendly_columns as empty dictionary (consistent with other functions)
            friendly_columns = {}
            
            # Get SQL query from cache
            sql_query = cached_pipeline.get('sql_query', '')
            tables_used = cached_pipeline.get('selected_tables', [])
            
            return {
                'success': True,
                'data_rows': data_rows,
                'columns': columns,
                'friendly_columns': friendly_columns,
                'pagination_info': paginated_result['pagination_info'],
                'total_rows': len(df),
                'sql_query': sql_query,
                'tables_used': tables_used,
                'message': f"Showing {len(data_rows)} of {len(df)} rows (Page {page})"
            }
            
        except Exception as e:
            log_message(f"❌ Error in paginate_cached_data: {e}")
            return {
                'success': False,
                'error': f"Pagination failed: {str(e)}",
                'data_rows': [],
                'columns': [],
                'friendly_columns': {},
                'total_rows': 0
            }

    def sort_and_paginate_cached_data(self, question: str, column: str, direction: str = 'asc', 
                                    page: int = 1, page_size: Optional[int] = 20) -> Dict[str, Any]:
        """
        Sort cached data and return paginated results - using exact matching
        
        Args:
            question: Original question to find cached data
            column: Column to sort by
            direction: Sort direction ('asc' or 'desc')
            page: Page number for pagination
            page_size: Number of rows per page
        
        Returns:
            Dictionary with sorted and paginated data
        """
        try:
            # Try exact cache match first
            cached_pipeline = self._get_cached_pipeline(question)
            
            if not cached_pipeline:
                return {
                    'success': False,
                    'error': 'No cached data found or sorting failed',
                    'data_rows': [],
                    'columns': [],
                    'friendly_columns': {},
                    'total_rows': 0
                }
            
            # Get the raw dataframe
            if 'raw_results_df' not in cached_pipeline:
                return {
                    'success': False,
                    'error': 'No raw results data found in cache',
                    'data_rows': [],
                    'columns': [],
                    'friendly_columns': {},
                    'total_rows': 0
                }
            
            df = cached_pipeline['raw_results_df'].copy()
            
            # Validate column exists
            if column not in df.columns:
                available_cols = list(df.columns)
                log_message(f"Column '{column}' not found in dataframe. Available: {available_cols}")
                return {
                    'success': False,
                    'error': f"Column '{column}' not found. Available columns: {', '.join(available_cols)}",
                    'data_rows': [],
                    'columns': available_cols,
                    'friendly_columns': {},
                    'total_rows': len(df)
                }
            
            # Sort the dataframe
            ascending = direction == 'asc'
            
            # Handle different data types for sorting
            if df[column].dtype == 'object':
                # For object/string columns, handle NaN/None values
                df_sorted = df.sort_values(
                    by=column, 
                    ascending=ascending, 
                    na_position='last',
                    key=lambda x: x.astype(str).str.lower() if x.dtype == 'object' else x
                )
            else:
                # For numeric columns
                df_sorted = df.sort_values(by=column, ascending=ascending, na_position='last')
            
            log_message(f"✅ Sorted dataframe by {column} ({direction}): {len(df_sorted)} rows")
            
            # Store the sorted dataframe in cache for future pagination
            cached_pipeline['raw_results_df'] = df_sorted.copy()
            cached_pipeline['sort_applied'] = {'column': column, 'direction': direction}
            log_message(f"💾 Stored sorted dataframe in cache for future pagination")
            
            # Paginate the sorted data
            should_paginate = self._detect_pagination_intent(question)
            paginated_result = self._paginate_results(df_sorted, page, page_size, should_paginate)
            page_df = paginated_result['page_data']
            
            # Get columns and friendly names
            columns = list(df_sorted.columns)
            
            # Format data for JSON
            data_rows = self._handle_dataframe_for_json(page_df, question, columns)
            
            # Initialize friendly_columns as empty dictionary (consistent with other functions)
            friendly_columns = {}
            
            return {
                'success': True,
                'data_rows': data_rows,
                'columns': columns,
                'friendly_columns': friendly_columns,
                'pagination_info': paginated_result['pagination_info'],
                'total_rows': len(df_sorted),
                'sort_applied': {'column': column, 'direction': direction},
                'message': f"Sorted by {column} ({direction}) - showing {len(data_rows)} of {len(df_sorted)} rows"
            }
            
        except Exception as e:
            log_message(f"❌ Error in sort_and_paginate_cached_data: {e}")
            return {
                'success': False,
                'error': f"Sorting failed: {str(e)}",
                'data_rows': [],
                'columns': [],
                'friendly_columns': {},
                'total_rows': 0
            }
    

    def get_agent_llm(self, agent_type: str):
        if agent_type not in self.agent_configs or self.agent_configs[agent_type]['llm'] is None:
            raise ValueError(f"LLM for agent type '{agent_type}' not initialized.")
        return self.agent_configs[agent_type]['llm']


    def _get_question_hash(self, question: str) -> str:
        # Use exact question for hashing (no normalization)
        return hashlib.md5(question.encode()).hexdigest()

    def _cache_query_results(self, question: str, selected_tables: List[str],
                             parsed_intent: Dict[str, Any], sql_query: str,
                             results_df: pd.DataFrame):
        question_hash = self._get_question_hash(question)
        cache_entry = {
            'question': question,
            'selected_tables': selected_tables,
            'parsed_intent': parsed_intent,
            'sql_query': sql_query,
            'raw_results_df': results_df.copy(),
            'timestamp': datetime.now()
        }
        self.query_pipeline_cache[question_hash] = cache_entry
        # Enhanced debugging
        log_message(f"🗄️ CACHING: Question: '{question}', Hash: {question_hash}")
        log_message(f"🗄️ CACHING: DataFrame shape: {results_df.shape}")
        log_message(f"🗄️ CACHING: Total cache entries: {len(self.query_pipeline_cache)}")
        log_message(f"🗄️ Cached complete pipeline (with raw results) for question: '{question[:50]}...'")

    def _get_cached_pipeline(self, question: str) -> Optional[Dict[str, Any]]:
        """Enhanced cache retrieval that handles both single and multi-sheet results"""
        question_hash = self._get_question_hash(question)
        log_message(f"🔍 Cache lookup - Question: '{question}', Hash: {question_hash}")
        log_message(f"🔍 Cache lookup - Available keys: {list(self.query_pipeline_cache.keys())}")
        if question_hash in self.query_pipeline_cache:
            cache_entry = self.query_pipeline_cache[question_hash]
            result_type = cache_entry.get('result_type', 'single_sheet')
            log_message(f"🎯 Pipeline cache HIT for question: '{question[:50]}...' (Type: {result_type})")
            return cache_entry
        else:
            log_message(f"❌ Pipeline cache MISS for question: '{question[:50]}...'")
            return None

    def _get_query_hash(self, sql_query: str) -> str:
        normalized_query = re.sub(r'\s+', ' ', sql_query.strip().lower())
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def _store_results(self, sql_query: str, results_df: pd.DataFrame):
        query_hash = self._get_query_hash(sql_query)
        self.stored_results[query_hash] = results_df.copy()
        log_message(f"💾 Stored {len(results_df)} raw results for future pagination (SQL Hash: {query_hash})")

    def _get_stored_results(self, sql_query: str) -> Optional[pd.DataFrame]:
        query_hash = self._get_query_hash(sql_query)
        if query_hash in self.stored_results:
            log_message(f"📋 Using stored raw results for pagination (SQL Hash: {query_hash})")
            return self.stored_results[query_hash].copy()
        return None
        

    def _check_timeout(self, start_time: float, operation_name: str = "operation"):
        """Check if total timeout has been exceeded"""
        if time.time() - start_time > self.total_timeout:
            raise TimeoutError(f"Total query processing exceeded {self.total_timeout} seconds during {operation_name}")

    def _get_default_intent(self) -> Dict[str, Any]:
        """Return default intent structure"""
        return {
            'query_rejected': False,
            'rejection_reason': None,
            'reasoning': 'Default fallback',
            'entities': [],
            'metrics': [],
            'time_periods': ['latest'],
            'exact_sector_filters': [],
            'exact_industry_filters': [],
            'has_specific_companies': False,
            'use_sector_industry_filters': True,
            'required_calculations': [],
            'output_format': 'list',
            'use_semantic_post_processing': False,
            'max_companies_to_output': 250,
            'max_companies_to_process': None
        }

    def _extract_balanced_json(self, json_str: str) -> Dict[str, Any]:
        """Extract balanced JSON from potentially malformed string"""
        # Find the first opening brace
        start = json_str.find('{')
        if start == -1:
            raise ValueError("No opening brace found")
        
        # Count braces to find matching closing brace
        brace_count = 0
        end = start
        for i, char in enumerate(json_str[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if brace_count != 0:
            raise ValueError("Unbalanced braces")
        
        balanced_json = json_str[start:end]
        
        # Apply common fixes to the balanced JSON
        balanced_json = balanced_json.replace(' True', ' true').replace(' False', ' false').replace(' None', ' null')
        balanced_json = re.sub(r',\s*}', '}', balanced_json)
        balanced_json = re.sub(r',\s*]', ']', balanced_json)
        
        return json.loads(balanced_json)

    def _robust_json_parse(self, json_str: str) -> Dict[str, Any]:
        """Robust JSON parsing with multiple fallback strategies"""
        json_str = json_str.strip()
        
        # Debug logging for JSON parsing issues
        log_message(f"DEBUG: Attempting to parse JSON string length: {len(json_str)}")
        log_message(f"DEBUG: First 100 chars: {json_str[:100]}")
        log_message(f"DEBUG: Last 100 chars: {json_str[-100:]}")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            log_message(f"JSON decode error at line {e.lineno}, column {e.colno}: {e.msg}")
            log_message(f"Problematic character near position {e.pos}: '{json_str[max(0, e.pos-10):e.pos+10]}'")
            
            # Debug: Show the exact content that's failing
            log_message(f"FULL JSON CONTENT CAUSING ERROR:\n{json_str}")
            
            # Strategy 1: Comprehensive Python-to-JSON value fixes
            try:
                # Fix Python-style boolean values (comprehensive patterns)
                fixed_json = json_str
                log_message(f"BEFORE Boolean fixes: {fixed_json[:200]}...")
                
                # Multiple approaches to fix booleans (order matters - more specific first)
                # Fix capitalized boolean values with word boundaries
                fixed_json = re.sub(r'\bTrue\b', 'true', fixed_json)
                fixed_json = re.sub(r'\bFalse\b', 'false', fixed_json)
                fixed_json = re.sub(r'\bNone\b', 'null', fixed_json)
                
                # Additional safety replacements for any remaining instances
                fixed_json = fixed_json.replace('True', 'true')
                fixed_json = fixed_json.replace('False', 'false')
                fixed_json = fixed_json.replace('None', 'null')
                
                log_message(f"AFTER Boolean fixes: {fixed_json[:200]}...")
                
                # Remove trailing commas before closing braces/brackets
                fixed_json = re.sub(r',\s*}', '}', fixed_json)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                
                result = json.loads(fixed_json)
                log_message("✅ Successfully parsed JSON after comprehensive Python-to-JSON fixes")
                return result
            except json.JSONDecodeError as e2:
                log_message(f"❌ Comprehensive fixes failed: {e2}")
            
            # Strategy 2: Fix common JSON formatting issues with spaces
            try:
                # Comprehensive Python-style boolean fixes
                fixed_json = json_str
                # Simple replacements first
                fixed_json = fixed_json.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                # Then specific context replacements
                fixed_json = fixed_json.replace(' True', ' true').replace(' False', ' false').replace(' None', ' null')
                fixed_json = fixed_json.replace('True ', 'true ').replace('False ', 'false ').replace('None ', 'null ')
                fixed_json = fixed_json.replace(':True', ':true').replace(':False', ':false').replace(':None', ':null')
                fixed_json = fixed_json.replace('True,', 'true,').replace('False,', 'false,').replace('None,', 'null,')
                
                # Fix unescaped newlines in strings
                fixed_json = re.sub(r'(?<!\\)\n', '\\n', fixed_json)
                # Remove trailing commas
                fixed_json = re.sub(r',\s*}', '}', fixed_json)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                result = json.loads(fixed_json)
                log_message("✅ Successfully parsed JSON after fixing common issues")
                return result
            except json.JSONDecodeError as e3:
                log_message(f"❌ Common fixes failed: {e3}")
            
            # Strategy 3: Extract balanced JSON content
            try:
                result = self._extract_balanced_json(json_str)
                log_message("✅ Successfully extracted balanced JSON")
                return result
            except Exception as e4:
                log_message(f"❌ Balanced JSON extraction failed: {e4}")
            
            # Strategy 4: Try to fix quotes and escape sequences
            try:
                # Fix unescaped quotes in strings
                fixed_json = re.sub(r'(?<!\\)"(?=.*":)', '\\"', json_str)
                # Fix Python-style values again with comprehensive patterns
                fixed_json = fixed_json.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                fixed_json = re.sub(r'\bTrue\b', 'true', fixed_json)
                fixed_json = re.sub(r'\bFalse\b', 'false', fixed_json)
                fixed_json = re.sub(r'\bNone\b', 'null', fixed_json)
                result = json.loads(fixed_json)
                log_message("✅ Successfully parsed JSON after fixing quotes")
                return result
            except Exception as e5:
                log_message(f"❌ Quote fixing failed: {e5}")
            
            # Strategy 5: Extract just the JSON object content and rebuild
            try:
                # Find the content between the first { and last }
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    content = json_str[start_idx:end_idx+1]
                    # Apply all fixes with comprehensive patterns
                    content = content.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                    content = re.sub(r'\bTrue\b', 'true', content)
                    content = re.sub(r'\bFalse\b', 'false', content)
                    content = re.sub(r'\bNone\b', 'null', content)
                    content = re.sub(r',\s*}', '}', content)
                    content = re.sub(r',\s*]', ']', content)
                    result = json.loads(content)
                    log_message("✅ Successfully parsed JSON after content extraction and cleanup")
                    return result
            except Exception as e6:
                log_message(f"❌ Content extraction failed: {e6}")
            
            # Strategy 6: Last resort - character-by-character fix
            try:
                # Clean the entire string more aggressively
                fixed_json = json_str
                # Remove any control characters
                fixed_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', fixed_json)
                # Fix all Python boolean patterns - simple replace first
                fixed_json = fixed_json.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                # Then regex patterns
                fixed_json = re.sub(r'\bTrue\b', 'true', fixed_json)
                fixed_json = re.sub(r'\bFalse\b', 'false', fixed_json)
                fixed_json = re.sub(r'\bNone\b', 'null', fixed_json)
                # Fix trailing commas
                fixed_json = re.sub(r',\s*([}\]])', r'\1', fixed_json)
                # Fix multiple spaces
                fixed_json = re.sub(r'\s+', ' ', fixed_json)
                result = json.loads(fixed_json)
                log_message("✅ Successfully parsed JSON after aggressive cleanup")
                return result
            except Exception as e7:
                log_message(f"❌ Aggressive cleanup failed: {e7}")
            
            # All strategies failed
            log_message(f"❌ All JSON parsing strategies failed. Original error: {e}")
            raise e
    


    def parse_query_intent(self, question: str):
        """Efficient intent parser with ticker symbol support and CAGR detection"""
        
        # Extract ticker symbols from $TICKER format
        ticker_matches = re.findall(r'\$([A-Z]{1,5})', question)
        extracted_tickers = list(set(ticker_matches))  # Remove duplicates
        
        # Also extract traditional ticker mentions
        traditional_tickers = re.findall(r'\b([A-Z]{2,5})\b', question)
        # Exclude common financial terms and abbreviations
        financial_terms = ['USD', 'CEO', 'IPO', 'SEC', 'NYSE', 'NASDAQ', 'PE', 'TTM', 'ROE', 'ROA', 'ROCE', 
                          'ROIC', 'EPS', 'EBITDA', 'EBIT', 'FCF', 'OCF', 'CAPEX', 'OPEX', 'R&D', 'SGA', 
                          'COGS', 'YOY', 'QOQ', 'LTM', 'NTM', 'P&L', 'B&S', 'CFS', 'GAAP', 'IFRS',
                          'RATIO', 'BELOW', 'ABOVE', 'ALL', 'AND', 'OR', 'WITH', 'THE', 'FOR', 'TO']
        traditional_tickers = [t for t in traditional_tickers if t not in financial_terms]
        
        all_tickers = list(set(extracted_tickers + traditional_tickers))
        
        # Detect growth-related queries for CAGR
        growth_keywords = ['growth', 'grow', 'growing', 'increase', 'trend', 'change over time', 
                          'rate', 'compound', 'cagr', 'annual growth', 'yearly growth', 
                          'over the years', 'historical performance', 'performance over',
                          'progression', 'trajectory', 'expand', 'expansion']
        has_growth_query = any(keyword in question.lower() for keyword in growth_keywords)
        
        # NEW: Detect time period specifications to determine CAGR eligibility and TTM usage
        time_period_keywords = {
            'annual': ['annual', 'yearly', 'year', 'years', 'fiscal year', 'calendar year'],
            'quarterly': ['quarter', 'quarterly', 'q1', 'q2', 'q3', 'q4'],
            'monthly': ['month', 'monthly'],
            'specific_years': ['2020', '2021', '2022', '2023', '2024', '2019', '2018', '2017', '2016', '2015'],
            'period_ranges': ['last 5 years', 'last 10 years', 'past 3 years', 'over 5 years', 'since 2020']
        }
        
        # Check for specific time period mentions
        has_specific_timeline = False
        mentioned_years = []
        for period_type, keywords in time_period_keywords.items():
            for keyword in keywords:
                if keyword.lower() in question.lower():
                    has_specific_timeline = True
                    if period_type == 'specific_years':
                        mentioned_years.append(keyword)
                    break
        
        # NEW: Determine CAGR eligibility based on time period
        # CAGR only for 2+ years, otherwise use annual growth
        uses_cagr = has_growth_query and (
            has_specific_timeline and (
                len(mentioned_years) >= 2 or 
                any('years' in keyword.lower() and any(str(i) in keyword.lower() for i in range(2, 11)) 
                    for keyword in time_period_keywords['period_ranges'] if keyword.lower() in question.lower())
            ) or
            not has_specific_timeline  # If no specific timeline mentioned, assume multi-year for CAGR
        )
        
        # NEW: Determine TTM usage - default to TTM unless specific timeline mentioned
        uses_ttm_by_default = not has_specific_timeline
        
        # NEW: Detect specific company count requests
        company_count_patterns = [
            r'top\s+(\d+)\s+companies?',
            r'(\d+)\s+companies?',
            r'show\s+me\s+(\d+)\s+companies?',
            r'find\s+(\d+)\s+companies?',
            r'list\s+(\d+)\s+companies?',
            r'(\d+)\s+best\s+companies?',
            r'(\d+)\s+top\s+companies?',
            r'(\d+)\s+leading\s+companies?',
            r'(\d+)\s+largest\s+companies?',
            r'(\d+)\s+companies?\s+with',
            r'companies?\s+with.*top\s+(\d+)',
            r'companies?\s+with.*(\d+)\s+best'
        ]
        
        requested_company_count = None
        for pattern in company_count_patterns:
            match = re.search(pattern, question.lower())
            if match:
                requested_company_count = int(match.group(1))
                break
        
        # Cap at 500 maximum companies
        if requested_company_count and requested_company_count > 500:
            requested_company_count = 500
        
        intent_prompt = """Analyze this financial query and choose the best strategy:

QUESTION: {question}
DETECTED TICKERS: {detected_tickers}

AVAILABLE SECTORS:
{available_sectors}

AVAILABLE INDUSTRIES:
{available_industries}

📊 AVAILABLE DATA SOURCES:
- Financial Statements: revenue, profit, expenses, assets, liabilities, cash flow (annual & TTM)
- Company Profiles: company name, sector, industry, business description, location, CEO name
- Stock Metrics: market cap, stock price, P/E ratio, P/B ratio, financial ratios
- Growth Analysis: CAGR calculations, year-over-year comparisons
- Investment Screening: filtering by ratios, metrics, financial health

✅ QUERIES WE CAN ANSWER:
- "Show me tech companies with low P/E ratios"
- "Apple's revenue growth over 5 years"
- "Companies with highest operating margins"
- "Cash flow analysis for Microsoft"
- "Companies with highest R&D spend"
- "Innovation metrics for tech companies"

❌ REJECTION CRITERIA:
Only reject queries that are completely unrelated to public equity markets, such as:
- Personal information about executives/celebrities (Warren Buffett's personal net worth)
- Non-financial topics (pets, family, hobbies, sports, entertainment)
- General knowledge questions, weather, politics
- Technical support or system questions

🚨 CRITICAL: If query is NOT about public equity markets or financial analysis, return:
{{"query_rejected": true, "rejection_reason": "This query is outside the scope of financial data analysis. We can help with company financial data, stock metrics, ratios, and investment analysis."}}

ANALYSIS APPROACH:
• Always use single-sheet analysis for all queries

TICKER FORMAT SUPPORT:
- $TICKER (e.g., $AAPL, $TSLA) - preferred format
- Plain TICKER (e.g., AAPL, TSLA) - also supported

SECTOR/INDUSTRY MAPPING (SMART FILTERING):
🚨 CRITICAL: Apply sector/industry filters when:
1. User EXPLICITLY mentions specific sectors or industries
2. User asks for business types that are clearly associated with specific sectors

SMART BUSINESS TYPE TO SECTOR MAPPING:
When user asks for business types, intelligently suggest relevant sectors to reduce post-processing workload:


- "cryptocurrency companies" → Suggest "Technology" sector, "Software" industries
- "fintech companies" → Suggest "Financial Services" sector, "Technology" sector
- "data center companies" → Suggest "Technology" sector, "Real Estate" sector (for REITs)
- "renewable energy companies" → Suggest "Energy" sector, "Utilities" sector

🚨 DO NOT make assumptions about which sectors are "relevant" for general metrics (R&D, revenue, etc.)
🚨 When user asks for "companies with highest X" without mentioning sectors or business types, analyze ALL companies.

If the question explicitly mentions sectors or industries, map them to exact names from the available lists above:
- Extract sector keywords and match to exact_sector_filters
- Extract industry keywords and match to exact_industry_filters
- Use fuzzy matching and smart guessing ONLY when sectors/industries are explicitly mentioned

MAPPING RULES (ONLY WHEN EXPLICITLY MENTIONED):
- "tech", "technology", "software" → "Technology" sector
- "health", "healthcare", "medical", "pharma", "biotech" → "Healthcare" sector
- "finance", "financial", "bank", "banking" → "Financial Services" sector
- "energy", "oil", "gas" → "Energy" sector


For industries, make best effort matches ONLY when explicitly mentioned:
- "software" → "Software - Application" or "Software - Infrastructure"
- "pharma", "pharmaceutical" → "Drug Manufacturers - General"
- "biotech", "biotechnology" → "Biotechnology"
- "semiconductor", "chips" → "Semiconductors"


🎯 EXAMPLES:
✅ "Show me tech companies with highest revenue" → Apply Technology sector filter
✅ "Healthcare companies with R&D growth" → Apply Healthcare sector filter  
✅ "biologics companies with >20% China exposure" → Apply Healthcare sector filter + semantic post-processing
✅ "AI companies with cloud revenue" → Apply Technology sector filter + semantic post-processing
✅ "fintech companies with payment processing" → Apply Financial Services + Technology sector filters + semantic post-processing
❌ "Companies with highest R&D growth" → NO sector filters (analyze all sectors)
❌ "1B+ market cap companies with revenue growth" → NO sector filters (analyze all sectors)

DESCRIPTION-BASED FILTERING (ONLY FOR SIMPLE EXACT MATCHES):
Only use description filtering for very specific, unambiguous terms:
- "data center companies" → Simple keyword filtering (exact match needed)
- "colocation companies" → Simple keyword filtering (exact match needed)

When using description filtering:
- Set "use_description_filtering": true
- Add relevant keywords to "description_keywords"
- Set "use_semantic_post_processing": false
- Set "exact_sector_filters" and "exact_industry_filters" to empty arrays

GROWTH RATE CALCULATIONS (UPDATED - CAGR FOR 2+ YEARS ONLY):
When user asks for growth, growth rate, trend analysis, or any multi-year performance analysis:

NEW RULES:
- CAGR (Compound Annual Growth Rate) ONLY for 2 or more years of data
- For single year or less than 2 years: Use simple annual growth rate
- If no specific timeline mentioned: Assume multi-year and use CAGR

CAGR ELIGIBILITY:
- "revenue growth over 5 years" → Use CAGR (2+ years)
- "profit growth 2022-2024" → Use CAGR (2+ years)  
- "growth over the years" → Use CAGR (assumed multi-year)
- "revenue growth 2024" → Use simple annual growth (single year)

GROWTH CALCULATION METHODS:
- CAGR Formula: ((End Value / Start Value) ^ (1/years)) - 1
- Simple Annual Growth: (Current - Previous) / Previous
- Include appropriate "growth_calculation_method" in required_calculations

TTM FINANCIAL METRICS (NEW DEFAULT):
NEW RULE: Default to TTM (Trailing Twelve Months) financial metrics unless user specifies a timeline.

TTM USAGE RULES:
- "revenue" (no timeline) → Use TTM revenue
- "current financials" → Use TTM financials
- "revenue 2024" → Use annual 2024 revenue (specific timeline)
- "profit last 5 years" → Use annual profit data (specific timeline)
- "TTM revenue" → Use TTM revenue (explicitly mentioned)

IMPORTANT: If the query has ANY indication of growth, trends, or multi-year analysis:
- Set "uses_cagr_default": true (only if 2+ years eligible)
- Include appropriate "growth_calculation_method" in required_calculations  
- Add growth-related terms to metrics list
- This applies to: {has_growth_query}

TTM DEFAULT BEHAVIOR:
- Set "uses_ttm_by_default": true (unless specific timeline mentioned)
- This applies to: {uses_ttm_by_default}

🚨 SEMANTIC POST-PROCESSING (PREFERRED FOR COMPLEX BUSINESS TYPES):
When the query asks for business types that require AI interpretation beyond simple keyword matching:

💡 OPTIMIZATION STRATEGY:
- Use sector/industry filters to reduce the candidate pool BEFORE semantic post-processing
- This dramatically reduces the workload on the LLM and improves accuracy
- Example: Instead of filtering 1000 companies for "biologics", filter 100 Healthcare sector companies first

🧠 INTELLIGENT BUSINESS TYPE DETECTION:
Think about whether the user is asking for companies that ARE a certain business type vs. companies that USE or SERVE that business type. Use semantic post-processing when the distinction requires contextual understanding.

📊 BROAD CATEGORY SUGGESTION (NEW):
When the query references ambiguous business types or categories, suggest broad related categories to help with initial filtering:

EXAMPLES OF BROAD CATEGORY SUGGESTIONS:
- "biologics companies" → broad_categories: ["Healthcare", "Biotechnology", "Pharmaceuticals", "Life Sciences"]
- "AI companies" → broad_categories: ["Technology", "Software", "Artificial Intelligence", "Machine Learning"]
- "cryptocurrency companies" → broad_categories: ["Technology", "Software", "Financial Services", "Digital Assets"]

🚨 CRITICAL: AVOID SEMANTIC POST-PROCESSING WHEN BROAD CATEGORIES ARE CLEARLY AVAILABLE!
When the query can be answered using simple sector/industry filters, prefer those over semantic post-processing:

PREFER SIMPLE FILTERING WHEN:
- User asks for "tech companies" → Use Technology sector filter (NO semantic post-processing)
- User asks for "healthcare companies" → Use Healthcare sector filter (NO semantic post-processing)
- User asks for "financial companies" → Use Financial Services sector filter (NO semantic post-processing)
- User asks for "software companies" → Use Software industry filter (NO semantic post-processing)
- User asks for "biotech companies" → Use Biotechnology industry filter (NO semantic post-processing)
- User asks for "pharmaceutical companies" → Use Drug Manufacturers industry filter (NO semantic post-processing)

ONLY USE SEMANTIC POST-PROCESSING WHEN:
- Query requires complex business type classification that can't be mapped to exact sectors/industries
- Query asks for ambiguous business types that need contextual understanding
- Query combines business types that require interpretation


EXAMPLES THAT REQUIRE SEMANTIC POST-PROCESSING:

BUSINESS TYPE CLASSIFICATION (use "description_semantic"):
- "AI companies" → Use semantic post-processing with type "description_semantic"
- "cryptocurrency companies" → Use semantic post-processing with type "description_semantic"
- "crypto companies" → Use semantic post-processing with type "description_semantic"
- "blockchain companies" → Use semantic post-processing with type "description_semantic"
- "cybersecurity companies" → Use semantic post-processing with type "description_semantic"


COMBINED QUERIES (use "description_semantic"):
- "biologics companies with specific characteristics" → Use semantic post-processing with type "description_semantic"
- "AI companies with specific business models" → Use semantic post-processing with type "description_semantic"

💡 SMART DETECTION RULES:
- If the query asks for companies that ARE a business type (not just use/serve it) → Use semantic post-processing with type "description_semantic"
- If the query requires distinguishing between primary business vs. secondary activities → Use semantic post-processing with type "description_semantic"


🚨 CRITICAL RULE: For ambiguous business categories that need contextual understanding, ALWAYS use semantic post-processing!

WHEN TO USE SEMANTIC POST-PROCESSING:
✅ Complex business descriptions that need AI interpretation
✅ Cryptocurrency/blockchain/crypto/digital asset related companies
✅ AI/machine learning/artificial intelligence companies  
✅ Ambiguous industry classifications that need semantic matching
✅ Questions asking for "companies that..." with descriptive criteria
❌ Simple exact filters (sector = "Technology")
❌ Specific company queries ($AAPL, $MSFT)
❌ Standard financial ratio queries

When using semantic post-processing:
- Set "use_semantic_post_processing": true
- Set "max_companies_to_output": 250 (default: 250)
- Set "max_companies_to_process": 350 (1.4 * max_companies_to_output)
- Set "use_description_filtering": false
- OPTIMIZATION: Set "exact_sector_filters" and "exact_industry_filters" to relevant sectors/industries to reduce post-processing workload
- Example: For "biologics companies", set "exact_sector_filters": ["Healthcare"] to filter down from 1000+ to ~100 companies first

OUTPUT SIZING (NEW):
When using semantic post-processing, specify:
- "max_companies_to_output": Expected final number of companies (default: 250)
- "max_companies_to_output": Expected final number of companies (default: 200)
- "max_companies_to_process": 1.4 * max_companies_to_output (for LLM filtering)

JSON Response:
{{
    "query_rejected": false,
    "rejection_reason": null,
    "reasoning": "Brief explanation of analysis approach",
    "entities": {entities_list},
    "metrics": ["revenue", "net_income"],
    "time_periods": ["latest"],
    "exact_sector_filters": ["Technology", "Healthcare"],
    "exact_industry_filters": ["Software - Application", "Drug Manufacturers - General"],
    "has_specific_companies": {has_companies},
    "use_sector_industry_filters": {use_filters},
    "use_description_filtering": false,
    "description_keywords": ["data center", "colocation"],
    "use_semantic_post_processing": false,
    "broad_categories": ["Healthcare", "Biotechnology", "Technology"],
    "filter_columns": ["industry", "sector", "description"],
    "max_companies_to_output": 200,
    "max_companies_to_process": 280,
    "required_calculations": [],
    "output_format": "list",
    "confidence_level": "high",
    "uses_cagr_default": {uses_cagr},
    "growth_calculation_method": "CAGR" or "annual_growth",
    "uses_ttm_by_default": {uses_ttm_by_default},
    "has_specific_timeline": {has_specific_timeline}
}}"""

        try:
            prompt = ChatPromptTemplate.from_template(intent_prompt)
            llm = self.get_agent_llm('intent_parser')
            
            # Format the prompt with extracted ticker information and available sectors/industries
            formatted_prompt = prompt.format(
                question=question,
                detected_tickers=all_tickers,
                available_sectors=', '.join(self.available_sectors),
                available_industries=', '.join(self.available_industries),
                entities_list=all_tickers if all_tickers else '[]',
                has_companies=len(all_tickers) > 0,
                use_filters=len(all_tickers) == 0,
                has_growth_query=has_growth_query,
                uses_cagr=uses_cagr,
                uses_ttm_by_default=uses_ttm_by_default,
                has_specific_timeline=has_specific_timeline,
                requested_company_count=requested_company_count
            )
            
            # Log prompt length for intent parser
            prompt_length = len(formatted_prompt)
            log_message(f"\nPROMPT LENGTH FOR INTENT PARSER:\n{prompt_length}\n")
            
            response = llm.invoke(formatted_prompt)
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            log_message(f"Efficient intent parsing RAW response: {response_text}")

            # Parse JSON from response
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                json_str = match.group(0)
                try:
                    parsed_intent = self._robust_json_parse(json_str)
                except Exception as parse_error:
                    log_message(f"❌ JSON parsing completely failed: {parse_error}")
                    return self._get_default_intent()
                
                # Merge detected tickers with LLM-extracted entities
                llm_entities = parsed_intent.get('entities', [])
                combined_entities = list(set(all_tickers + llm_entities))
                parsed_intent['entities'] = combined_entities
                
                # No post-processing needed - LLM should map to exact industry names directly
                
                # Check for query rejection FIRST
                if parsed_intent.get('query_rejected', False):
                    rejection_reason = parsed_intent.get('rejection_reason', 'Query is outside the scope of financial data analysis')
                    log_message(f"🚫 Query rejected by intent parser: {rejection_reason}")
                    # Return a special rejection intent
                    return {
                        'query_rejected': True,
                        'rejection_reason': rejection_reason,
                        'reasoning': rejection_reason,
                        'entities': [],
                        'metrics': [],
                        'time_periods': [],
                        'exact_sector_filters': [],
                        'exact_industry_filters': [],
                        'has_specific_companies': False,
                        'use_sector_industry_filters': False,
                        'use_description_filtering': False,
                        'description_keywords': [],
                        'required_calculations': [],
                        'output_format': 'list',
                        'use_semantic_post_processing': False,
                        'max_companies_to_output': 0,
                        'max_companies_to_process': None
                    }

                # Ensure required fields exist with proper defaults
                parsed_intent.setdefault('query_rejected', False)
                parsed_intent.setdefault('rejection_reason', None)
                parsed_intent.setdefault('reasoning', 'Default single sheet')
                parsed_intent.setdefault('metrics', [])
                parsed_intent.setdefault('time_periods', ['latest'])
                parsed_intent.setdefault('exact_sector_filters', [])
                parsed_intent.setdefault('exact_industry_filters', [])
                parsed_intent.setdefault('has_specific_companies', len(combined_entities) > 0)
                # Only use sector/industry filters if they were explicitly provided by the LLM
                parsed_intent.setdefault('use_sector_industry_filters', 
                    bool(parsed_intent.get('exact_sector_filters')) or bool(parsed_intent.get('exact_industry_filters')))
                parsed_intent.setdefault('use_description_filtering', False)
                parsed_intent.setdefault('description_keywords', [])
                parsed_intent.setdefault('required_calculations', [])
                parsed_intent.setdefault('output_format', 'list')
                
                # NEW: Add defaults for broad categories and filter columns
                parsed_intent.setdefault('broad_categories', [])
                parsed_intent.setdefault('filter_columns', [])
                
                # Enhanced CAGR detection and defaults
                parsed_intent.setdefault('uses_cagr_default', uses_cagr)
                parsed_intent.setdefault('growth_calculation_method', 'CAGR' if uses_cagr else 'annual_growth')
                
                # Add CAGR to required calculations if growth query detected
                if has_growth_query and 'growth_calculation_method' not in str(parsed_intent.get('required_calculations', [])):
                    current_calcs = parsed_intent.get('required_calculations', [])
                    if not isinstance(current_calcs, list):
                        current_calcs = []
                    current_calcs.append({'type': 'growth_calculation_method', 'method': 'CAGR' if uses_cagr else 'annual_growth'})
                    parsed_intent['required_calculations'] = current_calcs

                # Semantic post-processing defaults with dynamic company count
                parsed_intent.setdefault('use_semantic_post_processing', False)
                
                # Use detected company count or default to 500
                if requested_company_count:
                    parsed_intent.setdefault('max_companies_to_output', requested_company_count)
                    parsed_intent.setdefault('user_requested_limit', requested_company_count)  # Set user_requested_limit for non-semantic queries
                    log_message(f"🎯 User requested {requested_company_count} companies")
                else:
                    parsed_intent.setdefault('max_companies_to_output', 500)  # Default to 500
                    parsed_intent.setdefault('user_requested_limit', None)  # No specific limit requested
                    log_message(f"🎯 No specific count requested, defaulting to 500 companies")
                
                # Calculate max_companies_to_process if semantic post-processing is enabled
                if parsed_intent.get('use_semantic_post_processing', False):
                    max_output = parsed_intent.get('max_companies_to_output', 500)
                    parsed_intent.setdefault('max_companies_to_process', int(1.4 * max_output))
                else:
                    parsed_intent.setdefault('max_companies_to_process', None)
                
                
                log_message(f"🧠 Efficient Intent Parser: {parsed_intent['reasoning']}")
                log_message(f"   $TICKER symbols detected: {extracted_tickers}")
                log_message(f"   All entities: {combined_entities}")
                if has_growth_query:
                    log_message(f"   📈 Growth query detected - CAGR will be used as default calculation method")
                log_message(f"   Uses CAGR: {parsed_intent.get('uses_cagr_default', False)}")
                log_message(f"   Growth method: {parsed_intent.get('growth_calculation_method', 'None')}")
                log_message(f"   Use description filtering: {parsed_intent.get('use_description_filtering', False)}")
                log_message(f"   Description keywords: {parsed_intent.get('description_keywords', [])}")
                log_message(f"   Use intelligent filtering: {parsed_intent.get('use_semantic_post_processing', False)}")
                if parsed_intent.get('use_semantic_post_processing', False):
                    log_message(f"   Max companies to output: {parsed_intent.get('max_companies_to_output')}")
                    log_message(f"   Max companies to process: {parsed_intent.get('max_companies_to_process')}")
                
                # NEW: Log the new fields
                log_message(f"   Broad categories: {parsed_intent.get('broad_categories', [])}")
                log_message(f"   Filter columns: {parsed_intent.get('filter_columns', [])}")
                
                return parsed_intent
                
            else:
                log_message(f"Warning: Could not parse JSON from intent response: {response_text}")
                return self._get_default_intent()
                    
        except Exception as e:
            log_message(f"Enhanced intent parsing failed: {e}. Using default intent.")
            import traceback
            traceback.print_exc()
            return self._get_default_intent()


    def parse_query_intent_streaming(self, question: str):
        """Efficient streaming intent parser with ticker symbol support and CAGR detection"""
        
        # Extract ticker symbols from $TICKER format
        ticker_matches = re.findall(r'\$([A-Z]{1,5})', question)
        extracted_tickers = list(set(ticker_matches))  # Remove duplicates
        
        # Also extract traditional ticker mentions
        traditional_tickers = re.findall(r'\b([A-Z]{2,5})\b', question)
        # Exclude common financial terms and abbreviations
        financial_terms = ['USD', 'CEO', 'IPO', 'SEC', 'NYSE', 'NASDAQ', 'PE', 'TTM', 'ROE', 'ROA', 'ROCE', 
                          'ROIC', 'EPS', 'EBITDA', 'EBIT', 'FCF', 'OCF', 'CAPEX', 'OPEX', 'R&D', 'SGA', 
                          'COGS', 'YOY', 'QOQ', 'LTM', 'NTM', 'P&L', 'B&S', 'CFS', 'GAAP', 'IFRS',
                          'RATIO', 'BELOW', 'ABOVE', 'ALL', 'AND', 'OR', 'WITH', 'THE', 'FOR', 'TO']
        traditional_tickers = [t for t in traditional_tickers if t not in financial_terms]
        
        all_tickers = list(set(extracted_tickers + traditional_tickers))
        
        # Detect growth-related queries for CAGR
        growth_keywords = ['growth', 'grow', 'growing', 'increase', 'trend', 'change over time', 
                          'rate', 'compound', 'cagr', 'annual growth', 'yearly growth', 
                          'over the years', 'historical performance', 'performance over',
                          'progression', 'trajectory', 'expand', 'expansion']
        has_growth_query = any(keyword in question.lower() for keyword in growth_keywords)
        
        # NEW: Detect time period specifications to determine CAGR eligibility and TTM usage
        time_period_keywords = {
            'annual': ['annual', 'yearly', 'year', 'years', 'fiscal year', 'calendar year'],
            'quarterly': ['quarter', 'quarterly', 'q1', 'q2', 'q3', 'q4'],
            'monthly': ['month', 'monthly'],
            'specific_years': ['2020', '2021', '2022', '2023', '2024', '2019', '2018', '2017', '2016', '2015'],
            'period_ranges': ['last 5 years', 'last 10 years', 'past 3 years', 'over 5 years', 'since 2020']
        }
        
        # Check for specific time period mentions
        has_specific_timeline = False
        mentioned_years = []
        for period_type, keywords in time_period_keywords.items():
            for keyword in keywords:
                if keyword.lower() in question.lower():
                    has_specific_timeline = True
                    if period_type == 'specific_years':
                        mentioned_years.append(keyword)
                    break
        
        # NEW: Determine CAGR eligibility based on time period
        # CAGR only for 2+ years, otherwise use annual growth
        uses_cagr = has_growth_query and (
            has_specific_timeline and (
                len(mentioned_years) >= 2 or 
                any('years' in keyword.lower() and any(str(i) in keyword.lower() for i in range(2, 11)) 
                    for keyword in time_period_keywords['period_ranges'] if keyword.lower() in question.lower())
            ) or
            not has_specific_timeline  # If no specific timeline mentioned, assume multi-year for CAGR
        )
        
        # NEW: Determine TTM usage - default to TTM unless specific timeline mentioned
        uses_ttm_by_default = not has_specific_timeline
        
        # Detect company count requests
        company_count_patterns = [
            r'(\d+)\s*(?:companies|stocks|entities|firms|businesses)',
            r'(?:top|show me|find)\s*(\d+)\s*(?:companies|stocks|entities|firms|businesses)',
            r'(\d+)\s*(?:best|highest|top|leading|most|largest|significant|prominent|notable|major|key|primary|foremost|premier|foremost|eminent|distinguished|outstanding|exceptional|remarkable|superior|leading|preeminent|elite|premier|select|chosen|handpicked|carefully selected|meticulously chosen|thoroughly vetted|rigorously screened|stringently selected|exclusively chosen|carefully curated|meticulously curated|thoroughly curated|rigorously curated|stringently curated|exclusively curated|carefully selected|meticulously selected|thoroughly selected|rigorously selected|stringently selected|exclusively selected)',
            r'(?:top|show me|find)\s*(\d+)\s*(?:best|highest|top|leading|most|largest|significant|prominent|notable|major|key|primary|foremost|premier|foremost|eminent|distinguished|outstanding|exceptional|remarkable|superior|leading|preeminent|elite|premier|select|chosen|handpicked|carefully selected|meticulously chosen|thoroughly vetted|rigorously screened|stringently selected|exclusively chosen|carefully curated|meticulously curated|thoroughly curated|rigorously curated|stringently curated|exclusively curated|carefully selected|meticulously selected|thoroughly selected|rigorously selected|stringently selected|exclusively selected)'
        ]
        requested_company_count = None
        for pattern in company_count_patterns:
            match = re.search(pattern, question.lower())
            if match:
                requested_company_count = int(match.group(1))
                break
        
        # Yield initial analysis step
        if extracted_tickers:
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_start', f'Analyzing companies: {", ".join(["$" + t for t in extracted_tickers])}', {'step': 'ticker_detection'}).__dict__}
        
        # Yield CAGR detection if applicable
        if has_growth_query:
            if uses_cagr:
                yield {'type': 'reasoning', 'event': ReasoningEvent('info', '📈 Growth analysis detected - Will use CAGR (Compound Annual Growth Rate) for trend calculations', {'step': 'cagr_detection', 'uses_cagr': True}).__dict__}
            else:
                yield {'type': 'reasoning', 'event': ReasoningEvent('info', '📈 Growth analysis detected - Will use simple annual growth rate (less than 2 years of data)', {'step': 'cagr_detection', 'uses_cagr': False}).__dict__}
        
        # Yield TTM detection if applicable
        if uses_ttm_by_default:
            yield {'type': 'reasoning', 'event': ReasoningEvent('info', '📊 Using TTM (Trailing Twelve Months) financial metrics by default', {'step': 'ttm_detection', 'uses_ttm': True}).__dict__}
        
        yield {'type': 'reasoning', 'event': ReasoningEvent('step_start', 'Understanding your financial question...', {'step': 'intent_analysis'}).__dict__}
        
        intent_prompt = """Analyze this financial query and choose strategy:

QUESTION: {question}
DETECTED TICKERS: {detected_tickers}

AVAILABLE SECTORS:
{available_sectors}

AVAILABLE INDUSTRIES:
{available_industries}

📊 AVAILABLE DATA SOURCES:
- Financial Statements: revenue, profit, expenses, assets, liabilities, cash flow (annual & TTM)
- Company Profiles: company name, sector, industry, business description, location, CEO name
- Stock Metrics: market cap, stock price, P/E ratio, P/B ratio, financial ratios
- Growth Analysis: CAGR calculations, year-over-year comparisons
- Investment Screening: filtering by ratios, metrics, financial health
- Research & Development: R&D expenses from income statements

✅ QUERIES WE CAN ANSWER:
- "Show me tech companies with low P/E ratios"
- "Apple's revenue growth over 5 years"
- "Companies with highest operating margins"
- "Geographic revenue breakdown for NVIDIA"
- "Cash flow analysis for Microsoft"
- "Companies with highest R&D spend"
- "Innovation metrics for tech companies"

❌ REJECTION CRITERIA:
Only reject queries that are completely unrelated to public equity markets, such as:
- Personal information about executives/celebrities (Warren Buffett's personal net worth)
- Non-financial topics (pets, family, hobbies, sports, entertainment)
- General knowledge questions, weather, politics
- Technical support or system questions

🚨 CRITICAL: If query is NOT about public equity markets or financial analysis, return:
{{"query_rejected": true, "rejection_reason": "This query is outside the scope of financial data analysis. We can help with company financial data, stock metrics, ratios, and investment analysis."}}

ANALYSIS APPROACH:
• Always use single-sheet analysis for all queries

TICKER SUPPORT: $TICKER format (e.g., $AAPL, $TSLA) preferred

    SECTOR/INDUSTRY MAPPING (CRITICAL - ONLY WHEN EXPLICITLY MENTIONED):
    🚨 CRITICAL: Only apply sector/industry filters when the user EXPLICITLY mentions specific sectors or industries.
    🚨 DO NOT make assumptions about which sectors are "relevant" for certain metrics (R&D, revenue, etc.)
    🚨 When user asks for "companies with highest X" without mentioning sectors, analyze ALL companies.
    
    ONLY use exact names from the AVAILABLE SECTORS and AVAILABLE INDUSTRIES lists above.
    - Extract sector keywords → find exact match in available_sectors → exact_sector_filters  
    - Extract industry keywords → find exact match in available_industries → exact_industry_filters
    - NEVER invent or abbreviate industry names - ONLY use the exact strings from the lists above
    - Apply mapping ONLY when sectors/industries are explicitly mentioned in the query

    MANDATORY: For "exact_industry_filters", you MUST use the complete, exact industry names from the AVAILABLE INDUSTRIES list.

    🚨 SEMANTIC POST-PROCESSING WITH BROAD CATEGORY FILTERING (ENHANCED ACCURACY):
When the query asks for business types that require AI interpretation beyond simple keyword matching:

ENHANCED APPROACH:
1. FIRST: Identify broad categories that can be filtered in SQL (sectors, industries, keywords)
2. THEN: Apply semantic post-processing to the pre-filtered dataset for higher accuracy

🚨 CRITICAL: AVOID SEMANTIC POST-PROCESSING WHEN BROAD CATEGORIES ARE CLEARLY AVAILABLE!
When the query can be answered using simple sector/industry filters, prefer those over semantic post-processing:

PREFER SIMPLE FILTERING WHEN:
- User asks for "tech companies" → Use Technology sector filter (NO semantic post-processing)
- User asks for "healthcare companies" → Use Healthcare sector filter (NO semantic post-processing)
- User asks for "financial companies" → Use Financial Services sector filter (NO semantic post-processing)
- User asks for "software companies" → Use Software industry filter (NO semantic post-processing)
- User asks for "biotech companies" → Use Biotechnology industry filter (NO semantic post-processing)
- User asks for "pharmaceutical companies" → Use Drug Manufacturers industry filter (NO semantic post-processing)

ONLY USE SEMANTIC POST-PROCESSING WHEN:
- Query requires complex business type classification that can't be mapped to exact sectors/industries
- Query asks for ambiguous business types that need contextual understanding
- Query combines business types that require interpretation

EXAMPLES:
- "AI companies" → Broad categories: ["Technology", "Software"], then semantic post-processing
- "cryptocurrency companies" → Broad categories: ["Technology", "Software"], then semantic post-processing  
- "biotech companies" → Broad categories: ["Healthcare", "Biotechnology"], then semantic post-processing
- "fintech companies" → Broad categories: ["Financial Services", "Technology"], then semantic post-processing
- "renewable energy companies" → Broad categories: ["Energy", "Utilities"], then semantic post-processing

BROAD CATEGORY IDENTIFICATION RULES:
- "AI companies" → broad_categories: ["Technology"], filter_columns: ["sector", "industry", "description"]
- "biotech companies" → broad_categories: ["Healthcare"], filter_columns: ["sector", "industry", "description"]
- "fintech companies" → broad_categories: ["Financial Services", "Technology"], filter_columns: ["sector", "industry", "description"]
- "cryptocurrency companies" → broad_categories: ["Technology"], filter_columns: ["sector", "industry", "description"]
- "data center companies" → broad_categories: ["Technology", "Real Estate"], filter_columns: ["sector", "industry", "description"]

When using semantic post-processing with broad categories:
- Set "use_semantic_post_processing": true
- Set "broad_categories": [list of relevant sectors/industries]
- Set "filter_columns": ["sector", "industry", "description"]
- Set "max_companies_to_output": 200 (or user's preference)
- Set "max_companies_to_process": 280 (1.4 * max_companies_to_output)
- Set "use_description_filtering": false
- Set "exact_sector_filters" and "exact_industry_filters" to empty arrays

DESCRIPTION-BASED FILTERING (ONLY FOR SIMPLE EXACT MATCHES):
Only use description filtering for very specific, unambiguous terms:
- "data center companies" → Simple keyword filtering (exact match needed)
- "colocation companies" → Simple keyword filtering (exact match needed)

When using description filtering:
- Set "use_description_filtering": true
- Add relevant keywords to "description_keywords"
- Set "use_semantic_post_processing": false
- Set "exact_sector_filters" and "exact_industry_filters" to empty arrays

    MAPPING RULES (ONLY WHEN EXPLICITLY MENTIONED):
    - "tech", "technology", "software" → "Technology" sector
    - "health", "healthcare", "medical", "pharma", "biotech" → "Healthcare" sector
    - "finance", "financial", "bank", "banking" → "Financial Services" sector
    - "energy", "oil", "gas" → "Energy" sector
    - "real estate", "property", "reit" → "Real Estate" sector
    - "utility", "utilities", "electric" → "Utilities" sector
    - "consumer", "retail" → "Consumer Cyclical" or "Consumer Defensive"
    - "industrial", "manufacturing" → "Industrials" sector
    - "materials", "mining", "metals" → "Basic Materials" sector
    - "communication", "telecom", "media" → "Communication Services" sector

    For industries, make best effort matches ONLY when explicitly mentioned:
    - "software" → "Software - Application" or "Software - Infrastructure"
    - "pharma", "pharmaceutical" → "Drug Manufacturers - General"
    - "biotech", "biotechnology" → "Biotechnology"
    - "semiconductor", "chips" → "Semiconductors"
    - "auto", "automotive", "cars" → "Auto - Manufacturers"
    - "aerospace", "defense" → "Aerospace & Defense"
    - "banks", "banking" → "Banks - Diversified" or "Banks - Regional"
    - "insurance" → "Insurance - Diversified"
    - "oil", "petroleum" → "Oil & Gas Exploration & Production"
    - "gambling", "casino", "gaming", "casinos" → "Gambling, Resorts & Casinos"
    - "hotel", "hospitality", "lodging" → "Travel Lodging"
    - "restaurant", "food service" → "Restaurants"
    - "retail", "stores" → "Specialty Retail" or "Department Stores"
    - "real estate", "reit" → "REIT - Diversified" or specific REIT types
    - "utilities", "electric", "gas" → "Regulated Electric", "Regulated Gas", or "Diversified Utilities"

    🚨 CRITICAL: Only apply these mappings when sector/industry terms are explicitly mentioned in the query.
    🚨 DO NOT automatically apply sector filters based on query type (R&D, revenue, etc.).
    
    🎯 EXAMPLES:
    ✅ "Show me tech companies with highest revenue" → Apply Technology sector filter
    ✅ "Healthcare companies with R&D growth" → Apply Healthcare sector filter  
    ❌ "Companies with highest R&D growth" → NO sector filters (analyze all sectors)
    ❌ "1B+ market cap companies with revenue growth" → NO sector filters (analyze all sectors)

GROWTH RATE CALCULATIONS (UPDATED - CAGR FOR 2+ YEARS ONLY):
When user asks for growth, growth rate, or trend analysis:

NEW RULES:
- CAGR (Compound Annual Growth Rate) ONLY for 2+ years of data
- For single year or less than 2 years: Use simple annual growth rate
- If no specific timeline mentioned: Assume multi-year and use CAGR

CAGR ELIGIBILITY:
- "revenue growth over 5 years" → Use CAGR (2+ years)
- "profit growth 2022-2024" → Use CAGR (2+ years)  
- "growth over the years" → Use CAGR (assumed multi-year)
- "revenue growth 2024" → Use simple annual growth (single year)
- "profit growth last year" → Use simple annual growth (single year)

GROWTH CALCULATION METHODS:
- CAGR Formula: ((End Value / Start Value) ^ (1/years)) - 1
- Simple Annual Growth: (Current - Previous) / Previous
- Include appropriate "growth_calculation_method" in required_calculations

TTM FINANCIAL METRICS (NEW DEFAULT):
NEW RULE: Default to TTM (Trailing Twelve Months) financial metrics unless user specifies a timeline.

TTM USAGE RULES:
- "revenue" (no timeline) → Use TTM revenue
- "profit" (no timeline) → Use TTM profit  
- "latest revenue" → Use TTM revenue
- "current financials" → Use TTM financials
- "revenue 2024" → Use annual 2024 revenue (specific timeline)
- "profit last 5 years" → Use annual profit data (specific timeline)
- "TTM revenue" → Use TTM revenue (explicitly mentioned)

JSON:
{{
    "query_rejected": false,
    "rejection_reason": null,
    "reasoning": "Brief explanation of analysis approach",
    "entities": {entities_list},
    "metrics": ["revenue"],
    "time_periods": ["latest"],
    "exact_sector_filters": ["Technology", "Healthcare"],
    "exact_industry_filters": ["Software - Application"],
    "has_specific_companies": {has_companies},
    "use_sector_industry_filters": {use_filters},
    "use_description_filtering": false,
    "description_keywords": ["data center", "colocation"],
    "required_calculations": [],
    "output_format": "list",
    "confidence_level": "high",
    "uses_cagr_default": {uses_cagr},
    "growth_calculation_method": "CAGR" or "annual_growth",
    "uses_ttm_by_default": {uses_ttm_by_default},
    "has_specific_timeline": {has_specific_timeline}
}}"""

        try:
            # Get the appropriate client for streaming based on provider
            from openai import OpenAI
            
            intent_config = self.agent_configs['intent_parser']
            if intent_config['provider'] == 'groq':
                # Use Groq API with proper temperature handling
                groq_temperature = max(intent_config['temperature'], 1e-8)
                client = OpenAI(
                    api_key=self.groq_api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                model_temperature = groq_temperature
            else:
                # Use OpenAI API
                client = OpenAI(api_key=self.api_key)
                model_temperature = intent_config['temperature']
            
            prompt_text = intent_prompt.format(
                question=question,
                detected_tickers=all_tickers,
                available_sectors=', '.join(self.available_sectors),
                available_industries=', '.join(self.available_industries),
                entities_list=all_tickers if all_tickers else '[]',
                has_companies=len(all_tickers) > 0,
                use_filters=len(all_tickers) == 0,
                has_growth_query=has_growth_query,
                uses_cagr=uses_cagr,
                uses_ttm_by_default=uses_ttm_by_default,
                has_specific_timeline=has_specific_timeline,
                requested_company_count=requested_company_count
            )
            
            # Stream analysis process
            yield {'type': 'reasoning', 'event': ReasoningEvent('info', 'Processing your financial question...', {'step': 'intent_analysis'}).__dict__}
            
            # Start streaming response from the appropriate API
            stream = client.chat.completions.create(
                model=intent_config['model'],
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                temperature=model_temperature,
                max_completion_tokens=1000,  # Reduced for efficiency
                stream=True
            )
            
            # Stream the response token by token
            response_text = ""
            json_started = False
            json_content = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    response_text += chunk_content
                    
                    # Check if we're starting JSON content
                    if '{' in chunk_content and not json_started:
                        json_started = True
                        json_content = response_text[response_text.find('{'):]
                    elif json_started:
                        json_content += chunk_content

            log_message(f"Efficient intent parsing RAW response: {response_text}")

            # Parse JSON from response
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                json_str = match.group(0)
                try:
                    parsed_intent = self._robust_json_parse(json_str)
                except Exception as parse_error:
                    log_message(f"❌ JSON parsing completely failed: {parse_error}")
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', 'Unable to understand request format - using standard analysis approach', {'step': 'intent_analysis'}).__dict__}
                    yield {'type': 'intent_result', 'data': self._get_default_intent()}
                    return
                
                # Merge detected tickers with LLM-extracted entities
                llm_entities = parsed_intent.get('entities', [])
                combined_entities = list(set(all_tickers + llm_entities))
                parsed_intent['entities'] = combined_entities
                
                # No post-processing needed - LLM should map to exact industry names directly
                
                # Stream the decision
                reasoning = parsed_intent.get('reasoning', 'Standard analysis approach')
                confidence = parsed_intent.get('confidence_level', 'medium')
                
                # User-friendly analysis description
                yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', f"Analysis approach: Single analysis ({confidence} confidence)", {'step': 'intent_analysis'}).__dict__}
                yield {'type': 'reasoning', 'event': ReasoningEvent('info', f"📋 {reasoning}", {'step': 'intent_analysis'}).__dict__}
                
                # Check for query rejection FIRST
                if parsed_intent.get('query_rejected', False):
                    rejection_reason = parsed_intent.get('rejection_reason', 'Query is outside the scope of financial data analysis')
                    log_message(f"🚫 Query rejected by streaming intent parser: {rejection_reason}")
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', f'❌ Query rejected: {rejection_reason}', {'step': 'intent_analysis', 'rejected': True}).__dict__}
                    yield {'type': 'intent_result', 'data': {
                        'query_rejected': True,
                        'rejection_reason': rejection_reason,
                        'reasoning': rejection_reason,
                        'entities': [],
                        'metrics': [],
                        'time_periods': [],
                        'exact_sector_filters': [],
                        'exact_industry_filters': [],
                        'has_specific_companies': False,
                        'use_sector_industry_filters': False,
                        'use_description_filtering': False,
                        'description_keywords': [],
                        'required_calculations': [],
                        'output_format': 'list',
                        'use_semantic_post_processing': False,
                        'max_companies_to_output': 0,
                        'max_companies_to_process': None
                    }}
                    return

                # Ensure all required fields exist with defaults
                parsed_intent.setdefault('query_rejected', False)
                parsed_intent.setdefault('rejection_reason', None)
                parsed_intent.setdefault('reasoning', 'Standard analysis approach')
                parsed_intent.setdefault('metrics', [])
                parsed_intent.setdefault('time_periods', ['latest'])
                parsed_intent.setdefault('exact_sector_filters', [])
                parsed_intent.setdefault('exact_industry_filters', [])
                parsed_intent.setdefault('has_specific_companies', len(combined_entities) > 0)
                # Only use sector/industry filters if they were explicitly provided by the LLM
                parsed_intent.setdefault('use_sector_industry_filters', 
                    bool(parsed_intent.get('exact_sector_filters')) or bool(parsed_intent.get('exact_industry_filters')))
                parsed_intent.setdefault('required_calculations', [])
                parsed_intent.setdefault('output_format', 'list')
                
                # NEW: Add defaults for broad categories and filter columns
                parsed_intent.setdefault('broad_categories', [])
                parsed_intent.setdefault('filter_columns', [])
                
                # Enhanced CAGR detection and defaults
                parsed_intent.setdefault('uses_cagr_default', has_growth_query)
                parsed_intent.setdefault('growth_calculation_method', 'CAGR' if has_growth_query else 'annual_growth')
                
                # Add CAGR to required calculations if growth query detected
                if has_growth_query and 'growth_calculation_method' not in str(parsed_intent.get('required_calculations', [])):
                    current_calcs = parsed_intent.get('required_calculations', [])
                    if not isinstance(current_calcs, list):
                        current_calcs = []
                    current_calcs.append({'type': 'growth_calculation_method', 'method': 'CAGR' if has_growth_query else 'annual_growth'})
                    parsed_intent['required_calculations'] = current_calcs

                # Semantic post-processing defaults with dynamic company count
                parsed_intent.setdefault('use_semantic_post_processing', False)
                
                # Use detected company count or default to 500
                if requested_company_count:
                    parsed_intent.setdefault('max_companies_to_output', requested_company_count)
                    parsed_intent.setdefault('user_requested_limit', requested_company_count)  # Set user_requested_limit for non-semantic queries
                    log_message(f"🎯 User requested {requested_company_count} companies")
                else:
                    parsed_intent.setdefault('max_companies_to_output', 500)  # Default to 500
                    parsed_intent.setdefault('user_requested_limit', None)  # No specific limit requested
                    log_message(f"🎯 No specific count requested, defaulting to 500 companies")
                
                # Calculate max_companies_to_process if semantic post-processing is enabled
                if parsed_intent.get('use_semantic_post_processing', False):
                    max_output = parsed_intent.get('max_companies_to_output', 500)
                    parsed_intent.setdefault('max_companies_to_process', int(1.4 * max_output))
                else:
                    parsed_intent.setdefault('max_companies_to_process', None)
                
                # Log the final parsed intent for debugging
                log_message(f"📋 Final parsed intent:")
                log_message(f"   Strategy: Single analysis")
                log_message(f"   Reasoning: {parsed_intent.get('reasoning', 'No reasoning provided')}")
                log_message(f"   Entities: {parsed_intent.get('entities', [])}")
                log_message(f"   Metrics: {parsed_intent.get('metrics', [])}")
                log_message(f"   Sectors: {parsed_intent.get('exact_sector_filters', [])}")
                log_message(f"   Industries: {parsed_intent.get('exact_industry_filters', [])}")
                log_message(f"   Uses CAGR: {parsed_intent.get('uses_cagr_default', False)}")
                log_message(f"   Use description filtering: {parsed_intent.get('use_description_filtering', False)}")
                log_message(f"   Description keywords: {parsed_intent.get('description_keywords', [])}")
                log_message(f"   Use intelligent filtering: {parsed_intent.get('use_semantic_post_processing', False)}")
                if parsed_intent.get('use_semantic_post_processing', False):
                    log_message(f"   Max companies to output: {parsed_intent.get('max_companies_to_output')}")
                    log_message(f"   Max companies to process: {parsed_intent.get('max_companies_to_process')}")
                
                # NEW: Log the new fields
                log_message(f"   Broad categories: {parsed_intent.get('broad_categories', [])}")
                log_message(f"   Filter columns: {parsed_intent.get('filter_columns', [])}")
                
                yield {'type': 'intent_result', 'data': parsed_intent}
                
            else:
                log_message(f"Warning: Could not parse JSON from intent response: {response_text}")
                yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', 'Invalid response format - using standard analysis approach', {'step': 'intent_analysis'}).__dict__}
                yield {'type': 'intent_result', 'data': self._get_default_intent()}
                    
        except Exception as e:
            log_message(f"❌ Streaming intent parsing failed: {e}")
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', f'Error analyzing request: {e}', {'step': 'intent_analysis'}).__dict__}
            yield {'type': 'intent_result', 'data': self._get_default_intent()}






    def _query_with_streaming_core(self, question: str, page: int = 1, page_size: Optional[int] = None, override_intent: Optional[Dict[str, Any]] = None):
        """
        Core single-sheet streaming logic with fixed status updates
        """
        log_message(f"\n🤔 New Streaming Question: {question} (Page: {page})", is_milestone=True)
        # For complete dataset requests (page_size=None), don't paginate
        if page_size is None:
            effective_page_size = None
        else:
            effective_page_size = page_size or self.default_page_size
        start_time = time.time()
        timing_details = {}

        try:
            # Check for cached pipeline first (only if no override intent)
            if not override_intent:
                cached_pipeline_entry = self._get_cached_pipeline(question)
                if cached_pipeline_entry and cached_pipeline_entry.get('result_type', 'single_sheet') == 'single_sheet':
                    yield {'type': 'reasoning', 'event': ReasoningEvent('info', 'Found previously analyzed results for this financial question.', {'step': 'cache_check'}).__dict__}
                    log_message("⚡ Using cached pipeline - no LLM calls needed!")
                    selected_tables = cached_pipeline_entry['selected_tables']
                    sql_query = cached_pipeline_entry['sql_query']
                    full_raw_df = cached_pipeline_entry['raw_results_df']
                    
                    should_paginate = self._detect_pagination_intent(question)
                    # For complete dataset requests (page_size=None), don't paginate
                    if page_size is None:
                        should_paginate = False
                    paginated_raw_result = self._paginate_results(full_raw_df, page, effective_page_size, should_paginate)
                    df_page_raw_data = paginated_raw_result['page_data']
                    pagination_info = paginated_raw_result['pagination_info']

                    generated_sql_columns = list(df_page_raw_data.columns) if not df_page_raw_data.empty else list(full_raw_df.columns)
                    
                    data_rows_formatted_for_json = self._handle_dataframe_for_json(df_page_raw_data, question, generated_sql_columns)
                    
                    total_time = time.time() - start_time
                    timing_details['total_duration_sec'] = total_time
                    timing_details['cache_type_hit'] = "pipeline_cache"

                    result_dict = {
                        "sql_query_generated": sql_query, "columns": generated_sql_columns,
                        "friendly_columns": {}, "data_rows": data_rows_formatted_for_json,
                        "message": self._generate_result_message(data_rows_formatted_for_json, pagination_info),
                        "error": None, "tables_used": selected_tables, "pagination_info": pagination_info,
                        "is_paginated": should_paginate, "used_stored_results": True,
                        "cache_type_hit": "pipeline_cache", "timing_details": timing_details
                    }
                    log_message(f"🏁 Query processing finished using pipeline cache. Success: True. Page: {page}")
                    self._log_timing_details(timing_details)
                    
                    # Final status update for cached results
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', f'✅ Analysis complete: {len(data_rows_formatted_for_json)} results found', {'step': 'final_result'}).__dict__}
                    yield {'type': 'result', 'data': result_dict}
                    return

            # Step 1: Parse intent (or use override) - NOW WITH STREAMING
            if override_intent:
                parsed_intent = override_intent
                yield {'type': 'reasoning', 'event': ReasoningEvent('info', 'Using provided analysis approach (multi-company mode)', {'step': 'intent_parsing'}).__dict__}
                timing_details['intent_parsing_sec'] = 0  # No time spent parsing
            else:
                self._check_timeout(start_time, "intent parsing")
                intent_start = time.time()
                
                # NEW: Use streaming intent parser
                parsed_intent = None
                for intent_event in self.parse_query_intent_streaming(question):
                    if intent_event['type'] == 'reasoning':
                        yield intent_event  # Forward reasoning events
                    elif intent_event['type'] == 'intent_result':
                        parsed_intent = intent_event['data']
                        break
                
                if not parsed_intent:
                    log_message("⚠️ Intent parsing failed, using default intent")
                    parsed_intent = self._get_default_intent()
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_fallback', 'Using standard analysis approach due to parsing issue', {'step': 'intent_parsing', 'fallback': True}).__dict__}
                elif parsed_intent.get('query_rejected', False):
                    # Handle rejected queries
                    rejection_reason = parsed_intent.get('rejection_reason', 'Query is outside the scope of financial data analysis')
                    log_message(f"🚫 Query rejected: {rejection_reason}")
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', f'❌ Query rejected: {rejection_reason}', {'step': 'query_rejection'}).__dict__}
                    yield {'type': 'error', 'message': f'Sorry, this query is outside the scope of financial data analysis: {rejection_reason}'}
                    yield {'type': 'result', 'data': {
                        "error": f"Query rejected: {rejection_reason}",
                        "message": f"Sorry, this query is outside the scope of financial data analysis. I can help with company financial data, stock metrics, ratios, and investment analysis.",
                        "query_rejected": True,
                        "rejection_reason": rejection_reason,
                        "columns": None,
                        "friendly_columns": None,
                        "data_rows": None,
                        "tables_used": [],
                        "pagination_info": None,
                        "is_paginated": False
                    }}
                    return
                
                timing_details['intent_parsing_sec'] = time.time() - intent_start
            
            # Step 2: Select tables
            self._check_timeout(start_time, "table selection")
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_start', 'Identifying relevant financial data sources...', {'step': 'table_selection'}).__dict__}
            tables_start = time.time()
            selected_tables = self.select_relevant_tables(question, parsed_intent)
            timing_details['table_selection_sec'] = time.time() - tables_start
            
            # Step 2.5: Check data availability and reject if data not available
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_start', 'Verifying data availability...', {'step': 'data_availability_check'}).__dict__}
            data_rejection = self._check_data_availability_and_reject(question, selected_tables)
            if data_rejection:
                rejection_reason = data_rejection.get('rejection_reason', 'Data not available')
                log_message(f"🚫 Query rejected by data availability check: {rejection_reason}")
                yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', f'❌ Data not available: {data_rejection.get("missing_data_type", "unknown")}', {'step': 'data_availability_check'}).__dict__}
                yield {'type': 'error', 'message': f'Sorry, {rejection_reason}'}
                yield {'type': 'result', 'data': {
                    "error": f"Data not available: {rejection_reason}",
                    "message": f"Sorry, {rejection_reason}",
                    "query_rejected": True,
                    "rejection_reason": rejection_reason,
                    "rejection_type": data_rejection.get('rejection_type', 'data_unavailable'),
                    "columns": None,
                    "friendly_columns": None,
                    "data_rows": None,
                    "tables_used": [],
                    "pagination_info": None,
                    "is_paginated": False
                }}
                return
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', 'Data availability verified', {'step': 'data_availability_check'}).__dict__}
            
            # User-friendly table descriptions
            table_descriptions = {
                'company_profiles': 'company information',
                'income_statements': 'income statements',
                'balance_sheets': 'balance sheets', 
                'cash_flow_statements': 'cash flow statements',
                'key_metrics': 'key financial metrics (P/E, P/B, market cap, ROE, ROA)',
                'financial_ratios': 'financial ratios (current ratio, quick ratio, margins, debt ratios)',
                'historical_market_cap': 'historical market capitalization data',
                'income_statements_ttm': 'trailing twelve months income',
                'balance_sheets_ttm': 'trailing twelve months balance sheet',
                'cash_flow_statements_ttm': 'trailing twelve months cash flow',
            }
            
            table_names = [table_descriptions.get(tbl, tbl) for tbl in selected_tables]
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', f'Found relevant financial data: {", ".join(table_names)}', {'step': 'table_selection', 'tables': selected_tables}).__dict__}
            if not selected_tables:
                raise QueryProcessingError("Could not determine relevant financial data sources for your question.", "table_selection")

            # Step 3: Generate and execute SQL
            self._check_timeout(start_time, "SQL generation")
            log_message("Step 3: Generating and executing SQL...")
            
            # This is now a generator, so we must iterate through it
            sql_generator = self.generate_and_execute_sql(question, selected_tables, parsed_intent, page, effective_page_size)
            for event in sql_generator:
                if event['type'] == 'result':
                    result_dict = event['data']
                    
                    # FIXED: Final status update based on success/failure
                    if result_dict.get('error'):
                        yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', f'❌ Analysis failed: {result_dict.get("error")}', {'step': 'final_result'}).__dict__}
                    else:
                        data_count = len(result_dict.get('data_rows', []))
                        yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', f'✅ Analysis complete: {data_count} results found', {'step': 'final_result'}).__dict__}
                    
                    yield {'type': 'result', 'data': result_dict}
                else:
                    # This is a reasoning event from the sub-generator
                    yield event

        except (TimeoutError, IntentParsingError, QueryProcessingError) as e:
            error_type = 'timeout' if isinstance(e, TimeoutError) else 'processing_error'
            log_message(f"❌ Query processing error: {e}", is_milestone=True)
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', f'❌ An unexpected error occurred during financial analysis', {'step': 'error_handling'}).__dict__}
            yield {'type': 'error', 'message': 'An unexpected error occurred. Please try again.'}
            yield {
                'type': 'result',
                'data': {
                    "error": "An unexpected error occurred during financial analysis", 
                    "message": "An unexpected error occurred. Please try again.",
                    "error_type": error_type,
                    "columns": None, "friendly_columns": None, "data_rows": None,
                    "tables_used": [], "pagination_info": None, "is_paginated": False
                }
            }
        except Exception as e:
            log_message(f"❌ Unexpected error in query method: {e}", is_milestone=True)
            import traceback
            traceback.print_exc()
            yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', f'❌ Unexpected error in financial analysis: {str(e)}', {'step': 'error_handling'}).__dict__}
            yield {'type': 'error', 'message': f"An unexpected error occurred: {e}"}
            yield {'type': 'result', 'data': {"error": f"An unexpected error occurred: {str(e)}", "message": "An internal error occurred."}}

    def query(self, question: str, page: int = 1, page_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Main query method for financial data analysis
        """
        # Check cache FIRST - before any LLM calls!
        cached_pipeline_entry = self._get_cached_pipeline(question)
        if cached_pipeline_entry:
            # Handle cached single-sheet result
            log_message("⚡ Using cached single-sheet result - no LLM calls needed!")
            selected_tables = cached_pipeline_entry['selected_tables']
            sql_query = cached_pipeline_entry['sql_query']
            full_raw_df = cached_pipeline_entry['raw_results_df']
            
            should_paginate = self._detect_pagination_intent(question)
            # For complete dataset requests (page_size=None), don't paginate
            if page_size is None:
                effective_page_size = None
                should_paginate = False
            else:
                effective_page_size = page_size or self.default_page_size
            paginated_raw_result = self._paginate_results(full_raw_df, page, effective_page_size, should_paginate)
            df_page_raw_data = paginated_raw_result['page_data']
            pagination_info = paginated_raw_result['pagination_info']

            generated_sql_columns = list(df_page_raw_data.columns) if not df_page_raw_data.empty else list(full_raw_df.columns)
            data_rows_formatted_for_json = self._handle_dataframe_for_json(df_page_raw_data, question, generated_sql_columns)
                
            return {
                "sql_query_generated": sql_query, 
                "columns": generated_sql_columns,
                "friendly_columns": {}, 
                "data_rows": data_rows_formatted_for_json,
                "message": self._generate_result_message(data_rows_formatted_for_json, pagination_info),
                "error": None, 
                "tables_used": selected_tables, 
                "pagination_info": pagination_info,
                "is_paginated": should_paginate, 
                "used_stored_results": True,
                "cache_type_hit": "pipeline_cache"
            }
        
        # Parse intent and determine strategy (only if cache miss)
        parsed_intent = self.parse_query_intent(question)
        
        # Check for query rejection
        if parsed_intent.get('query_rejected', False):
            rejection_reason = parsed_intent.get('rejection_reason', 'Query is outside the scope of financial data analysis')
            log_message(f"🚫 Non-streaming query rejected: {rejection_reason}")
            return {
                "error": f"Query rejected: {rejection_reason}",
                "message": f"Sorry, this query is outside the scope of financial data analysis. I can help with company financial data, stock metrics, ratios, and investment analysis.",
                "query_rejected": True,
                "rejection_reason": rejection_reason,
                "columns": None,
                "friendly_columns": None,
                "data_rows": None,
                "tables_used": [],
                "pagination_info": None,
                "is_paginated": False
            }
        
        # Use single sheet analysis
        final_result = None
        for event in self._query_with_streaming_core(question, page, page_size):
            if event.get('type') == 'result':
                final_result = event.get('data')
                break
        
        return final_result or {
            "error": "Single-sheet processing failed to produce result.",
            "message": "An internal error occurred during query processing."
        }

    def query_with_streaming(self, question: str, page: int = 1, page_size: Optional[int] = None):
        """
        Main streaming method for financial data analysis
        """
        # Route to single-sheet streaming
        for event in self._query_with_streaming_core(question, page, page_size):
            yield event
    
    def query_multiple_quarters_parallel(self, question: str, quarters: List[str], page: int = 1, page_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Query multiple quarters in parallel for comparative analysis
        
        Args:
            question: The user's question
            quarters: List of quarters to query (e.g., ['2025_q1', '2025_q2'])
            page: Page number for pagination
            page_size: Number of results per page
            
        Returns:
            Combined results from all quarters
        """
        log_message(f"🚀 Starting parallel multi-quarter query for {len(quarters)} quarters")
        start_time = time.time()
        
        # Create quarter-specific questions
        quarter_questions = []
        for quarter in quarters:
            # Add quarter context to the question
            quarter_question = f"{question} (for {quarter})"
            quarter_questions.append((quarter, quarter_question))
        
        # Process quarters in parallel using ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(quarters), self.max_parallel_workers)) as executor:
            # Submit all quarter queries
            future_to_quarter = {
                executor.submit(self.query, quarter_question, page, page_size): quarter
                for quarter, quarter_question in quarter_questions
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_quarter):
                quarter = future_to_quarter[future]
                try:
                    result = future.result()
                    results[quarter] = result
                    log_message(f"✅ Completed query for {quarter}")
                except Exception as e:
                    log_message(f"❌ Error querying {quarter}: {e}")
                    results[quarter] = {
                        "error": f"Error processing {quarter}: {str(e)}",
                        "data_rows": [],
                        "columns": [],
                        "message": f"Failed to process {quarter}"
                    }
        
        # Combine results from all quarters
        combined_result = self._combine_quarter_results(results, question)
        
        processing_time = time.time() - start_time
        log_message(f"✅ Parallel multi-quarter query completed in {processing_time:.3f}s")
        
        return combined_result
    
    def _combine_quarter_results(self, quarter_results: Dict[str, Dict[str, Any]], original_question: str) -> Dict[str, Any]:
        """
        Combine results from multiple quarters into a single response
        
        Args:
            quarter_results: Dictionary mapping quarter to query results
            original_question: The original user question
            
        Returns:
            Combined results with quarter-specific data
        """
        log_message(f"🔄 Combining results from {len(quarter_results)} quarters")
        
        # Initialize combined result structure
        combined_result = {
            "sql_query_generated": "Multi-quarter parallel query",
            "columns": [],
            "friendly_columns": {},
            "data_rows": [],
            "message": "",
            "error": None,
            "tables_used": [],
            "pagination_info": {},
            "is_paginated": False,
            "used_stored_results": False,
            "cache_type_hit": "none",
            "quarter_results": {},
            "total_quarters": len(quarter_results),
            "successful_quarters": 0,
            "failed_quarters": 0
        }
        
        # Process each quarter's results
        all_columns = set()
        all_tables = set()
        successful_quarters = 0
        failed_quarters = 0
        
        for quarter, result in quarter_results.items():
            combined_result["quarter_results"][quarter] = result
            
            if result.get("error"):
                failed_quarters += 1
                log_message(f"❌ Quarter {quarter} failed: {result['error']}")
            else:
                successful_quarters += 1
                
                # Collect columns and tables
                if result.get("columns"):
                    all_columns.update(result["columns"])
                if result.get("tables_used"):
                    all_tables.update(result["tables_used"])
                
                # Add quarter prefix to data rows
                quarter_data = result.get("data_rows", [])
                for row in quarter_data:
                    row["quarter"] = quarter
                    combined_result["data_rows"].append(row)
        
        # Set combined metadata
        combined_result["columns"] = list(all_columns) + ["quarter"]
        combined_result["tables_used"] = list(all_tables)
        combined_result["successful_quarters"] = successful_quarters
        combined_result["failed_quarters"] = failed_quarters
        
        # Generate combined message
        if successful_quarters > 0:
            total_rows = len(combined_result["data_rows"])
            combined_result["message"] = f"Successfully analyzed {successful_quarters} quarters with {total_rows} total results. "
            if failed_quarters > 0:
                combined_result["message"] += f"{failed_quarters} quarters failed to process."
        else:
            combined_result["message"] = "No quarters were successfully processed."
            combined_result["error"] = "All quarter queries failed"
        
        log_message(f"✅ Combined results: {successful_quarters} successful, {failed_quarters} failed, {len(combined_result['data_rows'])} total rows")
        
        return combined_result
    
    def query_multiple_companies_parallel(self, question: str, companies: List[str], page: int = 1, page_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Query multiple companies in parallel for comparative analysis
        
        Args:
            question: The user's question
            companies: List of company tickers to query
            page: Page number for pagination
            page_size: Number of results per page
            
        Returns:
            Combined results from all companies
        """
        log_message(f"🚀 Starting parallel multi-company query for {len(companies)} companies")
        start_time = time.time()
        
        # Create company-specific questions
        company_questions = []
        for company in companies:
            # Add company context to the question
            company_question = f"{question} (for {company})"
            company_questions.append((company, company_question))
        
        # Process companies in parallel using ThreadPoolExecutor
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(companies), self.max_parallel_workers)) as executor:
            # Submit all company queries
            future_to_company = {
                executor.submit(self.query, company_question, page, page_size): company
                for company, company_question in company_questions
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_company):
                company = future_to_company[future]
                try:
                    result = future.result()
                    results[company] = result
                    log_message(f"✅ Completed query for {company}")
                except Exception as e:
                    log_message(f"❌ Error querying {company}: {e}")
                    results[company] = {
                        "error": f"Error processing {company}: {str(e)}",
                        "data_rows": [],
                        "columns": [],
                        "message": f"Failed to process {company}"
                    }
        
        # Combine results from all companies
        combined_result = self._combine_company_results(results, question)
        
        processing_time = time.time() - start_time
        log_message(f"✅ Parallel multi-company query completed in {processing_time:.3f}s")
        
        return combined_result
    
    def _combine_company_results(self, company_results: Dict[str, Dict[str, Any]], original_question: str) -> Dict[str, Any]:
        """
        Combine results from multiple companies into a single response
        
        Args:
            company_results: Dictionary mapping company to query results
            original_question: The original user question
            
        Returns:
            Combined results with company-specific data
        """
        log_message(f"🔄 Combining results from {len(company_results)} companies")
        
        # Initialize combined result structure
        combined_result = {
            "sql_query_generated": "Multi-company parallel query",
            "columns": [],
            "friendly_columns": {},
            "data_rows": [],
            "message": "",
            "error": None,
            "tables_used": [],
            "pagination_info": {},
            "is_paginated": False,
            "used_stored_results": False,
            "cache_type_hit": "none",
            "company_results": {},
            "total_companies": len(company_results),
            "successful_companies": 0,
            "failed_companies": 0
        }
        
        # Process each company's results
        all_columns = set()
        all_tables = set()
        successful_companies = 0
        failed_companies = 0
        
        for company, result in company_results.items():
            combined_result["company_results"][company] = result
            
            if result.get("error"):
                failed_companies += 1
                log_message(f"❌ Company {company} failed: {result['error']}")
            else:
                successful_companies += 1
                
                # Collect columns and tables
                if result.get("columns"):
                    all_columns.update(result["columns"])
                if result.get("tables_used"):
                    all_tables.update(result["tables_used"])
                
                # Add company prefix to data rows
                company_data = result.get("data_rows", [])
                for row in company_data:
                    row["company"] = company
                    combined_result["data_rows"].append(row)
        
        # Set combined metadata
        combined_result["columns"] = list(all_columns) + ["company"]
        combined_result["tables_used"] = list(all_tables)
        combined_result["successful_companies"] = successful_companies
        combined_result["failed_companies"] = failed_companies
        
        # Generate combined message
        if successful_companies > 0:
            total_rows = len(combined_result["data_rows"])
            combined_result["message"] = f"Successfully analyzed {successful_companies} companies with {total_rows} total results. "
            if failed_companies > 0:
                combined_result["message"] += f"{failed_companies} companies failed to process."
        else:
            combined_result["message"] = "No companies were successfully processed."
            combined_result["error"] = "All company queries failed"
        
        log_message(f"✅ Combined results: {successful_companies} successful, {failed_companies} failed, {len(combined_result['data_rows'])} total rows")
        
        return combined_result

    def _check_data_availability_and_reject(self, question: str, selected_tables: List[str]) -> Optional[Dict[str, Any]]:
        """
        Check if the query asks for data we don't have and return rejection if so
        
        Args:
            question: The user's question
            selected_tables: Tables that were selected
            
        Returns:
            Rejection dictionary if query should be rejected, None otherwise
        """
        # DISABLED: Column parser rejection logic removed to make system less restrictive
        # All queries will now be processed regardless of data availability patterns
        log_message(f"✅ Data availability check disabled - processing query: {question[:100]}...")
        return None

    def select_relevant_tables(self, question: str, parsed_intent: Optional[Dict[str, Any]] = None) -> List[str]:
        """Select relevant tables based on question and optionally parsed intent"""
        # If we don't have parsed intent yet (parallel execution), use a simplified approach
        if parsed_intent is None:
            simplified_intent_str = "Intent parsing in progress - using question text only"
        else:
            simplified_intent_str = json.dumps(parsed_intent, indent=2)
            
        table_selection_prompt = """You are an expert database analyst. Select which tables are needed based on the query.

QUESTION: {question}
PARSED INTENT: {parsed_intent}

Available tables:
- company_profiles: General company info (name, sector, industry, current market cap, current stock price)
- income_statements: Revenue, profit, expenses for specific years (BEST for "revenue 2024", "annual revenue")
- balance_sheets: Assets, liabilities, equity for specific years (BEST for "balance sheet 2024", "annual assets")
- cash_flow_statements: Operating, investing, financing cash flows for specific years (BEST for "cash flow 2024", "annual cash flow")

- income_statements_ttm: TTM revenue, profit, expenses (DEFAULT for "revenue", "latest revenue", "current revenue")
- balance_sheets_ttm: TTM assets, liabilities, equity (DEFAULT for "assets", "latest balance sheet", "current assets")
- cash_flow_statements_ttm: TTM cash flows (DEFAULT for "cash flow", "latest cash flow", "current cash flow")
- key_metrics_ttm: TTM key metrics including return on equity (ROE), return on assets (ROA), return on capital employed (ROCE), return on invested capital (ROIC) (DEFAULT for "key ratios", "return metrics", "performance ratios")

🚨 CRITICAL: For "key ratios" queries, BOTH key_metrics_ttm AND financial_ratios_ttm should be selected to provide comprehensive financial analysis!

🚨 GROWTH TABLES (HIGHEST PRIORITY FOR GROWTH QUERIES):
- income_statements_growth_rates: Pre-calculated LATEST QUARTER YoY growth rates for revenue, profit, expenses (BEST for "revenue growth", "profit growth", "earnings growth", "ttm profit growth")
- balance_sheets_growth_rates: Pre-calculated LATEST QUARTER YoY growth rates for assets, equity, debt (BEST for "asset growth", "equity growth", "debt growth")
- cash_flow_statements_growth_rates: Pre-calculated LATEST QUARTER YoY growth rates for cash flows (BEST for "cash flow growth", "operating cash flow growth")
- growth_rates: Combined table with ALL latest quarter YoY growth rates (BEST for growth analysis)


⚠️ CRITICAL: Growth tables contain ONLY LATEST QUARTER YoY growth rates, not historical growth data!

🚨 CRITICAL TABLE SELECTION RULES:
1. Always include "company_profiles" if using any other financial table (required for sector/industry context)
2. If question mentions sectors/industries, ensure "company_profiles" is selected
3. Match financial metrics in question to appropriate tables
4. For any valuation metrics (P/E, P/B, stock price), include "key_metrics" and "financial_ratios"
5. **GROWTH TABLES (HIGHEST PRIORITY)**: For ANY growth-related queries, use growth tables instead of calculating growth manually. Growth tables have pre-calculated YoY growth rates that are much faster and more accurate than manual calculations.
6. **TTM TABLES (NEW DEFAULT)**: Use TTM tables by default unless query explicitly mentions specific years or timelines
7. **REGULAR TABLES**: Only use regular tables when query explicitly mentions specific years, annual data, or historical periods
8. When in doubt, include company_profiles as it's the base table

NEW TTM DEFAULT RULES:
- "revenue" (no timeline) → Use "income_statements_ttm" (DEFAULT)
- "profit" (no timeline) → Use "income_statements_ttm" (DEFAULT)
- "latest revenue" → Use "income_statements_ttm" (DEFAULT)
- "current financials" → Use TTM tables (DEFAULT)
- "revenue 2024" → Use "income_statements" (specific year)
- "annual revenue" → Use "income_statements" (annual data)
- "revenue last 5 years" → Use "income_statements" (historical period)
- "TTM revenue" → Use "income_statements_ttm" (explicitly mentioned)

EXAMPLES:
✅ "show me all companies revenue" → ["company_profiles", "income_statements_ttm"] (TTM DEFAULT!)
✅ "show me all companies revenue 2024" → ["company_profiles", "income_statements"] (specific year)
✅ "show me all companies annual revenue" → ["company_profiles", "income_statements"] (annual data)
✅ "show me all companies TTM revenue" → ["company_profiles", "income_statements_ttm"] (explicit TTM)
✅ "latest financial results" → ["company_profiles", "income_statements_ttm", "balance_sheets_ttm"] (TTM DEFAULT!)

🚨 GROWTH TABLE EXAMPLES (HIGHEST PRIORITY):
✅ "revenue growth" → ["company_profiles", "income_statements_growth_rates"] (USE GROWTH TABLE!)

✅ "positive ttm profit growth" → ["company_profiles", "income_statements_growth_rates"] (USE GROWTH TABLE!)
✅ "asset growth" → ["company_profiles", "balance_sheets_growth_rates"] (USE GROWTH TABLE!)
✅ "cash flow growth" → ["company_profiles", "cash_flow_statements_growth_rates"] (USE GROWTH TABLE!)
✅ "highest revenue growth" → ["company_profiles", "income_statements_growth_rates"] (USE GROWTH TABLE!)
✅ "companies with growth" → ["company_profiles", "income_statements_growth_rates", "balance_sheets_growth_rates"] (USE GROWTH TABLES!)

🚨 SPECIFIC QUERY EXAMPLES:
✅ "Show me all companies with positive ttm profit growth" → ["company_profiles", "income_statements_growth_rates"] (USE GROWTH TABLE!)
✅ "Which companies have the highest revenue growth?" → ["company_profiles", "income_statements_growth_rates"] (USE GROWTH TABLE!)
✅ "Find companies with positive earnings growth" → ["company_profiles", "income_statements_growth_rates"] (USE GROWTH TABLE!)


Return only a JSON array of table names.
Example: ["company_profiles", "income_statements_ttm", "key_metrics", "financial_ratios"]

🚨 FINAL REMINDER: If the question mentions ANY type of growth (revenue growth, profit growth, earnings growth, ttm profit growth, etc.), you MUST select the appropriate growth table instead of trying to calculate growth manually from TTM or regular tables!"""

        try:
            prompt = ChatPromptTemplate.from_template(table_selection_prompt)
            llm = self.get_agent_llm('table_selection')
            formatted_prompt = prompt.format(
                question=question,
                parsed_intent=simplified_intent_str
            )
            
            # Log prompt length for table selection
            prompt_length = len(formatted_prompt)
            log_message(f"\nPROMPT LENGTH FOR TABLE SELECTION:\n{prompt_length}\n")
            
            response = llm.invoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            log_message(f"Table selection RAW response: {response_text}")

            match = re.search(r'\[\s*("?\w+"?\s*,\s*)*"?\w*"?\s*\]', response_text)
            if match:
                json_str = match.group(0)
                selected_tables_raw = json.loads(json_str)
                selected_tables = [tbl for tbl in selected_tables_raw if tbl in self.table_schemas]
                if len(selected_tables_raw) > 0 and len(selected_tables) == 0:
                    log_message(f"Warning: Table selection LLM hallucinated all table names: {selected_tables_raw}")
                    selected_tables = ["company_profiles"] if "company_profiles" in self.table_schemas else list(self.table_schemas.keys())[:1]
            else:
                log_message(f"Warning: Could not parse JSON list from table selection response: {response_text}")
                selected_tables = [tbl_key for tbl_key in self.table_schemas.keys() if tbl_key.replace("_", "") in response_text.lower().replace("_", "")]
                if not selected_tables:
                    selected_tables = ["company_profiles"] if "company_profiles" in self.table_schemas else list(self.table_schemas.keys())[:1]

            # Auto-include company_profiles if any financial tables are selected
            financial_tables = ["income_statements", "balance_sheets", "cash_flow_statements", "key_metrics", "financial_ratios", 
                                "income_statements_ttm", "balance_sheets_ttm", "cash_flow_statements_ttm", "financial_ratios_ttm", "key_metrics_ttm",
                                "income_statements_growth_rates", "balance_sheets_growth_rates", "cash_flow_statements_growth_rates"]
            if "company_profiles" in self.table_schemas and \
               any(tbl in selected_tables for tbl in financial_tables) and \
               "company_profiles" not in selected_tables:
                selected_tables.insert(0, "company_profiles")

            # NEW: Apply TTM default logic if no specific timeline mentioned
            # Check if parsed_intent indicates TTM should be used by default
            if parsed_intent and parsed_intent.get('uses_ttm_by_default', False):
                # Replace regular tables with TTM equivalents if they're selected
                ttm_mapping = {
                    'income_statements': 'income_statements_ttm',
                    'balance_sheets': 'balance_sheets_ttm', 
                    'cash_flow_statements': 'cash_flow_statements_ttm',
                }
                
                updated_tables = []
                for table in selected_tables:
                    if table in ttm_mapping and ttm_mapping[table] in self.table_schemas:
                        updated_tables.append(ttm_mapping[table])
                        log_message(f"🔄 TTM Default: Replaced {table} with {ttm_mapping[table]}")
                    else:
                        updated_tables.append(table)
                
                selected_tables = updated_tables

            # Always include key_metrics and financial_ratios for default financial metrics (P/E, P/B)
            if "key_metrics" in self.table_schemas and "key_metrics" not in selected_tables and any(tbl in selected_tables for tbl in financial_tables):
                selected_tables.append("key_metrics")
            if "financial_ratios" in self.table_schemas and "financial_ratios" not in selected_tables and any(tbl in selected_tables for tbl in financial_tables):
                selected_tables.append("financial_ratios")
            
            # NEW: Ensure both key_metrics_ttm and financial_ratios_ttm are included when "key ratios" is mentioned
            question_lower = question.lower()
            if "key ratios" in question_lower or "key metrics" in question_lower or "return on" in question_lower:
                if "key_metrics_ttm" in self.table_schemas and "key_metrics_ttm" not in selected_tables:
                    selected_tables.append("key_metrics_ttm")
                    log_message(f"🔄 Key ratios detected: Added key_metrics_ttm table")
                if "financial_ratios_ttm" in self.table_schemas and "financial_ratios_ttm" not in selected_tables:
                    selected_tables.append("financial_ratios_ttm")
                    log_message(f"🔄 Key ratios detected: Added financial_ratios_ttm table")



            selected_tables = list(dict.fromkeys(selected_tables))

            log_message(f"📋 Table Selection Agent ({self.agent_configs['table_selection']['model']}): {selected_tables}")
            return selected_tables if selected_tables else (["company_profiles"] if "company_profiles" in self.table_schemas else list(self.table_schemas.keys())[:1])
        except Exception as e:
            log_message(f"Table selection failed: {e}. Defaulting to company_profiles, key_metrics, and financial_ratios.")
            default_tables = []
            if "company_profiles" in self.table_schemas:
                default_tables.append("company_profiles")
            if "key_metrics" in self.table_schemas:
                default_tables.append("key_metrics")
            if "financial_ratios" in self.table_schemas:
                default_tables.append("financial_ratios")
            return default_tables if default_tables else list(self.table_schemas.keys())[:1]

    def generate_and_execute_sql(self, question: str, selected_tables: List[str], 
                                 parsed_intent: Dict[str, Any],
                                 page: int = 1, page_size: Optional[int] = None):
        should_paginate = self._detect_pagination_intent(question)
        # For complete dataset requests (page_size=None), don't paginate
        if page_size is None:
            effective_page_size = None
            should_paginate = False
        else:
            effective_page_size = page_size or self.default_page_size
        
        focused_schema = self._build_focused_schema(selected_tables, parsed_intent)
        max_attempts = 5
        attempt_history = []
        final_sql_query = None
        full_raw_df = pd.DataFrame()
        timing_details = {'generation_attempts': [], 'execution_attempts': []}

        # NEW: Zero results retry tracking
        zero_results_retry_used = False

        for attempt in range(1, max_attempts + 1):
            log_message(f"🔧 SQL Generation Attempt {attempt}/{max_attempts}")
            try:
                gen_start_time = time.time()
                
                # Check if CAGR should be communicated to user
                uses_cagr = parsed_intent.get('uses_cagr_default', False)
                if uses_cagr and attempt == 1:  # Only show on first attempt
                    yield {'type': 'reasoning', 'event': ReasoningEvent('info', '📊 Using CAGR (Compound Annual Growth Rate) for growth calculations by default', {'step': 'cagr_notice', 'calculation_method': 'CAGR'}).__dict__}
                
                # NEW: Check if this is a filtering query and inform user about negative value exclusion
                question_lower = question.lower()
                filtering_keywords = ['low', 'high', 'less than', 'greater than', 'under', 'over', 'below', 'above', 'minimum', 'maximum', 'best', 'worst', 'top', 'bottom']
                is_filtering_query = any(keyword in question_lower for keyword in filtering_keywords)
                
                if is_filtering_query and attempt == 1:  # Only show on first attempt
                    yield {'type': 'reasoning', 'event': ReasoningEvent('info', '🔍 Filtering query detected - Automatically excluding negative values (unprofitable/unhealthy companies)', {'step': 'negative_value_filtering', 'filtering_type': 'screening'}).__dict__}
                
                # Only show attempt information on first attempt to avoid cluttering the UI
                if attempt == 1:
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_start', 'Preparing financial analysis...', {'step': 'sql_generation'}).__dict__}

                sql_query = self._generate_sql_query(question, focused_schema, parsed_intent, attempt_history, attempt)
                gen_time = time.time() - gen_start_time
                timing_details['generation_attempts'].append({'attempt': attempt, 'duration_sec': gen_time, 'sql': sql_query})
                
                final_sql_query = sql_query
                if not sql_query:
                    error_msg = 'Unable to prepare financial analysis query.'
                    attempt_history.append({'attempt': attempt, 'error': error_msg, 'sql': None, 'error_type': 'empty_query'})
                    log_message(f"❌ Attempt {attempt} failed: {error_msg}")
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', error_msg, {'step': 'sql_generation'}).__dict__}
                    if attempt == max_attempts: break
                    continue
                
                yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', 'Financial analysis prepared successfully', {'step': 'sql_generation', 'sql': sql_query}).__dict__}
                
                stored_full_raw_results = self._get_stored_results(sql_query)
                if stored_full_raw_results is not None:
                    log_message(f"✅ Using stored full RAW results for SQL query (Page {page})")
                    full_raw_df = stored_full_raw_results
                    timing_details['execution_attempts'].append({'attempt': attempt, 'duration_sec': 0, 'cached': True, 'rows': len(full_raw_df)})
                    yield {'type': 'reasoning', 'event': ReasoningEvent('info', 'Found previously analyzed results for this request', {'step': 'sql_execution'}).__dict__}
                else:
                    log_message(f"💾 Executing PostgreSQL SQL and storing full RAW results")
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_start', 'Retrieving financial data...', {'step': 'sql_execution'}).__dict__}
                    execution_result = self._try_execute_query(sql_query)
                    timing_details['execution_attempts'].append({'attempt': attempt, 'duration_sec': execution_result['timing'], 'cached': False, 'error': execution_result.get('error')})

                    if execution_result['success']:
                        log_message(f"✅ PostgreSQL SQL execution succeeded on attempt {attempt}")
                        yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', f"Financial data retrieved successfully, found {len(execution_result['data'])} total records", {'step': 'sql_execution'}).__dict__}
                        full_raw_df = execution_result['data']
                        timing_details['execution_attempts'][-1]['rows'] = len(full_raw_df)
                        self._store_results(sql_query, full_raw_df)
                    else:
                        error_detail = execution_result['error']
                        error_type = execution_result['error_type']
                        attempt_history.append({'attempt': attempt, 'sql': sql_query, 'error': error_detail, 'error_type': error_type})
                        log_message(f"❌ Attempt {attempt} PostgreSQL SQL execution failed: {error_detail} (Type: {error_type})")
                        yield {'type': 'reasoning', 'event': ReasoningEvent('step_error', 'An unexpected error occurred during financial analysis', {'step': 'sql_execution'}).__dict__}
                        
                        # Don't retry on timeout errors
                        if error_type == 'timeout':
                            break
                        if attempt == max_attempts: break
                        continue
                
                # NEW: Check for zero results and retry once more (only once)
                if len(full_raw_df) == 0 and not zero_results_retry_used and attempt < max_attempts:
                    zero_results_retry_used = True
                    log_message(f"⚠️ Zero results detected on attempt {attempt}. Triggering zero results retry...")
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_warning', 'No financial data found with current analysis. Attempting to refine the approach...', {'step': 'zero_results_retry'}).__dict__}
                    
                    # Add zero results context to attempt history for next retry
                    attempt_history.append({
                        'attempt': attempt, 
                        'sql': sql_query, 
                        'error': 'Zero results returned. Query may be too restrictive or have GROUP BY issues with filing columns.', 
                        'error_type': 'zero_results',
                        'suggestion': 'Consider simplifying GROUP BY clause and removing individual filing columns from grouping.'
                    })
                    continue  # Retry with zero results context
                
                formatting_start_time = time.time()
                paginated_raw_result = self._paginate_results(full_raw_df, page, effective_page_size, should_paginate)
                df_page_raw_data = paginated_raw_result['page_data']
                pagination_info = paginated_raw_result['pagination_info']
                
                generated_sql_columns = list(df_page_raw_data.columns) if not df_page_raw_data.empty else list(full_raw_df.columns)

                data_rows_formatted_for_json = self._handle_dataframe_for_json(df_page_raw_data, question, generated_sql_columns)
                
                formatting_time = time.time() - formatting_start_time
                timing_details['formatting_and_pagination_sec'] = formatting_time

                
                # Apply advanced filtering if required
                advanced_filtering_results = None
                if parsed_intent.get('use_semantic_post_processing', False):
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_start', 'Filtering best candidates from the dataset...', {'step': 'advanced_filtering'}).__dict__}
                    
                    # Add warning for complex queries
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_warning', 
                        '⚠️ Note: This search isn\'t exhaustive at this stage and will be improved.', 
                        {'step': 'filtering_warning'}).__dict__}
                    
                    filtering_start = time.time()
                    advanced_filtering_results = self._apply_semantic_post_processing(
                        full_raw_df, question, parsed_intent
                    )
                    filtering_time = time.time() - filtering_start
                    timing_details['advanced_filtering_sec'] = filtering_time
                    
                    # Use the filtered DataFrame instead of the original
                    full_raw_df = advanced_filtering_results['filtered_df']
                    
                    yield {'type': 'reasoning', 'event': ReasoningEvent('step_complete', 
                        f"Filtering complete: Selected {advanced_filtering_results['total_selected']} best candidates from {advanced_filtering_results['total_processed']} companies", 
                        {'step': 'advanced_filtering', 'selection_rate': advanced_filtering_results['selection_rate']}).__dict__}
                    
                    # Recalculate pagination and formatting with filtered data
                    paginated_raw_result = self._paginate_results(full_raw_df, page, effective_page_size, should_paginate)
                    df_page_raw_data = paginated_raw_result['page_data']
                    pagination_info = paginated_raw_result['pagination_info']
                    
                    generated_sql_columns = list(df_page_raw_data.columns) if not df_page_raw_data.empty else list(full_raw_df.columns)
                    data_rows_formatted_for_json = self._handle_dataframe_for_json(df_page_raw_data, question, generated_sql_columns)

                # Cache the pipeline result for future use
                self._cache_query_results(question, selected_tables, parsed_intent, sql_query, full_raw_df)
                
                # Enhanced message with advanced filtering info
                enhanced_message = self._enhance_result_message_with_cagr(
                    self._generate_result_message(data_rows_formatted_for_json, pagination_info),
                    sql_query, generated_sql_columns, question
                )
                
                if advanced_filtering_results:
                    filtering_note = f" 🔍 FILTERING: Applied intelligent filtering to select the {advanced_filtering_results['total_selected']} best candidates from {advanced_filtering_results['total_processed']} companies (selection rate: {advanced_filtering_results['selection_rate']:.1%})."
                    enhanced_message += filtering_note
                
                # Check if this is a semantic query
                is_semantic_query = parsed_intent.get('use_semantic_post_processing', False) or any([
                    'semantic' in str(event).lower() for event in attempt_history
                ]) if attempt_history else False
                
                # This is a generator, so we yield the final result
                result_data = {
                    "sql_query_generated": sql_query, "columns": generated_sql_columns,
                    "friendly_columns": {}, "data_rows": data_rows_formatted_for_json,
                    "message": enhanced_message,
                    "error": None, "tables_used": selected_tables, "pagination_info": pagination_info,
                    "is_paginated": should_paginate, "used_stored_results": stored_full_raw_results is not None,
                    "timing_details": timing_details, "is_semantic_query": is_semantic_query
                }
                
                # Add advanced filtering metadata if available
                if advanced_filtering_results:
                    result_data["advanced_filtering"] = {
                        "applied": True,
                        "total_processed": advanced_filtering_results['total_processed'],
                        "total_selected": advanced_filtering_results['total_selected'],
                        "selection_rate": advanced_filtering_results['selection_rate'],
                        "processing_batches": advanced_filtering_results['processing_batches']
                    }
                else:
                    result_data["advanced_filtering"] = {
                        "applied": False
                    }
                
                yield {
                    "type": "result",
                    "data": result_data
                }
                return # Exit the generator successfully

            except Exception as e:
                error_msg = f"Error in generation/execution attempt {attempt}: {e}"
                attempt_history.append({'attempt': attempt, 'error': error_msg, 'sql': final_sql_query, 'error_type': 'generation_loop_error'})
                log_message(f"❌ {error_msg}")
                # Don't create new reasoning steps for attempt errors - just log them
                log_message(f"❌ Attempt {attempt} error: {e}")
                if attempt == max_attempts: break
        
        
        failure_summary = f"Failed to generate and execute SQL after {len(attempt_history)} attempts.\n"
        for i, hist_item in enumerate(attempt_history, 1):
            failure_summary += f"Attempt {i}: {hist_item.get('error', 'Unknown error')}\n"
        
        log_message(failure_summary)
        yield {
            "type": "result",
            "data": {
                "sql_query_generated": final_sql_query, "error": "An unexpected error occurred during financial analysis",
                "message": "An unexpected error occurred. Please try again.",
                "columns": None, "friendly_columns": None, "data_rows": None,
                "tables_used": selected_tables, "pagination_info": None, "is_paginated": False,
                "used_stored_results": False, "timing_details": timing_details,
                "show_retry": True
            }
        }

    def _format_cell_value(self, value: Any, column_name: str, question: str, is_numeric_col_type: bool) -> str:
        """ENHANCED - Formats a single cell value with ROBUST percentage and year handling"""
        if pd.isna(value) or value is None:
            return "-"

        col_lower = column_name.lower()
        


        # ================================
        # YEAR HANDLING - SECOND HIGHEST PRIORITY
        # ================================
        year_indicators = ["year", "calendaryear", "fiscal_year", "yr"]
        if any(indicator in col_lower for indicator in year_indicators):
            try:
                # Try to parse as year
                year_val = int(float(value))
                # Validate reasonable year range
                if 1900 <= year_val <= 2100:
                    return str(year_val)  # Return plain year - NO $ signs!
                # If outside range, fall through to other formatting
            except (ValueError, TypeError):
                # If can't parse as year, return as string
                return str(value)

        # ================================
        # EXTENSIVE FORMATTING MAPPING - ONLY FORMAT RECOGNIZED COLUMNS
        # ================================
        
        # PERCENTAGE COLUMNS (get % suffix)
        percentage_columns = [
            # Explicit percentage terms
            "percentage", "percent", "pct", "%",
            # Growth and rate terms
            "cagr", "growth_rate", "growthrate", "tax_rate", "taxrate", "interest_rate", "interestrate",
            # Specific financial ratios that are percentages
            "roe", "roa", "roic", "roce", "roi", "return_on_equity", "return_on_assets", "return_on_invested_capital",
            "returnonequity", "returnonassets", "returnoninvestedcapital",
            # Specific margin types only
            "gross_margin", "grossmargin", "net_margin", "netmargin", "operating_margin", "operatingmargin",
            "ebitda_margin", "ebitdamargin", "profit_margin", "profitmargin", "safety_margin", "safetymargin",
            # Yield and return ratios
            "dividend_yield", "dividendyield", "yield", "return_on",
            # Beta and volatility
            "beta", "volatility",
            # Payout and retention ratios
            "payout_ratio", "payoutratio", "retention_ratio", "retentionratio",
            # Specific growth metrics
            "revenue_growth_rate", "revenuegrowthrate", "income_growth_rate", "incomegrowthrate",
            "earnings_growth_rate", "earningsgrowthrate", "quarterly_revenue_growth", "quarterlyrevenuegrowth",
            "quarterly_earnings_growth", "quarterlyearningsgrowth",
            # Tax ratios
            "effective_tax_rate", "effectivetaxrate", "tax_rate", "taxrate"
        ]
        
        # RATIO COLUMNS (get x suffix)
        ratio_columns = [
            # Valuation ratios
            "pe", "pb", "ps", "ev", "pricetobook", "pricetoearnings", "pricetosales",
            "price_to_book", "price_to_earnings", "price_to_sales",
            "pricetobookratio", "pricetoearningsratio", "pricetosalesratio",
            # Debt ratios
            "debt_to_equity", "debttoequity", "debt_to_assets", "debttoassets",
            "current_ratio", "currentratio", "quick_ratio", "quickratio",
            # Coverage ratios
            "interest_coverage", "interestcoverage", "coverage_ratio", "coverageratio",
            # Turnover ratios
            "turnover_ratio", "turnoverratio", "asset_turnover", "assetturnover",
            # Other ratios
            "book_value_per_share", "bookvaluepershare", "cash_per_share", "cashpershare",
            "free_cash_flow_per_share", "freecashflowpershare", "operating_cash_flow_per_share", "operatingcashflowpershare"
        ]
        
        # CURRENCY/FINANCIAL COLUMNS (get $ with scaling)
        currency_columns = [
            # Revenue and income
            "revenue", "income", "sales", "total_revenue", "totalrevenue", "total_income", "totalincome",
            # Profit terms (these are currency, not percentages!)
            "gross_profit", "grossprofit", "net_profit", "netprofit", "operating_profit", "operatingprofit",
            "ebitda", "ebit", "operating_income", "operatingincome",
            # Assets and liabilities
            "assets", "liabilities", "equity", "debt", "cash", "total_assets", "totalassets",
            "total_liabilities", "totalliabilities", "total_equity", "totalequity", "total_debt", "totaldebt",
            "total_cash", "totalcash",
            # Market cap and value
            "market_cap", "marketcap", "mktcap", "market_value", "marketvalue", "enterprise_value", "enterprisevalue",
            # Costs and expenses
            "cost", "expenses", "total_expenses", "totalexpenses", "capex", "opex", "operating_expenses", "operatingexpenses",
            # R&D and research expenses
            "rnd", "research", "development", "research_development", "researchdevelopment",
            # Price and amount
            "price", "stockprice", "amount", "value", "fee", "usd"
        ]
        
        # COUNT/INTEGER COLUMNS (get comma formatting)
        count_columns = [
            "count", "number", "qty", "quantity", "shares", "outstanding", "shares_outstanding", "sharesoutstanding",
            "outstanding_shares", "outstandingshares"
        ]
        
        # Check if column matches any known patterns
        col_clean = col_lower.replace("_", "").replace(" ", "")
        
        # PERCENTAGE FORMATTING - REMOVE ALL % CONVERSIONS, RETURN RAW VALUE WITH LIMITED PRECISION
        if any(indicator in col_clean for indicator in [p.replace("_", "").replace(" ", "") for p in percentage_columns]):
            # LLM should add a % in the column name if required for display
            # Limit precision to max 2 decimal places for floating point numbers
            if is_numeric_col_type:
                try:
                    num_val = float(value)
                    return f"{num_val:.2f}"
                except (ValueError, TypeError):
                    pass
            return str(value)
        
        # CURRENCY/FINANCIAL FORMATTING (no $ symbol, just scaling) - PRIORITY OVER RATIOS
        elif any(indicator in col_clean for indicator in [c.replace("_", "").replace(" ", "") for c in currency_columns]):
            if is_numeric_col_type:
                try:
                    num_val = float(value)
                    
                    if abs(num_val) >= 1e12: 
                        suffix = f"{num_val / 1e12:.2f}T"
                    elif abs(num_val) >= 1e9: 
                        suffix = f"{num_val / 1e9:.2f}B"
                    elif abs(num_val) >= 1e6: 
                        suffix = f"{num_val / 1e6:.2f}M"
                    else:
                        suffix = f"{num_val:,.2f}"
                    
                    # Return without $ symbol
                    return suffix
                except (ValueError, TypeError):
                    pass
        
        # RATIO FORMATTING (no suffix) - AFTER CURRENCY CHECK
        elif any(indicator in col_clean for indicator in [r.replace("_", "").replace(" ", "") for r in ratio_columns]):
            if is_numeric_col_type:
                try:
                    return f"{float(value):.2f}"
                except (ValueError, TypeError):
                    pass
        
        # COUNT/INTEGER FORMATTING (comma-separated)
        elif any(indicator in col_clean for indicator in [c.replace("_", "").replace(" ", "") for c in count_columns]):
            if is_numeric_col_type:
                try:
                    int_val = int(float(value))
                    return f"{int_val:,}"  # Add commas for large numbers
                except (ValueError, TypeError):
                    pass

        # ================================
        # DATE FORMATTING
        # ================================
        if any(keyword in col_lower for keyword in ["date", "time", "day"]):
            try:
                if isinstance(value, (datetime, pd.Timestamp)):
                    return value.strftime('%Y-%m-%d')
                parsed_date = pd.to_datetime(value, errors='coerce')
                if pd.notna(parsed_date):
                    return parsed_date.strftime('%Y-%m-%d')
            except Exception:
                pass

        # ================================
        # DEFAULT NUMERIC FORMATTING (NO $ FALLBACK)
        # ================================
        if is_numeric_col_type:
            try:
                num_val = float(value)
                # Limit precision to max 2 decimal places for floating point numbers
                if abs(num_val) < 1:
                    return f"{num_val:.2f}"
                else:
                    return f"{num_val:,.2f}"
            except (ValueError, TypeError):
                pass

        # ================================
        # FALLBACK: STRING FORMATTING
        # ================================
        return str(value)


    def _format_dataframe_for_display(self, df: pd.DataFrame, question: str, generated_sql_columns: List[str]) -> pd.DataFrame:
        """Formats the entire DataFrame for display, converting values to strings."""
        if df.empty:
            return df

        formatted_df = df.copy()
        for col_name in generated_sql_columns:
            if col_name in formatted_df:
                is_numeric_col_type = pd.api.types.is_numeric_dtype(df[col_name])
                formatted_df[col_name] = df[col_name].apply(
                    lambda x: self._format_cell_value(x, col_name, question, is_numeric_col_type)
                )
        return formatted_df

    
    def _handle_dataframe_for_json(self, df: pd.DataFrame, question: str, generated_sql_columns: List[str]) -> List[Dict[str, Any]]:
        """Enhanced DataFrame to JSON conversion with intelligent column type detection"""
        if df.empty:
            return []
        
        actual_df_columns = list(df.columns)
        if not all(col in actual_df_columns for col in generated_sql_columns):
            log_message(f"Warning: Mismatch between generated_sql_columns and actual DataFrame columns. Using actual: {actual_df_columns}")
            columns_to_format = actual_df_columns
        else:
            columns_to_format = generated_sql_columns

        # SMART FORMATTING: Don't format year columns as financial data
        df_for_json = df.copy()
        
        for col_name in columns_to_format:
            if col_name in df_for_json:
                col_lower = col_name.lower()
                
                # YEAR COLUMNS: Keep as plain integers/strings - HIGHEST PRIORITY
                year_indicators = ["year", "calendaryear", "fiscal_year", "yr"]
                if any(indicator in col_lower for indicator in year_indicators):
                    # Don't format year columns - keep them as-is
                    df_for_json[col_name] = df_for_json[col_name].apply(
                        lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.','').replace('-','').isdigit() else str(x) if pd.notna(x) else "-"
                    )
                    continue
                
                # DATE COLUMNS: Format as dates
                if "date" in col_lower or "day" in col_lower or "time" in col_lower:
                    df_for_json[col_name] = df_for_json[col_name].apply(
                        lambda x: self._format_date_value(x)
                    )
                    continue
                
                # OTHER COLUMNS: Use existing formatting logic
                is_numeric_col_type = pd.api.types.is_numeric_dtype(df[col_name])
                df_for_json[col_name] = df[col_name].apply(
                    lambda x: self._format_cell_value(x, col_name, question, is_numeric_col_type)
                )
        
        return df_for_json.to_dict(orient='records')
    
    def _format_date_value(self, value: Any) -> str:
        """Helper method to format date values"""
        if pd.isna(value) or value is None:
            return "-"
        try:
            if isinstance(value, (datetime, pd.Timestamp)):
                return value.strftime('%Y-%m-%d')
            parsed_date = pd.to_datetime(value, errors='coerce')
            if pd.notna(parsed_date):
                return parsed_date.strftime('%Y-%m-%d')
        except Exception:
            pass
        return str(value)

    def _try_execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Enhanced execute query with PostgreSQL validation"""
        start_time = time.time()
        
        # Pre-validate SQL for PostgreSQL compatibility
        is_valid, validation_error = self._validate_sql_for_postgres(sql_query)
        if not is_valid:
            return {
                'success': False, 
                'data': None, 
                'error': f"SQL Validation Error: {validation_error}", 
                'error_type': 'validation_error', 
                'timing': time.time() - start_time
            }
        
        try:
            log_message(f"Executing PostgreSQL SQL: {sql_query}")
            
            query_start_time = time.time()
            conn = get_postgres_connection()
            try:
                df = pd.read_sql_query(sql_query, conn)
            finally:
                conn.close()
            query_execution_time = time.time() - query_start_time
            
            fetch_time = 0  # No separate fetch time for pandas read_sql_query
            
            # Enhanced duplicate detection and removal
            original_rows = len(df)
            if original_rows > 0:
                # Log sample data before deduplication for debugging
                log_message(f"📊 Sample data before deduplication (first 3 rows):")
                for i in range(min(3, len(df))):
                    sample_row = df.iloc[i]
                    log_message(f"  Row {i+1}: {dict(sample_row)}")
                
                # Remove duplicate rows - keep first occurrence
                df = df.drop_duplicates(keep='first').reset_index(drop=True)
                duplicates_removed = original_rows - len(df)
                
                if duplicates_removed > 0:
                    log_message(f"🧹 Removed {duplicates_removed} duplicate rows from results")
                    log_message(f"📊 Final results: {len(df)} unique rows")
                    
                    # Log sample data after deduplication
                    log_message(f"📊 Sample data after deduplication (first 3 rows):")
                    for i in range(min(3, len(df))):
                        sample_row = df.iloc[i]
                        log_message(f"  Row {i+1}: {dict(sample_row)}")
                else:
                    log_message(f"✅ No duplicates found in {original_rows} rows")
            else:
                log_message(f"⚠️ Query returned {original_rows} rows (no data)")
            
            total_time = time.time() - start_time
            
            log_message(f"✅ PostgreSQL query executed successfully in {total_time:.4f}s ({len(df)} rows)")
            
            return {'success': True, 'data': df, 'error': None, 'error_type': None, 'timing': total_time}
                    
        except Exception as e:
            total_time = time.time() - start_time
            error_msg = str(e)
            
            log_message(f"PostgreSQL execution error: {error_msg}")
            
            # Enhanced error categorization for PostgreSQL
            error_str = error_msg.lower()
            if 'column' in error_str and ('not exist' in error_str or 'does not exist' in error_str):
                error_type = 'column_not_found'
            elif 'relation' in error_str and ('does not exist' in error_str or 'not exist' in error_str):
                error_type = 'table_not_found'
            elif any(keyword in error_str for keyword in ['syntax error', 'parser error', 'unexpected token']):
                error_type = 'syntax_error'
            elif 'function' in error_str and ('does not exist' in error_str or 'not exist' in error_str):
                error_type = 'function_error'
            elif 'timeout' in error_str or 'canceling statement due to statement timeout' in error_str:
                error_type = 'timeout'
            elif 'permission denied' in error_str:
                error_type = 'permission_error'
            else:
                error_type = 'execution_error'
                    
            return {
                'success': False, 
                'data': None, 
                'error': f"PostgreSQL Error: {error_msg}", 
                'error_type': error_type, 
                'timing': total_time
            }

    def _detect_pagination_intent(self, question: str) -> bool:
        # Always paginate when there are multiple results
        return True

    def _paginate_results(self, full_df: pd.DataFrame, page: int, page_size: Optional[int], should_paginate: bool) -> Dict[str, Any]:
        total_records = len(full_df)

        # Handle case where page_size is None (complete dataset requested)
        if page_size is None:
            log_message(f"📄 Complete dataset requested: returning all {total_records} records")
            return {'page_data': full_df, 'pagination_info': None}

        # Always paginate when there are multiple results
        total_pages = math.ceil(total_records / page_size) if total_records > 0 else 1
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = full_df.iloc[start_idx:end_idx]
        
        showing_from = start_idx + 1 if len(page_df) > 0 else 0
        showing_to = start_idx + len(page_df) if len(page_df) > 0 else 0
        
        pagination_info = {
            "current_page": page, "page_size": page_size, "total_records": total_records,
            "total_pages": total_pages, "has_next": page < total_pages, "has_previous": page > 1,
            "showing_from": showing_from, "showing_to": showing_to
        }
        log_message(f"📄 Pagination: {showing_from}-{showing_to} of {total_records} records (Page {page}/{total_pages})")
        return {'page_data': page_df, 'pagination_info': pagination_info}

    def _generate_sql_query(self, question: str, focused_schema: str, parsed_intent: Dict[str, Any], attempt_history: List[Dict], attempt: int) -> str:
        """Generate SQL query with enhanced PostgreSQL syntax awareness and proper data year handling"""
        
        # Extract intent fields
        has_specific_companies = parsed_intent.get('has_specific_companies', False)
        use_sector_industry_filters = parsed_intent.get('use_sector_industry_filters', True)
        use_description_filtering = parsed_intent.get('use_description_filtering', False)
        entities = parsed_intent.get('entities', [])

        # --- LIMIT guidance for LLM ---
        use_semantic_post_processing = parsed_intent.get('use_semantic_post_processing', False)
        if use_semantic_post_processing:
            # For semantic post-processing, always use 500 companies to process
            max_companies_to_process = 500
            # For semantic post-processing, modify the question to remove specific count requirements
            # and add guidance to get more candidates
            original_question = question
            # Remove specific number requests from the question for SQL generation
            modified_question = re.sub(r'\b\d+\s+companies?\b', 'companies', question, flags=re.IGNORECASE)
            modified_question = re.sub(r'\btop\s+\d+\b', 'companies', modified_question, flags=re.IGNORECASE)
            modified_question = re.sub(r'\bshow\s+me\s+\d+\b', 'find', modified_question, flags=re.IGNORECASE)
            modified_question = re.sub(r'\blist\s+\d+\b', 'find', modified_question, flags=re.IGNORECASE)
            
            limit_guidance = (
                f"\nCRITICAL: For this query, return as many candidate companies as possible "
                f"(up to {max_companies_to_process}) for AI post-processing. "
                f"Use LIMIT {max_companies_to_process} in your SQL. Do NOT limit to 10 or 20. "
                f"Ignore any specific number requests in the question - get the maximum candidates possible."
            )
            question = modified_question  # Use modified question for SQL generation
        else:
            # For non-semantic queries, only apply LIMIT if user specifically requests it
            max_companies_to_process = parsed_intent.get('max_companies_to_process', 20)
            max_companies_to_output = parsed_intent.get('max_companies_to_output', 20)
            
            # Check if user explicitly requested a specific number of companies
            user_requested_limit = parsed_intent.get('user_requested_limit')
            if user_requested_limit:
                # User specifically asked for a certain number, so apply that limit
                limit_guidance = (
                    f"\nCRITICAL: For this query, return only the top {user_requested_limit} companies. "
                    f"Use LIMIT {user_requested_limit} in your SQL."
                )
            else:
                # User didn't specify a limit, so don't apply any LIMIT clause
                limit_guidance = (
                    f"\nCRITICAL: For this query, return ALL matching companies. "
                    f"Do NOT use LIMIT in your SQL unless the user specifically requested a certain number."
                )
        
        # NEW: Detect if this is a filtering query that should exclude negative values
        question_lower = question.lower()
        filtering_keywords = ['low', 'high', 'less than', 'greater than', 'under', 'over', 'below', 'above', 'minimum', 'maximum', 'best', 'worst', 'top', 'bottom']
        is_filtering_query = any(keyword in question_lower for keyword in filtering_keywords)
        
        # Build filter instructions
        if has_specific_companies and entities:
            filter_instructions = f"""
    COMPANY FILTERING:
    - Target specific companies/tickers: {entities}
    - Use WHERE conditions like: symbol IN ('TICKER1', 'TICKER2') OR companyName = 'ExactCompanyName'
    - Do not apply sector/industry filters for specific company queries
    - Use exact company name matching, not LIKE patterns
    """
        elif use_description_filtering:
            description_keywords = parsed_intent.get('description_keywords', [])
            filter_instructions = f"""
    DESCRIPTION-BASED FILTERING:
    - Use company description field to find specific business types
    - Keywords to search for: {description_keywords}
    - IMPORTANT: Do NOT use LIKE clauses for description filtering
    - Instead, use semantic post-processing for approximate matching
    - This targets specific companies rather than entire sectors/industries
    - Do NOT use sector/industry filters when using description filtering
    """
        elif use_sector_industry_filters and (parsed_intent.get('exact_sector_filters') or parsed_intent.get('exact_industry_filters')):
            sector_filters = parsed_intent.get('exact_sector_filters', [])
            industry_filters = parsed_intent.get('exact_industry_filters', [])
            filter_instructions = f"""
    SECTOR/INDUSTRY FILTERING:
    - Sector filters: {sector_filters}
    - Industry filters: {industry_filters}
    - For SECTORS: Use exact match: sector IN ('Technology', 'Healthcare')
    - For INDUSTRIES: Use exact match only: industry IN ('Software - Application', 'Banks - Diversified')
    - CRITICAL: Do NOT use LIKE clauses for industry matching
    - CRITICAL: Use only exact industry names from the database
    - CRITICAL: For approximate industry matching, rely on semantic post-processing instead of SQL LIKE
    """
        else:
            filter_instructions = "NO SECTOR/INDUSTRY FILTERING: Analyze ALL companies across ALL sectors and industries. Do not make assumptions about which sectors are relevant."

        # NEW: Add negative value filtering instructions for filtering queries
        negative_value_filtering = ""
        if is_filtering_query:
            negative_value_filtering = """
    🚨 CRITICAL NEGATIVE VALUE FILTERING RULES (FOR FILTERING QUERIES):
    When users ask for filtering queries (like "low PE ratio", "high ROE", "companies with debt < X"), 
    automatically exclude negative values as they are undesirable in screening contexts:
    
    VALUATION RATIOS (EXCLUDE NEGATIVE VALUES):
    - P/E Ratio (peRatio, priceEarningsRatio): WHERE peRatio > 0 AND peRatio < threshold
    - P/B Ratio (pbRatio, priceToBookRatio): WHERE pbRatio > 0 AND pbRatio < threshold  
    - P/S Ratio (priceToSalesRatio): WHERE priceToSalesRatio > 0 AND priceToSalesRatio < threshold
    - P/FCF Ratio (pfcfRatio, priceToFreeCashFlowsRatio): WHERE pfcfRatio > 0 AND pfcfRatio < threshold
    - P/OCF Ratio (pocfratio, priceToOperatingCashFlowsRatio): WHERE pocfratio > 0 AND pocfratio < threshold
    - P/TB Ratio (ptbRatio, priceToTangibleBookRatio): WHERE ptbRatio > 0 AND ptbRatio < threshold
    - EV/EBITDA (evEbitda): WHERE evEbitda > 0 AND evEbitda < threshold
    - EV/Revenue (evRevenue): WHERE evRevenue > 0 AND evRevenue < threshold
    
    PROFITABILITY METRICS (EXCLUDE NEGATIVE VALUES):
    - ROE (returnOnEquity): WHERE returnOnEquity > 0 AND returnOnEquity > threshold
    - ROA (returnOnAssets): WHERE returnOnAssets > 0 AND returnOnAssets > threshold
    - ROIC (returnOnInvestedCapital): WHERE returnOnInvestedCapital > 0 AND returnOnInvestedCapital > threshold
    - Net Profit Margin (netProfitMargin): WHERE netProfitMargin > 0 AND netProfitMargin > threshold
    - Operating Margin (operatingIncomeMargin): WHERE operatingIncomeMargin > 0 AND operatingIncomeMargin > threshold
    - Gross Margin (grossProfitMargin): WHERE grossProfitMargin > 0 AND grossProfitMargin > threshold
    
    FINANCIAL HEALTH METRICS (EXCLUDE NEGATIVE VALUES):
    - Current Ratio (currentRatio): WHERE currentRatio > 0 AND currentRatio > threshold
    - Quick Ratio (quickRatio): WHERE quickRatio > 0 AND quickRatio > threshold
    - Debt-to-Equity (debtToEquity): WHERE debtToEquity > 0 AND debtToEquity < threshold
    - Interest Coverage (interestCoverage): WHERE interestCoverage > 0 AND interestCoverage > threshold
    
    CASH FLOW METRICS (EXCLUDE NEGATIVE VALUES):
    - Operating Cash Flow (operatingCashFlow): WHERE operatingCashFlow > 0 AND operatingCashFlow > threshold
    - Free Cash Flow (freeCashFlow): WHERE freeCashFlow > 0 AND freeCashFlow > threshold
    - Cash Flow to Debt (cashFlowToDebtRatio): WHERE cashFlowToDebtRatio > 0 AND cashFlowToDebtRatio > threshold
    
    EXAMPLES:
    ✅ "Low PE ratio companies" → WHERE peRatio > 0 AND peRatio < 20
    ✅ "High ROE companies" → WHERE returnOnEquity > 0 AND returnOnEquity > 0.15
    ✅ "Companies with debt < 50%" → WHERE debtToEquity > 0 AND debtToEquity < 0.5
    ✅ "Strong cash flow companies" → WHERE operatingCashFlow > 0 AND operatingCashFlow > 1000000000
    
    🚨 CRITICAL: Always add > 0 condition BEFORE the user's threshold condition!
    🚨 CRITICAL: This applies to ALL filtering queries, not just specific metrics mentioned!
    🚨 CRITICAL: Negative values are excluded because they indicate poor financial health or unprofitable companies!
    """

        # NEW: Add critical table selection guidance for actual vs per-share values
        table_selection_guidance = """
    🎯 CRITICAL TABLE SELECTION GUIDANCE:
    
    🎯 ACTUAL FINANCIAL VALUES (use these tables):
    - "cash" → Use "balance_sheets_ttm.cashAndCashEquivalents" (NOT key_metrics.cashPerShare)
- "revenue" → Use "income_statements_ttm.revenue" (NOT key_metrics.revenuePerShare)  
    - "total assets" → Use "balance_sheets_ttm.totalAssets"
    - "total debt" → Use "balance_sheets_ttm.totalDebt"
    - "net income" → Use "income_statements_ttm.netIncome"
    - "operating cash flow" → Use "cash_flow_statements_ttm.operatingCashFlow"
    - "free cash flow" → Use "cash_flow_statements_ttm.freeCashFlow"
    
    📊 PER-SHARE METRICS (use key_metrics table):
- "cash per share" → Use "key_metrics.cashPerShare"
- "revenue per share" → Use "key_metrics.revenuePerShare"
- "book value per share" → Use "key_metrics.bookValuePerShare"
- "free cash flow per share" → Use "key_metrics.freeCashFlowPerShare"

🔢 RATIOS (use financial_ratios table):
- "PE ratio" → Use "key_metrics.priceEarningsRatio"
- "PB ratio" → Use "key_metrics.priceToBookRatio"
- "debt to equity" → Use "financial_ratios.debtEquityRatio"
- "ROE" → Use "financial_ratios.returnOnEquity"
- "ROIC" → Use "key_metrics.roic" (Return on Invested Capital)
- "ROCE" → Use "financial_ratios.returnOnCapitalEmployed" (Return on Capital Employed - DIFFERENT from ROIC!)
    
    ⚠️ CRITICAL FINANCIAL METRIC NUANCES:
    - ROIC ≠ ROCE: These are DIFFERENT metrics with different calculations and meanings
    - ROIC (Return on Invested Capital): Measures return on total invested capital
    - ROCE (Return on Capital Employed): Measures return on capital employed (typically equity + long-term debt)
    - When user asks for "ROCE", do NOT use ROIC - they are asking for different metrics!
    - Pay attention to exact metric names: "return on equity" ≠ "return on assets" ≠ "return on capital"
    - "profit margin" could mean gross, operating, or net profit margin - check context carefully
    - "debt ratio" could mean debt-to-equity, debt-to-assets, or total debt ratio - verify the exact metric
    
    ⚠️ IMPORTANT: When user asks for "cash" or "revenue" without "per share", use actual financial statement tables, NOT key_metrics!
    """

        # NEW: Add critical column name guidance for financial_ratios_ttm table
        column_name_guidance = """
    🚨 CRITICAL COLUMN NAME GUIDANCE FOR financial_ratios_ttm AND key_metrics_ttm TABLES:
    
    ✅ CORRECT COLUMN NAMES FOR financial_ratios_ttm TABLE (fr alias):
    - fr.currentRatioTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.quickRatioTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.grossProfitMarginTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.netProfitMarginTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.debtToEquityRatioTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.ebtPerEbitTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.assetTurnoverTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.cashRatioTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.effectiveTaxRateTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.fixedAssetTurnoverTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.operatingCycleTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.priceCashFlowRatioTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.priceEarningsToGrowthRatioTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.priceFairValueTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.totalDebtToCapitalizationTTM (CORRECT - exists in financial_ratios_ttm)
    
    ✅ CORRECT COLUMN NAMES FOR key_metrics_ttm TABLE (km alias):
    - km.returnOnEquityTTM (CORRECT - exists in key_metrics_ttm)
    - km.returnOnAssetsTTM (CORRECT - exists in key_metrics_ttm)
    - km.returnOnCapitalEmployedTTM (CORRECT - exists in key_metrics_ttm)
    - km.returnOnInvestedCapitalTTM (CORRECT - exists in key_metrics_ttm)
    - km.operatingReturnOnAssetsTTM (CORRECT - exists in key_metrics_ttm)
    - km.enterpriseValueTTM (CORRECT - exists in key_metrics_ttm)
    - km.evToSalesTTM (CORRECT - exists in key_metrics_ttm)
    - km.evToOperatingCashFlowTTM (CORRECT - exists in key_metrics_ttm)
    - km.evToFreeCashFlowTTM (CORRECT - exists in key_metrics_ttm)
    - km.earningsYieldTTM (CORRECT - exists in key_metrics_ttm)
    - km.freeCashFlowYieldTTM (CORRECT - exists in key_metrics_ttm)
    
    ✅ CORRECT COLUMN NAMES FOR financial_ratios_ttm TABLE (fr alias):
    - fr.priceToEarningsRatioTTM (CORRECT - PE ratio exists in financial_ratios_ttm)
    - fr.priceToBookRatioTTM (CORRECT - PB ratio exists in financial_ratios_ttm)
    - fr.priceToSalesRatioTTM (CORRECT - PS ratio exists in financial_ratios_ttm)
    - fr.priceToFreeCashFlowRatioTTM (CORRECT - exists in financial_ratios_ttm)
    - fr.priceToOperatingCashFlowRatioTTM (CORRECT - exists in financial_ratios_ttm)
    
    ❌ WRONG COLUMN NAMES (DO NOT USE - THESE DO NOT EXIST):
    - fr.returnOnEquityTTM (does not exist in financial_ratios_ttm)
    - fr.returnOnAssetsTTM (does not exist in financial_ratios_ttm)
    - km.peRatioTTM (does not exist in key_metrics_ttm - use fr.priceToEarningsRatioTTM instead)
    - km.priceToEarningsRatioTTM (does not exist in key_metrics_ttm - use fr.priceToEarningsRatioTTM instead)
    - fr.returnOnEquity (does not exist)
    - fr.returnOnAssets (does not exist)
    - fr.currentRatio (does not exist)
    - fr.quickRatio (does not exist)
    - fr.grossProfitMargin (does not exist)
    - fr.netProfitMargin (does not exist)
    - fr.debtEquityRatio (does not exist)
    
    🚨 CRITICAL: Return on equity and return on assets metrics are in key_metrics_ttm table, NOT financial_ratios_ttm!
    🚨 CRITICAL: The financial_ratios_ttm table uses TTM SUFFIX in ALL column names!
    🚨 CRITICAL: ALL columns in financial_ratios_ttm table end with "TTM"!
    🚨 CRITICAL: The data in financial_ratios_ttm is TTM data, AND the column names have TTM suffix!
    
    🚨 CRITICAL COLUMN-TO-TABLE MAPPING (MANDATORY):
    ✅ RETURN METRICS → key_metrics_ttm (km alias):
       - km.returnOnEquityTTM
       - km.returnOnAssetsTTM  
       - km.returnOnCapitalEmployedTTM
       - km.returnOnInvestedCapitalTTM
       - km.operatingReturnOnAssetsTTM
    
    ✅ VALUATION RATIOS → financial_ratios_ttm (fr alias):
       - fr.priceToEarningsRatioTTM (PE ratio)
       - fr.priceToBookRatioTTM (PB ratio)
       - fr.priceToSalesRatioTTM (PS ratio)
       - fr.priceToFreeCashFlowRatioTTM
       - fr.priceToOperatingCashFlowRatioTTM
    
    ✅ FINANCIAL RATIOS → financial_ratios_ttm (fr alias):
       - fr.currentRatioTTM
       - fr.quickRatioTTM
       - fr.debtToEquityRatioTTM
       - fr.grossProfitMarginTTM
       - fr.netProfitMarginTTM
    
    ✅ CORRECT EXAMPLE WITH BOTH TABLES:
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          km.returnOnEquityTTM, km.returnOnAssetsTTM, km.returnOnCapitalEmployedTTM,
    ✅          fr.priceToEarningsRatioTTM, fr.currentRatioTTM, fr.quickRatioTTM,
    ✅          fr.grossProfitMarginTTM, fr.netProfitMarginTTM, fr.debtToEquityRatioTTM
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN financial_ratios_ttm fr ON cp.symbol = fr.symbol
    ✅   JOIN financial_data.key_metrics_ttm km ON cp.symbol = km.symbol
    ✅   WHERE cp.industry = 'Gambling, Resorts & Casinos'
    ✅   ORDER BY cp.mktCap DESC;
    
    ❌ WRONG EXAMPLE:
    ❌   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ❌          fr.returnOnEquityTTM, fr.returnOnAssetsTTM, fr.currentRatioTTM
    ❌   FROM financial_data.company_profiles cp
    ❌   JOIN financial_ratios_ttm fr ON cp.symbol = fr.symbol
    ❌   WHERE cp.industry = 'Gambling, Resorts & Casinos'
    ❌   ORDER BY cp.mktCap DESC;
    """

        # NEW: Add broad category filtering for semantic post-processing
        broad_category_filtering = ""
        if parsed_intent.get('use_semantic_post_processing', False) and parsed_intent.get('broad_categories'):
            broad_categories = parsed_intent.get('broad_categories', [])
            filter_columns = parsed_intent.get('filter_columns', [])
            
            # Create SQL filter conditions
            sector_filters = []
            industry_filters = []
            description_filters = []
            
            # Map broad categories to SQL filters
            for category in broad_categories:
                if category in ['Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary', 'Real Estate', 'Industrials', 'Energy', 'Materials', 'Utilities', 'Communication Services']:
                    sector_filters.append(f"'{category}'")
                # Note: Removed all LIKE-based filtering for specific categories
                # These will be handled by semantic post-processing instead
            
            # Build the filtering guidance
            filter_conditions = []
            if sector_filters:
                filter_conditions.append(f"cp.sector IN ({', '.join(sector_filters)})")
            if industry_filters:
                filter_conditions.extend(industry_filters)
            if description_filters:
                filter_conditions.extend(description_filters)
            
            if filter_conditions:
                broad_category_filtering = f"""
    🚨 CRITICAL BROAD CATEGORY FILTERING (MANDATORY):
    
    You MUST apply these pre-filters in your SQL WHERE clause to improve semantic query accuracy:
    
    MANDATORY FILTERS:
    WHERE {' OR '.join(filter_conditions)}
    
    🚨 CRITICAL: These filters MUST be included in your SQL query!
    🚨 CRITICAL: This reduces the dataset before semantic post-processing!
    🚨 CRITICAL: Do NOT skip these filters - they are required for accuracy!
    🚨 CRITICAL: Only exact sector matches are used - no LIKE clauses!
    
    EXAMPLE: If you have a WHERE clause, add these conditions with OR:
    WHERE (your_existing_conditions) AND ({' OR '.join(filter_conditions)})
    """
        
        # NEW: Add specific guidance for margin calculations
        margin_calculation_guidance = """
    🎯 CRITICAL MARGIN CALCULATION GUIDANCE:
    
    GROSS PROFIT MARGIN (grossProfitMargin):
    - Values are stored as DECIMALS (0.543 = 54.3%)
    - For percentage calculations, use decimal arithmetic
    - Example: 5% difference = 0.05 in decimal form
    - Example: MIN(margin) >= MAX(margin) - 0.05 (for 5% threshold)
    
    MARGIN VOLATILITY ANALYSIS:
    - Calculate margin range: MAX(grossProfitMargin) - MIN(grossProfitMargin)
    - Check if range is significant: (MAX - MIN) > 0.001 (0.1% minimum difference)
    - Apply percentage thresholds: MIN >= MAX - 0.05 (5% threshold)
    
    EXAMPLES:
    ✅ "Companies with stable gross margins (within 5%)" 
       → HAVING (MAX(grossProfitMargin) - MIN(grossProfitMargin)) > 0.001 
          AND MIN(grossProfitMargin) >= MAX(grossProfitMargin) - 0.05
    
    ✅ "Companies with volatile gross margins (more than 10% range)"
       → HAVING (MAX(grossProfitMargin) - MIN(grossProfitMargin)) > 0.10
    
    ✅ "Companies with flat gross margins (less than 1% variation)"
       → HAVING (MAX(grossProfitMargin) - MIN(grossProfitMargin)) < 0.01
    
    🚨 CRITICAL: Always exclude companies with exactly flat margins (MAX = MIN) unless specifically requested!
    🚨 CRITICAL: Use decimal arithmetic for percentage calculations!
    🚨 CRITICAL: Add minimum threshold to avoid noise from tiny variations!
    """

        # NEW: Add critical growth table guidance
        growth_table_guidance = """
    🚨 CRITICAL GROWTH TABLE GUIDANCE (HIGHEST PRIORITY):
    
    🎯 GROWTH TABLES CONTAIN PRE-CALCULATED YoY GROWTH RATES:
    - income_statements_growth_rates: Pre-calculated YoY growth rates for revenue, profit, expenses
    - balance_sheets_growth_rates: Pre-calculated YoY growth rates for assets, equity, debt  
    - cash_flow_statements_growth_rates: Pre-calculated YoY growth rates for cash flows
    - growth_rates: Combined table with ALL latest quarter YoY growth rates
    
    🚨 CRITICAL: If growth tables are available in the schema, you MUST use them instead of calculating growth manually!
    🚨 CRITICAL: Growth tables are much faster and more accurate than manual calculations!
    🚨 CRITICAL: Growth tables contain ONLY LATEST QUARTER YoY growth rates, not historical growth data!
    
    ✅ CORRECT PATTERNS FOR GROWTH TABLES:
    
    🎯 CRITICAL: Always include current/previous values AND period context!
    
    ✅ "profit growth" or "earnings growth" → Use income_statements_growth_rates:
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          igr.latest_fiscal_year, igr.latest_period, igr.latest_date,
    ✅          igr.previous_fiscal_year, igr.previous_period, igr.previous_date,
    ✅          igr.netIncome_current, igr.netIncome_previous,
    ✅          igr.netIncome_yoy_growth_pct AS profit_growth_percent,
    ✅          igr.netIncome_yoy_growth_amount AS profit_growth_amount
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN financial_data.income_statements_growth_rates igr ON cp.symbol = igr.symbol
    ✅   WHERE igr.netIncome_yoy_growth_pct > 0
    ✅   ORDER BY igr.netIncome_yoy_growth_pct DESC;
    
    ✅ "revenue growth" → Use income_statements_growth_rates:
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          igr.latest_fiscal_year, igr.latest_period, igr.latest_date,
    ✅          igr.previous_fiscal_year, igr.previous_period, igr.previous_date,
    ✅          igr.revenue_current, igr.revenue_previous,
    ✅          igr.revenue_yoy_growth_pct AS revenue_growth_percent,
    ✅          igr.revenue_yoy_growth_amount AS revenue_growth_amount
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN financial_data.income_statements_growth_rates igr ON cp.symbol = igr.symbol
    ✅   WHERE igr.revenue_yoy_growth_pct > 0
    ✅   ORDER BY igr.revenue_yoy_growth_pct DESC;
    
    ✅ "asset growth" → Use balance_sheets_growth_rates:
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          bgr.latest_fiscal_year, bgr.latest_period, bgr.latest_date,
    ✅          bgr.previous_fiscal_year, bgr.previous_period, bgr.previous_date,
    ✅          bgr.totalAssets_current, bgr.totalAssets_previous,
    ✅          bgr.totalAssets_yoy_growth_pct AS asset_growth_percent,
    ✅          bgr.totalAssets_yoy_growth_amount AS asset_growth_amount
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN balance_sheets_growth_rates bgr ON cp.symbol = bgr.symbol
    ✅   WHERE bgr.totalAssets_yoy_growth_pct > 0
    ✅   ORDER BY bgr.totalAssets_yoy_growth_pct DESC;
    
    ✅ "cash flow growth" → Use cash_flow_statements_growth_rates:
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          cgr.latest_fiscal_year, cgr.latest_period, cgr.latest_date,
    ✅          cgr.previous_fiscal_year, cgr.previous_period, cgr.previous_date,
    ✅          cgr.operatingCashFlow_current, cgr.operatingCashFlow_previous,
    ✅          cgr.operatingCashFlow_yoy_growth_pct AS cash_flow_growth_percent,
    ✅          cgr.operatingCashFlow_yoy_growth_amount AS cash_flow_growth_amount
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN cash_flow_statements_growth_rates cgr ON cp.symbol = cgr.symbol
    ✅   WHERE cgr.operatingCashFlow_yoy_growth_pct > 0
    ✅   ORDER BY cgr.operatingCashFlow_yoy_growth_pct DESC;
    
    ❌ WRONG: Do NOT calculate growth manually from TTM or regular tables when growth tables are available!
    ❌ WRONG: Do NOT use income_statements_ttm to calculate profit growth when income_statements_growth_rates is available!
    ❌ WRONG: Do NOT use window functions or LAG() to calculate growth when growth tables have pre-calculated rates!
    ❌ WRONG: Do NOT omit period information (latest_fiscal_year, latest_period, etc.) - users need this context!
    ❌ WRONG: Do NOT only show growth percentages without the actual current/previous values!
    
    🚨 CRITICAL: Growth tables are selected by the table selection agent for a reason - USE THEM!
    🚨 CRITICAL: The growth rates in these tables are YoY (Year-over-Year) for the latest quarter only!
    🚨 CRITICAL: Do NOT try to calculate historical growth or multi-period growth from these tables!
    
    🎯 CRITICAL OUTPUT REQUIREMENTS:
    🚨 ALWAYS include both current and previous period values!
    🚨 ALWAYS include fiscal year and period information (latest_fiscal_year, latest_period, previous_fiscal_year, previous_period)!
    🚨 ALWAYS include the growth percentage AND growth amount!
    🚨 This gives users complete context: "Company A had $1.2B profit in Q1 2025 vs $1.0B in Q1 2024 = 20% growth"
    
    🚨 MANDATORY COLUMNS FOR ALL GROWTH QUERIES:
    🚨 You MUST include these columns in EVERY growth query:
    🚨 1. latest_fiscal_year, latest_period, latest_date
    🚨 2. previous_fiscal_year, previous_period, previous_date  
    🚨 3. [metric]_current, [metric]_previous
    🚨 4. [metric]_yoy_growth_pct, [metric]_yoy_growth_amount
    🚨 5. companyName, symbol, sector, industry
    
    ✅ EXAMPLE OUTPUT COLUMNS FOR PROFIT GROWTH:
    ✅   - companyName, symbol, sector, industry
    ✅   - latest_fiscal_year, latest_period, latest_date
    ✅   - previous_fiscal_year, previous_period, previous_date  
    ✅   - netIncome_current, netIncome_previous
    ✅   - netIncome_yoy_growth_pct, netIncome_yoy_growth_amount
    """

        # Build comprehensive error context for retries
        error_context = ""
        if attempt_history:
            error_context = f"\n\n🚨 RETRY ATTEMPT {attempt}/3 - Previous attempts failed:\n"
            for i, hist in enumerate(attempt_history[-3:], 1):  # Show last 3 attempts
                error_detail = hist.get('error', 'Unknown error')
                error_context += f"Attempt {i}: {error_detail}\n"
            
            error_context += "\n🔧 CRITICAL FIXES NEEDED:\n"
            
            # Specific guidance based on error types
            error_messages = [str(hist.get('error', '')) for hist in attempt_history]
            
            # NEW: Zero results retry guidance
            if any("zero_results" in str(hist.get('error_type', '')) for hist in attempt_history):
                error_context += "🚨 ZERO RESULTS DETECTED - LIKELY FISCAL CALENDAR ISSUE!\n"
                error_context += "❌ Problem: Using global fiscal year/period filters ignores company-specific fiscal calendars!\n"
                error_context += "❌ WRONG: WHERE fiscalYear = 2024 AND period = 'Q3' (assumes all companies have same fiscal calendar)\n"
                error_context += "✅ CORRECT: Use window functions to find each company's latest quarter individually:\n\n"
                error_context += "✅ COMPANY-SPECIFIC LATEST QUARTER PATTERN:\n"
                error_context += "✅   WITH latest_quarters AS (\n"
                error_context += "✅     SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn\n"
                error_context += "✅     FROM income_statements_ttm\n"
                error_context += "✅   )\n"
                error_context += "✅   SELECT ... FROM latest_quarters WHERE rn = 1\n"
                error_context += "✅ This handles different fiscal year-ends (Oct, Dec, Jun, etc.) properly!\n\n"
                error_context += "🚨 CRITICAL: Different companies have different fiscal calendars!\n"
                error_context += "🚨 Company A: Q4 ends Oct-31, Company B: Q4 ends Dec-31, Company C: Q4 ends Jun-30\n\n"
            
            # NEW: CAGR time period validation error detection
            if any("cagr" in str(hist.get('error', '')).lower() or "growth" in str(hist.get('error', '')).lower() for hist in attempt_history):
                error_context += "🚨 CAGR TIME PERIOD VALIDATION ERROR DETECTED!\n"
                error_context += "❌ PROBLEM: Query includes companies with incomplete time periods for CAGR calculation!\n"
                error_context += "❌ WRONG: Including companies with only 2-6 years of data when asking for 10-year CAGR!\n"
                error_context += "✅ CORRECT: MUST enforce complete time period data for CAGR calculations!\n\n"
                error_context += "✅ MANDATORY PATTERN FOR 10-YEAR CAGR:\n"
                error_context += "✅   WITH revenue_cagr AS (\n"
                error_context += "✅     SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,\n"
                error_context += "✅            MIN(CASE WHEN inc.calendarYear = 2014 THEN inc.revenue END) AS start_revenue,\n"
                error_context += "✅            MAX(CASE WHEN inc.calendarYear = 2024 THEN inc.revenue END) AS end_revenue,\n"
                error_context += "✅            COUNT(DISTINCT inc.calendarYear) AS years_with_data\n"
                error_context += "✅     FROM financial_data.company_profiles cp\n"
                error_context += "✅     JOIN financial_data.income_statements inc ON cp.symbol = inc.symbol\n"
                error_context += "✅     WHERE inc.calendarYear BETWEEN 2014 AND 2024\n"
                error_context += "✅     GROUP BY cp.companyName, cp.symbol, cp.sector, cp.industry\n"
                error_context += "✅     HAVING years_with_data = 11  -- MUST have all 11 years!\n"
                error_context += "✅       AND start_revenue > 0 AND end_revenue > 0\n"
                error_context += "✅   )\n"
                error_context += "✅   SELECT companyName, symbol, sector, industry,\n"
                error_context += "✅          revenue_2014, revenue_2024,\n"
                error_context += "✅          ROUND(((revenue_2024 / revenue_2014) ^ (1.0 / 10) - 1) * 100, 2) AS revenue_cagr_10yr\n"
                error_context += "✅   FROM revenue_cagr\n"
                error_context += "✅   HAVING ROUND(((revenue_2024 / revenue_2014) ^ (1.0 / 10) - 1) * 100, 2) > 20\n"
                error_context += "✅   ORDER BY revenue_cagr_10yr DESC;\n\n"
                error_context += "🚨 CRITICAL SQL FILTERING RULE:\n"
                error_context += "❌ NEVER use WHERE with a column alias (like revenue_cagr) in the outer query!\n"
                error_context += "✅ ALWAYS use HAVING and repeat the full expression for filtering on calculated columns!\n"
                error_context += "✅ Example: HAVING ROUND(((end_revenue / start_revenue) ^ (1.0 / 10) - 1) * 100, 2) > 20\n\n"
                error_context += "🚨 CRITICAL: ALWAYS use HAVING COUNT(DISTINCT calendarYear) = (X + 1) for X years of growth!\n"
                error_context += "🚨 CRITICAL: For 10 years = 11 data points, for 5 years = 6 data points, etc.\n"
                error_context += "🚨 CRITICAL: NEVER include companies with incomplete time periods!\n\n"
            
            # ENHANCED: GROUP BY error detection and guidance
            if any("must appear in the GROUP BY clause" in msg or "must be part of an aggregate function" in msg for msg in error_messages):
                error_context += "🚨 CRITICAL GROUP BY ERROR DETECTED - IMMEDIATE FIX REQUIRED:\n"
                error_context += "❌ RULE: When using ANY aggregate functions (MIN, MAX, SUM, COUNT, AVG), ALL non-aggregate columns MUST be in GROUP BY!\n"
                error_context += "❌ PROBLEMATIC: SELECT col1, col2, MIN(col3) FROM table; (Missing GROUP BY)\n"
                error_context += "✅ CORRECT OPTION 1: Add ALL non-aggregate columns to GROUP BY:\n"
                error_context += "✅   SELECT col1, col2, MIN(col3) FROM table GROUP BY col1, col2;\n"
                error_context += "✅ CORRECT OPTION 2: Use DISTINCT ON (PostgreSQL-specific):\n"
                error_context += "✅   SELECT DISTINCT ON (group_key) col1, col2, MIN(col3) FROM table ORDER BY group_key;\n"
                error_context += "✅ CORRECT OPTION 3: Use ONLY aggregate functions:\n"
                error_context += "✅   SELECT MIN(col1), MAX(col2), AVG(col3) FROM table;\n\n"
                
                error_context += "🎯 SPECIFIC FIX FOR COMPANY QUERIES WITH FILING DATA:\n"
                error_context += "✅ PATTERN 1 - WITH GROUP BY (RECOMMENDED):\n"
                error_context += "✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry, cp.cik,\n"
                error_context += "✅          km.priceEarningsRatio, km.marketCap,\n"
                error_context += "✅          MIN(inc.fillingDate) AS first_fillingDate,\n"
                error_context += "✅          MAX(inc.fillingDate) AS last_fillingDate\n"
                error_context += "✅   FROM financial_data.company_profiles cp\n"
                error_context += "✅   JOIN financial_data.key_metrics km ON cp.symbol = km.symbol\n"
                error_context += "✅   JOIN financial_data.income_statements inc ON cp.symbol = inc.symbol\n"
                error_context += "✅   WHERE km.calendarYear = 2024 AND km.period = 'FY'\n"
                error_context += "✅   GROUP BY cp.companyName, cp.symbol, cp.sector, cp.industry, cp.cik, km.priceEarningsRatio, km.marketCap;\n\n"
                
                error_context += "✅ PATTERN 2 - WITH DISTINCT ON (PostgreSQL alternative):\n"
                error_context += "✅   SELECT DISTINCT ON (cp.symbol) \n"
                error_context += "✅          cp.companyname AS companyname,\n"
                error_context += "✅          cp.symbol AS symbol,\n"
                error_context += "✅          cp.sector AS sector,\n"
                error_context += "✅          cp.industry AS industry,\n"
                error_context += "✅          cp.cik AS cik,\n"
                error_context += "✅          km.priceearningsratio AS priceearningsratio,\n"
                error_context += "✅          km.marketcap AS marketcap,\n"
                error_context += "✅          MIN(inc.fillingDate) AS first_fillingDate,\n"
                error_context += "✅          MAX(inc.fillingDate) AS last_fillingDate\n"
                error_context += "✅   FROM financial_data.company_profiles cp\n"
                error_context += "✅   JOIN financial_data.key_metrics km ON cp.symbol = km.symbol\n"
                error_context += "✅   JOIN financial_data.income_statements inc ON cp.symbol = inc.symbol\n"
                error_context += "✅   WHERE km.calendarYear = 2024 AND km.period = 'FY';\n\n"
                
                error_context += "🚨 CRITICAL: Choose ONE approach - either GROUP BY or DISTINCT ON, not mixed!\n"
                error_context += "🚨 CRITICAL: Include ALL non-aggregate SELECT columns in GROUP BY clause!\n\n"
                
                # NEW: Specific fix for large cap company queries (the exact failing case)
                if any("mktCap" in msg or "marketCap" in msg for msg in error_messages):
                    error_context += "🚨 LARGE CAP COMPANY QUERY FIX (EXACT FAILING CASE):\n"
                    error_context += "✅ CORRECT - LARGE CAP COMPANIES WITH PROPER GROUP BY:\n"
                    error_context += "✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry, cp.cik, cp.mktCap,\n"
                    error_context += "✅          km.priceEarningsRatio AS \"PE Ratio\",\n"
                    error_context += "✅          km.marketCap,\n"
                    error_context += "✅          inc.netIncome,\n"
                    error_context += "✅          inc.totalOtherIncomeExpensesNet,\n"
                    error_context += "✅          inc.calendarYear,\n"
                    error_context += "✅          inc.period,\n"
                    error_context += "✅          MIN(inc.fillingDate) AS first_fillingDate,\n"
                    error_context += "✅          MAX(inc.fillingDate) AS last_fillingDate,\n"
                    error_context += "✅          MIN(inc.acceptedDate) AS first_acceptedDate,\n"
                    error_context += "✅          MAX(inc.acceptedDate) AS last_acceptedDate,\n"
                    error_context += "✅          MIN(inc.link) AS first_link,\n"
                    error_context += "✅          MAX(inc.link) AS last_link,\n"
                    error_context += "✅          MIN(inc.finalLink) AS first_finalLink,\n"
                    error_context += "✅          MAX(inc.finalLink) AS last_finalLink\n"
                    error_context += "✅   FROM financial_data.company_profiles cp\n"
                    error_context += "✅   JOIN financial_data.key_metrics km ON cp.symbol = km.symbol\n"
                    error_context += "✅   AND km.calendarYear = 2024 AND km.period = 'FY'\n"
                    error_context += "✅   JOIN financial_data.income_statements inc ON cp.symbol = inc.symbol\n"
                    error_context += "✅   AND inc.calendarYear = 2024 AND inc.period = 'FY'\n"
                    error_context += "✅   WHERE cp.mktCap >= 1000000000\n"
                    error_context += "✅   GROUP BY cp.companyName, cp.symbol, cp.sector, cp.industry, cp.cik, cp.mktCap,\n"
                    error_context += "✅            km.priceEarningsRatio, km.marketCap, inc.netIncome, inc.totalOtherIncomeExpensesNet,\n"
                    error_context += "✅            inc.calendarYear, inc.period\n"
                    error_context += "✅   ORDER BY cp.mktCap DESC, cp.companyName;\n\n"
                
                # NEW: Specific fix for financial data columns
                if any("revenue" in msg or "income" in msg or "profit" in msg or "assets" in msg or "liabilities" in msg for msg in error_messages):
                    error_context += "🚨 FINANCIAL DATA GROUP BY FIX:\n"
                    error_context += "❌ PROBLEMATIC: SELECT cp.companyName, cp.symbol, inc.revenue, MIN(inc.fillingDate) FROM ...\n"
                    error_context += "✅ CORRECT: SELECT cp.companyName, cp.symbol, inc.revenue, MIN(inc.fillingDate) FROM ... GROUP BY cp.companyName, cp.symbol, inc.revenue;\n"
                    error_context += "✅ OR BETTER: Use DISTINCT ON for financial data: SELECT DISTINCT ON (cp.symbol) cp.companyname, cp.symbol, inc.revenue, MIN(inc.fillingdate) FROM ... ORDER BY cp.symbol;\n"
                    error_context += "✅ PATTERN: Financial data columns (revenue, netincome, totalassets, etc.) must be in GROUP BY or use DISTINCT ON\n\n"
            
            if any("ILIKE ANY" in msg or "~~*" in msg for msg in error_messages):
                error_context += "❌ Use PostgreSQL-compatible syntax!\n"
                error_context += "✅ ALWAYS use: (column ILIKE '%pattern1%' OR column ILIKE '%pattern2%')\n"
            

            
            # Critical fix for TTM table column errors
            if any("calendarYear" in msg and ("income_statements_ttm" in msg or "balance_sheets_ttm" in msg or "cash_flow_statements_ttm" in msg) for msg in error_messages):
                error_context += "❌ TTM tables (income_statements_ttm, balance_sheets_ttm, cash_flow_statements_ttm) use 'fiscalYear' NOT 'calendarYear'!\n"
                error_context += "✅ CORRECT: Use inc.fiscalYear, bs.fiscalYear, cf.fiscalYear (NOT calendarYear) for TTM tables\n"


            
            if any("fiscalYear" in msg and "Candidate bindings" in msg for msg in error_messages):
                error_context += "❌ Column name error! PostgreSQL is telling you the correct column name!\n"
                error_context += "✅ TTM tables use fiscalYear, regular financial tables use calendarYear\n"
            
            # Handle missing filing columns in TTM tables
            if any("link" in msg and ("income_statements_ttm" in msg or "balance_sheets_ttm" in msg or "cash_flow_statements_ttm" in msg) for msg in error_messages):
                error_context += "❌ TTM tables do NOT have 'link' or 'finalLink' columns!\n"
                error_context += "✅ TTM tables only have: acceptedDate, cik, period\n"
            
            if any("does not have a column" in msg for msg in error_messages):
                error_context += "❌ Wrong column name detected! Check schema carefully!\n"
                error_context += "✅ TTM tables (income_statements_ttm, balance_sheets_ttm, cash_flow_statements_ttm): USE fiscalYear\n"
                error_context += "✅ Regular tables (income_statements, balance_sheets, cash_flow_statements): USE calendarYear\n"
            
            if any("Parser Error" in msg for msg in error_messages):
                error_context += "❌ PostgreSQL parser error - check syntax carefully!\n"
                error_context += "✅ Avoid ANY/ALL with arrays, use OR conditions or IN lists\n"
                error_context += "✅ Check for missing quotes, brackets, or invalid operators\n"
            
            # NEW: Specific fix for WHERE clause aggregate errors
            if any("WHERE clause cannot contain aggregates" in msg for msg in error_messages):
                error_context += "🚨 CRITICAL WHERE CLAUSE AGGREGATE ERROR DETECTED!\n"
                error_context += "❌ NEVER use aggregate functions (MAX, MIN, SUM, COUNT, AVG) in WHERE clauses or JOIN conditions!\n"
                error_context += "❌ WRONG: WHERE fr.date = (SELECT MAX(date) FROM financial_ratios_ttm)\n"
                error_context += "❌ WRONG: AND inc.fiscalYear = (SELECT MAX(fiscalYear) FROM income_statements_ttm)\n"
                error_context += "❌ WRONG: JOIN financial_ratios_ttm fr ON fr.date = (SELECT MAX(date) FROM financial_ratios_ttm)\n\n"
                error_context += "✅ CORRECT APPROACHES for getting latest data:\n"
                error_context += "✅ APPROACH 1 - Use explicit year/date values (RECOMMENDED):\n"
                error_context += "✅   WHERE inc.fiscalYear = 2024  -- Use the actual latest year\n"
                error_context += "✅   AND fr.date = '2024-12-31'   -- Use the actual latest date\n\n"
                error_context += "✅ APPROACH 2 - Use window functions in subqueries:\n"
                error_context += "✅   JOIN (\n"
                error_context += "✅     SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn\n"
                error_context += "✅     FROM financial_ratios_ttm\n"
                error_context += "✅   ) fr ON cp.symbol = fr.symbol AND fr.rn = 1\n\n"
                error_context += "✅ APPROACH 3 - Use CTEs for latest data:\n"
                error_context += "✅   WITH latest_ratios AS (\n"
                error_context += "✅     SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn\n"
                error_context += "✅     FROM financial_ratios_ttm\n"
                error_context += "✅   )\n"
                error_context += "✅   SELECT ... FROM financial_data.company_profiles cp\n"
                error_context += "✅   JOIN latest_ratios fr ON cp.symbol = fr.symbol AND fr.rn = 1\n\n"
                error_context += "🚨 CRITICAL: The latest fiscal year for TTM data is typically 2024\n"
                error_context += "🚨 CRITICAL: The latest date for financial ratios is typically '2024-12-31'\n"
                error_context += "🚨 CRITICAL: Always use explicit values instead of subqueries with aggregates in WHERE/JOIN!\n\n"
            
            error_context += "\n🎯 Generate clean, working PostgreSQL SQL that avoids ALL previous errors!\n\n"

        base_prompt = """You are a PostgreSQL SQL expert. Generate ONLY a syntactically correct PostgreSQL query.

🚨 STOP! READ THIS FIRST! 🚨
🚨 NEVER use fr.returnOnCapitalEmployedTTM - it does NOT exist in financial_ratios_ttm!
🚨 NEVER use fr.returnOnEquityTTM - it does NOT exist in financial_ratios_ttm!
🚨 NEVER use fr.returnOnAssetsTTM - it does NOT exist in financial_ratios_ttm!
🚨 ALWAYS use km.returnOnCapitalEmployedTTM for ROCE!
🚨 ALWAYS use km.returnOnEquityTTM for ROE!
🚨 ALWAYS use km.returnOnAssetsTTM for ROA!
🚨 ALWAYS use fr.priceToEarningsRatioTTM for PE ratio!
🚨 ALWAYS use fr.debtToEquityRatioTTM for debt to equity!

🚨 CRITICAL INSTRUCTIONS:
- Return ONLY the SQL query - NO explanations, NO reasoning, NO comments
- Do NOT include any text before or after the SQL
- Do NOT use <think> tags or reasoning blocks
- Generate clean, executable SQL that can run directly in PostgreSQL
- ALWAYS prefix table names with 'financial_data.' schema (e.g., financial_data.company_profiles)

📋 QUERY INFORMATION:
QUESTION: {question}
PARSED INTENT: {parsed_intent}

{error_context}

📊 DATABASE SCHEMA:
{focused_schema}

🚨 CRITICAL BROAD CATEGORY FILTERING (HIGHEST PRIORITY):
{broad_category_filtering}

🎯 FILTERING INSTRUCTIONS:
{filter_instructions}

{negative_value_filtering}

📈 TABLE SELECTION GUIDANCE:
{table_selection_guidance}

📊 COLUMN NAME GUIDANCE:
{column_name_guidance}

📊 GROWTH TABLE GUIDANCE:
{growth_table_guidance}

📊 MARGIN CALCULATION GUIDANCE:
{margin_calculation_guidance}

📊 LIMIT GUIDANCE:
{limit_guidance}

    CRITICAL DATA AVAILABILITY RULES:
    - LATEST COMPLETE FINANCIAL YEAR: 2024 (NOT 2025!)
    - For "current", "latest", "recent", or "now" queries:
    * Other tables: use calendarYear = 2024
    - For "10 years ago" calculations, use 2024 - 10 = 2014
    - NEVER use EXTRACT(year FROM CURRENT_DATE) for financial data - always use explicit years
    - Default period is 'FY' (full year) unless specified otherwise
    - Available years typically range from 2014-2024

    🚨 CRITICAL "LAST X YEARS" INTERPRETATION RULES:
    When user asks for "improvement over last X years" or "growth in last X years":
    
    ❌ WRONG INTERPRETATION: "Last 5 years" = 5 data points (2020-2024) = 4 years of growth
    ✅ CORRECT INTERPRETATION: "Last 5 years" = 5 years of growth period = 6 data points (2019-2024)
    
    🎯 CORRECT MAPPING FOR "LAST X YEARS":
    - "Last 5 years" → calendarYear BETWEEN 2019 AND 2024 (6 data points, 5 years of growth)
    - "Last 10 years" → calendarYear BETWEEN 2014 AND 2024 (11 data points, 10 years of growth)  
    - "Last 3 years" → calendarYear BETWEEN 2021 AND 2024 (4 data points, 3 years of growth)
    
    📊 CAGR CALCULATION VALIDATION:
    - COUNT(DISTINCT calendarYear) = X + 1 (where X is the years of growth requested)
    - CAGR periods = COUNT(DISTINCT calendarYear) - 1 = X (the actual growth years)
    - For "5 years of growth": Need 6 data points, CAGR over 5 periods
    
    ✅ EXAMPLES:
    - "Companies with revenue growth over last 5 years" → WHERE calendarYear BETWEEN 2019 AND 2024, HAVING COUNT(DISTINCT calendarYear) = 6
    - "Gross margin improvement in last 3 years" → WHERE calendarYear BETWEEN 2021 AND 2024, HAVING COUNT(DISTINCT calendarYear) = 4
    - "10-year revenue CAGR" → WHERE calendarYear BETWEEN 2014 AND 2024, HAVING COUNT(DISTINCT calendarYear) = 11
    
    🚨 CRITICAL: When user says "last X years", they mean X YEARS OF GROWTH, not X data points!
    
    🚨 MANDATORY TIME PERIOD VALIDATION FOR CAGR QUERIES:
    When calculating CAGR for a specific time period (e.g., "last 10 years"), you MUST enforce that companies have data for the ENTIRE period:
    
    ✅ CORRECT PATTERN FOR 10-YEAR CAGR:
    ```sql
    WITH revenue_cagr AS (
      SELECT
        cp.companyName,
        cp.symbol,
        cp.sector,
        cp.industry,
        MIN(CASE WHEN inc.calendarYear = 2014 THEN inc.revenue END) AS revenue_2014,
        MAX(CASE WHEN inc.calendarYear = 2024 THEN inc.revenue END) AS revenue_2024,
        COUNT(DISTINCT inc.calendarYear) AS years_with_data
      FROM financial_data.company_profiles cp
      JOIN financial_data.income_statements inc ON cp.symbol = inc.symbol
      WHERE inc.calendarYear BETWEEN 2014 AND 2024
        AND cp.sector IN ('Consumer Cyclical', 'Consumer Defensive')
      GROUP BY cp.companyName, cp.symbol, cp.sector, cp.industry
      HAVING years_with_data = 11  -- MUST have all 11 years (2014-2024)
        AND revenue_2014 > 0 
        AND revenue_2024 > 0
    )
    SELECT
      companyName,
      symbol,
      sector,
      industry,
      revenue_2014,
      revenue_2024,
      ROUND(((revenue_2024 / revenue_2014) ^ (1.0 / 10) - 1) * 100, 2) AS revenue_cagr_10yr
    FROM revenue_cagr
    HAVING ROUND(((revenue_2024 / revenue_2014) ^ (1.0 / 10) - 1) * 100, 2) > 20
    ORDER BY revenue_cagr_10yr DESC;
    ```
    
    🚨 CRITICAL ENFORCEMENT RULES:
    1. ALWAYS use HAVING COUNT(DISTINCT calendarYear) = (X + 1) where X is the years of growth
    2. For "10 years": HAVING COUNT(DISTINCT calendarYear) = 11
    3. For "5 years": HAVING COUNT(DISTINCT calendarYear) = 6
    4. For "3 years": HAVING COUNT(DISTINCT calendarYear) = 4
    5. NEVER include companies with incomplete data periods
    6. ALWAYS validate that start and end values exist and are positive
    7. NEVER use WHERE with column aliases in outer queries - use HAVING and repeat the full expression

    MANDATORY POSTGRESQL SYNTAX RULES:
    ❌ FORBIDDEN: Using wrong column names (calendarYear vs fiscalYear)
    ❌ FORBIDDEN: Missing schema prefix for tables

    ✅ REQUIRED: (column ILIKE '%pattern1%' OR column ILIKE '%pattern2%')
    ✅ REQUIRED: column IN ('value1', 'value2', 'value3')
    ✅ REQUIRED: ALWAYS use financial_data.table_name for all tables
    ✅ REQUIRED: Correct column names per table

    🚨 CRITICAL GROUP BY RULES (MOST IMPORTANT):
    ❌ NEVER mix aggregate and non-aggregate columns without proper GROUP BY!
    ❌ PROBLEMATIC: SELECT col1, col2, MIN(col3) FROM table; (Missing GROUP BY)
    
    ✅ OPTION 1 - WITH GROUP BY (RECOMMENDED):
    ✅   SELECT col1, col2, MIN(col3) FROM table GROUP BY col1, col2;
    
    ✅ OPTION 2 - WITH DISTINCT ON (PostgreSQL-specific):
    ✅   SELECT DISTINCT ON (group_key) col1, col2, MIN(col3) FROM table ORDER BY group_key;
    
    ✅ OPTION 3 - ALL AGGREGATE:
    ✅   SELECT MIN(col1), MAX(col2), AVG(col3) FROM table;
    
    🚨 CRITICAL: Choose ONE approach consistently throughout the query!

    🚨 CRITICAL WHERE CLAUSE RULES:
    - NEVER use aggregate functions (MAX, MIN, SUM, COUNT, AVG) in WHERE clauses or JOIN conditions!
    - ❌ WRONG: WHERE fr.date = (SELECT MAX(date) FROM financial_ratios_ttm)
    - ❌ WRONG: AND inc.fiscalYear = (SELECT MAX(fiscalYear) FROM income_statements_ttm)
    - ❌ WRONG: JOIN financial_ratios_ttm fr ON fr.date = (SELECT MAX(date) FROM financial_ratios_ttm)
    
    ✅ CORRECT APPROACHES for getting latest data:
    
    ✅ APPROACH 1 - Use explicit year/date values (RECOMMENDED):
    ✅   WHERE inc.fiscalYear = 2024  -- Use the actual latest year
    ✅   AND fr.date = '2024-12-31'   -- Use the actual latest date
    
    ✅ APPROACH 2 - Use window functions in subqueries:
    ✅   JOIN (
    ✅     SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
    ✅     FROM financial_ratios_ttm
    ✅   ) fr ON cp.symbol = fr.symbol AND fr.rn = 1
    
    ✅ APPROACH 3 - Use CTEs for latest data:
    ✅   WITH latest_ratios AS (
    ✅     SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) as rn
    ✅     FROM financial_ratios_ttm
    ✅   )
    ✅   SELECT ... FROM financial_data.company_profiles cp
    ✅   JOIN latest_ratios fr ON cp.symbol = fr.symbol AND fr.rn = 1
    
    🚨 CRITICAL: The latest fiscal year for TTM data is typically 2024
    🚨 CRITICAL: The latest date for financial ratios is typically '2024-12-31'
    🚨 CRITICAL: Always use explicit values instead of subqueries with aggregates in WHERE/JOIN!

    TABLE-SPECIFIC REQUIREMENTS:

    


    TTM tables (income_statements_ttm, balance_sheets_ttm, cash_flow_statements_ttm, financial_ratios_ttm):
    - Use 'fiscalYear' (NOT calendarYear) - CRITICAL!
    - These are trailing twelve months financial statements
    - Column structure: fiscalYear, period, date
    - DEFAULT: Use TTM tables for financial metrics unless a specific timeline is mentioned in the query
    - 🚨 CRITICAL: When using TTM tables, ALWAYS alias output columns with a 'ttm_' prefix (or '_ttm' suffix if more natural) to clearly indicate TTM values. For example: 'SUM(revenue) AS ttm_revenue' or 'SUM(revenue) AS revenue_ttm'. This applies to all financial metrics (revenue, netIncome, totalAssets, etc.) that are calculated or selected from TTM tables. This ensures users can easily distinguish TTM values from annual or quarterly values in the output.

    🚨 CRITICAL TTM (TRAILING TWELVE MONTHS) RULES:

📊 TTM DATA STRUCTURE (5 QUARTERS):
- TTM data includes 5 quarters: current quarter + 4 quarters back

🚨 MANDATORY COLUMN NAMING RULES:
- ALWAYS include time period information in column names for clarity
- For TTM data: Use "ttm_" prefix (e.g., "ttm_revenue", "ttm_netIncome", "ttm_totalAssets")
- For historical data: Include year/period info (e.g., "revenue_2024", "netIncome_2023", "totalAssets_2024")
- For CAGR calculations: Include time period (e.g., "revenue_cagr_10yr", "profit_growth_5yr")
- NEVER use generic column names without time context
- The "years_with_data" column is irrelevant and should be removed from final output
- Example: Q2 2025, Q1 2025, Q4 2024, Q3 2024, Q2 2024
- This allows for YoY quarter comparison (Q2 2025 vs Q2 2024)

🎯 TWO DIFFERENT USE CASES:

1️⃣ HISTORICAL COMPARISON (CAGR, YoY Growth):
- For CAGR calculations, YoY growth analysis, or historical trends
- Use 2024 as the latest year for comparison
- Example: "companies with 20% CAGR" → compare 2024 vs 2023 vs 2022
- Pattern: WHERE fiscalYear IN (2022, 2023, 2024) ORDER BY fiscalYear

2️⃣ TTM CURRENT PERFORMANCE:
- For current TTM metrics (revenue, net income, etc.)
- Use the most recent 4 quarters for each company
- For P&L and cashflow: SUM over the latest 4 quarters
- For balance sheet: use the latest quarter only
- Pattern: Use window function (ROW_NUMBER) to select latest 4 quarters

✅ CORRECT PATTERNS:

✅ Historical Comparison (CAGR/YoY):
  SELECT symbol, 
         revenue_2024, revenue_2023, revenue_2022,
         ((revenue_2024 / revenue_2022) ^ (1/2) - 1) * 100 AS cagr_percent
  FROM (
    SELECT symbol,
           MAX(CASE WHEN fiscalYear = 2024 THEN revenue END) AS revenue_2024,
           MAX(CASE WHEN fiscalYear = 2023 THEN revenue END) AS revenue_2023,
           MAX(CASE WHEN fiscalYear = 2022 THEN revenue END) AS revenue_2022
    FROM income_statements_ttm
    WHERE fiscalYear IN (2022, 2023, 2024)
    GROUP BY symbol
  )
  WHERE revenue_2024 > 0 AND revenue_2022 > 0

✅ TTM Current Performance (Latest 4 Quarters):
  WITH latest_quarters AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY fiscalYear DESC, 
      CASE period WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 ELSE 0 END DESC) AS rn
    FROM income_statements_ttm
  )
  SELECT symbol, SUM(revenue) AS ttm_revenue, SUM(netIncome) AS ttm_netIncome
  FROM latest_quarters WHERE rn <= 4 GROUP BY symbol

✅ YoY Quarter Comparison (Latest vs Previous Year Same Quarter):
  WITH current_quarter AS (
    SELECT symbol, revenue, fiscalYear, period
    FROM income_statements_ttm
    WHERE fiscalYear = 2025 AND period = 'Q2'  -- Latest quarter
  ),
  previous_year_quarter AS (
    SELECT symbol, revenue, fiscalYear, period
    FROM income_statements_ttm
    WHERE fiscalYear = 2024 AND period = 'Q2'  -- Same quarter previous year
  )
  SELECT c.symbol, 
         c.revenue AS current_revenue,
         p.revenue AS previous_revenue,
         ((c.revenue - p.revenue) / p.revenue) * 100 AS yoy_growth_percent
  FROM current_quarter c
  JOIN previous_year_quarter p ON c.symbol = p.symbol
  WHERE p.revenue > 0

🚨 CRITICAL: Choose the right pattern based on user intent!
- "CAGR", "growth", "trend" → Use Historical Comparison (2024 as latest)
- "TTM", "current", "latest" → Use TTM Current Performance (latest 4 quarters)
- "YoY quarter" → Use YoY Quarter Comparison (current vs previous year same quarter)

🚨 CRITICAL: financial_ratios_ttm table has NO date column!
- financial_ratios_ttm table only has: symbol + TTM ratio columns (grossProfitMarginTTM, etc.)
- ❌ WRONG: ORDER BY date DESC (no date column exists)
- ✅ CORRECT: Use financial_ratios_ttm directly without window functions
- ✅ CORRECT: JOIN financial_ratios_ttm fr ON cp.symbol = fr.symbol (no date ordering needed)

                    Regular financial tables (income_statements, balance_sheets, cash_flow_statements, key_metrics, financial_ratios):
- Use 'calendarYear' (NOT fiscalYear) for: income_statements, balance_sheets, cash_flow_statements, key_metrics, financial_ratios

    - Use these tables only if the query explicitly mentions a specific year, annual data, or a historical period

    QUERY REQUIREMENTS:
    - Use standard SQL syntax compatible with PostgreSQL
    - ALWAYS prefix table names with 'financial_data.' schema (e.g., financial_data.company_profiles)
    - Include company context (name, symbol, sector, industry) when appropriate
    - Apply proper filtering based on the parsed intent
    - Sort results meaningfully (typically by year ASC for time series)
    - Use proper PostgreSQL date/time functions
    - Follow the syntax patterns shown in the guide above
    - For pattern matching, use ILIKE for case-insensitive searches
    
    🚨 CRITICAL FINANCIAL METRIC ACCURACY:
    - Pay EXTREME attention to financial metric names and their exact meanings
    - ROIC ≠ ROCE: These are completely different metrics with different calculations
    - "Return on equity" ≠ "Return on assets" ≠ "Return on capital" - verify exact metric requested
    - "Profit margin" could be gross, operating, or net - check user context carefully
    - "Debt ratio" could be debt-to-equity, debt-to-assets, or total debt ratio - verify exact metric
    - When in doubt about metric meaning, use the most commonly understood interpretation
    - ALWAYS verify you're using the correct table and column for the exact metric requested
    
    🚨 CRITICAL OUTPUT FORMAT RULE:
    - Return ONLY the SQL query - NO comments, NO explanations, NO markdown formatting
    - Do NOT include lines starting with "- " or "-- " 
    - Do NOT include explanatory text after the SQL
    - Do NOT include "This query..." or "The query..." explanations
    - Return ONLY the pure SQL statement that can be executed directly
    - Example of what NOT to include:
      ❌ - Selects company name, symbol, sector, industry, peRatio, cash, revenue, debtToEquity, and roce (using roic as ROCE).
      ❌ - Joins to company_profiles for company name, symbol, sector, and industry.
      ❌ - Uses correct column names and joins per schema.
    - Example of what TO return:
      ✅ SELECT cp.companyName, cp.symbol, cp.sector, cp.industry, km.priceEarningsRatio FROM financial_data.company_profiles cp JOIN financial_data.key_metrics km ON cp.symbol = km.symbol WHERE km.priceEarningsRatio > 0 AND km.priceEarningsRatio < 20 ORDER BY km.priceEarningsRatio ASC;
      

    GROWTH RATE CALCULATIONS (UPDATED):
    - Use CAGR (Compound Annual Growth Rate) ONLY if there are 2 or more years of data (multi-year analysis)
    - For single year or less than 2 years: Use simple annual growth rate ((Current - Previous) / Previous)
    - If no specific timeline is mentioned, assume multi-year and use CAGR
    - CAGR Formula: POWER((end_value / start_value), (1.0 / years)) - 1
    - For multi-year data: Calculate CAGR between first and last available years
    - Include CAGR calculation as additional column: 
      * "revenue_cagr" for revenue growth
      * "profit_cagr" for net income growth
      * "growth_cagr" for general growth metrics
    - Format CAGR as percentage in result
    - Add note in column alias like: "Revenue CAGR (%)" 
    - Communicate to user that CAGR is the default growth calculation method for multi-year queries
    - For single-year queries, communicate that annual growth is used

    CAGR CALCULATION EXAMPLES:
    -- Revenue CAGR over available years
    SELECT 
        cp.companyName,
        cp.symbol,
        MIN(inc.revenue) as start_revenue,
        MAX(inc.revenue) as end_revenue,
        COUNT(DISTINCT inc.calendarYear) - 1 as years,
        (POWER((MAX(inc.revenue) / MIN(inc.revenue)), (1.0 / (COUNT(DISTINCT inc.calendarYear) - 1))) - 1) * 100 as "Revenue CAGR (%)"
    FROM financial_data.company_profiles cp
    JOIN financial_data.income_statements inc ON cp.symbol = inc.symbol
    WHERE inc.calendarYear BETWEEN 2015 AND 2024
    GROUP BY cp.companyName, cp.symbol;

    -- Annual growth example (single year)
    SELECT 
        cp.companyName,
        cp.symbol,
        inc1.revenue as previous_revenue,
        inc2.revenue as current_revenue,
        ((inc2.revenue - inc1.revenue) / inc1.revenue) * 100 as "Annual Revenue Growth (%)"
    FROM financial_data.company_profiles cp
    JOIN financial_data.income_statements inc1 ON cp.symbol = inc1.symbol AND inc1.calendarYear = 2023
    JOIN financial_data.income_statements inc2 ON cp.symbol = inc2.symbol AND inc2.calendarYear = 2024;

    🚨 CRITICAL TIMELINE/PERIOD COLUMN REQUIREMENTS (HIGHEST PRIORITY):
    
    🎯 MANDATORY: ALWAYS include timeline/period columns when available in the query results!
    🎯 Users need to understand WHEN the data is from for proper analysis and decision-making!
    🎯 Timeline context is essential for financial data interpretation!
    
    📊 REQUIRED TIMELINE COLUMNS BY TABLE:
    
    ✅ FINANCIAL STATEMENTS (income_statements, balance_sheets, cash_flow_statements):
    ✅   ALWAYS include: calendarYear, period, date
    ✅   Example: SELECT cp.companyName, cp.symbol, inc.revenue, inc.calendarYear, inc.period, inc.date
    
    ✅ TTM TABLES (income_statements_ttm, balance_sheets_ttm, cash_flow_statements_ttm):
    ✅   ALWAYS include: fiscalYear, period, date
    ✅   Example: SELECT cp.companyName, cp.symbol, inc.revenue, inc.fiscalYear, inc.period, inc.date
    
    ✅ KEY_METRICS:
    ✅   ALWAYS include: period, date
    ✅   Example: SELECT cp.companyName, cp.symbol, km.priceEarningsRatio, km.period, km.date
    

    
    ✅ GROWTH TABLES (income_statements_growth_rates, balance_sheets_growth_rates, etc.):
    ✅   ALWAYS include: latest_fiscal_year, latest_period, latest_date, previous_fiscal_year, previous_period, previous_date
    ✅   Example: SELECT cp.companyName, cp.symbol, igr.revenue_yoy_growth_pct, 
    ✅            igr.latest_fiscal_year, igr.latest_period, igr.latest_date,
    ✅            igr.previous_fiscal_year, igr.previous_period, igr.previous_date
    
    ✅ HISTORICAL_MARKET_CAP:
    ✅   ALWAYS include: date
    ✅   Example: SELECT cp.companyName, cp.symbol, hmc.marketCap, hmc.date
    
    ✅ FINANCIAL_RATIOS:
    ✅   ALWAYS include: period, date
    ✅   Example: SELECT cp.companyName, cp.symbol, fr.returnOnEquity, fr.period, fr.date
    
    ✅ KEY_METRICS:
    ✅   ALWAYS include: period, date
    ✅   Example: SELECT cp.companyName, cp.symbol, km.revenuePerShare, km.period, km.date
    
    🚨 CRITICAL RULES:
    🚨 1. NEVER omit timeline columns - they are MANDATORY for data context!
    🚨 2. Users need to know WHEN the financial data is from!
    🚨 3. Timeline columns help users understand data freshness and relevance!
    🚨 4. For historical analysis, timeline columns are essential for trend identification!
    🚨 5. For current data queries, timeline columns show data recency!
    
    ✅ CORRECT PATTERNS WITH TIMELINE COLUMNS:
    
    ✅ "Top 10 companies by revenue":
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          inc.revenue, inc.calendarYear, inc.period, inc.date
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN financial_data.income_statements inc ON cp.symbol = inc.symbol
    ✅   WHERE inc.calendarYear = 2024 AND inc.period = 'FY'
    ✅   ORDER BY inc.revenue DESC
    ✅   LIMIT 10;
    
    ✅ "Companies with high ROE":
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          fr.returnOnEquity, fr.period, fr.date
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN financial_ratios fr ON cp.symbol = fr.symbol
    ✅   WHERE fr.returnOnEquity > 0.15
    ✅   ORDER BY fr.returnOnEquity DESC;
    
    ✅ "Revenue growth analysis":
    ✅   SELECT cp.companyName, cp.symbol, cp.sector, cp.industry,
    ✅          igr.revenue_yoy_growth_pct, igr.revenue_current, igr.revenue_previous,
    ✅          igr.latest_fiscal_year, igr.latest_period, igr.latest_date,
    ✅          igr.previous_fiscal_year, igr.previous_period, igr.previous_date
    ✅   FROM financial_data.company_profiles cp
    ✅   JOIN financial_data.income_statements_growth_rates igr ON cp.symbol = igr.symbol
    ✅   WHERE igr.revenue_yoy_growth_pct > 0
    ✅   ORDER BY igr.revenue_yoy_growth_pct DESC;
    
    ❌ WRONG PATTERNS (MISSING TIMELINE COLUMNS):
    ❌   SELECT cp.companyName, cp.symbol, inc.revenue  -- Missing calendarYear, period, date!
    ❌   SELECT cp.companyName, cp.symbol, cm.priceEarningsRatio  -- Missing period, date!
    
    🎯 TIMELINE COLUMN BENEFITS:
    🎯 - Users can verify data freshness and relevance
    🎯 - Users can understand if data is current or historical
    🎯 - Users can identify trends and patterns over time
    🎯 - Users can make informed decisions based on data timeliness
    🎯 - Users can cross-reference with market events and economic conditions
    
    TIME PERIODS: {time_periods}
    REQUIRED CALCULATIONS: {required_calculations}

    EXAMPLES OF CORRECT PATTERNS:

    🚨 FINAL OUTPUT REQUIREMENTS:
    - Return ONLY the SQL query - NO comments, explanations, or markdown formatting!
    - Return ONLY executable SQL that can be run directly in PostgreSQL
    - Do NOT include any reasoning text or explanations
    - Do NOT include "This query..." or "The query..." text
    - Start directly with SELECT, WITH, or other SQL keywords

    🚨 CRITICAL FINANCIAL METRIC ACCURACY:
    - Pay EXTREME attention to financial metric names and their exact meanings
    - ROIC ≠ ROCE: These are completely different metrics with different calculations
    - "Return on equity" ≠ "Return on assets" ≠ "Return on capital" - verify exact metric requested
    - "Profit margin" could be gross, operating, or net - check user context carefully
    - "Debt ratio" could be debt-to-equity, debt-to-assets, or total debt ratio - verify exact metric
    - When in doubt about metric meaning, use the most commonly understood interpretation
    - ALWAYS verify you're using the correct table and column for the exact metric requested

    🚨 OUTPUT FORMAT EXAMPLE:
    ✅ CORRECT:
    SELECT cp.companyName, cp.symbol, km.priceEarningsRatio 
    FROM financial_data.company_profiles cp 
    JOIN financial_data.key_metrics km ON cp.symbol = km.symbol 
    WHERE km.priceEarningsRatio > 0 AND km.priceEarningsRatio < 20 
    ORDER BY km.priceEarningsRatio ASC;

    ❌ WRONG:
    This query selects company name, symbol, and PE ratio...
    SELECT cp.companyName, cp.symbol, cm.priceEarningsRatio...
"""

        try:
            prompt_template = ChatPromptTemplate.from_template(base_prompt)
            llm = self.get_agent_llm('sql_generation')
            formatted_prompt = prompt_template.format(
                question=question,
                parsed_intent=json.dumps(parsed_intent, indent=2),
                focused_schema=focused_schema,
                filter_instructions=filter_instructions,
                negative_value_filtering=negative_value_filtering,
                broad_category_filtering=broad_category_filtering,
                table_selection_guidance=table_selection_guidance,
                column_name_guidance=column_name_guidance,
                growth_table_guidance=growth_table_guidance,
                error_context=error_context,
                time_periods=parsed_intent.get('time_periods', ['latest']),
                required_calculations=json.dumps(parsed_intent.get('required_calculations', []), indent=2),
                margin_calculation_guidance=margin_calculation_guidance,
                limit_guidance=limit_guidance
            )
            
            # Log prompt length for SQL generation
            prompt_length = len(formatted_prompt)
            log_message(f"\nPROMPT LENGTH FOR SQL GENERATION:\n{prompt_length}\n")
            
            response = llm.invoke(formatted_prompt)
            sql_query = response.content if hasattr(response, 'content') else str(response)
            log_message(f"LLM Generated PostgreSQL SQL (attempt {attempt}): {sql_query}")
            
            # Debug: Check if broad category filtering was applied
            if parsed_intent.get('use_semantic_post_processing', False) and parsed_intent.get('broad_categories'):
                broad_categories = parsed_intent.get('broad_categories', [])
                log_message(f"🔍 DEBUG: Expected broad categories: {broad_categories}")
                log_message(f"🔍 DEBUG: Generated SQL contains sector filter: {'sector' in sql_query.lower()}")
                log_message(f"🔍 DEBUG: Generated SQL contains industry filter: {'industry' in sql_query.lower()}")
                log_message(f"🔍 DEBUG: Generated SQL contains description filter: {'description' in sql_query.lower()}")
            
            return sql_query
        except Exception as e:
            log_message(f"SQL generation LLM invocation error on attempt {attempt}: {e}")
            return ""


    def _apply_semantic_post_processing(self, df: pd.DataFrame, question: str, parsed_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply AI post-processing to filter results using LLM with parallel processing
        
        Args:
            df: DataFrame with initial results
            question: Original user question
            parsed_intent: Parsed intent including AI processing configuration
            
        Returns:
            Dictionary with filtered results and metadata
        """
        max_companies_to_output = parsed_intent.get('max_companies_to_output', 500)
        # For semantic post-processing, use 500 companies to process, otherwise use user's limit
        if parsed_intent.get('use_semantic_post_processing', False):
            max_companies_to_process = 500
        else:
            max_companies_to_process = max_companies_to_output
        
        log_message(f"🧠 Starting AI post-processing")
        log_message(f"   📊 Input: {len(df)} companies")
        log_message(f"   🎯 Target output: {max_companies_to_output} companies")
        log_message(f"   📝 Max to process: {max_companies_to_process} companies")
        
        if df.empty:
            return {
                'filtered_df': df,
                'total_processed': 0,
                'total_selected': 0,
                'selection_rate': 0.0,
                'processing_batches': 0
            }
        
        # For post-processing, we want to process more companies than the target output
        # If we have fewer companies than max_companies_to_process, we need to get more from the database
        if len(df) < max_companies_to_process:
            log_message(f"   📊 Need more candidates for post-processing. Current: {len(df)}, Target: {max_companies_to_process}")
            # We'll need to modify the SQL to get more candidates
            # For now, process what we have, but ideally we'd regenerate SQL with higher LIMIT
            df_to_process = df
        else:
            # Limit input to max_companies_to_process
            df_to_process = df.head(max_companies_to_process)
        
        # Extract symbols for description lookup
        symbols = df_to_process['symbol'].tolist() if 'symbol' in df_to_process.columns else []
        
        # Get company descriptions
        descriptions = self._get_company_descriptions(symbols)
        
        # Process in batches of 20 with parallel processing
        batch_size = 20
        selected_rows = []
        total_batches = 0
        
        # Create batches for parallel processing
        batches = []
        for i in range(0, len(df_to_process), batch_size):
            batch_df = df_to_process.iloc[i:i+batch_size]
            batches.append((batch_df, i // batch_size + 1))
            total_batches += 1
        
        log_message(f"   🚀 Processing {total_batches} batches in parallel")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {
                executor.submit(self._semantic_filter_batch, batch_df, question,descriptions, parsed_intent): batch_num
                for batch_df, batch_num in batches
            }
            
            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_results = future.result()
                    selected_rows.extend(batch_results)
                    completed_batches += 1
                    log_message(f"   ✅ Completed batch {batch_num} ({len(batch_results)} companies selected)")
                    
                    # Stop early if we have the requested number of companies
                    if len(selected_rows) >= max_companies_to_output:
                        log_message(f"   🎯 Early termination: {len(selected_rows)} companies selected (target: {max_companies_to_output})")
                        # Cancel remaining futures
                        for remaining_future in future_to_batch:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                        
                except Exception as e:
                    batch_num = future_to_batch[future]
                    log_message(f"❌ Error processing batch {batch_num}: {e}")
                    # Continue with other batches
        
        # Convert selected rows back to DataFrame
        if selected_rows:
            # Sort by confidence score (highest first) and limit to max_companies_to_output
            selected_rows.sort(key=lambda x: x['confidence'], reverse=True)
            selected_rows = selected_rows[:max_companies_to_output]
            
            # Extract row indices from selected results
            selected_indices = [result['original_index'] for result in selected_rows]
            filtered_df = df_to_process.iloc[selected_indices].copy()
            
            # Add confidence scores and reasoning as new columns
            confidence_scores = [result['confidence'] for result in selected_rows]
            reasoning_texts = [result['reasoning'] for result in selected_rows]
            filtered_df['relevance'] = confidence_scores
            filtered_df['reasoning'] = reasoning_texts
            
            log_message(f"   🏆 Top confidence scores: {[f'{c:.2f}' for c in confidence_scores[:5]]}")
        else:
            filtered_df = pd.DataFrame()
            log_message(f"⚠️ No companies selected - this might indicate an issue with the query or data")
        
        selection_rate = len(selected_rows) / len(df_to_process) if len(df_to_process) > 0 else 0.0
        
        log_message(f"✅ AI post-processing complete:")
        log_message(f"   📊 Processed: {len(df_to_process)} companies in {total_batches} batches")
        log_message(f"   🎯 Selected: {len(selected_rows)} companies")
        log_message(f"   📈 Selection rate: {selection_rate:.1%}")
        
        return {
            'filtered_df': filtered_df,
            'total_processed': len(df_to_process),
            'total_selected': len(selected_rows),
            'selection_rate': selection_rate,
            'processing_batches': total_batches
        }

    def _semantic_filter_batch(self, batch_df: pd.DataFrame, question: str, 
                              descriptions: Dict[str, str], 
                              parsed_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply AI filtering to a batch of companies
        
        Args:
            batch_df: DataFrame batch to process
            question: Original question
            descriptions: Company descriptions lookup
            parsed_intent: Parsed intent for context
            
        Returns:
            List of dictionaries with selected companies and confidence scores
        """
        if batch_df.empty:
            return []
        
                 # Prepare company data for LLM analysis
        company_data = []
        for i, (idx, row) in enumerate(batch_df.iterrows()):
            symbol = row.get('symbol', 'Unknown')
            company_name = row.get('companyName', 'Unknown')
            sector = row.get('sector', 'Unknown')
            industry = row.get('industry', 'Unknown')
            description = descriptions.get(symbol, 'No description available')
            
            company_info = {
                'symbol': symbol,
                'company_name': company_name,
                'sector': sector,
                'industry': industry,
                'description': description,
                'original_index': idx,  # Use the actual DataFrame index
                'batch_index': i  # Index within this batch for LLM reference
            }
            company_data.append(company_info)
        
        # Create AI filtering prompt based on type
        prompt = self._create_semantic_filtering_prompt(
            question,company_data, parsed_intent
        )
        
        # Debug: Log what companies are being analyzed
        log_message(f"🔍 Analyzing batch with {len(company_data)} companies:")
        for i, company in enumerate(company_data[:3]):  # Log first 3 companies
            log_message(f"   {i+1}. {company['company_name']} ({company['symbol']}) - {company['sector']}/{company['industry']}")
        if len(company_data) > 3:
            log_message(f"   ... and {len(company_data)-3} more companies")
        
        # Retry logic with exponential backoff
        max_retries = 2
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Use the SQL generation LLM for AI filtering
                llm = self.get_agent_llm('sql_generation')
                # Log prompt length for semantic filtering
                prompt_length = len(prompt)
                log_message(f"\nPROMPT LENGTH FOR SEMANTIC FILTERING:\n{prompt_length}\n")
                
                response = llm.invoke(prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Parse the response
                selected_companies = self._parse_semantic_filtering_response(response_text, company_data)
                
                if selected_companies:
                    log_message(f"✅ AI filtering successful on attempt {attempt + 1}")
                    return selected_companies
                else:
                    log_message(f"⚠️ AI filtering returned no results on attempt {attempt + 1}")
                    if attempt < max_retries:
                        log_message(f"🔄 Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay = min(retry_delay * 1.5, 5)  # Exponential backoff, max 5 seconds
                    else:
                        log_message(f"❌ All retry attempts failed")
                        
            except Exception as e:
                log_message(f"❌ AI filtering error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    log_message(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 5)  # Exponential backoff, max 5 seconds
                else:
                    log_message(f"❌ All retry attempts failed")
        
        # If all attempts failed, return empty list - no fallback companies
        log_message(f"🔄 All AI filtering attempts failed - returning no companies")
        return []

    def _create_semantic_filtering_prompt(self, question: str,
                                        company_data: List[Dict], parsed_intent: Dict[str, Any]) -> str:
        """
        Create the prompt for AI filtering
        """
        base_prompt = f"""You are an expert financial analyst with deep knowledge of business classification and industry analysis. Your task is to intelligently analyze companies and determine which ones truly match the user's specific request.

ORIGINAL QUESTION: {question}
PARSED INTENT: {json.dumps(parsed_intent.get('description_keywords', []), indent=2)}

CRITICAL: The user is asking for: "{question}"
You must filter companies based on the FULL question, not just parts of it.

COMPANIES TO ANALYZE:
"""
        
        for company in company_data:
            base_prompt += f"""
{company['batch_index']}. {company['company_name']} ({company['symbol']})
   Sector: {company['sector']}
   Industry: {company['industry']}
   Description: {company['description'][:500]}...
"""
        
        base_prompt += f"""

ANALYSIS APPROACH:
Think like a sophisticated business analyst who understands the nuances of different industries and business models. Your goal is to identify companies that genuinely match the user's request, not just companies that have some tangential relationship.

KEY PRINCIPLES:
1. **Core Business Focus**: Look at what the company primarily does, not what they might use or sell to
2. **Industry Expertise**: Understand the difference between companies that ARE in an industry vs. companies that SERVE that industry
3. **Business Model Clarity**: Distinguish between primary revenue sources and secondary activities
4. **Context Matters**: Consider the full business description, not just keywords

ANALYSIS QUESTIONS TO ASK YOURSELF:
- What is this company's primary business model?
- Are they a direct participant in the requested industry, or do they just provide tools/services to that industry?
- Would an expert in the field consider this company part of the requested category?
- Is this company's core expertise and revenue primarily derived from the requested business type?

SPECIFIC GUIDANCE FOR BIOLOGICS COMPANIES:
When the user asks for "biologics companies", look for:
- Companies that develop, manufacture, or commercialize biologic drugs
- Companies with biologic drug pipelines or FDA-approved biologics
- Biopharmaceutical companies focused on large molecule therapeutics
- Companies developing monoclonal antibodies, therapeutic proteins, vaccines, cell therapies, gene therapies
- Companies with biologics manufacturing capabilities or biologic drug candidates
- AVOID: Companies that only provide services, tools, or equipment to biotech companies
- AVOID: Traditional pharmaceutical companies that only make small molecule drugs
- AVOID: Medical device companies that don't develop biologic drugs


EXAMPLES OF GOOD vs BAD CLASSIFICATION:
- "Biologics companies": 
  ✅ GOOD: Companies that develop/manufacture biologic drugs, therapeutic proteins, monoclonal antibodies, vaccines, cell therapies, gene therapies, recombinant proteins, biosimilars
  ✅ GOOD: Biopharmaceutical companies focused on biologic drug development
  ✅ GOOD: Companies with FDA-approved biologic drugs or biologics in clinical trials
  ❌ BAD: Semiconductor companies that make chips used in medical devices
  ❌ BAD: Software companies that provide tools to biotech companies
  ❌ BAD: Traditional pharmaceutical companies that only make small molecule drugs
  ❌ BAD: Medical device companies that don't develop biologic drugs
  ❌ BAD: Companies that only provide services to biotech companies

- "AI companies":
  ✅ GOOD: Companies whose core product is AI/ML software or services
  ❌ BAD: Traditional companies that use AI internally
  ❌ BAD: Hardware companies that make chips used in AI applications

- "Data center companies":
  ✅ GOOD: Companies that own/operate data centers, provide colocation, cloud hosting
  ❌ BAD: Software companies that run on data centers
  ❌ BAD: Hardware companies that sell equipment to data centers

- "Companies with cloud revenue":
  ✅ GOOD: Companies with significant cloud services revenue (AWS, Azure, Google Cloud, etc.)
  ✅ GOOD: Companies that provide cloud infrastructure or SaaS services
  ✅ GOOD: Companies with substantial cloud-related business operations
  ❌ BAD: Companies that only use cloud services internally
  ❌ BAD: Companies with minimal cloud revenue (<5% of total)

- "Companies with data center services":
  ✅ GOOD: Companies that provide colocation, hosting, or data center infrastructure
  ✅ GOOD: Companies with data center operations as a meaningful business line
  ❌ BAD: Companies that only use data centers for their own operations

CONFIDENCE SCORING:
- High (0.8-1.0): Perfect match - this company IS the requested business type
- Medium (0.6-0.79): Good match - significant operations in the requested area
- Low (0.4-0.59): Weak match - some involvement but not core business
- Exclude (0.0-0.39): Poor match - should not be included

IMPORTANT: Be reasonable in your assessment, BUT be strict about business type classification. 

For business type queries (like "biologics companies", "AI companies", etc.):
- ONLY include companies that ARE the requested business type
- Do NOT include companies that just use or serve that business type
- Be strict about the core business model


The key is to understand what the user is actually asking for and filter accordingly.

OUTPUT FORMAT (JSON only):
{{
    "selected_companies": [
        {{
            "index": 0,
            "symbol": "AAPL",
            "confidence": 0.85,
            "reasoning": "Clear explanation of why this company matches or doesn't match"
        }}
    ]
}}

Include companies with confidence >= 0.3 (lowered threshold). Be reasonable and inclusive rather than overly strict.
Return ONLY the JSON response, no additional text."""
        
        return base_prompt

    def _parse_semantic_filtering_response(self, response_text: str, company_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse the LLM response for semantic filtering
        """
        try:
            # Extract JSON from response
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                json_str = match.group(0)
                parsed_response = self._robust_json_parse(json_str)
                
                selected_companies = []
                for selection in parsed_response.get('selected_companies', []):
                    batch_idx = selection.get('index', 0)
                    confidence = selection.get('confidence', 0.5)
                    reasoning = selection.get('reasoning', 'No reasoning provided')
                    
                    # Find the company by batch index
                    matching_company = None
                    for company in company_data:
                        if company['batch_index'] == batch_idx:
                            matching_company = company
                            break
                    
                    if matching_company:
                        selected_companies.append({
                            'original_index': matching_company['original_index'],
                            'confidence': confidence,
                            'reasoning': reasoning
                        })
                
                return selected_companies
            else:
                log_message(f"❌ No JSON found in semantic filtering response")
                return []
                
        except Exception as e:
            log_message(f"❌ Error parsing semantic filtering response: {e}")
            return []






    def _detect_cagr_usage(self, sql_query: str, columns: List[str], question: str) -> bool:
        """Detect if CAGR calculation was used in the SQL query"""
        sql_lower = sql_query.lower()
        question_lower = question.lower()
        
        # Check for CAGR indicators in SQL
        cagr_indicators = [
            'cagr', 'compound annual growth rate', 'power(', 'exp(', 'ln(',
            'growth rate', 'annualized growth'
        ]
        
        # Check for CAGR indicators in question
        question_cagr_indicators = [
            'cagr', 'compound annual growth rate', 'annualized', 'growth rate over',
            'average annual growth', 'compounded growth'
        ]
        
        # Check SQL for CAGR patterns
        for indicator in cagr_indicators:
            if indicator in sql_lower:
                return True
        
        # Check question for CAGR patterns
        for indicator in question_cagr_indicators:
            if indicator in question_lower:
                return True
        
        return False

    def _enhance_result_message_with_cagr(self, base_message: str, sql_query: str, columns: List[str], question: str) -> str:
        """Enhance result message with CAGR detection and explanation"""
        # Check if CAGR was used in the query
        if self._detect_cagr_usage(sql_query, columns, question):
            cagr_explanation = "\n\n📈 **CAGR Analysis**: This analysis uses Compound Annual Growth Rate (CAGR) to measure growth over multiple periods, providing a smoothed annual growth rate that accounts for compounding effects."
            return base_message + cagr_explanation
        return base_message

    def _build_focused_schema(self, selected_tables: List[str], parsed_intent: Dict[str, Any]) -> str:
        """Enhanced schema builder using _get_enhanced_table_metadata for LLM-friendly format"""
        schema_parts = []
        
        for table_name in selected_tables:
            if table_name in self.table_schemas:
                # Get the enhanced metadata for this table
                enhanced_metadata = self._get_enhanced_table_metadata(table_name)
                
                if enhanced_metadata:
                    # Get actual columns from the database
                    actual_columns = self.table_schemas[table_name]['columns']
                    
                    schema_part = f"📊 TABLE: {table_name}\n"
                    schema_part += f"📝 Description: {enhanced_metadata.get('description', 'No description available')}\n"
                    
                    # Add actual columns in a clean, readable format
                    schema_part += f"🔢 Columns ({len(actual_columns)} total):\n"
                    # Group columns by type for better readability
                    key_cols = enhanced_metadata.get('key_columns', [])
                    filing_cols = enhanced_metadata.get('filing_columns', [])
                    
                    # Show key columns first
                    if key_cols:
                        schema_part += f"   🔑 Key columns: {', '.join(key_cols)}\n"
                    
                    # Show filing columns
                    if filing_cols:
                        schema_part += f"   📄 Filing columns: {', '.join(filing_cols)}\n"
                    
                    # Show all columns in a compact format
                    schema_part += f"   📋 All columns: {', '.join(actual_columns)}\n"
                    
                    # Add time series information if applicable
                    if enhanced_metadata.get('is_time_series'):
                        time_col = enhanced_metadata.get('time_column', 'calendarYear')
                        period_col = enhanced_metadata.get('period_column', 'period')
                        schema_part += f"⏰ Time series: YES\n"
                        schema_part += f"   📅 Time column: {time_col}\n"
                        schema_part += f"   📊 Period column: {period_col}\n"
                        
                        # Special handling for TTM tables
                        if 'ttm' in table_name.lower():
                            schema_part += f"   ⚠️ TTM TABLE RULES:\n"
                            schema_part += f"      • For TTM, use a window function (ROW_NUMBER) to select the latest 4 quarters for each company, ordered by fiscalYear DESC and period (Q4, Q3, Q2, Q1).\n"
                            schema_part += f"      • For P&L and cashflow, SUM over the latest 4 quarters. For balance sheet, use the latest quarter only.\n"
                            schema_part += f"      • If a specific year is requested, use that year instead.\n"
                            schema_part += f"      • Example for TTM P&L/cashflow SUM:\n        WITH latest_quarters AS (\n          SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY fiscalYear DESC, CASE period WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 ELSE 0 END DESC) AS rn\n          FROM {table_name}\n        )\n        SELECT symbol, SUM(revenue) AS revenue_ttm FROM latest_quarters WHERE rn <= 4 GROUP BY symbol\n"
                            schema_part += f"      • Example for TTM balance sheet (latest value):\n        WITH latest_quarter AS (\n          SELECT *, ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY fiscalYear DESC, CASE period WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 ELSE 0 END DESC) AS rn\n          FROM {table_name}\n        )\n        SELECT symbol, totalAssets FROM latest_quarter WHERE rn = 1\n"
                        else:
                            schema_part += f"    Default filters: {time_col} = <year>, {period_col} = 'FY'\n"
                    
                    # Add sorting information
                    if enhanced_metadata.get('primary_sort_column'):
                        sort_col = enhanced_metadata.get('primary_sort_column')
                        sort_dir = enhanced_metadata.get('sort_direction', 'DESC')
                        schema_part += f"📊 Default sort: {sort_col} {sort_dir}\n"
                    
                    # Add filter hints
                    if enhanced_metadata.get('filter_hints'):
                        schema_part += f"💡 Filter hints: {enhanced_metadata.get('filter_hints')}\n"
                    
                    # Add special notes
                    if enhanced_metadata.get('special_notes'):
                        schema_part += f"📌 Notes: {enhanced_metadata.get('special_notes')}\n"
                    
                    schema_parts.append(schema_part)
                else:
                    # Fallback to basic schema if enhanced metadata not available
                    all_columns = self.table_schemas[table_name]['columns']
                    description = self.table_schemas[table_name].get('description', 'No description available')
                    
                    schema_part = f"📊 TABLE: {table_name}\n"
                    schema_part += f"📝 Description: {description}\n"
                    schema_part += f"🔢 Columns: {', '.join(all_columns)}\n"
                    schema_parts.append(schema_part)
        
        schema_with_guidance = "\n\n".join(schema_parts) if schema_parts else "❌ No valid schemas available for selected tables."
        
        # Add general PostgreSQL syntax guidance
        schema_with_guidance += "\n\n" + self._get_postgres_syntax_guidance()
        
        return schema_with_guidance

    def _log_timing_details(self, timing_details: Dict[str, Any]):
        """Prints a formatted summary of the query timing details."""
        print("\n" + "=" * 70)
        print("⏱️  Query Timing Breakdown")
        print("-" * 70)
        
        total_duration = timing_details.get('total_duration_sec', 0)
        cache_hit = timing_details.get('cache_type_hit', 'N/A')
        
        print(f"  - {'Total Duration:':<35} {total_duration:.4f}s")
        print(f"  - {'Cache Status:':<35} {cache_hit}")
        
        if cache_hit != "pipeline_cache":
            print(f"  - {'Intent Parsing:':<35} {timing_details.get('intent_parsing_sec', 0):.4f}s")
            print(f"  - {'Table Selection:':<35} {timing_details.get('table_selection_sec', 0):.4f}s")
            
            if 'generation_attempts' in timing_details:
                for i, attempt in enumerate(timing_details['generation_attempts']):
                    print(f"  - {'SQL Generation (Attempt ' + str(i+1) + '):':<35} {attempt['duration_sec']:.4f}s")
            
            if 'execution_attempts' in timing_details:
                for i, attempt in enumerate(timing_details['execution_attempts']):
                    cached_str = " (CACHED)" if attempt.get('cached') else ""
                    print(f"  - {'SQL Execution (Attempt ' + str(i+1) + '):':<35} {attempt['duration_sec']:.4f}s{cached_str}")
            
            if 'formatting_and_pagination_sec' in timing_details:
                print(f"  - {'Formatting & Pagination:':<35} {timing_details.get('formatting_and_pagination_sec', 0):.4f}s")

        print("=" * 70 + "\n")

    def _get_postgres_syntax_guidance(self) -> str:
        """Provides comprehensive PostgreSQL syntax guidance for the LLM"""
        return """
    POSTGRESQL SQL SYNTAX GUIDE:

    🎯 CRITICAL: COMPANY-SPECIFIC FISCAL CALENDAR HANDLING
    
    ❌ WRONG APPROACH: Using global fiscal year/period filters
    ❌ NEVER USE: WHERE fiscalYear = 2024 AND period = 'Q3' (ignores company-specific fiscal calendars!)
    
    ✅ CORRECT APPROACH: Per-company latest quarter with window functions
    
    🎯 PERFECT TTM YOY PATTERN (HANDLES ALL FISCAL CALENDARS):
    WITH latest_quarters AS (
      SELECT *,
        ROW_NUMBER() OVER (
          PARTITION BY symbol 
          ORDER BY date DESC, calendaryear DESC, 
            CASE period WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 END DESC
        ) as rn
      FROM financial_data.income_statements
    ),
    current_data AS (
      SELECT * FROM latest_quarters WHERE rn = 1
    )
    SELECT 
      cp.companyname, cp.symbol, cp.sector, cp.industry, cp.cik,
      curr.netincome AS pat_latest_quarter,
      curr.calendaryear AS latest_calendaryear,
      curr.period AS latest_period,
      curr.date AS latest_quarter_end_date,
      curr.accepteddate AS latest_accepteddate,
      prev.netincome AS pat_prev_year_same_quarter,
      prev.calendaryear AS prev_calendaryear,
      prev.period AS prev_period,
      prev.date AS prev_quarter_end_date,
      prev.accepteddate AS prev_accepteddate,
      ((curr.netincome - prev.netincome) / NULLIF(prev.netincome, 0)) * 100 AS pat_yoy_growth_percent
    FROM financial_data.company_profiles cp
    JOIN current_data curr ON cp.symbol = curr.symbol
    JOIN financial_data.income_statements prev ON curr.symbol = prev.symbol
      AND curr.period = prev.period
      AND curr.calendaryear = prev.calendaryear + 1
    WHERE prev.netincome IS NOT NULL
      AND curr.netincome > prev.netincome
    ORDER BY pat_yoy_growth_percent DESC;
    
    🚨 KEY POINTS:
    1. ✅ Uses ROW_NUMBER() window function to find each company's latest quarter individually
    2. ✅ Orders by date DESC to handle different fiscal calendars properly
    3. ✅ Joins on period to compare same quarters year-over-year
    4. ✅ Uses calendaryear + 1 to get previous year's same quarter  
    5. ✅ Handles all fiscal year-ends (Oct, Dec, Jun, etc.) automatically
    
    🎯 CRITICAL POSTGRESQL SYNTAX RULES:
    ❌ FORBIDDEN: Missing schema prefix for tables
    ❌ FORBIDDEN: Using wrong column names (use lowercase: calendaryear, netincome, etc.)
    ❌ FORBIDDEN: CamelCase column names (PostgreSQL uses lowercase)
    
    ✅ REQUIRED: ALWAYS use financial_data.table_name for all tables
    ✅ REQUIRED: (column ILIKE '%pattern1%' OR column ILIKE '%pattern2%')
    ✅ REQUIRED: column IN ('value1', 'value2', 'value3')
    ✅ REQUIRED: Use lowercase column names: companyname, netincome, calendaryear, etc.
    
    🚨 CRITICAL GROUP BY RULES:
    ❌ NEVER mix aggregate and non-aggregate columns without proper GROUP BY!
    ❌ PROBLEMATIC: SELECT col1, col2, MIN(col3) FROM table; (Missing GROUP BY)
    
    ✅ OPTION 1 - WITH GROUP BY (RECOMMENDED):
    ✅   SELECT col1, col2, MIN(col3) FROM table GROUP BY col1, col2;
    
    ✅ OPTION 2 - WITH DISTINCT ON (PostgreSQL-specific):
    ✅   SELECT DISTINCT ON (col1) col1, col2, col3 FROM table ORDER BY col1, col3;
    
    ✅ OPTION 3 - ALL AGGREGATE:
    ✅   SELECT MIN(col1), MAX(col2), AVG(col3) FROM table;
    
         🚨 CRITICAL: Choose ONE approach consistently throughout the query!
     
     """

    def _get_company_descriptions(self, symbols: List[str]) -> Dict[str, str]:
        """
        Get company descriptions for the given symbols
        """
        descriptions = {}
        
        for symbol in symbols:
            try:
                # Query the company_profiles table for description using PostgreSQL
                query = "SELECT description FROM financial_data.company_profiles WHERE symbol = %s LIMIT 1"
                conn = get_postgres_connection()
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(query, (symbol,))
                        result = cursor.fetchone()
                        
                        if result and result[0]:
                            descriptions[symbol] = str(result[0])
                        else:
                            descriptions[symbol] = "No description available"
                finally:
                    conn.close()
                    
            except Exception as e:
                log_message(f"❌ Error getting description for {symbol}: {e}")
                descriptions[symbol] = "Error retrieving description"
        
        return descriptions


    def _format_value_json_safe(self, value: Any, column_name: str) -> str:
        """
        Format a value to be JSON-safe while preserving readability
        
        Args:
            value: The value to format
            column_name: The column name for context-aware formatting
            
        Returns:
            JSON-safe string representation of the value
        """
        if pd.isna(value) or value is None:
            return "-"
        
        # Handle numeric values
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                return "-"
            return str(value)
        
        # Handle datetime objects
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.strftime('%Y-%m-%d')
        
        # Handle boolean values
        if isinstance(value, bool):
            return str(value)
        
        # Handle everything else as string
        return str(value)



    def _validate_sql_for_postgres(self, sql_query: str) -> Tuple[bool, str]:
        """Validate SQL query for PostgreSQL compatibility and correct column names"""
        errors = []
        warnings = []
        
        # Ensure proper schema prefix for financial_data tables
        financial_tables = ['company_profiles', 'income_statements', 'balance_sheets', 'cash_flow_statements', 
                           'key_metrics', 'financial_ratios', 'historical_market_cap']
        
        for table in financial_tables:
            # Check if table is used without schema prefix
            pattern = rf'\b{table}\b(?!\s*\.)'  # Table name not followed by a dot
            if re.search(pattern, sql_query, re.IGNORECASE):
                # Replace table references with schema-prefixed versions
                sql_query = re.sub(rf'\b{table}\b', f'financial_data.{table}', sql_query, flags=re.IGNORECASE)
        
        # IMPROVED: Check for problematic GROUP BY with filing columns (more precise)
        if 'GROUP BY' in sql_query.upper():
            filing_columns = ['fillingDate', 'acceptedDate']
            
            # Extract the actual GROUP BY clause content
            try:
                # Split the query to isolate GROUP BY section
                upper_query = sql_query.upper()
                group_by_start = upper_query.find('GROUP BY') + len('GROUP BY')
                
                # Find the end of GROUP BY section (next major clause or end of query)
                group_by_end = len(sql_query)
                for clause in ['ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', ';']:
                    clause_pos = upper_query.find(clause, group_by_start)
                    if clause_pos != -1:
                        group_by_end = min(group_by_end, clause_pos)
                
                group_by_section = sql_query[group_by_start:group_by_end].strip()
                
                # Check for problematic patterns: bare filing column names in GROUP BY
                for filing_col in filing_columns:
                    # Look for patterns like ", fillingDate" or "fillingDate," (bare column references)
# But NOT patterns like "MIN(fillingDate)" or "MAX(fillingDate)" (aggregated references)
                    patterns = [
                        rf'\b{filing_col}\b(?!\s*\))',  # Column name not followed by closing paren (not in function)
                        rf',\s*{filing_col}\s*(?:,|$)',  # Column name between commas
                        rf'^\s*{filing_col}\s*(?:,|$)'   # Column name at start of GROUP BY
                    ]
                    
                    for pattern in patterns:
                        if re.search(pattern, group_by_section, re.IGNORECASE):
                            # Double-check it's not part of an aggregate function
                            function_pattern = rf'(MIN|MAX|AVG|SUM|COUNT)\s*\(\s*[^)]*{filing_col}[^)]*\)'
                            if not re.search(function_pattern, group_by_section, re.IGNORECASE):
                                warnings.append(f"Potential issue: '{filing_col}' in GROUP BY clause may cause data fragmentation. Consider using aggregate functions like MIN({filing_col}) instead.")
                                break  # Only warn once per column
                
            except Exception as e:
                # If parsing fails, fall back to simple check (but less likely to false positive)
                log_message(f"GROUP BY validation parsing failed: {e}, using simple check")
                for filing_col in filing_columns:
                    if f', {filing_col}' in sql_query or f'{filing_col},' in sql_query:
                        warnings.append(f"Potential issue: '{filing_col}' may be in GROUP BY clause. Verify it's properly aggregated.")
        

        # Match patterns like "FROM table_name alias" or "JOIN table_name alias"
        alias_pattern = r'\b(?:FROM|JOIN)\s+(\w+)\s+(\w+)(?:\s+ON|\s*,|\s*WHERE|\s*GROUP|\s*ORDER|\s*HAVING|\s*LIMIT|\s*$)'
        alias_matches = re.findall(alias_pattern, sql_query, re.IGNORECASE)
        
        reserved_aliases = ['is', 'as', 'in', 'on', 'or', 'and', 'not', 'null', 'true', 'false', 'where', 'select', 'from', 'group', 'order', 'having']
        for table_name, alias in alias_matches:
            if alias.lower() in reserved_aliases:
                errors.append(f"Reserved keyword '{alias}' used as table alias for table '{table_name}'")
        
        # Combine errors and warnings
        all_issues = errors + warnings
        if all_issues:
            return False, "; ".join(all_issues)
        return True, ""

    def _generate_result_message(self, data_rows: List[Dict[str, Any]], pagination_info: Optional[Dict[str, Any]]) -> str:
        if not data_rows:
            if pagination_info and pagination_info.get('total_records', 0) > 0:
                return (f"No results on this page. Showing {pagination_info['showing_from']}-{pagination_info['showing_to']} "
                        f"of {pagination_info['total_records']} results (Page {pagination_info['current_page']} of {pagination_info['total_pages']})")
            return "Query executed successfully, but no results found matching your criteria."
        
        if pagination_info:
            return (f"Showing {pagination_info['showing_from']}-{pagination_info['showing_to']} of {pagination_info['total_records']} results "
                    f"(Page {pagination_info['current_page']} of {pagination_info['total_pages']})")
        else:
            return f"{len(data_rows)} rows returned."



