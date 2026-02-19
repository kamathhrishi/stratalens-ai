#!/usr/bin/env python3
"""
10-K SEC Filings Database Ingestion with Sophisticated Processing

This script uses the full sophisticated data_loading.py processing pipeline
and stores results in PostgreSQL database instead of disk cache.

Features:
- Hierarchical content extraction with section identification
- Contextual chunking preserving document structure
- Table extraction and financial statement identification
- Cross-encoder for reranking
- TF-IDF for hybrid search
- Stores in PostgreSQL with pgvector
- Automatically skips already-processed ticker-year combinations

Usage:
    python ingest_10k_to_database.py --ticker AAPL
    python ingest_10k_to_database.py --tickers AAPL MSFT GOOGL
    python ingest_10k_to_database.py --all-financebench
"""

import argparse
import os
import sys
import json
import logging
import time
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.pool import SimpleConnectionPool
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import shutil
import tempfile
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from functools import partial
import signal

# Import the sophisticated data loading module
sys.path.insert(0, str(Path(__file__).parent))
from ingest_10k_filings_full import (
    DataProcessor,
    download_and_extract_10k,
    download_and_extract_filing,
    create_embeddings_for_filing_data,
    FINANCEBENCH_COMPANIES,
    FILING_TYPE_SECTIONS,
)

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('10k_db_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DatabaseIntegration:
    """Handles storing 10-K data in PostgreSQL database"""

    def __init__(self, db_url: str = None):
        # Use PG_VECTOR database (same as RAG system) which has pgvector extension
        # Priority: provided db_url > PG_VECTOR > DATABASE_URL
        if db_url:
            self.db_url = db_url
        else:
            self.db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")

        if not self.db_url:
            raise ValueError("PG_VECTOR or DATABASE_URL environment variable not set")
        self._connection = None

    def get_connection(self):
        """Get or create a database connection with proper error handling"""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                self.db_url,
                connect_timeout=30,
                options='-c statement_timeout=300000'  # 5 minute query timeout
            )
        return self._connection

    def close_connection(self):
        """Explicitly close the database connection"""
        if self._connection and not self._connection.closed:
            try:
                self._connection.close()
            except Exception as e:
                logger.debug(f"Error closing connection: {e}")
            finally:
                self._connection = None

    def ensure_tables(self):
        """Ensure database tables exist with proper schema"""
        logger.info("🔍 Ensuring database tables exist...")
        logger.info(f"   Database URL: {self.db_url[:50]}...")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # First, ensure pgvector extension is enabled
            logger.info("   Enabling pgvector extension...")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                logger.info("   ✅ pgvector extension enabled")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not enable pgvector extension: {e}")
                # Continue anyway - extension might already be enabled

            # Create ten_k_chunks table with full metadata support
            # Using pgvector database (PG_VECTOR) which has vector extension
            logger.info("   Creating ten_k_chunks table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ten_k_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding vector(384),
                    metadata JSONB,
                    ticker VARCHAR(10),
                    fiscal_year INTEGER,
                    filing_type VARCHAR(10) DEFAULT '10-K',
                    chunk_index INTEGER,
                    citation VARCHAR(200),
                    chunk_type VARCHAR(50),
                    sec_section VARCHAR(50),
                    sec_section_title VARCHAR(200),
                    path_string TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("   ✅ ten_k_chunks table created/verified")

            # Check if table has data before creating indexes
            cursor.execute("SELECT COUNT(*) FROM ten_k_chunks")
            row_count = cursor.fetchone()[0]

            if row_count == 0:
                # Table is empty - create all indexes
                logger.info("   Creating indexes (table is empty)...")

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ten_k_ticker_year
                    ON ten_k_chunks(ticker, fiscal_year);
                """)
                conn.commit()
                logger.info("   ✅ Index on (ticker, fiscal_year)")

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ten_k_section
                    ON ten_k_chunks(sec_section);
                """)
                conn.commit()
                logger.info("   ✅ Index on sec_section")

                # Create vector index
                logger.info("   Creating vector index (this may take a minute for large tables)...")
                try:
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_ten_k_embedding
                        ON ten_k_chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    """)
                    conn.commit()
                    logger.info("   ✅ Vector index created")
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not create vector index: {e}")
                    conn.rollback()
            else:
                # Table has data - skip all index creation (they should already exist)
                logger.info(f"   ⚠️  Table has {row_count:,} rows - skipping ALL index creation")
                logger.info(f"   💡 Indexes should already exist from initial setup")
                logger.info(f"   💡 If indexes are missing, create them manually or clear the table and re-run")

            # Create ten_k_tables table for storing extracted tables
            logger.info("   Creating ten_k_tables table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ten_k_tables (
                    id SERIAL PRIMARY KEY,
                    table_id VARCHAR(100) UNIQUE NOT NULL,
                    ticker VARCHAR(10),
                    fiscal_year INTEGER,
                    content TEXT NOT NULL,
                    table_data JSONB,
                    path_string TEXT,
                    sec_section VARCHAR(50),
                    sec_section_title VARCHAR(200),
                    is_financial_statement BOOLEAN,
                    statement_type VARCHAR(50),
                    priority VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("   ✅ ten_k_tables table created/verified")

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ten_k_tables_ticker_year
                ON ten_k_tables(ticker, fiscal_year);
            """)
            conn.commit()

            # Create complete_sec_filings table for storing full documents
            logger.info("   Creating complete_sec_filings table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS complete_sec_filings (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    company_name VARCHAR(255),
                    filing_type VARCHAR(10) NOT NULL,
                    fiscal_year INTEGER NOT NULL,
                    quarter INTEGER,
                    filing_date DATE,
                    filing_period VARCHAR(50),
                    document_text TEXT NOT NULL,
                    document_html TEXT,
                    document_length INTEGER,
                    sections JSONB,
                    section_offsets JSONB,
                    accession_number VARCHAR(50),
                    cik VARCHAR(20),
                    form_type VARCHAR(10),
                    source_url TEXT,
                    cache_path TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, filing_type, fiscal_year, quarter, filing_date)
                );
            """)
            conn.commit()
            logger.info("   ✅ complete_sec_filings table created/verified")

            # Create indexes for complete_sec_filings
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker
                ON complete_sec_filings(ticker);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker_type_year
                ON complete_sec_filings(ticker, filing_type, fiscal_year);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sec_filings_filing_date
                ON complete_sec_filings(filing_date);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sec_filings_accession
                ON complete_sec_filings(accession_number);
            """)
            conn.commit()
            logger.info("   ✅ Indexes created for complete_sec_filings")

            logger.info("✅ Database tables ensured successfully")

            # Verify tables exist
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('ten_k_chunks', 'ten_k_tables', 'complete_sec_filings')
            """)
            existing_tables = [row[0] for row in cursor.fetchall()]
            logger.info(f"   Verified tables: {existing_tables}")

        except Exception as e:
            logger.error(f"❌ Failed to ensure tables: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if cursor:
                cursor.close()

    def store_chunks(self, ticker: str, fiscal_year: int, processor: DataProcessor, filing_type: str = '10-K'):
        """Store processed chunks in database"""
        chunks = processor.get_chunks()
        embeddings = processor.get_embeddings()

        if not chunks or embeddings is None:
            logger.warning(f"⚠️ No chunks or embeddings to store for {ticker} FY{fiscal_year}")
            return 0

        logger.info(f"💾 Storing {len(chunks)} chunks for {ticker} {filing_type} FY{fiscal_year}")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Delete existing chunks for this ticker, fiscal year, and filing type
            cursor.execute("""
                DELETE FROM ten_k_chunks
                WHERE ticker = %s AND fiscal_year = %s AND filing_type = %s
            """, (ticker, fiscal_year, filing_type))
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.info(f"  🗑️ Deleted {deleted_count} existing chunks")

            # Prepare chunk data
            filing_type_abbr = filing_type.replace('-', '')
            chunk_data = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    'ticker': ticker,
                    'fiscal_year': fiscal_year,
                    'filing_type': filing_type,
                    'chunk_index': idx,
                    'type': chunk.get('type'),
                    'level': chunk.get('level'),
                    'path': chunk.get('path', []),
                    'sec_section': chunk.get('sec_section'),
                    'sec_section_title': chunk.get('sec_section_title')
                }

                citation = f"{ticker}_{filing_type_abbr}_FY{fiscal_year}_{idx}"

                chunk_data.append((
                    chunk['content'],
                    embedding.tolist(),  # Store as array for vector type
                    json.dumps(metadata),
                    ticker,
                    fiscal_year,
                    filing_type,
                    idx,
                    citation,
                    chunk.get('type', 'unknown'),
                    chunk.get('sec_section', 'unknown'),
                    chunk.get('sec_section_title', 'Unknown'),
                    chunk.get('path_string', '')
                ))

            # Batch insert
            execute_values(
                cursor,
                """
                INSERT INTO ten_k_chunks
                (chunk_text, embedding, metadata, ticker, fiscal_year, filing_type,
                 chunk_index, citation, chunk_type, sec_section, sec_section_title, path_string)
                VALUES %s
                """,
                chunk_data,
                template="(%s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )

            conn.commit()
            logger.info(f"  ✅ Stored {len(chunk_data)} chunks")
            return len(chunk_data)

        except Exception as e:
            logger.error(f"  ❌ Failed to store chunks: {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()

    def store_tables(self, ticker: str, fiscal_year: int, processor: DataProcessor):
        """Store extracted tables in database"""
        tables = processor.get_tables()

        if not tables:
            logger.warning(f"⚠️ No tables to store for {ticker} FY{fiscal_year}")
            return 0

        logger.info(f"📊 Storing {len(tables)} tables for {ticker} FY{fiscal_year}")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Delete existing tables for this ticker and fiscal year
            cursor.execute("""
                DELETE FROM ten_k_tables
                WHERE ticker = %s AND fiscal_year = %s
            """, (ticker, fiscal_year))

            # Prepare table data
            table_data = []
            for table_id, table_info in tables.items():
                # Make table_id unique across tickers by prepending ticker_fiscalyear
                unique_table_id = f"{ticker}_FY{fiscal_year}_{table_id}"
                table_data.append((
                    unique_table_id,
                    ticker,
                    fiscal_year,
                    table_info['content'],
                    json.dumps(table_info.get('table_data')),
                    table_info.get('path_string', ''),
                    table_info.get('sec_section', 'unknown'),
                    table_info.get('sec_section_title', 'Unknown'),
                    table_info.get('is_financial_statement', False),
                    table_info.get('statement_type'),
                    table_info.get('priority', 'NORMAL')
                ))

            # Batch insert
            execute_values(
                cursor,
                """
                INSERT INTO ten_k_tables
                (table_id, ticker, fiscal_year, content, table_data, path_string,
                 sec_section, sec_section_title, is_financial_statement, statement_type, priority)
                VALUES %s
                """,
                table_data,
                template="(%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)"
            )

            conn.commit()

            # Count financial statements
            financial_count = sum(1 for t in tables.values() if t.get('is_financial_statement'))
            logger.info(f"  ✅ Stored {len(table_data)} tables ({financial_count} financial statements)")
            return len(table_data)

        except Exception as e:
            logger.error(f"  ❌ Failed to store tables: {e}")
            if conn:
                conn.rollback()
            return 0
        finally:
            if cursor:
                cursor.close()

    def store_complete_filing(
        self,
        ticker: str,
        fiscal_year: int,
        filing_type: str,
        document_text: str,
        filing_period: Optional[str] = None,
        filing_date: Optional[str] = None,
        company_name: Optional[str] = None,
        sections: Optional[List[str]] = None,
        section_offsets: Optional[Dict[str, Dict[str, int]]] = None,
        accession_number: Optional[str] = None,
        cik: Optional[str] = None,
        source_url: Optional[str] = None,
        cache_path: Optional[str] = None
    ) -> bool:
        """
        Store complete SEC filing document in database

        Args:
            ticker: Company ticker symbol
            fiscal_year: Fiscal year of the filing
            filing_type: Type of filing ('10-K', '10-Q', '8-K')
            document_text: Full text of the SEC filing
            filing_period: Normalized period (e.g., 'Q2' for 10-Q, date for 8-K)
            filing_date: Actual filing date
            company_name: Company name
            sections: List of sections in the document
            section_offsets: Character offsets for each section
            accession_number: SEC accession number
            cik: Central Index Key
            source_url: SEC EDGAR URL
            cache_path: Original cache directory path

        Returns:
            True if successful, False otherwise
        """
        if not document_text or len(document_text.strip()) < 100:
            logger.warning(f"⚠️ Document text too short for {ticker} FY{fiscal_year} {filing_type}")
            return False

        logger.info(f"📄 Storing complete filing: {ticker} FY{fiscal_year} {filing_type}")

        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Parse quarter from filing_period if it's a 10-Q
            quarter = None
            if filing_type == '10-Q' and filing_period:
                if filing_period.startswith('Q'):
                    try:
                        quarter = int(filing_period[1])
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse quarter from filing_period: {filing_period}")

            # Calculate document length
            document_length = len(document_text)

            # Prepare data
            cursor.execute("""
                INSERT INTO complete_sec_filings (
                    ticker, company_name, filing_type, fiscal_year,
                    quarter, filing_date, filing_period,
                    document_text, document_length,
                    sections, section_offsets,
                    accession_number, cik, form_type,
                    source_url, cache_path
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, filing_type, fiscal_year, quarter, filing_date)
                DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    filing_period = EXCLUDED.filing_period,
                    document_text = EXCLUDED.document_text,
                    document_length = EXCLUDED.document_length,
                    sections = EXCLUDED.sections,
                    section_offsets = EXCLUDED.section_offsets,
                    accession_number = EXCLUDED.accession_number,
                    cik = EXCLUDED.cik,
                    form_type = EXCLUDED.form_type,
                    source_url = EXCLUDED.source_url,
                    cache_path = EXCLUDED.cache_path,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                ticker,
                company_name,
                filing_type,
                fiscal_year,
                quarter,
                filing_date,
                filing_period,
                document_text,
                document_length,
                json.dumps(sections) if sections else None,
                json.dumps(section_offsets) if section_offsets else None,
                accession_number,
                cik,
                filing_type,  # form_type same as filing_type
                source_url,
                cache_path
            ))

            conn.commit()
            logger.info(f"  ✅ Stored complete filing ({document_length:,} chars)")
            return True

        except Exception as e:
            logger.error(f"  ❌ Failed to store complete filing: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

    def check_data_exists(self, ticker: str, fiscal_year: int, filing_type: str = '10-K') -> bool:
        """
        Check if data already exists for this ticker-year-filing_type combination

        Returns True if data exists, False otherwise
        """
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check if chunks exist for this ticker-year-filing_type
            cursor.execute("""
                SELECT COUNT(*) FROM ten_k_chunks
                WHERE ticker = %s AND fiscal_year = %s AND filing_type = %s
            """, (ticker, fiscal_year, filing_type))

            count = cursor.fetchone()[0]
            return count > 0

        except Exception as e:
            logger.debug(f"Error checking if data exists: {e}")
            return False
        finally:
            if cursor:
                cursor.close()

def cleanup_temp_files():
    """Clean up temporary files created by datamule"""
    try:
        # Clean up temp directory
        temp_dir = Path(tempfile.gettempdir())
        datamule_patterns = ['datamule*', 'portfolio*', 'sec_*']

        cleaned_count = 0
        for pattern in datamule_patterns:
            for temp_file in temp_dir.glob(pattern):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                        cleaned_count += 1
                    elif temp_file.is_dir():
                        shutil.rmtree(temp_file)
                        cleaned_count += 1
                except Exception as e:
                    logger.debug(f"Could not delete temp file {temp_file}: {e}")

        if cleaned_count > 0:
            logger.info(f"🧹 Cleaned up {cleaned_count} temporary files")

        # Force garbage collection
        gc.collect()

    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Process timeout")


def ingest_ticker_worker(ticker: str, lookback_years: int, db_url: str, timeout_minutes: int = 30):
    """
    Worker function for parallel processing - must be picklable.
    Each worker creates its own database connection.

    IMPORTANT: This function includes aggressive garbage collection to prevent
    memory accumulation from datamule libraries.

    Args:
        ticker: Stock ticker symbol
        lookback_years: Number of years to look back
        db_url: Database connection URL
        timeout_minutes: Maximum time allowed for processing (default: 30 minutes)
    """
    # Set up timeout handler (Unix-based systems only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)

    db_integration = None
    try:
        # Create database integration for this worker with explicit db_url
        # This ensures workers use the same database as the main process
        db_integration = DatabaseIntegration(db_url=db_url)

        # Call the main ingestion function
        result = ingest_ticker(ticker, lookback_years, db_integration)

        # Clean up database connection
        db_integration.close_connection()

        # AGGRESSIVE CLEANUP to prevent memory leaks
        cleanup_temp_files()

        # Force garbage collection multiple times
        gc.collect()
        gc.collect()
        gc.collect()

        # Try to clear any remaining references
        db_integration = None

        # Cancel the alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        return result

    except TimeoutError:
        logger.error(f"⏱️ Worker timeout for {ticker} (exceeded {timeout_minutes} minutes)")
        return {'ticker': ticker, 'error': f'Timeout after {timeout_minutes} minutes'}

    except Exception as e:
        logger.error(f"❌ Worker failed for {ticker}: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return {'ticker': ticker, 'error': str(e)}

    finally:
        # Cancel the alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        # Ensure cleanup happens even on failure
        try:
            if db_integration:
                db_integration.close_connection()
            cleanup_temp_files()
            gc.collect()
        except:
            pass


def ingest_ticker(ticker: str, lookback_years: int, db_integration: DatabaseIntegration):
    """Ingest 10-K filings for a single ticker using sophisticated processing"""
    logger.info(f"📈 Processing {ticker}...")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    # Download and extract 10-K with sophisticated processing
    start_year = start_date.year - 1  # Buffer for fiscal year mismatch
    end_year = end_date.year + 1

    # Calculate minimum fiscal year to process (based on lookback period)
    min_fiscal_year = start_date.year
    logger.info(f"📅 Will only process fiscal years >= {min_fiscal_year}")

    logger.info(f"📥 Downloading 10-K filings from {start_year} to {end_year}")
    filings_by_year = download_and_extract_10k(ticker, start_year, end_year)

    if not filings_by_year:
        logger.warning(f"⚠️ No 10-K filings found for {ticker}")
        return {'ticker': ticker, 'filings_processed': 0, 'chunks_stored': 0, 'tables_stored': 0}

    # Filter to only recent fiscal years based on lookback period
    filtered_filings = {fy: data for fy, data in filings_by_year.items() if fy >= min_fiscal_year}

    if not filtered_filings:
        logger.warning(f"⚠️ No recent 10-K filings found for {ticker} (need FY >= {min_fiscal_year})")
        return {'ticker': ticker, 'filings_processed': 0, 'chunks_stored': 0, 'tables_stored': 0}

    skipped_years = set(filings_by_year.keys()) - set(filtered_filings.keys())
    if skipped_years:
        logger.info(f"⏭️  Skipping older fiscal years: {sorted(skipped_years)} (older than {lookback_years} year(s))")

    logger.info(f"📦 Found {len(filtered_filings)} fiscal year(s) of recent data: {sorted(filtered_filings.keys())}")

    # Process each fiscal year
    total_chunks = 0
    total_tables = 0
    filings_processed = 0
    filings_skipped = 0

    for fiscal_year, filing_data in filtered_filings.items():
        logger.info(f"\n🔧 Processing FY{fiscal_year}...")

        # Check if data already exists for this ticker-year
        if db_integration.check_data_exists(ticker, fiscal_year):
            logger.info(f"⏭️  Skipping {ticker} FY{fiscal_year} - already processed")
            filings_skipped += 1
            continue

        try:
            # Create processor and prepare chunks with hierarchical structure
            processor = DataProcessor()

            # Use hierarchical chunks (the sophisticated method)
            processor.prepare_chunks(filing_data, use_hierarchical=True, exclude_titles=True)

            # Create embeddings
            processor.create_embeddings()

            # Identify financial statements
            processor.identify_financial_statement_tables()

            # Store in database
            chunks_stored = db_integration.store_chunks(ticker, fiscal_year, processor)
            tables_stored = db_integration.store_tables(ticker, fiscal_year, processor)

            # Store complete filing document
            document_text = filing_data.get('document_text', '')
            sections_list = None
            if 'hierarchical_chunks' in filing_data and filing_data['hierarchical_chunks']:
                # Extract unique sections from hierarchical chunks
                sections_set = set()
                for chunk in filing_data['hierarchical_chunks']:
                    if 'sec_section' in chunk:
                        sections_set.add(chunk['sec_section'])
                sections_list = sorted(list(sections_set))

            filing_stored = db_integration.store_complete_filing(
                ticker=ticker,
                fiscal_year=fiscal_year,
                filing_type='10-K',
                document_text=document_text,
                sections=sections_list
            )

            total_chunks += chunks_stored
            total_tables += tables_stored
            filings_processed += 1

            logger.info(f"✅ FY{fiscal_year}: {chunks_stored} chunks, {tables_stored} tables, filing stored: {filing_stored}")

            # Clean up processor to free memory
            processor = None
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Failed to process FY{fiscal_year}: {e}")
            # Clean up on error too
            try:
                processor = None
                gc.collect()
            except:
                pass
            continue

    # Build result message
    if filings_processed == 0 and filings_skipped == 0:
        result_msg = f"⚠️  {ticker}: No 10-K found"
    elif filings_processed == 0 and filings_skipped > 0:
        result_msg = f"⏭️  {ticker}: All {filings_skipped} filings already processed (skipped)"
    elif filings_skipped > 0:
        result_msg = f"✅ {ticker}: {filings_processed} new, {filings_skipped} skipped, {total_chunks} chunks, {total_tables} tables"
    else:
        result_msg = f"✅ {ticker}: {filings_processed} filings, {total_chunks} chunks, {total_tables} tables"

    logger.info(result_msg)

    # Clean up temporary files after processing this ticker
    cleanup_temp_files()

    return {
        'ticker': ticker,
        'filings_processed': filings_processed,
        'filings_skipped': filings_skipped,
        'chunks_stored': total_chunks,
        'tables_stored': total_tables,
    }


def ingest_filing_worker(ticker: str, filing_type: str, lookback_years: int, db_url: str,
                         timeout_minutes: int = 30, max_filings: int = None):
    """
    Worker function for parallel processing of any filing type.

    Args:
        ticker: Stock ticker symbol
        filing_type: SEC filing type ('10-K', '10-Q', '8-K', 'S-11')
        lookback_years: Number of years to look back
        db_url: Database connection URL
        timeout_minutes: Maximum time allowed for processing
        max_filings: Maximum number of filings per ticker (useful for 8-K)
    """
    if filing_type == '10-K':
        return ingest_ticker_worker(ticker, lookback_years, db_url, timeout_minutes)

    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_minutes * 60)

    db_integration = None
    try:
        db_integration = DatabaseIntegration(db_url=db_url)
        result = ingest_filing(ticker, filing_type, lookback_years, db_integration,
                               max_filings=max_filings)
        db_integration.close_connection()
        cleanup_temp_files()
        gc.collect()
        db_integration = None

        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)

        return result

    except TimeoutError:
        logger.error(f"⏱️ Worker timeout for {ticker} {filing_type} (exceeded {timeout_minutes} minutes)")
        return {'ticker': ticker, 'filing_type': filing_type, 'error': f'Timeout after {timeout_minutes} minutes'}

    except Exception as e:
        logger.error(f"❌ Worker failed for {ticker} {filing_type}: {e}")
        return {'ticker': ticker, 'filing_type': filing_type, 'error': str(e)}

    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        try:
            if db_integration:
                db_integration.close_connection()
            cleanup_temp_files()
            gc.collect()
        except:
            pass


def ingest_filing(ticker: str, filing_type: str, lookback_years: int,
                  db_integration: DatabaseIntegration, max_filings: int = None):
    """
    Ingest any filing type for a single ticker.

    Args:
        ticker: Stock ticker symbol
        filing_type: SEC filing type ('10-K', '10-Q', '8-K', 'S-11')
        lookback_years: Number of years to look back
        db_integration: DatabaseIntegration instance
        max_filings: Maximum number of filings per ticker (useful for 8-K)
    """
    # For 10-K, delegate to the original function
    if filing_type == '10-K':
        return ingest_ticker(ticker, lookback_years, db_integration)

    logger.info(f"📈 Processing {ticker} ({filing_type})...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)

    start_year = start_date.year - 1
    end_year = end_date.year + 1
    min_fiscal_year = start_date.year

    logger.info(f"📥 Downloading {filing_type} filings from {start_year} to {end_year}")

    filings_by_key = download_and_extract_filing(
        ticker, start_year, end_year,
        filing_type=filing_type,
        max_filings=max_filings
    )

    if not filings_by_key:
        logger.warning(f"⚠️ No {filing_type} filings found for {ticker}")
        return {'ticker': ticker, 'filing_type': filing_type, 'filings_processed': 0,
                'chunks_stored': 0, 'tables_stored': 0}

    # For 8-K, filing keys can be dates or year-based — filter by year
    if filing_type == '8-K':
        filtered_filings = {}
        for key, data in filings_by_key.items():
            key_str = str(key)
            # Extract year from date string (YYYY-MM-DD) or integer key
            try:
                key_year = int(key_str[:4])
            except (ValueError, IndexError):
                key_year = min_fiscal_year  # include if we can't parse
            if key_year >= min_fiscal_year:
                filtered_filings[key] = data
    else:
        # 10-Q, S-11: filter by fiscal year
        filtered_filings = {fy: data for fy, data in filings_by_key.items()
                           if isinstance(fy, int) and fy >= min_fiscal_year}

    if not filtered_filings:
        logger.warning(f"⚠️ No recent {filing_type} filings found for {ticker}")
        return {'ticker': ticker, 'filing_type': filing_type, 'filings_processed': 0,
                'chunks_stored': 0, 'tables_stored': 0}

    logger.info(f"📦 Found {len(filtered_filings)} {filing_type} filing(s) to process")

    total_chunks = 0
    total_tables = 0
    filings_processed = 0
    filings_skipped = 0

    for filing_key, filing_data in filtered_filings.items():
        logger.info(f"\n🔧 Processing {filing_type} key={filing_key}...")

        # For DB storage, convert filing_key to an integer fiscal_year
        # 8-K date keys get converted to year integer
        if isinstance(filing_key, int):
            fiscal_year_int = filing_key
        else:
            try:
                fiscal_year_int = int(str(filing_key)[:4])
            except (ValueError, IndexError):
                fiscal_year_int = min_fiscal_year

        # Check if data already exists
        if db_integration.check_data_exists(ticker, fiscal_year_int, filing_type):
            logger.info(f"⏭️  Skipping {ticker} {filing_type} key={filing_key} - already processed")
            filings_skipped += 1
            continue

        try:
            processor = DataProcessor()
            processor.prepare_chunks(filing_data, use_hierarchical=True, exclude_titles=True)
            processor.create_embeddings()

            if filing_type not in ('8-K', 'S-11'):
                processor.identify_financial_statement_tables()

            chunks_stored = db_integration.store_chunks(ticker, fiscal_year_int, processor, filing_type=filing_type)
            tables_stored = db_integration.store_tables(ticker, fiscal_year_int, processor)

            # Store complete filing document
            document_text = filing_data.get('document_text', '')
            sections_list = None
            if 'hierarchical_chunks' in filing_data and filing_data['hierarchical_chunks']:
                sections_set = set()
                for chunk in filing_data['hierarchical_chunks']:
                    if 'sec_section' in chunk:
                        sections_set.add(chunk['sec_section'])
                sections_list = sorted(list(sections_set))

            # Determine filing_period for 10-Q and 8-K
            filing_period = None
            if filing_type == '10-Q':
                # Extract quarter from filing_data metadata if available
                # Otherwise use the filing_key format if it contains quarter info
                filing_period = filing_data.get('quarter', f"Q{filing_data.get('quarter_num', '')}")
            elif filing_type == '8-K':
                # For 8-K, use the filing date as the period
                filing_period = str(filing_key) if not isinstance(filing_key, int) else None

            filing_stored = db_integration.store_complete_filing(
                ticker=ticker,
                fiscal_year=fiscal_year_int,
                filing_type=filing_type,
                document_text=document_text,
                filing_period=filing_period,
                sections=sections_list
            )

            total_chunks += chunks_stored
            total_tables += tables_stored
            filings_processed += 1

            logger.info(f"✅ {filing_type} key={filing_key}: {chunks_stored} chunks, {tables_stored} tables, filing stored: {filing_stored}")

            processor = None
            gc.collect()

        except Exception as e:
            logger.error(f"❌ Failed to process {filing_type} key={filing_key}: {e}")
            try:
                processor = None
                gc.collect()
            except:
                pass
            continue

    # Build result message
    if filings_processed == 0 and filings_skipped == 0:
        result_msg = f"⚠️  {ticker} {filing_type}: No filings found"
    elif filings_processed == 0 and filings_skipped > 0:
        result_msg = f"⏭️  {ticker} {filing_type}: All {filings_skipped} filings already processed"
    elif filings_skipped > 0:
        result_msg = f"✅ {ticker} {filing_type}: {filings_processed} new, {filings_skipped} skipped, {total_chunks} chunks, {total_tables} tables"
    else:
        result_msg = f"✅ {ticker} {filing_type}: {filings_processed} filings, {total_chunks} chunks, {total_tables} tables"

    logger.info(result_msg)
    cleanup_temp_files()

    return {
        'ticker': ticker,
        'filing_type': filing_type,
        'filings_processed': filings_processed,
        'filings_skipped': filings_skipped,
        'chunks_stored': total_chunks,
        'tables_stored': total_tables,
    }


def load_all_tickers(ticker_file: str = None) -> List[str]:
    """Load all US tickers from file"""
    if ticker_file is None:
        ticker_file = Path(__file__).parent / "us_tickers.txt"
    else:
        ticker_file = Path(ticker_file)

    if not ticker_file.exists():
        logger.error(f"❌ Ticker file not found: {ticker_file}")
        return []

    try:
        with open(ticker_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]

        logger.info(f"📋 Loaded {len(tickers)} tickers from {ticker_file}")
        return tickers
    except Exception as e:
        logger.error(f"❌ Error reading ticker file: {e}")
        return []


def main():
    # Fix CUDA multiprocessing issue - use spawn instead of fork
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(description='Ingest 10-K SEC filings with sophisticated processing')
    parser.add_argument('--ticker', type=str, help='Single ticker to ingest')
    parser.add_argument('--tickers', type=str, nargs='+', help='Multiple tickers')
    parser.add_argument('--all-tickers', action='store_true', help='Ingest ALL US companies (9500+)')
    parser.add_argument('--all-financebench', action='store_true', help='Ingest all FinanceBench companies')
    parser.add_argument('--ticker-file', type=str, help='Path to ticker file (default: us_tickers.txt)')
    parser.add_argument('--lookback-years', type=int, default=1, help='Years to look back (default: 1)')
    parser.add_argument('--max-tickers', type=int, help='Limit number of tickers to process')
    parser.add_argument('--skip-first', type=int, default=0, help='Skip first N tickers (for resuming)')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers (default: 3)')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially (no parallel processing)')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout per ticker in minutes (default: 30)')
    parser.add_argument('--retry-failed', action='store_true', help='Retry failed tickers once')

    args = parser.parse_args()

    # Determine tickers
    tickers = []
    if args.ticker:
        tickers = [args.ticker.upper()]
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.all_tickers:
        tickers = load_all_tickers(args.ticker_file)
        if not tickers:
            logger.error("❌ Failed to load tickers")
            sys.exit(1)
    elif args.all_financebench:
        tickers = list(FINANCEBENCH_COMPANIES.values())
    else:
        parser.print_help()
        print("\nError: Must specify --ticker, --tickers, --all-tickers, or --all-financebench")
        sys.exit(1)

    # Apply skip and max limits
    if args.skip_first > 0:
        logger.info(f"⏭️ Skipping first {args.skip_first} tickers")
        tickers = tickers[args.skip_first:]

    if args.max_tickers:
        logger.info(f"🔢 Limiting to {args.max_tickers} tickers")
        tickers = tickers[:args.max_tickers]

    logger.info(f"🚀 10-K Database Ingestion Starting")
    logger.info(f"  Tickers: {len(tickers)} companies")
    logger.info(f"  Lookback: {args.lookback_years} year(s)")
    logger.info(f"  Workers: {args.workers if not args.sequential else 1} (parallel)" if not args.sequential else "  Mode: Sequential")
    logger.info(f"  Processing: Hierarchical extraction + contextual chunking + table identification")

    # Get database URL for workers (must be consistent across all processes)
    db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error("❌ PG_VECTOR or DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info(f"  Database: {db_url[:50]}...")

    # Initialize database (create tables) - use same URL as workers
    db_integration = DatabaseIntegration(db_url=db_url)
    db_integration.ensure_tables()

    # Process tickers
    results = []
    start_time = time.time()

    if args.sequential:
        # Sequential processing (old behavior)
        logger.info("🔄 Running in sequential mode")
        for idx, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"📊 Progress: {idx}/{len(tickers)} ({idx/len(tickers)*100:.1f}%)")
                logger.info(f"{'='*70}")

                result = ingest_ticker(ticker, args.lookback_years, db_integration)
                results.append(result)

                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = avg_time * (len(tickers) - idx)
                    logger.info(f"⏱️ Processed {idx}/{len(tickers)} in {elapsed/60:.1f}m, "
                              f"~{remaining/60:.1f}m remaining")

            except Exception as e:
                logger.error(f"❌ Failed to ingest {ticker}: {e}")
                results.append({'ticker': ticker, 'error': str(e)})
                continue

    else:
        # Parallel processing with multiple workers
        logger.info(f"🔥 Running with {args.workers} parallel workers")

        # Check Python version for max_tasks_per_child support
        python_version = sys.version_info
        supports_max_tasks = python_version >= (3, 11)

        if supports_max_tasks:
            logger.info("✅ Using max_tasks_per_child=1 to prevent memory accumulation")
        else:
            logger.info("⚠️  Python < 3.11 detected - workers will be reused (may accumulate memory)")
            logger.info("   Consider upgrading to Python 3.11+ for better memory management")

        completed = 0
        # Use max_tasks_per_child=1 to ensure each process handles only ONE ticker
        # This prevents memory accumulation from datamule and other libraries
        executor_kwargs = {'max_workers': args.workers}
        if supports_max_tasks:
            executor_kwargs['max_tasks_per_child'] = 1

        with ProcessPoolExecutor(**executor_kwargs) as executor:
            # Submit all tasks with timeout
            future_to_ticker = {
                executor.submit(ingest_ticker_worker, ticker, args.lookback_years, db_url, args.timeout): ticker
                for ticker in tickers
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_ticker, timeout=None):
                ticker = future_to_ticker[future]
                completed += 1

                try:
                    # Get result with a timeout buffer (add 2 minutes for overhead)
                    result = future.result(timeout=(args.timeout * 60) + 120)
                    results.append(result)

                    # Log progress every 10 completions or on error
                    if completed % 10 == 0 or 'error' in result:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed
                        remaining = avg_time * (len(tickers) - completed)

                        successful = len([r for r in results if r.get('filings_processed', 0) > 0])
                        failed = len([r for r in results if 'error' in r])
                        logger.info(f"⏱️ Progress: {completed}/{len(tickers)} "
                                  f"({completed/len(tickers)*100:.1f}%) | "
                                  f"✅ {successful} successful | ❌ {failed} failed | "
                                  f"Time: {elapsed/60:.1f}m elapsed, ~{remaining/60:.1f}m remaining")

                except TimeoutError:
                    logger.error(f"⏱️ Future timeout for {ticker} (process didn't respond)")
                    results.append({'ticker': ticker, 'error': 'Future timeout - process hung'})
                except Exception as e:
                    logger.error(f"❌ Worker exception for {ticker}: {e}")
                    results.append({'ticker': ticker, 'error': str(e)})

        logger.info(f"🎉 All workers completed!")

        # Retry failed tickers if requested
        if args.retry_failed:
            failed_tickers = [r['ticker'] for r in results if 'error' in r]
            if failed_tickers:
                logger.info(f"\n🔄 Retrying {len(failed_tickers)} failed tickers...")
                retry_results = []

                with ProcessPoolExecutor(**executor_kwargs) as executor:
                    future_to_ticker = {
                        executor.submit(ingest_ticker_worker, ticker, args.lookback_years, db_url, args.timeout): ticker
                        for ticker in failed_tickers
                    }

                    for future in as_completed(future_to_ticker, timeout=None):
                        ticker = future_to_ticker[future]
                        try:
                            result = future.result(timeout=(args.timeout * 60) + 120)
                            retry_results.append(result)

                            if 'error' not in result:
                                logger.info(f"✅ Retry successful for {ticker}")
                                # Update original result
                                for i, r in enumerate(results):
                                    if r['ticker'] == ticker:
                                        results[i] = result
                                        break
                            else:
                                logger.warning(f"❌ Retry failed for {ticker}: {result.get('error')}")
                        except Exception as e:
                            logger.error(f"❌ Retry exception for {ticker}: {e}")

                successful_retries = len([r for r in retry_results if 'error' not in r and r.get('filings_processed', 0) > 0])
                logger.info(f"🔄 Retry complete: {successful_retries}/{len(failed_tickers)} succeeded")

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 INGESTION SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Tickers processed: {len(results)}")

    total_filings = sum(r.get('filings_processed', 0) for r in results)
    total_skipped = sum(r.get('filings_skipped', 0) for r in results)
    total_chunks = sum(r.get('chunks_stored', 0) for r in results)
    total_tables = sum(r.get('tables_stored', 0) for r in results)
    logger.info(f"Total filings processed: {total_filings}")
    logger.info(f"Total filings skipped: {total_skipped}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"Total tables: {total_tables}")

    # Count successes and failures
    successful = [r for r in results if 'error' not in r and r.get('filings_processed', 0) > 0]
    skipped_all = [r for r in results if 'error' not in r and r.get('filings_processed', 0) == 0 and r.get('filings_skipped', 0) > 0]
    failed = [r for r in results if 'error' in r]
    no_data = [r for r in results if 'error' not in r and r.get('filings_processed', 0) == 0 and r.get('filings_skipped', 0) == 0]

    logger.info(f"\nResults breakdown:")
    logger.info(f"  ✅ Successful: {len(successful)}")
    logger.info(f"  ⏭️  Already processed (skipped): {len(skipped_all)}")
    logger.info(f"  ❌ Failed: {len(failed)}")
    logger.info(f"  ⚠️ No data found: {len(no_data)}")

    # Show sample of successful
    if successful:
        logger.info(f"\nSample successful ingestions:")
        for result in successful[:10]:
            skipped_msg = f", {result.get('filings_skipped', 0)} skipped" if result.get('filings_skipped', 0) > 0 else ""
            logger.info(f"  {result['ticker']}: {result['filings_processed']} filings{skipped_msg}, "
                       f"{result['chunks_stored']} chunks, {result['tables_stored']} tables")
        if len(successful) > 10:
            logger.info(f"  ... and {len(successful) - 10} more")

    # Show companies that were entirely skipped
    if skipped_all:
        logger.info(f"\n⏭️  Companies with all filings already processed:")
        sample_skipped = [r['ticker'] for r in skipped_all[:20]]
        logger.info(f"  {', '.join(sample_skipped)}")
        if len(skipped_all) > 20:
            logger.info(f"  ... and {len(skipped_all) - 20} more")

    # Show all failures
    if failed:
        logger.info(f"\nFailed tickers:")
        for result in failed[:20]:
            logger.info(f"  {result['ticker']}: {result.get('error', 'Unknown error')}")
        if len(failed) > 20:
            logger.info(f"  ... and {len(failed) - 20} more errors")

    logger.info(f"\n✅ Ingestion complete!")

    # Final cleanup
    try:
        db_integration.close_connection()
    except:
        pass
    cleanup_temp_files()


if __name__ == "__main__":
    main()
