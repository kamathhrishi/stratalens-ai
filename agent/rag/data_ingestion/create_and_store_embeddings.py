#!/usr/bin/env python3
"""
Creates embeddings from downloaded earnings transcripts and stores them in PostgreSQL with pgvector.

This script:
1. Reads transcript JSON files from earnings_transcripts/ folder
2. Chunks transcripts into smaller pieces with overlap
3. Creates vector embeddings using sentence-transformers
4. Stores embeddings and complete transcripts in PostgreSQL database
5. Creates indexes for efficient RAG retrieval

Usage:
    python create_and_store_embeddings.py [--max-files N] [--force-regenerate]
"""

import argparse
import os
import json
import logging
import time
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

# Load environment variables from main project directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Processes earnings transcripts: creates embeddings and stores them in PostgreSQL with pgvector."""
    
    def __init__(self):
        self.config = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "embedding_model": "all-MiniLM-L6-v2",
            "pgvector_url": os.getenv("DATABASE_URL", ""),
        }
        
        # Define available quarters for 2023, 2024, and 2025
        self.quarters = [
            # 2023 quarters
            {"quarter": "2023_q1", "folder": "earnings_transcripts", "year": 2023, "quarter_num": 1},
            {"quarter": "2023_q2", "folder": "earnings_transcripts", "year": 2023, "quarter_num": 2},
            {"quarter": "2023_q3", "folder": "earnings_transcripts", "year": 2023, "quarter_num": 3},
            {"quarter": "2023_q4", "folder": "earnings_transcripts", "year": 2023, "quarter_num": 4},
            # 2024 quarters
            {"quarter": "2024_q1", "folder": "earnings_transcripts", "year": 2024, "quarter_num": 1},
            {"quarter": "2024_q2", "folder": "earnings_transcripts", "year": 2024, "quarter_num": 2},
            {"quarter": "2024_q3", "folder": "earnings_transcripts", "year": 2024, "quarter_num": 3},
            {"quarter": "2024_q4", "folder": "earnings_transcripts", "year": 2024, "quarter_num": 4},
            # 2025 quarters
            {"quarter": "2025_q1", "folder": "earnings_transcripts", "year": 2025, "quarter_num": 1},
            {"quarter": "2025_q2", "folder": "earnings_transcripts", "year": 2025, "quarter_num": 2},
            {"quarter": "2025_q3", "folder": "earnings_transcripts", "year": 2025, "quarter_num": 3},
            {"quarter": "2025_q4", "folder": "earnings_transcripts", "year": 2025, "quarter_num": 4},
        ]
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config["embedding_model"])
        logger.info(f"Initialized embedding model: {self.config['embedding_model']}")
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"🧠 Memory usage at {stage}: {memory_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
    
    def force_garbage_collection(self, stage: str):
        """Force garbage collection and log memory cleanup."""
        logger.info(f"🧹 Running garbage collection at {stage}...")
        collected = gc.collect()
        logger.info(f"✅ Garbage collection completed: {collected} objects collected")
        self.log_memory_usage(f"{stage} (after GC)")
    
    @staticmethod
    def _extract_transcript_text(transcript: Dict[str, Any]) -> str:
        """Extract text content from a transcript dict, handling nested data structures."""
        text = ""

        # Check for nested data structure first
        if 'data' in transcript and isinstance(transcript['data'], dict):
            if 'transcript' in transcript['data']:
                text = transcript['data']['transcript']
            elif 'content' in transcript['data']:
                text = transcript['data']['content']

        # Fallback to direct fields
        if not text:
            if 'content' in transcript:
                text = transcript['content']
            elif 'transcript' in transcript:
                text = transcript['transcript']
            else:
                # Try to extract text from other fields
                for key, value in transcript.items():
                    if isinstance(value, str) and len(value) > 100:
                        text = value
                        break

        return text

    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = self.config["chunk_size"]
        if chunk_overlap is None:
            chunk_overlap = self.config["chunk_overlap"]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if len(chunk.strip()) > 0:
                chunks.append(chunk.strip())
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def load_transcripts_for_quarter(self, quarter_info: Dict, max_files: int = 1000000) -> List[Dict[str, Any]]:
        """Load earnings transcript files for a specific quarter."""
        transcripts_folder = Path(quarter_info["folder"])
        if not transcripts_folder.exists():
            logger.warning(f"Transcripts folder not found: {transcripts_folder}")
            return []
        
        # Filter files by year and quarter pattern, but exclude "no transcript" files
        year = quarter_info["year"]
        quarter_num = quarter_info["quarter_num"]
        
        # Pattern to match files for specific year and quarter, excluding no-transcript files
        pattern = f"*{year}_Q{quarter_num}.json"
        all_files = list(transcripts_folder.glob(pattern))
        
        # Filter out files with "status_no_transcript" in the filename
        transcript_files = [f for f in all_files if "status_no_transcript" not in f.name.lower()]
        
        # Apply max_files limit after filtering
        transcript_files = transcript_files[:max_files]
        
        logger.info(f"Found {len(all_files)} total files for {quarter_info['quarter']}, {len(transcript_files)} valid transcripts (excluded {len(all_files) - len(transcript_files)} no-transcript files)")
        
        transcripts = []
        for file_path in transcript_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcripts.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(transcripts)} transcripts from {quarter_info['quarter']}")
        return transcripts
    
    def create_embeddings_for_quarter(self, transcripts: List[Dict[str, Any]], quarter_info: Dict) -> tuple:
        """Create embeddings for all transcript chunks in a specific quarter."""
        all_chunks = []
        all_metadata = []
        skipped_count = 0
        
        for transcript in transcripts:
            text = self._extract_transcript_text(transcript)

            # Skip if no meaningful text found
            if not text or len(text.strip()) < 50:
                logger.debug(f"Skipping transcript with insufficient content: {transcript.get('ticker', 'unknown')} (length: {len(text) if text else 0})")
                skipped_count += 1
                continue
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            # Create metadata for each chunk with proper citations
            for i, chunk in enumerate(chunks):
                # Extract ticker and date from nested structure or direct fields
                ticker = transcript.get('ticker', 'unknown')
                date = transcript.get('date', 'unknown')
                transcript_id = transcript.get('id', 'unknown')
                
                # Check nested data structure for additional info
                if 'data' in transcript and isinstance(transcript['data'], dict):
                    if 'date' in transcript['data']:
                        date = transcript['data']['date']
                    if 'id' in transcript['data']:
                        transcript_id = transcript['data']['id']
                
                # Create citation in format: TICKER_DATE_chunk_N
                citation = f"{ticker}_{date}_chunk_{i}"
                
                metadata = {
                    'transcript_id': transcript_id,
                    'ticker': ticker,
                    'date': date,
                    'chunk_index': i,
                    'citation': citation,
                    'company_name': transcript.get('company_name', 'Unknown'),
                    'quarter': quarter_info['quarter_num'],
                    'year': quarter_info['year']
                }
                all_chunks.append(chunk)
                all_metadata.append(metadata)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(transcripts)} transcripts for {quarter_info['quarter']} (skipped {skipped_count} with insufficient content)")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {quarter_info['quarter']}...")
        embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape} for {quarter_info['quarter']}")
        
        return embeddings, all_metadata
    
    def format_existing_database(self):
        """Format/clear the existing transcript_chunks and complete_transcripts tables."""
        if not self.config["pgvector_url"]:
            logger.warning("DATABASE_URL not set, skipping database formatting")
            return
        
        try:
            logger.info("🔧 Formatting existing database...")
            conn = psycopg2.connect(self.config["pgvector_url"])
            cursor = conn.cursor()
            logger.info("✅ Database connection established")
            
            # Drop existing tables completely to ensure clean slate
            logger.info("🗑️ Dropping existing tables...")
            cursor.execute("DROP TABLE IF EXISTS transcript_chunks CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS complete_transcripts CASCADE;")
            conn.commit()
            logger.info("✅ Existing tables dropped")
            
            # Create fresh transcript_chunks table with proper structure
            logger.info("🏗️ Creating fresh transcript_chunks table...")
            cursor.execute("""
                CREATE TABLE transcript_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(384),
                    metadata JSONB,
                    ticker VARCHAR(10),
                    year INTEGER,
                    quarter INTEGER,
                    transcript_id VARCHAR(100),
                    chunk_index INTEGER,
                    citation VARCHAR(200),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✅ transcript_chunks table created")
            
            # Create fresh complete_transcripts table
            logger.info("🏗️ Creating fresh complete_transcripts table...")
            cursor.execute("""
                CREATE TABLE complete_transcripts (
                    id SERIAL PRIMARY KEY,
                    transcript_id VARCHAR(100) UNIQUE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    company_name VARCHAR(200),
                    date VARCHAR(50),
                    year INTEGER,
                    quarter INTEGER,
                    full_transcript TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            logger.info("✅ complete_transcripts table created")
            
            cursor.close()
            conn.close()
            logger.info("✅ Database formatting completed")
            
        except Exception as e:
            logger.error(f"Failed to format database: {e}")
            raise
    
    def save_complete_transcripts_to_db(self, transcripts: List[Dict[str, Any]], quarter_info: Dict):
        """Save complete transcripts to PostgreSQL database."""
        if not self.config["pgvector_url"]:
            logger.warning("DATABASE_URL not set, skipping complete transcripts storage")
            return
        
        try:
            logger.info(f"💾 Connecting to database to store {quarter_info['quarter']} complete transcripts...")
            conn = psycopg2.connect(self.config["pgvector_url"])
            cursor = conn.cursor()
            logger.info("✅ Database connection established")
            
            # Ensure complete_transcripts table exists
            logger.info("🔍 Checking if complete_transcripts table exists...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS complete_transcripts (
                    id SERIAL PRIMARY KEY,
                    transcript_id VARCHAR(100) UNIQUE NOT NULL,
                    ticker VARCHAR(10) NOT NULL,
                    company_name VARCHAR(200),
                    date VARCHAR(50),
                    year INTEGER,
                    quarter INTEGER,
                    full_transcript TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("✅ complete_transcripts table ensured")
            
            # Prepare data for complete transcripts
            complete_transcript_data = []
            skipped_complete_count = 0
            
            for transcript in transcripts:
                text = self._extract_transcript_text(transcript)

                # Skip if no meaningful text found
                if not text or len(text.strip()) < 50:
                    logger.debug(f"Skipping transcript with insufficient content for complete storage: {transcript.get('ticker', 'unknown')} (length: {len(text) if text else 0})")
                    skipped_complete_count += 1
                    continue
                
                # Extract metadata
                ticker = transcript.get('ticker', 'unknown')
                date = transcript.get('date', 'unknown')
                transcript_id = transcript.get('id', f"{ticker}_{date}")
                company_name = transcript.get('company_name', 'Unknown')
                
                # Check nested data structure for additional info
                if 'data' in transcript and isinstance(transcript['data'], dict):
                    if 'date' in transcript['data']:
                        date = transcript['data']['date']
                    if 'id' in transcript['data']:
                        transcript_id = transcript['data']['id']
                    if 'company_name' in transcript['data']:
                        company_name = transcript['data']['company_name']
                
                complete_transcript_data.append((
                    transcript_id,
                    ticker,
                    company_name,
                    date,
                    quarter_info['year'],
                    quarter_info['quarter_num'],
                    text,
                    json.dumps(transcript)  # Store full metadata
                ))
            
            logger.info(f"📊 Storing {len(complete_transcript_data)} complete transcripts for {quarter_info['quarter']} (skipped {skipped_complete_count} with insufficient content)")
            
            # Insert complete transcripts in batches
            batch_size = 100
            total_batches = (len(complete_transcript_data) + batch_size - 1) // batch_size
            logger.info(f"📊 Inserting {len(complete_transcript_data)} complete transcripts in {total_batches} batches...")
            
            for i in range(0, len(complete_transcript_data), batch_size):
                batch_num = (i // batch_size) + 1
                batch_end = min(i + batch_size, len(complete_transcript_data))
                progress_pct = (batch_num / total_batches) * 100
                
                logger.info(f"📦 Complete transcripts batch {batch_num}/{total_batches} ({progress_pct:.1f}%): Processing records {i+1}-{batch_end}...")
                
                batch_data = complete_transcript_data[i:batch_end]
                
                execute_values(
                    cursor,
                    """
                    INSERT INTO complete_transcripts (transcript_id, ticker, company_name, date, year, quarter, full_transcript, metadata)
                    VALUES %s
                    ON CONFLICT (transcript_id) DO UPDATE SET
                        ticker = EXCLUDED.ticker,
                        company_name = EXCLUDED.company_name,
                        date = EXCLUDED.date,
                        year = EXCLUDED.year,
                        quarter = EXCLUDED.quarter,
                        full_transcript = EXCLUDED.full_transcript,
                        metadata = EXCLUDED.metadata
                    """,
                    batch_data,
                    template=None,
                    page_size=50
                )
                
                conn.commit()
                logger.info(f"✅ Complete transcripts batch {batch_num}/{total_batches} completed ({progress_pct:.1f}%)")
                
                time.sleep(0.1)
            
            logger.info(f"✅ All {quarter_info['quarter']} complete transcripts inserted successfully")
            
            cursor.close()
            conn.close()
            
            logger.info(f"🎉 Successfully saved {len(complete_transcript_data)} complete transcripts for {quarter_info['quarter']} to database")
            
        except Exception as e:
            logger.error(f"Failed to save {quarter_info['quarter']} complete transcripts to database: {e}")
            raise
    
    def save_quarter_embeddings_to_db(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], quarter_info: Dict):
        """Save embeddings for a single quarter to PostgreSQL database with pgvector."""
        if not self.config["pgvector_url"]:
            logger.warning("DATABASE_URL not set, skipping database storage")
            return
        
        try:
            logger.info(f"💾 Connecting to database to store {quarter_info['quarter']} embeddings...")
            conn = psycopg2.connect(self.config["pgvector_url"])
            cursor = conn.cursor()
            logger.info("✅ Database connection established")
            
            # Ensure transcript_chunks table exists
            logger.info("🔍 Checking if transcript_chunks table exists...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcript_chunks (
                    id SERIAL PRIMARY KEY,
                    chunk_text TEXT NOT NULL,
                    embedding VECTOR(384),
                    metadata JSONB,
                    ticker VARCHAR(10),
                    year INTEGER,
                    quarter INTEGER,
                    transcript_id VARCHAR(100),
                    chunk_index INTEGER,
                    citation VARCHAR(200),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("✅ transcript_chunks table ensured")
            
            logger.info(f"📊 Storing {len(embeddings)} embeddings for {quarter_info['quarter']}")
            
            # Pre-serialize metadata to avoid repeated JSON serialization
            logger.info("📝 Pre-serializing metadata...")
            serialized_metadata = [json.dumps(meta) for meta in metadata]
            logger.info("✅ Metadata pre-serialized")
            
            # Prepare data in smaller chunks to avoid memory issues
            logger.info("📦 Preparing data for database insertion...")
            batch_size = 500
            total_batches = (len(embeddings) + batch_size - 1) // batch_size
            logger.info(f"📊 Inserting {len(embeddings)} records in {total_batches} batches of {batch_size}...")
            
            for i in range(0, len(embeddings), batch_size):
                batch_num = (i // batch_size) + 1
                batch_end = min(i + batch_size, len(embeddings))
                progress_pct = (batch_num / total_batches) * 100
                
                logger.info(f"📦 Batch {batch_num}/{total_batches} ({progress_pct:.1f}%): Processing records {i+1}-{batch_end}...")
                
                # Prepare batch data
                batch_data = []
                for j in range(i, batch_end):
                    meta = metadata[j]
                    batch_data.append((
                        meta['chunk_text'],
                        embeddings[j].tolist(),
                        serialized_metadata[j],
                        meta.get('ticker', 'unknown'),
                        meta.get('year'),
                        meta.get('quarter'),
                        meta.get('transcript_id', 'unknown'),
                        meta.get('chunk_index', j),
                        meta.get('citation', '')
                    ))
                
                # Show sample tickers from this batch
                sample_tickers = set(record[3] for record in batch_data[:5])
                logger.info(f"   📊 Sample tickers: {', '.join(sorted(sample_tickers))}")
                
                # Insert batch
                execute_values(
                    cursor,
                    """
                    INSERT INTO transcript_chunks (chunk_text, embedding, metadata, ticker, year, quarter, transcript_id, chunk_index, citation)
                    VALUES %s
                    """,
                    batch_data,
                    template=None,
                    page_size=100
                )
                
                # Commit each batch and clear batch data
                conn.commit()
                logger.info(f"✅ Batch {batch_num}/{total_batches} completed and committed ({progress_pct:.1f}%)")
                
                # Clear batch data from memory
                del batch_data
                
                # Small delay for visibility
                time.sleep(0.1)
            
            logger.info(f"✅ All {quarter_info['quarter']} embeddings inserted successfully")
            
            cursor.close()
            conn.close()
            
            logger.info(f"🎉 Successfully saved {len(embeddings)} embeddings for {quarter_info['quarter']} to database")
            
        except Exception as e:
            logger.error(f"Failed to save {quarter_info['quarter']} embeddings to database: {e}")
            raise
    
    def create_indexes_after_all_quarters(self):
        """Create database indexes after all quarters have been processed."""
        if not self.config["pgvector_url"]:
            logger.warning("DATABASE_URL not set, skipping index creation")
            return
        
        try:
            logger.info("🔍 Creating database indexes after all quarters processed...")
            conn = psycopg2.connect(self.config["pgvector_url"])
            cursor = conn.cursor()
            logger.info("✅ Database connection established for indexing")
            
            # Create vector similarity index for transcript_chunks
            logger.info("🔍 Creating vector similarity index...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS transcript_chunks_embedding_idx 
                ON transcript_chunks USING ivfflat (embedding vector_cosine_ops);
            """)
            logger.info("✅ Vector index ready")
            
            # Create indexes for transcript_chunks
            logger.info("🔍 Creating transcript_chunks indexes...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS transcript_chunks_ticker_idx 
                ON transcript_chunks (ticker);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS transcript_chunks_quarter_idx 
                ON transcript_chunks (year, quarter);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS transcript_chunks_transcript_id_idx 
                ON transcript_chunks (transcript_id);
            """)
            logger.info("✅ transcript_chunks indexes ready")
            
            # Create indexes for complete_transcripts
            logger.info("🔍 Creating complete_transcripts indexes...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS complete_transcripts_ticker_idx 
                ON complete_transcripts (ticker);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS complete_transcripts_quarter_idx 
                ON complete_transcripts (year, quarter);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS complete_transcripts_date_idx 
                ON complete_transcripts (date);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS complete_transcripts_company_name_idx 
                ON complete_transcripts (company_name);
            """)
            logger.info("✅ complete_transcripts indexes ready")
            
            # Final commit for indexes
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("🎉 All database indexes created successfully!")
            
        except Exception as e:
            logger.error(f"Failed to create database indexes: {e}")
            raise
    
    def process_all_quarters(self, max_files: int = 1000000, force_regenerate: bool = False):
        """Process all quarters and create embeddings with memory-efficient approach."""
        logger.info("🚀 Starting memory-efficient multi-quarter embeddings creation process")
        logger.info(f"📊 Processing quarters: {[q['quarter'] for q in self.quarters]}")
        logger.info(f"📊 Max files per quarter: {max_files}")
        
        quarter_stats = []
        
        # Format existing database once at the start
        self.format_existing_database()
        
        # Process each quarter individually to avoid memory issues
        for quarter_idx, quarter_info in enumerate(self.quarters):
            logger.info(f"\n📅 Processing quarter {quarter_idx + 1}/{len(self.quarters)}: {quarter_info['quarter']}")
            logger.info(f"📁 Folder: {quarter_info['folder']}")
            self.log_memory_usage(f"start of {quarter_info['quarter']}")
            
            # Load transcripts for this quarter
            transcripts = self.load_transcripts_for_quarter(quarter_info, max_files)
            if not transcripts:
                logger.warning(f"⚠️ No transcripts found for {quarter_info['quarter']}")
                continue
            
            self.log_memory_usage(f"after loading {quarter_info['quarter']} transcripts")
            
            # Create embeddings for this quarter
            embeddings, metadata = self.create_embeddings_for_quarter(transcripts, quarter_info)
            
            self.log_memory_usage(f"after creating {quarter_info['quarter']} embeddings")
            
            # Store complete transcripts first
            logger.info(f"💾 Storing {quarter_info['quarter']} complete transcripts to database...")
            self.save_complete_transcripts_to_db(transcripts, quarter_info)
            
            # Store embeddings immediately after creation to avoid memory accumulation
            logger.info(f"💾 Storing {quarter_info['quarter']} embeddings to database...")
            self.save_quarter_embeddings_to_db(embeddings, metadata, quarter_info)
            
            quarter_stats.append({
                'quarter': quarter_info['quarter'],
                'transcripts': len(transcripts),
                'chunks': len(embeddings),
                'folder': quarter_info['folder']
            })
            
            logger.info(f"✅ Completed {quarter_info['quarter']}: {len(transcripts)} transcripts → {len(embeddings)} chunks")
            
            # Force garbage collection to free memory
            del embeddings, metadata, transcripts
            self.force_garbage_collection(f"after {quarter_info['quarter']}")
        
        if not quarter_stats:
            logger.error("❌ No embeddings created for any quarter")
            return
        
        # Create indexes after all quarters are processed
        logger.info("\n🔍 Creating database indexes after all quarters...")
        self.create_indexes_after_all_quarters()
        
        # Print final statistics
        logger.info("\n🎉 Memory-efficient multi-quarter embeddings creation completed successfully!")
        logger.info("📊 Final Statistics:")
        total_transcripts = 0
        total_chunks = 0
        
        for stat in quarter_stats:
            logger.info(f"   📅 {stat['quarter']}: {stat['transcripts']} transcripts → {stat['chunks']} chunks")
            total_transcripts += stat['transcripts']
            total_chunks += stat['chunks']
        
        logger.info(f"   📊 Total: {total_transcripts} transcripts → {total_chunks} chunks")
        logger.info(f"   💾 Database: transcript_chunks table with {total_chunks} embeddings")
        logger.info(f"   💾 Database: complete_transcripts table with {total_transcripts} full transcripts")
        logger.info("   🔍 Indexes: Vector similarity, ticker, quarter, and company indexes created")
        logger.info("   ✅ RAG System: Ready for use with all quarters (2023-2025)!")
        logger.info("   📚 Complete transcripts available for full-text search and analysis!")


def main():
    """Main function."""
    # Check environment variables first
    pgvector_url = os.getenv("DATABASE_URL", "")
    if not pgvector_url:
        logger.error("❌ DATABASE_URL environment variable not set!")
        logger.error("Please ensure your .env file contains the DATABASE_URL database URL")
        logger.error("Current working directory:", os.getcwd())
        logger.error("Looking for .env file at:", os.path.abspath("../.env"))
        return
    
    logger.info("✅ Environment variables loaded successfully")
    logger.info(f"🔗 Database URL: {pgvector_url[:50]}...")

    parser = argparse.ArgumentParser(description='Format database and create embeddings for earnings transcripts (2023-2025)')
    parser.add_argument('--max-files', type=int, default=1000000, help='Maximum files to process per quarter')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regenerate embeddings')
    
    args = parser.parse_args()
    
    try:
        processor = EmbeddingProcessor()
        processor.process_all_quarters(args.max_files, args.force_regenerate)
        
    except Exception as e:
        logger.error(f"Multi-quarter embeddings creation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
