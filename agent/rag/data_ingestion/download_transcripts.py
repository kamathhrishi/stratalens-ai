#!/usr/bin/env python3
"""
Downloads earnings transcripts from API Ninjas for multiple quarters (2020-2025).
Saves transcripts as JSON files to be processed by create_and_store_embeddings.py.
"""

import os
import sys
import time
import json
import requests
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import gc
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('earnings_transcripts_fetch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def download_single_transcript(ticker: str, year: int, quarter: int, api_key: str) -> Dict:
    """
    Download a single transcript from API Ninjas - designed for multiprocessing.
    Returns the transcript data with metadata.
    """
    try:
        api_url = f'https://api.api-ninjas.com/v1/earningstranscript?ticker={ticker}&year={year}&quarter={quarter}'
        headers = {'X-Api-Key': api_key}
        
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return {
                    'ticker': ticker,
                    'year': year,
                    'quarter': quarter,
                    'data': data,
                    'fetched_at': datetime.now().isoformat(),
                    'status': 'success'
                }
            else:
                return {
                    'ticker': ticker,
                    'year': year,
                    'quarter': quarter,
                    'data': None,
                    'fetched_at': datetime.now().isoformat(),
                    'status': 'no_transcript'
                }
        elif response.status_code == 404:
            return {
                'ticker': ticker,
                'year': year,
                'quarter': quarter,
                'data': None,
                'fetched_at': datetime.now().isoformat(),
                'status': 'no_transcript'
            }
        elif response.status_code == 429:
            return {
                'ticker': ticker,
                'year': year,
                'quarter': quarter,
                'data': None,
                'fetched_at': datetime.now().isoformat(),
                'status': 'rate_limit'
            }
        else:
            return {
                'ticker': ticker,
                'year': year,
                'quarter': quarter,
                'data': None,
                'fetched_at': datetime.now().isoformat(),
                'status': 'api_error',
                'error_code': response.status_code,
                'error_message': response.text
            }
            
    except requests.exceptions.Timeout:
        return {
            'ticker': ticker,
            'year': year,
            'quarter': quarter,
            'data': None,
            'fetched_at': datetime.now().isoformat(),
            'status': 'timeout'
        }
    except requests.exceptions.RequestException as e:
        return {
            'ticker': ticker,
            'year': year,
            'quarter': quarter,
            'data': None,
            'fetched_at': datetime.now().isoformat(),
            'status': 'network_error',
            'error_message': str(e)
        }
    except Exception as e:
        return {
            'ticker': ticker,
            'year': year,
            'quarter': quarter,
            'data': None,
            'fetched_at': datetime.now().isoformat(),
            'status': 'other_error',
            'error_message': str(e)
        }

class TranscriptDownloader:
    def __init__(self, db_path: str = "financial_data_new.duckdb", output_folder: str = "earnings_transcripts",
                 ticker_file: str = "us_tickers.txt", max_workers: int = None):
        self.db_path = db_path
        self.output_folder = Path(output_folder)
        self.ticker_file = ticker_file

        # Read API key from environment
        self.api_key = os.getenv("API_NINJAS_KEY")
        if not self.api_key:
            raise ValueError("API_NINJAS_KEY environment variable not set. Please add it to your .env file.")

        self.max_workers = max_workers or 3  # Optimal for API limits and memory
        
        # Generate quarters for last 6 years (2020-2025)
        self.quarters = self._generate_quarters_for_6_years()
        
        # Create output folder
        self.output_folder.mkdir(exist_ok=True)
        
        # Build set of already downloaded transcripts
        self.downloaded_transcripts = self._get_downloaded_transcripts()
        
        # Statistics - track per quarter
        self.stats = {
            'total_tickers_processed': 0,
            'total_quarters_processed': 0,
            'quarter_stats': {},
            'skipped_downloaded': 0
        }
        
        # Error tracking
        self.errors = []
    
    def _generate_quarters_for_6_years(self) -> List[tuple]:
        """Generate list of (year, quarter) tuples for the last 6 years (2020-2025)"""
        from datetime import datetime
        
        current_year = datetime.now().year
        current_quarter = ((datetime.now().month - 1) // 3) + 1
        
        quarters = []
        
        # Generate quarters going back 6 years (2020-2025)
        for year_offset in range(6):
            year = current_year - year_offset
            
            # For current year, only include quarters up to current quarter
            if year == current_year:
                max_quarter = current_quarter
            else:
                max_quarter = 4
            
            # For previous years, include all 4 quarters
            for quarter in range(1, max_quarter + 1):
                quarters.append((year, quarter))
        
        logger.info(f"Generated {len(quarters)} quarters to process: {quarters}")
        return quarters
    
    def _get_downloaded_transcripts(self) -> set:
        """Get set of already downloaded transcripts"""
        downloaded = set()
        
        if not self.output_folder.exists():
            logger.info("📁 Output folder doesn't exist yet - no transcripts to skip")
            return downloaded
        
        logger.info("🔍 Scanning for already downloaded transcripts...")
        
        for file_path in self.output_folder.glob("*.json"):
            try:
                filename = file_path.name
                # Parse filename to extract ticker, year, quarter
                # Format: {ticker}_transcript_{year}_Q{quarter}.json or {ticker}_status_{status}_{year}_Q{quarter}.json
                parts = filename.replace('.json', '').split('_')
                
                if len(parts) >= 4:
                    if 'transcript' in parts:
                        # Success transcript: TICKER_transcript_YEAR_QQUARTER
                        ticker = parts[0]
                        year = int(parts[2])
                        quarter = int(parts[3].replace('Q', ''))
                        downloaded.add((ticker, year, quarter))
                    elif 'status' in parts:
                        # Status file: TICKER_status_STATUS_YEAR_QQUARTER
                        ticker = parts[0]
                        year = int(parts[3])
                        quarter = int(parts[4].replace('Q', ''))
                        downloaded.add((ticker, year, quarter))
                        
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse filename {filename}: {e}")
                continue
        
        logger.info(f"✅ Found {len(downloaded)} already downloaded transcripts")
        return downloaded
        
    def get_all_tickers(self) -> List[str]:
        """Get all ticker symbols from database or ticker file"""
        
        # If ticker file is provided, use that
        if self.ticker_file:
            return self.get_tickers_from_file()
        
        # Otherwise, try database
        return self.get_tickers_from_database()
    
    def get_tickers_from_file(self) -> List[str]:
        """Get ticker symbols from a text file"""
        try:
            if not os.path.exists(self.ticker_file):
                logger.error(f"❌ Ticker file not found: {self.ticker_file}")
                return []
            
            with open(self.ticker_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            
            logger.info(f"📋 Loaded {len(tickers)} ticker symbols from {self.ticker_file}")
            return tickers
            
        except Exception as e:
            logger.error(f"Error loading tickers from file: {e}")
            return []
    
    def get_tickers_from_database(self) -> List[str]:
        """Get all ticker symbols from the company_profiles table"""
        try:
            logger.info(f"Connecting to database: {self.db_path}")
            conn = duckdb.connect(self.db_path)
            
            # Check if company_profiles table exists
            tables = conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            
            if 'company_profiles' not in table_names:
                logger.error("company_profiles table not found in database")
                logger.info(f"Available tables: {table_names}")
                return []
            
            # Get all ticker symbols
            query = "SELECT DISTINCT symbol FROM company_profiles WHERE symbol IS NOT NULL AND symbol != '' ORDER BY symbol"
            result = conn.execute(query).fetchall()
            
            tickers = [row[0] for row in result if row[0]]
            conn.close()
            
            logger.info(f"Found {len(tickers)} ticker symbols in database")
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting tickers from database: {e}")
            return []
    
    def save_transcript(self, transcript_data: Dict):
        """Save transcript data to file"""
        try:
            ticker = transcript_data['ticker']
            year = transcript_data['year']
            quarter = transcript_data['quarter']
            status = transcript_data['status']
            
            # Clean ticker name to avoid filesystem issues
            clean_ticker = ticker.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            
            # Create filename based on status
            if status == 'success':
                filename = f"{clean_ticker}_transcript_{year}_Q{quarter}.json"
            else:
                filename = f"{clean_ticker}_status_{status}_{year}_Q{quarter}.json"
            
            filepath = self.output_folder / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"💾 Saved {filename}")
            
        except Exception as e:
            logger.error(f"Error saving transcript for {ticker}: {e}")
            self.errors.append({
                'ticker': ticker,
                'error': f"Save error: {e}",
                'timestamp': datetime.now().isoformat()
            })
    
    def update_stats(self, transcript_data: Dict, year: int, quarter: int):
        """Update statistics based on transcript fetch result"""
        status = transcript_data['status']
        quarter_key = f"{year}_Q{quarter}"
        
        # Initialize quarter stats if not exists
        if quarter_key not in self.stats['quarter_stats']:
            self.stats['quarter_stats'][quarter_key] = {
                'total_tickers': 0,
                'successful_fetches': 0,
                'failed_fetches': 0,
                'no_transcript_available': 0,
                'api_errors': 0,
                'other_errors': 0
            }
        
        if status == 'success':
            self.stats['quarter_stats'][quarter_key]['successful_fetches'] += 1
        elif status == 'no_transcript':
            self.stats['quarter_stats'][quarter_key]['no_transcript_available'] += 1
        elif status == 'api_error':
            self.stats['quarter_stats'][quarter_key]['api_errors'] += 1
        else:
            self.stats['quarter_stats'][quarter_key]['other_errors'] += 1
        
        self.stats['quarter_stats'][quarter_key]['total_tickers'] += 1
    
    def save_summary_report(self):
        """Save a summary report of the fetch operation"""
        try:
            # Calculate overall success rate
            total_successful = sum(q['successful_fetches'] for q in self.stats['quarter_stats'].values())
            total_processed = sum(q['total_tickers'] for q in self.stats['quarter_stats'].values())
            overall_success_rate = (total_successful / total_processed * 100) if total_processed > 0 else 0
            
            summary = {
                'fetch_summary': {
                    'date': datetime.now().isoformat(),
                    'quarters_processed': len(self.quarters),
                    'quarters_list': [f"{year}_Q{quarter}" for year, quarter in self.quarters],
                    'output_folder': str(self.output_folder),
                    'database_path': self.db_path
                },
                'overall_statistics': {
                    'total_quarters_processed': len(self.stats['quarter_stats']),
                    'total_tickers_processed': total_processed,
                    'total_successful_fetches': total_successful,
                    'overall_success_rate': f"{overall_success_rate:.2f}%"
                },
                'quarter_statistics': self.stats['quarter_stats'],
                'errors': self.errors[:100],  # Limit to first 100 errors
            }
            
            summary_file = self.output_folder / f"fetch_summary_6_years_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Summary report saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving summary report: {e}")
    
    def print_progress(self, current: int, total: int, year: int, quarter: int):
        """Print progress information"""
        percentage = (current / total * 100) if total > 0 else 0
        quarter_key = f"{year}_Q{quarter}"
        
        if quarter_key in self.stats['quarter_stats']:
            stats = self.stats['quarter_stats'][quarter_key]
            logger.info(f"📈 Progress {year} Q{quarter}: {current}/{total} ({percentage:.1f}%) - "
                       f"Success: {stats['successful_fetches']}, "
                       f"Failed: {stats['api_errors'] + stats['other_errors']}, "
                       f"No transcript: {stats['no_transcript_available']}")
        else:
            logger.info(f"📈 Progress {year} Q{quarter}: {current}/{total} ({percentage:.1f}%)")
    
    def run(self, max_tickers: Optional[int] = None, delay: float = 0.2, batch_size: int = 5000):
        """Main execution method with multiprocessing"""
        logger.info("🚀 Starting earnings transcript fetch operation for last 6 years (2020-2025)")
        logger.info(f"📁 Output folder: {self.output_folder}")
        logger.info(f"📅 Quarters to process: {len(self.quarters)}")
        logger.info(f"📋 Quarters: {[f'{year}_Q{quarter}' for year, quarter in self.quarters]}")
        logger.info(f"🔥 Using {self.max_workers} parallel workers")
        
        # Get all tickers
        tickers = self.get_all_tickers()
        if not tickers:
            logger.error("❌ No tickers found")
            return
        
        # Limit tickers if specified
        if max_tickers:
            tickers = tickers[:max_tickers]
            logger.info(f"🔢 Limited to {max_tickers} tickers")
        
        logger.info(f"🎯 Processing {len(tickers)} tickers across {len(self.quarters)} quarters")
        
        # Create all combinations of tickers and quarters, skipping already downloaded
        all_combinations = []
        skipped_count = 0
        
        for ticker in tickers:
            for year, quarter in self.quarters:
                # Clean ticker name to match the filename format
                clean_ticker = ticker.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                
                if (clean_ticker, year, quarter) not in self.downloaded_transcripts:
                    all_combinations.append((ticker, year, quarter))
                else:
                    skipped_count += 1
        
        logger.info(f"⏭️ Skipped {skipped_count} already downloaded transcripts")
        self.stats['skipped_downloaded'] = skipped_count
        
        logger.info(f"📊 Total API calls to make: {len(all_combinations)}")
        
        # Process combinations in batches to manage memory
        if len(all_combinations) > batch_size:
            logger.info(f"🔄 Processing in batches of {batch_size} to manage memory")
            for i in range(0, len(all_combinations), batch_size):
                batch = all_combinations[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(all_combinations) + batch_size - 1) // batch_size
                logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} combinations)")
                self.process_combinations_parallel(batch, delay)
                # Force garbage collection after each batch
                gc.collect()
                logger.debug("🧹 Batch completed, garbage collection performed")
        else:
            # Process all combinations at once if small enough
            self.process_combinations_parallel(all_combinations, delay)
        
        # Save summary report
        self.save_summary_report()
        
        # Print final statistics
        self.print_final_statistics()
    
    def process_combinations_parallel(self, combinations: List[tuple], delay: float):
        """Process all ticker-quarter combinations in parallel"""
        processed_count = 0
        total_combinations = len(combinations)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_combination = {
                executor.submit(download_single_transcript, ticker, year, quarter, self.api_key): (ticker, year, quarter)
                for ticker, year, quarter in combinations
            }
            
            # Process completed tasks
            for future in as_completed(future_to_combination):
                ticker, year, quarter = future_to_combination[future]
                
                try:
                    # Get the result
                    transcript_data = future.result()
                    
                    if transcript_data:
                        # Save transcript
                        self.save_transcript(transcript_data)
                        
                        # Update statistics
                        self.update_stats(transcript_data, year, quarter)
                    
                    processed_count += 1
                    
                    # Print progress every 100 completions
                    if processed_count % 100 == 0 or processed_count == total_combinations:
                        percentage = (processed_count / total_combinations) * 100
                        
                        # Calculate current stats
                        current_stats = self.get_current_stats()
                        logger.info(f"📈 Progress: {processed_count}/{total_combinations} ({percentage:.1f}%) - "
                                   f"Success: {current_stats['successful']}, "
                                   f"Failed: {current_stats['failed']}, "
                                   f"No transcript: {current_stats['no_transcript']}")
                        
                        # Force garbage collection every 1000 completions to manage memory
                        if processed_count % 1000 == 0:
                            gc.collect()
                            logger.debug("🧹 Garbage collection performed")
                    
                    # Small delay to avoid overwhelming the system
                    if delay > 0:
                        time.sleep(delay)
                        
                except KeyboardInterrupt:
                    logger.info("⏹️ Operation interrupted by user")
                    # Cancel remaining futures
                    for f in future_to_combination:
                        f.cancel()
                    break
                except Exception as e:
                    logger.error(f"💥 Unexpected error processing {ticker} {year} Q{quarter}: {e}")
                    self.errors.append({
                        'ticker': ticker,
                        'year': year,
                        'quarter': quarter,
                        'error': f"Processing error: {e}",
                        'timestamp': datetime.now().isoformat()
                    })
                    processed_count += 1
    
    def get_current_stats(self) -> Dict:
        """Get current statistics for progress reporting"""
        total_successful = sum(q['successful_fetches'] for q in self.stats['quarter_stats'].values())
        total_no_transcript = sum(q['no_transcript_available'] for q in self.stats['quarter_stats'].values())
        total_api_errors = sum(q['api_errors'] for q in self.stats['quarter_stats'].values())
        total_other_errors = sum(q['other_errors'] for q in self.stats['quarter_stats'].values())
        
        return {
            'successful': total_successful,
            'no_transcript': total_no_transcript,
            'failed': total_api_errors + total_other_errors
        }
    
    def print_final_statistics(self):
        """Print final statistics"""
        logger.info("🎉 Fetch operation completed!")
        
        # Calculate and display overall statistics
        total_processed = sum(q['total_tickers'] for q in self.stats['quarter_stats'].values())
        total_successful = sum(q['successful_fetches'] for q in self.stats['quarter_stats'].values())
        total_no_transcript = sum(q['no_transcript_available'] for q in self.stats['quarter_stats'].values())
        total_api_errors = sum(q['api_errors'] for q in self.stats['quarter_stats'].values())
        total_other_errors = sum(q['other_errors'] for q in self.stats['quarter_stats'].values())
        
        logger.info(f"📊 Overall Final Statistics:")
        logger.info(f"   Quarters processed: {len(self.stats['quarter_stats'])}")
        logger.info(f"   Total tickers processed: {total_processed}")
        logger.info(f"   Successful fetches: {total_successful}")
        logger.info(f"   No transcript available: {total_no_transcript}")
        logger.info(f"   API errors: {total_api_errors}")
        logger.info(f"   Other errors: {total_other_errors}")
        logger.info(f"   Skipped (already downloaded): {self.stats['skipped_downloaded']}")
        
        if total_processed > 0:
            success_rate = (total_successful / total_processed) * 100
            logger.info(f"   Overall success rate: {success_rate:.2f}%")
        
        # Display per-quarter statistics
        logger.info(f"📊 Per-Quarter Statistics:")
        for quarter_key, stats in self.stats['quarter_stats'].items():
            quarter_success_rate = (stats['successful_fetches'] / stats['total_tickers'] * 100) if stats['total_tickers'] > 0 else 0
            logger.info(f"   {quarter_key}: {stats['successful_fetches']}/{stats['total_tickers']} ({quarter_success_rate:.1f}%)")
        
        if self.errors:
            logger.warning(f"⚠️ {len(self.errors)} errors occurred. Check the log file for details.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch earnings transcripts for all tickers')
    parser.add_argument('--db-path', default='financial_data_new.duckdb', 
                       help='Path to the DuckDB database file')
    parser.add_argument('--output-folder', default='earnings_transcripts',
                       help='Output folder for transcripts')
    parser.add_argument('--max-tickers', type=int, default=None,
                       help='Maximum number of tickers to process (for testing)')
    parser.add_argument('--ticker-file', type=str, default='us_tickers.txt',
                       help='Text file containing ticker symbols (one per line). Default: us_tickers.txt')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3 - optimal for API limits)')
    parser.add_argument('--delay', type=float, default=0.2,
                       help='Delay between batches in seconds (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=5000,
                       help='Number of combinations to process per batch (default: 5000)')
    
    args = parser.parse_args()
    
    # Check if ticker file exists (default behavior)
    if not os.path.exists(args.ticker_file):
        logger.error(f"❌ Ticker file not found: {args.ticker_file}")
        logger.info("💡 Run 'python3 fetch_us_tickers.py' first to create the ticker file")
        sys.exit(1)
    
    # Create downloader and run
    downloader = TranscriptDownloader(
        db_path=args.db_path,
        output_folder=args.output_folder,
        ticker_file=args.ticker_file,
        max_workers=args.workers
    )
    
    try:
        downloader.run(max_tickers=args.max_tickers, delay=args.delay, batch_size=args.batch_size)
    except KeyboardInterrupt:
        logger.info("⏹️ Operation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 