#!/usr/bin/env python3
"""
S&P 500 10-K SEC Filings Ingestion Script

This script automatically downloads and processes 10-K filings for all S&P 500 companies
from the last 1 year (December 2024 to December 2025).

Features:
- Automatically fetches current S&P 500 ticker list from Wikipedia
- Fallback to SPY ETF holdings if Wikipedia fails
- Uses sophisticated hierarchical processing (from ingest_10k_to_database.py)
- Automatically skips already-processed ticker-year combinations
- Parallel processing for faster ingestion with memory management
- Each ticker processed in a separate process (prevents memory accumulation)
- Aggressive garbage collection to prevent memory leaks
- Automatic cleanup of temporary files
- Comprehensive error handling and logging

Usage:
    # Ingest all S&P 500 companies (recommended: use parallel processing)
    python ingest_sp500_10k.py --workers 5

    # Sequential processing (slower but simpler)
    python ingest_sp500_10k.py --sequential

    # Process subset for testing
    python ingest_sp500_10k.py --max-tickers 10 --workers 3

    # Resume from specific position
    python ingest_sp500_10k.py --skip-first 100 --workers 5

    # Custom lookback period
    python ingest_sp500_10k.py --lookback-years 2 --workers 5

Requirements:
    pip install datamule scikit-learn pandas openpyxl requests
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add parent directory to path to import from ingest_10k_to_database
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pandas as pd
except ImportError:
    print("Error: pandas library not found. Install with: pip install pandas")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("Error: requests library not found. Install with: pip install requests")
    sys.exit(1)

# Import the sophisticated ingestion module
try:
    from ingest_10k_to_database import (
        DatabaseIntegration,
        ingest_ticker,
        ingest_ticker_worker,
        cleanup_temp_files
    )
except ImportError as e:
    print(f"Error: Failed to import from ingest_10k_to_database.py: {e}")
    print("Make sure ingest_10k_to_database.py is in the same directory.")
    sys.exit(1)

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
        logging.FileHandler('sp500_10k_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SP500TickerFetcher:
    """Fetches S&P 500 ticker symbols from multiple sources with fallbacks."""

    def __init__(self):
        self.tickers = []

    def fetch_from_wikipedia(self) -> Optional[List[str]]:
        """
        Fetch S&P 500 tickers from Wikipedia.
        This is the most reliable and up-to-date source.
        """
        logger.info("📊 Fetching S&P 500 tickers from Wikipedia...")
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]

            # The first table contains the S&P 500 companies
            # Symbol column contains the ticker symbols
            tickers = df['Symbol'].tolist()

            # Clean up tickers (remove any whitespace)
            tickers = [ticker.strip().upper() for ticker in tickers if ticker]

            logger.info(f"✅ Successfully fetched {len(tickers)} tickers from Wikipedia")
            return tickers

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch from Wikipedia: {e}")
            return None

    def fetch_from_spy_etf(self) -> Optional[List[str]]:
        """
        Fetch S&P 500 tickers from SPY ETF holdings (State Street Global Advisors).
        This is an official source but may be slightly delayed.
        """
        logger.info("📊 Fetching S&P 500 tickers from SPY ETF holdings...")
        try:
            url = 'https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx'

            # Download with custom headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Save to temporary file
            temp_file = Path('/tmp/spy_holdings.xlsx')
            with open(temp_file, 'wb') as f:
                f.write(response.content)

            # Read Excel file (skip first few rows which are headers)
            df = pd.read_excel(temp_file, engine='openpyxl', skiprows=4)

            # Ticker column name may vary, try common variations
            ticker_column = None
            for col in ['Ticker', 'Symbol', 'ticker', 'symbol']:
                if col in df.columns:
                    ticker_column = col
                    break

            if ticker_column is None:
                logger.warning("⚠️ Could not find ticker column in SPY holdings")
                return None

            tickers = df[ticker_column].tolist()

            # Clean up tickers
            tickers = [str(ticker).strip().upper() for ticker in tickers if pd.notna(ticker)]

            # Remove temp file
            temp_file.unlink()

            logger.info(f"✅ Successfully fetched {len(tickers)} tickers from SPY ETF")
            return tickers

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch from SPY ETF: {e}")
            return None

    def fetch_from_slickcharts(self) -> Optional[List[str]]:
        """
        Fetch S&P 500 tickers from SlickCharts.
        Backup option if other sources fail.
        """
        logger.info("📊 Fetching S&P 500 tickers from SlickCharts...")
        try:
            url = 'https://www.slickcharts.com/sp500'

            # Custom headers required for SlickCharts
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            tables = pd.read_html(url, storage_options=headers)
            df = tables[0]

            # Symbol column contains tickers
            tickers = df['Symbol'].tolist()

            # Clean up tickers
            tickers = [ticker.strip().upper() for ticker in tickers if ticker]

            logger.info(f"✅ Successfully fetched {len(tickers)} tickers from SlickCharts")
            return tickers

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch from SlickCharts: {e}")
            return None

    def get_sp500_tickers(self) -> List[str]:
        """
        Get S&P 500 tickers with automatic fallback to multiple sources.
        Tries sources in order of reliability.
        """
        logger.info("🚀 Starting S&P 500 ticker fetch...")

        # Try sources in order of preference
        sources = [
            ('Wikipedia', self.fetch_from_wikipedia),
            ('SPY ETF', self.fetch_from_spy_etf),
            ('SlickCharts', self.fetch_from_slickcharts)
        ]

        for source_name, fetch_func in sources:
            tickers = fetch_func()
            if tickers and len(tickers) >= 400:  # S&P 500 should have ~500 tickers
                logger.info(f"✅ Successfully fetched {len(tickers)} tickers from {source_name}")
                self.tickers = sorted(list(set(tickers)))  # Remove duplicates and sort
                return self.tickers

        # All sources failed
        logger.error("❌ Failed to fetch S&P 500 tickers from all sources")
        raise Exception("Could not fetch S&P 500 ticker list from any source")

    def save_to_file(self, filename: str = 'sp500_tickers.txt'):
        """Save fetched tickers to a file for reference."""
        if not self.tickers:
            logger.warning("⚠️ No tickers to save")
            return

        filepath = Path(__file__).parent / filename
        try:
            with open(filepath, 'w') as f:
                for ticker in self.tickers:
                    f.write(f"{ticker}\n")
            logger.info(f"💾 Saved {len(self.tickers)} tickers to {filepath}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save tickers to file: {e}")


def main():
    # Fix CUDA multiprocessing issue
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(
        description='Ingest S&P 500 10-K SEC filings with sophisticated processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest all S&P 500 companies with 5 parallel workers (recommended)
  python ingest_sp500_10k.py --workers 5

  # Process first 50 companies for testing
  python ingest_sp500_10k.py --max-tickers 50 --workers 3

  # Resume from ticker #200
  python ingest_sp500_10k.py --skip-first 200 --workers 5

  # Get 2 years of data instead of 1
  python ingest_sp500_10k.py --lookback-years 2 --workers 5
        """
    )

    parser.add_argument('--lookback-years', type=int, default=1,
                       help='Number of years to look back (default: 1)')
    parser.add_argument('--max-tickers', type=int,
                       help='Limit number of tickers to process (for testing)')
    parser.add_argument('--skip-first', type=int, default=0,
                       help='Skip first N tickers (for resuming)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel workers (default: 3, recommended: 3-5)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run sequentially without parallel processing')
    parser.add_argument('--save-tickers', action='store_true',
                       help='Save fetched ticker list to sp500_tickers.txt')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Timeout per ticker in minutes (default: 30)')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry failed tickers once')
    parser.add_argument('--skip-tickers', type=str, nargs='+',
                       help='Specific tickers to skip (e.g., --skip-tickers AEP AES)')
    args = parser.parse_args()

    # Banner
    logger.info("=" * 80)
    logger.info("S&P 500 10-K FILINGS INGESTION")
    logger.info("=" * 80)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Lookback period: {args.lookback_years} year(s)")
    logger.info(f"Processing mode: {'Sequential' if args.sequential else f'Parallel ({args.workers} workers)'}")
    logger.info("=" * 80)

    # Step 1: Fetch S&P 500 tickers
    logger.info("\n📊 STEP 1: Fetching S&P 500 ticker list")
    logger.info("-" * 80)

    try:
        fetcher = SP500TickerFetcher()
        tickers = fetcher.get_sp500_tickers()

        # Optionally save to file
        if args.save_tickers:
            fetcher.save_to_file()

        logger.info(f"✅ Successfully fetched {len(tickers)} S&P 500 tickers")
        logger.info(f"📋 Sample tickers: {', '.join(tickers[:10])}...")

    except Exception as e:
        logger.error(f"❌ Failed to fetch S&P 500 tickers: {e}")
        logger.error("Please check your internet connection and try again.")
        sys.exit(1)

    # Apply skip and max limits
    original_count = len(tickers)

    # Filter out blacklisted tickers
    if args.skip_tickers:
        skip_set = set(ticker.upper() for ticker in args.skip_tickers)
        before_filter = len(tickers)
        tickers = [t for t in tickers if t not in skip_set]
        logger.info(f"🚫 Skipping {len(skip_set)} blacklisted tickers: {', '.join(sorted(skip_set))}")
        logger.info(f"   Filtered out {before_filter - len(tickers)} tickers from list")

    if args.skip_first > 0:
        logger.info(f"⏭️  Skipping first {args.skip_first} tickers")
        tickers = tickers[args.skip_first:]

    if args.max_tickers:
        logger.info(f"🔢 Limiting to {args.max_tickers} tickers")
        tickers = tickers[:args.max_tickers]

    logger.info(f"📊 Processing {len(tickers)} of {original_count} S&P 500 companies")

    # Step 2: Initialize database
    logger.info("\n🗄️  STEP 2: Initializing database")
    logger.info("-" * 80)

    # Get database URL (must be consistent across all processes)
    db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error("❌ PG_VECTOR or DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info(f"Database: {db_url[:50]}...")

    try:
        db_integration = DatabaseIntegration(db_url=db_url)
        db_integration.ensure_tables()
        logger.info("✅ Database tables ready")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        logger.error("Please check your PG_VECTOR or DATABASE_URL environment variable.")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

    # Step 3: Process tickers
    logger.info("\n🚀 STEP 3: Ingesting 10-K filings")
    logger.info("-" * 80)
    logger.info(f"Processing strategy: Hierarchical extraction + contextual chunking + table identification")
    logger.info(f"Storage: PostgreSQL with pgvector (tables: ten_k_chunks, ten_k_tables)")
    logger.info("-" * 80)
    results = []
    start_time = time.time()

    if args.sequential:
        # Sequential processing
        logger.info("🔄 Running in sequential mode\n")

        for idx, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"\n{'=' * 70}")
                logger.info(f"📊 Progress: {idx}/{len(tickers)} ({idx/len(tickers)*100:.1f}%)")
                logger.info(f"{'=' * 70}")

                result = ingest_ticker(ticker, args.lookback_years, db_integration)
                results.append(result)

                # Progress update every 10 tickers
                if idx % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / idx
                    remaining = avg_time * (len(tickers) - idx)
                    successful = len([r for r in results if r.get('filings_processed', 0) > 0])

                    logger.info(f"\n⏱️  PROGRESS UPDATE")
                    logger.info(f"Processed: {idx}/{len(tickers)} ({idx/len(tickers)*100:.1f}%)")
                    logger.info(f"Successful: {successful}")
                    logger.info(f"Time elapsed: {elapsed/60:.1f}m")
                    logger.info(f"Estimated remaining: {remaining/60:.1f}m")

            except Exception as e:
                logger.error(f"❌ Failed to ingest {ticker}: {e}")
                results.append({'ticker': ticker, 'error': str(e)})

    else:
        # Parallel processing
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

        logger.info(f"🔥 Running with {args.workers} parallel workers")
        logger.info(f"⏱️  Timeout per ticker: {args.timeout} minutes\n")

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

            # Process completed tasks
            for future in as_completed(future_to_ticker, timeout=None):
                ticker = future_to_ticker[future]
                completed += 1

                try:
                    # Get result with a timeout buffer (add 2 minutes for overhead)
                    result = future.result(timeout=(args.timeout * 60) + 120)
                    results.append(result)

                    # Progress update every 10 completions or on error
                    if completed % 10 == 0 or 'error' in result:
                        elapsed = time.time() - start_time
                        avg_time = elapsed / completed
                        remaining = avg_time * (len(tickers) - completed)
                        successful = len([r for r in results if r.get('filings_processed', 0) > 0])
                        failed = len([r for r in results if 'error' in r])

                        logger.info(f"\n⏱️  PROGRESS UPDATE")
                        logger.info(f"Completed: {completed}/{len(tickers)} ({completed/len(tickers)*100:.1f}%)")
                        logger.info(f"Successful: {successful}")
                        logger.info(f"Failed: {failed}")
                        logger.info(f"Time elapsed: {elapsed/60:.1f}m")
                        logger.info(f"Estimated remaining: {remaining/60:.1f}m\n")

                except TimeoutError:
                    logger.error(f"⏱️ Future timeout for {ticker} (process didn't respond)")
                    results.append({'ticker': ticker, 'error': 'Future timeout - process hung'})
                except Exception as e:
                    logger.error(f"❌ Worker exception for {ticker}: {e}")
                    results.append({'ticker': ticker, 'error': str(e)})

        logger.info("🎉 All workers completed!")

        # Retry failed tickers if requested
        if args.retry_failed:
            failed_tickers = [r['ticker'] for r in results if 'error' in r]
            if failed_tickers:
                logger.info(f"\n🔄 RETRYING FAILED TICKERS")
                logger.info(f"Retrying {len(failed_tickers)} failed tickers...")
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
                logger.info(f"🔄 Retry complete: {successful_retries}/{len(failed_tickers)} succeeded\n")

    # Step 4: Summary
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("📊 FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total execution time: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    logger.info(f"Tickers processed: {len(results)}")

    # Calculate statistics
    total_filings = sum(r.get('filings_processed', 0) for r in results)
    total_skipped = sum(r.get('filings_skipped', 0) for r in results)
    total_chunks = sum(r.get('chunks_stored', 0) for r in results)
    total_tables = sum(r.get('tables_stored', 0) for r in results)
    successful = [r for r in results if 'error' not in r and r.get('filings_processed', 0) > 0]
    skipped_all = [r for r in results if 'error' not in r and r.get('filings_processed', 0) == 0 and r.get('filings_skipped', 0) > 0]
    failed = [r for r in results if 'error' in r]
    no_data = [r for r in results if 'error' not in r and r.get('filings_processed', 0) == 0 and r.get('filings_skipped', 0) == 0]

    logger.info(f"\n📈 DATA INGESTED:")
    logger.info(f"  Total 10-K filings processed: {total_filings}")
    logger.info(f"  Total 10-K filings skipped: {total_skipped}")
    logger.info(f"  Total text chunks: {total_chunks:,}")
    logger.info(f"  Total tables extracted: {total_tables:,}")
    logger.info(f"\n📊 RESULTS BREAKDOWN:")
    logger.info(f"  ✅ Successful: {len(successful)} companies")
    logger.info(f"  ⏭️  Already processed (skipped): {len(skipped_all)} companies")
    logger.info(f"  ⚠️  No data found: {len(no_data)} companies")
    logger.info(f"  ❌ Failed: {len(failed)} companies")

    # Show sample successful ingestions
    if successful:
        logger.info(f"\n✅ SAMPLE SUCCESSFUL INGESTIONS:")
        for result in successful[:15]:
            skipped_msg = f", {result.get('filings_skipped', 0)} skipped" if result.get('filings_skipped', 0) > 0 else ""
            logger.info(f"  {result['ticker']}: {result['filings_processed']} filings{skipped_msg}, "
                       f"{result['chunks_stored']:,} chunks, {result['tables_stored']} tables")
        if len(successful) > 15:
            logger.info(f"  ... and {len(successful) - 15} more")

    # Show companies that were entirely skipped
    if skipped_all:
        logger.info(f"\n⏭️  COMPANIES WITH ALL FILINGS ALREADY PROCESSED:")
        sample_skipped = [r['ticker'] for r in skipped_all[:20]]
        logger.info(f"  {', '.join(sample_skipped)}")
        if len(skipped_all) > 20:
            logger.info(f"  ... and {len(skipped_all) - 20} more")

    # Show companies with no data
    if no_data:
        logger.info(f"\n⚠️  COMPANIES WITH NO DATA (likely no 10-K in date range):")
        sample_no_data = [r['ticker'] for r in no_data[:20]]
        logger.info(f"  {', '.join(sample_no_data)}")
        if len(no_data) > 20:
            logger.info(f"  ... and {len(no_data) - 20} more")

    # Show failures
    if failed:
        logger.info(f"\n❌ FAILED INGESTIONS:")
        for result in failed[:20]:
            logger.info(f"  {result['ticker']}: {result.get('error', 'Unknown error')}")
        if len(failed) > 20:
            logger.info(f"  ... and {len(failed) - 20} more errors")

    # Performance metrics
    if elapsed > 0:
        avg_time_per_ticker = elapsed / len(results)
        logger.info(f"\n⚡ PERFORMANCE METRICS:")
        logger.info(f"  Average time per ticker: {avg_time_per_ticker:.1f}s")
        logger.info(f"  Tickers per minute: {len(results) / (elapsed/60):.2f}")
        if successful:
            logger.info(f"  Average chunks per successful ticker: {total_chunks / len(successful):.0f}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ S&P 500 INGESTION COMPLETE!")
    logger.info("=" * 80)

    # Final cleanup
    try:
        db_integration.close_connection()
    except:
        pass
    cleanup_temp_files()

    # Exit code based on success rate
    success_rate = len(successful) / len(results) if results else 0
    if success_rate < 0.5:
        logger.warning(f"⚠️  Low success rate ({success_rate*100:.1f}%), please review errors")
        sys.exit(1)
    else:
        logger.info(f"🎉 Success rate: {success_rate*100:.1f}%")
        sys.exit(0)


if __name__ == "__main__":
    main()
