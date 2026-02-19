#!/usr/bin/env python3
"""
Unified SEC Filings Ingestion Script

Download and ingest any SEC filing type (10-K, 10-Q, 8-K, S-11) into PostgreSQL.
Uses the same sophisticated processing pipeline as ingest_10k_to_database.py.

Usage:
    # Ingest 10-Q for specific tickers
    python ingest_sec_filings.py --tickers AAPL MSFT --types 10-Q --lookback-years 2

    # Ingest 8-K for a ticker (limit to 10 filings)
    python ingest_sec_filings.py --tickers AAPL --types 8-K --lookback-years 1 --max-filings 10

    # Multiple filing types at once
    python ingest_sec_filings.py --tickers AAPL MSFT NVDA --types 10-Q 8-K --workers 3

    # S-11 (raw text mode)
    python ingest_sec_filings.py --tickers AAPL --types S-11 --lookback-years 5

    # All filing types for a ticker
    python ingest_sec_filings.py --tickers AAPL --types 10-K 10-Q 8-K --lookback-years 1
"""

import argparse
import os
import sys
import logging
import time
import gc
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ingest_10k_to_database import (
    DatabaseIntegration,
    ingest_filing,
    ingest_filing_worker,
    cleanup_temp_files,
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
        logging.FileHandler('sec_filings_ingestion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

SUPPORTED_FILING_TYPES = ['10-K', '10-Q', '8-K', 'S-11']


def main():
    # Fix CUDA multiprocessing issue
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description='Ingest SEC filings (10-K, 10-Q, 8-K, S-11) into PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_sec_filings.py --tickers AAPL MSFT --types 10-Q --lookback-years 2
  python ingest_sec_filings.py --tickers AAPL --types 8-K --lookback-years 1 --max-filings 10
  python ingest_sec_filings.py --tickers AAPL MSFT NVDA --types 10-Q 8-K --workers 3
        """
    )
    parser.add_argument('--tickers', type=str, nargs='+', required=True,
                        help='Ticker symbols to process')
    parser.add_argument('--types', type=str, nargs='+', required=True,
                        choices=SUPPORTED_FILING_TYPES,
                        help=f'Filing types to ingest ({", ".join(SUPPORTED_FILING_TYPES)})')
    parser.add_argument('--lookback-years', type=int, default=1,
                        help='Years to look back (default: 1)')
    parser.add_argument('--max-filings', type=int, default=None,
                        help='Max filings per ticker per type (useful for 8-K)')
    parser.add_argument('--workers', type=int, default=3,
                        help='Number of parallel workers (default: 3)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run sequentially (no parallel processing)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout per ticker in minutes (default: 30)')
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    filing_types = [t.upper() for t in args.types]

    # Build work items: (ticker, filing_type) pairs
    work_items = []
    for ticker in tickers:
        for filing_type in filing_types:
            work_items.append((ticker, filing_type))

    logger.info(f"{'='*70}")
    logger.info(f"SEC Filing Ingestion Starting")
    logger.info(f"{'='*70}")
    logger.info(f"  Tickers: {len(tickers)} ({', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''})")
    logger.info(f"  Filing types: {', '.join(filing_types)}")
    logger.info(f"  Work items: {len(work_items)} (ticker x type combinations)")
    logger.info(f"  Lookback: {args.lookback_years} year(s)")
    logger.info(f"  Max filings per ticker: {args.max_filings or 'unlimited'}")
    logger.info(f"  Mode: {'Sequential' if args.sequential else f'{args.workers} parallel workers'}")

    # Database setup
    db_url = os.getenv("PG_VECTOR", "") or os.getenv("DATABASE_URL", "")
    if not db_url:
        logger.error("PG_VECTOR or DATABASE_URL environment variable not set")
        sys.exit(1)

    logger.info(f"  Database: {db_url[:50]}...")

    # Ensure tables exist
    db_integration = DatabaseIntegration(db_url=db_url)
    db_integration.ensure_tables()

    results = []
    start_time = time.time()

    if args.sequential:
        logger.info("\nRunning in sequential mode")
        for idx, (ticker, filing_type) in enumerate(work_items, 1):
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"Progress: {idx}/{len(work_items)} ({idx/len(work_items)*100:.1f}%)")
                logger.info(f"{'='*70}")

                result = ingest_filing(
                    ticker, filing_type, args.lookback_years,
                    db_integration, max_filings=args.max_filings
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to ingest {ticker} {filing_type}: {e}")
                results.append({'ticker': ticker, 'filing_type': filing_type, 'error': str(e)})
                continue
    else:
        logger.info(f"\nRunning with {args.workers} parallel workers")

        python_version = sys.version_info
        executor_kwargs = {'max_workers': args.workers}
        if python_version >= (3, 11):
            executor_kwargs['max_tasks_per_child'] = 1

        completed = 0
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            future_to_item = {}
            for ticker, filing_type in work_items:
                future = executor.submit(
                    ingest_filing_worker,
                    ticker, filing_type, args.lookback_years,
                    db_url, args.timeout,
                    args.max_filings
                )
                future_to_item[future] = (ticker, filing_type)

            for future in as_completed(future_to_item):
                ticker, filing_type = future_to_item[future]
                completed += 1

                try:
                    result = future.result(timeout=args.timeout * 60 + 60)
                    results.append(result)

                    if result.get('error'):
                        logger.error(f"[{completed}/{len(work_items)}] {ticker} {filing_type}: ERROR - {result['error']}")
                    else:
                        chunks = result.get('chunks_stored', 0)
                        processed = result.get('filings_processed', 0)
                        skipped = result.get('filings_skipped', 0)
                        logger.info(f"[{completed}/{len(work_items)}] {ticker} {filing_type}: "
                                   f"{processed} processed, {skipped} skipped, {chunks} chunks")

                except Exception as e:
                    logger.error(f"[{completed}/{len(work_items)}] {ticker} {filing_type}: EXCEPTION - {e}")
                    results.append({'ticker': ticker, 'filing_type': filing_type, 'error': str(e)})

    # Final summary
    elapsed = time.time() - start_time

    logger.info(f"\n{'='*70}")
    logger.info(f"INGESTION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total time: {elapsed/60:.1f} minutes")

    total_chunks = 0
    total_tables = 0
    total_processed = 0
    total_skipped = 0
    errors = []

    for result in results:
        if result.get('error'):
            errors.append(f"  {result.get('ticker', '?')} {result.get('filing_type', '?')}: {result['error']}")
        else:
            total_chunks += result.get('chunks_stored', 0)
            total_tables += result.get('tables_stored', 0)
            total_processed += result.get('filings_processed', 0)
            total_skipped += result.get('filings_skipped', 0)

    logger.info(f"  Filings processed: {total_processed}")
    logger.info(f"  Filings skipped: {total_skipped}")
    logger.info(f"  Total chunks stored: {total_chunks}")
    logger.info(f"  Total tables stored: {total_tables}")
    logger.info(f"  Errors: {len(errors)}")

    if errors:
        logger.info(f"\nErrors:")
        for err in errors:
            logger.info(err)

    # Cleanup
    cleanup_temp_files()
    gc.collect()


if __name__ == '__main__':
    main()
