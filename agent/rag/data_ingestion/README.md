# Data Ingestion Scripts

Scripts for downloading and processing financial data for the RAG system.
All data is stored in PostgreSQL (pgvector) and the Railway S3 bucket.

## 📁 Script Reference

### Ongoing / Reusable

| Script | Purpose |
|--------|---------|
| `download_transcripts.py` | Download earnings transcripts (API Ninjas, 2020–2025). Saves as JSON. |
| `create_and_store_embeddings.py` | Chunk transcripts, generate embeddings, store in PostgreSQL. |
| `ingest_10k_filings_full.py` | **Core module** (~2,350 lines). Download, parse, chunk, embed SEC filings. Imported by other scripts. |
| `ingest_10k_to_database.py` | Main 10-K ingestion script. Stores to PostgreSQL. Use `--ticker`, `--tickers`, or `--all`. |
| `ingest_sec_filings.py` | Unified ingestion for any filing type (10-K, 10-Q, 8-K, S-11). |
| `ingest_with_structure.py` | Structured ingestion that preserves markdown headings and `char_offset` for precise highlighting. |
| `ingest_sp500_10k.py` | Bulk ingestion for all S&P 500 companies. Parallel processing. |
| `fetch_us_tickers.py` | Fetch all US equity tickers via `financedatabase`. Saves to file. |
| `drop_tables.py` | Clear all 10-K data from the database. Requires confirmation prompt. |
| `test_db_connection.py` | Verify DB connection and table existence. |

### One-Time Migration Scripts

These were run once to migrate data to the current infrastructure (Railway S3 bucket).
Keep for reference but do not re-run unless resetting.

| Script | Purpose |
|--------|---------|
| `migrate_filings_to_bucket.py` | Uploaded SEC filing markdown from PostgreSQL to Railway S3. **Already complete.** |
| `migrate_transcripts_to_bucket.py` | Uploaded transcript text from PostgreSQL to Railway S3. **Already complete.** |
| `backfill_complete_filings.py` | Migrated SEC docs from embeddings_cache to `complete_sec_filings` DB table. |
| `backfill_10k_2019_2025.py` | Backfilled 10-K filings for 118 companies (FY 2019–2025). |
| `download_sample_filings.py` | Downloaded sample 10-Q/8-K/S-11 filings for structure exploration. |

## 🚀 Quick Start (New Setup)

```bash
# 1. Copy env and add keys
cp .env.example .env

# 2. Run DB migrations (creates tables and columns)
psql $DATABASE_URL -f db/create_sec_filings_tables.sql
psql $DATABASE_URL -f db/migrate_add_structure_columns.sql

# 3. Download transcripts
python agent/rag/data_ingestion/download_transcripts.py

# 4. Create transcript embeddings
python agent/rag/data_ingestion/create_and_store_embeddings.py

# 5. Ingest 10-K filings (single ticker example)
python agent/rag/data_ingestion/ingest_10k_to_database.py --ticker AAPL
```

## ⚙️ Environment Variables

```
DATABASE_URL          # PostgreSQL + pgvector connection string
PG_VECTOR             # Alias for DATABASE_URL (used by some scripts)
API_NINJAS_KEY        # For transcript downloads
RAILWAY_BUCKET_ENDPOINT      # S3-compatible bucket endpoint
RAILWAY_BUCKET_ACCESS_KEY_ID
RAILWAY_BUCKET_SECRET_KEY
RAILWAY_BUCKET_NAME
```

## 💾 Storage

Data is split between PostgreSQL and Railway S3:

- **PostgreSQL**: chunk embeddings, metadata, section structure
- **S3 bucket**: full filing markdown, transcript text (fetched on demand)

Expected DB sizes:
- Transcript chunks: ~500MB per 1,000 companies
- 10-K chunks + embeddings: ~5–10GB per 500 companies

## 🔧 Troubleshooting

```bash
# Test database connection
python agent/rag/data_ingestion/test_db_connection.py

# Check if a filing is available
psql $DATABASE_URL -c "SELECT ticker, filing_type, fiscal_year, bucket_key IS NOT NULL as in_bucket FROM complete_sec_filings WHERE ticker='AAPL' ORDER BY fiscal_year DESC;"
```
