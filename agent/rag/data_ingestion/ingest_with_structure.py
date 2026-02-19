#!/usr/bin/env python3
"""
Structured SEC Filing Ingestion

Downloads SEC filings via datamule, preserves markdown structure (headings, tables),
stores document_markdown alongside document_text, and creates chunks annotated with
char_offset and heading_path for accurate offset-based highlighting.

Usage:
    python ingest_with_structure.py --ticker AMZN --year-start 2022 --year-end 2024
    python ingest_with_structure.py --ticker AMZN --year-start 2019 --year-end 2019
"""

import argparse
import bisect
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras
import numpy as np
import boto3
from botocore.config import Config
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from datamule import Portfolio
from sentence_transformers import SentenceTransformer

# Load env
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(dotenv_path=project_root / ".env", override=True)

# ── Railway S3 bucket ──────────────────────────────────────────────────────────
_s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("RAILWAY_BUCKET_ENDPOINT", "").strip(),
    aws_access_key_id=os.getenv("RAILWAY_BUCKET_ACCESS_KEY_ID", "").strip(),
    aws_secret_access_key=os.getenv("RAILWAY_BUCKET_SECRET_KEY", "").strip(),
    region_name="auto",
    config=Config(signature_version="s3v4"),
)
_BUCKET_NAME = os.getenv("RAILWAY_BUCKET_NAME", "").strip()


def upload_markdown_to_bucket(ticker: str, filing_type: str, fiscal_year: int, markdown: str) -> Optional[str]:
    """Upload markdown to bucket, return the key or None on failure."""
    if not _BUCKET_NAME:
        return None
    safe_type = filing_type.replace("/", "-")
    key = f"sec-filings/{ticker.upper()}/{safe_type}/{fiscal_year}.md"
    try:
        _s3_client.put_object(
            Bucket=_BUCKET_NAME,
            Key=key,
            Body=markdown.encode("utf-8"),
            ContentType="text/markdown; charset=utf-8",
        )
        return key
    except Exception as e:
        logging.getLogger(__name__).warning(f"Bucket upload failed for {key}: {e}")
        return None


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Embedding model (same as rest of pipeline)
_EMBED_MODEL = None
_MODEL_LOCK = None  # initialized lazily to avoid issues with multiprocessing

def _get_model_lock():
    global _MODEL_LOCK
    if _MODEL_LOCK is None:
        import threading
        _MODEL_LOCK = threading.Lock()
    return _MODEL_LOCK

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        with _get_model_lock():
            if _EMBED_MODEL is None:
                logger.info("Loading embedding model...")
                _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _EMBED_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# 1. MARKDOWN STRUCTURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_document_structure(markdown_text: str) -> List[Dict]:
    """
    Parse markdown headings into a hierarchical tree.

    Returns list of root-level nodes. Each node:
        {level: int, text: str, char_offset: int, children: [...]}
    """
    lines = markdown_text.split('\n')
    flat_headings: List[Dict] = []
    char_pos = 0
    in_fence = False

    for line in lines:
        stripped = line.rstrip()
        # Track fenced code blocks
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_fence = not in_fence
            char_pos += len(line) + 1
            continue

        if not in_fence:
            m = re.match(r'^(#{1,6})\s+(.+)', stripped)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                flat_headings.append({
                    'level': level,
                    'text': text,
                    'char_offset': char_pos,
                    'children': []
                })

        char_pos += len(line) + 1

    # Build tree from flat list
    root: List[Dict] = []
    stack: List[Dict] = []  # stack of (level, node)

    for node in flat_headings:
        # Pop stack until we find a parent with smaller level
        while stack and stack[-1]['level'] >= node['level']:
            stack.pop()

        if stack:
            stack[-1]['children'].append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def flatten_headings(structure: List[Dict]) -> List[Tuple[int, str, List[str]]]:
    """
    DFS flatten the heading tree to a sorted list of
    (char_offset, heading_text, full_path_list).
    """
    result: List[Tuple[int, str, List[str]]] = []

    def dfs(nodes: List[Dict], path: List[str]):
        for node in nodes:
            current_path = path + [node['text']]
            result.append((node['char_offset'], node['text'], current_path))
            dfs(node['children'], current_path)

    dfs(structure, [])
    result.sort(key=lambda x: x[0])
    return result


def extract_table_of_contents(markdown_text: str) -> List[Dict]:
    """
    Find the first markdown table that looks like a filing index (TOC).
    Returns list of {title: str, page: str}.
    """
    lines = markdown_text.split('\n')
    in_table = False
    table_rows: List[str] = []
    toc: List[Dict] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|'):
            in_table = True
            table_rows.append(stripped)
        elif in_table and table_rows:
            # Table ended — parse and return
            break

    if not table_rows:
        return []

    # Parse the first real table
    for row in table_rows:
        # Skip separator rows
        if re.match(r'^\|[\s\-|]+\|$', row):
            continue

        cells = [c.strip() for c in row.strip('|').split('|')]
        if len(cells) >= 2:
            title = cells[0]
            page = cells[-1] if cells[-1].strip().isdigit() else ''
            if title and title not in ('', 'Page', 'Item'):
                toc.append({'title': title, 'page': page})

    # Only return if it looks like an actual filing TOC (has Item 1, Item 7, etc.)
    has_items = any('Item' in e['title'] or 'PART' in e['title'].upper() for e in toc)
    return toc[:60] if has_items else []


def build_headings_map(structure: List[Dict]) -> Dict[str, int]:
    """
    Flat map: {heading_text -> char_offset} for jump-to-section.
    """
    result: Dict[str, int] = {}

    def dfs(nodes: List[Dict]):
        for node in nodes:
            result[node['text']] = node['char_offset']
            dfs(node['children'])

    dfs(structure)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. STRUCTURED CHUNK CREATION
# ─────────────────────────────────────────────────────────────────────────────

# Map heading text patterns to sec_section codes
# Use [\s\xa0]* to handle both regular spaces and non-breaking spaces
# (\xa0 is common in SEC filings: "Item\xa07." instead of "Item 7.")
_S = r'[\s\xa0]*'
SECTION_PATTERNS = [
    (r'item' + _S + r'1[aA]', 'item_1a', 'Risk Factors'),
    (r'item' + _S + r'1[bB]', 'item_1b', 'Unresolved Staff Comments'),
    (r'item' + _S + r'1\b', 'item_1', 'Business'),
    (r'item' + _S + r'2\b', 'item_2', 'Properties'),
    (r'item' + _S + r'3\b', 'item_3', 'Legal Proceedings'),
    (r'item' + _S + r'4\b', 'item_4', 'Mine Safety Disclosures'),
    (r'item' + _S + r'5\b', 'item_5', 'Market for Common Stock'),
    (r'item' + _S + r'6\b', 'item_6', 'Selected Financial Data'),
    (r'item' + _S + r'7[aA]', 'item_7a', 'Quantitative and Qualitative Disclosures'),
    (r'item' + _S + r'7\b', 'item_7', "Management's Discussion and Analysis"),
    (r'item' + _S + r'8\b', 'item_8', 'Financial Statements'),
    (r'item' + _S + r'9[aA]', 'item_9a', 'Controls and Procedures'),
    (r'item' + _S + r'9[bB]', 'item_9b', 'Other Information'),
    (r'item' + _S + r'9\b', 'item_9', 'Disagreements with Accountants'),
    (r'item' + _S + r'10\b', 'item_10', 'Directors and Executive Officers'),
    (r'item' + _S + r'11\b', 'item_11', 'Executive Compensation'),
    (r'item' + _S + r'12\b', 'item_12', 'Security Ownership'),
    (r'item' + _S + r'13\b', 'item_13', 'Certain Relationships'),
    (r'item' + _S + r'14\b', 'item_14', 'Principal Accountant Fees'),
    (r'item' + _S + r'15\b', 'item_15', 'Exhibits'),
]

_PARA_REFS = re.compile(
    r'see\s+item|as\s+described\s+in\s+item|refer\s+to\s+item|'
    r'pursuant\s+to|in\s+accordance\s+with|as\s+a\s+result\s+of|'
    r'prior\s+period|for\s+additional\s+information',
    re.IGNORECASE
)

def identify_section(heading_path: List[str]) -> Tuple[str, str]:
    """Return (sec_section, sec_section_title) from heading path.

    Skips heading elements that look like paragraph/footnote text:
    - Very long (>150 chars), OR
    - Contains cross-reference phrases like "See Item X"
    Real section headings ("Item 7. Management's Discussion...") are kept.
    """
    for heading in reversed(heading_path):
        # Skip paragraph/footnote text
        if len(heading) > 150:
            continue
        if _PARA_REFS.search(heading):
            continue
        for pattern, code, title in SECTION_PATTERNS:
            if re.search(pattern, heading, re.IGNORECASE):
                return code, title
    return 'general', 'General'


def create_structured_chunks(
    markdown_text: str,
    structure: List[Dict],
    ticker: str,
    fiscal_year: int,
    filing_type: str = '10-K',
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[Dict]:
    """
    Split markdown_text into overlapping chunks annotated with heading context.

    Each chunk dict:
        content, char_offset, parent_heading, heading_path,
        sec_section, sec_section_title, chunk_type
    """
    flat = flatten_headings(structure)
    offsets = [h[0] for h in flat]  # sorted list of heading char offsets

    chunks: List[Dict] = []
    text_len = len(markdown_text)
    start = 0
    chunk_idx = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Don't cut inside a word
        if end < text_len:
            newline = markdown_text.rfind('\n', start, end)
            if newline > start + chunk_size // 2:
                end = newline

        chunk_text = markdown_text[start:end].strip()
        if not chunk_text:
            start = end + 1
            continue

        # Find active heading at chunk start
        idx = bisect.bisect_right(offsets, start) - 1
        if idx >= 0:
            parent_heading = flat[idx][1]
            heading_path = flat[idx][2]
        else:
            parent_heading = None
            heading_path = []

        sec_section, sec_section_title = identify_section(heading_path)

        # Detect table chunks
        lines = chunk_text.split('\n')
        table_lines = sum(1 for l in lines if l.strip().startswith('|'))
        chunk_type = 'table' if table_lines > len(lines) * 0.3 else 'text'

        # Build path string (for backwards compat with path_string column)
        path_string = ' > '.join(heading_path) if heading_path else ''

        chunks.append({
            'content': chunk_text,
            'char_offset': start,
            'parent_heading': parent_heading,
            'heading_path': heading_path,
            'sec_section': sec_section,
            'sec_section_title': sec_section_title,
            'chunk_type': chunk_type,
            'path_string': path_string,
            'chunk_index': chunk_idx,
        })

        # Advance with overlap
        start = end - overlap if end < text_len else text_len
        chunk_idx += 1

    logger.info(f"  Created {len(chunks)} chunks from {text_len:,} char markdown")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATABASE STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def get_connection(db_url: Optional[str] = None):
    url = db_url or os.getenv('PG_VECTOR')
    if not url:
        raise ValueError("PG_VECTOR env var not set")
    return psycopg2.connect(url)


def store_structured_filing(
    conn,
    ticker: str,
    fiscal_year: int,
    filing_type: str,
    document_text: str,
    document_markdown: str,
    document_structure: List[Dict],
    table_of_contents: List[Dict],
    headings_map: Dict[str, int],
    filing_date: Optional[str] = None,
    accession_number: Optional[str] = None,
    filing_period: Optional[str] = None,
) -> bool:
    """
    Upsert into complete_sec_filings.
    Markdown is uploaded to the Railway S3 bucket; only the bucket_key is stored in DB.
    """
    cursor = conn.cursor()
    try:
        # Upload markdown to bucket first
        bucket_key = upload_markdown_to_bucket(ticker, filing_type, fiscal_year, document_markdown)
        if bucket_key:
            logger.info(f"  ✅ Uploaded markdown to bucket: {bucket_key}")
        else:
            logger.warning(f"  ⚠️  Bucket upload failed — markdown will NOT be stored in DB")

        # Check if row exists
        cursor.execute(
            "SELECT id FROM complete_sec_filings WHERE UPPER(ticker)=%s AND filing_type=%s AND fiscal_year=%s LIMIT 1",
            (ticker.upper(), filing_type.upper(), fiscal_year)
        )
        existing = cursor.fetchone()

        structure_json = json.dumps(document_structure)
        toc_json = json.dumps(table_of_contents)
        headings_json = json.dumps(headings_map)
        doc_len = len(document_text)

        if existing:
            cursor.execute("""
                UPDATE complete_sec_filings SET
                    document_text = %s,
                    document_markdown = NULL,
                    bucket_key = COALESCE(%s, bucket_key),
                    document_structure = %s::jsonb,
                    table_of_contents = %s::jsonb,
                    headings_map = %s::jsonb,
                    document_length = %s,
                    accession_number = COALESCE(%s, accession_number),
                    filing_date = COALESCE(%s::date, filing_date),
                    updated_at = CURRENT_TIMESTAMP
                WHERE UPPER(ticker) = %s AND filing_type = %s AND fiscal_year = %s
            """, (
                document_text,
                bucket_key,
                structure_json, toc_json, headings_json,
                doc_len,
                accession_number,
                filing_date,
                ticker.upper(), filing_type.upper(), fiscal_year
            ))
            logger.info(f"  ✅ Updated existing filing row")
        else:
            cursor.execute("""
                INSERT INTO complete_sec_filings (
                    ticker, filing_type, fiscal_year,
                    document_text, document_markdown, bucket_key, document_length,
                    document_structure, table_of_contents, headings_map,
                    filing_date, filing_period, accession_number, form_type
                ) VALUES (%s,%s,%s, %s,NULL,%s,%s, %s::jsonb,%s::jsonb,%s::jsonb, %s::date,%s,%s,%s)
                ON CONFLICT (ticker, filing_type, fiscal_year, quarter, filing_date)
                DO UPDATE SET
                    bucket_key = EXCLUDED.bucket_key,
                    document_markdown = NULL,
                    document_structure = EXCLUDED.document_structure,
                    table_of_contents = EXCLUDED.table_of_contents,
                    headings_map = EXCLUDED.headings_map,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                ticker.upper(), filing_type.upper(), fiscal_year,
                document_text, bucket_key, doc_len,
                structure_json, toc_json, headings_json,
                filing_date, filing_period, accession_number, filing_type.upper()
            ))
            logger.info(f"  ✅ Inserted new filing row")

        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"  ❌ Failed to store filing: {e}")
        raise
    finally:
        cursor.close()


def store_structured_chunks(
    conn,
    ticker: str,
    fiscal_year: int,
    filing_type: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
) -> int:
    """
    Replace all chunks for (ticker, fiscal_year, filing_type) with new structured chunks.
    """
    cursor = conn.cursor()
    try:
        # Delete existing chunks
        cursor.execute(
            "DELETE FROM ten_k_chunks WHERE ticker=%s AND fiscal_year=%s AND filing_type=%s",
            (ticker.upper(), fiscal_year, filing_type.upper())
        )
        deleted = cursor.rowcount
        logger.info(f"  Deleted {deleted} existing chunks")

        # Prepare rows
        rows = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Build metadata JSON
            metadata = {
                'ticker': ticker,
                'fiscal_year': fiscal_year,
                'filing_type': filing_type,
                'heading_path': chunk['heading_path'],
                'char_offset': chunk['char_offset'],
            }

            # Convert embedding to pgvector format
            emb_list = emb.tolist()

            rows.append((
                chunk['content'],           # chunk_text
                emb_list,                   # embedding
                json.dumps(metadata),       # metadata
                ticker.upper(),             # ticker
                fiscal_year,                # fiscal_year
                filing_type.upper(),        # filing_type
                chunk['chunk_index'],       # chunk_index
                chunk['sec_section'],       # sec_section (used as citation)
                chunk['chunk_type'],        # chunk_type
                chunk['sec_section'],       # sec_section
                chunk['sec_section_title'], # sec_section_title
                chunk['path_string'],       # path_string
                chunk['parent_heading'],    # parent_heading
                chunk['heading_path'],      # heading_path (array)
                chunk['char_offset'],       # char_offset
            ))

        if rows:
            psycopg2.extras.execute_values(
                cursor,
                """
                INSERT INTO ten_k_chunks (
                    chunk_text, embedding, metadata,
                    ticker, fiscal_year, filing_type,
                    chunk_index, citation, chunk_type,
                    sec_section, sec_section_title, path_string,
                    parent_heading, heading_path, char_offset
                )
                VALUES %s
                """,
                rows,
                template="(%s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::text[], %s)"
            )

        conn.commit()
        logger.info(f"  ✅ Inserted {len(rows)} structured chunks")
        return len(rows)

    except Exception as e:
        conn.rollback()
        logger.error(f"  ❌ Failed to store chunks: {e}")
        raise
    finally:
        cursor.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_structured_ingestion(
    ticker: str,
    year_start: int,
    year_end: int,
    filing_type: str = '10-K',
    db_url: Optional[str] = None,
    force: bool = False,
) -> Dict:
    """
    Full structured ingestion pipeline for one company.

    1. Download via datamule
    2. Parse markdown, extract structure
    3. Create chunks with char_offset
    4. Create embeddings
    5. Store everything in DB

    Returns summary dict.
    """
    ticker = ticker.upper()
    logger.info(f"\n{'='*60}")
    logger.info(f"STRUCTURED INGESTION: {ticker} {filing_type} {year_start}-{year_end}")
    logger.info(f"{'='*60}\n")

    conn = get_connection(db_url)
    model = get_embed_model()

    years_processed = []
    chunks_stored = 0
    errors = []

    try:
        # Download filings from SEC EDGAR
        logger.info(f"📥 Downloading {ticker} {filing_type} filings ({year_start}-{year_end})...")
        portfolio = Portfolio(ticker)
        portfolio.download_submissions(
            ticker=ticker,
            filing_date=(f'{year_start}-01-01', f'{year_end}-12-31'),
            submission_type=[filing_type]
        )

        documents = list(portfolio.document_type(filing_type))
        logger.info(f"✅ Found {len(documents)} filing(s)")

        for doc in documents:
            try:
                logger.info(f"\n📄 Processing filing: {doc.path}")

                # Parse the document
                doc.parse()
                logger.info(f"  ✅ Parsed document")

                # Extract fiscal year from filing date or document content
                filing_date_str = str(doc.filing_date) if hasattr(doc, 'filing_date') else None

                # Get the markdown text
                markdown_text = doc.markdown
                if not markdown_text:
                    logger.warning(f"  ⚠️ No markdown available, skipping")
                    continue

                # Get plain text
                plain_text = str(doc.text)

                # Determine fiscal year from filing date (e.g. filed 2020-01 → FY2019)
                # SEC 10-Ks are typically filed in Jan-Apr for the prior fiscal year
                if filing_date_str:
                    from datetime import datetime
                    fd = datetime.strptime(filing_date_str[:10], '%Y-%m-%d')
                    fiscal_year = fd.year if fd.month > 6 else fd.year - 1
                else:
                    fiscal_year = year_start

                # Check if we've already processed this year (unless force)
                if not force:
                    check_cur = conn.cursor()
                    check_cur.execute(
                        "SELECT document_markdown IS NOT NULL FROM complete_sec_filings WHERE UPPER(ticker)=%s AND fiscal_year=%s AND filing_type=%s LIMIT 1",
                        (ticker, fiscal_year, filing_type.upper())
                    )
                    row = check_cur.fetchone()
                    check_cur.close()
                    if row and row[0]:
                        logger.info(f"  ⏭️  FY{fiscal_year} already has markdown, skipping (use --force to overwrite)")
                        continue

                logger.info(f"  📊 FY{fiscal_year}: markdown={len(markdown_text):,} chars, text={len(plain_text):,} chars")

                # Extract document structure
                logger.info(f"  🌳 Extracting document structure...")
                structure = extract_document_structure(markdown_text)
                toc = extract_table_of_contents(markdown_text)
                headings_map = build_headings_map(structure)
                logger.info(f"  ✅ Structure: {count_nodes(structure)} headings, {len(toc)} TOC entries")

                # Create structured chunks
                logger.info(f"  ✂️  Creating structured chunks...")
                chunks = create_structured_chunks(
                    markdown_text, structure, ticker, fiscal_year, filing_type
                )

                # Create embeddings
                logger.info(f"  🔢 Creating embeddings for {len(chunks)} chunks...")
                texts = [c['content'] for c in chunks]
                embeddings = model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )

                # Store filing with structure
                logger.info(f"  💾 Storing filing with structure...")
                store_structured_filing(
                    conn=conn,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    filing_type=filing_type,
                    document_text=plain_text,
                    document_markdown=markdown_text,
                    document_structure=structure,
                    table_of_contents=toc,
                    headings_map=headings_map,
                    filing_date=filing_date_str,
                    accession_number=getattr(doc, 'accession', None),
                )

                # Store structured chunks
                logger.info(f"  💾 Storing structured chunks...")
                n = store_structured_chunks(
                    conn=conn,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    filing_type=filing_type,
                    chunks=chunks,
                    embeddings=embeddings,
                )

                years_processed.append(fiscal_year)
                chunks_stored += n
                logger.info(f"  ✅ FY{fiscal_year} complete: {n} chunks stored")

            except Exception as e:
                logger.error(f"  ❌ Error processing document: {e}", exc_info=True)
                errors.append(str(e))
                continue

    finally:
        conn.close()
        # Clean up datamule's local tar cache for this ticker to free disk space
        import shutil
        ticker_cache = Path(ticker)
        if ticker_cache.exists() and ticker_cache.is_dir():
            shutil.rmtree(ticker_cache, ignore_errors=True)
            logger.info(f"  🧹 Cleaned up local cache: {ticker_cache}/")

    result = {
        'ticker': ticker,
        'filing_type': filing_type,
        'years_processed': years_processed,
        'chunks_stored': chunks_stored,
        'errors': errors,
        'success': len(years_processed) > 0
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"INGESTION COMPLETE: {ticker}")
    logger.info(f"  Years processed: {years_processed}")
    logger.info(f"  Chunks stored:   {chunks_stored}")
    logger.info(f"  Errors:          {len(errors)}")
    logger.info(f"{'='*60}\n")

    return result


def count_nodes(structure: List[Dict]) -> int:
    """Count total nodes in heading tree."""
    count = len(structure)
    for node in structure:
        count += count_nodes(node['children'])
    return count


if __name__ == '__main__':
    import concurrent.futures

    parser = argparse.ArgumentParser(
        description='Ingest SEC filings with full markdown structure preservation'
    )
    parser.add_argument('--ticker', default=None, help='Single company ticker symbol')
    parser.add_argument('--tickers', nargs='+', help='Multiple ticker symbols for batch mode')
    parser.add_argument('--tickers-json', help='Path to JSON file with large-cap ticker list')
    parser.add_argument('--year-start', type=int, default=2019)
    parser.add_argument('--year-end', type=int, default=2025)
    parser.add_argument('--filing-type', default='10-K')
    parser.add_argument('--force', action='store_true', help='Overwrite existing data')
    parser.add_argument('--workers', type=int, default=3, help='Parallel workers for batch mode (default: 3)')
    args = parser.parse_args()

    # Build ticker list
    tickers = []
    if args.tickers_json:
        with open(args.tickers_json) as f:
            jdata = json.load(f)
        tickers = [c['ticker'] for c in jdata['companies'] if c.get('market_cap') == 'Large Cap']
        # Remove known non-primary tickers
        _drop = {
            'BOE', 'BOEI', 'DISCB', 'DISCK', 'FOX', 'NWS', 'PRSI', 'SQD', 'RSMDF',
            'TBC', 'VIACA', 'VIACP', 'ZG', 'ZMD', 'AVGOP', 'MCHPP', 'IBMA',
            'STRD', 'STRK', 'QRTEP', 'SABRP', 'LBRDB', 'XYZ', 'PSKY', 'UPSN',
        }
        tickers = [t for t in tickers if t not in _drop]
        logger.info(f"Loaded {len(tickers)} large-cap tickers from {args.tickers_json}")
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        tickers = ['AMZN']

    def _worker(ticker):
        return run_structured_ingestion(
            ticker=ticker,
            year_start=args.year_start,
            year_end=args.year_end,
            filing_type=args.filing_type,
            force=args.force,
        )

    if len(tickers) == 1:
        result = _worker(tickers[0])
        print(json.dumps(result, indent=2))
    else:
        logger.info(f"Batch mode: {len(tickers)} tickers, {args.workers} workers, {args.year_start}-{args.year_end}")
        # Pre-load model in main thread to avoid CUDA meta-tensor race condition in threads
        logger.info("Pre-loading embedding model in main thread...")
        get_embed_model()
        logger.info("Embedding model ready.")
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_ticker = {executor.submit(_worker, t): t for t in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results.append(result)
                    done = len(results)
                    logger.info(f"[{done}/{len(tickers)}] {ticker}: years={result.get('years_processed')}, chunks={result.get('chunks_stored')}")
                except Exception as e:
                    logger.error(f"[FAILED] {ticker}: {e}")
                    results.append({'ticker': ticker, 'error': str(e), 'success': False})

        successes = sum(1 for r in results if r.get('success'))
        logger.info(f"\nBatch complete: {successes}/{len(tickers)} succeeded")
        with open('ingest_structured_batch_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to ingest_structured_batch_results.json")
