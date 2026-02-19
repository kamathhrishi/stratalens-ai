"""
SEC Filings Router

Serves 10-K, 10-Q, and 8-K filing content from the bucket + database.
- Metadata (ticker, fiscal_year, etc.) lives in PostgreSQL
- Document markdown lives in the Railway S3 bucket
"""

import os
import re
import asyncio
import logging
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncpg
import boto3
from botocore.config import Config

router = APIRouter(prefix="/sec-filings", tags=["sec-filings"])

_pg_vector_url = os.getenv("PG_VECTOR", "").strip() or os.getenv("DATABASE_URL", "").strip()
_pool: asyncpg.Pool | None = None

# S3 client for Railway bucket
_s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("RAILWAY_BUCKET_ENDPOINT", "").strip(),
    aws_access_key_id=os.getenv("RAILWAY_BUCKET_ACCESS_KEY_ID", "").strip(),
    aws_secret_access_key=os.getenv("RAILWAY_BUCKET_SECRET_KEY", "").strip(),
    region_name="auto",
    config=Config(signature_version="s3v4"),
)
_BUCKET_NAME = os.getenv("RAILWAY_BUCKET_NAME", "").strip()

# In-memory cache for recently fetched markdown
_markdown_cache: dict[str, str] = {}
_MAX_CACHE = 30


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not _pg_vector_url:
            raise HTTPException(status_code=503, detail="PG_VECTOR database not configured")
        _pool = await asyncpg.create_pool(_pg_vector_url, min_size=2, max_size=5)
    return _pool


async def _fetch_markdown_from_bucket(bucket_key: str) -> str:
    """Fetch markdown content from the Railway bucket (with in-memory cache)."""
    if bucket_key in _markdown_cache:
        return _markdown_cache[bucket_key]

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _s3.get_object(Bucket=_BUCKET_NAME, Key=bucket_key)
    )
    markdown = response["Body"].read().decode("utf-8")

    # Evict oldest if full
    if len(_markdown_cache) >= _MAX_CACHE:
        _markdown_cache.pop(next(iter(_markdown_cache)))
    _markdown_cache[bucket_key] = markdown
    return markdown


class HighlightRequest(BaseModel):
    ticker: str
    filing_type: str
    fiscal_year: int
    quarter: Optional[int] = None
    filing_date: Optional[str] = None
    relevant_chunks: Optional[List[dict]] = None


async def _fetch_filing_meta(ticker: str, filing_type: str, fiscal_year: int) -> dict:
    """Fetch filing metadata row from DB."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ticker, filing_type, fiscal_year, filing_date,
                   document_length, company_name, bucket_key
            FROM complete_sec_filings
            WHERE UPPER(ticker) = UPPER($1)
              AND filing_type = $2
              AND fiscal_year <= $3
            ORDER BY fiscal_year DESC, filing_date DESC NULLS LAST
            LIMIT 1
            """,
            ticker, filing_type, fiscal_year
        )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No {filing_type} filing found for {ticker.upper()} FY{fiscal_year}"
        )
    return dict(row)


def _inject_highlights(markdown: str, chunks: List[dict]) -> str:
    if not markdown or not chunks:
        return markdown
    result = markdown
    for chunk in chunks:
        text = (chunk.get("chunk_text") or "").strip()
        if not text or len(text) < 20:
            continue

        # Strategy 1: char_offset — direct position hint (most reliable for tables)
        char_offset = chunk.get("char_offset")
        if char_offset is not None and isinstance(char_offset, int):
            end = char_offset + len(text)
            if 0 <= char_offset < len(result) and end <= len(result) + 200:
                window_start = max(0, char_offset - 50)
                window_end = min(len(result), char_offset + 50)
                if text[:30].strip() in result[window_start:window_end]:
                    actual_start = result.find(text[:30].strip(), window_start)
                    if actual_start != -1:
                        actual_end = actual_start + len(text)
                        result = result[:actual_start] + "<mark>" + result[actual_start:actual_end] + "</mark>" + result[actual_end:]
                        continue

        # Strategy 2: Exact text match
        needle = text[:120]
        idx = result.find(needle)
        if idx != -1:
            result = result[:idx] + "<mark>" + text + "</mark>" + result[idx + len(text):]
            continue

        # Strategy 3: Flexible whitespace match on first 80 chars
        short = text[:80].strip()
        tokens = re.split(r'\s+', short)
        if len(tokens) >= 3:
            pattern = r'\s+'.join(re.escape(t) for t in tokens if t)
            m = re.search(pattern, result)
            if m:
                start = m.start()
                if len(text) > 40:
                    tail_tokens = re.split(r'\s+', text[-40:].strip())
                    tail_pattern = r'\s+'.join(re.escape(t) for t in tail_tokens if t)
                    end_m = re.search(tail_pattern, result[start:start + len(text) + 300])
                    end = start + end_m.end() if end_m else start + len(text) + 50
                else:
                    end = start + len(text) + 50
                result = result[:start] + "<mark>" + result[start:end] + "</mark>" + result[end:]
                continue

        # Strategy 4: Per-line matching — strip outer table pipes, search each data line
        # Handles the case where chunk omits leading/trailing | but S3 includes them.
        line_found = False
        for line in text.split('\n'):
            clean = line.strip().strip('|').strip()
            if len(clean) < 15:
                continue
            if re.match(r'^[-|:\s]+$', clean):  # separator row
                continue
            # Exact match of stripped line (works when S3 has `| CONTENT |` and chunk has `CONTENT`)
            idx = result.find(clean)
            if idx != -1:
                chunk_start = max(0, idx - 300)
                chunk_end = min(len(result), idx + len(text) + 300)
                result = result[:chunk_start] + "<mark>" + result[chunk_start:chunk_end] + "</mark>" + result[chunk_end:]
                line_found = True
                break
            # Flexible spacing match between cells (handles ` | ` vs `|` variations)
            parts = [p.strip() for p in clean.split('|') if p.strip() and len(p.strip()) >= 3]
            if len(parts) >= 2:
                pat = r'\s*\|\s*'.join(re.escape(p) for p in parts)
                m = re.search(pat, result)
                if m:
                    chunk_start = max(0, m.start() - 300)
                    chunk_end = min(len(result), m.start() + len(text) + 300)
                    result = result[:chunk_start] + "<mark>" + result[chunk_start:chunk_end] + "</mark>" + result[chunk_end:]
                    line_found = True
                    break
        if line_found:
            continue
    return result


@router.get("/{ticker}/available")
async def get_available_filings(ticker: str):
    """List all available filings for a ticker."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ticker, filing_type, fiscal_year, filing_date, document_length
            FROM complete_sec_filings
            WHERE UPPER(ticker) = UPPER($1)
            ORDER BY fiscal_year DESC, filing_date DESC
            """,
            ticker
        )
    return {
        "ticker": ticker.upper(),
        "filings": [dict(r) for r in rows]
    }


@router.get("/{ticker}/{filing_type}/{fiscal_year}")
async def get_filing(
    ticker: str,
    filing_type: str,
    fiscal_year: int,
    quarter: Optional[int] = Query(None),
    filing_date: Optional[str] = Query(None),
):
    """Fetch a filing — metadata from DB, content from bucket."""
    meta = await _fetch_filing_meta(ticker, filing_type, fiscal_year)

    bucket_key = meta.get("bucket_key")
    if not bucket_key:
        raise HTTPException(status_code=404, detail="Filing content not yet migrated to bucket")

    markdown = await _fetch_markdown_from_bucket(bucket_key)

    return ORJSONResponse({
        "success": True,
        "ticker": meta["ticker"],
        "company_name": meta.get("company_name"),
        "filing_type": meta["filing_type"],
        "fiscal_year": meta["fiscal_year"],
        "filing_date": str(meta["filing_date"]) if meta.get("filing_date") else None,
        "document_text": "",
        "document_markdown": markdown,
        "document_length": meta.get("document_length"),
    })


@router.post("/with-highlights")
async def get_filing_with_highlights(req: HighlightRequest):
    """Fetch a filing and inject highlight marks around relevant chunks."""
    meta = await _fetch_filing_meta(req.ticker, req.filing_type, req.fiscal_year)

    bucket_key = meta.get("bucket_key")
    if not bucket_key:
        raise HTTPException(status_code=404, detail="Filing content not yet migrated to bucket")

    markdown = await _fetch_markdown_from_bucket(bucket_key)
    highlighted = _inject_highlights(markdown, req.relevant_chunks or [])

    return ORJSONResponse({
        "success": True,
        "ticker": meta["ticker"],
        "company_name": meta.get("company_name"),
        "filing_type": meta["filing_type"],
        "fiscal_year": meta["fiscal_year"],
        "filing_date": str(meta["filing_date"]) if meta.get("filing_date") else None,
        "document_text": "",
        "document_markdown": markdown,
        "highlighted_markdown": highlighted,
        "document_length": meta.get("document_length"),
    })
