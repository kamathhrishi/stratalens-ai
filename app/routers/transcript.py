"""
Transcript Router - Handles earnings transcript endpoints

Transcript text lives in the Railway S3 bucket.
Metadata (ticker, year, quarter, etc.) lives in PostgreSQL (PG_VECTOR).
"""

import asyncio
import logging
import os
import re

import asyncpg
import boto3
from botocore.config import Config
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import ORJSONResponse
from app.auth.auth_utils import get_current_user, get_optional_user

router = APIRouter()
logger = logging.getLogger(__name__)

# ── S3 bucket client ──────────────────────────────────────────────────────────
_s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("RAILWAY_BUCKET_ENDPOINT", "").strip(),
    aws_access_key_id=os.getenv("RAILWAY_BUCKET_ACCESS_KEY_ID", "").strip(),
    aws_secret_access_key=os.getenv("RAILWAY_BUCKET_SECRET_KEY", "").strip(),
    region_name="auto",
    config=Config(signature_version="s3v4"),
)
_BUCKET_NAME = os.getenv("RAILWAY_BUCKET_NAME", "").strip()

# ── DB pool (PG_VECTOR) ───────────────────────────────────────────────────────
_pg_url = os.getenv("PG_VECTOR", "").strip()
_pool: asyncpg.Pool | None = None

async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(_pg_url, min_size=2, max_size=5)
    return _pool

# ── In-memory cache for transcript text ──────────────────────────────────────
_transcript_cache: dict[str, str] = {}
_MAX_CACHE = 50


async def _fetch_transcript_from_bucket(bucket_key: str) -> str:
    if bucket_key in _transcript_cache:
        return _transcript_cache[bucket_key]
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _s3.get_object(Bucket=_BUCKET_NAME, Key=bucket_key)
    )
    text = response["Body"].read().decode("utf-8")
    if len(_transcript_cache) >= _MAX_CACHE:
        _transcript_cache.pop(next(iter(_transcript_cache)))
    _transcript_cache[bucket_key] = text
    return text


def _inject_highlights(text: str, chunks: list[dict]) -> str:
    """Wrap relevant chunk text in <span class="highlighted-chunk"> tags."""
    if not chunks:
        return text
    sorted_chunks = sorted(chunks, key=lambda x: len(x.get("chunk_text", "")), reverse=True)
    for i, chunk in enumerate(sorted_chunks):
        chunk_text = re.sub(r'\s+', ' ', (chunk.get("chunk_text") or "").strip())
        if len(chunk_text) < 10:
            continue
        replacement = (
            f'<span class="highlighted-chunk chunk-{i}" '
            f'data-chunk-id="{chunk.get("chunk_id", "")}">'
            f'{chunk_text}</span>'
        )
        pattern = re.escape(chunk_text)
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE, count=1)
    return text


@router.get("/transcript/{ticker}/{year}/{quarter}")
async def get_complete_transcript(
    ticker: str,
    year: int,
    quarter: int,
    current_user: dict = Depends(get_optional_user)
):
    """Fetch transcript metadata from DB, content from bucket."""
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ticker, company_name, year, quarter, date, bucket_key, metadata, created_at
            FROM complete_transcripts
            WHERE UPPER(ticker) = UPPER($1) AND year = $2 AND quarter = $3
            LIMIT 1
            """,
            ticker, year, quarter
        )

    if not row or not row["bucket_key"]:
        raise HTTPException(
            status_code=404,
            detail=f"Full earnings transcript not yet available for {ticker.upper()} {year} Q{quarter}"
        )
    bucket_key = row["bucket_key"]

    try:
        text = await _fetch_transcript_from_bucket(bucket_key)
    except Exception as e:
        logger.error(f"Failed to fetch transcript from bucket: {e}")
        raise HTTPException(status_code=503, detail="Could not load transcript from storage")

    return ORJSONResponse({
        "success": True,
        "ticker": row["ticker"],
        "company_name": row["company_name"] or "Unknown",
        "year": row["year"],
        "quarter": row["quarter"],
        "date": row["date"] or "Unknown",
        "transcript_text": text,
        "transcript": text,
        "transcript_length": len(text),
        "source": "bucket",
        "metadata": row["metadata"] if isinstance(row["metadata"], dict) else {},
    })


@router.post("/transcript/with-highlights")
async def get_transcript_with_highlights(
    request: dict,
    current_user: dict = Depends(get_optional_user)
):
    """Fetch transcript from bucket and inject highlight spans around relevant chunks."""
    ticker = (request.get("ticker") or "").upper()
    year = request.get("year")
    quarter = request.get("quarter")
    relevant_chunks = request.get("relevant_chunks", [])

    if not all([ticker, year, quarter]):
        raise HTTPException(status_code=400, detail="Missing required parameters: ticker, year, quarter")

    pool = await _get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ticker, company_name, year, quarter, date, bucket_key, metadata
            FROM complete_transcripts
            WHERE UPPER(ticker) = UPPER($1) AND year = $2 AND quarter = $3
            LIMIT 1
            """,
            ticker, year, quarter
        )

    if not row or not row["bucket_key"]:
        raise HTTPException(status_code=404, detail=f"Transcript not found for {ticker} {year} Q{quarter}")

    try:
        text = await _fetch_transcript_from_bucket(row["bucket_key"])
    except Exception as e:
        logger.error(f"Failed to fetch transcript from bucket: {e}")
        raise HTTPException(status_code=503, detail="Could not load transcript from storage")

    highlighted = _inject_highlights(text, relevant_chunks)

    return ORJSONResponse({
        "success": True,
        "transcript_text": text,
        "transcript": text,
        "highlighted_transcript": highlighted,
        "transcript_length": len(text),
        "metadata": {
            "ticker": row["ticker"],
            "year": row["year"],
            "quarter": row["quarter"],
            "date": row["date"] or "N/A",
        },
    })


