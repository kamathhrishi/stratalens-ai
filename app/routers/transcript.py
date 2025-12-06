"""
Transcript Router - Handles earnings transcript endpoints
"""

import json
import logging
import os
import re
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from app.auth.auth_utils import get_current_user, get_optional_user

# RAG system imports (optional)
try:
    from agent.rag.transcript_service import TranscriptService
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

router = APIRouter()
logger = logging.getLogger(__name__)

# Global transcript service instance
_transcript_service = None

def get_transcript_service() -> TranscriptService:
    """Get or create the transcript service instance"""
    global _transcript_service
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Transcript features disabled - RAG system not available")
    
    if _transcript_service is None:
        # Create config and database manager
        from agent.rag.config import Config
        config = Config()
        from agent.rag.database_manager import DatabaseManager
        database_manager = DatabaseManager(config)
        _transcript_service = TranscriptService(database_manager)
    
    return _transcript_service


@router.get("/transcript/{ticker}/{year}/{quarter}")
async def get_complete_transcript(
    ticker: str,
    year: int,
    quarter: int,
    current_user: dict = Depends(get_optional_user)
):
    """Get complete earnings transcript for a specific ticker, year, and quarter from database
    
    Works for both authenticated users and demo users (no authentication required)
    """
    try:
        # Get transcript service instance
        transcript_service = get_transcript_service()
        
        # First check if chunks exist for this company/quarter
        logger.info(f"🔍 Checking chunks availability for {ticker} {year} Q{quarter}")
        chunks_available = await transcript_service.check_chunks_availability_async(ticker, year, quarter)
        logger.info(f"🔍 Chunks available result: {chunks_available}")
        
        # Retrieve transcript from database
        transcript_data = await transcript_service.get_complete_transcript_async(ticker, year, quarter)
        
        if not transcript_data:
            if chunks_available:
                # Chunks exist but complete transcript doesn't
                logger.info(f"🔍 Chunks exist but transcript missing for {ticker} {year} Q{quarter}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"Complete transcript not available for {ticker.upper()} {year} Q{quarter}. Chunks are available for search but the full transcript hasn't been processed yet."
                )
            else:
                # Neither chunks nor transcript exist
                logger.info(f"🔍 Neither chunks nor transcript exist for {ticker} {year} Q{quarter}")
                raise HTTPException(
                    status_code=404, 
                    detail=f"No transcript data found for {ticker.upper()} {year} Q{quarter}. This transcript may not be available in our database."
                )
        
        # Determine if this is a demo request
        is_demo = current_user is None
        user_type = "demo" if is_demo else "authenticated"
        username = current_user.get('username', 'anonymous') if current_user else 'demo'
        
        # Log the request
        logger.info(f"{user_type.capitalize()} user {username} requested transcript for {ticker.upper()} {year} Q{quarter}")
        
        response_data = {
            "success": True,
            "ticker": transcript_data['ticker'],
            "company_name": transcript_data.get('company_name', 'Unknown'),
            "year": transcript_data['year'],
            "quarter": transcript_data['quarter'],
            "date": transcript_data.get('date', 'Unknown'),
            "transcript_text": transcript_data['full_transcript'],
            "transcript": transcript_data['full_transcript'],  # Keep both for compatibility
            "transcript_length": len(transcript_data['full_transcript']),
            "source": "Database: complete_transcripts table",
            "metadata": transcript_data.get('metadata', {}),
            "created_at": transcript_data.get('created_at')
        }
        
        # Add demo_mode flag for demo users
        if is_demo:
            response_data["demo_mode"] = True
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving transcript from database: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/transcript/with-highlights")
async def get_transcript_with_highlights(
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    """Get transcript with highlighted chunks based on citation data"""
    try:
        ticker = request.get("ticker", "").upper()
        year = request.get("year")
        quarter = request.get("quarter")
        relevant_chunks = request.get("relevant_chunks", [])
        
        logger.info(f"🎨 DEBUG: get_transcript_with_highlights called with ticker={ticker}, year={year}, quarter={quarter}")
        logger.info(f"🎨 DEBUG: Received {len(relevant_chunks)} relevant chunks")
        logger.info(f"🎨 DEBUG: First few chunks: {[chunk.get('chunk_text', '')[:50] + '...' for chunk in relevant_chunks[:3]]}")
        
        if not all([ticker, year, quarter]):
            raise HTTPException(status_code=400, detail="Missing required parameters: ticker, year, quarter")
        
        # Construct the filename
        filename = f"{ticker}_transcript_{year}_Q{quarter}.json"
        
        # Try different quarter folders
        possible_folders = [
            f"rag/earnings_transcripts_{year}_q{quarter}",
            f"rag/earnings_transcripts_{year}_q{quarter-1}",  # Fallback to previous quarter
        ]
        
        transcript_data = None
        used_folder = None
        
        for folder in possible_folders:
            file_path = os.path.join(folder, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    used_folder = folder
                    break
        
        if not transcript_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Transcript not found for {ticker} {year} Q{quarter}"
            )
        
        # Extract transcript text
        transcript_text = ""
        if 'data' in transcript_data and 'transcript' in transcript_data['data']:
            transcript_text = transcript_data['data']['transcript']
        elif 'transcript' in transcript_data:
            transcript_text = transcript_data['transcript']
        
        if not transcript_text:
            raise HTTPException(status_code=404, detail="No transcript text found")
        
        # Create highlighted version
        logger.info(f"🎨 Creating highlighted transcript with {len(relevant_chunks)} chunks")
        logger.debug(f"🎨 Relevant chunks: {relevant_chunks}")
        logger.info(f"🎨 DEBUG: About to call highlight_chunks_in_transcript")
        highlighted_transcript = highlight_chunks_in_transcript(transcript_text, relevant_chunks)
        logger.info(f"🎨 Highlighted transcript created, length: {len(highlighted_transcript)}")
        logger.info(f"🎨 DEBUG: Highlighted transcript contains HTML: {'<span' in highlighted_transcript}")
        logger.info(f"🎨 DEBUG: Highlighted transcript contains highlighted-chunk: {'highlighted-chunk' in highlighted_transcript}")
        
        return {
            "success": True,
            "transcript_text": transcript_text,
            "highlighted_transcript": highlighted_transcript,
            "metadata": {
                "ticker": ticker,
                "year": year,
                "quarter": quarter,
                "date": transcript_data.get("data", {}).get("date", "N/A"),
                "cik": transcript_data.get("data", {}).get("cik", "N/A"),
                "source_folder": used_folder
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        from app.utils import raise_sanitized_http_exception
        raise_sanitized_http_exception(
            e, 
            "transcript with highlights fetch", 
            current_user.get("id"),
            "We're having trouble retrieving the transcript. Please try again later.",
            status_code=500
        )


def highlight_chunks_in_transcript(transcript_text: str, relevant_chunks: list) -> str:
    """Highlight relevant chunks in the transcript text"""
    logger.info(f"🎨 DEBUG: highlight_chunks_in_transcript called with {len(relevant_chunks) if relevant_chunks else 0} chunks")
    logger.info(f"🎨 DEBUG: Transcript text length: {len(transcript_text) if transcript_text else 0}")
    
    if not relevant_chunks:
        logger.info("🎨 No relevant chunks provided, returning original transcript")
        return transcript_text
    
    logger.info(f"🎨 Starting highlighting with {len(relevant_chunks)} chunks")
    logger.info(f"🎨 DEBUG: First few chunks: {[chunk.get('chunk_text', '')[:50] + '...' for chunk in relevant_chunks[:3]]}")
    
    # Sort chunks by length (longer first) to avoid conflicts with shorter matches
    sorted_chunks = sorted(relevant_chunks, key=lambda x: len(x.get("chunk_text", "")), reverse=True)
    
    highlighted_text = transcript_text
    highlighted_count = 0
    
    for i, chunk in enumerate(sorted_chunks):
        chunk_text = chunk.get("chunk_text", "").strip()
        if not chunk_text or len(chunk_text) < 10:  # Skip very short chunks
            continue
            
        # Clean up the chunk text - remove extra whitespace and normalize
        chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
        
        # Try multiple matching strategies
        highlighted = False
        
        # Strategy 1: Exact match with word boundaries
        escaped_chunk = re.escape(chunk_text)
        pattern = r'\b' + escaped_chunk + r'\b'
        
        if re.search(pattern, highlighted_text, re.IGNORECASE):
            replacement = f'<span class="highlighted-chunk chunk-{i}" data-chunk-id="{chunk.get("chunk_id", "")}" title="Relevant chunk from search results">{chunk_text}</span>'
            highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
            highlighted = True
            highlighted_count += 1
            logger.info(f"🎨 Successfully highlighted chunk {i+1} with exact match")
        
        # Strategy 2: If exact match fails, try without word boundaries
        if not highlighted:
            pattern = re.escape(chunk_text)
            if re.search(pattern, highlighted_text, re.IGNORECASE):
                replacement = f'<span class="highlighted-chunk chunk-{i}" data-chunk-id="{chunk.get("chunk_id", "")}" title="Relevant chunk from search results">{chunk_text}</span>'
                highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
                highlighted = True
                highlighted_count += 1
                logger.info(f"🎨 Successfully highlighted chunk {i+1} with non-word-boundary match")
        
        # Strategy 3: If still no match, try with first 5 words
        if not highlighted and len(chunk_text.split()) >= 5:
            words = chunk_text.split()[:5]
            partial_chunk = " ".join(words)
            pattern = re.escape(partial_chunk)
            if re.search(pattern, highlighted_text, re.IGNORECASE):
                replacement = f'<span class="highlighted-chunk chunk-{i}" data-chunk-id="{chunk.get("chunk_id", "")}" title="Relevant chunk from search results">{partial_chunk}</span>'
                highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
                highlighted = True
                highlighted_count += 1
                logger.info(f"🎨 Successfully highlighted chunk {i+1} with 5-word partial match")
        
        # Strategy 4: Try with first 3 words as last resort
        if not highlighted and len(chunk_text.split()) >= 3:
            words = chunk_text.split()[:3]
            partial_chunk = " ".join(words)
            pattern = re.escape(partial_chunk)
            if re.search(pattern, highlighted_text, re.IGNORECASE):
                replacement = f'<span class="highlighted-chunk chunk-{i}" data-chunk-id="{chunk.get("chunk_id", "")}" title="Relevant chunk from search results">{partial_chunk}</span>'
                highlighted_text = re.sub(pattern, replacement, highlighted_text, flags=re.IGNORECASE)
                highlighted_count += 1
                logger.info(f"🎨 Successfully highlighted chunk {i+1} with 3-word partial match")
        
        if not highlighted:
            logger.warning(f"🎨 Could not highlight chunk {i+1}: '{chunk_text[:50]}...'")
    
    logger.info(f"🎨 Highlighting complete: {highlighted_count}/{len(sorted_chunks)} chunks highlighted")
    return highlighted_text
