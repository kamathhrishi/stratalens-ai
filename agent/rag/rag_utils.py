#!/usr/bin/env python3
"""
RAG Utilities - Common utility functions for the RAG system

This module contains utility functions that are commonly used across the RAG system,
including text processing, JSON repair, keyword extraction, and ticker extraction.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)


def extract_keywords(query: str) -> List[str]:
    """Extract keywords from query for keyword search."""
    # Remove common stop words and special characters
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'what', 'how', 'when', 'where', 'why', 'which', 'who', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'must', 'shall', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }

    # Clean and split query
    words = re.findall(r'\b\w+\b', query.lower())

    # Filter out stop words and short words
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)

    return unique_keywords[:10]  # Limit to 10 keywords


# Common ticker aliases: maps company name variants to actual stock tickers
TICKER_ALIASES = {
    'TSMC': 'TSM',
    'GOOGLE': 'GOOGL',
    'FACEBOOK': 'META',
    'FB': 'META',
    'BERKSHIRE': 'BRK.B',
    'BRKA': 'BRK.A',
    'BRKB': 'BRK.B',
}


def normalize_ticker(ticker: str) -> str:
    """Normalize a ticker symbol using the alias map."""
    if not ticker:
        return ticker
    upper = ticker.upper()
    return TICKER_ALIASES.get(upper, upper)


def extract_ticker_simple(query: str) -> Optional[str]:
    """Simple synchronous ticker extraction using regex patterns."""
    # Look for $TICKER format (most common in financial queries)
    ticker_match = re.search(r'\$([A-Z]{1,5})\b', query.upper())
    if ticker_match:
        return normalize_ticker(ticker_match.group(1))

    # Look for standalone uppercase words that might be tickers (2-5 chars)
    words = query.upper().split()
    for word in words:
        # Clean the word of punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        if len(clean_word) >= 2 and len(clean_word) <= 5 and clean_word.isalpha():
            # This is a potential ticker, but we should be conservative
            # Only return if it's a common ticker pattern
            if clean_word in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA']:
                return normalize_ticker(clean_word)

    return None


def repair_unterminated_strings(json_text: str) -> str:
    """Repair unterminated strings in JSON."""
    # Find unterminated strings and close them
    # This is a simple heuristic - look for unclosed quotes
    lines = json_text.split('\n')
    repaired_lines = []
    
    for line in lines:
        # Count quotes in the line
        quote_count = line.count('"')
        if quote_count % 2 == 1:  # Odd number of quotes means unterminated
            # Try to close the string at the end of the line
            if line.strip().endswith(','):
                line = line.rstrip(',') + '",'
            else:
                line = line + '"'
        repaired_lines.append(line)
    
    return '\n'.join(repaired_lines)


def repair_escape_issues(json_text: str) -> str:
    """Repair common escape issues in JSON strings.

    Note: Currently a no-op. Escape issues are typically handled by json.loads itself.
    This function exists as a placeholder for future more sophisticated escape repair logic.
    """
    # TODO: Implement proper escape sequence repair if needed
    return json_text


def repair_incomplete_structure(json_text: str) -> str:
    """Repair incomplete JSON structure."""
    # Remove trailing commas
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    # If the JSON seems to be cut off, try to close it properly
    if not json_text.strip().endswith('}'):
        # Count open braces
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        
        if open_braces > close_braces:
            # Add missing closing braces
            json_text += '}' * (open_braces - close_braces)
    
    return json_text


def repair_truncated_response(json_text: str) -> str:
    """Repair truncated JSON responses by completing missing fields."""
    # Check if the response looks truncated
    if not json_text.strip().endswith('}'):
        # Try to complete common truncated patterns
        if '"suggested_improvements":[' in json_text and not json_text.strip().endswith(']'):
            # Complete the array
            json_text = json_text.rstrip() + ']'
        
        if '"extracted_tickers":[' in json_text and not json_text.strip().endswith(']'):
            # Complete the array
            json_text = json_text.rstrip() + ']'
        
        # Add missing closing braces
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)
    
    return json_text


def extract_json_from_content(json_text: str) -> str:
    """Extract JSON from mixed content (e.g., text before/after JSON)."""
    # Look for JSON object pattern
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, json_text, re.DOTALL)
    
    if matches:
        # Return the longest match (most likely to be complete)
        return max(matches, key=len)
    
    # If no pattern found, try to find the start of JSON
    start_idx = json_text.find('{')
    if start_idx != -1:
        # Try to extract from the first opening brace
        potential_json = json_text[start_idx:]
        # Try to find a reasonable end point
        brace_count = 0
        end_idx = 0
        for i, char in enumerate(potential_json):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > 0:
            return potential_json[:end_idx]
    
    return json_text


def parse_json_with_repair(json_text: str, attempt: int, QuestionAnalysisResult=None, rag_logger=None) -> Dict[str, Any]:
    """Parse JSON with repair attempts for common malformation patterns."""
    # Clean the input text first
    json_text = json_text.strip()
    if rag_logger:
        rag_logger.info(f"🔍 JSON repair attempt {attempt + 1}: Processing {len(json_text)} characters")
        rag_logger.info(f"🔍 JSON text preview: {json_text[:200]}...")
    
    # Check if we have any content at all
    if len(json_text) == 0:
        if rag_logger:
            rag_logger.error("❌ CRITICAL: Empty JSON text provided for parsing!")
        raise json.JSONDecodeError("Empty JSON text", json_text, 0)
    
    # First, try direct parsing with Pydantic validation
    parsed_json = None
    needs_news_from_raw = False
    try:
        parsed_json = json.loads(json_text)
        # Extract needs_latest_news before validation in case validation fails
        needs_news_from_raw = parsed_json.get('needs_latest_news', False)
        
        # Validate with Pydantic model if provided
        if QuestionAnalysisResult:
            validated_result = QuestionAnalysisResult(**parsed_json)
            if rag_logger:
                rag_logger.info("✅ JSON successfully parsed and validated with Pydantic")
            result = validated_result.dict()
            # Ensure needs_latest_news is preserved
            if 'needs_latest_news' not in result and needs_news_from_raw:
                result['needs_latest_news'] = needs_news_from_raw
            return result
        else:
            if rag_logger:
                rag_logger.info("✅ JSON successfully parsed")
            return parsed_json
    except json.JSONDecodeError as e:
        if rag_logger:
            rag_logger.warning(f"🔧 Direct JSON parsing failed, attempting repairs: {e}")
    except Exception as e:
        if rag_logger:
            rag_logger.warning(f"🔧 Pydantic validation failed, attempting JSON repairs: {e}")
            # If we have parsed_json, preserve needs_latest_news for later use
            if parsed_json and 'needs_latest_news' in parsed_json:
                needs_news_from_raw = parsed_json.get('needs_latest_news', False)
                if rag_logger:
                    rag_logger.info(f"📰 Preserving needs_latest_news={needs_news_from_raw} from raw JSON")
    
    # Repair attempt 1: Fix unterminated strings
    try:
        repaired_text = repair_unterminated_strings(json_text)
        parsed_json = json.loads(repaired_text)
        if QuestionAnalysisResult:
            validated_result = QuestionAnalysisResult(**parsed_json)
            if rag_logger:
                rag_logger.info("✅ Unterminated string repair successful with Pydantic validation")
            return validated_result.dict()
        else:
            if rag_logger:
                rag_logger.info("✅ Unterminated string repair successful")
            return parsed_json
    except (json.JSONDecodeError, Exception) as e:
        if rag_logger:
            rag_logger.warning(f"🔧 Unterminated string repair failed: {e}")
    
    # Repair attempt 2: Fix common escape issues
    try:
        repaired_text = repair_escape_issues(json_text)
        parsed_json = json.loads(repaired_text)
        if QuestionAnalysisResult:
            validated_result = QuestionAnalysisResult(**parsed_json)
            if rag_logger:
                rag_logger.info("✅ Escape issue repair successful with Pydantic validation")
            return validated_result.dict()
        else:
            if rag_logger:
                rag_logger.info("✅ Escape issue repair successful")
            return parsed_json
    except (json.JSONDecodeError, Exception) as e:
        if rag_logger:
            rag_logger.warning(f"🔧 Escape issue repair failed: {e}")
    
    # Repair attempt 3: Fix incomplete JSON structure
    try:
        repaired_text = repair_incomplete_structure(json_text)
        parsed_json = json.loads(repaired_text)
        if QuestionAnalysisResult:
            validated_result = QuestionAnalysisResult(**parsed_json)
            if rag_logger:
                rag_logger.info("✅ Incomplete structure repair successful with Pydantic validation")
            return validated_result.dict()
        else:
            if rag_logger:
                rag_logger.info("✅ Incomplete structure repair successful")
            return parsed_json
    except (json.JSONDecodeError, Exception) as e:
        if rag_logger:
            rag_logger.warning(f"🔧 Incomplete structure repair failed: {e}")
    
    # Repair attempt 4: Fix truncated responses
    try:
        repaired_text = repair_truncated_response(json_text)
        parsed_json = json.loads(repaired_text)
        if QuestionAnalysisResult:
            validated_result = QuestionAnalysisResult(**parsed_json)
            if rag_logger:
                rag_logger.info("✅ Truncated response repair successful with Pydantic validation")
            return validated_result.dict()
        else:
            if rag_logger:
                rag_logger.info("✅ Truncated response repair successful")
            return parsed_json
    except (json.JSONDecodeError, Exception) as e:
        if rag_logger:
            rag_logger.warning(f"🔧 Truncated response repair failed: {e}")
    
    # Repair attempt 5: Extract JSON from mixed content
    try:
        repaired_text = extract_json_from_content(json_text)
        parsed_json = json.loads(repaired_text)
        if QuestionAnalysisResult:
            validated_result = QuestionAnalysisResult(**parsed_json)
            if rag_logger:
                rag_logger.info("✅ JSON extraction repair successful with Pydantic validation")
            return validated_result.dict()
        else:
            if rag_logger:
                rag_logger.info("✅ JSON extraction repair successful")
            return parsed_json
    except (json.JSONDecodeError, Exception) as e:
        if rag_logger:
            rag_logger.warning(f"🔧 JSON extraction repair failed: {e}")
    
    # If all repairs fail, raise the original error
    raise json.JSONDecodeError("All JSON repair attempts failed", json_text, 0)


def generate_user_friendly_limit_message(limits_exceeded: Dict[str, Any]) -> str:
    """Generate user-friendly messages about limits that were exceeded."""
    messages = []
    
    # Handle ticker limits
    if 'tickers' in limits_exceeded:
        ticker_info = limits_exceeded['tickers']
        requested = ticker_info['requested']
        processed = ticker_info['processed']
        skipped = ticker_info['skipped']
        
        if requested > processed:
            ticker_message = f"📊 **Note**: You requested analysis for {requested} companies, but I've limited the analysis to the first {processed} companies ({', '.join(skipped)} were skipped) to ensure optimal performance."
            messages.append(ticker_message)
    
    # Handle quarter limits
    if 'quarters' in limits_exceeded:
        quarter_info = limits_exceeded['quarters']
        requested = quarter_info['requested']
        processed = quarter_info['processed']
        skipped = quarter_info['skipped']
        
        if requested > processed:
            # Convert quarter format to user-friendly format
            skipped_friendly = []
            for q in skipped:
                if '_q' in q:
                    year, quarter = q.split('_q')
                    skipped_friendly.append(f"{year} Q{quarter}")
                else:
                    skipped_friendly.append(q)
            
            if processed == 12:
                year_info = " (3 years)"
            else:
                year_info = f" ({processed/4:.1f} years)" if processed >= 4 else ""
            quarter_message = f"📅 **Note**: You requested data for {requested} quarters, but I've limited the analysis to the most recent {processed} quarters{year_info} ({', '.join(skipped_friendly)} were skipped) to ensure optimal performance."
            messages.append(quarter_message)
    
    # Combine messages
    if messages:
        return "⚠️ " + " ".join(messages) + "\n\n"
    
    return ""


def deduplicate_citations_and_chunks(
    best_citations: List[Any],
    best_chunks: List[Dict[str, Any]],
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """
    Deduplicate citations and chunks from iterative improvement results.
    Returns (unique_citations, unique_chunks) for use in final response.
    """
    log = logger_instance or logger

    unique_citations = []
    seen_citations = set()

    news_in_best = [c for c in best_citations if isinstance(c, dict) and c.get('type') == 'news']
    tenk_in_best = [c for c in best_citations if isinstance(c, dict) and c.get('type') == '10-K']
    log.info(
        f"🔍 DEBUG: best_citations contains {len(news_in_best)} news citations, "
        f"{len(tenk_in_best)} 10-K citations, {len(best_citations)} total"
    )

    for citation in best_citations:
        citation_key = citation
        if isinstance(citation, dict):
            if citation.get('type') == 'news':
                citation_key = citation.get('marker', citation.get('url', str(citation)))
                log.debug(
                    f"🔍 Processing news citation: marker={citation.get('marker')}, "
                    f"url={citation.get('url', '')[:50] if citation.get('url') else 'None'}"
                )
            elif citation.get('type') == '10-K':
                citation_key = citation.get('marker', str(citation))
            else:
                citation_key = (
                    citation.get('citation') or citation.get('id')
                    or citation.get('chunk_index') or str(citation)
                )

        if citation_key not in seen_citations:
            seen_citations.add(citation_key)
            unique_citations.append(citation)
            if isinstance(citation, dict) and citation.get('type') == 'news':
                log.info(f"✅ Added news citation to unique_citations: {citation.get('marker')}")
        elif isinstance(citation, dict) and citation.get('type') == 'news':
            log.warning(
                f"⚠️ Duplicate news citation skipped: {citation.get('marker')}, "
                f"url={citation.get('url', '')[:50] if citation.get('url') else 'None'}"
            )

    unique_chunks = []
    seen_chunk_citations = set()

    for chunk in best_chunks:
        chunk_citation = chunk.get('citation')
        if isinstance(chunk_citation, dict):
            chunk_citation = (
                chunk_citation.get('citation') or chunk_citation.get('id')
                or chunk_citation.get('chunk_index') or str(chunk_citation)
            )
        if chunk_citation not in seen_chunk_citations:
            seen_chunk_citations.add(chunk_citation)
            unique_chunks.append(chunk)

    log.info(
        f"📎 Deduplicated citations: {len(best_citations)} -> {len(unique_citations)} "
        f"unique citations from all iterations"
    )
    log.info(
        f"📄 Deduplicated chunks: {len(best_chunks)} -> {len(unique_chunks)} "
        f"unique chunks from all iterations"
    )
    return (unique_citations, unique_chunks)


def combine_search_results(vector_results: List[Dict[str, Any]], keyword_results: List[Dict[str, Any]], 
                           vector_weight: float = 0.7, keyword_weight: float = 0.3, 
                           similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Combine and rank results from vector and keyword search using weighted scoring."""
    logger.info(f"🔄 Combining search results: vector={len(vector_results)}, keyword={len(keyword_results)}")
    
    # Create a dictionary to store combined results by citation
    combined_results = {}
    
    # Add vector search results
    for chunk in vector_results:
        citation = chunk['citation']
        chunk['search_type'] = 'vector'
        chunk['combined_score'] = chunk['similarity'] * vector_weight
        combined_results[citation] = chunk
    
    # Add keyword search results
    for chunk in keyword_results:
        citation = chunk['citation']
        if citation in combined_results:
            # If chunk exists in both, combine the scores
            existing_chunk = combined_results[citation]
            existing_chunk['combined_score'] += chunk['similarity'] * keyword_weight
            existing_chunk['search_type'] = 'hybrid'  # Mark as hybrid result
            logger.info(f"🔄 Combined scores for citation {citation}: vector={existing_chunk['similarity']:.3f}, keyword={chunk['similarity']:.3f}, combined={existing_chunk['combined_score']:.3f}")
        else:
            # New chunk from keyword search
            chunk['search_type'] = 'keyword'
            chunk['combined_score'] = chunk['similarity'] * keyword_weight
            combined_results[citation] = chunk
    
    # Sort by combined score
    sorted_results = sorted(combined_results.values(), key=lambda x: x['combined_score'], reverse=True)
    
    # Apply similarity threshold to combined results
    filtered_results = [chunk for chunk in sorted_results if chunk['combined_score'] >= similarity_threshold]
    
    logger.info(f"✅ Combined search results: {len(sorted_results)} total, {len(filtered_results)} above threshold {similarity_threshold}")
    
    # Log search type distribution
    search_types = {}
    for chunk in filtered_results:
        search_type = chunk.get('search_type', 'unknown')
        search_types[search_type] = search_types.get(search_type, 0) + 1
    
    logger.info(f"📊 Search type distribution: {search_types}")
    
    return filtered_results


def assess_answer_quality(answer: str, chunk_count: int) -> Dict[str, Any]:
    """Assess if an answer needs chunk context for evaluation based on chunk count."""
    if chunk_count <= 3:
        return {'is_insufficient': True, 'reason': f'Only {chunk_count} chunks used - providing chunk context'}
    
    return {'is_insufficient': False, 'reason': f'{chunk_count} chunks used - using standard evaluation'}
