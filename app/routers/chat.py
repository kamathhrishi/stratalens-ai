from fastapi import APIRouter, HTTPException, Query, Depends, status, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import time
import pandas as pd
import logging
import traceback
import os
import json
import uuid
import datetime
import asyncpg
import asyncio
from typing import Optional
from datetime import datetime
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.schemas import (
    ChatMessage, ChatResponse, ChatCitation, ChatHistoryItem,
    ChatHistoryResponse, ChatClearRequest, ChatClearResponse,
    ChatConversation, ChatConversationMessage, ChatConversationsResponse, ChatConversationRequest
)
from config import settings
from app.utils import rate_limiter, RATE_LIMIT_PER_MONTH, record_successful_query_usage
from agent import Agent as RAGSystem, create_agent as create_rag_system
from app.utils.logging_utils import log_info, log_error, log_warning
from app.utils import create_error_response, raise_sanitized_http_exception
from db.db_connection_utils import get_postgres_connection
from app.auth.auth_utils import get_current_user

# Import PostgreSQL connection utilities
import psycopg2
from psycopg2.extras import RealDictCursor

# Import centralized utilities
from app.auth.auth_utils import get_current_user, get_optional_user
from db.db_utils import get_db
from app.utils import create_error_response, raise_sanitized_http_exception
from analytics.analytics_utils import log_chat_analytics, get_analytics_summary, get_analytics_data
from analytics.analytics import UserType, AnalyticsQuery, AnalyticsResponse, AnalyticsSummary


# Set up logging
logger = logging.getLogger(__name__)

# Import Logfire for observability (optional)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False

# Shared utilities
from db.db_connection_utils import get_postgres_connection

# Create router
router = APIRouter(prefix="", tags=["chat"])

# Global dictionary to track active chat requests for cancellation
active_chat_requests = {}

# Global message counter for server-wide message limit
_global_message_count = 0
_MAX_MESSAGES_PER_SERVER = 100

# Global demo tracking dictionary
_demo_ip_tracking = {}

# Create RAG system instance for chat functionality
try:
    rag_system = create_rag_system()
    log_info("✅ RAG system initialized for chat router")
except Exception as e:
    log_error(f"❌ Failed to initialize RAG system for chat: {e}")
    rag_system = None

def get_chat_rag_system():
    """Get the RAG system instance for chat functionality"""
    if rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Chat features disabled - RAG system not available"
        )
    return rag_system

def get_max_iterations() -> int:
    """Get the max iterations for agent mode.

    Returns:
        Number of iterations (default: 3)
    """
    return 3

def check_and_increment_message_count():
    """Check if server message limit is reached and increment counter.
    
    Raises:
        HTTPException: If message limit is exceeded
    """
    global _global_message_count
    
    if _global_message_count >= _MAX_MESSAGES_PER_SERVER:
        logger.warning(f"Server message limit reached: {_global_message_count}/{_MAX_MESSAGES_PER_SERVER}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service temporarily unavailable. Maximum message capacity reached. Please contact admin."
        )
    
    _global_message_count += 1
    logger.info(f"Message count: {_global_message_count}/{_MAX_MESSAGES_PER_SERVER}")

# =============================================================================
# CHAT ENDPOINTS (RAG-POWERED)
# =============================================================================


@router.post("/message/stream-v2")
async def stream_chat_message_v2(
    request: Request,
    chat_request: ChatMessage,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Stream chat message processing with step-by-step updates using POST"""
    
    # Check global message limit
    check_and_increment_message_count()
    
    logger.info(f"✅ POST /message/stream-v2 called successfully!")
    logger.info(f"📦 Request body received: {chat_request}")
    
    user_id = current_user["id"]
    is_admin = current_user.get("is_admin", False)
    
    message = chat_request.message
    comprehensive = chat_request.comprehensive
    max_iterations = get_max_iterations()

    logger.info(f"🔄 Agent mode: max_iterations: {max_iterations}")
    
    # Create Logfire span for this chat request
    if LOGFIRE_AVAILABLE:
        logfire.info(
            "chat.stream_v2",
            user_id=user_id,
            is_admin=is_admin,
            max_iterations=max_iterations,
            comprehensive=comprehensive,
            message_length=len(message)
        )
    
    # Rate limiting
    try:
        allowed, limit_info = await rate_limiter.check_rate_limit_with_monthly(user_id, is_admin, db)
        if not allowed:
            error_event = {
                'type': 'error',
                'error': 'RATE_LIMIT_EXCEEDED',
                'message': limit_info['message']
            }
            async def error_generator():
                yield f"data: {json.dumps(error_event)}\n\n"
            return StreamingResponse(error_generator(), media_type="text/event-stream")
        rate_limiter.record_request(user_id)
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}")
        error_event = {'type': 'error', 'message': 'Rate limit check failed'}
        async def error_generator():
            yield f"data: {json.dumps(error_event)}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    # Check RAG system
    if rag_system is None:
        error_event = {'type': 'error', 'message': 'Chat service unavailable'}
        async def error_generator():
            yield f"data: {json.dumps(error_event)}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")
    
    # Get or create conversation
    conversation_id = None
    if chat_request.conversation_id:
        # Validate existing conversation
        try:
            conv_uuid = uuid.UUID(chat_request.conversation_id)
            existing_conv = await db.fetchrow('''
                SELECT id FROM chat_conversations 
                WHERE id = $1 AND user_id = $2
            ''', conv_uuid, uuid.UUID(user_id))
            
            if existing_conv:
                conversation_id = conv_uuid
                logger.info(f"📂 Using existing conversation: {conversation_id}")
            else:
                logger.warning(f"⚠️ Conversation not found, creating new")
        except ValueError:
            logger.warning(f"⚠️ Invalid conversation ID format, creating new")
    
    if not conversation_id:
        # Create new conversation
        title = message[:60] + ("..." if len(message) > 60 else "")
        conversation_id = await db.fetchval('''
            INSERT INTO chat_conversations (user_id, title)
            VALUES ($1, $2)
            RETURNING id
        ''', uuid.UUID(user_id), title)
        logger.info(f"✅ Created new conversation: {conversation_id}")
    
    # Save user message first
    user_message_id = await db.fetchval('''
        INSERT INTO chat_messages (conversation_id, role, content)
        VALUES ($1, 'user', $2)
        RETURNING id
    ''', conversation_id, message)
    logger.info(f"💬 Saved user message: {user_message_id}")
    
    async def event_generator():
        # STREAMING + DATABASE CONNECTION PATTERN:
        # =========================================
        # Acquire a dedicated connection INSIDE the generator that lives for the entire stream.
        # 
        # Why? FastAPI's Depends(get_db) returns connections via async context manager.
        # When the endpoint returns StreamingResponse, the function exits immediately,
        # but the generator is still running. Without this pattern, the connection would
        # be returned to the pool while streaming is still happening, causing errors.
        #
        # Solution: Acquire a fresh connection inside the generator's lifetime.
        # Combined with contextvars in ConversationMemory, this ensures:
        # 1. Connection lives for entire stream duration
        # 2. Request isolation (no cross-user contamination)
        # 3. Thread-safe async operation
        from db.db_utils import _db_pool
        if _db_pool is None:
            error_event = {'type': 'error', 'message': 'Database unavailable'}
            yield f"data: {json.dumps(error_event)}\n\n"
            return
        
        async with _db_pool.acquire() as stream_db:
            # Set database connection for RAG system (request-scoped via contextvars)
            rag_system.set_database_connection(stream_db)
            
            query_successful = False
            start_time = time.time()
            
            try:
                logger.info(f"Starting streaming chat for user {user_id}: '{message[:100]}...'")
                logger.info(f"🔄 MAX_ITERATIONS parameter received: {max_iterations}")
                
                # Execute RAG flow with streaming - events are yielded directly
                final_result = None
                
                async for event in rag_system.execute_rag_flow(
                    question=message,
                    show_details=False,
                    comprehensive=comprehensive,
                    max_iterations=max_iterations,
                    conversation_id=str(conversation_id),
                    stream=True  # Enable streaming
                ):
                    # Check if client disconnected
                    if await request.is_disconnected():
                        logger.warning(f"Client disconnected for user {user_id}")
                        break
                    
                    # Check for cancellation
                    if user_id in active_chat_requests and active_chat_requests[user_id].get("cancelled", False):
                        logger.info(f"Request cancelled for user {user_id}")
                        break
                    
                    # Filter out evaluation reasoning from events before sending
                    # Remove evaluation text from message and data fields
                    if event.get('message') and isinstance(event['message'], str):
                        msg = event['message']
                        # Check if it's evaluation reasoning
                        if (('The answer' in msg or 'The response' in msg) and 
                            ('omits' in msg or 'lacks' in msg or 'gaps' in msg or 'missing' in msg or 
                             'does not provide' in msg or 'prevent' in msg)):
                            # Replace with brief action message
                            if 'iteration' in event.get('type', ''):
                                event['message'] = 'Analyzing answer quality'
                            else:
                                event['message'] = 'Processing...'
                    
                    # Also filter data fields
                    if event.get('data') and isinstance(event['data'], dict):
                        for field in ['reasoning', 'iteration_reasoning', 'evaluation_summary', 'reason']:
                            if field in event['data'] and isinstance(event['data'][field], str):
                                reasoning_text = event['data'][field]
                                if (('The answer' in reasoning_text or 'The response' in reasoning_text) and 
                                    ('omits' in reasoning_text or 'lacks' in reasoning_text or 'gaps' in reasoning_text)):
                                    # Clear evaluation reasoning
                                    event['data'][field] = ''
                    
                    # Log event being forwarded
                    logger.info(f"📡 ROUTER: Forwarding event to client: type={event.get('type')}, step={event.get('step')}")
                    
                    # Send event
                    if event.get('type') == 'result':
                        final_result = event.get('data')
                        query_successful = True
                        logger.info(f"✅ ROUTER: Final result event - query successful")
                        # Inject conversation_id into the final result
                        event['conversation_id'] = str(conversation_id)
                        logger.info(f"📂 Added conversation_id to result: {conversation_id}")
                    
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # For token events, add small delay to ensure browser receives them in real-time
                    # Without this, tokens arrive too fast and get buffered
                    if event.get('type') == 'token':
                        await asyncio.sleep(0.01)  # 10ms delay = ~100 tokens/sec for smooth streaming
                    else:
                        # Minimal yield to allow context switching for other events
                        await asyncio.sleep(0)
                
                # Record usage if successful
                if query_successful and final_result:
                    await record_successful_query_usage(user_id, stream_db, settings.RATE_LIMITING.COST_PER_REQUEST)
                    
                    # Save assistant message to conversation
                    try:
                        assistant_message_id = await stream_db.fetchval('''
                            INSERT INTO chat_messages (conversation_id, role, content, citations, context)
                            VALUES ($1, 'assistant', $2, $3, $4)
                            RETURNING id
                        ''', conversation_id, final_result.get('answer', ''), json.dumps(final_result.get('citations', [])), json.dumps({
                            'timing': final_result.get('timing', {}),
                            'comprehensive': comprehensive
                        }))
                        
                        # Update conversation timestamp
                        await stream_db.execute('''
                            UPDATE chat_conversations 
                            SET updated_at = CURRENT_TIMESTAMP 
                            WHERE id = $1
                        ''', conversation_id)
                        
                        logger.info(f"✅ Assistant message saved: {assistant_message_id}")
                    except Exception as e:
                        logger.error(f"Failed to save assistant message: {e}")
                    
                    # Log analytics
                    try:
                        execution_time = time.time() - start_time
                        await log_chat_analytics(
                            db=stream_db,
                            ip_address="unknown",
                            user_type=UserType.AUTHORIZED,
                            query_text=message,
                            comprehensive_search=comprehensive,
                            success=True,
                            response_time_ms=execution_time * 1000,
                            citations_count=len(final_result.get('citations', [])),
                            user_id=user_id
                        )
                    except Exception as e:
                        logger.error(f"Failed to log analytics: {e}")
            
            except Exception as e:
                logger.error(f"Error during chat streaming: {e}", exc_info=True)
                error_event = {
                    'type': 'error',
                    'message': 'Sorry, an error occurred while processing your request. Please try again.'
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                
                # Log failed analytics
                try:
                    await log_chat_analytics(
                        db=stream_db,
                        ip_address="unknown",
                        user_type=UserType.AUTHORIZED,
                        query_text=message,
                        comprehensive_search=comprehensive,
                        success=False,
                        response_time_ms=0,
                        citations_count=0,
                        error_message=str(e),
                        user_id=user_id
                    )
                except Exception as analytics_error:
                    logger.error(f"Failed to log failed analytics: {analytics_error}")
    
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    }
    logger.info(f"✅ Returning StreamingResponse for authenticated stream-v2")
    return StreamingResponse(event_generator(), headers=headers)

@router.post("/landing/demo/stream-v2")
async def stream_landing_demo_message_v2(
    request: Request,
    chat_request: ChatMessage,
    db: asyncpg.Connection = Depends(get_db)
):
    """Stream demo chat message with step-by-step updates using POST"""
    
    # Check global message limit
    check_and_increment_message_count()
    
    logger.info(f"✅ POST /landing/demo/stream-v2 called successfully!")
    logger.info(f"📦 Request body received: {chat_request}")
    
    message = chat_request.message
    comprehensive = chat_request.comprehensive
    max_iterations = get_max_iterations()
    session_id = chat_request.session_id or f"session_{int(time.time() * 1000)}"

    logger.info(f"🔄 Agent mode: max_iterations: {max_iterations}")
    
    logger.info(f"🌟 Landing page demo stream from session {session_id}: {message[:100]}...")
    logger.info(f"🔄 MAX_ITERATIONS parameter received for demo: {max_iterations}")
    
    # Global in-memory rate limiting for demo (per session)
    global _demo_ip_tracking
    
    current_time = time.time()
    cutoff_time = current_time - (24 * 60 * 60)
    _demo_ip_tracking = {
        session: data for session, data in _demo_ip_tracking.items() 
        if isinstance(data, dict) and data.get('last_used', 0) > cutoff_time
    }
    
    # Check demo limit (skip in local/development environment)
    # Only enforce demo limit in production
    env = settings.ENVIRONMENT.ENVIRONMENT.lower()
    is_local_or_dev = env in ['local', 'development']
    
    logger.info(f"🔍 Demo limit check - Environment: {env}, is_local_or_dev: {is_local_or_dev}")
    
    if not is_local_or_dev:
        if session_id not in _demo_ip_tracking:
            _demo_ip_tracking[session_id] = {'count': 0, 'last_used': current_time}
        
        current_count = _demo_ip_tracking[session_id]['count']
        max_demo_messages = 5
        
        if current_count >= max_demo_messages:
            error_event = {
                'type': 'error',
                'error': 'DEMO_LIMIT_EXCEEDED',
                'message': f"You've used all {max_demo_messages} free demo messages! Sign up to continue chatting."
            }
            async def error_generator():
                yield f"data: {json.dumps(error_event)}\n\n"
            return StreamingResponse(error_generator(), media_type="text/event-stream")
        
        # Increment count
        _demo_ip_tracking[session_id]['count'] += 1
        _demo_ip_tracking[session_id]['last_used'] = current_time
    else:
        # Local/development environment: skip demo limit
        # Also clear any existing tracking for this session to reset the count
        if session_id in _demo_ip_tracking:
            _demo_ip_tracking[session_id]['count'] = 0
        logger.info(f"🔓 Demo limit disabled (environment: {env})")
    
    # Get or create conversation for demo user
    conversation_id = None
    if chat_request.conversation_id:
        # Demo user wants to continue an existing conversation
        try:
            conv_uuid = uuid.UUID(chat_request.conversation_id)
            existing_conv = await db.fetchrow('''
                SELECT id FROM chat_conversations 
                WHERE id = $1 AND user_id IS NULL AND title LIKE 'Demo:%'
            ''', conv_uuid)
            
            if existing_conv:
                conversation_id = conv_uuid
                logger.info(f"📂 Demo stream continuing conversation: {conversation_id}")
        except (ValueError, Exception) as e:
            logger.warning(f"⚠️ Invalid demo conversation ID: {e}")
    
    if not conversation_id:
        # Create new demo conversation
        title = f"Demo: {message[:50]}..." if len(message) > 50 else f"Demo: {message}"
        conversation_id = await db.fetchval('''
            INSERT INTO chat_conversations (user_id, title)
            VALUES (NULL, $1)
            RETURNING id
        ''', title)
        logger.info(f"✅ Created new demo stream conversation: {conversation_id}")
    
    # Save user message
    user_message_id = await db.fetchval('''
        INSERT INTO chat_messages (conversation_id, role, content)
        VALUES ($1, 'user', $2)
        RETURNING id
    ''', conversation_id, message)
    logger.info(f"💬 Saved demo stream user message: {user_message_id}")
    
    async def event_generator():
        # STREAMING + DATABASE CONNECTION PATTERN (see authenticated endpoint for full explanation)
        # Acquire connection inside generator to ensure it lives for entire stream duration
        from db.db_utils import _db_pool
        if _db_pool is None:
            error_event = {'type': 'error', 'message': 'Database unavailable'}
            yield f"data: {json.dumps(error_event)}\n\n"
            return
        
        async with _db_pool.acquire() as stream_db:
            # Get RAG system
            if not rag_system:
                error_event = {'type': 'error', 'message': 'Chat service unavailable'}
                yield f"data: {json.dumps(error_event)}\n\n"
                return
            
            # Set database connection for RAG system (request-scoped via contextvars)
            rag_system.set_database_connection(stream_db)
            
            query_successful = False
            start_time = time.time()
            
            try:
                logger.info(f"Starting demo stream for session {session_id}")
                
                # Execute RAG flow with streaming
                final_result = None
                
                async for event in rag_system.execute_rag_flow(
                    question=message,
                    show_details=False,
                    comprehensive=comprehensive,
                    max_iterations=max_iterations,
                    conversation_id=str(conversation_id),
                    stream=True  # Enable streaming
                ):
                    # Check if client disconnected
                    if await request.is_disconnected():
                        logger.warning(f"Client disconnected for demo session {session_id}")
                        break
                    
                    # Filter out evaluation reasoning from events before sending (same as authenticated endpoint)
                    if event.get('message') and isinstance(event['message'], str):
                        msg = event['message']
                        if (('The answer' in msg or 'The response' in msg) and 
                            ('omits' in msg or 'lacks' in msg or 'gaps' in msg or 'missing' in msg or 
                             'does not provide' in msg or 'prevent' in msg)):
                            if 'iteration' in event.get('type', ''):
                                event['message'] = 'Analyzing answer quality'
                            else:
                                event['message'] = 'Processing...'
                    
                    if event.get('data') and isinstance(event['data'], dict):
                        for field in ['reasoning', 'iteration_reasoning', 'evaluation_summary', 'reason']:
                            if field in event['data'] and isinstance(event['data'][field], str):
                                reasoning_text = event['data'][field]
                                if (('The answer' in reasoning_text or 'The response' in reasoning_text) and 
                                    ('omits' in reasoning_text or 'lacks' in reasoning_text or 'gaps' in reasoning_text)):
                                    event['data'][field] = ''
                    
                    # Log event being forwarded
                    logger.info(f"📡 DEMO ROUTER: Forwarding event to client: type={event.get('type')}, step={event.get('step')}")
                    
                    # Send event
                    if event.get('type') == 'result':
                        final_result = event.get('data')
                        query_successful = True
                        logger.info(f"✅ DEMO ROUTER: Final result event - query successful")
                        # Inject conversation_id into the final result
                        event['conversation_id'] = str(conversation_id)
                        logger.info(f"📂 Added conversation_id to result: {conversation_id}")
                    
                    yield f"data: {json.dumps(event)}\n\n"
                    
                    # For token events, add small delay to ensure browser receives them in real-time
                    # Without this, tokens arrive too fast and get buffered
                    if event.get('type') == 'token':
                        await asyncio.sleep(0.01)  # 10ms delay = ~100 tokens/sec for smooth streaming
                    else:
                        # Minimal yield to allow context switching for other events
                        await asyncio.sleep(0)
                
                # Save assistant message and log analytics for demo
                if final_result:
                    # Save assistant message to conversation
                    try:
                        assistant_message_id = await stream_db.fetchval('''
                            INSERT INTO chat_messages (conversation_id, role, content, citations, context)
                            VALUES ($1, 'assistant', $2, $3, $4)
                            RETURNING id
                        ''', conversation_id, final_result.get('answer', ''), json.dumps(final_result.get('citations', [])), json.dumps({
                            'timing': final_result.get('timing', {}),
                            'comprehensive': comprehensive
                        }))
                        
                        # Update conversation timestamp
                        await stream_db.execute('''
                            UPDATE chat_conversations 
                            SET updated_at = CURRENT_TIMESTAMP 
                            WHERE id = $1
                        ''', conversation_id)
                        
                        logger.info(f"✅ Demo stream assistant message saved: {assistant_message_id}")
                    except Exception as e:
                        logger.error(f"Failed to save demo stream assistant message: {e}")
                    
                    # Log analytics
                    try:
                        client_ip = request.client.host if request.client else "unknown"
                        execution_time = time.time() - start_time
                        await log_chat_analytics(
                            db=stream_db,
                            ip_address=client_ip,
                            user_type=UserType.DEMO,
                            query_text=message,
                            comprehensive_search=comprehensive,
                            success=True,
                            response_time_ms=execution_time * 1000,
                            citations_count=len(final_result.get('citations', [])),
                            session_id=session_id
                        )
                    except Exception as e:
                        logger.error(f"Failed to log demo analytics: {e}")
            
            except Exception as e:
                logger.error(f"Error during demo streaming: {e}", exc_info=True)
                error_event = {
                    'type': 'error',
                    'message': 'Sorry, an error occurred while processing your request. Please try again.'
                }
                yield f"data: {json.dumps(error_event)}\n\n"

                # Log failed analytics
                try:
                    client_ip = request.client.host if request.client else "unknown"
                    await log_chat_analytics(
                        db=stream_db,
                        ip_address=client_ip,
                        user_type=UserType.DEMO,
                        query_text=message,
                        comprehensive_search=comprehensive,
                        success=False,
                        response_time_ms=0,
                        citations_count=0,
                        error_message=str(e),
                        session_id=session_id
                    )
                except Exception as analytics_error:
                    logger.error(f"Failed to log demo failed analytics: {analytics_error}")
    
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*"
    }
    logger.info(f"✅ Returning StreamingResponse for demo stream-v2")
    return StreamingResponse(event_generator(), headers=headers)


@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: int = Query(50, le=100, description="Maximum number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    search: Optional[str] = Query(None, description="Search term to filter messages"),
    date_from: Optional[str] = Query(None, description="Filter messages from this date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter messages to this date (YYYY-MM-DD)"),
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get chat history for the current user with search and filtering"""
    try:
        user_id = current_user["id"]

        # Build the base query with optional filters
        where_conditions = ["user_id = $1"]
        query_params = [uuid.UUID(user_id)]
        param_counter = 2

        # Add search filter
        if search:
            where_conditions.append(f"(user_message ILIKE ${param_counter} OR assistant_response ILIKE ${param_counter})")
            query_params.append(f"%{search}%")
            param_counter += 1

        # Add date filters
        if date_from:
            try:
                date_from_parsed = datetime.strptime(date_from, "%Y-%m-%d").date()
                where_conditions.append(f"created_at::date >= ${param_counter}")
                query_params.append(date_from_parsed)
                param_counter += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")

        if date_to:
            try:
                date_to_parsed = datetime.strptime(date_to, "%Y-%m-%d").date()
                where_conditions.append(f"created_at::date <= ${param_counter}")
                query_params.append(date_to_parsed)
                param_counter += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        where_clause = " AND ".join(where_conditions)
        
        # Get total count with filters
        count_query = f"SELECT COUNT(*) FROM chat_history WHERE {where_clause}"
        total_count = await db.fetchval(count_query, *query_params)
        
        # Get chat messages with pagination and filters
        messages_query = f'''
            SELECT id, user_message, assistant_response, context, created_at
            FROM chat_history 
            WHERE {where_clause}
            ORDER BY created_at DESC 
            LIMIT ${param_counter} OFFSET ${param_counter + 1}
        '''
        query_params.extend([limit, offset])
        
        messages = await db.fetch(messages_query, *query_params)
        
        # Convert to response format
        chat_items = []
        for message in messages:
            context = json.loads(message['context']) if message['context'] else {}
            
            # Extract citations from context
            citations_data = context.get('citations', [])
            citations = []
            for citation_data in citations_data:
                citations.append(ChatCitation(**citation_data))
            
            chat_items.append(ChatHistoryItem(
                id=str(message['id']),
                user_message=message['user_message'],
                assistant_response=message['assistant_response'],
                citations=citations,
                created_at=message['created_at'],
                context=context
            ))
        
        return ChatHistoryResponse(
            success=True,
            messages=chat_items,
            total_count=total_count,
            filtered_count=len(chat_items) if search or date_from or date_to else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chat history for user {current_user['id']}: {e}")
        error_response = create_error_response(e, "chat history fetch", current_user.get("id"))
        return ChatHistoryResponse(
            success=False,
            error=error_response["error"]
        )


@router.get("/history/{chat_id}")
async def get_chat_by_id(
    chat_id: str,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get a specific chat message by ID"""
    try:
        # Validate UUID format
        try:
            chat_uuid = uuid.UUID(chat_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid chat ID format")
        
        # Get the specific chat message
        message = await db.fetchrow('''
            SELECT id, user_message, assistant_response, context, created_at
            FROM chat_history 
            WHERE id = $1 AND user_id = $2
        ''', chat_uuid, uuid.UUID(current_user["id"]))
        
        if not message:
            raise HTTPException(status_code=404, detail="Chat message not found or access denied")
        
        # Parse context and extract citations
        context = json.loads(message['context']) if message['context'] else {}
        citations_data = context.get('citations', [])
        citations = []
        for citation_data in citations_data:
            citations.append(ChatCitation(**citation_data))
        
        chat_item = ChatHistoryItem(
            id=str(message['id']),
            user_message=message['user_message'],
            assistant_response=message['assistant_response'],
            citations=citations,
            created_at=message['created_at'],
            context=context
        )
        
        return {
            "success": True,
            "message": chat_item,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            f"chat message fetch for ID {chat_id}", 
            current_user.get("id"),
            status_code=500
        )


@router.get("/stats")
async def get_chat_stats(
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get chat statistics for the current user"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        # Get basic stats
        stats = await db.fetchrow('''
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT DATE(created_at)) as active_days,
                MIN(created_at) as first_message,
                MAX(created_at) as last_message,
                AVG(LENGTH(user_message)) as avg_question_length,
                AVG(LENGTH(assistant_response)) as avg_response_length
            FROM chat_history 
            WHERE user_id = $1
        ''', user_id)
        
        # Get messages by day for the last 30 days
        daily_stats = await db.fetch('''
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as message_count
            FROM chat_history 
            WHERE user_id = $1 
                AND created_at >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        ''', user_id)
        
        # Get top topics (based on user messages)
        top_topics = await db.fetch('''
            SELECT 
                SUBSTRING(user_message, 1, 100) as topic_preview,
                LENGTH(assistant_response) as response_length,
                created_at
            FROM chat_history 
            WHERE user_id = $1
            ORDER BY LENGTH(assistant_response) DESC
            LIMIT 10
        ''', user_id)
        
        return {
            "success": True,
            "stats": {
                "total_messages": stats['total_messages'],
                "active_days": stats['active_days'],
                "first_message": stats['first_message'].isoformat() if stats['first_message'] else None,
                "last_message": stats['last_message'].isoformat() if stats['last_message'] else None,
                "avg_question_length": round(float(stats['avg_question_length']) if stats['avg_question_length'] else 0, 2),
                "avg_response_length": round(float(stats['avg_response_length']) if stats['avg_response_length'] else 0, 2)
            },
            "daily_activity": [
                {
                    "date": day['date'].isoformat(),
                    "message_count": day['message_count']
                } for day in daily_stats
            ],
            "top_conversations": [
                {
                    "topic_preview": topic['topic_preview'],
                    "response_length": topic['response_length'],
                    "created_at": topic['created_at'].isoformat()
                } for topic in top_topics
            ]
        }
        
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            "chat statistics fetch", 
            current_user.get("id"),
            status_code=500
        )


@router.post("/export")
async def export_chat_history(
    request: dict,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Export chat history in various formats"""
    try:
        export_format = request.get("format", "json").lower()
        include_citations = request.get("include_citations", True)
        date_from = request.get("date_from")
        date_to = request.get("date_to")
        
        if export_format not in ["json", "csv", "txt"]:
            raise HTTPException(status_code=400, detail="Supported formats: json, csv, txt")
        
        user_id = uuid.UUID(current_user["id"])
        
        # Build query with optional date filters
        where_conditions = ["user_id = $1"]
        query_params = [user_id]
        param_counter = 2
        
        if date_from:
            try:
                date_from_parsed = datetime.strptime(date_from, "%Y-%m-%d").date()
                where_conditions.append(f"created_at::date >= ${param_counter}")
                query_params.append(date_from_parsed)
                param_counter += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use YYYY-MM-DD")
        
        if date_to:
            try:
                date_to_parsed = datetime.strptime(date_to, "%Y-%m-%d").date()
                where_conditions.append(f"created_at::date <= ${param_counter}")
                query_params.append(date_to_parsed)
                param_counter += 1
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use YYYY-MM-DD")
        
        where_clause = " AND ".join(where_conditions)
        
        # Fetch all messages
        messages = await db.fetch(f'''
            SELECT id, user_message, assistant_response, context, created_at
            FROM chat_history 
            WHERE {where_clause}
            ORDER BY created_at ASC
        ''', *query_params)
        
        # Format data based on requested format
        if export_format == "json":
            export_data = []
            for message in messages:
                context = json.loads(message['context']) if message['context'] else {}
                
                item = {
                    "id": str(message['id']),
                    "user_message": message['user_message'],
                    "assistant_response": message['assistant_response'],
                    "created_at": message['created_at'].isoformat()
                }
                
                if include_citations:
                    item["citations"] = context.get('citations', [])
                    item["analysis"] = context.get('analysis', {})
                    item["timing"] = context.get('timing', {})
                
                export_data.append(item)
            
            return {
                "success": True,
                "format": "json",
                "data": export_data,
                "total_messages": len(export_data),
                "exported_at": datetime.utcnow().isoformat()
            }
        
        elif export_format == "csv":
            import io
            output = io.StringIO()
            
            # Write CSV header
            headers = ["ID", "User Message", "Assistant Response", "Created At"]
            if include_citations:
                headers.extend(["Citations Count", "Has Analysis"])
            
            output.write(",".join(f'"{h}"' for h in headers) + "\n")
            
            # Write data rows
            for message in messages:
                context = json.loads(message['context']) if message['context'] else {}
                
                row = [
                    str(message['id']),
                    message['user_message'].replace('"', '""'),
                    message['assistant_response'].replace('"', '""'),
                    message['created_at'].isoformat()
                ]
                
                if include_citations:
                    citations = context.get('citations', [])
                    row.extend([
                        str(len(citations)),
                        "Yes" if context.get('analysis') else "No"
                    ])
                
                output.write(",".join(f'"{field}"' for field in row) + "\n")
            
            csv_content = output.getvalue()
            output.close()
            
            return {
                "success": True,
                "format": "csv",
                "content": csv_content,
                "total_messages": len(messages),
                "exported_at": datetime.utcnow().isoformat()
            }
        
        elif export_format == "txt":
            text_content = f"Chat History Export - {datetime.utcnow().isoformat()}\n"
            text_content += "=" * 60 + "\n\n"
            
            for i, message in enumerate(messages, 1):
                context = json.loads(message['context']) if message['context'] else {}
                
                text_content += f"Conversation #{i}\n"
                text_content += f"Date: {message['created_at'].isoformat()}\n"
                text_content += f"ID: {message['id']}\n"
                text_content += "-" * 40 + "\n"
                text_content += f"Question: {message['user_message']}\n\n"
                text_content += f"Response: {message['assistant_response']}\n"
                
                if include_citations and context.get('citations'):
                    citations = context['citations']
                    text_content += f"\nCitations ({len(citations)}):\n"
                    for j, citation in enumerate(citations, 1):
                        text_content += f"  {j}. {citation.get('company', 'Unknown')} - {citation.get('quarter', 'Unknown')}\n"
                
                text_content += "\n" + "=" * 60 + "\n\n"
            
            return {
                "success": True,
                "format": "txt",
                "content": text_content,
                "total_messages": len(messages),
                "exported_at": datetime.utcnow().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            "chat history export", 
            current_user.get("id"),
            status_code=500
        )


# Legacy endpoint for backward compatibility
@router.delete("/clear")
async def clear_chat_history_legacy(
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Legacy DELETE endpoint - automatically converts to POST /chat/clear"""
    # For backward compatibility, automatically clear with confirmation
    clear_request = ChatClearRequest(confirm=True)
    return await clear_chat_history(clear_request, current_user, db)

# Legacy endpoint for backward compatibility  
@router.post("/send")
async def send_chat_message_legacy(
    request: dict,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Legacy send endpoint - automatically converts to POST /chat/message"""
    # Extract message from old format and convert to new format
    message = request.get("message", "")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    chat_message = ChatMessage(message=message, comprehensive=True)
    return await send_chat_message(chat_message, current_user, db)

@router.post("/clear", response_model=ChatClearResponse)
async def clear_chat_history(
    clear_request: ChatClearRequest,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Clear all chat history for the current user"""
    try:
        if not clear_request.confirm:
            return ChatClearResponse(
                success=False,
                message="Confirmation required to clear chat history"
            )
        
        user_id = current_user["id"]
        
        # Count conversations before deletion
        count_before = await db.fetchval('''
            SELECT COUNT(*) FROM chat_conversations WHERE user_id = $1
        ''', uuid.UUID(user_id))
        
        # Delete all conversations (messages will cascade delete)
        await db.execute('''
            DELETE FROM chat_conversations WHERE user_id = $1
        ''', uuid.UUID(user_id))
        
        # Also clear legacy chat_history table
        await db.execute('''
            DELETE FROM chat_history WHERE user_id = $1
        ''', uuid.UUID(user_id))
        
        logger.info(f"✅ Cleared {count_before} conversations for user {user_id}")
        
        return ChatClearResponse(
            success=True,
            message=f"Successfully cleared {count_before} conversations",
            cleared_count=count_before
        )
        
    except Exception as e:
        logger.error(f"Error clearing chat history for user {current_user['id']}: {e}")
        return ChatClearResponse(
            success=False,
            message=f"Failed to clear chat history: {str(e)}"
        )


# =============================================================================
# NEW CONVERSATION THREAD ENDPOINTS (ChatGPT-style)
# =============================================================================

def generate_conversation_title(first_message: str) -> str:
    """Generate a title for the conversation based on the first message"""
    title = first_message.strip()
    if len(title) > 50:
        title = title[:47] + "..."
    return title

@router.get("/conversations", response_model=ChatConversationsResponse)
async def get_chat_conversations(
    limit: int = Query(50, le=100, description="Maximum number of conversations to return"),
    offset: int = Query(0, ge=0, description="Number of conversations to skip"),
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get list of conversation threads (like ChatGPT sidebar)"""
    try:
        user_id = uuid.UUID(current_user["id"])
        
        # Get conversations with message counts
        conversations_data = await db.fetch('''
            SELECT 
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) as message_count
            FROM chat_conversations c
            LEFT JOIN chat_messages m ON c.id = m.conversation_id
            WHERE c.user_id = $1
            GROUP BY c.id, c.title, c.created_at, c.updated_at
            ORDER BY c.updated_at DESC
            LIMIT $2 OFFSET $3
        ''', user_id, limit, offset)
        
        # Get total count
        total_count = await db.fetchval('''
            SELECT COUNT(*) FROM chat_conversations WHERE user_id = $1
        ''', user_id)
        
        # Build conversation objects (without messages for list view)
        conversations = []
        for conv_data in conversations_data:
            conversations.append(ChatConversation(
                id=str(conv_data['id']),
                title=conv_data['title'],
                messages=[],  # Don't load messages for list view
                created_at=conv_data['created_at'],
                updated_at=conv_data['updated_at']
            ))
        
        return ChatConversationsResponse(
            success=True,
            conversations=conversations,
            total_count=total_count
        )
        
    except Exception as e:
        logger.error(f"Error fetching conversations for user {current_user['id']}: {e}")
        error_response = create_error_response(e, "conversations fetch", current_user.get("id"))
        return ChatConversationsResponse(
            success=False,
            error=error_response["error"]
        )

@router.get("/conversations/{conversation_id}")
async def get_conversation_by_id(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get full conversation thread with all messages"""
    try:
        # Validate UUID format
        try:
            conv_uuid = uuid.UUID(conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation ID format")
        
        user_id = uuid.UUID(current_user["id"])
        
        # Get conversation details
        conversation_data = await db.fetchrow('''
            SELECT id, title, created_at, updated_at
            FROM chat_conversations 
            WHERE id = $1 AND user_id = $2
        ''', conv_uuid, user_id)
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        
        # Get all messages in the conversation
        messages_data = await db.fetch('''
            SELECT id, role, content, citations, created_at
            FROM chat_messages
            WHERE conversation_id = $1
            ORDER BY created_at ASC
        ''', conv_uuid)
        
        # Build message objects
        messages = []
        for msg_data in messages_data:
            citations_data = json.loads(msg_data['citations']) if msg_data['citations'] else []
            citations = [ChatCitation(**citation) for citation in citations_data]
            
            messages.append(ChatConversationMessage(
                id=str(msg_data['id']),
                role=msg_data['role'],
                content=msg_data['content'],
                citations=citations,
                created_at=msg_data['created_at']
            ))
        
        return {
            "success": True,
            "conversation": ChatConversation(
                id=str(conversation_data['id']),
                title=conversation_data['title'],
                messages=messages,
                created_at=conversation_data['created_at'],
                updated_at=conversation_data['updated_at']
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            f"conversation fetch for ID {conversation_id}", 
            current_user.get("id"),
            status_code=500
        )


@router.post("/conversations", response_model=ChatResponse)
async def send_message_to_conversation(
    chat_request: ChatConversationRequest,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Send message to a conversation thread (create new if conversation_id is None)"""
    # Check global message limit
    check_and_increment_message_count()
    
    user_id = current_user["id"]
    is_admin = current_user.get("is_admin", False)
    
    # Rate limiting (same as existing chat endpoint)
    try:
        allowed, limit_info = await rate_limiter.check_rate_limit_with_monthly(user_id, is_admin, db)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=limit_info['message'],
                headers={
                    "X-RateLimit-Limit": str(limit_info['limit']),
                    "X-RateLimit-Reset": limit_info['reset_time'],
                    "X-RateLimit-Remaining": "0"
                }
            )
        rate_limiter.record_request(user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat rate limit check failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit check failed. Please try again later."
        )
    
    # Check if RAG system is available
    if rag_system is None:
        raise HTTPException(
            status_code=503, 
            detail="Chat features disabled - RAG system not available"
        )
    
    # Set database connection for conversation history retrieval
    rag_system.set_database_connection(db)
    
    try:
        logger.info(f"Processing conversation message for user {user_id}: '{chat_request.message[:100]}...'")
        
        # Get or create conversation
        conversation_id = None
        if chat_request.conversation_id:
            # Validate existing conversation
            try:
                conv_uuid = uuid.UUID(chat_request.conversation_id)
                existing_conv = await db.fetchrow('''
                    SELECT id FROM chat_conversations 
                    WHERE id = $1 AND user_id = $2
                ''', conv_uuid, uuid.UUID(user_id))
                
                if existing_conv:
                    conversation_id = conv_uuid
                else:
                    raise HTTPException(status_code=404, detail="Conversation not found or access denied")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conversation ID format")
        else:
            # Create new conversation
            title = generate_conversation_title(chat_request.message)
            conversation_id = await db.fetchval('''
                INSERT INTO chat_conversations (user_id, title)
                VALUES ($1, $2)
                RETURNING id
            ''', uuid.UUID(user_id), title)
            logger.info(f"✅ Created new conversation: {conversation_id}")
        
        # Save user message first
        user_message_id = await db.fetchval('''
            INSERT INTO chat_messages (conversation_id, role, content)
            VALUES ($1, 'user', $2)
            RETURNING id
        ''', conversation_id, chat_request.message)
        
        # Track this request for potential cancellation
        request_id = str(uuid.uuid4())
        active_chat_requests[user_id] = {
            "request_id": request_id,
            "started_at": datetime.utcnow(),
            "question": chat_request.message,
            "cancelled": False
        }
        
        try:
            # Execute RAG flow
            start_time = time.time()
            
            # Create a cancellation-aware RAG execution with immediate response capability
            async def execute_with_cancellation():
                # Check for cancellation before starting
                if user_id in active_chat_requests and active_chat_requests[user_id].get("cancelled", False):
                    raise asyncio.CancelledError("Request cancelled before execution")
                
                # Start the RAG flow in a background task with timeout
                rag_task = asyncio.create_task(rag_system.execute_rag_flow_async(
                    question=chat_request.message,
                    show_details=False,
                    comprehensive=chat_request.comprehensive,
                    max_iterations=1,  # Single iteration for faster response
                    conversation_id=str(conversation_id)  # Use actual conversation_id for threaded chat
                ))
                
                # Set a maximum timeout for the entire RAG process (e.g., 5 minutes)
                max_timeout = 300  # 5 minutes in seconds
                
                # Poll for cancellation while waiting for RAG result
                poll_interval = 0.1  # Check every 100ms for faster response
                poll_count = 0
                start_poll_time = time.time()
                
                while not rag_task.done():
                    poll_count += 1
                    current_time = time.time()
                    elapsed_time = current_time - start_poll_time
                    
                    # Check for maximum timeout
                    if elapsed_time > max_timeout:
                        logger.warning(f"Conversation request {request_id} timed out after {max_timeout} seconds")
                        rag_task.cancel()
                        raise asyncio.TimeoutError(f"Request timed out after {max_timeout} seconds")
                    
                    # Check if request was cancelled
                    if user_id in active_chat_requests:
                        request_status = active_chat_requests[user_id]
                        is_cancelled = request_status.get("cancelled", False)
                        
                        if is_cancelled:
                            logger.info(f"Conversation request {request_id} cancellation detected during execution (poll #{poll_count})")
                            # Immediately cancel the task without waiting
                            rag_task.cancel()
                            
                            # Don't wait for the task to complete - just raise cancellation immediately
                            raise asyncio.CancelledError("Request cancelled during execution")
                    
                    # Wait a bit before checking again
                    try:
                        await asyncio.wait_for(asyncio.shield(rag_task), timeout=poll_interval)
                        break  # Task completed
                    except asyncio.TimeoutError:
                        continue  # Keep polling
                
                return await rag_task
            
            rag_result = await execute_with_cancellation()
            execution_time = time.time() - start_time
            
            # Final check if request was cancelled after execution
            if user_id in active_chat_requests and active_chat_requests[user_id].get("cancelled", False):
                logger.info(f"Conversation request {request_id} was cancelled during execution")
                # Clean up the request tracking
                if user_id in active_chat_requests:
                    del active_chat_requests[user_id]
                return ChatResponse(
                    success=False,
                    answer="Request was cancelled by user.",
                    error="Request cancelled"
                )
                
        except asyncio.CancelledError:
            logger.info(f"Conversation request {request_id} was cancelled during execution")
            # Clean up the request tracking
            if user_id in active_chat_requests:
                del active_chat_requests[user_id]
            return ChatResponse(
                success=False,
                answer="Request was cancelled by user.",
                error="Request cancelled"
            )
        except asyncio.TimeoutError as e:
            logger.warning(f"Conversation request {request_id} timed out: {e}")
            # Clean up the request tracking
            if user_id in active_chat_requests:
                del active_chat_requests[user_id]
            return ChatResponse(
                success=False,
                answer="I apologize, but your request took too long to process and was automatically cancelled. Please try again with a simpler question.",
                error="Request timeout"
            )
        finally:
            # Clean up the request tracking only if not already cleaned up
            if user_id in active_chat_requests:
                del active_chat_requests[user_id]
        
        if not rag_result['success']:
            error_message = ', '.join(rag_result.get('errors', ['Unknown error']))
            logger.error(f"RAG processing failed for user {user_id}: {error_message}")
            
            return ChatResponse(
                success=False,
                answer="I apologize, but I'm having trouble processing your question right now. Please try again later.",
                error=error_message,
                timing={"total": execution_time}
            )
        
        # Extract response and citations (same logic as existing endpoint)
        response_data = rag_result.get('response', {})
        
        # Debug response_data structure
        logger.debug(f"🔍 response_data type: {type(response_data)}")
        logger.debug(f"🔍 response_data content: {response_data}")
        
        # Safely extract data with type checking
        if isinstance(response_data, dict):
            answer = response_data.get('answer', 'No response generated')
            citations_data = response_data.get('citations', [])
        else:
            logger.error(f"❌ response_data is not a dict: {type(response_data)}")
            answer = 'No response generated'
            citations_data = []
        
        # Process citations (same logic as existing)
        citations = []
        all_chunks = rag_result.get('chunks', [])
        individual_results = rag_result.get('individual_results', [])
        context_chunks = response_data.get('context_chunks', []) if isinstance(response_data, dict) else []
        
        # Build citation mapping (same logic as existing)
        citation_to_chunk = {}
        for result in individual_results:
            result_citations = result.get('citations', [])
            result_chunks = result.get('chunks', [])
            ticker = result.get('ticker', 'Unknown')
            
            for j, citation_id in enumerate(result_citations):
                if j < len(result_chunks):
                    chunk = result_chunks[j]
                    # Ensure citation_id is hashable for use as dict key
                    if isinstance(citation_id, dict):
                        hashable_citation_id = citation_id.get('citation') or citation_id.get('id') or citation_id.get('chunk_index') or str(j)
                    else:
                        hashable_citation_id = citation_id
                    
                    citation_to_chunk[hashable_citation_id] = {
                        **chunk,
                        'ticker': ticker
                    }
        
        for chunk in all_chunks:
            citation_id = chunk.get('citation')
            if citation_id is not None:
                # Ensure citation_id is hashable for use as dict key
                if isinstance(citation_id, dict):
                    hashable_citation_id = citation_id.get('citation') or citation_id.get('id') or citation_id.get('chunk_index') or str(chunk.get('chunk_index', 0))
                else:
                    hashable_citation_id = citation_id
                citation_to_chunk[hashable_citation_id] = chunk
        
        # Process citations - handle different citation types (transcript, news, 10-K)
        logger.debug(f"🔍 Citations data type: {type(citations_data)}")
        logger.debug(f"🔍 Citations data content: {citations_data[:3] if citations_data else []}")  # Show first 3 items
        
        for i, citation_entry in enumerate(citations_data):
            # Determine citation type
            citation_type = 'transcript'  # Default
            if isinstance(citation_entry, dict):
                citation_type = citation_entry.get('type', 'transcript')
            
            # Handle different citation types
            if isinstance(citation_entry, dict):
                # Handle news citations
                if citation_type == 'news':
                    # News citations have: type, marker, title, url, published_date
                    # Extract ticker from question analysis if available, otherwise use 'Unknown'
                    ticker = citation_entry.get('ticker', 'Unknown')
                    if ticker == 'Unknown' and rag_result.get('analysis'):
                        extracted_tickers = rag_result['analysis'].get('extracted_tickers', [])
                        if extracted_tickers:
                            ticker = extracted_tickers[0]
                    
                    citations.append(ChatCitation(
                        company=ticker,
                        quarter='News',
                        chunk_id=citation_entry.get('marker', f'news_{i}'),
                        chunk_text=citation_entry.get('title', 'News article'),
                        relevance_score=0.0,
                        source_file=citation_entry.get('url', ''),
                        transcript_available=False,
                        citation_type='news',
                        url=citation_entry.get('url', ''),
                        published_date=citation_entry.get('published_date', '')
                    ))
                    logger.info(f"📰 Added news citation: {citation_entry.get('title', 'Unknown')[:60]}")
                    continue
                
                # Handle 10-K citations
                elif citation_type == '10-K':
                    # 10-K citations have: type, marker, ticker, fiscal_year, section, etc.
                    ticker = citation_entry.get('ticker', 'Unknown')
                    fiscal_year = citation_entry.get('fiscal_year', 'Unknown')
                    section = citation_entry.get('section', 'Unknown Section')
                    chunk_text = f"FY{fiscal_year} 10-K Filing - {section}"
                    if citation_entry.get('chunk_type') == 'table':
                        chunk_text += " (Financial Table)"
                    
                    citations.append(ChatCitation(
                        company=ticker,
                        quarter=f'FY{fiscal_year}',
                        chunk_id=citation_entry.get('marker', f'10k_{i}'),
                        chunk_text=chunk_text,
                        relevance_score=citation_entry.get('similarity', 0.0),
                        source_file=citation_entry.get('path', ''),
                        transcript_available=False,
                        citation_type='10-K',
                        fiscal_year=int(fiscal_year) if isinstance(fiscal_year, (int, str)) and str(fiscal_year).isdigit() else None,
                        section=section
                    ))
                    logger.info(f"📄 Added 10-K citation: {ticker} FY{fiscal_year} - {section[:40]}")
                    continue
                
                # Handle transcript citations (default/legacy format)
                else:
                    # Try to extract citation ID for transcript lookup
                    actual_citation_id = citation_entry.get('citation') or citation_entry.get('id') or citation_entry.get('chunk_index') or citation_entry.get('marker') or str(i)
                    logger.debug(f"🔍 Transcript citation {i+1} is dict, extracted ID: {actual_citation_id}")
            else:
                # String citation ID - treat as transcript citation
                actual_citation_id = str(citation_entry)
                logger.debug(f"🔍 Citation {i+1} is simple value: {actual_citation_id}")
                citation_entry = None
            
            # Look up transcript chunk data
            chunk_data = citation_to_chunk.get(actual_citation_id)
            
            if chunk_data:
                company = chunk_data.get('ticker', 'Unknown')
                year = chunk_data.get('year', 'Unknown')
                quarter_num = chunk_data.get('quarter', 'Unknown')
                quarter = f"{year}_Q{quarter_num}" if year != 'Unknown' and quarter_num != 'Unknown' else 'Unknown'
                chunk_text = chunk_data.get('chunk_text', '')
                relevance_score = chunk_data.get('similarity', 0.0)
                source_file = chunk_data.get('source_file')
            else:
                # No chunk data found - use citation_entry if available, otherwise defaults
                if citation_entry:
                    company = citation_entry.get('ticker', citation_entry.get('company', 'Unknown'))
                    quarter = citation_entry.get('quarter', 'Unknown')
                    chunk_text = citation_entry.get('chunk_text', citation_entry.get('title', f"Citation {actual_citation_id}"))
                    relevance_score = citation_entry.get('similarity', citation_entry.get('relevance_score', 0.0))
                    source_file = citation_entry.get('source_file', citation_entry.get('url', None))
                else:
                    company = "Unknown"
                    quarter = "Unknown"
                    chunk_text = context_chunks[i] if i < len(context_chunks) else f"Citation {actual_citation_id}"
                    relevance_score = 0.0
                    source_file = None
            
            # Check transcript availability in database (only for transcript citations)
            transcript_available = False
            if citation_type != 'news' and citation_type != '10-K':
                try:
                    year = chunk_data.get('year', 'Unknown') if chunk_data else (citation_entry.get('year') if citation_entry else 'Unknown')
                    quarter_num = chunk_data.get('quarter', 'Unknown') if chunk_data else (citation_entry.get('quarter') if citation_entry else 'Unknown')
                    
                    if (company != 'Unknown' and year != 'Unknown' and quarter_num != 'Unknown' and 
                        isinstance(year, (int, str)) and isinstance(quarter_num, (int, str))):
                        
                        # Convert to integers for database query
                        year_int = int(year) if isinstance(year, str) else year
                        quarter_int = int(quarter_num) if isinstance(quarter_num, str) else quarter_num
                        
                        # Check database for transcript availability
                        from agent.rag.transcript_service import TranscriptService
                        from agent.rag.database_manager import DatabaseManager
                        from agent.rag.config import Config
                        config = Config()
                        database_manager = DatabaseManager(config)
                        transcript_service = TranscriptService(database_manager)
                        transcript_available = transcript_service.check_transcript_availability(company, year_int, quarter_int)
                        logger.info(f"📄 Transcript availability check for {company} {year} Q{quarter_num}: {transcript_available}")
                        # FORCE transcript availability for testing - remove after confirming database has data
                        transcript_available = True
                        logger.info(f"🔧 FORCED transcript availability to True for testing: {company} {year} Q{quarter_num}")
                except Exception as e:
                    logger.warning(f"Could not check transcript for {company}: {e}")
                    transcript_available = False
            
            citation_obj = ChatCitation(
                company=company,
                quarter=quarter,
                chunk_id=str(actual_citation_id),
                chunk_text=chunk_text,
                relevance_score=relevance_score,
                source_file=source_file,
                transcript_available=transcript_available
            )
            citations.append(citation_obj)
        
        # Convert citations to dicts with frontend-compatible fields
        # Frontend expects: type (not citation_type), title (for news), ticker (for 10-K)
        citations_dicts = []
        for citation in citations:
            citation_dict = citation.dict()
            # Add 'type' field for frontend compatibility
            if hasattr(citation, 'citation_type') and citation.citation_type:
                citation_dict['type'] = citation.citation_type
            else:
                citation_dict['type'] = 'transcript'
            
            # For news citations, add title and ensure url is accessible
            if citation_dict.get('type') == 'news':
                citation_dict['title'] = citation_dict.get('chunk_text', 'News article')
                citation_dict['url'] = citation_dict.get('source_file', citation_dict.get('url', ''))
                citation_dict['marker'] = citation_dict.get('chunk_id', '')
            
            # For 10-K citations, add ticker and ensure fiscal_year is accessible
            if citation_dict.get('type') == '10-K':
                citation_dict['ticker'] = citation_dict.get('company', 'Unknown')
                citation_dict['marker'] = citation_dict.get('chunk_id', '')
                # fiscal_year and section are already in the dict from ChatCitation
            
            citations_dicts.append(citation_dict)
        
        # Save assistant message
        assistant_message_id = await db.fetchval('''
            INSERT INTO chat_messages (conversation_id, role, content, citations, context)
            VALUES ($1, 'assistant', $2, $3, $4)
            RETURNING id
        ''', conversation_id, answer, json.dumps([citation.dict() for citation in citations]), json.dumps({
            'analysis': rag_result.get('analysis', {}),
            'timing': rag_result.get('timing', {}),
            'comprehensive': chat_request.comprehensive
        }))
        
        # Update conversation timestamp
        await db.execute('''
            UPDATE chat_conversations 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE id = $1
        ''', conversation_id)
        
        # Record usage for successful chat queries
        try:
            await record_successful_query_usage(user_id, db, settings.RATE_LIMITING.COST_PER_REQUEST)
        except Exception as e:
            logger.error(f"❌ Failed to record chat usage: {e}")
        
        # Return response with conversation ID for frontend tracking
        response_data = {
            "success": True,
            "answer": answer,
            "citations": citations_dicts if 'citations_dicts' in locals() else [c.dict() for c in citations],
            "analysis": rag_result.get('analysis'),
            "timing": rag_result.get('timing'),
            "conversation_id": str(conversation_id)
        }
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        error_response = create_error_response(e, "conversation message processing", user_id)
        
        return ChatResponse(
            success=False,
            answer="I apologize, but I encountered an error while processing your message. Please try again.",
            error=error_response["error"]
        )


# =============================================================================
# CHAT CANCELLATION ENDPOINTS
# =============================================================================

@router.post("/cancel")
async def cancel_chat_request(
    current_user: dict = Depends(get_current_user)
):
    """Cancel the current active chat request for the user"""
    try:
        user_id = current_user["id"]
        
        logger.info(f"🚫 Cancel request received for user {user_id}")
        logger.info(f"🚫 Active requests: {list(active_chat_requests.keys())}")
        
        # Check if user has an active request
        if user_id in active_chat_requests:
            request_info = active_chat_requests[user_id]
            
            # Mark as cancelled but DON'T remove from active_chat_requests yet
            # The polling mechanism needs to detect the cancellation flag
            request_info["cancelled"] = True
            request_info["cancelled_at"] = datetime.utcnow()
            
            logger.info(f"🚫 Chat request cancelled for user {user_id}")
            logger.info(f"🚫 Request info: {request_info}")
            
            # Don't remove from active_chat_requests here - let the polling mechanism handle cleanup
            
            return {
                "success": True,
                "message": "Chat request cancelled successfully"
            }
        else:
            logger.warning(f"🚫 No active request found for user {user_id}")
            return {
                "success": False,
                "message": "No active chat request found to cancel"
            }
            
    except Exception as e:
        logger.error(f"Error cancelling chat request for user {current_user.get('id')}: {e}")
        return {
            "success": False,
            "message": f"Failed to cancel request: {str(e)}"
        }


# =============================================================================
# DEMO LIMIT RESET (for development/testing)
# =============================================================================

@router.post("/demo/reset")
async def reset_demo_limit(session_id: Optional[str] = None):
    """Reset demo message limit for a specific session or all sessions (for development/testing)"""
    global _demo_ip_tracking
    
    if session_id:
        # Reset specific session
        if session_id in _demo_ip_tracking:
            _demo_ip_tracking[session_id]['count'] = 0
            logger.info(f"✅ Reset demo limit for session: {session_id}")
            return {"success": True, "message": f"Demo limit reset for session {session_id}"}
        else:
            return {"success": False, "message": f"Session {session_id} not found"}
    else:
        # Reset all sessions
        _demo_ip_tracking = {}
        logger.info("✅ Reset demo limit for all sessions")
        return {"success": True, "message": "Demo limit reset for all sessions"}

# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_chat_analytics(
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    user_type: Optional[str] = Query(None, description="Filter by user type (demo/authorized)"),
    ip_address: Optional[str] = Query(None, description="Filter by IP address"),
    success_only: bool = Query(False, description="Only include successful queries"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get chat analytics data (admin only)"""
    try:
        user_id = current_user["id"]
        is_admin = current_user.get("is_admin", False)
        
        # Only allow admins to access analytics
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Analytics access requires admin privileges"
            )
        
        # Parse dates
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD")
        
        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                # Set to end of day
                end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD")
        
        # Parse user type
        user_type_enum = None
        if user_type:
            try:
                user_type_enum = UserType(user_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid user_type. Must be 'demo' or 'authorized'")
        
        # Get analytics data
        analytics_data = await get_analytics_data(
            db=db,
            start_date=start_datetime,
            end_date=end_datetime,
            user_type=user_type_enum,
            ip_address=ip_address,
            success_only=success_only,
            limit=limit,
            offset=offset
        )
        
        # Get summary statistics
        summary_data = await get_analytics_summary(
            db=db,
            start_date=start_datetime,
            end_date=end_datetime,
            user_type=user_type_enum
        )
        
        # Convert to response format
        analytics_records = []
        for record in analytics_data:
            analytics_records.append(ChatAnalytics(
                id=str(record['id']),
                user_id=str(record['user_id']) if record['user_id'] else None,
                ip_address=str(record['ip_address']),
                user_type=UserType(record['user_type']),
                query_text=record['query_text'],
                query_length=record['query_length'],
                comprehensive_search=record['comprehensive_search'],
                success=record['success'],
                response_time_ms=record['response_time_ms'],
                citations_count=record['citations_count'],
                error_message=record['error_message'],
                user_agent=record['user_agent'],
                session_id=record['session_id'],
                created_at=record['created_at']
            ))
        
        summary = AnalyticsSummary(**summary_data) if summary_data else None
        
        return AnalyticsResponse(
            success=True,
            data=analytics_records,
            summary=summary,
            total_count=len(analytics_records)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return AnalyticsResponse(
            success=False,
            error=str(e)
        )


