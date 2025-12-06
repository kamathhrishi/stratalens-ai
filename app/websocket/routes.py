"""
WebSocket Routes

WebSocket endpoint handlers for real-time communication.
"""

import asyncio
import json
import logging
import time
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect, HTTPException

from app.auth.auth_utils import authenticate_user_by_id
from app.lifespan import get_db_pool, get_websocket_manager
from config import settings
from app.websocket.types import MessageType, WebSocketMessage

logger = logging.getLogger(__name__)


async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time financial analysis communication
    
    Usage: Connect to ws://localhost:8000/ws/{user_id}
    """
    session_id = None
    
    # Check if analyzer is available
    try:
        from agent.screener import FinancialDataAnalyzer
        ANALYZER_AVAILABLE = True
    except ImportError:
        ANALYZER_AVAILABLE = False
    
    if not ANALYZER_AVAILABLE:
        await websocket.close(code=1000, reason="Financial analyzer not available")
        return
    
    # Get WebSocket manager
    websocket_manager = get_websocket_manager()
    if not websocket_manager:
        logger.error("❌ WebSocket manager not initialized")
        await websocket.close(code=1000, reason="WebSocket manager not available")
        return
    
    # Get database pool for authentication
    db_pool = get_db_pool()
    if not db_pool:
        logger.error("❌ Database pool not available for WebSocket authentication")
        await websocket.close(code=1000, reason="Database not available")
        return
    
    # 🔐 AUTHENTICATION CHECK - Use centralized authentication
    try:
        current_user = await authenticate_user_by_id(user_id, db_pool)
    except HTTPException as e:
        logger.error(f"❌ WebSocket authentication failed: {e.detail}")
        await websocket.close(code=1008, reason=e.detail)
        return
    except Exception as e:
        logger.error(f"❌ WebSocket authentication error: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return
    
    try:
        # Generate unique session ID
        session_id = f"{user_id}_{int(time.time())}"
        
        # Get session manager from websocket manager
        session_manager = websocket_manager.session_manager
        
        # Connect WebSocket with better error handling
        try:
            await websocket_manager.connect(websocket, session_id, user_id)
        except Exception as e:
            logger.error(f"❌ Failed to establish WebSocket connection for user {user_id}: {e}")
            try:
                await websocket.close(code=1000, reason="Connection failed")
            except:
                pass
            return
        
        # Create session
        await session_manager.create_session(
            user_id=user_id, 
            metadata={"connection_time": datetime.utcnow().isoformat()}
        )
        
        # Small delay to ensure connection is fully established
        await asyncio.sleep(settings.WEBSOCKET.CONNECTION_DELAY_SECONDS)
        
        # Send welcome message using new WebSocketMessage structure
        welcome_message = WebSocketMessage(
            type=MessageType.CONNECTED,
            session_id=session_id,
            data={
                "status": "connected",
                "message": "WebSocket connected successfully"
            },
            user_id=user_id
        )
        success = await websocket_manager.send_message(session_id, welcome_message)
        if not success:
            logger.warning(f"⚠️ Failed to send welcome message for session {session_id}")
        
        logger.info(f"🎯 WebSocket session {session_id} established for user {user_id}")
        
        # Listen for messages
        while True:
            try:
                # Receive message from client
                message_data = await websocket.receive_text()
                message = json.loads(message_data)
                
                # Handle message using the new WebSocket handlers
                await websocket_manager.handle_message(session_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                logger.error(f"❌ Error handling WebSocket message for session {session_id}: {e}")
                # Send error message to client
                error_message = WebSocketMessage(
                    type=MessageType.ERROR,
                    session_id=session_id,
                    data={
                        "error": "Message processing failed",
                        "details": str(e)
                    }
                )
                await websocket_manager.send_message(session_id, error_message)
    
    except Exception as e:
        logger.error(f"❌ WebSocket endpoint error for user {user_id}: {e}")
    finally:
        # Clean up session
        if session_id:
            await websocket_manager.disconnect(session_id)
            if session_manager:
                await session_manager.delete_session(session_id)
            logger.info(f"🧹 WebSocket session {session_id} cleaned up")

