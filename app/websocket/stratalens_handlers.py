"""
Custom WebSocket Handlers for StrataLens AI

Handlers for chat and query functionality specific to the StrataLens application.
"""

import json
import asyncio
import time
from datetime import datetime, date
from typing import Dict, Any, Optional
import uuid

from .types import WebSocketMessage, MessageType
from .manager import WebSocketManager
from .session_manager import SessionManager


class StrataLensWebSocketHandlers:
    """Custom handlers for StrataLens-specific WebSocket functionality"""
    
    def __init__(self, websocket_manager: WebSocketManager, session_manager: SessionManager):
        self.websocket_manager = websocket_manager
        self.session_manager = session_manager
        
        # Register custom handlers
        self._register_custom_handlers()
    
    def _register_custom_handlers(self):
        """Register custom message handlers"""
        self.websocket_manager.register_handler(MessageType.PING, self.handle_ping)
        self.websocket_manager.register_handler(MessageType.QUERY, self.handle_query)
        self.websocket_manager.register_handler(MessageType.CHAT_MESSAGE, self.handle_chat)
        self.websocket_manager.register_handler(MessageType.GET_STATUS, self.handle_get_status)
        self.websocket_manager.register_handler(MessageType.GET_RESULT, self.handle_get_result)
    
    async def handle_ping(self, session_id: str, message_data: Dict[str, Any]):
        """Handle ping/keepalive message"""
        try:
            # Update session activity
            await self.session_manager.update_session(session_id, last_activity=datetime.utcnow())
            
            # Send pong response
            pong_message = WebSocketMessage(
                type=MessageType.PONG,
                session_id=session_id,
                data={
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, pong_message)
            
        except Exception as e:
            print(f"Error handling ping for session {session_id}: {e}")
    
    async def handle_query(self, session_id: str, message_data: Dict[str, Any]):
        """Handle financial analysis query"""
        try:
            question = message_data.get("data", {}).get("question", "")
            page = message_data.get("data", {}).get("page", 1)
            page_size = message_data.get("data", {}).get("page_size")
            
            if not question:
                error_message = WebSocketMessage(
                    type=MessageType.ERROR,
                    session_id=session_id,
                    data={
                        "error": "Question is required",
                        "message": "Question is required"
                    }
                )
                await self.websocket_manager.send_message(session_id, error_message)
                return
            
            # Send acknowledgment
            ack_message = WebSocketMessage(
                type=MessageType.QUERY_RECEIVED,
                session_id=session_id,
                data={
                    "question": question,
                    "message": "Query received, starting analysis...",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self.websocket_manager.send_message(session_id, ack_message)
            
            # Start background analysis
            # Note: This would need to be integrated with the existing analyzer
            # For now, we'll just send a placeholder response
            result_message = WebSocketMessage(
                type=MessageType.RESULT,
                session_id=session_id,
                data={
                    "result": {"message": "Query processing not yet implemented in new structure"},
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self.websocket_manager.send_message(session_id, result_message)
            
        except Exception as e:
            print(f"Error handling query for session {session_id}: {e}")
            
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                session_id=session_id,
                data={
                    "error": "Failed to process query",
                    "details": str(e)
                }
            )
            await self.websocket_manager.send_message(session_id, error_message)
    
    async def handle_chat(self, session_id: str, message_data: Dict[str, Any]):
        """Handle chat message"""
        try:
            chat_question = message_data.get("data", {}).get("message", "")
            comprehensive = message_data.get("data", {}).get("comprehensive", True)
            
            if not chat_question:
                error_message = WebSocketMessage(
                    type=MessageType.CHAT_ERROR,
                    session_id=session_id,
                    data={
                        "error": "Chat message is required",
                        "message": "Chat message is required"
                    }
                )
                await self.websocket_manager.send_message(session_id, error_message)
                return
            
            # Check character limit
            if len(chat_question) > 4000:
                error_message = WebSocketMessage(
                    type=MessageType.CHAT_ERROR,
                    session_id=session_id,
                    data={
                        "error": f"Message too long! Please keep it under 4000 characters. Current length: {len(chat_question)} characters.",
                        "message": f"Message too long! Please keep it under 4000 characters. Current length: {len(chat_question)} characters."
                    }
                )
                await self.websocket_manager.send_message(session_id, error_message)
                return
            
            # Send acknowledgment
            ack_message = WebSocketMessage(
                type=MessageType.CHAT_RECEIVED,
                session_id=session_id,
                data={
                    "message": "Chat message received, processing...",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self.websocket_manager.send_message(session_id, ack_message)
            
            # Start background chat processing
            # Note: This would need to be integrated with the existing RAG system
            # For now, we'll just send a placeholder response
            result_message = WebSocketMessage(
                type=MessageType.CHAT_RESULT,
                session_id=session_id,
                data={
                    "result": {"message": "Chat processing not yet implemented in new structure"},
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self.websocket_manager.send_message(session_id, result_message)
            
        except Exception as e:
            print(f"Error handling chat for session {session_id}: {e}")
            
            error_message = WebSocketMessage(
                type=MessageType.CHAT_ERROR,
                session_id=session_id,
                data={
                    "error": "Failed to process chat message",
                    "details": str(e)
                }
            )
            await self.websocket_manager.send_message(session_id, error_message)
    
    async def handle_get_status(self, session_id: str, message_data: Dict[str, Any]):
        """Handle get status request"""
        try:
            session = await self.session_manager.get_session(session_id)
            if session:
                status_message = WebSocketMessage(
                    type=MessageType.STATUS,
                    session_id=session_id,
                    data={
                        "status": session.status if hasattr(session, 'status') else "unknown",
                        "progress": session.metadata.get("progress", {}),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                await self.websocket_manager.send_message(session_id, status_message)
            else:
                error_message = WebSocketMessage(
                    type=MessageType.ERROR,
                    session_id=session_id,
                    data={
                        "error": "Session not found",
                        "message": "Session not found"
                    }
                )
                await self.websocket_manager.send_message(session_id, error_message)
                
        except Exception as e:
            print(f"Error handling get_status for session {session_id}: {e}")
            
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                session_id=session_id,
                data={
                    "error": "Failed to get status",
                    "details": str(e)
                }
            )
            await self.websocket_manager.send_message(session_id, error_message)
    
    async def handle_get_result(self, session_id: str, message_data: Dict[str, Any]):
        """Handle get result request"""
        try:
            # Note: This would need to be integrated with the background task manager
            # For now, we'll just send a placeholder response
            result_message = WebSocketMessage(
                type=MessageType.RESULT,
                session_id=session_id,
                data={
                    "result": {"message": "Result retrieval not yet implemented in new structure"},
                    "success": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self.websocket_manager.send_message(session_id, result_message)
            
        except Exception as e:
            print(f"Error handling get_result for session {session_id}: {e}")
            
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                session_id=session_id,
                data={
                    "error": "Failed to get result",
                    "details": str(e)
                }
            )
            await self.websocket_manager.send_message(session_id, error_message)
    
    async def send_reasoning_event(self, session_id: str, reasoning_event: Dict[str, Any]):
        """Send reasoning event to client"""
        try:
            message = WebSocketMessage(
                type=MessageType.REASONING,
                session_id=session_id,
                data=reasoning_event,
                timestamp=datetime.utcnow()
            )
            
            await self.websocket_manager.send_message(session_id, message)
            
        except Exception as e:
            print(f"Error sending reasoning event for session {session_id}: {e}")
    
    async def send_chat_stream(self, session_id: str, content: str):
        """Send streaming chat content to client"""
        try:
            message = WebSocketMessage(
                type=MessageType.CHAT_STREAM,
                session_id=session_id,
                data={
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, message)
            
        except Exception as e:
            print(f"Error sending chat stream for session {session_id}: {e}")
    
    async def send_chat_progress(self, session_id: str, progress_data: Dict[str, Any]):
        """Send chat progress update to client"""
        try:
            message = WebSocketMessage(
                type=MessageType.CHAT_PROGRESS,
                session_id=session_id,
                data={
                    "stage": progress_data.get("stage", "unknown"),
                    "progress": progress_data.get("progress", 0.0),
                    "message": progress_data.get("message", ""),
                    "details": progress_data.get("details", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, message)
            
        except Exception as e:
            print(f"Error sending chat progress for session {session_id}: {e}")
    
    async def send_result(self, session_id: str, result: Dict[str, Any], success: bool = True):
        """Send final result to client"""
        try:
            message = WebSocketMessage(
                type=MessageType.RESULT,
                session_id=session_id,
                data={
                    "result": result,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, message)
            
        except Exception as e:
            print(f"Error sending result for session {session_id}: {e}")
    
    async def send_chat_result(self, session_id: str, result: Dict[str, Any], success: bool = True):
        """Send final chat result to client"""
        try:
            message = WebSocketMessage(
                type=MessageType.CHAT_RESULT,
                session_id=session_id,
                data={
                    "result": result,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, message)
            
        except Exception as e:
            print(f"Error sending chat result for session {session_id}: {e}")
    
    async def send_error(self, session_id: str, error_message: str, details: str = None):
        """Send error message to client"""
        try:
            message = WebSocketMessage(
                type=MessageType.ERROR,
                session_id=session_id,
                data={
                    "error": error_message,
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, message)
            
        except Exception as e:
            print(f"Error sending error message for session {session_id}: {e}")
    
    async def send_chat_error(self, session_id: str, error_message: str, details: str = None):
        """Send chat error message to client"""
        try:
            message = WebSocketMessage(
                type=MessageType.CHAT_ERROR,
                session_id=session_id,
                data={
                    "error": error_message,
                    "details": details,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, message)
            
        except Exception as e:
            print(f"Error sending chat error message for session {session_id}: {e}")
