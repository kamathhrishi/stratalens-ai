"""
WebSocket Message Handlers

Centralized handlers for different types of WebSocket messages.
"""

import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from .types import WebSocketMessage, MessageType
from .manager import WebSocketManager
from .session_manager import SessionManager


class WebSocketHandlers:
    """Handles different types of WebSocket messages"""
    
    def __init__(self, websocket_manager: WebSocketManager, session_manager: SessionManager):
        self.websocket_manager = websocket_manager
        self.session_manager = session_manager
        
        # Register default handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.websocket_manager.register_handler(MessageType.CHAT_MESSAGE, self.handle_chat_message)
        self.websocket_manager.register_handler(MessageType.HEARTBEAT, self.handle_heartbeat)
        self.websocket_manager.register_handler(MessageType.SESSION_UPDATE, self.handle_session_update)
    
    async def handle_chat_message(self, session_id: str, message_data: Dict[str, Any]):
        """Handle chat message from client"""
        try:
            # Extract message content
            content = message_data.get("data", {}).get("message", "")
            user_id = message_data.get("user_id")
            
            # Update session activity
            await self.session_manager.update_session(session_id, last_activity=datetime.utcnow())
            
            # Echo back confirmation
            confirmation = WebSocketMessage(
                type=MessageType.CHAT_MESSAGE,
                session_id=session_id,
                data={
                    "status": "received",
                    "message": content,
                    "timestamp": datetime.utcnow().isoformat()
                },
                user_id=user_id
            )
            
            await self.websocket_manager.send_message(session_id, confirmation)
            
        except Exception as e:
            print(f"Error handling chat message for session {session_id}: {e}")
            
            # Send error response
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                session_id=session_id,
                data={
                    "error": "Failed to process chat message",
                    "details": str(e)
                }
            )
            
            await self.websocket_manager.send_message(session_id, error_message)
    
    async def handle_heartbeat(self, session_id: str, message_data: Dict[str, Any]):
        """Handle heartbeat/ping message"""
        try:
            # Update session activity
            await self.session_manager.update_session(session_id, last_activity=datetime.utcnow())
            
            # Send pong response
            pong_message = WebSocketMessage(
                type=MessageType.HEARTBEAT,
                session_id=session_id,
                data={
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await self.websocket_manager.send_message(session_id, pong_message)
            
        except Exception as e:
            print(f"Error handling heartbeat for session {session_id}: {e}")
    
    async def handle_session_update(self, session_id: str, message_data: Dict[str, Any]):
        """Handle session metadata updates"""
        try:
            metadata = message_data.get("data", {}).get("metadata", {})
            
            # Update session metadata
            session = await self.session_manager.get_session(session_id)
            if session:
                updated_metadata = {**session.metadata, **metadata}
                await self.session_manager.update_session(
                    session_id,
                    metadata=updated_metadata,
                    last_activity=datetime.utcnow()
                )
                
                # Send confirmation
                confirmation = WebSocketMessage(
                    type=MessageType.SESSION_UPDATE,
                    session_id=session_id,
                    data={
                        "status": "updated",
                        "metadata": updated_metadata
                    }
                )
                
                await self.websocket_manager.send_message(session_id, confirmation)
            
        except Exception as e:
            print(f"Error handling session update for session {session_id}: {e}")
            
            # Send error response
            error_message = WebSocketMessage(
                type=MessageType.ERROR,
                session_id=session_id,
                data={
                    "error": "Failed to update session",
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
    
    def register_custom_handler(self, message_type: MessageType, handler: Callable):
        """Register a custom message handler"""
        self.websocket_manager.register_handler(message_type, handler)
