"""
WebSocket Connection Manager

Centralized management of WebSocket connections, message routing, and connection state.
"""

import json
import asyncio
from typing import Dict, Set, Optional, Any, Callable
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

from .types import WebSocketMessage, MessageType, ConnectionState
from .session_manager import SessionManager


class WebSocketManager:
    """Manages WebSocket connections and message routing"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[str, ConnectionState] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None) -> bool:
        """Accept WebSocket connection and register session"""
        try:
            await websocket.accept()
            self.active_connections[session_id] = websocket
            self.connection_states[session_id] = ConnectionState.CONNECTED
            
            # Update session in session manager
            await self.session_manager.update_session(
                session_id,
                connection_state=ConnectionState.CONNECTED,
                user_id=user_id
            )
            
            return True
        except Exception as e:
            print(f"Error connecting WebSocket for session {session_id}: {e}")
            return False
    
    async def disconnect(self, session_id: str) -> bool:
        """Disconnect WebSocket and clean up session"""
        try:
            # Remove from active connections
            if session_id in self.active_connections:
                del self.active_connections[session_id]
            
            # Update connection state
            self.connection_states[session_id] = ConnectionState.DISCONNECTED
            
            # Update session in session manager
            await self.session_manager.update_session(
                session_id,
                connection_state=ConnectionState.DISCONNECTED
            )
            
            return True
        except Exception as e:
            print(f"Error disconnecting WebSocket for session {session_id}: {e}")
            return False
    
    async def send_message(self, session_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific WebSocket connection"""
        if session_id not in self.active_connections:
            return False
            
        websocket = self.active_connections[session_id]
        
        try:
            await websocket.send_text(json.dumps(message.to_dict()))
            return True
        except WebSocketDisconnect:
            await self.disconnect(session_id)
            return False
        except Exception as e:
            print(f"Error sending message to session {session_id}: {e}")
            return False
    
    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> int:
        """Send message to all connections for a specific user"""
        sent_count = 0
        active_sessions = await self.session_manager.get_active_sessions()
        
        for session_id, session in active_sessions.items():
            if session.user_id == user_id and session_id in self.active_connections:
                if await self.send_message(session_id, message):
                    sent_count += 1
                    
        return sent_count
    
    async def broadcast(self, message: WebSocketMessage, exclude_sessions: Set[str] = None) -> int:
        """Broadcast message to all active connections"""
        if exclude_sessions is None:
            exclude_sessions = set()
            
        sent_count = 0
        
        for session_id in list(self.active_connections.keys()):
            if session_id not in exclude_sessions:
                if await self.send_message(session_id, message):
                    sent_count += 1
                    
        return sent_count
    
    def is_connection_ready(self, session_id: str) -> bool:
        """Check if WebSocket connection is ready for sending messages"""
        return (
            session_id in self.active_connections and
            self.connection_states.get(session_id) == ConnectionState.CONNECTED
        )
    
    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_active_sessions(self) -> Set[str]:
        """Get set of active session IDs"""
        return set(self.active_connections.keys())
    
    async def handle_message(self, session_id: str, message_data: Dict[str, Any]) -> bool:
        """Handle incoming WebSocket message"""
        try:
            message_type = MessageType(message_data.get("type"))
            
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                await handler(session_id, message_data)
                return True
            else:
                print(f"No handler registered for message type: {message_type}")
                return False
                
        except ValueError:
            print(f"Invalid message type: {message_data.get('type')}")
            return False
        except Exception as e:
            print(f"Error handling message for session {session_id}: {e}")
            return False
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def cleanup_stale_connections(self) -> int:
        """Clean up stale connections that are no longer active"""
        stale_sessions = []
        
        for session_id, websocket in self.active_connections.items():
            try:
                # Try to ping the connection
                await websocket.ping()
            except:
                stale_sessions.append(session_id)
        
        # Remove stale connections
        for session_id in stale_sessions:
            await self.disconnect(session_id)
            
        return len(stale_sessions)
    
    async def get_connection_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get connection information for a session"""
        if session_id not in self.active_connections:
            return None
            
        session = await self.session_manager.get_session(session_id)
        if not session:
            return None
            
        return {
            "session_id": session_id,
            "user_id": session.user_id,
            "connected_at": session.connected_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "connection_state": self.connection_states.get(session_id, ConnectionState.DISCONNECTED).value,
            "metadata": session.metadata
        }
