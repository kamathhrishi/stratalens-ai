"""
Session Manager for WebSocket Connections

Handles session persistence using Redis with memory fallback.
Extracted from fastapi_server.py for better organization.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import asyncio

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .types import SessionData, ConnectionState


class SessionManager:
    """Manages WebSocket sessions with Redis persistence and memory fallback"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self.memory_sessions: Dict[str, SessionData] = {}
        self.session_timeout = 3600  # 1 hour default
        
    async def create_session(self, user_id: Optional[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        session_data = SessionData(
            session_id=session_id,
            user_id=user_id,
            connected_at=now,
            last_activity=now,
            connection_state=ConnectionState.CONNECTED,
            metadata=metadata or {}
        )
        
        # Store in Redis if available, otherwise use memory
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_timeout,
                    json.dumps(session_data.to_dict())
                )
            except Exception as e:
                print(f"Redis error, falling back to memory: {e}")
                self.memory_sessions[session_id] = session_data
        else:
            self.memory_sessions[session_id] = session_data
            
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by session ID"""
        if self.redis_client:
            try:
                data = await self.redis_client.get(f"session:{session_id}")
                if data:
                    session_dict = json.loads(data)
                    return SessionData(
                        session_id=session_dict["session_id"],
                        user_id=session_dict["user_id"],
                        connected_at=datetime.fromisoformat(session_dict["connected_at"]),
                        last_activity=datetime.fromisoformat(session_dict["last_activity"]),
                        connection_state=ConnectionState(session_dict["connection_state"]),
                        metadata=session_dict["metadata"]
                    )
            except Exception as e:
                print(f"Redis error, checking memory: {e}")
        
        # Fallback to memory
        return self.memory_sessions.get(session_id)
    
    async def update_session(self, session_id: str, **updates) -> bool:
        """Update session data"""
        session = await self.get_session(session_id)
        if not session:
            return False
            
        # Apply updates
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)
        
        session.last_activity = datetime.utcnow()
        
        # Store updated session
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_timeout,
                    json.dumps(session.to_dict())
                )
                return True
            except Exception as e:
                print(f"Redis error, updating memory: {e}")
                self.memory_sessions[session_id] = session
                return True
        else:
            self.memory_sessions[session_id] = session
            return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if self.redis_client:
            try:
                await self.redis_client.delete(f"session:{session_id}")
            except Exception as e:
                print(f"Redis error: {e}")
        
        # Also remove from memory
        if session_id in self.memory_sessions:
            del self.memory_sessions[session_id]
            
        return True
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions from memory"""
        now = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.memory_sessions.items():
            if now - session.last_activity > timedelta(seconds=self.session_timeout):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.memory_sessions[session_id]
            
        return len(expired_sessions)
    
    async def get_active_sessions(self) -> Dict[str, SessionData]:
        """Get all active sessions"""
        active_sessions = {}
        
        # Get from Redis if available
        if self.redis_client:
            try:
                keys = await self.redis_client.keys("session:*")
                for key in keys:
                    session_id = key.decode().replace("session:", "")
                    session = await self.get_session(session_id)
                    if session:
                        active_sessions[session_id] = session
            except Exception as e:
                print(f"Redis error, using memory sessions: {e}")
        
        # Merge with memory sessions
        active_sessions.update(self.memory_sessions)
        
        return active_sessions
