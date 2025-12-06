"""
WebSocket Type Definitions

Centralized type definitions for WebSocket-related data structures.
"""

from typing import Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class ConnectionState(Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


class MessageType(Enum):
    """WebSocket message types"""
    CHAT_MESSAGE = "chat_message"
    CHAT_STREAM = "chat_stream"
    CHAT_PROGRESS = "chat_progress"
    CHAT_RESULT = "chat_result"
    CHAT_ERROR = "chat_error"
    CHAT_RECEIVED = "chat_received"
    QUERY = "query"
    QUERY_RECEIVED = "query_received"
    REASONING = "reasoning"
    RESULT = "result"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    PING = "ping"
    PONG = "pong"
    SESSION_UPDATE = "session_update"
    STATUS = "status"
    GET_STATUS = "get_status"
    GET_RESULT = "get_result"
    CONNECTED = "connected"


@dataclass
class WebSocketMessage:
    """Standard WebSocket message structure"""
    type: MessageType
    session_id: str
    data: Dict[str, Any]
    timestamp: Optional[datetime] = None
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.type.value,
            "session_id": self.session_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id
        }


@dataclass
class SessionData:
    """Session data structure"""
    session_id: str
    user_id: Optional[str]
    connected_at: datetime
    last_activity: datetime
    connection_state: ConnectionState
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "connected_at": self.connected_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "connection_state": self.connection_state.value,
            "metadata": self.metadata
        }


@dataclass
class BackgroundTaskConfig:
    """Configuration for background tasks"""
    task_id: str
    task_type: str
    timeout: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
