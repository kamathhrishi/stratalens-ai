from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class WSMessage(BaseModel):
    type: str
    data: Dict[str, Any] = {}
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


class WSQueryRequest(BaseModel):
    type: str = "query"
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question about financial data (max 1000 characters)")
    page: int = 1
    page_size: Optional[int] = None
    session_id: Optional[str] = None


class WSProgressUpdate(BaseModel):
    type: str = "progress"
    session_id: str
    progress: Dict[str, Any]
    message: str
    timestamp: str


class WSResultResponse(BaseModel):
    type: str = "result"
    session_id: str
    result: Dict[str, Any]
    success: bool
    timestamp: str
