"""
Chat models for agent-powered conversation endpoints
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ChatCitation(BaseModel):
    """Citation information for RAG responses - supports transcript, news, and 10-K citations"""
    # All fields are optional to support different citation types
    # Transcript citations: company, quarter, chunk_id, chunk_text, relevance_score
    # News citations: title, url, published_date
    # 10-K citations: ticker, fiscal_year, section, path, chunk_type

    # Common fields
    company: Optional[str] = Field(None, description="Company ticker symbol")
    ticker: Optional[str] = Field(None, description="Ticker symbol (alias for company)")
    chunk_id: Optional[str] = Field(None, description="Unique identifier for the text chunk")
    chunk_text: Optional[str] = Field(None, description="Relevant text chunk or title")
    relevance_score: Optional[float] = Field(None, description="Similarity score (0-1)")
    source_file: Optional[str] = Field(None, description="Original source file name or URL")
    transcript_available: Optional[bool] = Field(False, description="Whether complete transcript is available")

    # Citation type
    citation_type: Optional[str] = Field(None, description="Type of citation: 'transcript', 'news', '10-K'")
    type: Optional[str] = Field(None, description="Type of citation (alias)")

    # Transcript-specific fields
    quarter: Optional[str] = Field(None, description="Quarter (e.g., '2025_q1')")
    year: Optional[int] = Field(None, description="Year for transcript citations")

    # News-specific fields
    title: Optional[str] = Field(None, description="Title for news citations")
    url: Optional[str] = Field(None, description="URL for news citations")
    published_date: Optional[str] = Field(None, description="Published date for news citations")

    # 10-K specific fields
    fiscal_year: Optional[int] = Field(None, description="Fiscal year for 10-K citations")
    section: Optional[str] = Field(None, description="SEC section for 10-K citations")
    path: Optional[str] = Field(None, description="Document path for 10-K citations")
    chunk_type: Optional[str] = Field(None, description="Chunk type for 10-K citations")

    # Marker/source number
    marker: Optional[str] = Field(None, description="Citation marker (e.g., [1], [N1])")
    source_number: Optional[int] = Field(None, description="Source number in the response")

    class Config:
        extra = "allow"  # Allow additional fields not defined in schema


class ChatMessage(BaseModel):
    """Chat message model"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message (max 1000 characters)")
    comprehensive: bool = Field(True, description="Whether to use comprehensive search")
    session_id: Optional[str] = Field(None, description="Client-generated session ID for anonymous users")
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID (None for new conversation)")


class ChatResponse(BaseModel):
    """Chat response with RAG results"""
    success: bool = Field(..., description="Whether the request was successful")
    answer: str = Field(..., description="Generated response from RAG system")
    citations: List[ChatCitation] = Field(default=[], description="Supporting citations")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Question analysis details")
    timing: Optional[Dict[str, float]] = Field(None, description="Processing timing breakdown")
    error: Optional[str] = Field(None, description="Error message if any")
    session_id: Optional[str] = Field(None, description="Session ID used for this response (for anonymous users)")
    conversation_id: Optional[str] = Field(None, description="Conversation thread ID")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


# New conversation thread models
class ChatConversationMessage(BaseModel):
    """Single message within a conversation"""
    id: str = Field(..., description="Unique message ID")
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    citations: List[ChatCitation] = Field(default=[], description="Citations (for assistant messages)")
    reasoning: List[dict] = Field(default=[], description="Reasoning steps (for assistant messages)")
    created_at: datetime = Field(..., description="When the message was created")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ChatConversation(BaseModel):
    """Complete conversation thread (like ChatGPT)"""
    id: str = Field(..., description="Unique conversation ID")
    title: str = Field(..., description="Auto-generated conversation title")
    messages: List[ChatConversationMessage] = Field(default=[], description="All messages in conversation")
    created_at: datetime = Field(..., description="When conversation was created")
    updated_at: datetime = Field(..., description="When conversation was last updated")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ChatConversationsResponse(BaseModel):
    """Response with list of conversation threads"""
    success: bool = Field(..., description="Whether the request was successful")
    conversations: List[ChatConversation] = Field(default=[], description="List of conversation threads")
    total_count: int = Field(0, description="Total number of conversations")
    error: Optional[str] = Field(None, description="Error message if any")


class ChatConversationRequest(BaseModel):
    """Request to start or continue a conversation"""
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID (None for new conversation)")
    message: str = Field(..., min_length=1, max_length=1000, description="User message (max 1000 characters)")
    comprehensive: bool = Field(True, description="Whether to use comprehensive search")


# Legacy models for backward compatibility
class ChatHistoryItem(BaseModel):
    """Single chat history item (legacy)"""
    id: str = Field(..., description="Unique message ID")
    user_message: str = Field(..., description="Original user message")
    assistant_response: str = Field(..., description="Assistant's response")
    citations: List[ChatCitation] = Field(default=[], description="Citations used in response")
    created_at: datetime = Field(..., description="When the message was created")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ChatHistoryResponse(BaseModel):
    """Chat history response (legacy)"""
    success: bool = Field(..., description="Whether the request was successful")
    messages: List[ChatHistoryItem] = Field(default=[], description="Chat history messages")
    total_count: int = Field(0, description="Total number of messages")
    filtered_count: Optional[int] = Field(None, description="Number of messages after filtering")
    error: Optional[str] = Field(None, description="Error message if any")


class ChatClearRequest(BaseModel):
    """Request to clear chat history"""
    confirm: bool = Field(True, description="Confirmation to clear history")


class ChatClearResponse(BaseModel):
    """Response to chat clear request"""
    success: bool = Field(..., description="Whether the clear was successful")
    message: str = Field(..., description="Success or error message")
    cleared_count: int = Field(0, description="Number of messages cleared")
