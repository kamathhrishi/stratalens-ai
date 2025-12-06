"""Analytics models for tracking chat search queries and user behavior"""
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class UserType(str, Enum):
    """User type enumeration"""
    DEMO = "demo"
    AUTHORIZED = "authorized"


class ChatAnalytics(BaseModel):
    """Analytics record for chat search queries"""
    id: Optional[str] = Field(None, description="Unique identifier (UUID)")
    user_id: Optional[str] = Field(None, description="User ID if authenticated (UUID)")
    ip_address: str = Field(..., description="Client IP address")
    user_type: UserType = Field(..., description="Type of user: demo or authorized")
    query_text: str = Field(..., description="The search query text")
    query_length: int = Field(..., description="Length of the query text")
    comprehensive_search: bool = Field(True, description="Whether comprehensive search was used")
    success: bool = Field(..., description="Whether the query was successful")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    citations_count: int = Field(0, description="Number of citations returned")
    error_message: Optional[str] = Field(None, description="Error message if query failed")
    user_agent: Optional[str] = Field(None, description="User agent string")
    session_id: Optional[str] = Field(None, description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when query was made")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class AnalyticsSummary(BaseModel):
    """Summary statistics for analytics"""
    total_queries: int = Field(..., description="Total number of queries")
    demo_queries: int = Field(..., description="Number of demo queries")
    authorized_queries: int = Field(..., description="Number of authorized user queries")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time in milliseconds")
    unique_ips: int = Field(..., description="Number of unique IP addresses")
    unique_users: int = Field(..., description="Number of unique authorized users")
    period_start: datetime = Field(..., description="Start of the analytics period")
    period_end: datetime = Field(..., description="End of the analytics period")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class AnalyticsQuery(BaseModel):
    """Query parameters for analytics data"""
    start_date: Optional[datetime] = Field(None, description="Start date for analytics query")
    end_date: Optional[datetime] = Field(None, description="End date for analytics query")
    user_type: Optional[UserType] = Field(None, description="Filter by user type")
    ip_address: Optional[str] = Field(None, description="Filter by IP address")
    success_only: bool = Field(False, description="Only include successful queries")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of records to return")
    offset: int = Field(0, ge=0, description="Number of records to skip")


class AnalyticsResponse(BaseModel):
    """Response containing analytics data"""
    success: bool = Field(..., description="Whether the request was successful")
    data: list[ChatAnalytics] = Field(default=[], description="Analytics records")
    summary: Optional[AnalyticsSummary] = Field(None, description="Summary statistics")
    total_count: int = Field(0, description="Total number of records matching query")
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
