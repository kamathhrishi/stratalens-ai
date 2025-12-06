"""Central schemas/models package for StrataLens AI"""

# Auth schemas
from .auth import (
    UserLogin, UserRegistration, OnboardingCompletion,
    PasswordReset, PasswordChange, MagicLinkRequest, MagicLinkVerify
)

# User schemas
from .user import UserProfile, UserPreferences, UserProfileUpdate, UserProfileResponse

# Query schemas
from .query import (
    QueryHistoryItem, QueryRequest, QueryResponse, ExportRequest, SortRequest, SortResponse,
    ExpandValueRequest, ExpandValueResponse, ExpandScreenValueRequest,
    PaginationRequest
)

# Company schemas
from .company import (
    CompanySearchRequest, CompanySearchResult, CompanySearchResponse
)

# Admin schemas
from .admin import AdminUserApproval, AdminUsersList, CSVUserRequest, OnboardingCompleteRequest

# WebSocket schemas
from .websocket import WSMessage, WSQueryRequest, WSProgressUpdate, WSResultResponse

# Usage schemas
from .usage import UsageStats, RateLimitInfo

# Chat schemas
from .chat import (
    ChatMessage, ChatResponse, ChatCitation, ChatHistoryItem,
    ChatHistoryResponse, ChatClearRequest, ChatClearResponse,
    ChatConversation, ChatConversationMessage, ChatConversationsResponse, ChatConversationRequest
)

# Screen schemas
from .screens import SaveScreenRequest, SavedScreen, ScreenData, UpdateScreenRequest

# Analytics schemas
from .analytics import (
    ChatAnalytics, AnalyticsSummary, AnalyticsQuery, AnalyticsResponse, UserType
)

# RAG schemas
from .rag import QuestionAnalysisResult

__all__ = [
    # Auth schemas
    "UserLogin", "UserRegistration", "OnboardingCompletion",
    "PasswordReset", "PasswordChange", "MagicLinkRequest", "MagicLinkVerify",
    # User schemas
    "UserProfile", "UserPreferences", "UserProfileUpdate", "UserProfileResponse",
    # Query schemas
    "QueryHistoryItem", "QueryRequest", "QueryResponse", "ExportRequest", "SortRequest", "SortResponse",
    "ExpandValueRequest", "ExpandValueResponse", "ExpandScreenValueRequest",
    "PaginationRequest",
    # Company schemas
    "CompanySearchRequest", "CompanySearchResult", "CompanySearchResponse",
    # Admin schemas
    "AdminUserApproval", "AdminUsersList", "CSVUserRequest", "OnboardingCompleteRequest",
    # WebSocket schemas
    "WSMessage", "WSQueryRequest", "WSProgressUpdate", "WSResultResponse",
    # Usage schemas
    "UsageStats", "RateLimitInfo",
    # Chat schemas
    "ChatMessage", "ChatResponse", "ChatCitation", "ChatHistoryItem",
    "ChatHistoryResponse", "ChatClearRequest", "ChatClearResponse",
    "ChatConversation", "ChatConversationMessage", "ChatConversationsResponse", "ChatConversationRequest",
    # Screen schemas
    "SaveScreenRequest", "SavedScreen", "ScreenData", "UpdateScreenRequest",
    # Analytics schemas
    "ChatAnalytics", "AnalyticsSummary", "AnalyticsQuery", "AnalyticsResponse", "UserType",
    # RAG schemas
    "QuestionAnalysisResult"
]
