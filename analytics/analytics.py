"""
Analytics models for tracking chat search queries and user behavior

NOTE: All models have been moved to schemas/analytics.py
This file is kept for backward compatibility with imports.
"""

# Re-export all analytics models from schemas
from app.schemas.analytics import (
    UserType,
    ChatAnalytics,
    AnalyticsSummary,
    AnalyticsQuery,
    AnalyticsResponse
)

__all__ = [
    'UserType',
    'ChatAnalytics',
    'AnalyticsSummary',
    'AnalyticsQuery',
    'AnalyticsResponse'
]
