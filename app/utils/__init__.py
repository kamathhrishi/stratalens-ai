"""
Utilities package for StrataLens API

This package contains:
- error_handlers: Sanitized error responses and exception handling
- general_utils: Common utilities, rate limiting, helper functions
"""

from .error_handlers import (
    create_error_response,
    raise_sanitized_http_exception,
    handle_database_error,
    handle_query_error,
    handle_rate_limit_error,
    handle_authentication_error,
    handle_authorization_error,
    handle_validation_error,
    handle_service_unavailable_error,
    handle_generic_error,
    sanitize_error_message,
    sanitize_error_for_frontend
)

from .general_utils import (
    RateLimiter,
    get_rate_limiter,
    rate_limiter,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_PER_MONTH,
    ADMIN_RATE_LIMIT_PER_MONTH,
    record_successful_query_usage,
    format_currency,
    format_percentage,
    format_number,
    safe_divide,
    generate_request_id,
    log_request,
    log_response,
    validate_ticker,
    validate_date_range,
    get_user_type,
    is_demo_user,
    is_authorized_user
)

__all__ = [
    # Error handlers
    'create_error_response',
    'raise_sanitized_http_exception',
    'handle_database_error', 
    'handle_query_error',
    'handle_rate_limit_error',
    'handle_authentication_error',
    'handle_authorization_error',
    'handle_validation_error',
    'handle_service_unavailable_error',
    'handle_generic_error',
    'sanitize_error_message',
    'sanitize_error_for_frontend',
    
    # General utilities
    'RateLimiter',
    'get_rate_limiter',
    'rate_limiter',
    'RATE_LIMIT_PER_MINUTE',
    'RATE_LIMIT_PER_MONTH',
    'ADMIN_RATE_LIMIT_PER_MONTH',
    'record_successful_query_usage',
    'format_currency',
    'format_percentage', 
    'format_number',
    'safe_divide',
    'generate_request_id',
    'log_request',
    'log_response',
    'validate_ticker',
    'validate_date_range',
    'get_user_type',
    'is_demo_user',
    'is_authorized_user'
]
