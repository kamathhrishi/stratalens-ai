"""
Error handling utilities for StrataLens API
Provides sanitized error responses that hide backend system details from frontend users
"""

import logging
import traceback
from typing import Dict, Any, Optional
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Generic error messages that don't expose backend details
GENERIC_ERROR_MESSAGES = {
    "database_error": "We're experiencing a temporary issue with our data service. Please try again in a few moments.",
    "rate_limit": "You've reached your request limit. Please wait a moment before trying again.",
    "authentication_error": "Please log in again to continue.",
    "authorization_error": "You don't have permission to access this resource.",
    "validation_error": "The information you provided is invalid. Please check your input and try again.",
    "service_unavailable": "This service is temporarily unavailable. Please try again later.",
    "query_timeout": "Your request is taking longer than expected. Please try again with a simpler query.",
    "data_not_found": "The requested data could not be found.",
    "processing_error": "We encountered an error while processing your request. Please try again.",
    "server_error": "We're sorry, but something went wrong on our end. Please try again or contact support if the problem persists."
}

def sanitize_error_for_frontend(
    error: Exception, 
    operation: str = "operation",
    user_id: Optional[str] = None,
    custom_message: Optional[str] = None
) -> str:
    """
    Sanitize error messages for frontend consumption
    
    Args:
        error: The exception that occurred
        operation: Description of what operation failed (for logging)
        user_id: User ID for logging context
        custom_message: Custom user-friendly message to return
        
    Returns:
        User-friendly error message that doesn't expose backend details
    """
    
    # Log the actual error details for debugging
    error_details = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "user_id": user_id
    }
    
    logger.error(f"❌ {operation} failed: {error_details}")
    logger.debug(f"❌ Traceback: {traceback.format_exc()}")
    
    # Return custom message if provided
    if custom_message:
        return custom_message
    
    # Categorize common error types and return appropriate generic messages
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Database-related errors
    if any(keyword in error_str for keyword in ['connection', 'database', 'sql', 'asyncpg', 'duckdb']):
        return GENERIC_ERROR_MESSAGES["database_error"]
    
    # Rate limiting errors
    if any(keyword in error_str for keyword in ['rate limit', 'too many requests', '429']):
        return GENERIC_ERROR_MESSAGES["rate_limit"]
    
    # Authentication/authorization errors
    if any(keyword in error_str for keyword in ['unauthorized', 'forbidden', 'token', 'auth']):
        if '401' in error_str or 'unauthorized' in error_str:
            return GENERIC_ERROR_MESSAGES["authentication_error"]
        elif '403' in error_str or 'forbidden' in error_str:
            return GENERIC_ERROR_MESSAGES["authorization_error"]
    
    # Validation errors
    if any(keyword in error_str for keyword in ['validation', 'invalid', 'bad request', '400']):
        return GENERIC_ERROR_MESSAGES["validation_error"]
    
    # Service unavailable
    if any(keyword in error_str for keyword in ['service unavailable', '503', 'not available', 'disabled']):
        return GENERIC_ERROR_MESSAGES["service_unavailable"]
    
    # Timeout errors
    if any(keyword in error_str for keyword in ['timeout', 'time out', 'took too long']):
        return GENERIC_ERROR_MESSAGES["query_timeout"]
    
    # Data not found
    if any(keyword in error_str for keyword in ['not found', '404', 'no data', 'empty result']):
        return GENERIC_ERROR_MESSAGES["data_not_found"]
    
    # Default to generic server error
    return GENERIC_ERROR_MESSAGES["server_error"]

def create_error_response(
    error: Exception,
    operation: str = "operation",
    user_id: Optional[str] = None,
    custom_message: Optional[str] = None,
    status_code: int = 500
) -> Dict[str, Any]:
    """
    Create a standardized error response for API endpoints
    
    Args:
        error: The exception that occurred
        operation: Description of what operation failed
        user_id: User ID for logging context
        custom_message: Custom user-friendly message
        status_code: HTTP status code to return
        
    Returns:
        Standardized error response dictionary
    """
    
    sanitized_message = sanitize_error_for_frontend(
        error, operation, user_id, custom_message
    )
    
    return {
        "success": False,
        "error": sanitized_message,
        "message": sanitized_message
    }

def raise_sanitized_http_exception(
    error: Exception,
    operation: str = "operation",
    user_id: Optional[str] = None,
    custom_message: Optional[str] = None,
    status_code: int = 500
) -> None:
    """
    Raise an HTTPException with sanitized error message
    
    Args:
        error: The exception that occurred
        operation: Description of what operation failed
        user_id: User ID for logging context
        custom_message: Custom user-friendly message
        status_code: HTTP status code to return
    """
    
    sanitized_message = sanitize_error_for_frontend(
        error, operation, user_id, custom_message
    )
    
    raise HTTPException(status_code=status_code, detail=sanitized_message)

# Specific error handlers for common scenarios
def handle_database_error(error: Exception, operation: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Handle database-related errors"""
    return create_error_response(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["database_error"]
    )

def handle_query_error(error: Exception, operation: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Handle query processing errors"""
    return create_error_response(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["processing_error"]
    )

def handle_authentication_error(error: Exception, operation: str, user_id: Optional[str] = None) -> None:
    """Handle authentication errors"""
    raise_sanitized_http_exception(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["authentication_error"],
        status.HTTP_401_UNAUTHORIZED
    )

def handle_authorization_error(error: Exception, operation: str, user_id: Optional[str] = None) -> None:
    """Handle authorization errors"""
    raise_sanitized_http_exception(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["authorization_error"],
        status.HTTP_403_FORBIDDEN
    )

def handle_validation_error(error: Exception, operation: str, user_id: Optional[str] = None) -> None:
    """Handle validation errors"""
    raise_sanitized_http_exception(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["validation_error"],
        status.HTTP_400_BAD_REQUEST
    )

def handle_service_unavailable_error(error: Exception, operation: str, user_id: Optional[str] = None) -> None:
    """Handle service unavailable errors"""
    raise_sanitized_http_exception(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["service_unavailable"],
        status.HTTP_503_SERVICE_UNAVAILABLE
    )

def handle_rate_limit_error(error: Exception, operation: str, user_id: Optional[str] = None) -> None:
    """Handle rate limiting errors"""
    raise_sanitized_http_exception(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["rate_limit"],
        status.HTTP_429_TOO_MANY_REQUESTS
    )

def handle_generic_error(error: Exception, operation: str, user_id: Optional[str] = None) -> None:
    """Handle generic errors"""
    raise_sanitized_http_exception(
        error, 
        operation, 
        user_id, 
        GENERIC_ERROR_MESSAGES["server_error"],
        status.HTTP_500_INTERNAL_SERVER_ERROR
    )

def sanitize_error_message(error: Exception, operation: str = "operation", user_id: Optional[str] = None) -> str:
    """Alias for sanitize_error_for_frontend for backward compatibility"""
    return sanitize_error_for_frontend(error, operation, user_id)