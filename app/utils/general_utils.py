"""
General utilities for the StrataLens API
This module contains common utilities used across the application including:
- Rate limiting system
- Common helper functions
- Shared constants and configurations
"""

import os
import time
import uuid
import logging
import traceback
from datetime import datetime, date
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict

import asyncpg
from fastapi import HTTPException, status

# Import configuration
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# RATE LIMITING SYSTEM
# =============================================================================

# Rate limiting constants
RATE_LIMIT_PER_MINUTE = settings.RATE_LIMITING.PER_MINUTE
RATE_LIMIT_PER_MONTH = settings.RATE_LIMITING.PER_MONTH
ADMIN_RATE_LIMIT_PER_MONTH = settings.RATE_LIMITING.ADMIN_PER_MONTH

class RateLimiter:
    """
    Rate limiting middleware class
    Note: This tracks requests for rate limiting purposes only, not for billing
    Billing is handled separately in record_successful_query_usage() only for successful queries
    """
    
    def __init__(self):
        self.minute_requests = defaultdict(list)  # user_id -> list of timestamps
        self.monthly_requests = defaultdict(list)   # user_id -> list of dates
        
    def _cleanup_old_requests(self, user_id: str):
        """Clean up old requests from memory"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean minute-based requests
        if user_id in self.minute_requests:
            self.minute_requests[user_id] = [
                ts for ts in self.minute_requests[user_id] 
                if ts > minute_ago
            ]
        
        # Clean monthly requests (keep only current month)
        today = date.today()
        current_month_start = today.replace(day=1)
        if user_id in self.monthly_requests:
            self.monthly_requests[user_id] = [
                d for d in self.monthly_requests[user_id] 
                if d >= current_month_start
            ]
    
    def check_rate_limit(self, user_id: str, is_admin: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """Check if user is within rate limits (minute limit only - monthly is checked separately)"""
        self._cleanup_old_requests(user_id)
        
        now = time.time()
        
        # Check minute limit (in-memory only)
        minute_count = len(self.minute_requests[user_id])
        minute_remaining = max(0, RATE_LIMIT_PER_MINUTE - minute_count)
        
        # Check if minute limit exceeded
        if minute_count >= RATE_LIMIT_PER_MINUTE:
            return False, {
                "limit_type": "minute",
                "limit": RATE_LIMIT_PER_MINUTE,
                "reset_time": datetime.fromtimestamp(now + 60).isoformat(),
                "message": f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute."
            }
        
        return True, {
            "minute_remaining": minute_remaining,
            "monthly_remaining": 0,  # Will be calculated separately from database
            "monthly_limit": ADMIN_RATE_LIMIT_PER_MONTH if is_admin else RATE_LIMIT_PER_MONTH
        }
    
    async def check_rate_limit_with_monthly(self, user_id: str, is_admin: bool, db: asyncpg.Connection) -> Tuple[bool, Dict[str, Any]]:
        """Check both minute and monthly rate limits together"""
        # First check minute limit
        minute_allowed, minute_info = self.check_rate_limit(user_id, is_admin)
        
        if not minute_allowed:
            return False, minute_info
        
        # Now check monthly limit from database
        today = date.today()
        month_start = today.replace(day=1)
        monthly_limit = ADMIN_RATE_LIMIT_PER_MONTH if is_admin else RATE_LIMIT_PER_MONTH
        
        # FIXED: Clean up old monthly requests first
        self._cleanup_old_requests(user_id)
        
        # Get monthly count from database
        monthly_count_db = await db.fetchval('''
            SELECT COALESCE(SUM(request_count), 0) 
            FROM user_usage 
            WHERE user_id = $1 AND request_date >= $2
        ''', uuid.UUID(user_id), month_start)
        
        # FIXED: Get in-memory monthly count (for pending requests)
        monthly_count_memory = len(self.monthly_requests[user_id])
        
        # FIXED: Use the higher of the two counts to prevent race conditions
        monthly_count = max(monthly_count_db, monthly_count_memory)
        
        logger.info(f"📊 Unified rate limit check for user {user_id}: monthly_count_db={monthly_count_db}, monthly_count_memory={monthly_count_memory}, monthly_count={monthly_count}, monthly_limit={monthly_limit}")
        
        # Check if monthly limit exceeded
        if monthly_count >= monthly_limit:
            # Calculate next month's start time for reset
            if today.month == 12:
                next_month = today.replace(year=today.year + 1, month=1, day=1)
            else:
                next_month = today.replace(month=today.month + 1, day=1)
            
            logger.warning(f"Rate limit exceeded for user {user_id}: Monthly limit exceeded ({monthly_count}/{monthly_limit})")
            
            return False, {
                "limit_type": "month",
                "limit": monthly_limit,
                "reset_time": next_month.isoformat(),
                "message": f"Monthly limit exceeded. Maximum {monthly_limit} requests per month."
            }
        
        # Both limits are OK
        monthly_remaining = max(0, monthly_limit - monthly_count)
        return True, {
            "minute_remaining": minute_info.get("minute_remaining", 0),
            "monthly_remaining": monthly_remaining,
            "monthly_limit": monthly_limit,
            "monthly_count": monthly_count
        }
    
    def record_request(self, user_id: str):
        """Record a new request for the user"""
        now = time.time()
        today = date.today()
        
        self.minute_requests[user_id].append(now)
        self.monthly_requests[user_id].append(today)

# Global rate limiter instance
rate_limiter = RateLimiter()

def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    return rate_limiter

# =============================================================================
# COMMON UTILITY FUNCTIONS
# =============================================================================

def safe_uuid_conversion(value: str) -> Optional[uuid.UUID]:
    """Safely convert string to UUID, return None if invalid"""
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError):
        return None

def validate_user_id(user_id: str) -> uuid.UUID:
    """Validate and convert user ID string to UUID"""
    try:
        return uuid.UUID(user_id)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user ID format"
        )

def format_error_message(error: Exception, context: str = "") -> str:
    """Format error message with context for logging"""
    error_msg = str(error)
    if context:
        return f"{context}: {error_msg}"
    return error_msg

def log_error_with_traceback(message: str, error: Exception, logger_instance: logging.Logger = None):
    """Log error with full traceback"""
    if logger_instance is None:
        logger_instance = logger
    
    logger_instance.error(f"{message}: {str(error)}")
    logger_instance.error(f"Traceback: {traceback.format_exc()}")

def get_current_month_start() -> date:
    """Get the first day of the current month"""
    today = date.today()
    return today.replace(day=1)

def get_next_month_start() -> date:
    """Get the first day of next month"""
    today = date.today()
    if today.month == 12:
        return today.replace(year=today.year + 1, month=1, day=1)
    else:
        return today.replace(month=today.month + 1, day=1)

def is_valid_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    import re
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')
    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    return sanitized

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length with optional suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result

def remove_none_values(data: Dict) -> Dict:
    """Remove None values from dictionary recursively"""
    if isinstance(data, dict):
        return {k: remove_none_values(v) for k, v in data.items() if v is not None}
    elif isinstance(data, list):
        return [remove_none_values(item) for item in data if item is not None]
    else:
        return data

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return os.path.splitext(filename)[1].lower()

def is_supported_file_type(filename: str, supported_extensions: List[str]) -> bool:
    """Check if file extension is in supported list"""
    return get_file_extension(filename) in supported_extensions

def generate_unique_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix"""
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id

def parse_query_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and clean query parameters"""
    cleaned = {}
    for key, value in params.items():
        if value is not None:
            # Convert string numbers to appropriate types
            if isinstance(value, str):
                if value.isdigit():
                    cleaned[key] = int(value)
                elif value.replace('.', '').isdigit():
                    try:
                        cleaned[key] = float(value)
                    except ValueError:
                        cleaned[key] = value
                elif value.lower() in ['true', 'false']:
                    cleaned[key] = value.lower() == 'true'
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value
    return cleaned

# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_currency(value: float, currency: str = "USD") -> str:
    """Format a number as currency"""
    if currency == "USD":
        return f"${value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage"""
    return f"{value:.{decimals}f}%"

def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with specified decimal places"""
    return f"{value:,.{decimals}f}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    if denominator == 0:
        return default
    return numerator / denominator

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())

def log_request(request_id: str, method: str, path: str, user_id: Optional[str] = None):
    """Log incoming request"""
    logger.info(f"📥 Request {request_id}: {method} {path} (user: {user_id or 'anonymous'})")

def log_response(request_id: str, status_code: int, response_time: float):
    """Log response"""
    logger.info(f"📤 Response {request_id}: {status_code} ({response_time:.3f}s)")

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_ticker(ticker: str) -> bool:
    """Validate stock ticker format"""
    if not ticker or not isinstance(ticker, str):
        return False
    # Basic ticker validation: 1-5 uppercase letters
    import re
    return re.match(r'^[A-Z]{1,5}$', ticker.upper()) is not None

def validate_date_range(start_date: str, end_date: str) -> bool:
    """Validate date range format and logic"""
    try:
        from datetime import datetime
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        return start <= end
    except (ValueError, AttributeError):
        return False

# =============================================================================
# USER TYPE FUNCTIONS
# =============================================================================

def get_user_type(user_id: Optional[str]) -> str:
    """Get user type based on user ID"""
    if not user_id:
        return "DEMO"
    # In a real implementation, you might check the database or user properties
    # For now, we'll assume all authenticated users are AUTHORIZED
    return "AUTHORIZED"

def is_demo_user(user_id: Optional[str]) -> bool:
    """Check if user is a demo user"""
    return get_user_type(user_id) == "DEMO"

def is_authorized_user(user_id: Optional[str]) -> bool:
    """Check if user is an authorized user"""
    return get_user_type(user_id) == "AUTHORIZED"

# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    except ImportError:
        # Fallback if psutil is not available
        return 0.0
    except Exception:
        # Fallback if there's any error
        return 0.0

# =============================================================================
# USAGE TRACKING FUNCTIONS
# =============================================================================

async def record_successful_query_usage(user_id: str, db: asyncpg.Connection, cost_per_request: float = None):
    """Record usage for a successful query"""
    try:
        # Get cost per request from settings if not provided
        if cost_per_request is None:
            cost_per_request = settings.RATE_LIMITING.COST_PER_REQUEST
            
        today = date.today()
        current_hour = datetime.now().hour
        logger.info(f"📊 Recording successful query usage for user {user_id}: date={today}, hour={current_hour}, cost=${cost_per_request}")
        
        await db.execute('''
            INSERT INTO user_usage (user_id, request_date, request_hour, request_count, total_cost)
            VALUES ($1, $2, $3, 1, $4)
            ON CONFLICT (user_id, request_date, request_hour)
            DO UPDATE SET 
                request_count = user_usage.request_count + 1,
                total_cost = user_usage.total_cost + $4,
                updated_at = CURRENT_TIMESTAMP
        ''', uuid.UUID(user_id), today, current_hour, cost_per_request)
        
        logger.info(f"✅ Usage recorded successfully for user {user_id}")
    except Exception as e:
        logger.error(f"❌ Failed to record usage for user {user_id}: {e}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")

# =============================================================================
# RATE LIMITING CONSTANTS EXPORT
# =============================================================================

__all__ = [
    'RateLimiter',
    'get_rate_limiter',
    'rate_limiter',
    'RATE_LIMIT_PER_MINUTE',
    'RATE_LIMIT_PER_MONTH', 
    'ADMIN_RATE_LIMIT_PER_MONTH',
    'record_successful_query_usage',
    'safe_uuid_conversion',
    'validate_user_id',
    'format_error_message',
    'log_error_with_traceback',
    'get_current_month_start',
    'get_next_month_start',
    'is_valid_email',
    'sanitize_filename',
    'format_bytes',
    'truncate_string',
    'deep_merge_dicts',
    'remove_none_values',
    'get_file_extension',
    'is_supported_file_type',
    'generate_unique_id',
    'parse_query_params',
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
    'is_authorized_user',
    'get_memory_usage'
]
