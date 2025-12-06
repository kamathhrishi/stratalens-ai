import logging
import os
import json
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import asyncpg
from dataclasses import dataclass, asdict
import traceback
import sys

# Configure comprehensive logging with single file and console output
def setup_logging():
    """Setup comprehensive logging configuration with single log file"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger with single file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Single file handler for ALL logs
            logging.FileHandler('logs/stratalens.log'),
            # Console handler for immediate visibility
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    return logger

# Initialize logging
logger = setup_logging()

def log_message(message: str, is_milestone: bool = False, 
               user_id: Optional[str] = None, username: Optional[str] = None,
               level: str = "INFO", extra_data: Optional[Dict[str, Any]] = None,
               logger_name: str = "system"):
    """
    Comprehensive logging function that logs to single file and console
    
    Args:
        message: The message to log
        is_milestone: Whether this is a milestone event
        user_id: User ID for user-specific logging
        username: Username for user-specific logging
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        extra_data: Additional data to log
        logger_name: Which logger to use (system, search, auth, etc.)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Add user context if available
    user_context = ""
    if user_id and username:
        user_context = f" [User: {username}({user_id})]"
    elif user_id:
        user_context = f" [User: {user_id}]"
    
    # Format the message
    if is_milestone:
        formatted_message = f"=== MILESTONE ==={user_context} {message}"
    else:
        formatted_message = f"{level.upper()}:{user_context} {message}"
    
    # Get the appropriate logger
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Log to file and console via Python logging
    logger.log(log_level, formatted_message)
    
    # Also print to console for immediate visibility (redundant but ensures visibility)
    console_message = f"[{timestamp}] {formatted_message}"
    print(console_message)
    
    # Log extra data if provided
    if extra_data:
        try:
            extra_log = f"EXTRA DATA: {json.dumps(extra_data, indent=2)}"
            logger.log(log_level, extra_log)
            print(f"[{timestamp}] {extra_log}")
        except Exception as e:
            error_msg = f"Failed to log extra data: {e}"
            logger.error(error_msg)
            print(f"[{timestamp}] ERROR: {error_msg}")

def log_error(message: str, error: Optional[Exception] = None, 
              user_id: Optional[str] = None, username: Optional[str] = None,
              extra_data: Optional[Dict[str, Any]] = None, logger_name: str = "system"):
    """Log an error message with optional exception details"""
    if error:
        error_details = f"{message}: {str(error)}"
        if hasattr(error, '__traceback__'):
            error_details += f"\nTraceback: {traceback.format_exc()}"
    else:
        error_details = message
    
    log_message(error_details, is_milestone=False, user_id=user_id, username=username,
                level="ERROR", extra_data=extra_data, logger_name=logger_name)

def log_warning(message: str, user_id: Optional[str] = None, username: Optional[str] = None,
                extra_data: Optional[Dict[str, Any]] = None, logger_name: str = "system"):
    """Log a warning message"""
    log_message(message, is_milestone=False, user_id=user_id, username=username,
                level="WARNING", extra_data=extra_data, logger_name=logger_name)

def log_info(message: str, user_id: Optional[str] = None, username: Optional[str] = None,
             extra_data: Optional[Dict[str, Any]] = None, logger_name: str = "system"):
    """Log an info message"""
    log_message(message, is_milestone=False, user_id=user_id, username=username,
                level="INFO", extra_data=extra_data, logger_name=logger_name)

def log_debug(message: str, user_id: Optional[str] = None, username: Optional[str] = None,
              extra_data: Optional[Dict[str, Any]] = None, logger_name: str = "system"):
    """Log a debug message"""
    log_message(message, is_milestone=False, user_id=user_id, username=username,
                level="DEBUG", extra_data=extra_data, logger_name=logger_name)

def log_milestone(message: str, user_id: Optional[str] = None, username: Optional[str] = None,
                  extra_data: Optional[Dict[str, Any]] = None, logger_name: str = "system"):
    """Log a milestone message"""
    log_message(message, is_milestone=True, user_id=user_id, username=username,
                level="INFO", extra_data=extra_data, logger_name=logger_name)

# Database logging support
class DatabaseLogger:
    """Database logging support for storing logs in PostgreSQL"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
    
    async def log_to_database(self, message: str, level: str, is_milestone: bool,
                             user_id: Optional[str], username: Optional[str],
                             extra_data: Optional[Dict[str, Any]], timestamp: str,
                             logger_name: str = "system"):
        """Log message to database"""
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO system_logs (
                        user_id, username, message, level, is_milestone, 
                        extra_data, created_at, logger_name
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''', user_id, username, message, level, is_milestone,
                    json.dumps(extra_data) if extra_data else None,
                    datetime.fromisoformat(timestamp.replace(' ', 'T')),
                    logger_name)
        except Exception as e:
            # Fallback to console if database logging fails
            error_msg = f"Database logging failed: {e}"
            log_error(error_msg, extra_data={"original_message": message})

# Global database logger instance
db_logger = None

def set_database_logger(db_pool: asyncpg.Pool):
    """Set the global database logger instance"""
    global db_logger
    db_logger = DatabaseLogger(db_pool)

def get_database_logger() -> Optional[DatabaseLogger]:
    """Get the global database logger instance"""
    return db_logger

# Enhanced logging function with database support
def log_with_database(message: str, is_milestone: bool = False, 
                     user_id: Optional[str] = None, username: Optional[str] = None,
                     level: str = "INFO", extra_data: Optional[Dict[str, Any]] = None,
                     logger_name: str = "system"):
    """Enhanced logging function that also logs to database if available"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Log to file and console
    log_message(message, is_milestone, user_id, username, level, extra_data, logger_name)
    
    # Also log to database if available
    if db_logger:
        asyncio.create_task(db_logger.log_to_database(
            message=message,
            level=level,
            is_milestone=is_milestone,
            user_id=user_id,
            username=username,
            extra_data=extra_data,
            timestamp=timestamp,
            logger_name=logger_name
        ))

# Profiling utilities
from contextlib import contextmanager
import functools

@contextmanager
def timer(label: str, log_result: bool = True):
    """Context manager for timing code blocks.
    
    Usage:
        with timer("search_operation"):
            # do search
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if log_result:
            logger.info(f"⏱️  {label}: {elapsed:.4f}s")

def profile_function(name: str = None):
    """Decorator to profile function execution time.
    
    Usage:
        @profile_function("my_function")
        async def my_function():
            # do work
    """
    def decorator(func):
        func_name = name or func.__name__
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with timer(func_name):
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with timer(func_name):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Initialize logging on module import
setup_logging()
