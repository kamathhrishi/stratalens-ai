"""
Centralized authentication utilities for the StrataLens API
"""
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncpg
from .jwt_config import decode_access_token

logger = logging.getLogger(__name__)

# Security setup
security = HTTPBearer()

# Import database utilities
from db.db_utils import get_db

async def authenticate_user_by_token(token: str, db: asyncpg.Connection, auth_type: str = "MAIN") -> Dict[str, Any]:
    """Common authentication logic for all endpoints"""
    try:
        logger.info(f"🔐 {auth_type} AUTH: Verifying token: {token[:20]}...")
        
        # Use the shared JWT decode function
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            logger.error(f"🔐 {auth_type} AUTH: Token payload missing 'sub' field")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        logger.info(f"🔐 {auth_type} AUTH: Token valid for user_id: {user_id}")
        
    except Exception as e:
        logger.error(f"🔐 {auth_type} AUTH: JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Verify user exists and is active
    user = await db.fetchrow(
        "SELECT id, username, email, full_name, is_active, is_approved, is_admin FROM users WHERE id = $1",
        uuid.UUID(user_id)
    )
    
    if not user:
        logger.error(f"🔐 {auth_type} AUTH: User not found in database: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    if not user['is_active'] or not user['is_approved']:
        logger.error(f"🔐 {auth_type} AUTH: User not active or approved: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account not active or not approved"
        )
    
    logger.info(f"🔐 {auth_type} AUTH: User authenticated successfully: {user['username']}")
    return {
        "id": str(user['id']),
        "username": user['username'],
        "email": user['email'],
        "full_name": user['full_name'],
        "is_admin": user['is_admin']
    }

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security), 
    db: asyncpg.Connection = Depends(get_db)
) -> Dict[str, Any]:
    """Get current authenticated user - standard dependency for all endpoints"""
    token = credentials.credentials
    return await authenticate_user_by_token(token, db, "MAIN")

async def get_current_user_for_stream(
    request: Request,
    db: asyncpg.Connection = Depends(get_db)
) -> Dict[str, Any]:
    """Get current user for streaming endpoints (supports token in header or query)"""
    # Get token from header or query
    token = None
    try:
        # Try to get from Authorization header first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        else:
            # Try to get from query parameter
            token = request.query_params.get("token")
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication token required. Provide token in Authorization header or as query parameter.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    except Exception as e:
        logger.error(f"Token extraction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return await authenticate_user_by_token(token, db, "STREAM")

async def get_optional_user(
    request: Request,
    db: asyncpg.Connection = Depends(get_db)
) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None if not - for optional auth endpoints"""
    try:
        # Try to get from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            return await authenticate_user_by_token(token, db, "OPTIONAL")
    except Exception as e:
        logger.info(f"🔓 Optional auth failed (this is ok for anonymous requests): {e}")
    
    return None

async def authenticate_user_by_id(user_id: str, db_pool) -> Dict[str, Any]:
    """Authenticate user by ID for WebSocket connections"""
    try:
        async with db_pool.acquire() as db:
            # Verify user exists and is active
            user = await db.fetchrow(
                "SELECT id, username, email, full_name, is_active, is_approved, is_admin FROM users WHERE id = $1", 
                uuid.UUID(user_id)
            )
        
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        if not user['is_active'] or not user['is_approved']:
            raise HTTPException(
                status_code=403,
                detail="Account not active or not approved"
            )
        
        logger.info(f"🔐 WebSocket authenticated user: {user['username']}")
        return {
            "id": str(user['id']),
            "username": user['username'],
            "email": user['email'],
            "full_name": user['full_name'],
            "is_admin": user['is_admin']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ WebSocket authentication error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication failed"
        )
