# Create a new file: jwt_config.py
"""
Centralized JWT configuration to ensure consistent SECRET_KEY across modules
"""
import os
import secrets

# Generate a consistent SECRET_KEY
# In production, this should be set as an environment variable
_SECRET_KEY = None

def get_secret_key():
    """Get the SECRET_KEY, generating one if needed"""
    global _SECRET_KEY
    if _SECRET_KEY is None:
        # Try to get from environment first
        _SECRET_KEY = os.getenv("JWT_SECRET_KEY")
        
        # If not in environment, generate a random one
        # In production, you should set JWT_SECRET_KEY as an environment variable
        if not _SECRET_KEY:
            # Generate a random secret key if not provided (different on each restart)
            _SECRET_KEY = secrets.token_urlsafe(32)
            print("⚠️  WARNING: Generated random JWT secret key (will change on restart). Set JWT_SECRET_KEY environment variable for production!")
    
    return _SECRET_KEY

# Constants
SECRET_KEY = get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

print("🔐 JWT Config initialized successfully")

# JWT utility functions that should be used across all modules
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT access token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise jwt.ExpiredSignatureError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")

def verify_token(token: str) -> Optional[str]:
    """Verify token and return user_id if valid"""
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        return user_id
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None