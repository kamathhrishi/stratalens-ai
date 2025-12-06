from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Query, Request as FastAPIRequest, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, EmailStr
from typing import List, Dict, Optional, Any, Union
import asyncpg
import uuid
import logging
from datetime import datetime, timedelta
from passlib.context import CryptContext
import os
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import RedirectResponse

import json

# Import the shared JWT configuration
from .jwt_config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, create_access_token, decode_access_token, verify_token

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper function to validate required environment variables
# Environment variable helpers imported from config
from config import get_required_env_var, get_optional_env_var, settings

from fastapi import APIRouter
# Import comprehensive logging
from app.utils.logging_utils import log_message, log_error, log_warning, log_info, log_debug, log_milestone


router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

# Environment variable printing will be moved to startup phase

# Security setup
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth setup
oauth = OAuth()

# Magic link serializer
magic_link_serializer = URLSafeTimedSerializer(SECRET_KEY)

# Email configuration
SMTP_SERVER = get_optional_env_var("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(get_optional_env_var("SMTP_PORT", "587"))
SMTP_USERNAME = get_optional_env_var("SMTP_USERNAME")
SMTP_PASSWORD = get_optional_env_var("SMTP_PASSWORD")

# Google OAuth configuration
GOOGLE_CLIENT_ID = get_optional_env_var("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = get_optional_env_var("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = get_optional_env_var("GOOGLE_REDIRECT_URI")

# Configure Google OAuth client
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'}
    )

# Import schemas from centralized location
from app.schemas.auth import (
    UserLogin, UserRegistration, MagicLinkRequest, MagicLinkVerify,
    PasswordReset, PasswordChange, OnboardingCompletion
)


def generate_invitation_code() -> str:
    """Generate a unique invitation code"""
    return secrets.token_urlsafe(16)

def generate_invitation_url(invitation_code: str) -> str:
    """Generate the invitation URL"""
    base_url = get_required_env_var("BASE_URL", "Base URL for invitation links")
    return f"{base_url}/onboard/{invitation_code}"

# Security functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Email sending utility
async def send_email(to_email: str, subject: str, html_content: str, text_content: str = None):
    """Send email via SMTP"""
    if not SMTP_USERNAME or not SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured, cannot send email")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SMTP_USERNAME
        msg['To'] = to_email
        
        if text_content:
            text_part = MIMEText(text_content, 'plain')
            msg.attach(text_part)
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

def generate_magic_link_token(email: str) -> str:
    """Generate a secure magic link token"""
    return magic_link_serializer.dumps(email, salt='magic-link')

def verify_magic_link_token(token: str, max_age: int = 900) -> Optional[str]:  # 15 minutes
    """Verify magic link token and return email if valid"""
    try:
        email = magic_link_serializer.loads(token, salt='magic-link', max_age=max_age)
        return email
    except (BadSignature, SignatureExpired):
        return None

# Database dependency - will be set by main app
def get_db_dependency():
    """
    This will be set by the main app to provide the actual database dependency.
    This allows the auth module to work independently while still accessing the shared db_pool.
    """
    raise HTTPException(status_code=503, detail="Database dependency not initialized")

# This will be set by the main app
get_db = get_db_dependency

def set_db_dependency(db_func):
    """Called by main app to set the database dependency function"""
    global get_db
    get_db = db_func

# Authentication functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    try:
        token = credentials.credentials
        logger.info(f"🔐 AUTH: Verifying token: {token[:20]}...")
        
        # Use the shared JWT decode function
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            logger.error("🔐 AUTH: Token payload missing 'sub' field")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        logger.info(f"🔐 AUTH: Token valid for user_id: {user_id}")
        
    except Exception as e:
        logger.error(f"🔐 AUTH: JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    # Get the database connection and verify user
    async for connection in get_db():
        user = await connection.fetchrow(
            "SELECT id, email, full_name, is_active, is_approved, is_admin FROM users WHERE id = $1",
            uuid.UUID(user_id)
        )
        
        if not user:
            logger.error(f"🔐 AUTH: User not found in database: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        if not user['is_active'] or not user['is_approved']:
            logger.error(f"🔐 AUTH: User not active or approved: {user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account not active or not approved"
            )
        
        logger.info(f"🔐 AUTH: User authenticated successfully: {user['email']}")
        return {
            "id": str(user['id']),
            "email": user['email'],
            "full_name": user['full_name'],
            "is_active": user['is_active'],
            "is_approved": user['is_approved'],
            "is_admin": user['is_admin']
        }

async def get_current_user_for_stream(request: FastAPIRequest):
    """Authenticates user for streaming endpoints using token from header or query."""
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
    
    try:
        logger.info(f"🔐 STREAM AUTH: Verifying token: {token[:20]}...")
        
        # Use the shared JWT decode function  
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            logger.error("Token payload missing 'sub' field")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, 
                detail="Invalid token payload"
            )
        
        logger.info(f"🔐 STREAM AUTH: Token valid for user_id: {user_id}")
        
    except Exception as e:
        logger.error(f"🔐 STREAM AUTH: JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid token"
        )

    try:
        # Get database connection and fetch user
        async for db in get_db():
            user = await db.fetchrow(
                "SELECT id, email, full_name, is_active, is_approved, is_admin FROM users WHERE id = $1", 
                uuid.UUID(user_id)
            )
            
            if not user:
                logger.error(f"User not found in database: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, 
                    detail="User not found"
                )
            
            if not user['is_active'] or not user['is_approved']:
                logger.error(f"User account not active or approved: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, 
                    detail="Account not active or not approved"
                )

            logger.info(f"🔐 STREAM AUTH: User authenticated successfully: {user['email']}")
            return {
                "id": str(user['id']), 
                "email": user['email'], 
                "full_name": user['full_name'], 
                "is_admin": user['is_admin']
            }
        
    except ValueError as e:
        logger.error(f"Invalid UUID format: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid user ID format"
        )
    except Exception as e:
        logger.error(f"Database error during user lookup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="Internal server error during authentication"
        )

# Explicit OPTIONS handlers to fix CORS preflight issues
@router.options("/login")
async def options_login():
    return JSONResponse(content={"message": "OK"})

@router.options("/register")
async def options_register():
    return JSONResponse(content={"message": "OK"})

@router.options("/validate")
async def options_validate():
    return JSONResponse(content={"message": "OK"})

@router.options("/query")
async def options_query():
    return JSONResponse(content={"message": "OK"})

@router.options("/query/stream")
async def options_stream():
    return JSONResponse(content={"message": "OK"})

@router.post("/register")
async def register_user(registration: UserRegistration):
    """Register a new user with password - SELF-SERVE FLOW (auto-approved)"""
    if not settings.APPLICATION.ENABLE_SELF_SERVE_REGISTRATION:
        raise HTTPException(status_code=403, detail="Self-serve registration is disabled by admin")
    async for db in get_db():
        # Check if user already exists
        existing_user = await db.fetchrow("SELECT id FROM users WHERE username = $1 OR email = $2", registration.username, registration.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username or email already registered")
        
        # Hash password
        hashed_password = hash_password(registration.password)
        
        # Insert new user (self-serve registration - auto-approved)
        user_id = await db.fetchval('''
            INSERT INTO users (username, email, full_name, company, role, hashed_password, is_active, is_approved, onboarded_via_invitation)
            VALUES ($1, $2, $3, $4, $5, $6, TRUE, TRUE, FALSE)
            RETURNING id
        ''', registration.username, registration.email, registration.full_name, registration.company, 
            registration.role, hashed_password)
        
        # Create default preferences
        await db.execute('INSERT INTO user_preferences (user_id) VALUES ($1)', user_id)
        
        log_info(f"👤 New user registered (self-serve): {registration.full_name} ({registration.username})")
        
        return {
            "message": "Registration successful! You can now login with your credentials.",
            "status": "approved",
            "registration_type": "self_serve",
            "user_id": str(user_id)
        }

@router.post("/login")
async def login_user(login_request: UserLogin):
    """Login with username and password - PRIMARY LOGIN METHOD"""
    if not settings.APPLICATION.ENABLE_LOGIN:
        raise HTTPException(status_code=403, detail="Password login is disabled by admin")
    async for db in get_db():
        user = await db.fetchrow('''
            SELECT id, username, email, full_name, hashed_password, is_active, is_approved, is_admin, has_completed_onboarding
            FROM users WHERE username = $1
        ''', login_request.username)
        
        if not user:
            raise HTTPException(status_code=400, detail="Invalid username or password")
        
        # Verify password
        if not user['hashed_password'] or not verify_password(login_request.password, user['hashed_password']):
            raise HTTPException(status_code=400, detail="Invalid username or password")
        
        if not user['is_approved']:
            raise HTTPException(status_code=403, detail="Account pending admin approval")
        
        if not user['is_active']:
            raise HTTPException(status_code=403, detail="Account is disabled")
        
        # Update last login and set first_login_at if it's the first time
        await db.execute('''
            UPDATE users SET 
                is_active = TRUE, 
                last_login = CURRENT_TIMESTAMP,
                first_login_at = COALESCE(first_login_at, CURRENT_TIMESTAMP)
            WHERE id = $1
        ''', user['id'])
        
        # Create access token using shared function
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user['id'])}, expires_delta=access_token_expires
        )
        
        logger.info(f"🔐 LOGIN: Generated token for {user['username']}: {access_token[:20]}...")
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "id": str(user['id']),
                "username": user['username'],
                "email": user['email'],
                "full_name": user['full_name'],
                "is_admin": user['is_admin'],
                "has_completed_onboarding": user.get('has_completed_onboarding', False)
            }
        }

@router.get("/validate")
async def validate_token(current_user: dict = Depends(get_current_user)):
    """
    Validates the user's token. If the token is valid, returns success.
    The get_current_user dependency handles all validation.
    """
    return {"status": "ok", "message": "Token is valid", "user_id": current_user["id"]}


@router.post("/forgot-password")
async def forgot_password(reset_request: PasswordReset):
    """Request password reset - DISABLED (magic token functionality removed)"""
    return {"message": "Password reset functionality is currently disabled. Please contact an administrator for assistance."}

@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: dict = Depends(get_current_user)
):
    """Change password for authenticated user"""
    async for db in get_db():
        user = await db.fetchrow(
            "SELECT id, hashed_password FROM users WHERE id = $1",
            uuid.UUID(current_user["id"])
        )
        
        if not user or not user['hashed_password']:
            raise HTTPException(status_code=400, detail="Current password not set")
        
        # Verify current password
        if not verify_password(password_change.current_password, user['hashed_password']):
            raise HTTPException(status_code=400, detail="Invalid current password")
        
        # Update password
        new_hashed_password = hash_password(password_change.new_password)
        await db.execute('''
            UPDATE users SET hashed_password = $1 WHERE id = $2
        ''', new_hashed_password, user['id'])
        
        return {"message": "Password changed successfully"}

@router.post("/magic-link")
async def send_magic_link(request: MagicLinkRequest):
    """Send a magic link to the user's email for passwordless authentication"""
    async for db in get_db():
        # Check if user exists
        user = await db.fetchrow("SELECT id, email, full_name, is_active, is_approved FROM users WHERE email = $1", request.email)
        
        if not user:
            # For security, don't reveal whether email exists or not
            return {"message": "If this email is registered, you will receive a magic link shortly."}
        
        if not user['is_active'] or not user['is_approved']:
            raise HTTPException(status_code=403, detail="Account not active or not approved")
        
        # Generate magic link token
        token = generate_magic_link_token(request.email)
        base_url = get_required_env_var("BASE_URL", "Base URL for magic links")
        magic_link = f"{base_url}/auth/magic-link/verify?token={token}"
        
        # Send email
        html_content = f"""
        <html>
        <body>
            <h2>Sign in to StrataLens</h2>
            <p>Hello {user['full_name']},</p>
            <p>Click the link below to sign in to your StrataLens account:</p>
            <p><a href="{magic_link}" style="background-color: #329ef6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px;">Sign In to StrataLens</a></p>
            <p>This link will expire in 15 minutes for security purposes.</p>
            <p>If you didn't request this, you can safely ignore this email.</p>
            <br>
            <p>Best regards,<br>The StrataLens Team</p>
        </body>
        </html>
        """
        
        text_content = f"""
        Sign in to StrataLens
        
        Hello {user['full_name']},
        
        Click the link below to sign in to your StrataLens account:
        {magic_link}
        
        This link will expire in 15 minutes for security purposes.
        If you didn't request this, you can safely ignore this email.
        
        Best regards,
        The StrataLens Team
        """
        
        email_sent = await send_email(
            request.email,
            "Sign in to StrataLens - Magic Link",
            html_content,
            text_content
        )
        
        if email_sent:
            log_info(f"🔗 Magic link sent to {request.email}")
            return {"message": "Magic link sent! Check your email and click the link to sign in."}
        else:
            raise HTTPException(status_code=500, detail="Failed to send magic link. Please try again later.")

@router.get("/magic-link/verify")
async def verify_magic_link(token: str):
    """Verify magic link token and authenticate user"""
    # Verify token
    email = verify_magic_link_token(token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired magic link")
    
    async for db in get_db():
        # Get user
        user = await db.fetchrow('''
            SELECT id, username, email, full_name, is_active, is_approved, is_admin, has_completed_onboarding
            FROM users WHERE email = $1
        ''', email)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user['is_active'] or not user['is_approved']:
            raise HTTPException(status_code=403, detail="Account not active or not approved")
        
        # Update last login
        await db.execute('''
            UPDATE users SET 
                last_login = CURRENT_TIMESTAMP,
                first_login_at = COALESCE(first_login_at, CURRENT_TIMESTAMP)
            WHERE id = $1
        ''', user['id'])
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user['id'])}, expires_delta=access_token_expires
        )
        
        log_info(f"🔗 Magic link authentication successful for {user['email']}")
        
        # Redirect to frontend with token
        frontend_url = get_required_env_var("FRONTEND_URL", "Frontend URL for redirects")
        return RedirectResponse(url=f"{frontend_url}?token={access_token}")

@router.get("/google")
async def google_login(request: FastAPIRequest):
    """Initiate Google OAuth flow"""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="Google authentication not configured")
    
    redirect_uri = GOOGLE_REDIRECT_URI or f"{get_required_env_var('BASE_URL')}/auth/google/callback"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/google/callback")
async def google_callback(request: FastAPIRequest):
    """Handle Google OAuth callback"""
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(status_code=501, detail="Google authentication not configured")
    
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user information from Google")
        
        email = user_info.get('email')
        name = user_info.get('name')
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Google")
        
        async for db in get_db():
            # Check if user exists
            user = await db.fetchrow("SELECT id, username, email, full_name, is_active, is_approved, is_admin, has_completed_onboarding FROM users WHERE email = $1", email)
            
            if user:
                # Existing user - log them in
                if not user['is_active'] or not user['is_approved']:
                    raise HTTPException(status_code=403, detail="Account not active or not approved")
                
                # Update last login
                await db.execute('''
                    UPDATE users SET 
                        last_login = CURRENT_TIMESTAMP,
                        first_login_at = COALESCE(first_login_at, CURRENT_TIMESTAMP)
                    WHERE id = $1
                ''', user['id'])
                
                user_id = user['id']
                log_info(f"🔍 Google OAuth login for existing user: {email}")
            
            else:
                # New user - create account
                username = email.split('@')[0]  # Use email prefix as username
                counter = 1
                original_username = username
                
                # Ensure username is unique
                while await db.fetchrow("SELECT id FROM users WHERE username = $1", username):
                    username = f"{original_username}{counter}"
                    counter += 1
                
                user_id = await db.fetchval('''
                    INSERT INTO users (username, email, full_name, is_active, is_approved, onboarded_via_invitation)
                    VALUES ($1, $2, $3, TRUE, TRUE, FALSE)
                    RETURNING id
                ''', username, email, name or email)
                
                # Create default preferences
                await db.execute('INSERT INTO user_preferences (user_id) VALUES ($1)', user_id)
                
                log_info(f"👤 New user created via Google OAuth: {name} ({email})")
            
            # Create access token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": str(user_id)}, expires_delta=access_token_expires
            )
            
            # Redirect to frontend with token
            frontend_url = get_required_env_var("FRONTEND_URL", "Frontend URL for redirects")
            return RedirectResponse(url=f"{frontend_url}?token={access_token}")
            
    except Exception as e:
        logger.error(f"Google OAuth error: {e}")
        raise HTTPException(status_code=400, detail="Google authentication failed")
