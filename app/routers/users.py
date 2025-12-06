"""
User management router for StrataLens API
Handles user profile, usage, and onboarding endpoints
"""

import uuid
import logging
import traceback
from datetime import date
from typing import Dict, Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer

# Import models
from app.schemas import (
    UserProfileResponse, UserProfileUpdate, OnboardingCompleteRequest, UsageStats
)

# Import dependencies
from app.auth.auth_utils import get_current_user
from db.db_utils import get_db
from app.utils.logging_utils import log_info, log_error
from app.utils.error_handlers import create_error_response, raise_sanitized_http_exception

# Import rate limiting and constants from main server
# These will be set by the main server
rate_limiter = None
RATE_LIMIT_PER_MINUTE = 3
RATE_LIMIT_PER_MONTH = 20
ADMIN_RATE_LIMIT_PER_MONTH = 100
COST_PER_REQUEST = 0.02

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/user", tags=["user"])

def set_user_globals(rate_limiter_instance, rate_limits, cost_per_request):
    """Set global variables from main server"""
    global rate_limiter, RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_MONTH, ADMIN_RATE_LIMIT_PER_MONTH, COST_PER_REQUEST
    rate_limiter = rate_limiter_instance
    RATE_LIMIT_PER_MINUTE = rate_limits['per_minute']
    RATE_LIMIT_PER_MONTH = rate_limits['per_month']
    ADMIN_RATE_LIMIT_PER_MONTH = rate_limits['admin_per_month']
    COST_PER_REQUEST = cost_per_request

@router.get("/usage", response_model=UsageStats)
async def get_user_usage(
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get current user's usage statistics"""
    try:
        user_id = uuid.UUID(current_user["id"])
        today = date.today()
        month_start = today.replace(day=1)
        is_admin = current_user.get("is_admin", False)
        
        logger.info(f"📊 Fetching usage stats for user {current_user['email']} (ID: {user_id}, admin: {is_admin})")
        
        # Get total requests
        total_requests = await db.fetchval('''
            SELECT COALESCE(SUM(request_count), 0) 
            FROM user_usage 
            WHERE user_id = $1
        ''', user_id)
        
        # Get monthly requests
        monthly_requests = await db.fetchval('''
            SELECT COALESCE(SUM(request_count), 0) 
            FROM user_usage 
            WHERE user_id = $1 AND request_date >= $2
        ''', user_id, month_start)
        
        # Calculate costs
        total_cost = total_requests * COST_PER_REQUEST
        monthly_cost = monthly_requests * COST_PER_REQUEST
        
        # Get rate limit info with admin status
        if rate_limiter:
            _, limit_info = rate_limiter.check_rate_limit(current_user["id"], is_admin)
        else:
            limit_info = {"minute_remaining": RATE_LIMIT_PER_MINUTE, "monthly_remaining": RATE_LIMIT_PER_MONTH}
        
        logger.info(f"📊 Usage stats for {current_user['email']} (admin: {is_admin}): Total={total_requests}, Monthly={monthly_requests}")
        
        return UsageStats(
            total_requests=total_requests,
            monthly_requests=monthly_requests,
            total_cost=round(total_cost, 4),
            monthly_cost=round(monthly_cost, 4),
            rate_limit_remaining=limit_info.get("minute_remaining", 0),
            monthly_limit_remaining=limit_info.get("monthly_remaining", 0)
        )
    except Exception as e:
        logger.error(f"❌ Error fetching usage stats for user {current_user.get('email', 'unknown')}: {e}")
        # Return default values if there's an error
        monthly_limit = ADMIN_RATE_LIMIT_PER_MONTH if current_user.get("is_admin", False) else RATE_LIMIT_PER_MONTH
        return UsageStats(
            total_requests=0,
            monthly_requests=0,
            total_cost=0.0,
            monthly_cost=0.0,
            rate_limit_remaining=RATE_LIMIT_PER_MINUTE,
            monthly_limit_remaining=monthly_limit
        )

@router.post("/complete-onboarding")
async def complete_onboarding(
    request: OnboardingCompleteRequest,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Mark user onboarding as complete"""
    try:
        log_info(f"🎓 Complete onboarding request - current_user_id: {current_user['id']}, request_user_id: {request.user_id}")
        
        # Verify user is updating their own onboarding status
        if current_user['id'] != request.user_id:
            raise HTTPException(status_code=403, detail="Can only update your own onboarding status")
        
        # Update onboarding status
        result = await db.execute('''
            UPDATE users SET has_completed_onboarding = TRUE
            WHERE id = $1
        ''', str(request.user_id))
        
        log_info(f"🎓 Update result: {result}")
        
        return {
            "success": True,
            "message": "Onboarding completed successfully"
        }
        
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            "onboarding completion", 
            current_user.get("id"),
            status_code=500
        )

@router.get("/onboarding-status")
async def get_onboarding_status(
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get user's onboarding status"""
    try:
        log_info(f"🎓 Getting onboarding status for user_id: {current_user['id']}")
        
        # Check if user has completed onboarding AND if they have made any queries
        user_data = await db.fetchrow('''
            SELECT 
                u.has_completed_onboarding,
                COUNT(qh.id) as query_count
            FROM users u
            LEFT JOIN query_history qh ON u.id = qh.user_id
            WHERE u.id = $1
            GROUP BY u.id, u.has_completed_onboarding
        ''', str(current_user['id']))
        
        log_info(f"🎓 Database result: {user_data}")
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        # SIMPLE LOGIC: Show onboarding if user hasn't made any queries yet
        should_show_onboarding = user_data['query_count'] == 0
        
        log_info(f"🎓 SIMPLE LOGIC: query_count = {user_data['query_count']}, should_show_onboarding = {should_show_onboarding}")
        
        return {
            "has_completed_onboarding": user_data['has_completed_onboarding'],
            "query_count": user_data['query_count'],
            "should_show_onboarding": should_show_onboarding
        }
        
    except Exception as e:
        raise_sanitized_http_exception(
            e, 
            "onboarding status fetch", 
            current_user.get("id"),
            status_code=500
        )

@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get current user's profile information"""
    user = await db.fetchrow('''
        SELECT id, username, email, full_name, first_name, last_name, company, role,
               organization, department, title, access_level, is_active, is_approved,
               is_admin, created_at, last_login, onboarded_via_invitation, has_completed_onboarding
        FROM users WHERE id = $1
    ''', uuid.UUID(current_user["id"]))
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": str(user['id']),
        "username": user['username'],
        "email": user['email'],
        "full_name": user['full_name'],
        "first_name": user['first_name'],
        "last_name": user['last_name'],
        "company": user['company'],
        "role": user['role'],
        "organization": user['organization'],
        "department": user['department'],
        "title": user['title'],
        "access_level": user['access_level'],
        "is_active": user['is_active'],
        "is_approved": user['is_approved'],
        "is_admin": user['is_admin'],
        "created_at": user['created_at'].isoformat(),
        "last_login": user['last_login'].isoformat() if user['last_login'] else None,
        "onboarded_via_invitation": user['onboarded_via_invitation'],
        "has_completed_onboarding": user['has_completed_onboarding']
    }

@router.put("/profile", response_model=UserProfileResponse)
async def update_user_profile(
    profile_update: UserProfileUpdate,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Update current user's profile information"""
    user_id = uuid.UUID(current_user["id"])
    
    # Check if username is being changed and if it's already taken
    if profile_update.username:
        existing_user = await db.fetchrow(
            "SELECT id FROM users WHERE username = $1 AND id != $2",
            profile_update.username, user_id
        )
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already taken")
    
    # Check if email is being changed and if it's already taken
    if profile_update.email:
        existing_user = await db.fetchrow(
            "SELECT id FROM users WHERE email = $1 AND id != $2",
            profile_update.email, user_id
        )
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already taken")
    
    # Build update query dynamically based on provided fields
    update_fields = []
    update_values = []
    param_count = 1
    
    if profile_update.username is not None:
        update_fields.append(f"username = ${param_count}")
        update_values.append(profile_update.username)
        param_count += 1
    
    if profile_update.email is not None:
        update_fields.append(f"email = ${param_count}")
        update_values.append(profile_update.email)
        param_count += 1
    
    if profile_update.full_name is not None:
        update_fields.append(f"full_name = ${param_count}")
        update_values.append(profile_update.full_name)
        param_count += 1
    
    if profile_update.company is not None:
        update_fields.append(f"company = ${param_count}")
        update_values.append(profile_update.company)
        param_count += 1
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    # Add user_id as the last parameter
    update_values.append(user_id)
    
    # Execute update
    update_query = f'''
        UPDATE users SET {', '.join(update_fields)}
        WHERE id = ${param_count}
        RETURNING id, username, email, full_name, first_name, last_name, company, role,
                  organization, department, title, access_level, is_active, is_approved,
                  is_admin, created_at, last_login, onboarded_via_invitation, has_completed_onboarding
    '''
    
    user = await db.fetchrow(update_query, *update_values)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": str(user['id']),
        "username": user['username'],
        "email": user['email'],
        "full_name": user['full_name'],
        "first_name": user['first_name'],
        "last_name": user['last_name'],
        "company": user['company'],
        "role": user['role'],
        "organization": user['organization'],
        "department": user['department'],
        "title": user['title'],
        "access_level": user['access_level'],
        "is_active": user['is_active'],
        "is_approved": user['is_approved'],
        "is_admin": user['is_admin'],
        "created_at": user['created_at'].isoformat(),
        "last_login": user['last_login'].isoformat() if user['last_login'] else None,
        "onboarded_via_invitation": user['onboarded_via_invitation'],
        "has_completed_onboarding": user['has_completed_onboarding']
    }
