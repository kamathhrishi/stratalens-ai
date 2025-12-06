"""Authentication schemas for StrataLens AI"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict


class UserLogin(BaseModel):
    """User login request schema"""
    username: str = Field(..., min_length=1, max_length=50, description="Username")
    password: str = Field(..., min_length=1)


class UserRegistration(BaseModel):
    """User registration request schema"""
    username: str = Field(..., min_length=3, max_length=50, description="Username (min 3 characters)")
    email: Optional[EmailStr] = None
    full_name: str = Field(..., min_length=2, max_length=100)
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 characters)")
    company: Optional[str] = Field(None, max_length=100)
    role: Optional[str] = Field(None, max_length=50)
    reason_for_access: Optional[str] = Field(None, max_length=500)


class OnboardingCompletion(BaseModel):
    """Complete onboarding with invitation code"""
    invitation_code: str
    password: str = Field(..., min_length=8, max_length=128, description="Create your password")
    terms_accepted: bool = True
    preferences: Optional[Dict] = None


class PasswordReset(BaseModel):
    """Password reset request schema"""
    email: Optional[EmailStr] = None


class PasswordChange(BaseModel):
    """Password change request schema"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)


class MagicLinkRequest(BaseModel):
    """Magic link authentication request"""
    email: EmailStr = Field(..., description="Email address to send magic link to")


class MagicLinkVerify(BaseModel):
    """Magic link verification"""
    token: str = Field(..., description="Magic link token")
