from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List


class UserProfile(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    full_name: str
    first_name: Optional[str]
    last_name: Optional[str]
    company: Optional[str]
    role: Optional[str]
    organization: Optional[str]
    department: Optional[str]
    title: Optional[str]
    access_level: str
    is_active: bool
    is_approved: bool
    is_admin: bool
    created_at: str
    last_login: Optional[str]
    onboarded_via_invitation: bool


class UserPreferences(BaseModel):
    default_page_size: int = Field(default=20, ge=5, le=100)
    preferred_sectors: List[str] = Field(default=[])
    email_notifications: bool = Field(default=True)
    theme: str = Field(default='light', pattern='^(light|dark)$')


class UserProfileUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50, description="Username (min 3 characters)")
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    company: Optional[str] = Field(None, max_length=100)


class UserProfileResponse(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    full_name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    organization: Optional[str] = None
    department: Optional[str] = None
    title: Optional[str] = None
    access_level: str
    is_active: bool
    is_approved: bool
    is_admin: bool
    created_at: str
    last_login: Optional[str] = None
    onboarded_via_invitation: bool
    has_completed_onboarding: bool
