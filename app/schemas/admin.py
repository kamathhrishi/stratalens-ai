from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any


class AdminUserApproval(BaseModel):
    user_id: str
    approved: bool
    notes: Optional[str] = None


class AdminUsersList(BaseModel):
    users: List[Dict[str, Any]]
    total_count: int
    pending_approvals: int


class CSVUserRequest(BaseModel):
    username: str = Field(..., description="User username")
    email: Optional[str] = Field(None, description="User email")
    password: str = Field(..., min_length=8, description="User password")
    full_name: str = Field(..., description="User full name")
    company: Optional[str] = Field(None, description="User company")
    role: Optional[str] = Field(None, description="User role")
    is_admin: bool = Field(default=False, description="Whether user is admin")


class OnboardingCompleteRequest(BaseModel):
    user_id: str = Field(..., description="User ID to mark onboarding as complete")
