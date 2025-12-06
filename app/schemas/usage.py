from pydantic import BaseModel


class UsageStats(BaseModel):
    total_requests: int
    monthly_requests: int
    total_cost: float
    monthly_cost: float
    rate_limit_remaining: int
    monthly_limit_remaining: int


class RateLimitInfo(BaseModel):
    requests_remaining: int
    reset_time: str
    limit_type: str  # "minute" or "month"
