from pydantic import BaseModel, Field
from typing import Optional, List


class CompanySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100, description="Search term for company name or ticker")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum number of results")


class CompanySearchResult(BaseModel):
    symbol: str
    companyName: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    marketCap: Optional[float] = None
    country: Optional[str] = None
    exchangeShortName: Optional[str] = None
    description: Optional[str] = None


class CompanySearchResponse(BaseModel):
    success: bool
    companies: List[CompanySearchResult]
    total_found: int
    query: str
    execution_time: float
    message: str
