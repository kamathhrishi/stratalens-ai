"""Data models for the RAG system"""
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class QuestionAnalysisResult(BaseModel):
    """Pydantic model for strict validation of question analysis results"""

    is_valid: bool = Field(description="Whether the question is appropriate for earnings transcripts")
    reason: str = Field(description="Explanation of why it's valid/invalid")
    question_type: str = Field(description="Type of question")
    extracted_ticker: Optional[str] = Field(description="First/primary company ticker mentioned")
    extracted_tickers: List[str] = Field(description="All company tickers mentioned")
    rephrased_question: str = Field(description="Improved version for better retrieval")
    suggested_improvements: List[str] = Field(description="Suggestions for better questions")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the analysis (0.0 to 1.0)")
    quarter_reference: Optional[str] = Field(description="Detected quarter reference in database format")
    quarter_context: str = Field(description="Context about quarter timing")
    quarter_count: Optional[int] = Field(description="Number of quarters requested")
    data_source: str = Field(default="earnings_transcripts", description="Primary data source: '10k', 'latest_news', 'earnings_transcripts', or 'hybrid'")
    needs_latest_news: bool = Field(default=False, description="Whether the question requires latest news search")
    needs_10k: bool = Field(default=False, description="Whether the question requires 10-K SEC filings")

    @validator('question_type')
    def validate_question_type(cls, v):
        valid_types = [
            "specific_company", "multiple_companies", "general_market",
            "financial_metrics", "guidance", "challenges", "outlook",
            "industry_analysis", "executive_leadership", "business_strategy",
            "company_info", "latest_news", "invalid"
        ]
        if v not in valid_types:
            raise ValueError(f"question_type must be one of {valid_types}")
        return v

    @validator('quarter_context')
    def validate_quarter_context(cls, v):
        valid_contexts = ["latest", "previous", "specific", "multiple"]
        if v not in valid_contexts:
            raise ValueError(f"quarter_context must be one of {valid_contexts}")
        return v
    
    @validator('data_source')
    def validate_data_source(cls, v):
        valid_sources = ["10k", "latest_news", "earnings_transcripts", "hybrid"]
        if v not in valid_sources:
            raise ValueError(f"data_source must be one of {valid_sources}")
        return v
