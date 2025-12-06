from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class QueryHistoryItem(BaseModel):
    id: str
    question: str
    sql_query_generated: Optional[str]
    success: bool
    execution_time: Optional[float]
    error_message: Optional[str]
    tables_used: List[str]
    result_count: Optional[int]
    used_cache: bool
    created_at: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question about financial data (max 1000 characters)")
    page: int = Field(default=1, ge=1, description="Page number for pagination")
    page_size: Optional[int] = Field(default=None, ge=1, le=1000, description="Number of records per page")
    save_to_history: bool = Field(default=True, description="Whether to save this query to user history")


class QueryResponse(BaseModel):
    success: bool
    execution_time: float
    sql_query_generated: Optional[str] = None
    columns: Optional[List[str]] = None
    friendly_columns: Optional[Dict[str, str]] = None
    data_rows: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    tables_used: Optional[List[str]] = None
    pagination_info: Optional[Dict] = None
    used_stored_results: Optional[bool] = False
    used_pipeline_cache: Optional[bool] = False
    query_id: Optional[str] = None


class ExportRequest(BaseModel):
    sheet_ids: Optional[List[str]] = None  # If None, export all
    format: str = Field(default="csv", pattern="^(csv|excel)$")
    include_metadata: bool = Field(default=True)


class SortRequest(BaseModel):
    column: str = Field(..., description="Column name to sort by")
    direction: str = Field(..., pattern="^(asc|desc)$", description="Sort direction: asc or desc")
    query_id: Optional[str] = Field(None, description="Query ID for cached results")


class SortResponse(BaseModel):
    success: bool
    data_rows: List[Dict[str, Any]]
    columns: List[str]
    friendly_columns: Dict[str, str]
    message: str
    sort_applied: Dict[str, str]  # column and direction
    total_rows: int
    pagination_info: Optional[Dict[str, Any]] = None


class ExpandValueRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Original question to find cached data (max 1000 characters)")
    row_index: int = Field(..., ge=0, description="Row index in the cached data")
    column_name: str = Field(..., description="Column name to get the full value for")


class ExpandValueResponse(BaseModel):
    success: bool
    full_value: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


class ExpandScreenValueRequest(BaseModel):
    screen_id: str = Field(..., description="Screen ID to get data from")
    row_index: int = Field(..., ge=0, description="Row index in the screen data")
    column_name: str = Field(..., description="Column name to get the full value for")


class PaginationRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Original question to find cached data (max 1000 characters)")
    page: int = Field(default=1, ge=1, description="Page number for pagination")
    page_size: Optional[int] = Field(default=None, ge=1, le=1000, description="Number of records per page")
