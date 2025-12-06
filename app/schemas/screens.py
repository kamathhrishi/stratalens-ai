"""Saved screens schemas for StrataLens AI"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class SaveScreenRequest(BaseModel):
    """Request to save query results as a screen"""
    screen_name: str = Field(..., min_length=1, max_length=100, description="User-defined name for the screen")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    query: str = Field(..., description="Original query that generated this screen")
    query_type: str = Field(..., description="Type of query: single_sheet or multi_sheet")

    # Single sheet data
    columns: Optional[List[str]] = None
    friendly_columns: Optional[Dict[str, str]] = None
    data_rows: Optional[List[Dict[str, Any]]] = None

    # Multi-sheet data
    sheets: Optional[List[Dict[str, Any]]] = None
    companies: Optional[List[str]] = None

    # Metadata
    tables_used: Optional[List[str]] = Field(default=[])
    total_rows: int = Field(default=0)


class SavedScreen(BaseModel):
    """Saved screen metadata"""
    id: str
    screen_name: str
    description: Optional[str]
    query: str
    query_type: str
    total_rows: int
    tables_used: List[str]
    companies: Optional[List[str]]
    created_at: str
    updated_at: str


class ScreenData(BaseModel):
    """Complete screen data with content"""
    screen: SavedScreen
    columns: Optional[List[str]] = None
    friendly_columns: Optional[Dict[str, str]] = None
    data_rows: Optional[List[Dict[str, Any]]] = None
    sheets: Optional[List[Dict[str, Any]]] = None


class UpdateScreenRequest(BaseModel):
    """Request to update screen metadata"""
    screen_name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
