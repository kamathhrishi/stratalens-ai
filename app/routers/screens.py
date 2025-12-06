"""
Saved Screens Management Router
Handles saving query results as reusable screens that users can access later
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import asyncpg
import uuid
import json
from datetime import datetime
import logging

# Import centralized utilities
from app.auth.auth_utils import get_current_user
from db.db_utils import get_db
from app.utils import create_error_response, raise_sanitized_http_exception
from app.utils.logging_utils import log_message, log_error, log_warning, log_info, log_debug, log_milestone

# Import schemas from centralized location
from app.schemas import ExpandValueResponse, ExpandScreenValueRequest
from app.schemas.screens import SaveScreenRequest, SavedScreen, ScreenData, UpdateScreenRequest

logger = logging.getLogger(__name__)

# Database Utilities
async def init_screens_tables(db_pool):
    """Initialize the saved screens table"""
    async with db_pool.acquire() as conn:
        # Create saved_screens table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS saved_screens (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                screen_name VARCHAR(100) NOT NULL,
                description TEXT,
                query TEXT NOT NULL,
                query_type VARCHAR(20) NOT NULL,
                
                -- Single sheet data
                columns JSONB,
                friendly_columns JSONB,
                data_rows JSONB,
                
                -- Multi-sheet data  
                sheets JSONB,
                companies JSONB DEFAULT '[]',
                
                -- Metadata
                tables_used JSONB DEFAULT '[]',
                total_rows INTEGER DEFAULT 0,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Ensure unique screen names per user
                UNIQUE(user_id, screen_name)
            )
        ''')
        
        # Create indexes for better performance
        try:
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_saved_screens_user_id ON saved_screens(user_id)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_saved_screens_created_at ON saved_screens(created_at)')
            await conn.execute('CREATE INDEX IF NOT EXISTS idx_saved_screens_query_type ON saved_screens(query_type)')
        except Exception as e:
            logger.warning(f"Could not create some indexes: {e}")

# Screen Management Functions
async def save_screen_to_db(
    user_id: str, 
    screen_request: SaveScreenRequest, 
    db: asyncpg.Connection
) -> str:
    """Save a screen to the database"""
    try:
        # Check if screen name already exists for this user
        existing = await db.fetchrow(
            "SELECT id FROM saved_screens WHERE user_id = $1 AND screen_name = $2",
            uuid.UUID(user_id), screen_request.screen_name
        )
        
        if existing:
            raise HTTPException(
                status_code=409, 
                detail=f"Screen name '{screen_request.screen_name}' already exists. Please choose a different name."
            )
        
        # Insert new screen
        screen_id = await db.fetchval('''
            INSERT INTO saved_screens 
            (user_id, screen_name, description, query, query_type, columns, 
             friendly_columns, data_rows, sheets, companies, tables_used, total_rows)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            RETURNING id
        ''',
            uuid.UUID(user_id),
            screen_request.screen_name,
            screen_request.description,
            screen_request.query,
            screen_request.query_type,
            json.dumps(screen_request.columns) if screen_request.columns else None,
            json.dumps(screen_request.friendly_columns) if screen_request.friendly_columns else None,
            json.dumps(screen_request.data_rows) if screen_request.data_rows else None,
            json.dumps(screen_request.sheets) if screen_request.sheets else None,
            json.dumps(screen_request.companies) if screen_request.companies else None,
            json.dumps(screen_request.tables_used),
            screen_request.total_rows
        )
        
        logger.info(f"✅ Screen '{screen_request.screen_name}' saved for user {user_id}")
        return str(screen_id)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to save screen: {e}")
        raise HTTPException(status_code=500, detail="Failed to save screen")

async def get_user_screens(
    user_id: str, 
    db: asyncpg.Connection,
    page: int = 1,
    limit: int = 20,
    query_type_filter: Optional[str] = None
) -> Dict[str, Any]:
    """Get all saved screens for a user"""
    try:
        offset = (page - 1) * limit
        
        # Build query conditions
        where_conditions = ["user_id = $1"]
        params = [uuid.UUID(user_id)]
        
        if query_type_filter:
            where_conditions.append("query_type = $2")
            params.append(query_type_filter)
        
        where_clause = " AND ".join(where_conditions)
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM saved_screens WHERE {where_clause}"
        total_count = await db.fetchval(count_query, *params)
        
        # Get screens
        screens_query = f'''
            SELECT id, screen_name, description, query, query_type, 
                   total_rows, tables_used, companies, created_at, updated_at
            FROM saved_screens 
            WHERE {where_clause}
            ORDER BY updated_at DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        '''
        params.extend([limit, offset])
        
        screens = await db.fetch(screens_query, *params)
        
        return {
            "screens": [
                SavedScreen(
                    id=str(screen['id']),
                    screen_name=screen['screen_name'],
                    description=screen['description'],
                    query=screen['query'],
                    query_type=screen['query_type'],
                    total_rows=screen['total_rows'],
                    tables_used=json.loads(screen['tables_used']) if screen['tables_used'] else [],
                    companies=json.loads(screen['companies']) if screen['companies'] else None,
                    created_at=screen['created_at'].isoformat(),
                    updated_at=screen['updated_at'].isoformat()
                )
                for screen in screens
            ],
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total_count,
                "pages": (total_count + limit - 1) // limit
            },
            "summary": {
                "total_screens": total_count,
                "single_sheet_screens": await db.fetchval(
                    "SELECT COUNT(*) FROM saved_screens WHERE user_id = $1 AND query_type = 'single_sheet'",
                    uuid.UUID(user_id)
                ),
                "multi_sheet_screens": await db.fetchval(
                    "SELECT COUNT(*) FROM saved_screens WHERE user_id = $1 AND query_type = 'multi_sheet'",
                    uuid.UUID(user_id)
                )
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get user screens: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve screens")

async def get_screen_data(
    screen_id: str, 
    user_id: str, 
    db: asyncpg.Connection
) -> ScreenData:
    """Get full screen data including the saved query results"""
    try:
        screen_uuid = uuid.UUID(screen_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    
    try:
        screen = await db.fetchrow('''
            SELECT * FROM saved_screens 
            WHERE id = $1 AND user_id = $2
        ''', screen_uuid, user_uuid)
        
        if not screen:
            raise HTTPException(status_code=404, detail="Screen not found or access denied")
        
        return ScreenData(
            screen=SavedScreen(
                id=str(screen['id']),
                screen_name=screen['screen_name'],
                description=screen['description'],
                query=screen['query'],
                query_type=screen['query_type'],
                total_rows=screen['total_rows'],
                tables_used=json.loads(screen['tables_used']) if screen['tables_used'] else [],
                companies=json.loads(screen['companies']) if screen['companies'] else None,
                created_at=screen['created_at'].isoformat(),
                updated_at=screen['updated_at'].isoformat()
            ),
            columns=json.loads(screen['columns']) if screen['columns'] else None,
            friendly_columns=json.loads(screen['friendly_columns']) if screen['friendly_columns'] else None,
            data_rows=json.loads(screen['data_rows']) if screen['data_rows'] else None,
            sheets=json.loads(screen['sheets']) if screen['sheets'] else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get screen data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve screen data")

async def update_screen(
    screen_id: str,
    user_id: str,
    update_request: UpdateScreenRequest,
    db: asyncpg.Connection
) -> SavedScreen:
    """Update a saved screen's metadata"""
    try:
        screen_uuid = uuid.UUID(screen_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    
    try:
        # Check if screen exists and belongs to user
        existing = await db.fetchrow(
            "SELECT id FROM saved_screens WHERE id = $1 AND user_id = $2",
            screen_uuid, user_uuid
        )
        
        if not existing:
            raise HTTPException(status_code=404, detail="Screen not found or access denied")
        
        # Check for name conflicts if updating name
        if update_request.screen_name:
            name_conflict = await db.fetchrow(
                "SELECT id FROM saved_screens WHERE user_id = $1 AND screen_name = $2 AND id != $3",
                user_uuid, update_request.screen_name, screen_uuid
            )
            
            if name_conflict:
                raise HTTPException(
                    status_code=409,
                    detail=f"Screen name '{update_request.screen_name}' already exists"
                )
        
        # Build update query
        updates = []
        params = []
        param_count = 1
        
        if update_request.screen_name is not None:
            updates.append(f"screen_name = ${param_count}")
            params.append(update_request.screen_name)
            param_count += 1
            
        if update_request.description is not None:
            updates.append(f"description = ${param_count}")
            params.append(update_request.description)
            param_count += 1
        
        if not updates:
            raise HTTPException(status_code=400, detail="No updates provided")
        
        updates.append(f"updated_at = CURRENT_TIMESTAMP")
        params.extend([screen_uuid, user_uuid])
        
        # Execute update
        await db.execute(f'''
            UPDATE saved_screens 
            SET {", ".join(updates)}
            WHERE id = ${param_count} AND user_id = ${param_count + 1}
        ''', *params)
        
        # Return updated screen
        updated_screen = await db.fetchrow('''
            SELECT id, screen_name, description, query, query_type, 
                   total_rows, tables_used, companies, created_at, updated_at
            FROM saved_screens 
            WHERE id = $1
        ''', screen_uuid)
        
        return SavedScreen(
            id=str(updated_screen['id']),
            screen_name=updated_screen['screen_name'],
            description=updated_screen['description'],
            query=updated_screen['query'],
            query_type=updated_screen['query_type'],
            total_rows=updated_screen['total_rows'],
            tables_used=json.loads(updated_screen['tables_used']) if updated_screen['tables_used'] else [],
            companies=json.loads(updated_screen['companies']) if updated_screen['companies'] else None,
            created_at=updated_screen['created_at'].isoformat(),
            updated_at=updated_screen['updated_at'].isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to update screen: {e}")
        raise HTTPException(status_code=500, detail="Failed to update screen")

async def delete_screen(
    screen_id: str,
    user_id: str,
    db: asyncpg.Connection
) -> bool:
    """Delete a saved screen"""
    try:
        screen_uuid = uuid.UUID(screen_id)
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")
    
    try:
        result = await db.execute(
            "DELETE FROM saved_screens WHERE id = $1 AND user_id = $2",
            screen_uuid, user_uuid
        )
        
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Screen not found or access denied")
        
        logger.info(f"✅ Screen {screen_id} deleted for user {user_id}")
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete screen: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete screen")

# Create router
router = APIRouter(prefix="/screens", tags=["screens"])

@router.post("/save", response_model=Dict[str, Any])
async def save_screen(
    screen_request: SaveScreenRequest,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Save current query results as a reusable screen"""
    screen_id = await save_screen_to_db(current_user["id"], screen_request, db)
    
    return {
        "success": True,
        "screen_id": screen_id,
        "message": f"Screen '{screen_request.screen_name}' saved successfully!"
    }

@router.get("/list")
async def list_screens(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=50),
    query_type: Optional[str] = Query(default=None, description="Filter by query_type"),
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get user's saved screens"""
    return await get_user_screens(
        current_user["id"], 
        db, 
        page=page, 
        limit=limit, 
        query_type_filter=query_type
    )

@router.get("/{screen_id}", response_model=ScreenData)
async def get_screen(
    screen_id: str,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get a specific saved screen with all its data"""
    return await get_screen_data(screen_id, current_user["id"], db)

@router.put("/{screen_id}", response_model=SavedScreen)
async def update_screen_metadata(
    screen_id: str,
    update_request: UpdateScreenRequest,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Update screen name and description"""
    return await update_screen(screen_id, current_user["id"], update_request, db)

@router.delete("/{screen_id}")
async def delete_saved_screen(
    screen_id: str,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Delete a saved screen"""
    await delete_screen(screen_id, current_user["id"], db)
    return {"success": True, "message": "Screen deleted successfully"}

@router.post("/expand-value", response_model=ExpandValueResponse)
async def expand_screen_truncated_value(
    expand_request: ExpandScreenValueRequest,
    current_user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db)
):
    """Get the full value for a truncated cell in a saved screen"""
    try:
        # Validate screen ID format
        try:
            screen_uuid = uuid.UUID(expand_request.screen_id)
        except ValueError:
            return ExpandValueResponse(
                success=False,
                error="Invalid screen ID format"
            )
        
        # Get the saved screen data
        screen = await db.fetchrow('''
            SELECT data_rows 
            FROM saved_screens 
            WHERE id = $1 AND user_id = $2
        ''', screen_uuid, uuid.UUID(current_user["id"]))
        
        if not screen:
            return ExpandValueResponse(
                success=False,
                error="Screen not found or access denied"
            )
        
        # Extract the data
        data_rows = json.loads(screen['data_rows']) if screen['data_rows'] else []
        
        # Validate row index
        if expand_request.row_index >= len(data_rows):
            return ExpandValueResponse(
                success=False,
                error=f"Row index {expand_request.row_index} is out of range"
            )
        
        # Get the row data
        row_data = data_rows[expand_request.row_index]
        
        # Get the full value for the specified column
        if expand_request.column_name not in row_data:
            return ExpandValueResponse(
                success=False,
                error=f"Column '{expand_request.column_name}' not found in the data"
            )
        
        full_value = row_data[expand_request.column_name]
        
        # Convert to string if needed
        if full_value is None:
            full_value = ""
        else:
            full_value = str(full_value)
        
        return ExpandValueResponse(
            success=True,
            full_value=full_value,
            message="Full value retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error expanding screen value: {e}")
        error_response = create_error_response(e, "screen value retrieval", current_user.get("id"))
        return ExpandValueResponse(
            success=False,
            error=error_response["error"]
        )
