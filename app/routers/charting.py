"""
Charting Router for Financial Data Visualization

This router handles all charting-related endpoints for financial data visualization,
including time series data, multi-company comparisons, and segment analysis.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, List
import time
import pandas as pd
import logging
import traceback

# Import PostgreSQL connection utilities
import psycopg2
from psycopg2.extras import RealDictCursor

# Import centralized utilities
from app.auth.auth_utils import get_current_user
from db.db_utils import get_db
from app.utils.error_handlers import create_error_response, raise_sanitized_http_exception

# Set up logging
logger = logging.getLogger(__name__)

# Shared utilities
from db.db_connection_utils import get_postgres_connection

# Create router
router = APIRouter(prefix="/charting", tags=["charting"])


@router.post("/multi-company-time-series")
async def get_multi_company_time_series(
    request: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Get time series data for multiple companies - optimized for charting."""
    start_time = time.time()
    
    try:
        # Parse request
        symbols = request.get('symbols', [])
        metric = request.get('metric')
        metric_type = request.get('metric_type', 'absolute')  # Default to absolute values
        statement_type = request.get('statement_type')
        years = min(request.get('years', 20), 50)  # Cap at 50 years max
        
        logger.info(f"📊 CHARTING REQUEST: symbols={symbols}, metric={metric}, metric_type={metric_type}, statement_type={statement_type}, years={years}")
        
        # Log the metric type being processed
        if metric_type == 'growth':
            logger.info(f"📊 PROCESSING GROWTH RATE CALCULATION for metric: {metric}")
        else:
            logger.info(f"📊 PROCESSING ABSOLUTE VALUES for metric: {metric}")
        
        if not symbols or not metric or not statement_type:
            raise HTTPException(status_code=400, detail="Missing required parameters: symbols, metric, statement_type")
        
        # Map statement types to table names
        table_map = {
            'income': 'income_statements',
            'balance': 'balance_sheets', 
            'cashflow': 'cash_flow_statements'
        }
        
        table_name = table_map.get(statement_type)
        if not table_name:
            raise HTTPException(status_code=400, detail=f"Invalid statement type: {statement_type}")
        
        # Create SQL with multiple symbols
        symbols_upper = [s.upper() for s in symbols]
        symbols_placeholders = ','.join(['%s'] * len(symbols_upper))
        
        logger.info(f"📊 Using table: {table_name}, metric: {metric}")
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # First, let's check what data is available for each company
                availability_query = f"""
                SELECT 
                    symbol,
                    COUNT(*) as total_records,
                    MIN(calendaryear) as earliest_year,
                    MAX(calendaryear) as latest_year,
                    COUNT(CASE WHEN {metric} IS NOT NULL THEN 1 END) as records_with_metric
                FROM financial_data.{table_name}
                WHERE UPPER(symbol) IN ({symbols_placeholders})
                AND period = 'FY'
                GROUP BY symbol
                ORDER BY symbol
                """
                
                cursor.execute(availability_query, symbols_upper)
                availability_results = cursor.fetchall()
                logger.info(f"📊 DATA AVAILABILITY CHECK:")
                for row in availability_results:
                    logger.info(f"📊   {row['symbol']}: {row['records_with_metric']}/{row['total_records']} records with {metric}, years {row['earliest_year']}-{row['latest_year']}")
                
                # Build SQL query based on metric type
                if metric_type == 'growth':
                    # For growth rates, calculate year-over-year growth
                    sql_query = f"""
                    WITH latest_per_year AS (
                        SELECT *,
                            ROW_NUMBER() OVER (PARTITION BY symbol, calendarYear ORDER BY date DESC) as rn
                        FROM financial_data.{table_name}
                        WHERE UPPER(symbol) IN ({symbols_placeholders})
                        AND period = 'FY'
                        AND {metric} IS NOT NULL
                    ),
                    unique_data AS (
                        SELECT symbol, calendarYear, {metric}
                        FROM latest_per_year
                        WHERE rn = 1
                    ),
                    growth_calculation AS (
                        SELECT 
                            symbol,
                            calendarYear,
                            {metric} as current_value,
                            LAG({metric}) OVER (PARTITION BY symbol ORDER BY calendarYear) as previous_value,
                            CASE 
                                WHEN LAG({metric}) OVER (PARTITION BY symbol ORDER BY calendarYear) IS NOT NULL 
                                AND LAG({metric}) OVER (PARTITION BY symbol ORDER BY calendarYear) != 0
                                THEN ROUND(CAST((({metric} - LAG({metric}) OVER (PARTITION BY symbol ORDER BY calendarYear)) / LAG({metric}) OVER (PARTITION BY symbol ORDER BY calendarYear)) * 100 AS NUMERIC), 2)
                                ELSE NULL 
                            END as growth_rate
                        FROM unique_data
                    ),
                    ranked_growth AS (
                        SELECT 
                            symbol,
                            calendarYear as year,
                            current_value as value,
                            growth_rate,
                            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY calendarYear DESC) as company_rank
                        FROM growth_calculation
                        WHERE growth_rate IS NOT NULL
                    )
                    SELECT 
                        symbol,
                        year,
                        value,
                        growth_rate
                    FROM ranked_growth
                    WHERE company_rank <= %s
                    ORDER BY symbol, year DESC
                    """
                    cursor.execute(sql_query, symbols_upper + [years])
                else:
                    # For absolute values, just get the latest data per year
                    sql_query = f"""
                    WITH latest_per_year AS (
                        SELECT *,
                            ROW_NUMBER() OVER (PARTITION BY symbol, calendarYear ORDER BY date DESC) as rn
                        FROM financial_data.{table_name}
                        WHERE UPPER(symbol) IN ({symbols_placeholders})
                        AND period = 'FY'
                        AND {metric} IS NOT NULL
                    ),
                    ranked_data AS (
                        SELECT 
                            symbol,
                            calendarYear as year,
                            {metric} as value,
                            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY calendarYear DESC) as company_rank
                        FROM latest_per_year
                        WHERE rn = 1
                    )
                    SELECT 
                        symbol,
                        year,
                        value
                    FROM ranked_data
                    WHERE company_rank <= %s
                    ORDER BY symbol, year DESC
                    """
                    cursor.execute(sql_query, symbols_upper + [years])
                
                results = cursor.fetchall()
                execution_time = time.time() - start_time
                
                # Group data by company
                companies_data = {}
                for row in results:
                    symbol = row['symbol']
                    if symbol not in companies_data:
                        companies_data[symbol] = []
                    
                    data_point = {
                        'year': row['year'],
                        'value': row['value']
                    }
                    
                    if metric_type == 'growth' and 'growth_rate' in row:
                        data_point['growth_rate'] = row['growth_rate']
                    
                    companies_data[symbol].append(data_point)
                
                # Sort each company's data by year ascending
                for symbol in companies_data:
                    companies_data[symbol].sort(key=lambda x: x['year'])
                
                # Convert companies_data object to array format expected by frontend
                companies_array = []
                for symbol, data in companies_data.items():
                    companies_array.append({
                        "symbol": symbol,
                        "data": data
                    })
                
                return {
                    "success": True,
                    "symbols": symbols_upper,
                    "metric": metric,
                    "metric_type": metric_type,
                    "statement_type": statement_type,
                    "years_requested": years,
                    "companies": companies_array,
                    "execution_time": execution_time,
                    "message": f"Multi-company time series data retrieved for {len(symbols)} companies"
                }
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            "multi-company time series data retrieval", 
            current_user.get("id"),
            status_code=500
        )

