"""
Financial statements endpoints for companies router
"""
import time
import traceback
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends

# Global variables that will be injected
ANALYZER_AVAILABLE = False
analyzer_instance = None

def set_analyzer_globals(available: bool, instance=None):
    """Set the analyzer globals from the main server"""
    global ANALYZER_AVAILABLE, analyzer_instance
    ANALYZER_AVAILABLE = available
    analyzer_instance = instance

# Import centralized utilities
from app.auth.auth_utils import get_current_user
from db.db_utils import get_db
from app.utils import create_error_response, raise_sanitized_http_exception
from app.utils.logging_utils import logger

def get_analyzer():
    """Get the analyzer instance"""
    if not ANALYZER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Query features disabled - analyzer not available")
    if analyzer_instance is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    return analyzer_instance

# Shared utilities
from db.db_connection_utils import get_postgres_connection
from db.data_transformation_utils import convert_postgres_to_camelcase

# Create router
router = APIRouter(prefix="/companies", tags=["companies-financials"])

@router.get("/{symbol}/income-statement")
async def get_company_income_statement(
    symbol: str,
    years: int = Query(default=5, ge=1, le=10, description="Number of years of data"),
    current_user: dict = Depends(get_current_user)
):
    """Get income statement data for a company."""
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get income statement data for the specified number of years
                sql_query = """
                WITH latest_per_year AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY calendaryear ORDER BY date DESC) as rn
                    FROM financial_data.income_statements
                    WHERE UPPER(symbol) = %s 
                    AND period = 'FY'
                )
                SELECT 
                    symbol,
                    date,
                    calendaryear,
                    period,
                    revenue,
                    costofrevenue,
                    grossprofit,
                    grossprofitratio,
                    researchanddevelopmentexpenses,
                    generalandadministrativeexpenses,
                    sellingandmarketingexpenses,
                    operatingexpenses,
                    costandexpenses,
                    interestexpense,
                    depreciationandamortization,
                    ebitda,
                    ebitdaratio,
                    operatingincome,
                    operatingincomeratio,
                    totalotherincomeexpensesnet,
                    incomebeforetax,
                    incomebeforetaxratio,
                    incometaxexpense,
                    netincome,
                    netincomeratio,
                    eps,
                    epsdiluted,
                    weightedaverageshsout,
                    weightedaverageshsoutdil
                FROM latest_per_year
                WHERE rn = 1
                ORDER BY calendaryear DESC
                LIMIT %s
                """
                
                cursor.execute(sql_query, (symbol_upper, years))
                results = cursor.fetchall()
                execution_time = time.time() - start_time
                
                # Convert results to list of dicts with camelCase conversion
                statements = []
                for row in results:
                    statement_dict = convert_postgres_to_camelcase(dict(row))
                    # Clean up None values
                    for key, value in statement_dict.items():
                        if value is None or value == '':
                            statement_dict[key] = None
                    statements.append(statement_dict)
                
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "statements": statements,
                    "years_requested": years,
                    "execution_time": execution_time,
                    "message": f"Income statement data retrieved for {symbol.upper()}"
                }
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"income statement retrieval for {symbol}", 
            current_user.get("id"),
            status_code=500
        )

@router.get("/{symbol}/balance-sheet")
async def get_company_balance_sheet(
    symbol: str,
    years: int = Query(default=5, ge=1, le=10, description="Number of years of data"),
    current_user: dict = Depends(get_current_user)
):
    """Get balance sheet data for a company."""
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                sql_query = """
                WITH latest_per_year AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY calendaryear ORDER BY date DESC) as rn
                    FROM financial_data.balance_sheets
                    WHERE UPPER(symbol) = %s 
                    AND period = 'FY'
                )
                SELECT 
                    symbol,
                    date,
                    calendaryear,
                    period,
                    cashandcashequivalents,
                    shortterminvestments,
                    cashandshortterminvestments,
                    netreceivables,
                    inventory,
                    othercurrentassets,
                    totalcurrentassets,
                    propertyplantequipmentnet,
                    goodwill,
                    intangibleassets,
                    goodwillandintangibleassets,
                    longterminvestments,
                    taxassets,
                    othernoncurrentassets,
                    totalnoncurrentassets,
                    otherassets,
                    totalassets,
                    accountpayables,
                    shorttermdebt,
                    taxpayables,
                    deferredrevenue,
                    othercurrentliabilities,
                    totalcurrentliabilities,
                    longtermdebt,
                    deferredrevenuenoncurrent,
                    deferredtaxliabilitiesnoncurrent,
                    othernoncurrentliabilities,
                    totalnoncurrentliabilities,
                    otherliabilities,
                    totalliabilities,
                    commonstock,
                    retainedearnings,
                    accumulatedothercomprehensiveincomeloss,
                    othertotalstockholdersequity,
                    totalstockholdersequity,
                    totalliabilitiesandstockholdersequity,
                    totalinvestments,
                    totaldebt,
                    netdebt
                FROM latest_per_year
                WHERE rn = 1
                ORDER BY calendaryear DESC
                LIMIT %s
                """
                
                cursor.execute(sql_query, (symbol_upper, years))
                results = cursor.fetchall()
                execution_time = time.time() - start_time
                
                statements = []
                for row in results:
                    statement_dict = convert_postgres_to_camelcase(dict(row))
                    for key, value in statement_dict.items():
                        if value is None or value == '':
                            statement_dict[key] = None
                    statements.append(statement_dict)
                
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "statements": statements,
                    "years_requested": years,
                    "execution_time": execution_time,
                    "message": f"Balance sheet data retrieved for {symbol.upper()}"
                }
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"balance sheet retrieval for {symbol}", 
            current_user.get("id"),
            status_code=500
        )

@router.get("/{symbol}/cash-flow")
async def get_company_cash_flow(
    symbol: str,
    years: int = Query(default=5, ge=1, le=10, description="Number of years of data"),
    current_user: dict = Depends(get_current_user)
):
    """Get cash flow statement data for a company."""
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                sql_query = """
                WITH latest_per_year AS (
                    SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY calendaryear ORDER BY date DESC) as rn
                    FROM financial_data.cash_flow_statements
                    WHERE UPPER(symbol) = %s 
                    AND period = 'FY'
                )
                SELECT 
                    symbol,
                    date,
                    calendaryear,
                    period,
                    netincome,
                    depreciationandamortization,
                    deferredincometax,
                    stockbasedcompensation,
                    changeinworkingcapital,
                    accountsreceivables,
                    inventory,
                    accountspayables,
                    otherworkingcapital,
                    othernoncashitems,
                    netcashprovidedbyoperatingactivities,
                    investmentsInPropertyPlantAndEquipment,
                    acquisitionsnet,
                    purchasesofinvestments,
                    salesmaturitiesofinvestments,
                    otherinvestingactivites,
                    netcashusedforinvestingactivites,
                    debtrepayment,
                    commonstockissued,
                    commonstockrepurchased,
                    dividendspaid,
                    otherfinancingactivites,
                    netcashusedprovidedbyfinancingactivities,
                    effectofforexchangesoncash,
                    netchangeincash,
                    cashatendofperiod,
                    cashatbeginningofperiod,
                    operatingcashflow,
                    capitalexpenditure,
                    freecashflow
                FROM latest_per_year
                WHERE rn = 1
                ORDER BY calendaryear DESC
                LIMIT %s
                """
                
                cursor.execute(sql_query, (symbol_upper, years))
                results = cursor.fetchall()
                execution_time = time.time() - start_time
                
                statements = []
                for row in results:
                    statement_dict = convert_postgres_to_camelcase(dict(row))
                    for key, value in statement_dict.items():
                        if value is None or value == '':
                            statement_dict[key] = None
                    statements.append(statement_dict)
                
                return {
                    "success": True,
                    "symbol": symbol.upper(),
                    "statements": statements,
                    "years_requested": years,
                    "execution_time": execution_time,
                    "message": f"Cash flow data retrieved for {symbol.upper()}"
                }
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"cash flow retrieval for {symbol}", 
            current_user.get("id"),
            status_code=500
        )
