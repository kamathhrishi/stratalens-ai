"""
Companies router for financial data endpoints
"""
import time
import traceback
import pandas as pd
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.security import HTTPAuthorizationCredentials

from app.schemas import CompanySearchResult, CompanySearchResponse
from agent.screener import FinancialDataAnalyzer
from app.utils.logging_utils import log_info, log_error, log_warning
import logging

logger = logging.getLogger(__name__)

# Global variables that will be injected
ANALYZER_AVAILABLE = False
analyzer_instance = None

def set_analyzer_globals(available: bool, instance: Optional[FinancialDataAnalyzer] = None):
    """Set the analyzer globals from the main server"""
    global ANALYZER_AVAILABLE, analyzer_instance
    ANALYZER_AVAILABLE = available
    analyzer_instance = instance

def get_analyzer() -> FinancialDataAnalyzer:
    """Get the analyzer instance"""
    if not ANALYZER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Query features disabled - analyzer not available")
    if analyzer_instance is None:
        raise HTTPException(status_code=503, detail="Analyzer not initialized")
    return analyzer_instance

def post_process_segment_data(segments: List[Dict], segment_type: str, symbol: str) -> Dict[str, Any]:
    """
    Post-process segment data to combine segments by ticker and format for display
    
    Args:
        segments: List of segment dictionaries from database
        segment_type: Either 'geographic' or 'product'
        symbol: Company symbol
        
    Returns:
        Formatted segment data with summaries and combined information
    """
    if not segments:
        return {
            "success": True,
            "symbol": symbol.upper(),
            "segment_type": segment_type,
            "segments": [],
            "summary": {
                "total_segments": 0,
                "total_revenue": 0,
                "year": None,
                "date": None
            },
            "message": f"No {segment_type} segment data found for {symbol.upper()}"
        }
    
    # Group segments by year and calculate totals
    segments_by_year = {}
    for segment in segments:
        year = segment.get('year')
        if year not in segments_by_year:
            segments_by_year[year] = []
        segments_by_year[year].append(segment)
    
    # Use the most recent year (should be 2024 based on our default)
    latest_year = max(segments_by_year.keys()) if segments_by_year else None
    latest_segments = segments_by_year.get(latest_year, [])
    
    # Calculate totals
    total_revenue = sum(seg.get('revenue', 0) for seg in latest_segments)
    
    # Format segments for display
    formatted_segments = []
    for segment in latest_segments:
        revenue = segment.get('revenue', 0)
        percentage = segment.get('percentage', 0)
        
        # Format revenue as currency
        if revenue >= 1e9:
            formatted_revenue = f"${revenue/1e9:.2f}B"
        elif revenue >= 1e6:
            formatted_revenue = f"${revenue/1e6:.2f}M"
        elif revenue >= 1e3:
            formatted_revenue = f"${revenue/1e3:.2f}K"
        else:
            formatted_revenue = f"${revenue:.2f}"
        
        formatted_segments.append({
            "segment": segment.get('segment', 'Unknown'),
            "revenue": revenue,
            "formatted_revenue": formatted_revenue,
            "percentage": percentage,
            "formatted_percentage": f"{percentage:.2f}%" if percentage else "0.00%",
            "year": year  # Add year to each segment
        })
    
    # Sort by revenue descending
    formatted_segments.sort(key=lambda x: x['revenue'], reverse=True)
    
    # Create summary
    summary = {
        "total_segments": len(formatted_segments),
        "total_revenue": total_revenue,
        "formatted_total_revenue": f"${total_revenue/1e9:.2f}B" if total_revenue >= 1e9 else f"${total_revenue/1e6:.2f}M",
        "year": latest_year,
        "date": latest_segments[0].get('date') if latest_segments else None,
        "data_type": "Historical Annual Data",
        "description": f"{segment_type.title()} Segments (Historical Annual Data)"
    }
    
    return {
        "success": True,
        "symbol": symbol.upper(),
        "segment_type": segment_type,
        "segments": formatted_segments,
        "summary": summary,
        "message": f"{segment_type.title()} segment data retrieved for {symbol.upper()}",
        "raw_data": segments  # Keep raw data for debugging
    }

# Create router
router = APIRouter(prefix="/companies", tags=["companies"])

# Import centralized utilities
from app.auth.auth_utils import get_current_user
from db.db_utils import get_db
from app.utils import create_error_response, raise_sanitized_http_exception
import psycopg2
from psycopg2.extras import RealDictCursor

# Shared utilities
from db.db_connection_utils import get_postgres_connection
from db.data_transformation_utils import convert_postgres_to_camelcase

@router.get("/public/search")
async def search_companies_public(
    query: str = Query(..., min_length=1, max_length=100, description="Search term for company name or ticker"),
    limit: int = Query(default=8, ge=1, le=20, description="Maximum number of results"),
    db = Depends(get_db)
):
    """Public company search endpoint for ticker autocomplete (no authentication required)"""
    start_time = time.time()
    
    logger.info(f"🔍 Public company search called for query: {query}")
    
    try:
        # Clean the search query
        search_term = query.strip().upper()
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build SQL query to search companies in PostgreSQL
                sql_query = """
        SELECT 
            symbol,
                    companyname,
            sector,
            industry,
                    mktcap,
            country,
                    exchangeshortname,
            description
                FROM financial_data.company_profiles 
        WHERE 
                    UPPER(symbol) LIKE %s 
                    OR UPPER(companyname) LIKE %s
                    OR UPPER(sector) LIKE %s
                    OR UPPER(industry) LIKE %s
        ORDER BY 
            CASE 
                        WHEN UPPER(symbol) = %s THEN 1
                        WHEN UPPER(symbol) LIKE %s THEN 2
                        WHEN UPPER(companyname) LIKE %s THEN 3
                ELSE 4
            END,
                    mktcap DESC NULLS LAST
                LIMIT %s
                """
                
                # Execute the query with parameterized values for security
                cursor.execute(sql_query, (
                    f'%{search_term}%',  # symbol LIKE
                    f'%{search_term}%',  # companyname LIKE
                    f'%{search_term}%',  # sector LIKE
                    f'%{search_term}%',  # industry LIKE
                    search_term,         # symbol = (exact match)
                    f'{search_term}%',   # symbol LIKE (starts with)
                    f'{search_term}%',   # companyname LIKE (starts with)
                    limit
                ))
                
                results = cursor.fetchall()
                
                execution_time = time.time() - start_time
                
                # Convert to response format
                companies = []
                for row in results:
                    # Convert PostgreSQL row dict to camelCase
                    row_dict = dict(row)
                    company = {
                        "symbol": row_dict.get("symbol", ""),
                        "companyName": row_dict.get("companyname", ""),
                        "sector": row_dict.get("sector", ""),
                        "industry": row_dict.get("industry", ""),
                        "marketCap": row_dict.get("mktcap"),
                        "country": row_dict.get("country", ""),
                        "exchangeShortName": row_dict.get("exchangeshortname", ""),
                        "description": row_dict.get("description", "")
                    }
                    companies.append(company)
                
                logger.info(f"✅ Public company search completed in {execution_time:.3f}s, found {len(companies)} companies")
                
                return CompanySearchResponse(
                    success=True,
                    companies=companies,
                    total_found=len(companies),
                    query=query,
                    execution_time=execution_time,
                    message=f"Found {len(companies)} companies matching '{query}'"
                )
                
        finally:
            conn.close()
            
    except Exception as e:
        execution_time = time.time() - start_time
        error_message = f"Company search failed: {str(e)}"
        logger.error(f"❌ {error_message}")
        
        return CompanySearchResponse(
            success=False,
            companies=[],
            total_found=0,
            query=query if 'query' in locals() else "",
            execution_time=execution_time,
            message=error_message
        )

@router.get("/search")
async def search_companies(
    query: str = Query(..., min_length=1, max_length=100, description="Search term for company name or ticker"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of results"),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db)
):
    """Search companies by ticker symbol or company name"""
    start_time = time.time()

    logger.info(f"🔍 Company search router called for query: {query}")
    
    try:
        # Clean the search query
        search_term = query.strip().upper()
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build SQL query to search companies in PostgreSQL
                sql_query = """
        SELECT 
            symbol,
                    companyname,
            sector,
            industry,
                    mktcap,
            country,
                    exchangeshortname,
            description
                FROM financial_data.company_profiles 
        WHERE 
                    UPPER(symbol) LIKE %s 
                    OR UPPER(companyname) LIKE %s
                    OR UPPER(sector) LIKE %s
                    OR UPPER(industry) LIKE %s
        ORDER BY 
            CASE 
                        WHEN UPPER(symbol) = %s THEN 1
                        WHEN UPPER(symbol) LIKE %s THEN 2
                        WHEN UPPER(companyname) LIKE %s THEN 3
                ELSE 4
            END,
                    mktcap DESC NULLS LAST
                LIMIT %s
                """
                
                # Execute the query with parameterized values for security
                cursor.execute(sql_query, (
                    f'%{search_term}%',  # symbol LIKE
                    f'%{search_term}%',  # companyname LIKE
                    f'%{search_term}%',  # sector LIKE
                    f'%{search_term}%',  # industry LIKE
                    search_term,         # symbol = (exact match)
                    f'{search_term}%',   # symbol LIKE (starts with)
                    f'{search_term}%',   # companyname LIKE (starts with)
                    limit
                ))
                
                results = cursor.fetchall()
                
                execution_time = time.time() - start_time
                
                # Convert to response format
                companies = []
                for row in results:
                    # Convert PostgreSQL row dict to camelCase
                    row_dict = dict(row)
                    camel_row = convert_postgres_to_camelcase(row_dict)
                    
                    companies.append(CompanySearchResult(
                        symbol=camel_row.get('symbol', ''),
                        companyName=camel_row.get('companyName', ''),
                        sector=camel_row.get('sector'),
                        industry=camel_row.get('industry'),
                        marketCap=float(camel_row.get('marketCap')) if camel_row.get('marketCap') else None,
                        country=camel_row.get('country'),
                        exchangeShortName=camel_row.get('exchangeShortName'),
                        description=camel_row.get('description')
                    ))
                
                logger.info(f"✅ Company search completed: {len(companies)} results in {execution_time:.3f}s")
                
                return CompanySearchResponse(
                    success=True,
                    companies=companies,
                    total_found=len(companies),
                    query=query,
                    execution_time=execution_time,
                    message=f"Found {len(companies)} companies matching '{query}'"
                )
                
        except Exception as e:
            logger.error(f"❌ Error searching companies: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error searching companies: {str(e)}")
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_message = f"Company search failed: {str(e)}"
        logger.error(f"❌ COMPANY SEARCH ERROR: {error_message}")
        logger.error(f"❌ COMPANY SEARCH EXCEPTION: {type(e).__name__}: {str(e)}")
        logger.error(f"❌ COMPANY SEARCH TRACEBACK: {traceback.format_exc()}")
        
        return CompanySearchResponse(
            success=False,
            companies=[],
            total_found=0,
            query=query,
            execution_time=execution_time,
            message=error_message
        )

@router.get("/{symbol}")
async def get_company_details(
    symbol: str,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive company information by joining multiple tables"""
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        # Get comprehensive company data by joining multiple tables
                # Note: PostgreSQL doesn't have TTM tables, so we'll use the main tables
                sql_query = """
        SELECT 
            -- Basic company info from company_profiles
            cp.symbol,
                    cp.companyname,
            cp.sector,
            cp.industry,
            cp.country,
            cp.city,
            cp.state,
            cp.address,
            cp.phone,
            cp.website,
            cp.description,
            cp.ceo,
                    cp.fulltimeemployees,
                    cp.ipodate,
            cp.exchange,
                    cp.exchangeshortname,
            cp.price,
            cp.changes,
                    cp.mktcap,
            cp.range,
            cp.beta,
                    cp.volavg,
                    cp.lastdiv,
            
            -- Financial ratios from latest period
                    fr.priceearningsratio,
                    fr.pricetobookratio,
                    fr.pricetosalesratio,
                    fr.pricetofreecashflowsratio,
                    fr.priceearningstogrowthratio,
                    fr.returnonequity,
                    fr.returnonassets,
                    fr.dividendyield,
                    fr.dividendpayoutratio,
                    fr.currentratio,
                    fr.quickratio,
                    fr.debtequityratio,
                    fr.grossprofitmargin,
                    fr.operatingprofitmargin,
                    fr.netprofitmargin,
                    fr.cashpershare,
                    fr.freecashflowpershare,
                    fr.interestcoverage,
            
            -- Key metrics from latest period
                    km.peratio,
                    km.pbratio,
                    km.enterprisevalue,
                    km.enterprisevalueoverebitda,
                    km.evtosales,
                    km.evtofreecashflow,
                    km.evtooperatingcashflow,
                    km.marketcap,
                    km.bookvaluepershare,
                    km.roe,
            km.roic,
                    km.freecashflowyield,
                    km.earningsyield,
                    km.payoutratio,
                    km.dividendyield,
                    km.netincomepershare,
                    km.revenuepershare,
                    km.tangiblebookvaluepershare,
                    km.workingcapital,
                    km.investedcapital,
                    km.returnontangibleassets,
                    
                    -- Latest income statement data
                    is_latest.revenue,
                    is_latest.netincome,
                    is_latest.grossprofit,
                    is_latest.operatingincome,
                    is_latest.eps,
                    is_latest.epsdiluted,
                    is_latest.ebitda,
                    is_latest.weightedaverageshsout,
                    is_latest.weightedaverageshsoutdil,
                    
                    -- Latest balance sheet data
                    bs_latest.totalassets,
                    bs_latest.totaldebt,
                    bs_latest.totalstockholdersequity,
                    bs_latest.cashandcashequivalents,
                    bs_latest.totalcurrentassets,
                    bs_latest.totalcurrentliabilities,
                    
                    -- Latest cash flow data
                    cs_latest.freecashflow,
                    cs_latest.operatingcashflow,
                    cs_latest.capitalexpenditure,
            
            -- Revenue per share calculation
            CASE 
                        WHEN is_latest.revenue IS NOT NULL 
                        AND is_latest.weightedaverageshsout IS NOT NULL 
                        AND is_latest.weightedaverageshsout > 0
                        THEN is_latest.revenue / is_latest.weightedaverageshsout
                ELSE NULL 
                    END as revenuepershare_calc,
            
            -- Working capital calculation
            CASE 
                        WHEN bs_latest.totalcurrentassets IS NOT NULL 
                        AND bs_latest.totalcurrentliabilities IS NOT NULL
                        THEN bs_latest.totalcurrentassets - bs_latest.totalcurrentliabilities
                ELSE NULL 
                    END as workingcapital_calc
            
                FROM financial_data.company_profiles cp
        
        -- Get latest financial ratios
        LEFT JOIN (
                    SELECT * FROM financial_data.financial_ratios 
                    WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
        ) fr ON cp.symbol = fr.symbol
        
        -- Get latest key metrics
        LEFT JOIN (
                    SELECT * FROM financial_data.key_metrics 
                    WHERE symbol = %s 
            AND period = 'FY'
            ORDER BY date DESC 
            LIMIT 1
        ) km ON cp.symbol = km.symbol
        
                -- Get latest income statement
        LEFT JOIN (
                    SELECT * FROM financial_data.income_statements 
                    WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
                ) is_latest ON cp.symbol = is_latest.symbol
        
                -- Get latest balance sheet
        LEFT JOIN (
                    SELECT * FROM financial_data.balance_sheets 
                    WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
                ) bs_latest ON cp.symbol = bs_latest.symbol
        
                -- Get latest cash flow statement
        LEFT JOIN (
                    SELECT * FROM financial_data.cash_flow_statements 
                    WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
                ) cs_latest ON cp.symbol = cs_latest.symbol
                
                WHERE UPPER(cp.symbol) = %s
        LIMIT 1
        """
        
                cursor.execute(sql_query, (symbol_upper, symbol_upper, symbol_upper, symbol_upper, symbol_upper, symbol_upper))
                result = cursor.fetchone()
        
                if not result:
                    raise HTTPException(status_code=404, detail=f"Company with symbol '{symbol}' not found")
        
                # Convert PostgreSQL row dict to camelCase
                company_data = convert_postgres_to_camelcase(dict(result))
                
                # Debug: Log key financial ratios to see what data we're getting
                logger.info(f"🔍 DEBUG - Key ratios for {symbol_upper}:")
                logger.info(f"  P/E Ratio (fr.priceearningsratio): {result.get('priceearningsratio')}")
                logger.info(f"  P/E Ratio (km.peratio): {result.get('peratio')}")
                logger.info(f"  P/B Ratio (fr.pricetobookratio): {result.get('pricetobookratio')}")
                logger.info(f"  P/B Ratio (km.pbratio): {result.get('pbratio')}")
                logger.info(f"  EV/EBITDA (fr.enterprisevalueoverebitda): {result.get('enterprisevalueoverebitda')}")
                logger.info(f"  EV/EBITDA (km.enterprisevalueoverebitda): {result.get('enterprisevalueoverebitda')}")
        
                # Clean up None values and format some fields
                for key, value in company_data.items():
                    if value == 'nan' or value == '':
                        company_data[key] = None
                
                # Add some calculated fields for better frontend display
                if company_data.get('price') and company_data.get('changes'):
                    try:
                        company_data['previousClose'] = float(company_data['price']) - float(company_data['changes'])
                    except (TypeError, ValueError):
                        company_data['previousClose'] = None
                
                # Format range field if it exists
                if company_data.get('range'):
                    try:
                        range_parts = str(company_data['range']).split('-')
                        if len(range_parts) == 2:
                            company_data['dayLow'] = float(range_parts[0].strip())
                            company_data['dayHigh'] = float(range_parts[1].strip()) 
                    except (ValueError, AttributeError):
                        pass
                        
                # Add some default volume if avgVolume exists
                if company_data.get('avgVolume'):
                    company_data['volume'] = company_data['avgVolume']
                
                execution_time = time.time() - start_time
                logger.info(f"✅ Company details completed for {symbol_upper} in {execution_time:.3f}s")
        
                return {
                    "success": True,
                    "company": company_data,
                    "message": f"Comprehensive company details for {symbol.upper()}"
                }
                
        finally:
            conn.close()
        
    except HTTPException:
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"company details fetch for {symbol}", 
            current_user.get("id"),
            status_code=500
        )

@router.get("/{symbol}/ttm-metrics")
async def get_company_ttm_metrics(
    symbol: str,
    current_user: dict = Depends(get_current_user)
):
    """Get TTM (Trailing Twelve Months) metrics for a company with last 5 quarters and YoY growth."""
    logger.info(f"🔍 TTM METRICS REQUEST: symbol={symbol}, user={current_user.get('email', 'unknown')}")
    
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        logger.info(f"📊 TTM METRICS: Processing symbol {symbol_upper}")
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Since PostgreSQL doesn't have TTM tables, we'll get the latest available data
                # and simulate TTM by getting the most recent quarters
                sql_query = """
        WITH latest_quarters AS (
            SELECT 
                symbol,
                date,
                        calendaryear as fiscalyear,
                period,
                revenue,
                        costofrevenue,
                        grossprofit,
                        operatingexpenses,
                        operatingincome,
                        incomebeforetax,
                        incometaxexpense,
                        netincome,
                        eps,
                        epsdiluted,
                        ROW_NUMBER() OVER (ORDER BY date DESC, calendaryear DESC, 
                            CASE period WHEN 'FY' THEN 5 WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 END DESC) as rn
                    FROM financial_data.income_statements
                    WHERE UPPER(symbol) = %s
        ),
        current_data AS (
            SELECT * FROM latest_quarters WHERE rn = 1
        ),
        previous_year_data AS (
            SELECT 
                lq.*,
                        ROW_NUMBER() OVER (ORDER BY lq.date DESC, lq.fiscalyear DESC, 
                            CASE lq.period WHEN 'FY' THEN 5 WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 END DESC) as prev_rn
            FROM latest_quarters lq
            JOIN current_data cd ON lq.symbol = cd.symbol 
                        AND lq.fiscalyear = cd.fiscalyear - 1
        )
        SELECT 
            -- Current quarter data
            curr.symbol,
            curr.date as current_date,
                    curr.fiscalyear as current_fiscalyear,
            curr.period as current_period,
            
            -- Income Statement (Current)
                    curr.revenue as revenuettm,
                    curr.costofrevenue as costofrevenuettm,
                    curr.grossprofit as grossprofitttm,
                    curr.operatingexpenses as operatingexpensesttm,
                    curr.operatingincome as operatingincomettm,
                    curr.incomebeforetax as incomebeforetaxttm,
                    curr.incometaxexpense as incometaxexpensettm,
                    curr.netincome as netincomettm,
                    curr.eps as epsttm,
                    curr.epsdiluted as epsdilutedttm,
            
            -- Balance Sheet (Current) - Get latest balance sheet data
                    bs.cashandcashequivalents as cashttm,
                    bs.totalcurrentassets as currentassetsttm,
                    bs.totalassets as totalassetsttm,
                    bs.totalcurrentliabilities as currentliabilitiesttm,
                    bs.longtermdebt as longtermdebtttm,
                    bs.totalliabilities as totalliabilitiesttm,
                    bs.totaldebt as totaldebtttm,
                    bs.totalequity as stockholdersequityttm,
                    (bs.totalcurrentassets - bs.totalcurrentliabilities) as workingcapitalttm,
            
            -- Cash Flow (Current) - Get latest cash flow data
                    cf.operatingcashflow as operatingcashflowttm,
                    cf.netcashusedforinvestingactivites as investingcashflowttm,
                    cf.netcashusedprovidedbyfinancingactivities as financingcashflowttm,
                    cf.capitalexpenditure as capexttm,
                    cf.freecashflow as freecashflowttm,
                    cf.netchangeincash as netchangeincashttm,
            
            -- Previous year same quarter data
                    prev.revenue as revenuettm_prev,
                    prev.netincome as netincomettm_prev,
                    prev.operatingincome as operatingincomettm_prev,
                    prev.grossprofit as grossprofitttm_prev,
            
            -- YoY Growth Calculations
            CASE 
                WHEN prev.revenue IS NOT NULL AND prev.revenue != 0 
                THEN ROUND(CAST(((curr.revenue - prev.revenue) / prev.revenue) * 100 AS NUMERIC), 2)
                ELSE NULL 
            END as revenue_yoy_growth_pct,
            
            CASE 
                        WHEN prev.netincome IS NOT NULL AND prev.netincome != 0 
                        THEN ROUND(CAST(((curr.netincome - prev.netincome) / prev.netincome) * 100 AS NUMERIC), 2)
                ELSE NULL 
                    END as netincome_yoy_growth_pct,
            
            CASE 
                        WHEN prev.operatingincome IS NOT NULL AND prev.operatingincome != 0 
                        THEN ROUND(CAST(((curr.operatingincome - prev.operatingincome) / prev.operatingincome) * 100 AS NUMERIC), 2)
                ELSE NULL 
                    END as operatingincome_yoy_growth_pct,
            
            CASE 
                        WHEN prev.grossprofit IS NOT NULL AND prev.grossprofit != 0 
                        THEN ROUND(CAST(((curr.grossprofit - prev.grossprofit) / prev.grossprofit) * 100 AS NUMERIC), 2)
                ELSE NULL 
                    END as grossprofit_yoy_growth_pct
            
        FROM current_data curr
        LEFT JOIN previous_year_data prev ON curr.symbol = prev.symbol AND prev.prev_rn = 1
        LEFT JOIN (
            SELECT * FROM financial_data.balance_sheets 
            WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
        ) bs ON curr.symbol = bs.symbol
        LEFT JOIN (
            SELECT * FROM financial_data.cash_flow_statements 
            WHERE symbol = %s 
            ORDER BY date DESC 
            LIMIT 1
        ) cf ON curr.symbol = cf.symbol
        """
        
                logger.info(f"🔧 TTM METRICS SQL: {sql_query}")
                
                cursor.execute(sql_query, (symbol_upper, symbol_upper, symbol_upper))
                result = cursor.fetchone()
                
                logger.info(f"📝 TTM METRICS DB RESULT: found={result is not None}")
                if result:
                    logger.info(f"📝 TTM METRICS SAMPLE DATA: {dict(result)}")
                
                ttm_data = {}
                if result:
                    ttm_dict = convert_postgres_to_camelcase(dict(result))
                    for key, value in ttm_dict.items():
                        if value is None or value == '':
                            ttm_data[key] = None
                        else:
                            ttm_data[key] = value
                
                execution_time = time.time() - start_time
                
                response_data = {
                    "success": True,
                    "symbol": symbol.upper(),
                    "metrics": ttm_data,
                    "execution_time": execution_time,
                    "message": f"TTM metrics retrieved for {symbol.upper()}"
                }
                
                logger.info(f"✅ TTM METRICS SUCCESS: symbol={symbol_upper}, metrics_count={len(ttm_data)}, execution_time={execution_time:.3f}s")
                logger.info(f"📤 TTM METRICS RESPONSE: {response_data}")
                
                return response_data
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"TTM metrics retrieval for {symbol}", 
            current_user.get("id"),
            status_code=500
        )


@router.get("/{symbol}/ttm-history")
async def get_company_ttm_history(
    symbol: str,
    quarters: int = Query(default=5, ge=1, le=10, description="Number of quarters to retrieve"),
    current_user: dict = Depends(get_current_user)
):
    """Get historical TTM (Trailing Twelve Months) data for the last N quarters."""
    logger.info(f"🔍 TTM HISTORY REQUEST: symbol={symbol}, quarters={quarters}, user={current_user.get('email', 'unknown')}")
    
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        logger.info(f"📊 TTM HISTORY: Processing symbol {symbol_upper} for {quarters} quarters")
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get last N quarters of data (simulating TTM since PostgreSQL doesn't have TTM tables)
                sql_query = """
        WITH latest_quarters AS (
            SELECT 
                symbol,
                date,
                        calendaryear as fiscalyear,
                period,
                revenue,
                        costofrevenue,
                        grossprofit,
                        operatingincome,
                        netincome,
                        eps,
                        epsdiluted,
                        ROW_NUMBER() OVER (ORDER BY date DESC, calendaryear DESC, 
                            CASE period WHEN 'FY' THEN 5 WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 END DESC) as rn
                    FROM financial_data.income_statements
                    WHERE UPPER(symbol) = %s
        )
        SELECT 
            symbol,
            date,
                    fiscalyear,
            period,
            
            -- Income Statement
            revenue,
                    costofrevenue,
                    grossprofit,
                    operatingincome,
                    netincome,
                    eps,
                    epsdiluted
            
        FROM latest_quarters 
                WHERE rn <= %s
                ORDER BY date DESC, fiscalyear DESC, 
                    CASE period WHEN 'FY' THEN 5 WHEN 'Q4' THEN 4 WHEN 'Q3' THEN 3 WHEN 'Q2' THEN 2 WHEN 'Q1' THEN 1 END DESC
        """
        
                logger.info(f"🔧 TTM HISTORY SQL: {sql_query}")
                
                cursor.execute(sql_query, (symbol_upper, quarters))
                results = cursor.fetchall()
                
                logger.info(f"📝 TTM HISTORY DB RESULT: found={len(results)} rows")
                if results:
                    logger.info(f"📝 TTM HISTORY SAMPLE DATA: {dict(results[0])}")
                
                quarters_data = []
                for row in results:
                    quarter_dict = convert_postgres_to_camelcase(dict(row))
                    # Clean up None values
                    for key, value in quarter_dict.items():
                        if value is None or value == '':
                            quarter_dict[key] = None
                    quarters_data.append(quarter_dict)
                
                execution_time = time.time() - start_time
                
                response_data = {
                    "success": True,
                    "symbol": symbol.upper(),
                    "quarters": quarters_data,
                    "quarters_requested": quarters,
                    "quarters_returned": len(quarters_data),
                    "execution_time": execution_time,
                    "message": f"TTM history retrieved for {symbol.upper()} - {len(quarters_data)} quarters"
                }
                
                logger.info(f"✅ TTM HISTORY SUCCESS: symbol={symbol_upper}, quarters_count={len(quarters_data)}, execution_time={execution_time:.3f}s")
                logger.info(f"📤 TTM HISTORY RESPONSE: quarters={len(quarters_data)} items")
                
                return response_data
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"TTM history retrieval for {symbol}", 
            current_user.get("id"),
            status_code=500
        )


@router.get("/{symbol}/product-segments")
async def get_company_product_segments(
    symbol: str,
    current_user: dict = Depends(get_current_user)
):
    """Get product/revenue segment data for a company."""
    logger.info(f"🔍 PRODUCT SEGMENTS REQUEST: symbol={symbol}, user={current_user.get('email', 'unknown')}")
    
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        logger.info(f"📊 PRODUCT SEGMENTS: Processing symbol {symbol_upper}")
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                sql_query = """
        -- MANDATORY: Default to 2024 for historical segment data unless user specifies otherwise
        WITH latest_year AS (
            SELECT COALESCE(
                        MAX(CASE WHEN COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) <= 2024 
                                 THEN COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) END),
                2024
            ) as max_year
                    FROM financial_data.revenue_product_segmentation
                    WHERE UPPER(symbol) = %s
        ),
        distinct_segments AS (
            SELECT DISTINCT
                symbol,
                date,
                        COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) as year,
                segment,
                revenue
                    FROM financial_data.revenue_product_segmentation, latest_year
                    WHERE UPPER(symbol) = %s
                    AND COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) = latest_year.max_year
        )
        SELECT 
            symbol,
            date,
            year,
            segment,
            revenue,
            -- Calculate percentage of total revenue for the same year (historical data)
            CASE 
                WHEN SUM(revenue) OVER (PARTITION BY symbol, year) > 0 
                THEN ROUND(CAST((revenue * 100.0) / SUM(revenue) OVER (PARTITION BY symbol, year) AS NUMERIC), 2)
                ELSE NULL 
            END as percentage
        FROM distinct_segments
        ORDER BY revenue DESC
        """
        
                logger.info(f"🔧 PRODUCT SEGMENTS SQL: {sql_query}")
                
                cursor.execute(sql_query, (symbol_upper, symbol_upper))
                results = cursor.fetchall()
                
                logger.info(f"📝 PRODUCT SEGMENTS DB RESULT: found={len(results)} rows")
                if results:
                    logger.info(f"📝 PRODUCT SEGMENTS SAMPLE DATA: {dict(results[0])}")
                
                segments = []
                for row in results:
                    segment_dict = convert_postgres_to_camelcase(dict(row))
                    # Clean up None values
                    for key, value in segment_dict.items():
                        if value is None or value == '':
                            segment_dict[key] = None
                    segments.append(segment_dict)
                
                # Apply post-processing to combine segments by ticker and format for display
                response_data = post_process_segment_data(segments, "product", symbol.upper())
                execution_time = time.time() - start_time
                response_data["execution_time"] = execution_time
                
                logger.info(f"✅ PRODUCT SEGMENTS SUCCESS: symbol={symbol_upper}, segments_count={len(segments)}, execution_time={execution_time:.3f}s")
                logger.info(f"📤 PRODUCT SEGMENTS RESPONSE: segments={len(segments)} items")
                if segments:
                    logger.info(f"📤 PRODUCT SEGMENTS SAMPLE: {segments[0] if segments else 'None'}")
                
                return response_data
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"product segments retrieval for {symbol}", 
            current_user.get("id"),
            status_code=500
        )


@router.get("/{symbol}/geographic-segments")
async def get_company_geographic_segments(
    symbol: str,
    current_user: dict = Depends(get_current_user)
):
    """Get geographic/revenue segment data for a company."""
    logger.info(f"🔍 GEOGRAPHIC SEGMENTS REQUEST: symbol={symbol}, user={current_user.get('email', 'unknown')}")
    
    start_time = time.time()
    
    try:
        symbol_upper = symbol.upper()
        logger.info(f"📊 GEOGRAPHIC SEGMENTS: Processing symbol {symbol_upper}")
        
        # Get PostgreSQL connection
        conn = get_postgres_connection()
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                sql_query = """
        -- MANDATORY: Default to 2024 for historical segment data unless user specifies otherwise
        WITH latest_year AS (
            SELECT COALESCE(
                        MAX(CASE WHEN COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) <= 2024 
                                 THEN COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) END),
                2024
            ) as max_year
                    FROM financial_data.revenue_geographic_segmentation
                    WHERE UPPER(symbol) = %s
        ),
        distinct_segments AS (
            SELECT DISTINCT
                symbol,
                date,
                        COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) as year,
                segment,
                revenue
                    FROM financial_data.revenue_geographic_segmentation, latest_year
                    WHERE UPPER(symbol) = %s
                    AND COALESCE(fiscalyear, EXTRACT(YEAR FROM date)) = latest_year.max_year
        )
        SELECT 
            symbol,
            date,
            year,
            segment,
            revenue,
            -- Calculate percentage of total revenue for the same year (historical data)
            CASE 
                WHEN SUM(revenue) OVER (PARTITION BY symbol, year) > 0 
                THEN ROUND(CAST((revenue * 100.0) / SUM(revenue) OVER (PARTITION BY symbol, year) AS NUMERIC), 2)
                ELSE NULL 
            END as percentage
        FROM distinct_segments
        ORDER BY revenue DESC
        """
        
                logger.info(f"🔧 GEOGRAPHIC SEGMENTS SQL: {sql_query}")
                
                cursor.execute(sql_query, (symbol_upper, symbol_upper))
                results = cursor.fetchall()
                
                logger.info(f"📝 GEOGRAPHIC SEGMENTS DB RESULT: found={len(results)} rows")
                if results:
                    logger.info(f"📝 GEOGRAPHIC SEGMENTS SAMPLE DATA: {dict(results[0])}")
                
                segments = []
                for row in results:
                    segment_dict = convert_postgres_to_camelcase(dict(row))
                    # Clean up None values
                    for key, value in segment_dict.items():
                        if value is None or value == '':
                            segment_dict[key] = None
                    segments.append(segment_dict)
                
                # Apply post-processing to combine segments by ticker and format for display
                response_data = post_process_segment_data(segments, "geographic", symbol.upper())
                execution_time = time.time() - start_time
                response_data["execution_time"] = execution_time
                
                logger.info(f"✅ GEOGRAPHIC SEGMENTS SUCCESS: symbol={symbol_upper}, segments_count={len(segments)}, execution_time={execution_time:.3f}s")
                logger.info(f"📤 GEOGRAPHIC SEGMENTS RESPONSE: segments={len(segments)} items")
                if segments:
                    logger.info(f"📤 GEOGRAPHIC SEGMENTS SAMPLE: {segments[0] if segments else 'None'}")
                
                return response_data
                
        finally:
            conn.close()
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise_sanitized_http_exception(
            e, 
            f"geographic segments retrieval for {symbol}", 
            current_user.get("id"),
            status_code=500
        )



