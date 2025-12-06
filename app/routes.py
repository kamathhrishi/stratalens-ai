"""
Route Configuration

Sets up all API routes, frontend routes, and static file serving.
"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse

from app.auth import auth
from app.auth import router as auth_router
from app.auth.auth_utils import get_db
from config import get_cors_origins, settings
from app.routers.charting import router as charting_router
from app.routers.chat import router as chat_router
from app.routers.companies import router as companies_router
from app.routers.companies_financials import router as companies_financials_router
from app.routers.screens import router as screens_router
from app.routers.screener import router as screener_router
from app.routers.transcript import router as transcript_router
from app.routers.users import router as users_router
from app.websocket.routes import websocket_endpoint


def setup_routes(app: FastAPI):
    """Configure all routes for the FastAPI application"""
    
    # Setup authentication dependency
    auth.set_db_dependency(get_db)
    
    # Include API routers
    app.include_router(auth_router)
    app.include_router(screens_router)
    app.include_router(charting_router)
    app.include_router(companies_router)
    app.include_router(companies_financials_router)
    app.include_router(chat_router, prefix="/chat")
    app.include_router(screener_router)
    app.include_router(transcript_router)
    app.include_router(users_router)
    
    # WebSocket route
    app.websocket("/ws/{user_id}")(websocket_endpoint)
    
    # Frontend routes
    setup_frontend_routes(app)
    
    # Static file endpoints
    setup_static_routes(app)
    
    # CORS preflight handlers
    setup_cors_handlers(app)
    
    # Public endpoints
    setup_public_endpoints(app)


def setup_frontend_routes(app: FastAPI):
    """Setup frontend serving routes"""
    
    @app.get("/")
    async def serve_landing():
        """Serve the landing page at root"""
        return FileResponse("frontend/landing.html")
    
    @app.get("/app")
    async def serve_app():
        """Serve the main application"""
        return FileResponse("frontend/index.html")


def setup_static_routes(app: FastAPI):
    """Setup static file serving routes"""
    
    @app.get("/styles.css")
    async def serve_styles():
        """Serve main stylesheet"""
        return FileResponse("frontend/styles.css")
    
    @app.get("/index-styles.css")
    async def serve_index_styles():
        """Serve index-specific stylesheet"""
        return FileResponse("frontend/index-styles.css")
    
    @app.get("/utils.js")
    async def serve_utils():
        """Serve utils JavaScript"""
        return FileResponse("frontend/utils.js")
    
    @app.get("/config.js")
    async def serve_config():
        """Serve config JavaScript"""
        return FileResponse("frontend/config.js")
    
    @app.get("/chat.js")
    async def serve_chat():
        """Serve chat JavaScript"""
        return FileResponse("frontend/chat.js")
    
    @app.get("/app.js")
    async def serve_app_js():
        """Serve app JavaScript"""
        return FileResponse("frontend/app.js")
    
    @app.get("/charts.js")
    async def serve_charts():
        """Serve charts JavaScript"""
        return FileResponse("frontend/charts.js")
    
    @app.get("/session.js")
    async def serve_session():
        """Serve session JavaScript"""
        return FileResponse("frontend/session.js")
    
    @app.get("/landing-integration.js")
    async def serve_landing_integration():
        """Serve landing integration JavaScript"""
        return FileResponse("frontend/landing-integration.js")
    
    @app.get("/favicon.ico")
    async def serve_favicon():
        """Serve favicon - return 204 No Content to avoid 404 errors"""
        return Response(status_code=204)


def setup_cors_handlers(app: FastAPI):
    """Setup CORS preflight handlers"""
    allowed_origins = get_cors_origins()
    dev_hosts = settings.SERVER.DEV_HOSTS
    
    @app.options("/auth/{path:path}")
    async def auth_options_handler(request: Request):
        """Handle CORS preflight requests for auth endpoints"""
        origin = request.headers.get("Origin")
        
        # Check if origin is in allowed origins
        if origin and origin in allowed_origins:
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                    "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Cache-Control, Pragma, User-Agent, Referer",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "86400",
                }
            )
        
        # Development mode: allow localhost and null origins
        if origin and any(host in origin for host in dev_hosts):
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                    "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Cache-Control, Pragma, User-Agent, Referer",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "86400",
                }
            )
        
        # Allow null origin for file:// URLs in development only
        if not origin or origin == "null":
            if any(host in str(allowed_origins) for host in dev_hosts):
                return Response(
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                        "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Cache-Control, Pragma, User-Agent, Referer",
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Max-Age": "86400",
                    }
                )
        
        return Response(status_code=403)
    
    @app.options("/{path:path}")
    async def global_options_handler(request: Request):
        """Handle CORS preflight requests for any endpoint"""
        origin = request.headers.get("Origin")
        
        # Check if origin is in allowed origins
        if origin and origin in allowed_origins:
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                    "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Cache-Control, Pragma, User-Agent, Referer",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "86400",
                }
            )
        
        # Development mode: allow localhost and null origins
        if origin and any(host in origin for host in dev_hosts):
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                    "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Cache-Control, Pragma, User-Agent, Referer",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "86400",
                }
            )
        
        # Allow null origin for file:// URLs in development only
        if not origin or origin == "null":
            if any(host in str(allowed_origins) for host in dev_hosts):
                return Response(
                    status_code=200,
                    headers={
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, HEAD",
                        "Access-Control-Allow-Headers": "Accept, Accept-Language, Content-Language, Content-Type, Authorization, X-Requested-With, Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Cache-Control, Pragma, User-Agent, Referer",
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Max-Age": "86400",
                    }
                )
        
        return Response(status_code=403)


def setup_public_endpoints(app: FastAPI):
    """Setup public API endpoints"""
    from config import get_cors_origins
    from app.lifespan import get_analyzer_instance, get_db_pool
    from app.utils.general_utils import get_memory_usage
    from config import settings
    from datetime import datetime
    
    allowed_origins = get_cors_origins()
    
    @app.get("/api", response_model=dict)
    async def api_info():
        """API information endpoint"""
        analyzer_instance = get_analyzer_instance()
        
        return {
            "message": "StrataLens Complete API with Financial Analysis & Saved Screens",
            "version": app.version,
            "cors_origins": allowed_origins,
            "features": [
                "💹 Intelligent Financial Data Analysis",
                "📊 Company Comparison & Analysis",
                "⚡ Real-time AI Reasoning Stream",
                "🔐 Secure Token Validation",
                "👥 Regular User Registration & Authentication",
                "💎 Self-Serve User Registration", 
                "🐘 PostgreSQL Database with Enhanced Schema",
                "🏗️ User Management & Admin Features", 
                "📈 Enhanced Query Interface with History",
                "💹 Financial Data Analysis with DuckDB",
                "🖥️ Saved Screens - Save & Reuse Query Results",
                "🤖 RAG-Powered Chat with Earnings Transcripts",
                "📚 Citation-Based Responses from Financial Documents"
            ],
            "auth_types": ["regular", "premium_invitation"],
            "auth_flags": {
                "enable_login": settings.APPLICATION.ENABLE_LOGIN,
                "enable_self_serve_registration": settings.APPLICATION.ENABLE_SELF_SERVE_REGISTRATION,
                "enable_regular_auth": settings.APPLICATION.ENABLE_REGULAR_AUTH,
                "enable_premium_onboarding": settings.APPLICATION.ENABLE_PREMIUM_ONBOARDING
            },
            "docs_url": "/docs",
            "endpoints": {
                "query_stream": "/screener/query/stream",
                "query_sort": "/screener/query/sort",
                "query_complete_dataset": "/screener/query/complete-dataset",
                "query_expand_value": "/screener/query/expand-value",
                "query_paginate": "/screener/query/paginate",
                "query_parallel_quarters": "/screener/query/parallel/quarters",
                "query_parallel_companies": "/screener/query/parallel/companies",
                "validate_token": "/auth/validate",
                "regular_register": "/auth/register",
                "premium_onboard": "/onboard/{invitation_code}",
                "screens_save": "/screens/save",
                "screens_list": "/screens/list",
                "screens_get": "/screens/{screen_id}",
                "screens_update": "/screens/{screen_id}",
                "screens_delete": "/screens/{screen_id}",
                "user_usage": "/user/usage",
                "user_profile": "/user/profile",
                "user_onboarding_status": "/user/onboarding-status",
                "user_complete_onboarding": "/user/complete-onboarding",
                "chat_message": "/chat/message",
                "chat_history": "/chat/history",
                "chat_history_by_id": "/chat/history/{chat_id}",
                "chat_stats": "/chat/stats",
                "chat_export": "/chat/export",
                "chat_clear": "/chat/clear",
                "chat_conversations": "/chat/conversations",
                "chat_conversation_by_id": "/chat/conversations/{conversation_id}"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint that doesn't require authentication"""
        try:
            # Check memory usage
            memory_mb = get_memory_usage()
            memory_status = "healthy" if memory_mb < settings.SERVER.MEMORY_LIMIT_MB * settings.SERVER.MEMORY_WARNING_THRESHOLD else "warning"
            
            # Check analyzer
            analyzer_instance = get_analyzer_instance()
            analyzer_status = "available" if analyzer_instance else "unavailable"
            
            # Check RAG system
            try:
                from agent import Agent as RAGSystem
                rag_status = "available"
            except ImportError:
                rag_status = "unavailable"
            
            # Check PostgreSQL without authentication dependency
            db_pool = get_db_pool()
            db_status = "unknown"
            system_tables = 0
            pool_size = 0
            free_connections = 0
            
            try:
                if db_pool:
                    async with db_pool.acquire() as db:
                        await db.fetchval("SELECT 1")
                        db_status = "connected"
                        
                        # Check database tables
                        system_tables = await db.fetchval("""
                            SELECT COUNT(*) FROM information_schema.tables 
                            WHERE table_name IN ('users', 'query_history')
                        """)
                        
                        # Get pool statistics
                        pool_size = db_pool.get_size()
                        free_connections = db_pool.get_idle_size()
                else:
                    db_status = "pool_not_initialized"
            except Exception as db_error:
                db_status = f"error: {str(db_error)}"
            
            # Check Redis
            from app.lifespan import get_redis_client
            redis_client = get_redis_client()
            redis_status = "unknown"
            try:
                if redis_client:
                    await redis_client.ping()
                    redis_status = "connected"
                else:
                    redis_status = "not_configured"
            except Exception as redis_error:
                redis_status = f"error: {str(redis_error)}"
            
            # Determine overall health status
            is_healthy = db_status == "connected" and memory_status == "healthy"
            
            return {
                "status": "healthy" if is_healthy else "unhealthy", 
                "message": "All systems operational" if is_healthy else f"Database issues: {db_status}",
                "database": db_status,
                "database_pool": {
                    "size": pool_size,
                    "idle_connections": free_connections
                },
                "redis": redis_status,
                "memory": {
                    "usage_mb": round(memory_mb, 1),
                    "limit_mb": settings.SERVER.MEMORY_LIMIT_MB,
                    "status": memory_status
                },
                "analyzer": analyzer_status,
                "rag_system": rag_status,
                "system_tables": system_tables,
                "features": {
                    "financial_queries": analyzer_status == "available",
                    "chat_rag": rag_status == "available",
                    "streaming": True,
                    "authentication": True,
                    "premium_onboarding": True
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "message": f"Service issues: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

