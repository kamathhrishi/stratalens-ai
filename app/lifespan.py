"""
Application Lifespan Management

Handles startup and shutdown lifecycle of the FastAPI server.
"""

import asyncio
import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from config import settings
from app.auth.auth import hash_password
from db.db_utils import get_db, set_db_pool
from app.utils.database_init import init_database
from app.utils.logfire_config import init_logfire
from app.utils.logging_utils import log_error, log_info, log_warning

# Global instances (will be initialized in lifespan)
db_pool = None
redis_client = None
analyzer_instance = None
session_manager = None
background_task_manager = None
websocket_manager = None
stratalens_handlers = None

# Database configuration
DATABASE_URL = None
REDIS_URL = None
SECRET_KEY = None

# Import optional dependencies
try:
    from agent.screener import FinancialDataAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    FinancialDataAnalyzer = None

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from logging_config import (
        ComprehensiveLogger, get_comprehensive_logger, set_comprehensive_logger
    )
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    ComprehensiveLogger = None
    set_comprehensive_logger = None

# Rate limiting constants
REDIS_TIMEOUT = settings.REDIS.TIMEOUT


def log_stage_header(stage_num: int, emoji: str, title: str):
    """Helper function to log stage headers consistently"""
    log_info("\n" + "="*60)
    log_info(f"{emoji} STAGE {stage_num}: {title}")
    log_info("="*60)


def validate_environment_variables():
    """Validate and log environment variables"""
    # Required environment variables
    required_vars = [
        ('DATABASE_URL', 'PostgreSQL database connection string'),
        ('REDIS_URL', 'Redis connection URL for WebSocket sessions'),
        ('BASE_URL', 'Base URL for invitation links'),
        ('OPENAI_API_KEY', 'OpenAI API key for LLM operations'),
        ('GROQ_API_KEY', 'Groq API key for LLM operations'),
    ]
    
    # Optional environment variables
    optional_vars = [
        ('JWT_SECRET_KEY', 'JWT secret key (auto-generated if not provided)'),
        ('PORT', 'FastAPI server port (default: 8000)'),
        ('WEBSOCKET_HOST', 'WebSocket host (default: 0.0.0.0)'),
        ('WEBSOCKET_PORT', 'WebSocket port (default: 8765)'),
        ('BASE_URL', 'Base URL for the application (required for invitation links)')
    ]
    
    log_info("Required Environment Variables:")
    for i, (var_name, description) in enumerate(required_vars, 1):
        value = os.getenv(var_name)
        if value:
            # Mask sensitive values
            if 'API_KEY' in var_name or 'SECRET' in var_name or 'PASSWORD' in var_name:
                masked_value = value[:8] + '*' * (len(value) - 12) + value[-4:] if len(value) > 12 else '*' * len(value)
                log_info(f"  {i}. {var_name}: {masked_value} ✓")
            else:
                log_info(f"  {i}. {var_name}: {value} ✓")
        else:
            log_info(f"  {i}. {var_name}: NOT SET ❌")
    
    log_info("\nOptional Environment Variables:")
    for i, (var_name, description) in enumerate(optional_vars, len(required_vars) + 1):
        value = os.getenv(var_name)
        if value:
            # Mask sensitive values
            if 'SECRET' in var_name:
                masked_value = value[:8] + '*' * (len(value) - 12) + value[-4:] if len(value) > 12 else '*' * len(value)
                log_info(f"  {i}. {var_name}: {masked_value} ✓")
            else:
                log_info(f"  {i}. {var_name}: {value} ✓")
        else:
            log_info(f"  {i}. {var_name}: NOT SET (will use default) ⚠️")
    
    # Check if any required variables are missing
    missing_required = [var_name for var_name, _ in required_vars if not os.getenv(var_name)]
    if missing_required:
        log_info("❌ MISSING REQUIRED ENVIRONMENT VARIABLES:")
        for var_name in missing_required:
            log_info(f"   - {var_name}")
        log_info("\nPlease set these variables in your Railway project's Variables tab.")
        log_info("The application will continue to start but may fail during operation.")


async def create_default_admin():
    """Create default admin account if it doesn't exist"""
    global db_pool
    
    admin_username = settings.SECURITY.ADMIN_USERNAME
    admin_email = settings.SECURITY.ADMIN_EMAIL
    admin_password = settings.SECURITY.ADMIN_PASSWORD
    
    try:
        async with db_pool.acquire() as conn:
            # Check if admin exists
            existing_admin = await conn.fetchrow(
                "SELECT id FROM users WHERE username = $1 OR email = $2", 
                admin_username, admin_email
            )
            
            if not existing_admin:
                # Create admin account with minimal required fields
                hashed_password = hash_password(admin_password)
                
                admin_id = await conn.fetchval('''
                    INSERT INTO users 
                    (username, email, full_name, hashed_password, is_active, is_approved, is_admin)
                    VALUES ($1, $2, $3, $4, TRUE, TRUE, TRUE)
                    RETURNING id
                ''', admin_username, admin_email, settings.SECURITY.ADMIN_FULL_NAME, hashed_password)
                
                # Create default preferences
                await conn.execute('INSERT INTO user_preferences (user_id) VALUES ($1)', admin_id)
                
                log_info(f"✅ Default admin account created:")
                log_info(f"   👤 Username: {admin_username}")
                log_info(f"   📧 Email: {admin_email}")
                log_info(f"   🔐 Password: [REDACTED - check ADMIN_PASSWORD env var or config]")
                log_info(f"   ⚠️  CHANGE PASSWORD AFTER FIRST LOGIN!")
                
            else:
                log_info(f"ℹ️  Admin account already exists: {admin_username} ({admin_email})")
                
    except Exception as e:
        log_info(f"❌ Failed to create admin account: {e}")


async def initialize_redis_and_websocket():
    """Initialize Redis and WebSocket infrastructure"""
    global redis_client, session_manager, background_task_manager, websocket_manager, stratalens_handlers
    
    # Initialize Redis for WebSocket sessions
    if REDIS_AVAILABLE:
        try:
            log_info("🔄 Connecting to Redis for WebSocket sessions...")
            redis_client = redis.from_url(REDIS_URL, socket_timeout=REDIS_TIMEOUT, socket_connect_timeout=REDIS_TIMEOUT)
            # Test Redis connection
            await redis_client.ping()
            log_info(f"✅ Redis connected successfully: {REDIS_URL}")
        except Exception as e:
            log_info(f"❌ Redis connection failed: {e}")
            log_info("📝 WebSocket sessions will use memory-only storage")
            redis_client = None
    else:
        log_info("⚠️ Redis not available. WebSocket sessions will use memory-only storage")
        redis_client = None
    
    # Initialize WebSocket components
    log_info("🔌 Initializing WebSocket components...")
    from app.websocket import SessionManager, BackgroundTaskManager, WebSocketManager, StrataLensWebSocketHandlers
    
    session_manager = SessionManager(redis_client)
    background_task_manager = BackgroundTaskManager()
    websocket_manager = WebSocketManager(session_manager)
    
    # Initialize StrataLens-specific handlers
    stratalens_handlers = StrataLensWebSocketHandlers(websocket_manager, session_manager)
    
    # Set up periodic cleanup task for WebSocket connections
    async def cleanup_websocket_connections():
        while True:
            try:
                await asyncio.sleep(settings.RATE_LIMITING.CLEANUP_INTERVAL_SECONDS)
                if websocket_manager:
                    cleaned = await websocket_manager.cleanup_stale_connections()
                    if cleaned > 0:
                        log_info(f"🧹 Cleaned up {cleaned} stale WebSocket connections")
            except Exception as e:
                log_error(f"❌ Error in WebSocket cleanup task: {e}")
    
    # Start the cleanup task
    asyncio.create_task(cleanup_websocket_connections())
    log_info("✅ WebSocket components initialized successfully")


async def initialize_database():
    """Initialize PostgreSQL database"""
    global db_pool
    
    # Test database connection and tables
    log_info("🔍 Testing database connection and query_history table...")
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                # Test basic connection
                test_result = await conn.fetchval("SELECT 1")
                log_info(f"✅ Database connection test passed: {test_result}")
                
                # Check if query_history table exists
                table_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'query_history'
                    )
                """)
                log_info(f"✅ Query history table exists: {table_exists}")
                
                if table_exists:
                    # Check table structure
                    columns = await conn.fetch("""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = 'query_history'
                        ORDER BY ordinal_position
                    """)
                    log_info(f"✅ Query history table columns: {[col['column_name'] for col in columns]}")
                else:
                    log_warning("⚠️ Query history table does not exist!")
                    
    except Exception as e:
        log_error(f"❌ Database connection test failed: {e}")
        log_error(f"❌ Traceback: {traceback.format_exc()}")
    
    # Initialize PostgreSQL
    log_info("🐘 Setting up PostgreSQL database...")
    try:
        db_pool = await init_database(DATABASE_URL)
        # Set the global db_pool for dependency injection
        set_db_pool(db_pool)
        log_info("✅ Database initialized successfully")
        
        # Set connection pool limits for AWS
        if db_pool:
            log_info(f"🔧 Database pool size: {db_pool.get_size()}")
            log_info(f"🔧 Database pool idle connections: {db_pool.get_idle_size()}")
            
            # Monitor connection pool health
            if settings.ENVIRONMENT.is_production:
                log_info("🏭 Production environment detected - monitoring connection pool")
        
        # Test database connection pool health
        if db_pool:
            log_info("🔍 Testing database connection pool health...")
            try:
                async with db_pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                    log_info(f"✅ Database connection pool health check passed: {result}")
            except Exception as e:
                log_info(f"⚠️ Database connection pool health check failed: {e}")
                log_info("🔄 Attempting to reinitialize database connection...")
                try:
                    await db_pool.close()
                    await init_database(DATABASE_URL)
                    log_info("✅ Database connection pool reinitialized successfully")
                except Exception as reinit_error:
                    log_info(f"❌ Database reinitialization failed: {reinit_error}")
        
        # Initialize comprehensive logging system
        if LOGGING_AVAILABLE and db_pool:
            log_info("📝 Initializing comprehensive logging system...")
            comprehensive_logger = ComprehensiveLogger(db_pool)
            set_comprehensive_logger(comprehensive_logger)
            log_info("✅ Comprehensive logging system initialized")
        else:
            log_info("⚠️ Comprehensive logging not available - using console-only logging")
            
    except Exception as e:
        log_info(f"❌ Database initialization failed: {e}")
        log_info("⚠️  App will start but database features won't work")
        log_info("💡 Add a PostgreSQL plugin in Railway or set DATABASE_URL variable")


async def initialize_analyzer_and_rag():
    """Initialize Financial Analyzer and RAG systems"""
    global analyzer_instance
    
    # Initialize analyzer (if available)
    log_info("🔍 Checking analyzer availability...")
    log_info(f"   ANALYZER_AVAILABLE: {ANALYZER_AVAILABLE}")
    log_info(f"   FinancialDataAnalyzer: {FinancialDataAnalyzer is not None}")
    
    if ANALYZER_AVAILABLE and FinancialDataAnalyzer:
        log_info("🤖 Initializing Financial Data Analyzer with PostgreSQL...")
        import time
        start_time = time.time()
        
        try:
            log_info("🔌 Creating FinancialDataAnalyzer instance...")
            analyzer_instance = FinancialDataAnalyzer(
                default_page_size=settings.DATABASE.DEFAULT_PAGE_SIZE
            )
            end_time = time.time()
            log_info(f"✅ Analyzer initialized successfully in {end_time - start_time:.2f} seconds.")
            
        except Exception as e:
            log_info(f"❌ Failed to initialize analyzer: {e}")
            log_info(f"🔍 Error type: {type(e).__name__}")
            traceback.print_exc()
            analyzer_instance = None
    else:
        log_info("⚠️  Analyzer not available - running in auth-only mode")
        log_info(f"   ANALYZER_AVAILABLE: {ANALYZER_AVAILABLE}")
        log_info(f"   FinancialDataAnalyzer: {FinancialDataAnalyzer}")
        analyzer_instance = None
    
    # Initialize RAG system (if available)
    log_info("🤖 Checking RAG system availability...")
    try:
        from agent import Agent as RAGSystem
        RAG_AVAILABLE = True
        log_info(f"   RAG_AVAILABLE: {RAG_AVAILABLE}")
    except ImportError:
        RAG_AVAILABLE = False
        log_info(f"   RAG_AVAILABLE: {RAG_AVAILABLE}")
    
    # RAG system is now initialized in the chat router
    log_info("ℹ️ RAG system will be initialized by chat router when needed")


async def setup_authentication_and_routing():
    """Set up authentication and routing configuration"""
    global analyzer_instance, db_pool
    
    from app.auth import auth
    from app.utils import rate_limiter, RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_MONTH, ADMIN_RATE_LIMIT_PER_MONTH
    from app.routers.screener import set_analyzer_instance
    from app.routers.users import set_user_globals
    
    # Set up centralized database and authentication
    log_info("🔐 Setting up centralized database and authentication...")
    set_db_pool(db_pool)
    auth.set_db_dependency(get_db)
    log_info("✅ Centralized database and authentication set up successfully")
    
    # Set analyzer instance in screener router
    if analyzer_instance:
        set_analyzer_instance(analyzer_instance)
        log_info("✅ Analyzer instance set in screener router")
    
    # Set up user router globals
    log_info("👤 Setting up user router globals...")
    set_user_globals(
        rate_limiter_instance=rate_limiter,
        rate_limits={
            'per_minute': RATE_LIMIT_PER_MINUTE,
            'per_month': RATE_LIMIT_PER_MONTH,
            'admin_per_month': ADMIN_RATE_LIMIT_PER_MONTH
        },
        cost_per_request=settings.RATE_LIMITING.COST_PER_REQUEST
    )
    log_info("✅ User router globals set up successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    =============================================================================
    🚀 STRATALENS SERVER LIFECYCLE MANAGER
    =============================================================================
    
    This function manages the complete startup and shutdown lifecycle of the FastAPI server.
    It initializes all core systems, validates configuration, and ensures proper cleanup.
    
    STAGES:
    1. 🔍 Environment Validation & Configuration
    2. 🔥 Observability & Logging Setup  
    3. 🔄 Redis & WebSocket Infrastructure
    4. 🐘 PostgreSQL Database Initialization
    5. 👤 User Management & Admin Setup
    6. 🤖 Financial Analyzer & RAG Systems
    7. 🔐 Authentication & Rate Limiting
    8. 🧹 Cleanup & Resource Management
    """
    global analyzer_instance, session_manager, background_task_manager, websocket_manager, stratalens_handlers, redis_client, db_pool
    global DATABASE_URL, REDIS_URL, SECRET_KEY
    
    try:
        log_info("🚀 Initializing StrataLens Complete API with WebSocket...")
        
        # STAGE 1: Environment Validation & Configuration
        log_stage_header(1, "🔍", "ENVIRONMENT VALIDATION & CONFIGURATION")
        validate_environment_variables()
        
        # TEMPORARY: Use defaults to get the app running
        log_info("⚠️  USING TEMPORARY DEFAULTS - APP WILL START BUT MAY NOT WORK FULLY")
        
        # Set required variables with defaults from centralized config
        DATABASE_URL = settings.get_database_url()
        REDIS_URL = settings.get_redis_url()
        
        # Set optional variables
        SECRET_KEY = settings.get_jwt_secret_key()
        
        log_info(f"📝 Using DATABASE_URL: {DATABASE_URL[:30]}...")
        log_info(f"📝 Using REDIS_URL: {REDIS_URL}")
        log_info("⚠️  Set proper environment variables in Railway for production use!")
        
        # STAGE 2: Observability & Logging Setup
        log_stage_header(2, "🔥", "OBSERVABILITY & LOGGING SETUP")
        log_info("\n🔥 Initializing Logfire observability...")
        init_logfire()
        print()
        
        # STAGE 3: Redis & WebSocket Infrastructure
        log_stage_header(3, "🔄", "REDIS & WEBSOCKET INFRASTRUCTURE")
        await initialize_redis_and_websocket()
        
        # STAGE 4: PostgreSQL Database Initialization
        log_stage_header(4, "🐘", "POSTGRESQL DATABASE INITIALIZATION")
        await initialize_database()
        
        # STAGE 5: User Management & Admin Setup
        log_stage_header(5, "👤", "USER MANAGEMENT & ADMIN SETUP")
        await create_default_admin()
        
        # STAGE 6: Financial Analyzer & RAG Systems
        log_stage_header(6, "🤖", "FINANCIAL ANALYZER & RAG SYSTEMS")
        await initialize_analyzer_and_rag()
        
        # STAGE 7: Authentication & Rate Limiting
        log_stage_header(7, "🔐", "AUTHENTICATION & RATE LIMITING")
        await setup_authentication_and_routing()
        
        # APPLICATION READY - YIELD CONTROL TO FASTAPI
        log_stage_header(8, "🎯", "APPLICATION READY - YIELDING CONTROL TO FASTAPI")
        log_info("✅ All systems initialized successfully!")
        log_info("🚀 StrataLens API is now ready to serve requests")
        log_info("="*60)
        
        yield
        
    except Exception as e:
        log_info(f"❌ CRITICAL: Failed to initialize: {e}")
        traceback.print_exc()
        analyzer_instance = None
    finally:
        # STAGE 8: Cleanup & Resource Management
        log_stage_header(8, "🧹", "CLEANUP & RESOURCE MANAGEMENT")
        log_info("🔄 Shutting down StrataLens API server...")
        
        # Close Redis connection
        if redis_client:
            try:
                await redis_client.close()
                log_info("🔄 Redis connection closed")
            except Exception as e:
                log_info(f"❌ Error closing Redis connection: {e}")
        
        # Close database pool
        if db_pool:
            await db_pool.close()
            log_info("🔄 Database connection pool closed")
        
        if analyzer_instance:
            log_info("🔄 Shutting down analyzer...")
            # Add any analyzer cleanup here if needed
        
        log_info("✅ Cleanup completed successfully")
        log_info("="*60)


# Export global instances for use in other modules
def get_analyzer_instance():
    """Get the global analyzer instance"""
    return analyzer_instance


def get_db_pool():
    """Get the global database pool"""
    return db_pool


def get_websocket_manager():
    """Get the global WebSocket manager"""
    return websocket_manager


def get_redis_client():
    """Get the global Redis client"""
    return redis_client

