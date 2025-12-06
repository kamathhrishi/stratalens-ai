"""
Database initialization utilities for StrataLens AI
Handles PostgreSQL database setup, table creation, and schema management
"""

import asyncpg
from typing import Optional
from config import settings, get_database_pool_config
from app.utils.logging_utils import log_info
from app.routers.screens import init_screens_tables


async def setup_connection(connection):
    """Setup function for database connections to prevent concurrent access issues"""
    await connection.execute(f"SET application_name = '{settings.DATABASE.APPLICATION_NAME}'")
    await connection.execute(f"SET statement_timeout = {settings.DATABASE.STATEMENT_TIMEOUT_MS}")
    await connection.execute(f"SET idle_in_transaction_session_timeout = {settings.DATABASE.IDLE_IN_TRANSACTION_TIMEOUT_MS}")


async def init_database(database_url: str, db_pool: Optional[asyncpg.Pool] = None) -> asyncpg.Pool:
    """
    Initialize PostgreSQL database tables and return the database pool
    
    Args:
        database_url: PostgreSQL connection string
        db_pool: Optional existing database pool (if None, will create new one)
    
    Returns:
        asyncpg.Pool: The initialized database pool
    """
    # Get database pool configuration from centralized settings
    pool_config = get_database_pool_config()
    min_size = pool_config['min_size']
    max_size = pool_config['max_size']
    command_timeout = pool_config['command_timeout']
    timeout = pool_config['timeout']
    
    if settings.ENVIRONMENT.is_production:
        log_info("🏭 Production environment detected - using stricter connection limits")
    else:
        log_info("🏠 Development environment detected - using increased connection limits to prevent conflicts")
    
    # Create database pool if not provided
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            database_url, 
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            timeout=timeout,
            # Improved connection pool settings to prevent concurrent access issues
            setup=setup_connection,
            server_settings={
                'application_name': settings.DATABASE.APPLICATION_NAME,
                'tcp_keepalives_idle': str(settings.DATABASE.TCP_KEEPALIVES_IDLE),
                'tcp_keepalives_interval': str(settings.DATABASE.TCP_KEEPALIVES_INTERVAL),
                'tcp_keepalives_count': str(settings.DATABASE.TCP_KEEPALIVES_COUNT)
            }
        )
    
    log_info(f"🔧 Database pool created: min_size={min_size}, max_size={max_size}")
    
    async with db_pool.acquire() as conn:
        # Check what columns exist in the users table
        try:
            existing_columns = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'users' AND table_schema = 'public'
            """)
            existing_column_names = [row['column_name'] for row in existing_columns]
            log_info(f"📋 Existing users table columns: {existing_column_names}")
        except:
            existing_column_names = []
        
        # Create users table only if it doesn't exist
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE,
                full_name VARCHAR(255) NOT NULL,
                first_name VARCHAR(255),
                last_name VARCHAR(255),
                company VARCHAR(255),
                role VARCHAR(100),
                organization VARCHAR(255),
                department VARCHAR(255),
                title VARCHAR(255),
                hashed_password VARCHAR(255),
                is_active BOOLEAN DEFAULT FALSE,
                is_approved BOOLEAN DEFAULT FALSE,
                is_admin BOOLEAN DEFAULT FALSE,
                access_level VARCHAR(50) DEFAULT 'standard',
                onboarded_via_invitation BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                approval_notes TEXT,
                has_completed_onboarding BOOLEAN DEFAULT FALSE,
                first_login_at TIMESTAMP
            )
        ''')
        
        # Add missing columns if needed (safely)
        missing_columns = {
            'username': 'VARCHAR(50) UNIQUE NOT NULL',
            'hashed_password': 'VARCHAR(255)',
            'first_name': 'VARCHAR(255)',
            'last_name': 'VARCHAR(255)',
            'organization': 'VARCHAR(255)',
            'department': 'VARCHAR(255)',
            'title': 'VARCHAR(255)',
            'access_level': "VARCHAR(50) DEFAULT 'standard'",
            'onboarded_via_invitation': 'BOOLEAN DEFAULT FALSE',
            'is_admin': 'BOOLEAN DEFAULT FALSE',
            'has_completed_onboarding': 'BOOLEAN DEFAULT FALSE',
            'first_login_at': 'TIMESTAMP'
        }
        
        for col_name, col_definition in missing_columns.items():
            if col_name not in existing_column_names:
                try:
                    await conn.execute(f'ALTER TABLE users ADD COLUMN {col_name} {col_definition}')
                    log_info(f"✅ Added {col_name} column")
                except:
                    pass

        # Create invitations table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS invitations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                invitation_code VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) NOT NULL,
                first_name VARCHAR(255) NOT NULL,
                last_name VARCHAR(255) NOT NULL,
                organization VARCHAR(255),
                department VARCHAR(255),
                title VARCHAR(255),
                access_level VARCHAR(50) DEFAULT 'standard',
                invitation_url TEXT,
                is_used BOOLEAN DEFAULT FALSE,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_at TIMESTAMP,
                custom_fields JSONB DEFAULT '{}'
            )
        ''')
        
        # User preferences table
        await conn.execute(f'''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                default_page_size INTEGER DEFAULT {settings.DATABASE.DEFAULT_PAGE_SIZE},
                preferred_sectors JSONB DEFAULT '[]',
                email_notifications BOOLEAN DEFAULT TRUE,
                theme VARCHAR(20) DEFAULT 'light',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Enhanced query history table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS query_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                query_type VARCHAR(20) DEFAULT 'single_sheet',
                sql_query_generated TEXT,
                success BOOLEAN NOT NULL,
                execution_time REAL,
                error_message TEXT,
                tables_used JSONB,
                result_count INTEGER,
                used_cache BOOLEAN DEFAULT FALSE,
                companies JSONB DEFAULT '[]',
                sheet_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        
        # Add missing columns to existing query_history table
        try:
            existing_qh_columns = await conn.fetch("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'query_history' AND table_schema = 'public'
            """)
            existing_qh_column_names = [row['column_name'] for row in existing_qh_columns]
            
            qh_missing_columns = {
                'query_type': "VARCHAR(20) DEFAULT 'single_sheet'",
                'companies': "JSONB DEFAULT '[]'",
                'sheet_count': 'INTEGER DEFAULT 1'
            }
            
            for col_name, col_definition in qh_missing_columns.items():
                if col_name not in existing_qh_column_names:
                    try:
                        await conn.execute(f'ALTER TABLE query_history ADD COLUMN {col_name} {col_definition}')
                        log_info(f"✅ Added {col_name} to query_history")
                    except Exception as e:
                        log_info(f"❌ Failed to add {col_name} to query_history: {e}")
        except Exception as e:
            log_info(f"⚠️ Could not check query_history columns: {e}")
        
        # Create usage tracking table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS user_usage (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                request_date DATE NOT NULL,
                request_hour INTEGER NOT NULL,
                request_count INTEGER DEFAULT 1,
                total_cost DECIMAL(10,4) DEFAULT 0.02,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, request_date, request_hour)
            )
        ''')
        log_info("✅ Usage tracking table created")
        
        # Create comprehensive search queries table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS search_queries (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                username VARCHAR(255) NOT NULL,
                question TEXT NOT NULL,
                query_type VARCHAR(50) DEFAULT 'single_sheet',
                status VARCHAR(50) NOT NULL, -- 'started', 'completed', 'failed', 'cancelled'
                execution_time REAL,
                error_message TEXT,
                sql_query_generated TEXT,
                tables_used JSONB,
                result_count INTEGER,
                companies JSONB,
                sheet_count INTEGER,
                used_cache BOOLEAN DEFAULT FALSE,
                ip_address INET,
                user_agent TEXT,
                session_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        log_info("✅ Search queries table created")
        
        # Create system logs table for comprehensive logging
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                username VARCHAR(255),
                message TEXT NOT NULL,
                level VARCHAR(20) DEFAULT 'INFO',
                is_milestone BOOLEAN DEFAULT FALSE,
                extra_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        log_info("✅ System logs table created")
        
        # Create indexes for better performance
        indexes = [
            ('idx_users_username', 'users', 'username'),
            ('idx_users_email', 'users', 'email'),
            ('idx_search_queries_user_id', 'search_queries', 'user_id'),
            ('idx_search_queries_created_at', 'search_queries', 'created_at'),
            ('idx_search_queries_status', 'search_queries', 'status'),
            ('idx_search_queries_query_type', 'search_queries', 'query_type'),
            ('idx_system_logs_user_id', 'system_logs', 'user_id'),
            ('idx_system_logs_created_at', 'system_logs', 'created_at'),
            ('idx_system_logs_level', 'system_logs', 'level'),
            ('idx_invitations_code', 'invitations', 'invitation_code'),
            ('idx_query_history_user_id', 'query_history', 'user_id'),
            ('idx_query_history_created_at', 'query_history', 'created_at'),
            ('idx_query_history_query_type', 'query_history', 'query_type'),
            ('idx_user_usage_user_id', 'user_usage', 'user_id'),
            ('idx_user_usage_date', 'user_usage', 'request_date'),
            ('idx_user_usage_user_date', 'user_usage', 'user_id, request_date'),
            ('idx_chat_history_user_id', 'chat_history', 'user_id'),
            ('idx_chat_history_created_at', 'chat_history', 'created_at'),
            ('idx_chat_conversations_user_id', 'chat_conversations', 'user_id'),
            ('idx_chat_conversations_updated_at', 'chat_conversations', 'updated_at'),
            ('idx_chat_messages_conversation_id', 'chat_messages', 'conversation_id'),
            ('idx_chat_messages_created_at', 'chat_messages', 'created_at'),
            ('idx_chat_analytics_user_id', 'chat_analytics', 'user_id'),
            ('idx_chat_analytics_ip_address', 'chat_analytics', 'ip_address'),
            ('idx_chat_analytics_user_type', 'chat_analytics', 'user_type'),
            ('idx_chat_analytics_created_at', 'chat_analytics', 'created_at'),
            ('idx_chat_analytics_success', 'chat_analytics', 'success'),
        ]
        
        for index_name, table_name, column_name in indexes:
            try:
                await conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name})')
            except Exception as e:
                log_info(f"⚠️ Could not create index {index_name}: {e}")
        
        # Initialize screens tables
        log_info("🖥️  Initializing saved screens tables...")
        await init_screens_tables(db_pool)
        log_info("✅ Saved screens tables initialized")
        
        # Create query_results_log table for complete query results logging
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS query_results_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                question TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                query_type VARCHAR(20) DEFAULT 'single_sheet',
                sql_query_generated TEXT,
                tables_used TEXT[],
                results JSONB, -- Complete results data
                error_message TEXT,
                execution_time FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        log_info("✅ Query results log table created")
        
        # Create chat_conversations table for conversation threads (like ChatGPT)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_conversations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL, -- Auto-generated title from first message
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        log_info("✅ Chat conversations table created")
        
        # Create chat_messages table for individual messages within conversations
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                conversation_id UUID REFERENCES chat_conversations(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
                content TEXT NOT NULL,
                citations JSONB DEFAULT '[]',
                context JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        log_info("✅ Chat messages table created")
        
        # Create chat_history table for backward compatibility (will migrate data)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                context JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        log_info("✅ Chat history table created (legacy)")
        
        # Create chat analytics table for tracking search queries
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_analytics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE SET NULL,
                ip_address INET NOT NULL,
                user_type VARCHAR(20) NOT NULL CHECK (user_type IN ('demo', 'authorized')),
                query_text TEXT NOT NULL,
                query_length INTEGER NOT NULL,
                comprehensive_search BOOLEAN DEFAULT TRUE,
                success BOOLEAN NOT NULL,
                response_time_ms REAL,
                citations_count INTEGER DEFAULT 0,
                error_message TEXT,
                user_agent TEXT,
                session_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        log_info("✅ Chat analytics table created")
    
    return db_pool
