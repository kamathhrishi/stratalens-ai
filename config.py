"""
Centralized configuration for StrataLens FastAPI server.

This module contains all configuration constants and settings that were previously
hardcoded throughout the application. It provides a clean, organized way to manage
all application settings in one place.

Usage:
    from config import settings
    
    # Access configuration values
    rate_limit = settings.RATE_LIMITING.PER_MINUTE
    db_config = settings.DATABASE.CONNECTION_POOL
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ClerkConfig:
    """Clerk authentication configuration."""

    # Clerk API keys (loaded from environment)
    SECRET_KEY: Optional[str] = field(default_factory=lambda: os.getenv("CLERK_SECRET_KEY"))
    PUBLISHABLE_KEY: Optional[str] = field(default_factory=lambda: os.getenv("CLERK_PUBLISHABLE_KEY"))
    WEBHOOK_SECRET: Optional[str] = field(default_factory=lambda: os.getenv("CLERK_WEBHOOK_SECRET"))

    # Clerk JWKS configuration
    JWKS_CACHE_TTL_SECONDS: int = 3600  # Cache JWKS for 1 hour

    @property
    def is_configured(self) -> bool:
        """Check if Clerk is properly configured."""
        return bool(self.SECRET_KEY and self.PUBLISHABLE_KEY)

    @property
    def jwks_url(self) -> Optional[str]:
        """Get the JWKS URL from the publishable key."""
        if not self.PUBLISHABLE_KEY:
            return None
        # Extract the Clerk frontend API from publishable key
        # Format: pk_test_xxx or pk_live_xxx
        # The JWKS URL is at the Clerk frontend API
        return None  # Will be set dynamically based on issuer from token


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""

    # JWT Configuration (legacy - kept for backwards compatibility during migration)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    DEFAULT_TOKEN_EXPIRE_MINUTES: int = 15  # Default for create_access_token

    # Admin account defaults
    ADMIN_USERNAME: str = "admin"
    ADMIN_EMAIL: str = "admin@stratalens.ai"
    ADMIN_PASSWORD: str = "CHANGE_ME_IN_PRODUCTION"  # Must be changed via environment variable
    ADMIN_FULL_NAME: str = "System Administrator"


@dataclass
class RateLimitingConfig:
    """Rate limiting configuration."""

    # Rate limits
    PER_MINUTE: int = 30
    PER_MONTH: int = 10000
    ADMIN_PER_MONTH: int = 100000
    
    # Billing
    COST_PER_REQUEST: float = 0.02  # $0.02 per request
    
    # Cleanup intervals
    CLEANUP_INTERVAL_SECONDS: int = 60


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    # Connection pool settings
    class ConnectionPool:
        PRODUCTION_MIN_SIZE: int = 5
        PRODUCTION_MAX_SIZE: int = 30  # Increased from 10 to support more concurrent streaming requests
        PRODUCTION_COMMAND_TIMEOUT: int = 20
        PRODUCTION_TIMEOUT: int = 15
        
        DEVELOPMENT_MIN_SIZE: int = 10
        DEVELOPMENT_MAX_SIZE: int = 50  # Increased from 20 for development multi-user testing
        DEVELOPMENT_COMMAND_TIMEOUT: int = 30
        DEVELOPMENT_TIMEOUT: int = 20
    
    # Statement timeouts
    STATEMENT_TIMEOUT_MS: int = 30000  # 30 seconds
    IDLE_IN_TRANSACTION_TIMEOUT_MS: int = 60000  # 1 minute
    
    # Application name
    APPLICATION_NAME: str = "stratalens_fastapi"
    
    # TCP keepalive settings
    TCP_KEEPALIVES_IDLE: int = 600
    TCP_KEEPALIVES_INTERVAL: int = 30
    TCP_KEEPALIVES_COUNT: int = 3
    
    # Default page size for queries
    DEFAULT_PAGE_SIZE: int = 20
    
    # Connection pool instance
    CONNECTION_POOL: ConnectionPool = field(default_factory=ConnectionPool)


@dataclass
class RedisConfig:
    """Redis configuration for WebSocket sessions."""
    
    # Connection settings
    TIMEOUT: int = 30  # Connection timeout in seconds
    
    # Session settings
    SESSION_TIMEOUT: int = 3600  # 1 hour default session timeout
    
    # Cleanup settings
    CLEANUP_INTERVAL_SECONDS: int = 60


@dataclass
class FilePathsConfig:
    """File paths configuration."""
    
    # Database files
    DUCKDB_FILENAME: str = "agent/screener/financial_data_new.duckdb"
    
    # CSV files
    USERS_CSV_PATH: str = "users.csv"
    INVITATIONS_CSV_PATH: str = "invitations.csv"
    
    # Log files
    LOG_DIRECTORY: str = "logs"
    MAIN_LOG_FILE: str = "stratalens.log"


@dataclass
class ServerConfig:
    """Server configuration."""
    
    # Ports
    DEFAULT_PORT: int = 8000
    DEFAULT_WEBSOCKET_PORT: int = 8765
    
    # Hosts
    DEFAULT_HOST: str = "0.0.0.0"
    
    # Memory limits
    MEMORY_LIMIT_MB: int = 1024  # 1GB memory limit for health checks
    MEMORY_WARNING_THRESHOLD: float = 0.8  # 80% of limit triggers warning
    
    # CORS origins (comma-separated) - Strict production configuration
    # Production domains: stratalens.ai, www.stratalens.ai, Railway deployment
    # Development: localhost and 127.0.0.1 for local development
    DEFAULT_CORS_ORIGINS: str = (
        "https://stratalens.ai,"
        "https://www.stratalens.ai,"
        "https://web-production-835f4.up.railway.app,"
        "http://localhost:3000,"
        "http://localhost:8000,"
        "http://127.0.0.1:3000,"
        "http://127.0.0.1:8000,"
        "http://localhost:5000,"
        "http://127.0.0.1:5000"
    )
    
    # Development hosts (for CORS)
    DEV_HOSTS: List[str] = field(default_factory=lambda: ["localhost", "127.0.0.1"])


@dataclass
class WebSocketConfig:
    """WebSocket configuration."""
    
    # Message retry settings
    MAX_RETRIES: int = 2
    RETRY_BACKOFF_FACTOR: float = 0.1  # Exponential backoff factor
    
    # Connection settings
    CONNECTION_DELAY_SECONDS: float = 0.1  # Small delay to ensure connection is established
    
    # Background task settings
    ANALYSIS_PROGRESS_UPDATE_INTERVAL: int = 50  # Progress percentage for updates


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    # Log levels
    DEFAULT_LOG_LEVEL: str = "INFO"
    
    # Log formats
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    MAX_LOG_SIZE_MB: int = 10
    BACKUP_COUNT: int = 5


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    
    # Environment detection
    ENVIRONMENT: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == 'development'


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    
    # App metadata
    TITLE: str = "StrataLens Complete API"
    DESCRIPTION: str = "Complete financial data analysis platform with intelligent queries, regular auth AND premium CSV onboarding"
    VERSION: str = "5.0.0"
    
    # Feature flags
    ENABLE_DOCS: bool = False  # Disable Swagger UI by default
    ENABLE_REDOC: bool = False  # Disable ReDoc by default
    
    # Database features
    ENABLE_DATABASE: bool = True
    ENABLE_REDIS: bool = True
    ENABLE_ANALYZER: bool = True
    
    # Authentication features
    ENABLE_REGULAR_AUTH: bool = True
    ENABLE_PREMIUM_ONBOARDING: bool = True
    ENABLE_CSV_USER_LOADING: bool = True
    
    # Fine-grained auth flags (admin-togglable via env vars)
    # When disabled, server will reject the corresponding endpoint and UI should hide it
    ENABLE_LOGIN: bool = field(default_factory=lambda: os.getenv('ENABLE_LOGIN', 'false').lower() == 'true')
    ENABLE_SELF_SERVE_REGISTRATION: bool = field(default_factory=lambda: os.getenv('ENABLE_SELF_SERVE_REGISTRATION', 'false').lower() == 'true')

    # Auth bypass - currently disabled for all environments
    # Set AUTH_DISABLED=false to re-enable authentication
    AUTH_DISABLED: bool = field(default_factory=lambda: os.getenv('AUTH_DISABLED', 'true').lower() == 'true')
    
    # Query features
    ENABLE_STREAMING: bool = True
    ENABLE_WEBSOCKET: bool = True
    ENABLE_SAVED_SCREENS: bool = True


class Settings:
    """
    Main settings class that aggregates all configuration sections.
    
    Usage:
        from config import settings
        
        # Access specific configurations
        rate_limit = settings.RATE_LIMITING.PER_MINUTE
        db_pool_config = settings.DATABASE.CONNECTION_POOL.PRODUCTION_MIN_SIZE
    """
    
    def __init__(self):
        self.CLERK = ClerkConfig()
        self.SECURITY = SecurityConfig()
        self.RATE_LIMITING = RateLimitingConfig()
        self.DATABASE = DatabaseConfig()
        self.REDIS = RedisConfig()
        self.FILE_PATHS = FilePathsConfig()
        self.SERVER = ServerConfig()
        self.WEBSOCKET = WebSocketConfig()
        self.LOGGING = LoggingConfig()
        self.ENVIRONMENT = EnvironmentConfig()
        self.APPLICATION = ApplicationConfig()
    
    def get_database_pool_config(self) -> Dict[str, int]:
        """
        Get database connection pool configuration based on environment.
        
        Returns:
            Dict with min_size, max_size, command_timeout, timeout
        """
        if self.ENVIRONMENT.is_production:
            return {
                'min_size': self.DATABASE.CONNECTION_POOL.PRODUCTION_MIN_SIZE,
                'max_size': self.DATABASE.CONNECTION_POOL.PRODUCTION_MAX_SIZE,
                'command_timeout': self.DATABASE.CONNECTION_POOL.PRODUCTION_COMMAND_TIMEOUT,
                'timeout': self.DATABASE.CONNECTION_POOL.PRODUCTION_TIMEOUT
            }
        else:
            return {
                'min_size': self.DATABASE.CONNECTION_POOL.DEVELOPMENT_MIN_SIZE,
                'max_size': self.DATABASE.CONNECTION_POOL.DEVELOPMENT_MAX_SIZE,
                'command_timeout': self.DATABASE.CONNECTION_POOL.DEVELOPMENT_COMMAND_TIMEOUT,
                'timeout': self.DATABASE.CONNECTION_POOL.DEVELOPMENT_TIMEOUT
            }
    
    def get_cors_origins(self) -> List[str]:
        """
        Get CORS origins from environment or use defaults.
        
        Returns:
            List of allowed origins
        """
        cors_origins = os.getenv("CORS_ORIGINS", self.SERVER.DEFAULT_CORS_ORIGINS)
        return [origin.strip() for origin in cors_origins.split(",")]
    
    def get_extended_cors_origins(self) -> List[str]:
        """
        Get extended CORS origins with strict production settings.
        
        Returns:
            List of allowed origins (strict in production, permissive in development)
        """
        allowed_origins = self.get_cors_origins()
        
        # Check if we're in development mode (localhost present in origins)
        is_development = any(host in str(allowed_origins) for host in self.SERVER.DEV_HOSTS)
        
        if is_development:
            # Development mode: allow wildcard and null for local development
            return ["*"]
        else:
            # Production mode: strict CORS - only allow explicitly listed origins
            return allowed_origins
    
    def get_duckdb_path(self) -> str:
        """
        Get the DuckDB file path, checking local and Railway volume locations.
        
        Returns:
            Path to the DuckDB file
        """
        # Both local and Railway use the same filename
        duckdb_path = self.FILE_PATHS.DUCKDB_FILENAME
        
        # Check if file exists
        if os.path.exists(duckdb_path):
            return duckdb_path
        
        # If file doesn't exist, return the path anyway (will create new DB)
        return duckdb_path
    
    def get_server_port(self) -> int:
        """Get server port from environment or use default."""
        return int(os.getenv("PORT", self.SERVER.DEFAULT_PORT))
    
    def get_redis_url(self) -> str:
        """Get Redis URL from environment or use default."""
        return os.getenv("REDIS_URL", "redis://localhost:6379")
    
    def get_database_url(self) -> str:
        """Get database URL from environment or use default."""
        return os.getenv("DATABASE_URL", "postgresql://postgres:changeme@localhost:5432/stratalens")
    
    def get_jwt_secret_key(self) -> str:
        """Get JWT secret key from environment or generate one."""
        import secrets
        return os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    
    def get_base_url(self) -> str:
        """Get base URL from environment or use default."""
        return os.getenv("BASE_URL", "http://localhost:8000")
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")
    
    def get_groq_api_key(self) -> Optional[str]:
        """Get Groq API key from environment."""
        return os.getenv("GROQ_API_KEY")


# Global settings instance
settings = Settings()


# Convenience functions for backward compatibility
def get_database_pool_config() -> Dict[str, int]:
    """Get database connection pool configuration."""
    return settings.get_database_pool_config()


def get_cors_origins() -> List[str]:
    """Get CORS origins."""
    return settings.get_cors_origins()


def get_extended_cors_origins() -> List[str]:
    """Get extended CORS origins."""
    return settings.get_extended_cors_origins()


def get_duckdb_path() -> str:
    """Get DuckDB file path."""
    return settings.get_duckdb_path()


# Environment variable helpers
def get_required_env_var(var_name: str, description: str = None) -> str:
    """Get a required environment variable or raise an error if not found."""
    value = os.getenv(var_name)
    if value is None:
        desc = f" ({description})" if description else ""
        raise ValueError(f"Required environment variable '{var_name}' is not set{desc}")
    return value


def get_optional_env_var(var_name: str, default: str = None) -> str:
    """Get an optional environment variable with a default value."""
    return os.getenv(var_name, default)


# Export commonly used configurations
__all__ = [
    'settings',
    'Settings',
    'ClerkConfig',
    'SecurityConfig',
    'RateLimitingConfig',
    'DatabaseConfig',
    'RedisConfig',
    'FilePathsConfig',
    'ServerConfig',
    'WebSocketConfig',
    'LoggingConfig',
    'EnvironmentConfig',
    'ApplicationConfig',
    'get_database_pool_config',
    'get_cors_origins',
    'get_extended_cors_origins',
    'get_duckdb_path',
    'get_required_env_var',
    'get_optional_env_var'
]
