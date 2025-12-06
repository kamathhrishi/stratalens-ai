"""
FastAPI Application Factory

This module creates and configures the FastAPI application instance.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from config import settings
from app.lifespan import lifespan
from app.middleware import setup_middleware
from app.routes import setup_routes
from app.utils.logfire_config import init_logfire, is_logfire_active

# Create FastAPI app
app = FastAPI(
    title=settings.APPLICATION.TITLE,
    description=settings.APPLICATION.DESCRIPTION,
    version=settings.APPLICATION.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.APPLICATION.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.APPLICATION.ENABLE_REDOC else None
)

# Instrument FastAPI with Logfire (must be done before adding routes)
try:
    import logfire
    if is_logfire_active():
        logfire.instrument_fastapi(app)
        from app.utils.logging_utils import log_info
        log_info("✅ FastAPI instrumented with Logfire")
except Exception as e:
    from app.utils.logging_utils import log_warning
    log_warning(f"⚠️ Could not instrument FastAPI with Logfire: {e}")

# Setup middleware
setup_middleware(app)

# Setup routes
setup_routes(app)

# Mount static files directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

