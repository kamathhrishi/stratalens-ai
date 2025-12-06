"""
Middleware Configuration

Sets up CORS and other middleware for the FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_cors_origins, get_extended_cors_origins
from app.utils.logging_utils import log_info


def setup_middleware(app: FastAPI):
    """Configure middleware for the FastAPI application"""
    
    # Get CORS origins from centralized configuration
    allowed_origins = get_cors_origins()
    extended_origins = get_extended_cors_origins()
    
    log_info(f"🌐 CORS Configuration:")
    log_info(f"   Allowed origins: {allowed_origins}")
    log_info(f"   Extended origins: {extended_origins}")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=extended_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=[
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
            "Cache-Control",
            "Pragma",
            "User-Agent",
            "Referer",
        ],
        expose_headers=["Content-Length", "Content-Type", "Authorization"],
        max_age=86400,  # Cache preflight requests for 24 hours
    )

