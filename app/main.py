"""
Application Entry Point

Main entry point for running the FastAPI server.
"""

import os
from dotenv import load_dotenv
import uvicorn

from app import app
from config import settings
from app.utils.logging_utils import log_info

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        log_info("⚠️  WARNING: OPENAI_API_KEY not set. Some features may not work.")
        log_info("Set it with: export OPENAI_API_KEY='your-key-here'")
    
    if not os.getenv("DATABASE_URL"):
        log_info("⚠️  WARNING: DATABASE_URL not set. Database features may not work.")
        log_info("Set it with: export DATABASE_URL='your-database-url'")
    
    port = settings.get_server_port()
    
    log_info("🚀 Starting StrataLens Enhanced API server...")
    log_info(f"🌐 Server will run on: http://localhost:{port}")
    log_info(f"📖 API Documentation: http://localhost:{port}/docs")
    try:
        from agent.screener import FinancialDataAnalyzer  # noqa: F401
        log_info("🧠 Financial Analysis Support: ✅ Enabled")
    except ImportError:
        log_info("🧠 Financial Analysis Support: ❌ Disabled (analyzer not available)")
    
    uvicorn.run(
        "app:app",  # Use the app module
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )

