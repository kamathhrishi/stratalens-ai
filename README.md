# StrataLens AI

Stratalens AI is equity research platform. You can ask questions and get answers to questions from 10K filings, earnings calls and news.

**Live Platform:** [stratalens.ai](https://stratalens.ai)

## Agent System

Core agent system implementing **Retrieval-Augmented Generation (RAG)** with **intelligent tool routing** and **self-reflection** for financial Q&A. This powers the chat and analysis features on stratalens.ai.

### Architecture Overview

```
                              AGENT PIPELINE
 ═══════════════════════════════════════════════════════════════════════

 ┌──────────┐    ┌───────────────────┐    ┌──────────────────────────┐
 │ Question │───►│ Question Analyzer │───►│     Tool Selection       │
 └──────────┘    │   (Cerebras LLM)  │    │                          │
                 │                   │    │  • Earnings Transcripts  │
                 │ Extracts:         │    │  • SEC 10-K Filings      │
                 │ • Tickers         │    │  • Real-Time News        │
                 │ • Time periods    │    │                          │
                 │ • Data source     │    └────────────┬─────────────┘
                 └───────────────────┘                 │
                                                       ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                  RETRIEVAL LAYER                     │
                 │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
                 │  │  Earnings   │  │  SEC 10-K   │  │   Tavily    │  │
                 │  │ Transcripts │  │   Filings   │  │    News     │  │
                 │  │             │  │             │  │             │  │
                 │  │ Vector DB   │  │ Vector DB   │  │  Live API   │  │
                 │  │ + Hybrid    │  │ + Section   │  │             │  │
                 │  │   Search    │  │   Routing   │  │             │  │
                 │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
                 └─────────┴───────────┬────┴────────────────┴─────────┘
                                       │ ▲
                                       │ │ Re-query with
                                       │ │ different sources
                                       ▼ │
                 ┌─────────────────────────────────────────────────────┐
                 │               ITERATIVE IMPROVEMENT                  │
                 │                                                      │
                 │    ┌──────────┐    ┌──────────┐    ┌──────────┐     │
                 │    │ Generate │───►│ Evaluate │───►│ Iterate? │─────┼───┐
                 │    │  Answer  │    │ Quality  │    │          │     │   │
                 │    └──────────┘    └──────────┘    └──────────┘     │   │
                 │                                         │ NO        │   │ YES
                 └─────────────────────────────────────────┼───────────┘   │
                                                           ▼               │
                                                    ┌─────────────┐        │
                                                    │   ANSWER    │        │
                                                    │ + Citations │        │
                                                    └─────────────┘        │
                                                           ▲               │
                                                           └───────────────┘
```

**Key Concepts:**
1. **Question Analysis** - Cerebras LLM extracts tickers, time periods, and determines data sources
2. **Tool Routing** - Automatically routes to earnings transcripts, SEC filings, or news based on question
3. **Self-Reflection** - Evaluates answer quality and iterates until confident (≥90%) or max iterations reached

For detailed agent documentation, see [agent/README.md](agent/README.md).

---

## Features

- **Earnings Transcripts** (2022-2025) - Word-for-word executive commentary
- **SEC Filings** (10K of 2024-25) - Official 10-K and 10-Q reports
- **Financial Screener** - Natural language queries over company fundamentals [not in production]

Unlike generic LLMs that rely on web content, StrataLens uses the same authoritative documents that professional analysts depend on.

## Tech Stack

- **Backend:** FastAPI, PostgreSQL (pgvector), DuckDB
- **AI/ML:** OpenAI, Groq, LangChain, RAG (Retrieval-Augmented Generation)
- **Frontend:** Vanilla JS, Tailwind CSS

## Project Structure

```
stratalens_ai/
├── agent/                  # AI agent & RAG system         → see agent/README.md
│   ├── rag/               # RAG implementation
│   │   └── data_ingestion/# Data pipeline                  → see data_ingestion/README.md
│   └── screener/          # Financial screener
├── app/                   # FastAPI application
│   ├── routers/           # API endpoints
│   ├── schemas/           # Pydantic models
│   ├── auth/              # Authentication
│   └── websocket/         # WebSocket handlers
├── frontend/              # Web interface
├── db/                    # Database utilities
├── analytics/             # Usage analytics
└── experiments/           # Development & benchmarking (gitignored)
```

## Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ with pgvector extension
- See [Requirements](#requirements) for full dependency list

### Installation

```bash
# Clone repository
git clone https://github.com/kamathhrishi/stratalensai.git
cd stratalens_ai

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials

# Configure environment (see Configuration section below)
```

### Configuration

Before running the application, configure the following files based on your environment:

**Backend (`.env`):**
- `BASE_URL` - Set to your server URL (e.g., `localhost:8000` for local, your production URL for deployment)
- `RAG_DEBUG_MODE` - Set to `false` for production, `true` for development debugging
- `ENABLE_LOGIN` / `ENABLE_SELF_SERVE_REGISTRATION` - Toggle authentication features as needed

**Frontend (`frontend/config.js`):**
- `ENVIRONMENT` - Set to `'local'` for development or `'production'` for deployment
- Update the `ENVIRONMENTS.production` URLs to match your production server

```bash
# Initialize database
python utils/database_init.py

# Ingest data (optional - see data_ingestion/README.md)
python agent/rag/data_ingestion/create_tables.py
python agent/rag/data_ingestion/download_transcripts.py

# Run server
python fastapi_server.py
```

Access the application at `http://localhost:8000`

## Requirements

### API Keys

| Service | Environment Variable | Required |
|---------|---------------------|----------|
| OpenAI | `OPENAI_API_KEY` | Yes |
| Cerebras | `CEREBRAS_API_KEY` | Yes |
| API Ninjas | `API_NINJAS_KEY` | Yes |
| Tavily | `TAVILY_API_KEY` | Optional |
| Logfire | `LOGFIRE_TOKEN` | Optional |
| Google OAuth | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Optional |

### Database

- **PostgreSQL** with [pgvector](https://github.com/pgvector/pgvector) extension (`DATABASE_URL`)
- **Redis** (optional, for caching) (`REDIS_URL`)

### Python Dependencies

See `requirements.txt` for full list.

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

- `POST /message/stream-v2` - Chat with streaming RAG responses
- `GET /companies/search` - Search companies by ticker/name
- `GET /transcript/{ticker}/{year}/{quarter}` - Get specific earnings transcript
- `POST /screener/query/stream` - Natural language financial queries

## Data Sources

All downloaded data is stored in `agent/rag/data_downloads/` (gitignored):
- Earnings transcripts (~1-2GB per 1000 companies)
- Vector embeddings (~500MB per 1000 companies)
- SEC filings (~5-10GB per 500 companies)

See `agent/rag/data_ingestion/README.md` for detailed ingestion instructions.

## AI Agent Documentation

For detailed documentation on the AI agent architecture and RAG system, see:

- **[agent/README.md](agent/README.md)** - Complete agent architecture, RAG pipeline, self-reflection system, and usage examples
- **[agent/rag/data_ingestion/README.md](agent/rag/data_ingestion/README.md)** - Data ingestion scripts for transcripts, embeddings, and SEC filings

## Development Status

**Production (stratalens.ai):**
- Earnings transcript chat with RAG
- SEC 10-K filings (2024-25)
- Real-time streaming responses
- User authentication

**In Development:**
- Enhanced financial screener
- Performance optimizations

## Contributing

Contributions welcome! Please open an issue to discuss major changes before submitting PRs.

## License

MIT License - see LICENSE file for details

## Contact

For questions or access requests: hrishi@stratalens.ai
