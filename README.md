# StrataLens AI

Stratalens AI is equity research platform. You can ask questions and get answers to questions from 10K filings, earnings calls and news.

**Live Platform:** [stratalens.ai](https://stratalens.ai)

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

### Required API Keys

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY` | Response generation, embeddings |
| Groq | `GROQ_API_KEY` | Fast question analysis |
| API Ninjas | `API_NINJAS_KEY` | Transcript downloads |

### Optional API Keys

| Service | Environment Variable | Purpose |
|---------|---------------------|---------|
| Tavily | `TAVILY_API_KEY` | Web search augmentation |
| Logfire | `LOGFIRE_TOKEN` | Monitoring/observability |
| Google OAuth | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Google login |

### Database

| Service | Environment Variable | Notes |
|---------|---------------------|-------|
| PostgreSQL | `DATABASE_URL` | Requires [pgvector](https://github.com/pgvector/pgvector) extension |
| Redis | `REDIS_URL` | Optional, for caching |

### Python Dependencies

#### Core Framework
| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | latest | Async web framework |
| uvicorn[standard] | latest | ASGI server |
| starlette | latest | ASGI toolkit |

#### Database
| Package | Version | Purpose |
|---------|---------|---------|
| asyncpg | 0.30.0 | Async PostgreSQL driver |
| psycopg2-binary | latest | PostgreSQL adapter |
| SQLAlchemy | latest | ORM |
| redis | 5.2.1 | Caching |

#### Authentication
| Package | Version | Purpose |
|---------|---------|---------|
| python-jose[cryptography] | latest | JWT handling |
| PyJWT | latest | JWT tokens |
| passlib[bcrypt] | 1.7.4 | Password hashing |
| Authlib | 1.2.1 | OAuth support |
| google-auth | 2.23.4 | Google OAuth |
| google-auth-httplib2 | 0.1.1 | Google HTTP client |
| google-auth-oauthlib | 1.1.0 | Google OAuth flow |

#### AI/ML
| Package | Version | Purpose |
|---------|---------|---------|
| openai | latest | GPT API |
| langchain | 0.3.18 | LLM orchestration |
| langchain-community | 0.3.17 | Community integrations |
| langchain-core | 0.3.34 | Core abstractions |
| langchain-openai | latest | OpenAI integration |
| langchain-text-splitters | 0.3.6 | Text chunking |
| sentence-transformers | 3.4.1 | Embedding models |
| tiktoken | 0.8.0 | Token counting |
| cerebras-cloud-sdk | latest | Cerebras inference |

#### Data Processing
| Package | Version | Purpose |
|---------|---------|---------|
| pandas | latest | Data manipulation |
| numpy | 2.2.5 | Numerical operations |
| finqual | latest | Financial data |
| datasets | latest | Dataset handling |

#### Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| pydantic | 2.10.6 | Data validation |
| pydantic-settings | 2.7.1 | Settings management |
| httpx | latest | Async HTTP client |
| websockets | 15.0.1 | WebSocket support |
| aiofiles | 24.1.0 | Async file I/O |
| tavily | 1.1.0 | Web search |
| tenacity | 9.0.0 | Retry logic |
| logfire[fastapi,asyncpg] | latest | Observability |
| python-multipart | latest | Form data |
| python-dotenv | latest | Environment variables |
| email-validator | latest | Email validation |

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
