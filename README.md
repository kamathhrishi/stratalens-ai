# StrataLens AI

Stratalens AI is an equity research platform. Ask questions and get answers from 10-K filings, earnings calls, and news.

**Live Platform:** [www.stratalens.ai](https://www.stratalens.ai)

**10K filings agent blogpost:** [Blogpost](https://substack.com/home/post/p-181608263)

## Agent System

Core agent system implementing **Retrieval-Augmented Generation (RAG)** with **semantic data source routing**, **research planning**, and **iterative self-improvement** for financial Q&A.

### Architecture Overview

```
                              AGENT PIPELINE
 ═══════════════════════════════════════════════════════════════════════

 ┌──────────┐    ┌───────────────────┐    ┌──────────────────────────┐
 │ Question │───►│ Question Analyzer │───►│  Semantic Data Routing   │
 └──────────┘    │   (Cerebras LLM)  │    │                          │
                 │                   │    │  • Earnings Transcripts  │
                 │ Extracts:         │    │  • SEC 10-K Filings      │
                 │ • Tickers         │    │  • Real-Time News        │
                 │ • Time periods    │    │  • Hybrid (multi-source) │
                 │ • Intent          │    └────────────┬─────────────┘
                 └───────────────────┘                 │
                                                       ▼
                 ┌─────────────────────────────────────────────────────┐
                 │              RESEARCH PLANNING                       │
                 │  Agent generates reasoning: "I need to find..."     │
                 └────────────────────────┬────────────────────────────┘
                                          ▼
                 ┌─────────────────────────────────────────────────────┐
                 │                  RETRIEVAL LAYER                     │
                 │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
                 │  │  Earnings   │  │  SEC 10-K   │  │   Tavily    │  │
                 │  │ Transcripts │  │  Retrieval  │  │    News     │  │
                 │  │             │  │   Agent     │  │             │  │
                 │  │ Vector DB   │  │ (10-K only) │  │  Live API   │  │
                 │  │ + Hybrid    │  │ Planning +  │  │             │  │
                 │  │   Search    │  │  Iterative  │  │             │  │
                 │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
                 └─────────┴───────────┬────┴────────────────┴─────────┘
                                       │ ▲
                                       │ │ Re-query with
                                       │ │ follow-up questions
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
1. **Semantic Routing** - Routes to data sources based on question **intent**, not keywords
2. **Research Planning** - Agent explains reasoning before searching ("I need to find...")
3. **Multi-Source RAG** - Combines earnings transcripts, SEC 10-K filings, and news
4. **Self-Reflection** - Evaluates answer quality and iterates until confident
5. **Answer Modes** - Configurable iteration depth (2-10 iterations) and quality thresholds (70-95%)
6. **Search-Optimized Follow-ups** - Generates keyword phrases for better RAG retrieval

**Benchmark:** 91% accuracy on [FinanceBench](https://github.com/patronus-ai/financebench) (112 10-K questions), ~10s per question, evaluated using LLM-as-a-judge.

### Documentation

| Document | Description |
|----------|-------------|
| **[agent/README.md](agent/README.md)** | Complete agent architecture, pipeline stages, configuration |
| **[docs/SEC_AGENT.md](docs/SEC_AGENT.md)** | SEC 10-K agent: section routing, table selection, reranking |
| **[agent/rag/data_ingestion/README.md](agent/rag/data_ingestion/README.md)** | Data ingestion pipelines for transcripts and 10-K filings |

---

## Features

- **Earnings Transcripts** (2022-2025) - Word-for-word executive commentary from earnings calls
- **SEC 10-K Filings** (2024-25) - Official annual reports via specialized retrieval agent (10-Q/8-K coming soon)
- **Real-Time News** - Latest market developments via Tavily search
- **Financial Screener** - Natural language queries over company fundamentals [in development]

Unlike generic LLMs that rely on web content, StrataLens uses the same authoritative documents that professional analysts depend on.

## Tech Stack

- **Backend:** FastAPI, PostgreSQL (pgvector), DuckDB
- **AI/ML:** Cerebras (Qwen-3-235B), OpenAI (fallback), RAG with iterative self-improvement
- **Search:** Hybrid vector (pgvector) + TF-IDF with cross-encoder reranking
- **Frontend:** React + TypeScript, Tailwind CSS

## Project Structure

```
stratalens_ai/
├── agent/                  # AI agent & RAG system         → see agent/README.md
│   ├── __init__.py        # Public API: Agent, RAGAgent, create_agent()
│   ├── agent_config.py    # Iteration/quality threshold settings
│   ├── prompts.py         # Centralized LLM prompt templates
│   ├── llm/               # Unified LLM client (OpenAI/Cerebras)  → see agent/llm/README.md
│   ├── rag/               # RAG implementation
│   │   ├── rag_agent.py                          # Main orchestration
│   │   ├── sec_filings_service_smart_parallel.py  # SEC 10-K agent
│   │   ├── response_generator.py   # LLM response & evaluation
│   │   ├── question_analyzer.py    # Semantic routing
│   │   ├── search_engine.py        # Hybrid transcript search
│   │   ├── tavily_service.py       # Real-time news
│   │   └── data_ingestion/         # Data pipeline → see data_ingestion/README.md
│   └── screener/          # Financial screener
├── app/                   # FastAPI application
│   ├── routers/           # API endpoints
│   └── schemas/           # Pydantic models
├── frontend/              # React + TypeScript frontend
├── docs/                  # Documentation
│   └── SEC_AGENT.md       # 10-K agent deep dive
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

Before running the application, configure the following in `.env`:

- `BASE_URL` - Set to your server URL (e.g., `http://localhost:8000` for local, your production URL for deployed)
- `RAG_DEBUG_MODE` - Set to `false` for production, `true` for development debugging
- `AUTH_DISABLED` - Set to `true` to bypass Clerk auth (dev only), `false` for production
- `CLERK_SECRET_KEY` / `CLERK_PUBLISHABLE_KEY` - Required for production authentication (get from Clerk Dashboard)

Frontend env vars (read from root `.env` via `envDir: '../'` in `vite.config.ts`):
- `VITE_CLERK_PUBLISHABLE_KEY` - Same value as `CLERK_PUBLISHABLE_KEY` (Vite requires `VITE_` prefix)
- `VITE_API_BASE_URL` - Leave empty for same-origin requests (default); set to an explicit URL only if backend is on a separate domain

```bash
# Ingest data (optional - see agent/rag/data_ingestion/README.md)
python agent/rag/data_ingestion/download_transcripts.py
python agent/rag/data_ingestion/ingest_10k_to_database.py --ticker AAPL

# Run server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access the application at `http://localhost:8000`

## Requirements

### API Keys

| Service | Environment Variable | Required |
|---------|---------------------|----------|
| OpenAI | `OPENAI_API_KEY` | Yes |
| Cerebras | `CEREBRAS_API_KEY` | Yes |
| API Ninjas | `API_NINJAS_KEY` | Yes |
| Clerk | `CLERK_SECRET_KEY`, `CLERK_PUBLISHABLE_KEY` | Yes (production) |
| Tavily | `TAVILY_API_KEY` | Optional |
| Logfire | `LOGFIRE_TOKEN` | Optional |

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

Data is split between PostgreSQL (embeddings, metadata) and Railway S3 (full filing documents, transcript text). See `agent/rag/data_ingestion/README.md` for detailed ingestion instructions.

## AI Agent Documentation

| Document | Description |
|----------|-------------|
| **[agent/README.md](agent/README.md)** | Complete agent architecture, pipeline stages, semantic routing, iterative self-improvement |
| **[docs/SEC_AGENT.md](docs/SEC_AGENT.md)** | SEC 10-K agent: planning-driven retrieval, 91% accuracy on FinanceBench |
| **[agent/rag/data_ingestion/README.md](agent/rag/data_ingestion/README.md)** | Data ingestion pipelines for transcripts and SEC filings |

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





