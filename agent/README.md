# Agent System

Core agent system implementing **Retrieval-Augmented Generation (RAG)** with **intelligent tool routing** and **self-reflection** for financial Q&A. This powers the chat and analysis features on stratalens.ai.

## Architecture Overview

The agent follows a **broad-to-deep** execution pattern:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIGH-LEVEL FLOW                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Question ──► Analyze & Route ──► Retrieve from Tools ──► Generate     │
│                     │                      │                    │        │
│                     │                      │                    ▼        │
│                     │                      │              ┌──────────┐   │
│                     │                      │              │ Evaluate │   │
│                     │                      │              │ Quality  │   │
│                     │                      │              └────┬─────┘   │
│                     │                      │                   │         │
│                     │                      │         confident?│         │
│                     │                      │              NO ──┴── YES   │
│                     │                      │              │         │    │
│                     │                      ◄─────────────┘         ▼    │
│                     │                   (iterate)            Final Answer│
│                     │                                                    │
│                     ▼                                                    │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │              TOOL ROUTING (Question Analyzer)            │           │
│   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐    │           │
│   │  │  Earnings   │ │  SEC 10-K   │ │   Real-Time     │    │           │
│   │  │ Transcripts │ │   Filings   │ │     News        │    │           │
│   │  │  (default)  │ │             │ │    (Tavily)     │    │           │
│   │  └─────────────┘ └─────────────┘ └─────────────────┘    │           │
│   └─────────────────────────────────────────────────────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Concepts:**
1. **Question Analysis** - LLM determines which data sources to query
2. **Tool Routing** - Routes to earnings transcripts, SEC filings, or news
3. **Self-Reflection** - Evaluates answer quality and iterates if needed (Agent Mode)

---

## Self-Reflection Loop (Agent Mode)

When running in Agent Mode (`max_iterations > 1`), the system performs iterative self-improvement. This is the core intelligence that separates a simple RAG from an agentic system.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ITERATION LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ Generate Answer  │ ◄─────────────────────────────────┐       │
│  └────────┬─────────┘                                   │       │
│           │                                              │       │
│           ▼                                              │       │
│  ┌──────────────────┐                                   │       │
│  │ Evaluate Quality │                                   │       │
│  │ • completeness   │                                   │       │
│  │ • accuracy       │                                   │       │
│  │ • clarity        │                                   │       │
│  │ • specificity    │                                   │       │
│  └────────┬─────────┘                                   │       │
│           │                                              │       │
│           ▼                                              │       │
│  ┌──────────────────┐      YES    ┌─────────────────┐   │       │
│  │ Should Iterate?  │ ──────────► │ Generate        │   │       │
│  │ (confidence<0.9) │             │ Follow-up       │ ──┘       │
│  └────────┬─────────┘             │ Questions       │           │
│           │ NO                    └─────────────────┘           │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │   Final Answer   │                                           │
│  └──────────────────┘                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Evaluation Criteria:**
- `completeness_score` (0-10): Does the answer fully address the question?
- `accuracy_score` (0-10): Is the information factually correct based on context?
- `clarity_score` (0-10): Is the answer well-structured and easy to understand?
- `specificity_score` (0-10): Does it include specific numbers, dates, quotes?
- `overall_confidence` (0-1): Weighted combination used for iteration decisions

**During iteration, the agent can autonomously decide to:**
- Search for more transcripts via `needs_transcript_search`
- Search for news via `needs_news_search` (triggers Tavily)

**Stopping Conditions:**
1. Confidence score ≥ 90% threshold
2. Agent determines answer is sufficient (`should_iterate=false`)
3. Max iterations reached
4. No follow-up questions generated

---

## Operating Modes

| Mode | Config | Latency | Use Case |
|------|--------|---------|----------|
| **Chat Mode** | `max_iterations=1` | ~3-5s | Production on stratalens.ai |
| **Agent Mode** | `max_iterations=3-4` | ~10-20s | Local testing, complex queries |

---

## How the Agent Chooses Tools

The agent doesn't blindly search all sources. It uses **LLM-based routing** in the Question Analyzer to determine which data sources to use based on the question's content.

### Data Source Routing (Question Analyzer)

When a question comes in, `question_analyzer.py` uses Cerebras LLM to analyze it and returns a `data_source` field:

| `data_source` Value | Description | Tools Used |
|---------------------|-------------|------------|
| `earnings_transcripts` | Default - quarterly earnings questions | Earnings transcript vector search |
| `10k` | Annual report questions (financials, compensation, risks) | SEC 10-K filing search |
| `latest_news` | Current events, breaking news | Tavily real-time news API |
| `hybrid` | Questions needing multiple sources | Combination of above |

**Routing Rules (from question_analyzer.py):**

```
10K is chosen when question contains:
├── "10k", "10-k", "annual report", "SEC filing"
├── "balance sheet", "income statement", "cash flow statement"
├── "executive compensation", "CEO salary", "CEO pay"
├── "risk factors", "legal proceedings", "MD&A"
└── "assets", "liabilities", "stockholders equity"

LATEST_NEWS is chosen when question contains:
├── "latest news", "recent news", "current news", "breaking news"
├── "what's happening", "latest updates", "recent developments"
└── Questions about very recent events (within days/weeks)

EARNINGS_TRANSCRIPTS is the default for:
├── Quarterly performance questions
├── Management commentary and guidance
├── Analyst Q&A discussions
└── Revenue, margins, growth discussions
```

### Tool Execution Flow (rag_agent.py)

After routing, `rag_agent.py` orchestrates tool execution in this order:

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUESTION ANALYSIS                              │
│  question_analyzer.py determines:                                 │
│  • data_source: "10k" | "latest_news" | "earnings_transcripts"   │
│  • needs_10k: boolean                                             │
│  • needs_latest_news: boolean                                     │
│  • extracted_tickers: ["AAPL", "MSFT"]                           │
│  • quarter_context: "latest" | "multiple" | "specific"           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2.5: NEWS SEARCH                         │
│  IF needs_latest_news == true:                                    │
│    → tavily_service.search_news(query)                           │
│    → Returns: articles with titles, URLs, content, dates         │
│    → Formats as context with [N1], [N2] citation markers         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2.6: 10-K SEARCH                         │
│  IF data_source in ["10k", "hybrid"] OR needs_10k == true:       │
│    → sec_filings_service.search_10k_filings_advanced_async()     │
│    → Uses LLM section routing (Cerebras)                         │
│    → Uses LLM table selection (Cerebras)                         │
│    → Hybrid search + cross-encoder reranking                     │
│    → Returns chunks with [10K1], [10K2] markers                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 3: TRANSCRIPT SEARCH                     │
│  IF data_source NOT in ["10k", "latest_news"]:                   │
│    → search_engine.search_similar_chunks()                       │
│    → Vector search (70%) + keyword BM25 (30%)                    │
│    → Returns chunks with citation markers                        │
│    → SKIPPED if pure 10K or news-only query                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 4: RESPONSE GENERATION                   │
│  All context combined → response_generator                       │
│  • news_context (from Tavily)                                    │
│  • ten_k_context (from SEC service)                              │
│  • transcript chunks (from search engine)                        │
│  → Single LLM call with all available context                    │
└─────────────────────────────────────────────────────────────────┘
```

## Deep Dive: Tavily (Real-Time News)

`tavily_service.py` provides real-time web search for current events that aren't in historical transcripts or filings.

### When Tavily is Used

1. **Question Analyzer Detection**: If question contains news keywords, sets `needs_latest_news=true`
2. **Agent Mode Iteration**: During self-reflection, if the agent determines current information is needed, it can trigger Tavily search via `needs_news_search` in evaluation

### How Tavily Works

```python
# tavily_service.py
class TavilyService:
    def search_news(self, query: str, max_results: int = 5, include_answer: str = "advanced"):
        """
        Searches Tavily API for latest news articles.

        Returns:
            {
                "answer": "AI-generated summary of results",
                "results": [
                    {
                        "title": "Article headline",
                        "url": "https://...",
                        "content": "Article text preview",
                        "published_date": "2024-01-15",
                        "score": 0.95
                    }
                ]
            }
        """

    def format_news_context(self, news_results):
        """Formats results with [N1], [N2] citation markers for LLM context"""

    def get_news_citations(self, news_results):
        """Extracts citation metadata for frontend display"""
```

### Example Flow

```
User: "What's the latest news on NVIDIA?"

1. Question Analyzer:
   - Detects "latest news" keyword
   - Sets data_source="latest_news", needs_latest_news=true
   - Extracts ticker: NVDA

2. rag_agent.py Stage 2.5:
   - Calls tavily_service.search_news("What's the latest news on NVIDIA? NVDA")
   - Returns 5 recent articles

3. Context Formation:
   === LATEST NEWS (from Tavily) ===
   Summary: NVIDIA announced record Q4 earnings...

   [N1] NVIDIA Stock Surges on AI Chip Demand
      Published: 2024-01-20
      Source: https://reuters.com/...
      NVIDIA's stock rose 5% following...

   [N2] Jensen Huang Keynote at CES 2024
      ...
   === END NEWS ===

4. Response Generator:
   - Receives news_context parameter
   - Generates answer citing [N1], [N2]
```

## Deep Dive: SEC 10-K Filings

`sec_filings_service.py` provides sophisticated access to annual SEC 10-K filings with LLM-based intelligent routing.

### When 10-K is Used

1. **Explicit Request**: Question mentions "10k", "10-K", "annual report", "SEC filing"
2. **Content Detection**: Questions about balance sheets, income statements, executive compensation, risk factors
3. **Automatic Detection**: Executive compensation questions ALWAYS use 10-K (this data isn't in earnings transcripts)

### 10-K Search Pipeline (4 Stages)

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 0: LLM Section Routing (Cerebras)                         │
│  ────────────────────────────────────────                        │
│  Question: "What are Apple's risk factors?"                      │
│                                                                   │
│  LLM analyzes and routes to relevant SEC sections:               │
│  → ["item_1a"] (Risk Factors section)                            │
│                                                                   │
│  Quantitative questions → item_7 (MD&A), item_8 (Financials)     │
│  Qualitative questions → item_1 (Business), item_1a (Risks)      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Hybrid Search (TF-IDF + Semantic)                      │
│  ────────────────────────────────────────────                    │
│  • Semantic search: 70% weight (vector similarity)               │
│  • Keyword search: 30% weight (TF-IDF)                           │
│  • Filter by routed sections from Phase 0                        │
│  • Retrieve ~100 candidate chunks                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Cross-Encoder Reranking                                │
│  ────────────────────────────────────                            │
│  • Uses cross-encoder/ms-marco-MiniLM-L-6-v2                     │
│  • Scores each (query, chunk) pair for relevance                 │
│  • Reorders results by cross-encoder score                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: LLM-Based Table Selection (Cerebras)                   │
│  ────────────────────────────────────────────                    │
│  • Fetches ALL tables for the ticker from ten_k_tables           │
│  • Prioritizes core financial statements:                        │
│    🌟 Income Statement (revenue, profit, expenses)               │
│    🌟 Balance Sheet (assets, liabilities, equity)                │
│    🌟 Cash Flow Statement (cash flows, capex)                    │
│  • LLM selects 2-5 most relevant tables                          │
│  • Selected tables placed BEFORE text chunks in context          │
└─────────────────────────────────────────────────────────────────┘
```

### Table Selection Prompt (Cerebras LLM)

The agent uses a detailed prompt for intelligent table selection:

```python
# From sec_filings_service.py
prompt = """
QUESTION: {question}

AVAILABLE TABLES:
1. [🌟 CORE FINANCIAL STATEMENT] Income Statement (item_8) - income_statement
2. [🌟 CORE FINANCIAL STATEMENT] Balance Sheet (item_8) - balance_sheet
3. Revenue by Segment (item_7)
...

STEP 1: DEEP QUESTION ANALYSIS
- What EXACTLY is being asked?
- Identify key financial metrics
- Determine if numbers, ratios, or trends needed

STEP 2: SYSTEMATIC TABLE EVALUATION
- For EACH table: Does it DIRECTLY answer the question?
- Create relevance score (1-10)

STEP 3: MAKE SELECTION
- PRIORITIZE core financial statements marked with 🌟
- Maximum 5 tables, prefer fewer highly relevant ones
- Quality over quantity

Return JSON:
{"selected_table_indices": [1, 2, 5], "reasoning": "..."}
"""
```

### Example 10-K Flow

```
User: "What was Tim Cook's compensation in 2023?"

1. Question Analyzer:
   - Detects "compensation" keyword
   - Sets data_source="10k", needs_10k=true
   - Note: Executive compensation is ONLY in 10-K filings

2. sec_filings_service Phase 0 (Section Routing):
   - LLM routes to: ["item_11"] (Executive Compensation)

3. Stage 1 (Hybrid Search):
   - Searches item_11 chunks for "compensation" "Tim Cook"
   - Returns 100 candidate chunks

4. Stage 2 (Cross-Encoder Reranking):
   - Reranks by relevance to exact question
   - Top chunks about CEO compensation float up

5. Stage 3 (Table Selection):
   - LLM sees: "Executive Compensation Table", "Stock Awards", etc.
   - Selects: Summary Compensation Table, Stock Awards Table

6. Context Formation:
   === 10-K SEC FILINGS DATA ===
   [10K1] AAPL - FY2023 - Executive Compensation
   Type: Financial Table
   Content: [Summary Compensation Table with Tim Cook's salary...]

   [10K2] AAPL - FY2023 - Executive Compensation
   Content: The CEO's total compensation for fiscal 2023...
   === END 10-K ===

7. Response Generation:
   - Uses ten_k_context parameter
   - Generates answer with specific salary figures
```

## Earnings Transcript Search

For quarterly earnings questions, the agent uses hybrid search over transcript chunks.

### Search Strategy

```python
# search_engine.py
def search_similar_chunks(query, top_k, quarter):
    """
    Hybrid search combining:
    - Vector search: 70% weight (semantic similarity via pgvector)
    - Keyword search: 30% weight (BM25 via PostgreSQL full-text)
    """
```

### Chunk Storage

```
PostgreSQL Table: transcript_chunks
├── chunk_text: TEXT (1000 chars max, 200 overlap)
├── embedding: VECTOR (all-MiniLM-L6-v2, 384 dimensions)
├── ticker: VARCHAR (e.g., "AAPL")
├── year: INTEGER (e.g., 2024)
├── quarter: INTEGER (1-4)
└── metadata: JSONB
```

## Key Components

### Core Files

| File | Description |
|------|-------------|
| `agent.py` | Main entry point - unified Agent API for financial Q&A |
| `rag/rag_agent.py` | Orchestration engine with tool routing and self-reflection |
| `rag/question_analyzer.py` | LLM-based query analysis and data source routing (Cerebras) |

### Data Sources (Tools)

| File | Tool | Description |
|------|------|-------------|
| `rag/search_engine.py` | Transcript Search | Hybrid vector + keyword search over earnings transcripts |
| `rag/sec_filings_service.py` | 10-K Search | SEC annual filings with LLM section routing and table selection |
| `rag/tavily_service.py` | News Search | Real-time news via Tavily API |

### Supporting Components

| File | Description |
|------|-------------|
| `rag/response_generator.py` | LLM response generation with streaming and quality evaluation |
| `rag/database_manager.py` | PostgreSQL/pgvector operations and connection pooling |
| `rag/conversation_memory.py` | Multi-turn conversation state for context-aware questions |
| `prompts.py` | Centralized LLM prompt templates |
| `rag/config.py` | RAG configuration (chunk sizes, search weights, model names) |

## Data Storage

### Database Schema

```
PostgreSQL + pgvector
├── transcript_chunks       # Earnings call transcripts
│   ├── chunk_text          # 1000 chars, 200 overlap
│   ├── embedding           # all-MiniLM-L6-v2 (384 dim)
│   ├── ticker, year, quarter
│   └── metadata (JSONB)
│
├── ten_k_chunks            # 10-K filing text
│   ├── chunk_text, embedding
│   ├── sec_section         # item_1, item_7, item_8, etc.
│   ├── sec_section_title   # Human-readable section name
│   └── is_financial_statement
│
└── ten_k_tables            # 10-K extracted tables (JSONB)
    ├── content             # Table data
    ├── statement_type      # income_statement, balance_sheet, cash_flow
    └── is_financial_statement
```

## Key Features

**Intelligent Tool Routing:**
- LLM-based data source selection (earnings, 10-K, news)
- Automatic detection of question intent
- Skip unnecessary searches based on question type

**Multi-Source RAG:**
- Earnings transcripts: Hybrid vector + keyword search
- SEC 10-K filings: LLM section routing + table selection + cross-encoder reranking
- Real-time news: Tavily API integration

**Core Capabilities:**
- Multi-ticker comparative analysis (up to 8 tickers)
- Quarter-aware filtering (e.g., "Q4 2024", "latest quarter", "last 3 quarters")
- Citation tracking with source attribution ([N1] for news, [10K1] for filings)
- Streaming response generation
- Multi-turn conversation memory

## Limitations

- Requires `$TICKER` format for company identification
- Quarter availability varies by company
- Companies describe fiscal years differently (cross-company comparison challenges)
- No real-time stock price data
- No strict evals for earnings transcripts at the moment

## Usage

```python
from agent import create_agent

agent = create_agent()

# Earnings transcript question (automatic routing)
result = await agent.execute_rag_flow_async(
    question="What did $AAPL say about iPhone sales in Q4 2024?",
    max_iterations=1
)

# 10-K question (automatically routes to SEC filings)
result = await agent.execute_rag_flow_async(
    question="What was Tim Cook's compensation in 2023?",
    max_iterations=1
)

# News question (automatically routes to Tavily)
result = await agent.execute_rag_flow_async(
    question="What's the latest news on $NVDA?",
    max_iterations=1
)

# Streaming
async for event in agent.execute_rag_flow(
    question="Compare $MSFT and $GOOGL cloud revenue",
    max_iterations=1,
    stream=True
):
    if event['type'] == 'streaming_token':
        print(event['data'], end='', flush=True)
```

## Configuration

**Environment Variables**:
```bash
OPENAI_API_KEY=...           # Response generation
CEREBRAS_API_KEY=...         # Question analysis, section routing, table selection
TAVILY_API_KEY=...           # Real-time news search
DATABASE_URL=postgresql://...
```

**Agent Config** (`agent_config.py`):
- `max_iterations`: Refinement iterations (default: 4)
- `min_confidence_threshold`: Quality threshold for early stopping (default: 0.90)

**RAG Config** (`rag/config.py`):
- `chunks_per_quarter`: Max chunks per quarter (default: 15)
- `keyword_weight` / `vector_weight`: Hybrid search (0.3 / 0.7)
- `cerebras_model`: Question analysis model (default: qwen-3-235b)
- `openai_model`: Generation model (default: gpt-4.1-mini)

## Development Status

| Component | Status |
|-----------|--------|
| Earnings Transcript Search | ✅ Production |
| SEC 10-K Filing Search | ✅ Production |
| Tavily News Search | ✅ Production |
| LLM Data Source Routing | ✅ Production |
| Streaming | ✅ Production |
| Multi-ticker/quarter | ✅ Production |
| Conversation memory | ✅ Production |
| Agent mode (self-reflection) | 🧪 Experimental |
| Screener agent | 🧪 Experimental |

## Data Ingestion

See `agent/rag/data_ingestion/README.md` for transcript and 10-K ingestion pipelines.

## Related

- API endpoints: See main `README.md` in project root
- Prompt templates: `prompts.py`
- FastAPI integration: `fastapi_server.py`
