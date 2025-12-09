# Agent System

Core agent system implementing **Retrieval-Augmented Generation (RAG)** with **self-reflection** for financial Q&A over earnings transcripts and SEC 10-K filings. This is what runs the chat and analysis features on stratalens.ai.

## File Structure

```
agent/
├── agent.py                    # Main entry point - Agent class
├── agent_config.py             # Agent configuration (iterations, thresholds)
├── prompts.py                  # Centralized LLM prompt templates
├── screener_agent.py           # Financial screener (text-to-SQL)
│
├── rag/                        # RAG implementation
│   ├── rag_agent.py            # Orchestration engine & iteration loop
│   ├── question_analyzer.py    # Query parsing (tickers, quarters, intent)
│   ├── search_engine.py        # Hybrid search (vector + keyword)
│   ├── response_generator.py   # LLM response & evaluation
│   ├── database_manager.py     # PostgreSQL/pgvector operations
│   ├── conversation_memory.py  # Multi-turn conversation state
│   ├── transcript_service.py   # Transcript metadata
│   ├── sec_filings_service.py  # SEC 10-K retrieval
│   ├── tavily_service.py       # Web search augmentation
│   ├── config.py               # RAG configuration
│   ├── rag_utils.py            # Utility functions
│   └── data_ingestion/         # Data pipeline → see data_ingestion/README.md
│
└── screener/                   # Financial screener
    └── metadata.py             # Screener metadata
```

## Overview

Agentic RAG system that combines retrieval, generation, and autonomous quality evaluation:

1. **Query Analysis** - LLM-based extraction of tickers, quarters, and intent with conversation context
2. **Hybrid Retrieval** - Vector search (70%) + keyword search (30%) with cross-encoder reranking
3. **Response Generation** - Multi-model LLM generation with citations
4. **Self-Reflection** - Autonomous evaluation and iterative refinement (agent mode only)

## Architecture

### System Design

The agent system follows a modular architecture where the `Agent` class provides a clean API interface to the underlying `RAGAgent` orchestration engine:

```
┌──────────────────────────────────────────────────────────────┐
│                        Agent (agent.py)                       │
│                   Main Entry Point & API Layer                │
│                     (delegates to RAGAgent)                    │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                    RAGAgent (rag/rag_agent.py)                │
│              Orchestration & Self-Reflection Engine           │
│                                                                │
│  Initializes and orchestrates all components:                 │
└─┬──────────┬───────────┬──────────────┬───────────────────┬──┘
  │          │           │              │                   │
  │          │           │              │                   │
  ▼          ▼           ▼              ▼                   ▼
┌────────┐ ┌────────────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────┐
│Database│ │ Question       │ │  Search  │ │  Response    │ │  Analytics  │
│Manager │ │ Analyzer       │ │  Engine  │ │  Generator   │ │   Logger    │
└────┬───┘ │                │ └─────┬────┘ └──────────────┘ └─────────────┘
     │     │  ┌──────────┐  │       │
     │     │  │Conversa- │  │       │
     │     │  │tion      │  │       │
     │     │  │Memory    │  │       │
     │     │  └──────────┘  │       │
     │     └────────────────┘       │
     │                              │
     └──────────uses────────────────┘

Additional modules:
• Config (rag/config.py) - shared configuration
• prompts.py - centralized LLM prompts
• rag_utils.py - utility functions
• agent_config.py - agent-specific configuration
```

### RAG Pipeline Flow

```
User Question
     │
     ▼
┌─────────────────────────────────────────────┐
│  1. Question Analysis                        │
│  • Extract tickers, quarters, intent         │
│  • Conversation context integration          │
│  • Validate query appropriateness            │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  2. Hybrid Retrieval (RAG Core)              │
│  • Vector search (semantic similarity)       │
│  • Keyword search (BM25)                     │
│  • Cross-encoder reranking                   │
│  • Quarter-aware filtering                   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  3. Response Generation                      │
│  • Context-aware LLM prompting               │
│  • Multi-quarter parallel processing         │
│  • Citation and source attribution           │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  4. Self-Reflection (Agent Mode Only)        │
│  • Quality scoring (completeness, accuracy)  │
│  • Gap identification                        │
│  • Follow-up question generation             │
│  • Iterative refinement until threshold met  │
└──────────────┬──────────────────────────────┘
               │
               ▼
          Final Answer
```

### Key Components

#### Core Files

- **`agent.py`** - Main entry point providing the unified Agent API for financial Q&A. Handles both streaming and non-streaming execution flows.

- **`rag/rag_agent.py`** - RAG orchestration engine with self-reflection capabilities. Coordinates the complete pipeline from question analysis through iterative refinement.

#### Retrieval Layer (RAG Foundation)

- **`rag/question_analyzer.py`** - Question analysis using Groq (`openai/gpt-oss-20b`). Extracts tickers, quarters, intent. Uses conversation memory to provide context for follow-up questions.

- **`rag/search_engine.py`** - Hybrid search: vector (all-MiniLM-L6-v2 embeddings) + keyword (BM25). Cross-encoder reranking for top results.

- **`rag/database_manager.py`** - PostgreSQL with pgvector extension. Connection pooling and query optimization.

#### Generation Layer

- **`rag/response_generator.py`** - Response generation using OpenAI (`gpt-4.1-mini-2025-04-14`). Supports single/multi-ticker, multi-quarter, streaming. Includes quality evaluation logic for agent mode.

#### Supporting Components

- **`rag/conversation_memory.py`** - Multi-turn conversation state. Used by question analyzer (context for follow-ups) and response generator (evaluation with history).

- **`rag/transcript_service.py`** - Transcript metadata and quarter availability.

- **`agent_config.py`** - Iteration limits, confidence thresholds for agent mode.

- **`prompts.py`** - Centralized LLM prompt templates.

- **`rag/config.py`** - RAG configuration: chunk sizes, search weights, model names.

## Operating Modes

### Chat Mode (Production)
- **Status**: Production on stratalens.ai
- **Config**: `max_iterations=1` (single-pass RAG)
- **Latency**: ~3-5s
- **Behavior**: Question → Retrieve → Generate → Answer

### Agent Mode (Experimental)
- **Status**: Local testing only
- **Config**: `max_iterations=3-4` (with self-reflection)
- **Latency**: ~10-20s (3-4x slower)
- **Behavior**: Question → Retrieve → Generate → Evaluate → (if needed) Refine Query → Retrieve → Generate → Answer

#### Self-Reflection Loop

The agent mode implements an iterative self-improvement loop:

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

**Stopping Conditions:**
1. Confidence score ≥ 90% threshold
2. Agent determines answer is sufficient (`should_iterate=false`)
3. Max iterations reached
4. No follow-up questions generated

**Evaluation Criteria:**
- `completeness_score` (0-10): Does the answer fully address the question?
- `accuracy_score` (0-10): Is the information factually correct based on context?
- `clarity_score` (0-10): Is the answer well-structured and easy to understand?
- `specificity_score` (0-10): Does it include specific numbers, dates, quotes?
- `overall_confidence` (0-1): Weighted combination used for iteration decisions

**Follow-up Question Generation:**
When the agent decides to iterate, it generates targeted follow-up questions to:
- Fill gaps in the current answer
- Retrieve missing financial data
- Get additional context from different quarters
- Clarify ambiguous information

## Data Processing & Chunking

### Earnings Transcripts

Transcripts are processed with character-based chunking:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 1000 chars | Size of each text chunk |
| `chunk_overlap` | 200 chars | Overlap between consecutive chunks |
| `embedding_model` | all-MiniLM-L6-v2 | Sentence transformer for embeddings |

**Storage:** PostgreSQL with pgvector extension
- Table: `transcript_chunks`
- Columns: `chunk_text`, `embedding` (vector), `ticker`, `year`, `quarter`, `metadata`

### SEC 10-K Filings

10-K filings are processed with hierarchical chunking that preserves document structure:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunk_size` | 1000 chars | Size of each text chunk |
| `chunk_overlap` | 200 chars | Overlap between chunks |
| `chunk_type` | text/table | Type of content |

**Storage:**
- Table: `ten_k_chunks` - Text chunks with embeddings
- Table: `ten_k_tables` - Extracted tables with structured data (JSONB)

**Chunk Metadata:**
- `sec_section` - SEC section identifier (e.g., "item1", "item7", "item8")
- `sec_section_title` - Human-readable section title
- `path_string` - Hierarchical path in document
- `chunk_type` - Content type (text, table, heading)
- `is_financial_statement` - Boolean flag for core financial tables
- `statement_type` - Type: `income_statement`, `balance_sheet`, `cash_flow`

#### LLM-Based Table Selection

For 10-K queries, tables are selected using an LLM (Cerebras) rather than just vector similarity:

```
┌─────────────────────────────────────────────────────────────────┐
│                   10-K Table Selection Flow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Fetch all tables for ticker from ten_k_tables               │
│                         │                                        │
│                         ▼                                        │
│  2. Prioritize core financial statements:                       │
│     • Income Statement (revenue, profit, expenses)              │
│     • Balance Sheet (assets, liabilities, equity)               │
│     • Cash Flow Statement (cash flows, capex)                   │
│                         │                                        │
│                         ▼                                        │
│  3. LLM analyzes question and selects relevant tables           │
│     • Deep question analysis (metrics, timeframes)              │
│     • Systematic table evaluation (relevance scoring)           │
│     • Quality over quantity (2-3 highly relevant > 10 loose)    │
│                         │                                        │
│                         ▼                                        │
│  4. Selected tables + text chunks combined for response         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Table Selection Criteria:**
- Financial metrics → Core financial statements prioritized
- Segment data → Segment reporting tables
- Specific notes → Exact note tables (e.g., "NOTE 13. EARNINGS PER SHARE")
- Ratios → Multiple related tables for calculation

## Key Features

**Core RAG:**
- Hybrid retrieval: 30% keyword (BM25) + 70% vector (cosine similarity)
- Cross-encoder reranking on top results
- Quarter-aware filtering (e.g., "Q4 2024", "latest quarter")
- Multi-ticker comparative analysis (up to 8 tickers)
- Citation tracking with source attribution
- Streaming response generation

**Conversation Handling:**
- Multi-turn conversation memory
- Context injection for follow-up questions
- Conversation-aware question analysis and evaluation

**Additional:**
- Stock screener agent (text-to-SQL conversion)
- Configurable chunk sizes, search weights, generation params
- Analytics logging for query tracking

## Limitations

- Earnings transcripts only (no real-time market data)
- Limited to quarterly earnings calls
- Quarter availability varies by company
- Companies describe fiscal years differently, so there can be issues when doing cross company comparison. Resolving this. 
- Currently user has to strictly type the ticker name with a $ symbol. This cna be a issue doing cross company queries like: "Describe more about $ADBE and $FIG merger"
- Right now it searches latest quarter by default, we need to also resolve earlier quarters where data is available.
- Retry mechanism when no relevant chunks would be great 
- No strict evals for earnings transcripts at the moment

## Usage

### Chat Mode (Production)
```python
from agent import create_agent

agent = create_agent()

# Non-streaming
result = await agent.execute_rag_flow_async(
    question="What was Apple's revenue in Q4 2024?",
    max_iterations=1
)

# Streaming
async for event in agent.execute_rag_flow(
    question="Compare Microsoft and Google's cloud revenue",
    max_iterations=1,
    stream=True
):
    if event['type'] == 'streaming_token':
        print(event['data'], end='', flush=True)
```

### Agent Mode (Experimental)
```python
# Local testing only
result = await agent.execute_rag_flow_async(
    question="Analyze Apple's profitability trends Q2-Q4 2024",
    max_iterations=3,
    comprehensive=True
)

# Includes evaluation metadata
print(f"Iterations: {result['metadata']['iterations']}")
print(f"Quality scores: {result['metadata']['quality_scores']}")
```

## Configuration

**Agent Config** (`agent_config.py`):
- `max_iterations`: Refinement iterations (default: 4)
- `min_confidence_threshold`: Quality threshold for early stopping (default: 0.90)
- `evaluation_model`: OpenAI model for self-evaluation (default: gpt-4.1-mini-2025-04-14)

**RAG Config** (`rag/config.py`):
- `chunks_per_quarter`: Max chunks per quarter (default: 15)
- `chunk_size`: Tokens per chunk (default: 1000)
- `keyword_weight` / `vector_weight`: Hybrid search (0.3 / 0.7)
- `openai_model`: Generation model (default: gpt-4.1-mini-2025-04-14)
- `groq_model`: Analysis model (default: openai/gpt-oss-20b)
- `embedding_model`: Sentence transformer (default: all-MiniLM-L6-v2)

**Environment Variables**:
```bash
OPENAI_API_KEY=...
GROQ_API_KEY=...
DATABASE_URL=postgresql://...
```

## Development Status

| Component | Status |
|-----------|--------|
| Chat Mode (single-pass RAG) | ✅ Production |
| Streaming | ✅ Production |
| Multi-ticker/quarter | ✅ Production |
| Conversation memory | ✅ Production |
| Agent mode (self-reflection) | 🧪 Experimental |
| Screener agent | 🧪 Experimental |

## Data Ingestion

See `agent/rag/data_ingestion/README.md` for transcript ingestion pipeline.

## Related

- API endpoints: See main `README.md` in project root
- Prompt templates: `prompts.py`
- FastAPI integration: `fastapi_server.py`
