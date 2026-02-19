# Agent System

Core agent system implementing **Retrieval-Augmented Generation (RAG)** with **semantic data source routing**, **research planning**, and **iterative self-improvement** for financial Q&A. This powers the chat and analysis features on stratalens.ai.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Complete Pipeline](#complete-pipeline)
- [Semantic Data Source Routing](#semantic-data-source-routing)
- [Question Planning & Reasoning](#question-planning--reasoning)
- [Self-Reflection Loop](#self-reflection-loop)
- [Follow-Up Search Strategy](#follow-up-search-strategy)
- [Data Sources](#data-sources)
  - [Earnings Transcripts](#earnings-transcript-search)
  - [SEC 10-K Filings](#sec-10-k-filings-agent)
  - [Real-Time News](#tavily-real-time-news)
- [Multi-Ticker Synthesis](#multi-ticker-synthesis)
- [Streaming Events](#streaming-events)
- [Configuration](#configuration)
- [Usage](#usage)
- [Key Components](#key-components)

---

## Architecture Overview

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
                 │  │ Transcripts │  │   Filings   │  │    News     │  │
                 │  │             │  │             │  │             │  │
                 │  │ Vector DB   │  │ Section     │  │  Live API   │  │
                 │  │ + Hybrid    │  │ Routing +   │  │             │  │
                 │  │   Search    │  │ Reranking   │  │             │  │
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
3. **Multi-Source RAG** - Combines earnings transcripts, SEC filings, and news
4. **Self-Reflection** - Evaluates answer quality and iterates until confident
5. **Answer Modes** - Configurable iteration depth (2-10 iterations) and quality thresholds (70-95%)
6. **Search-Optimized Follow-ups** - Generates keyword phrases, not verbose questions, for better RAG retrieval

---

## Complete Pipeline

The agent executes a **6-stage pipeline** for each question:

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: SETUP & INITIALIZATION                                          │
│ • Initialize RAG components (search engine, response generator)          │
│ • Load configuration and available quarters                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: QUESTION ANALYSIS & PLANNING (Cerebras LLM)                     │
│ Single LLM call that performs:                                           │
│ • Extract company tickers ($AAPL, $MSFT)                                 │
│ • Detect time periods (Q4 2024, last 3 quarters, latest)                 │
│ • Semantic routing → Choose data source based on INTENT                  │
│ • Detect answer_mode (direct/standard/detailed/deep_search)              │
│ • Generate semantically-grounded search query                            │
│ • Validate question (reject off-topic/invalid)                           │
│                                                                           │
│ Quarter Resolution (company-specific database queries):                  │
│ • "latest" → get_last_n_quarters_for_company(ticker, 1)                  │
│ • "last 3 quarters" → get_last_n_quarters_for_company(ticker, 3)         │
│ • Uses DB query: SELECT DISTINCT year, quarter FROM transcript_chunks    │
│   WHERE ticker = %s ORDER BY year DESC, quarter DESC                     │
│ • Each company gets its own most recent quarters (not global)            │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2.1: RESEARCH REASONING (Cerebras LLM)                             │
│ • Separate LLM call generates transparent research reasoning             │
│ • Example: "The user is asking about Azure revenue, so I need to find    │
│   quarterly growth rates, management commentary on cloud competition..." │
│ • Explains what metrics/data points to search for                        │
│ • Used later for evaluation (did we find what we planned to find?)       │
│ • Streamed to frontend as 'reasoning' event                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2.5: NEWS SEARCH (if needs_latest_news=true)                       │
│ • Query Tavily API for real-time news                                    │
│ • Format with [N1], [N2] citation markers                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2.6: SEC 10-K RETRIEVAL AGENT (if data_source="10k")               │
│ Invokes specialized retrieval agent for SEC 10-K annual filings:         │
│ • Planning-driven sub-question generation                                │
│ • LLM-based section routing (Item 1, Item 7, Item 8, etc.)               │
│ • Hybrid search (TF-IDF + semantic) with cross-encoder reranking         │
│ • LLM-based table selection from financial statements                    │
│ • Iterative retrieval (up to 5 iterations, self-evaluates)               │
│ • Format with [10K1], [10K2] citation markers                            │
│ Note: 10-K only for now (10-Q and 8-K support coming)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: TRANSCRIPT SEARCH (Vector + Keyword Hybrid)                     │
│ • Single-ticker: Direct search with quarter filtering                    │
│ • Multi-ticker: Parallel search per company                              │
│ • Hybrid scoring: 70% vector + 30% keyword                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: INITIAL ANSWER GENERATION                                       │
│ • Single ticker → generate_openai_response()                             │
│ • Multiple tickers → generate_multi_ticker_response() with synthesis     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: ITERATIVE IMPROVEMENT (varies by answer mode)                   │
│ For each iteration:                                                      │
│   1. Evaluate answer quality (confidence, completeness, specificity)     │
│   2. Check if reasoning goals are met                                    │
│   3. Generate search-optimized keyword phrases (not verbose questions)   │
│   4. Search ALL target quarters in parallel with each keyword phrase     │
│   5. Agent may request news/transcript search                            │
│   6. Regenerate answer with expanded context                             │
│ Stop when: confidence ≥ threshold, max iterations, or agent satisfied    │
│                                                                           │
│ Answer Modes:                                                            │
│ • direct: 2 iterations, 70% threshold - quick factual answers            │
│ • standard: 3 iterations, 80% threshold - balanced analysis              │
│ • detailed: 4 iterations, 90% threshold - comprehensive research         │
│ • deep_search: 10 iterations, 95% threshold - exhaustive search          │
│   (only triggers on explicit request: "search thoroughly", "dig deep")   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: FINAL RESPONSE ASSEMBLY                                         │
│ • Stream final answer with citations                                     │
│ • Include all source attributions (transcripts, 10-K, news)              │
│ • Return metadata (confidence, chunks used, timing)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Semantic Data Source Routing

The agent routes questions based on **intent**, not just keywords. This is a key differentiator from simple keyword matching.

The main agent orchestrates access to **three specialized data source tools**:
1. **Earnings Transcript Search** (`search_engine.py`) - Hybrid vector + keyword search over earnings calls
2. **SEC 10-K Filings Agent** (`sec_filings_service_smart_parallel.py`) - Specialized retrieval agent for SEC 10-K annual filings
3. **Tavily News Search** (`tavily_service.py`) - Real-time web search for breaking news

The Question Analyzer automatically selects which tool(s) to use based on question intent. The SEC agent is a full retrieval agent (not just search) with its own iterative improvement loop optimized for structured SEC filings.

### How It Works

The Question Analyzer uses Cerebras LLM to understand what type of information would **best answer** the question:

```
QUESTION INTENT → DATA SOURCE DECISION

┌─────────────────────────────────────────────────────────────────────────┐
│ SEC 10-K FILINGS (data_source="10k")  [Specialized Retrieval Agent]     │
│ Currently: 10-K only (annual reports) | Coming: 10-Q, 8-K                │
│                                                                           │
│ Best for:                                                                │
│ • Annual/full-year financial data, audited figures                       │
│ • Balance sheets, income statements, cash flow statements                │
│ • Executive compensation, CEO pay, stock awards (ONLY in 10-K!)          │
│ • Risk factors, legal proceedings, regulatory matters                    │
│ • Detailed business descriptions, segment breakdowns                     │
│ • Multi-year historical comparisons                                      │
│ • Total assets, liabilities, debt structure                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ EARNINGS TRANSCRIPTS (data_source="earnings_transcripts")                │
│ Best for:                                                                │
│ • Quarterly performance discussions, recent quarter results              │
│ • Management commentary, executive statements, tone/sentiment            │
│ • Forward guidance, outlook, projections                                 │
│ • Analyst Q&A, investor concerns, management responses                   │
│ • Product launches, strategic initiatives                                │
│ • Quarter-over-quarter comparisons                                       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ LATEST NEWS (data_source="latest_news")                                  │
│ Best for:                                                                │
│ • Very recent events (last few days/weeks)                               │
│ • Breaking developments, announcements                                   │
│ • Market reactions, stock movements                                      │
│ • Recent partnerships, acquisitions, leadership changes                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ HYBRID (data_source="hybrid")                                            │
│ Best for:                                                                │
│ • Questions explicitly requesting multiple perspectives                  │
│ • Comparing official filings with recent developments                    │
│ • Comprehensive analysis needing historical + current data               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Routing Decision Process

The LLM considers:
1. **Intent**: What is the user trying to learn?
2. **Time Period**: Annual=10K, Quarterly=Transcripts, Recent=News
3. **Formality**: Official/Audited=10K, Commentary=Transcripts, Current=News
4. **Completeness**: Would combining sources provide a better answer?

### Examples

| Question | Routed To | Reasoning |
|----------|-----------|-----------|
| "What was Apple's Q4 2024 revenue?" | Transcripts | Quarterly data, recent results |
| "What is Tim Cook's compensation?" | 10-K | Executive compensation only in SEC filings |
| "Show me Microsoft's balance sheet" | 10-K | Financial statements from annual reports |
| "What did management say about AI?" | Transcripts | Management commentary from earnings calls |
| "What's the latest news on NVIDIA?" | News | Recent developments |
| "Compare 10-K risks with recent news" | Hybrid | Needs multiple sources |

---

## Question Planning & Reasoning

**New Feature**: Before searching, the agent generates a reasoning statement explaining its research approach.

### Purpose

- Makes the agent's thinking transparent
- Guides evaluation (did we find what we planned to find?)
- Improves answer quality through structured research

### Example

```
User: "What is Microsoft's cloud strategy and how is Azure performing?"

Agent Reasoning:
"The user is asking about Microsoft's cloud business strategy and Azure
performance. I need to find:
- Azure revenue figures and growth rates (quarterly)
- Management commentary on competitive positioning vs AWS/Google Cloud
- Margin trends and profitability metrics
- Forward guidance for cloud segment
Key metrics: quarterly revenue, YoY growth %, operating margins.
I'll focus on the most recent quarters available and look for strategic
commentary from executives."
```

### Implementation

```python
# From prompts.py
QUESTION_PLANNING_SYSTEM_PROMPT = """You are a financial research analyst
who thinks through questions before searching. You explain your reasoning
process in a natural, verbose way - like thinking out loud about how to
approach a research question."""

# Generates 3-5 sentence reasoning explaining:
# - What the user is really trying to understand
# - What specific metrics/data points needed
# - What to focus the search on
# - How to approach given available data
```

---

## Self-Reflection Loop

The agent performs iterative self-improvement until the answer meets quality thresholds.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ITERATION LOOP                                │
│                                                                  │
│  ┌──────────────────┐                                           │
│  │ Generate Answer  │◄──────────────────────────────────┐       │
│  └────────┬─────────┘                                   │       │
│           │                                             │       │
│           ▼                                             │       │
│  ┌──────────────────┐                                   │       │
│  │ Evaluate Quality │                                   │       │
│  │ • completeness   │                                   │       │
│  │ • specificity    │                                   │       │
│  │ • accuracy       │                                   │       │
│  │ • vs. reasoning  │ ← Checks if reasoning goals met   │       │
│  └────────┬─────────┘                                   │       │
│           │                                             │       │
│           ▼                                             │       │
│  ┌──────────────────┐    YES    ┌─────────────────┐    │       │
│  │ Confidence < 90% │─────────► │ Search for more │────┘       │
│  │ & iterations left│           │ context (tools) │            │
│  └────────┬─────────┘           └─────────────────┘            │
│           │ NO                                                  │
│           ▼                                                     │
│     ┌───────────┐                                               │
│     │  OUTPUT   │                                               │
│     └───────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

**Evaluation Scores (0-100):**
- `completeness_score`: Does the answer fully address the question?
- `specificity_score`: Does it include specific numbers, quotes?
- `accuracy_score`: Is the information factually correct?
- `clarity_score`: Is the response well-structured?
- `overall_confidence` (0-1): Weighted combination

**During iteration, the agent can:**
- Generate search-optimized keyword phrases (NOT verbose questions)
  - Example: "capex guidance 2025 AI allocation" instead of "What guidance..."
  - Optimized for semantic/vector search in RAG context
  - Each phrase searches ALL target quarters in parallel
- Request additional transcript search (`needs_transcript_search`)
- Request news search (`needs_news_search`)

**Stops when:**
1. Confidence ≥ threshold (varies by answer mode: 70-95%)
2. Max iterations reached (2-10 depending on answer mode)
3. Agent decides answer is sufficient
4. No follow-up keyword phrases generated

**Answer Mode Configuration:**
| Mode | Iterations | Confidence | When Used |
|------|------------|------------|-----------|
| `direct` | 2 | 70% | Quick factual lookups |
| `standard` | 3 | 80% | Default balanced analysis |
| `detailed` | 4 | 90% | Comprehensive research |
| `deep_search` | 10 | 95% | Explicit user request only |

---

## Follow-Up Search Strategy

When the agent needs more information during iteration, it generates **search-optimized keyword phrases** (not verbose analytical questions) for better semantic/vector search.

### Keyword Phrase Generation

**OLD Approach (verbose questions):**
```
❌ "What specific revenue growth percentage was reported and how does
   it compare to the previous quarter?"
❌ "Did executives provide updated capex guidance for 2025, and what
   portion was specifically tied to AI?"
```

**NEW Approach (search-optimized keywords):**
```
✅ "revenue growth percentage quarter comparison"
✅ "capex guidance 2025 AI allocation"
✅ "specific metrics last three quarters"
```

### Why Keyword Phrases?

1. **Better semantic search**: Vector databases work better with declarative keyword phrases than natural language questions
2. **Removes noise**: Question framing words ("What", "How", "Did") don't help retrieval
3. **Focuses on entities**: Extracts core concepts, metrics, and temporal scope
4. **Preserves context**: Includes tickers and time periods from original question

### Parallel Execution

Each keyword phrase searches **ALL target quarters** in parallel:

```
Example: "last 3 quarters" = [2024_q4, 2025_q1, 2025_q2]
Follow-up phrases: ["capex guidance", "AI investments", "margin trends"]

Execution:
├── "capex guidance" → searches 2024_q4, 2025_q1, 2025_q2 (parallel)
├── "AI investments" → searches 2024_q4, 2025_q1, 2025_q2 (parallel)
└── "margin trends" → searches 2024_q4, 2025_q1, 2025_q2 (parallel)

Result: All chunks deduped by citation, merged into context
```

This ensures comprehensive coverage - if "capex guidance" appears in Q2 and Q4, both chunks are retrieved.

---

## Data Sources

### Earnings Transcript Search

For quarterly earnings questions, uses hybrid search over transcript chunks.

```python
# search_engine.py
def search_similar_chunks(query, top_k, quarter):
    """
    Hybrid search combining:
    - Vector search: 70% weight (semantic similarity via pgvector)
    - Keyword search: 30% weight (TF-IDF)
    """
```

**Database Schema:**
```
PostgreSQL Table: transcript_chunks
├── chunk_text: TEXT (1000 chars max, 200 overlap)
├── embedding: VECTOR (all-MiniLM-L6-v2, 384 dimensions)
├── ticker: VARCHAR (e.g., "AAPL")
├── year: INTEGER (e.g., 2024)
├── quarter: INTEGER (1-4)
└── metadata: JSONB
```

---

### SEC 10-K Filings Agent

**Dedicated documentation:** [docs/SEC_AGENT.md](../docs/SEC_AGENT.md)

**What it is:** A specialized **retrieval agent** optimized for extracting information from **SEC 10-K annual filings**. It uses planning-driven parallel retrieval with intelligent section routing and table selection.

**Current scope:** 10-K filings only (annual reports). Support for 10-Q (quarterly) and 8-K (current events) is under development.

**How the main agent uses it:**

The main agent uses the SEC agent as a specialized data source tool. When the Question Analyzer (Stage 2) determines that a question requires 10-K data (based on semantic routing), it automatically invokes the SEC agent during Stage 2.6 of the pipeline.

**Invocation flow:**
1. Question Analyzer sets `data_source="10k"` or `needs_10k=true`
2. Main agent calls SEC agent during Stage 2.6
3. SEC agent performs its own iterative retrieval (up to 5 iterations)
4. Results are formatted with `[10K1]`, `[10K2]` citation markers
5. Context flows back into main agent's answer generation

**Why a separate retrieval agent?** SEC 10-K filings have unique structure (15 sections, complex tables, financial statements) requiring specialized retrieval strategies. The SEC agent handles:
- Section-level routing (Item 1, Item 7, Item 8, etc.)
- LLM-based table selection from financial statements
- Hybrid search (TF-IDF + semantic) with cross-encoder reranking
- Planning-driven sub-question generation

**Benchmark:** 91% accuracy on FinanceBench (112 questions), ~10s per question

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         10-K SEARCH FLOW (max 5 iterations)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │ PHASE 0: PLAN   │   Generate sub-questions + search plan                │
│  │ • Sub-questions │   "What is inventory turnover?" →                     │
│  │ • Search plan   │     - "What is COGS?" [TABLE]                         │
│  └────────┬────────┘     - "What is inventory?" [TABLE]                    │
│           │              - "Inventory valuation?" [TEXT]                   │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ PHASE 1: PARALLEL RETRIEVAL                                         │   │
│  │ ├── Execute ALL searches in parallel (6 workers)                    │   │
│  │ │   ├── TABLE: "cost of goods sold" → LLM selects tables            │   │
│  │ │   ├── TABLE: "inventory balance" → LLM selects tables             │   │
│  │ │   └── TEXT: "inventory valuation" → hybrid search                 │   │
│  │ └── Deduplicate and combine chunks                                  │   │
│  └────────┬────────────────────────────────────────────────────────────┘   │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ PHASE 2: ANSWER │   Generate answer with ALL retrieved chunks          │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │ PHASE 3: EVAL   │   If quality >= 90% → DONE                            │
│  │                 │   Else → Replan and loop back                         │
│  └─────────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Generates targeted sub-questions for retrieval (not just original question)
- Parallel search execution for speed
- Dynamic replanning based on evaluation gaps


---

### Tavily (Real-Time News)

`tavily_service.py` provides real-time web search for current events.

**When Used:**
1. Question contains news keywords ("latest news", "recent developments")
2. Agent requests during iteration (`needs_news_search=true`)

**How It Works:**
```python
class TavilyService:
    def search_news(self, query: str, max_results: int = 5):
        """
        Returns:
            {
                "answer": "AI-generated summary",
                "results": [
                    {
                        "title": "Article headline",
                        "url": "https://...",
                        "content": "Article text",
                        "published_date": "2024-01-15"
                    }
                ]
            }
        """

    def format_news_context(self, news_results):
        """Formats with [N1], [N2] citation markers"""
```

---

## Multi-Ticker Synthesis

For questions comparing multiple companies, the agent:

1. **Parallel Processing**: Searches each ticker concurrently
2. **Ticker-Specific Rephrasing**: Creates company-specific search queries
3. **Synthesis**: Combines results into unified comparative analysis

```
Input: "Compare $AAPL and $MSFT revenue"

Process:
├── Rephrase for AAPL: "revenue and sales performance"
├── Rephrase for MSFT: "revenue and sales performance"
├── Search AAPL chunks (parallel)
├── Search MSFT chunks (parallel)
├── Synthesis prompt combines both
└── Output: Comparative analysis with both companies

Synthesis Requirements:
• ALWAYS maintain period metadata (Q1 2025, FY 2024)
• ALWAYS include ALL financial figures from ALL sources
• Show trends and comparisons across companies
• Use human-friendly format: "Q1 2025" not "2025_q1"
```

---

## Streaming Events

The agent streams real-time progress events to the frontend:

| Event Type | Description |
|------------|-------------|
| `progress` | Generic progress updates |
| `analysis` | Question analysis complete |
| `reasoning` | Agent's research planning statement |
| `news_search` | News search results |
| `10k_search` | 10-K SEC search results |
| `iteration_start` | Beginning of iteration N |
| `agent_decision` | Agent's quality assessment |
| `iteration_followup` | Follow-up questions being searched |
| `iteration_search` | New chunks found |
| `iteration_complete` | Iteration finished |
| `result` | Final answer with citations |
| `rejected` | Question rejected (out of scope) |
| `error` | Error occurred |

**Event Structure:**
```json
{
  "type": "reasoning",
  "message": "The user is asking about Microsoft's cloud strategy...",
  "step": "planning",
  "data": {
    "reasoning": "Full reasoning statement..."
  }
}
```

---

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=...           # Response generation (fallback)
CEREBRAS_API_KEY=...         # Question analysis, routing, planning
TAVILY_API_KEY=...           # Real-time news search
DATABASE_URL=postgresql://...# Main database
PG_VECTOR=postgresql://...   # Vector search database
LOGFIRE_TOKEN=...            # Observability (optional)
```

### Agent Config (`agent_config.py`)

```python
{
    "max_iterations": 4,              # General questions
    "sec_max_iterations": 5,          # 10-K questions (more thorough)
    "min_confidence_threshold": 0.90, # High bar for early stopping
    "min_completeness_threshold": 0.90,
}
```

### RAG Config (`rag/config.py`)

```python
{
    "chunks_per_quarter": 15,         # Results per quarter
    "max_quarters": 12,               # Max 3 years of data
    "max_tickers": 8,                 # Max companies per query

    # Hybrid search weights
    "keyword_weight": 0.3,
    "vector_weight": 0.7,

    # Models
    "cerebras_model": "qwen-3-235b-a22b-instruct-2507",
    "openai_model": "gpt-4.1-mini-2025-04-14",
    "embedding_model": "all-MiniLM-L6-v2",
}
```

### Answer Mode Config (`rag/config.py`)

Controls iteration depth and quality thresholds:

```python
from enum import Enum

class AnswerMode(str, Enum):
    DIRECT = "direct"           # Quick factual answers
    STANDARD = "standard"       # Balanced analysis (default)
    DETAILED = "detailed"       # Comprehensive research
    DEEP_SEARCH = "deep_search" # Exhaustive search (explicit only)

ANSWER_MODE_CONFIG = {
    AnswerMode.DIRECT: {
        "max_iterations": 2,
        "max_tokens": 2000,
        "confidence_threshold": 0.7,
    },
    AnswerMode.STANDARD: {
        "max_iterations": 3,
        "max_tokens": 6000,
        "confidence_threshold": 0.8,
    },
    AnswerMode.DETAILED: {
        "max_iterations": 4,
        "max_tokens": 16000,
        "confidence_threshold": 0.9,
    },
    AnswerMode.DEEP_SEARCH: {
        "max_iterations": 10,
        "max_tokens": 20000,
        "confidence_threshold": 0.95,
    },
}
```

**Important:** `deep_search` mode only triggers when user explicitly requests:
- "search thoroughly"
- "dig deep"
- "exhaustive search"
- "find everything"

The agent will also nudge users with "Want me to search thoroughly?" or "Should I dig deeper?" when appropriate.

---

## Usage

```python
from agent import create_agent

agent = create_agent()

# Earnings transcript question (automatic routing)
async for event in agent.execute_rag_flow(
    question="What did $AAPL say about iPhone sales in Q4 2024?",
    stream=True
):
    if event['type'] == 'reasoning':
        print(f"Planning: {event['message']}")
    elif event['type'] == 'result':
        print(f"Answer: {event['data']['answer']}")

# 10-K question (automatically routes to SEC filings)
result = await agent.execute_rag_flow_async(
    question="What was Tim Cook's compensation in 2023?"
)

# News question (automatically routes to Tavily)
result = await agent.execute_rag_flow_async(
    question="What's the latest news on $NVDA?"
)

# Multi-ticker comparison
async for event in agent.execute_rag_flow(
    question="Compare $MSFT and $GOOGL cloud revenue",
    stream=True,
    max_iterations=4
):
    print(event)
```

---

## Key Components

### Core Files

| File | Description |
|------|-------------|
| `__init__.py` | Public API — exports `Agent`, `RAGAgent`, `create_agent()` |
| `agent_config.py` | Agent configuration and iteration settings |
| `prompts.py` | Centralized LLM prompt templates (including planning) |
| `rag/rag_agent.py` | Orchestration engine with pipeline stages |
| `rag/question_analyzer.py` | LLM-based semantic routing (Cerebras) |

### Data Sources (Tools)

| File | Tool | Description |
|------|------|-------------|
| `rag/search_engine.py` | Transcript Search | Hybrid vector + keyword search |
| `rag/sec_filings_service_smart_parallel.py` | 10-K Agent | Planning-driven parallel retrieval |
| `rag/tavily_service.py` | News Search | Real-time news via Tavily API |

### Supporting Components

| File | Description |
|------|-------------|
| `rag/response_generator.py` | LLM response generation, evaluation, planning |
| `rag/database_manager.py` | PostgreSQL/pgvector operations |
| `rag/conversation_memory.py` | Multi-turn conversation state |
| `rag/config.py` | RAG configuration |

---

## Database Schema

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

---

## Limitations

- Requires `$TICKER` format for company identification
- Quarter availability varies by company
- Companies describe fiscal years differently
- No real-time stock price data
- 10-K data limited to 2024-25 filings currently

---

## Development Status

| Component | Status |
|-----------|--------|
| Semantic Data Source Routing | ✅ Production |
| Question Planning/Reasoning | ✅ Production |
| Earnings Transcript Search | ✅ Production |
| SEC 10-K Filing Search | ✅ Production (91% accuracy on FinanceBench) |
| Tavily News Search | ✅ Production |
| Multi-ticker Synthesis | ✅ Production |
| Iterative Improvement | ✅ Production |
| Streaming Events | ✅ Production |
| Conversation Memory | ✅ Production |

---

## Related Documentation

- **[Main README](../README.md)** - Project overview and setup
- **[SEC Agent](../docs/SEC_AGENT.md)** - Detailed 10-K agent: planning-driven retrieval, 91% accuracy
- **[Data Ingestion](rag/data_ingestion/README.md)** - Transcript and 10-K ingestion pipelines
