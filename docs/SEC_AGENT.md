# SEC 10-K Filings Agent

The SEC Agent provides sophisticated access to annual 10-K SEC filings using **planning-driven parallel retrieval**. It achieves **91% accuracy** on the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark with an average response time of **~10 seconds per question**.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Step-by-Step Flow](#step-by-step-flow)
- [Key Design Decisions](#key-design-decisions)
- [Configuration](#configuration)
- [Version in Use](#version-in-use)
- [Database Schema](#database-schema)
- [Examples](#examples)
- [Benchmark Results](#benchmark-results)

---

## Overview

The SEC Agent retrieves data from 10-K filings using a multi-phase approach:

| Feature | Description |
|---------|-------------|
| **Planning-Driven** | Generates targeted sub-questions, not just the original question |
| **Parallel Retrieval** | Executes multiple searches concurrently (6 workers) |
| **Dynamic Replanning** | Adjusts search strategy based on evaluation feedback |
| **Early Termination** | Stops when confidence ≥ 90% (typically 1-3 iterations) |
| **Hybrid Search** | Combines semantic (70%) + TF-IDF (30%) with cross-encoder reranking |

### Benchmark Performance

```
Accuracy:     91% on FinanceBench (112 10-K questions)
Avg Time:     ~10 seconds per question
Avg Iters:    2.4 iterations (out of max 5)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SMART PARALLEL SEC AGENT FLOW                              │
│                    (max 5 iterations, typically 1-3)                          │
└──────────────────────────────────────────────────────────────────────────────┘

                              USER QUESTION
         "What is AMD's inventory turnover ratio for FY2022?"
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: INTELLIGENT PLANNING                                                │
│  ═════════════════════════════                                                │
│                                                                               │
│  LLM generates targeted sub-questions (NOT just the original question):      │
│                                                                               │
│  {                                                                            │
│    "sub_questions": [                                                         │
│      "What is the cost of goods sold (COGS)?",                               │
│      "What is the ending inventory balance?",                                 │
│      "What is the beginning inventory balance?",                              │
│      "How is inventory valued and managed?"                                   │
│    ],                                                                         │
│    "search_plan": [                                                           │
│      {"query": "cost of goods sold COGS", "type": "table", "priority": 1},   │
│      {"query": "inventory balance", "type": "table", "priority": 1},          │
│      {"query": "inventory valuation method", "type": "text", "priority": 2}   │
│    ]                                                                          │
│  }                                                                            │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PARALLEL MULTI-QUERY RETRIEVAL                                      │
│  ═══════════════════════════════════════                                      │
│                                                                               │
│  ThreadPoolExecutor (6 workers) executes ALL searches concurrently:          │
│                                                                               │
│  ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐        │
│  │ SubQ 1: COGS       │ │ SubQ 2: Inventory  │ │ SubQ 3: Valuation  │        │
│  │ Type: TABLE        │ │ Type: TABLE        │ │ Type: TEXT         │        │
│  │                    │ │                    │ │                    │        │
│  │ LLM selects:       │ │ LLM selects:       │ │ Hybrid search:     │        │
│  │ • Income Statement │ │ • Balance Sheet    │ │ • Semantic 70%     │        │
│  │                    │ │                    │ │ • TF-IDF 30%       │        │
│  │                    │ │                    │ │ • Cross-encoder    │        │
│  └─────────┬──────────┘ └─────────┬──────────┘ └─────────┬──────────┘        │
│            │                      │                      │                    │
│            └──────────────────────┼──────────────────────┘                    │
│                                   │                                           │
│                                   ▼                                           │
│                      ┌────────────────────────┐                               │
│                      │   COMBINE & DEDUPE     │                               │
│                      │   All retrieved chunks │                               │
│                      └────────────────────────┘                               │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: ANSWER GENERATION                                                   │
│  ══════════════════════════                                                   │
│                                                                               │
│  LLM generates answer using ALL accumulated chunks:                           │
│  • Address each sub-question                                                  │
│  • Cite sources as [10K1], [10K2], etc.                                       │
│  • Calculate derived metrics (e.g., turnover ratio)                           │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: QUALITY EVALUATION                                                  │
│  ═══════════════════════════                                                  │
│                                                                               │
│  Evaluate answer quality (0-100 scale):                                       │
│  • completeness_score: Does it fully answer the question?                     │
│  • specificity_score: Does it include specific numbers?                       │
│  • accuracy_score: Is it factually correct?                                   │
│  • clarity_score: Is it well-structured?                                      │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ IF quality >= 90%  →  EARLY TERMINATION (return answer)                 │ │
│  │ IF quality < 90%   →  Continue to PHASE 4 (replanning)                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────┬─────────────────────────────────────────────┘
                                 │ (if quality < 90%)
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: DYNAMIC REPLANNING                                                  │
│  ═══════════════════════════                                                  │
│                                                                               │
│  Based on evaluation.missing_info, generate NEW search queries:               │
│                                                                               │
│  Evaluation says: "Missing prior year inventory for average calculation"      │
│                   │                                                           │
│                   ▼                                                           │
│  New search plan: [{"query": "FY2021 ending inventory", "type": "table"}]    │
│                                                                               │
│  Loop back to PHASE 1 with new queries (max 5 total iterations)               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Flow

### Phase 0: Intelligent Planning

The agent generates targeted sub-questions instead of repeating the original question:

```python
def plan_investigation_with_search_strategy(self, question, model):
    """
    Input: "What is AMD's inventory turnover ratio?"

    Output:
    {
        "sub_questions": [
            "What is COGS?",
            "What is ending inventory?",
            "What is beginning inventory?"
        ],
        "search_plan": [
            {"query": "cost of goods sold", "type": "table"},
            {"query": "inventory balance", "type": "table"}
        ]
    }
    """
```

**Why this matters:** Using targeted sub-questions instead of the original question retrieves different, specific information for each information need.

### Phase 1: Parallel Retrieval

All search queries execute concurrently using ThreadPoolExecutor:

```python
async def parallel_multi_query_retrieval(self, search_plan, model):
    """
    Executes searches in parallel:
    - TABLE queries: LLM selects relevant tables
    - TEXT queries: Hybrid search + cross-encoder reranking

    Returns deduplicated combined chunks.
    """
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(self._execute_search, item)
            for item in search_plan
        ]
        results = [f.result() for f in futures]

    return self._dedupe_and_combine(results)
```

### Phase 2: Answer Generation

Uses ALL accumulated chunks to generate a comprehensive answer:

```python
def _generate_answer(self, question, sub_questions, chunks, previous_answer):
    """
    Generates answer addressing:
    - Original question
    - Each sub-question
    - Calculations where needed

    Citations: [10K1], [10K2], etc.
    """
```

### Phase 3: Quality Evaluation

Strict evaluation on 0-100 scale:

```python
evaluation = {
    "quality_score": 0.85,
    "issues": ["missing YoY comparison"],
    "missing_info": ["prior year inventory figure"],
    "suggestions": ["search for FY2021 balance sheet"]
}

# Early termination if quality >= 0.90
if evaluation["quality_score"] >= 0.90:
    return final_answer
```

### Phase 4: Dynamic Replanning

Generates new search queries based on evaluation gaps:

```python
def replan_based_on_evaluation(self, evaluation, current_subquestions):
    """
    Input: evaluation.missing_info = ["prior year inventory"]

    Output: [{"query": "FY2021 inventory balance", "type": "table"}]
    """
```

---

## Key Design Decisions

### 1. Sub-Questions Over Original Question

**Problem:** Iterative approach used same query repeatedly, getting same results.
**Solution:** Generate targeted sub-questions for specific information needs.

### 2. Parallel Over Sequential

**Problem:** Sequential iterations were slow (~170s/question).
**Solution:** Execute all searches concurrently with ThreadPoolExecutor.

### 3. Table-First for Numeric Questions

Financial data questions prioritize table retrieval:

```python
FINANCIAL_KEYWORDS = [
    'revenue', 'income', 'profit', 'assets', 'liabilities',
    'earnings', 'sales', 'expenses', 'equity', 'cash flow',
    'ratio', 'margin', 'million', 'billion', 'percent', 'eps'
]

if any(kw in question.lower() for kw in FINANCIAL_KEYWORDS):
    prioritize_tables = True
```

### 4. LLM-Based Table Selection

Instead of retrieving all tables, LLM selects the most relevant:

```python
def select_tables_by_llm(self, question, available_tables, iteration):
    """
    LLM sees all available tables and selects 1-2 most relevant.

    Prioritizes core financial statements:
    - 🌟 Income Statement
    - 🌟 Balance Sheet
    - 🌟 Cash Flow Statement

    Avoids selecting same tables as previous iterations.
    """
```

### 5. Cross-Encoder Reranking

Text chunks are reranked using ms-marco cross-encoder for better relevance:

```python
def rerank_chunks(self, query, chunks, top_k=10):
    """
    Uses cross-encoder (ms-marco-MiniLM-L-6-v2) to rerank
    hybrid search results for better precision.
    """
```

---

## Configuration

### Environment Variables

```bash
CEREBRAS_API_KEY=...         # Primary LLM (Qwen-3-235B)
OPENAI_API_KEY=...           # Fallback LLM
DATABASE_URL=postgresql://...# 10-K chunks and tables
```

### Agent Settings

```python
# In The agentSECFilingsService
max_iterations = 5           # Maximum iterations per question
confidence_threshold = 0.90  # Quality score for early termination
parallel_workers = 6         # ThreadPoolExecutor workers

# Hybrid search weights
semantic_weight = 0.70
tfidf_weight = 0.30
```

---

## Version in Use

Loaded in `agent/rag/rag_agent.py`:

```python
from .sec_filings_service_smart_parallel import SmartParallelSECFilingsService as SECFilingsService
```

---

## Database Schema

### ten_k_chunks

```sql
CREATE TABLE ten_k_chunks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    sec_section VARCHAR(20),        -- 'item_1', 'item_7', 'item_8', etc.
    sec_section_title TEXT,         -- 'Business', 'MD&A', 'Financial Statements'
    chunk_text TEXT NOT NULL,
    chunk_type VARCHAR(20),         -- 'text' or 'table'
    embedding VECTOR(384),          -- all-MiniLM-L6-v2
    is_financial_statement BOOLEAN DEFAULT FALSE,
    statement_type VARCHAR(50),
    path_string TEXT,
    metadata JSONB
);
```

### ten_k_tables

```sql
CREATE TABLE ten_k_tables (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    sec_section VARCHAR(20),
    sec_section_title TEXT,
    content TEXT,                   -- Table content as markdown
    statement_type VARCHAR(50),     -- 'income_statement', 'balance_sheet', 'cash_flow'
    is_financial_statement BOOLEAN DEFAULT FALSE,
    path_string TEXT,
    metadata JSONB
);
```

---

## Examples

### Example 1: Inventory Turnover Ratio (Numeric)

```
User: "What is AMD's inventory turnover ratio for FY2022?"

PHASE 0: PLANNING
  Sub-questions:
  - "What is cost of goods sold (COGS)?"
  - "What is ending inventory balance?"
  - "What is beginning inventory balance?"

  Search plan:
  - {"query": "cost of goods sold COGS", "type": "table"}
  - {"query": "inventory balance assets", "type": "table"}

PHASE 1: PARALLEL RETRIEVAL
  ├── TABLE: COGS → LLM selects Income Statement
  ├── TABLE: Inventory → LLM selects Balance Sheet
  └── Combines both tables

PHASE 2: ANSWER GENERATION
  "AMD's inventory turnover ratio for FY2022:
   - COGS: $13.5B [10K1]
   - Avg Inventory: ($4.3B + $1.9B) / 2 = $3.1B [10K2]
   - Turnover: 13.5 / 3.1 = 4.35x"

PHASE 3: EVALUATION
  quality_score = 0.92 ≥ 0.90 → EARLY TERMINATION

Result: 1 iteration, ~8 seconds
```

### Example 2: Risk Factors (Qualitative)

```
User: "What are Tesla's main risk factors?"

PHASE 0: PLANNING
  Sub-questions:
  - "What operational risks does Tesla face?"
  - "What regulatory/legal risks exist?"
  - "What financial/market risks are mentioned?"

  Search plan:
  - {"query": "operational risks manufacturing", "type": "text"}
  - {"query": "regulatory legal risks", "type": "text"}
  - {"query": "market competition risks", "type": "text"}

PHASE 1: PARALLEL RETRIEVAL
  ├── TEXT: Operational → Item 1A chunks
  ├── TEXT: Regulatory → Item 1A chunks
  └── TEXT: Market → Item 1A chunks

PHASE 2: ANSWER GENERATION
  "Tesla's main risk factors include:
   1. Supply chain dependencies [10K1]
   2. Regulatory uncertainty [10K2]
   3. Competition from legacy automakers [10K3]..."

PHASE 3: EVALUATION
  quality_score = 0.88 < 0.90 → Continue

PHASE 4: REPLANNING
  Missing: "Specific examples of each risk"
  New query: {"query": "risk factor examples incidents", "type": "text"}

ITERATION 2: Retrieves more specific examples
  quality_score = 0.93 ≥ 0.90 → EARLY TERMINATION

Result: 2 iterations, ~12 seconds
```

### Example 3: Complex Multi-Part Question

```
User: "What was Microsoft's operating margin and why did it change?"

PHASE 0: PLANNING
  Sub-questions:
  - "What was the operating income?"
  - "What was total revenue?"
  - "What factors affected operating expenses?"
  - "How did margin compare to prior year?"

  Search plan:
  - {"query": "operating income revenue", "type": "table"}
  - {"query": "operating expenses changes", "type": "text"}
  - {"query": "margin drivers cost efficiency", "type": "text"}

PHASE 1: PARALLEL RETRIEVAL (all concurrent)
PHASE 2: ANSWER with numbers + explanation
PHASE 3: EVALUATION → 0.85 (missing YoY comparison)
PHASE 4: REPLAN → add prior year search

ITERATION 2: Add prior year data
  quality_score = 0.91 → EARLY TERMINATION

Result: 2 iterations, ~14 seconds
```

---

## Benchmark Results

### FinanceBench Evaluation (112 10-K Questions)

| Metric | Result |
|--------|--------|
| Accuracy | **91%** |
| Avg Time | **10.7s** per question |
| Avg Iterations | 2.4 (out of max 5) |

### Performance Breakdown

```
Timing per question:
├── Phase 0 (Planning):     ~1.5s
├── Phase 1 (Retrieval):    ~3.5s (parallel)
├── Phase 2 (Answer):       ~2.5s
├── Phase 3 (Evaluation):   ~1.5s
└── Total (1 iteration):    ~9s

With 2.4 avg iterations: ~10.7s total
```

### Why It's Fast

1. **Parallel execution** - 6 searches run concurrently
2. **Targeted sub-questions** - Each query retrieves different, specific information
3. **Fewer iterations** - Better initial retrieval means earlier termination

---

## Related Documentation

- **[Agent README](../agent/README.md)** - Full agent architecture
- **[Data Ingestion](../agent/rag/data_ingestion/README.md)** - 10-K ingestion pipeline
