# SEC 10-K Filings Agent (Iterative)

The SEC Agent provides sophisticated access to annual 10-K SEC filings using an **iterative step-by-step approach** where the agent decides at each iteration whether to retrieve **TABLE** or **TEXT** chunks, evaluates the answer quality, and refines until confident.

## Table of Contents

- [Overview](#overview)
- [Iterative Architecture](#iterative-architecture)
- [Step-by-Step Flow](#step-by-step-flow)
- [Decision Logic](#decision-logic)
- [Configuration](#configuration)
- [Switching Between Versions](#switching-between-versions)
- [Database Schema](#database-schema)
- [Examples](#examples)

---

## Overview

The SEC Agent implements an **agentic iterative loop** for 10-K filing search:

| Feature | Description |
|---------|-------------|
| **Iterative Decisions** | At each iteration, LLM decides: retrieve TABLE or TEXT? |
| **Quality Evaluation** | After each iteration, evaluates answer completeness |
| **Dynamic Switching** | If tables don't yield results, switches to text (and vice versa) |
| **Early Termination** | Stops when confidence ≥ 90% or max iterations (5) reached |
| **Table-First Rule** | For numeric/financial questions, starts with tables |

### Why Iterative?

Unlike the one-pass approach (fetch everything at once), the iterative approach:
- **Makes intelligent decisions** about what to retrieve next
- **Avoids wasting tokens** on irrelevant data
- **Builds answers step-by-step** with increasing quality
- **Can course-correct** if initial retrieval doesn't yield good results

---

## Iterative Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ITERATIVE 10-K SEARCH PIPELINE                         │
└──────────────────────────────────────────────────────────────────────────┘

                              USER QUESTION
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  PHASE 0: DECIDE INITIAL RETRIEVAL TYPE                                   │
│  ═══════════════════════════════════════                                  │
│                                                                           │
│  LLM analyzes question:                                                   │
│  • Financial keywords (revenue, assets, EPS)? → Start with TABLE          │
│  • Qualitative (risks, strategy, description)? → Start with TEXT          │
│                                                                           │
│  TABLE-FIRST RULE: Numeric queries ALWAYS start with tables               │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ITERATION LOOP (max 5 iterations)                                        │
│  ═════════════════════════════════                                        │
│                                                                           │
│  FOR EACH ITERATION:                                                      │
│                                                                           │
│  ┌─ STEP 1: DECIDE RETRIEVAL TYPE ─────────────────────────────────────┐ │
│  │  LLM decides based on:                                               │ │
│  │  • Previous evaluation feedback                                      │ │
│  │  • What sources have been tried                                      │ │
│  │  • What's missing from current answer                                │ │
│  │                                                                      │ │
│  │  DYNAMIC SWITCHING:                                                  │ │
│  │  • If only TABLE tried → force TEXT next                             │ │
│  │  • If only TEXT tried → force TABLE next                             │ │
│  │  • If low quality after one type → try the other                     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                            │                                              │
│                            ▼                                              │
│  ┌─ STEP 2: RETRIEVE CHUNKS ───────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  IF TABLE:                                                           │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • LLM sees ALL available tables for ticker                     │ │ │
│  │  │ • LLM selects 1-2 most relevant tables                         │ │ │
│  │  │ • Prioritizes 🌟 CORE financial statements:                    │ │ │
│  │  │   - Income Statement                                           │ │ │
│  │  │   - Balance Sheet                                              │ │ │
│  │  │   - Cash Flow Statement                                        │ │ │
│  │  │ • Avoids selecting same tables as previous iterations          │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  IF TEXT:                                                            │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │ • Hybrid search: 70% semantic + 30% TF-IDF                     │ │ │
│  │  │ • Cross-encoder reranking (ms-marco model)                     │ │ │
│  │  │ • Returns top 10 text chunks                                   │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  Add retrieved chunks to accumulated_chunks (avoid duplicates)       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                            │                                              │
│                            ▼                                              │
│  ┌─ STEP 3: GENERATE/REFINE ANSWER ────────────────────────────────────┐ │
│  │  • Uses ALL accumulated chunks (tables + text from all iterations)   │ │
│  │  • Builds on previous answer with new information                    │ │
│  │  • Cites sources as [1], [2], etc.                                   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                            │                                              │
│                            ▼                                              │
│  ┌─ STEP 4: EVALUATE ANSWER QUALITY ───────────────────────────────────┐ │
│  │  LLM evaluates:                                                      │ │
│  │  • quality_score (0.0 - 1.0)                                         │ │
│  │  • issues: ["missing revenue figure", "no context for change"]       │ │
│  │  • missing_info: ["need specific numbers", "need explanation"]       │ │
│  │  • suggestions: ["try tables for numbers", "try text for context"]   │ │
│  │                                                                      │ │
│  │  EARLY TERMINATION:                                                  │ │
│  │  If quality_score >= 0.9 (90% confidence) → STOP                     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                            │                                              │
│                            ▼                                              │
│  ┌─ STEP 5: DECIDE NEXT RETRIEVAL STRATEGY ────────────────────────────┐ │
│  │  Based on evaluation:                                                │ │
│  │  • Missing numbers/metrics → next type = TABLE                       │ │
│  │  • Missing context/explanation → next type = TEXT                    │ │
│  │  • Low quality after TABLE → switch to TEXT                          │ │
│  │  • Low quality after TEXT → switch to TABLE                          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  REPEAT until: quality >= 0.9 OR iteration == 5                           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  FINAL RESULT                                                             │
│  ════════════                                                             │
│                                                                           │
│  Returns:                                                                 │
│  • accumulated_chunks: All chunks from all iterations                     │
│  • session: Full iteration history for debugging                          │
│  • Citations formatted as [10K1], [10K2], etc.                            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Flow

### Phase 0: Initial Decision

```python
# Decides whether to start with TABLE or TEXT

Financial keywords detected? → Start with TABLE
├── revenue, income, profit, assets, liabilities
├── earnings, sales, expenses, equity, cash flow
├── ratio, margin, growth rate, million, billion
└── percent, dollar, amount, total, figure, eps

No financial keywords? → Start with TEXT
├── risks, strategy, description
├── management, business, operations
└── qualitative information
```

### Iteration Steps

**Step 1: Decide Retrieval Type**
```
Input: question, current_answer, iteration, evaluation_history

DYNAMIC SWITCHING RULES:
├── If table_count=0 and text_count>0 → Force TABLE
├── If text_count=0 and table_count>0 → Force TEXT
├── If quality<0.7 after TABLE → Try TEXT
└── If quality<0.7 after TEXT → Try TABLE
```

**Step 2: Retrieve Chunks**
```
IF TABLE:
  1. Get all tables for ticker from database
  2. LLM selects 1-2 most relevant (avoids duplicates)
  3. Converts to chunk format with selection_reasoning

IF TEXT:
  1. Hybrid search (semantic + TF-IDF)
  2. Cross-encoder reranking
  3. Returns top 10 text chunks
```

**Step 3: Generate Answer**
```
Uses ALL accumulated chunks from ALL iterations
Builds on previous answer with new information
Cites sources for traceability
```

**Step 4: Evaluate Quality**
```
quality_score: 0.0 - 1.0
├── COMPLETENESS: Does it fully answer the question?
├── SPECIFICITY: Does it include specific numbers?
├── ACCURACY: Is it supported by the context?
└── CLARITY: Is it well-structured?

If quality_score >= 0.9 → EARLY TERMINATION
```

**Step 5: Decide Next Strategy**
```
Based on evaluation feedback:
├── missing_info contains "number/figure/metric" → TABLE
├── missing_info contains "context/explanation" → TEXT
├── Low quality after tables → TEXT
└── Low quality after text → TABLE
```

---

## Decision Logic

### Table-First Rule

For questions with financial keywords, the agent **always starts with tables**:

```python
FINANCIAL_KEYWORDS = [
    'revenue', 'income', 'profit', 'loss', 'earnings', 'sales', 'expenses',
    'assets', 'liabilities', 'equity', 'cash flow', 'ratio', 'margin',
    'million', 'billion', 'percent', 'dollar', 'amount', 'total', 'eps'
]

if any(kw in question.lower() for kw in FINANCIAL_KEYWORDS):
    initial_type = 'table'
```

### Dynamic Switching

The agent ensures **both sources are explored**:

```python
# If only text tried, force trying tables
if table_count == 0 and text_count > 0:
    next_type = 'table'
    reasoning = 'Dynamic switch: text tried but tables not yet explored'

# If only tables tried, force trying text
elif text_count == 0 and table_count > 0:
    next_type = 'text'
    reasoning = 'Dynamic switch: tables tried but text not yet explored'
```

### Quality-Based Switching

If quality is low after one type, switch to the other:

```python
if quality_score < 0.7:
    if last_type == 'table':
        next_type = 'text'
        reasoning = f'Low quality ({quality_score:.2f}) after tables - trying text'
    else:
        next_type = 'table'
        reasoning = f'Low quality ({quality_score:.2f}) after text - trying tables'
```

---

## Configuration

### Environment Variables

```bash
CEREBRAS_API_KEY=...         # For LLM decisions (table selection, evaluation)
OPENAI_API_KEY=...           # Fallback
DATABASE_URL=postgresql://...# 10-K chunks and tables
```

### Iteration Limits

```python
# Hard-coded in IterativeSECFilingsService
self.max_iterations = 5  # Maximum iterations per question

# In RAG agent call
max_iterations=5,        # Passed to execute_iterative_search_async
confidence_threshold=0.9 # 90% confidence for early termination
```

---

## Switching Between Versions

The iterative and one-pass versions can be switched by changing the import in `rag_agent.py`:

### Use Iterative (Current Default)
```python
# In agent/rag/rag_agent.py
from .sec_filings_service_iterative import IterativeSECFilingsService as SECFilingsService
```

### Use One-Pass (Fallback)
```python
# In agent/rag/rag_agent.py
from .sec_filings_service import SECFilingsService
```

### Files

| File | Description |
|------|-------------|
| `sec_filings_service_iterative.py` | **Current**: Iterative step-by-step approach |
| `sec_filings_service.py` | **Fallback**: One-pass approach |

---

## Database Schema

### ten_k_chunks

```sql
CREATE TABLE ten_k_chunks (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    fiscal_year INTEGER NOT NULL,
    sec_section VARCHAR(20),
    sec_section_title TEXT,
    chunk_text TEXT NOT NULL,
    chunk_type VARCHAR(20),  -- 'text' or 'table'
    embedding VECTOR(384),
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
    content TEXT,  -- Table content as text/markdown
    statement_type VARCHAR(50),  -- income_statement, balance_sheet, cash_flow
    is_financial_statement BOOLEAN DEFAULT FALSE,
    path_string TEXT,
    metadata JSONB
);
```

---

## Examples

### Example 1: Revenue Question (Numeric)

```
User: "What was Apple's revenue in FY 2024?"

PHASE 0: Financial keyword "revenue" detected → Start with TABLE

ITERATION 1:
  Step 1: Retrieve TABLE (initial decision)
  Step 2: LLM selects "Income Statement" table
  Step 3: Generate answer with revenue figures
  Step 4: Evaluate → quality_score = 0.85 (has numbers, missing context)
  Step 5: Next type = TEXT (to add context)

ITERATION 2:
  Step 1: Retrieve TEXT
  Step 2: Hybrid search returns MD&A discussion of revenue
  Step 3: Refine answer with context about growth drivers
  Step 4: Evaluate → quality_score = 0.92 ≥ 0.9
  → EARLY TERMINATION

Result: "Apple's revenue in FY 2024 was $XXX billion, representing X% YoY growth..."
  - 2 iterations
  - 1 table retrieval, 1 text retrieval
```

### Example 2: Risk Factors (Qualitative)

```
User: "What are Tesla's main risk factors?"

PHASE 0: No financial keywords → Start with TEXT

ITERATION 1:
  Step 1: Retrieve TEXT (initial decision)
  Step 2: Hybrid search returns Item 1A risk factor chunks
  Step 3: Generate answer listing risks
  Step 4: Evaluate → quality_score = 0.88 (good list, missing specific examples)
  Step 5: Next type = TEXT (more detail needed)

ITERATION 2:
  Step 1: Retrieve TEXT
  Step 2: Returns more specific risk factor details
  Step 3: Refine answer with examples
  Step 4: Evaluate → quality_score = 0.93 ≥ 0.9
  → EARLY TERMINATION

Result: "Tesla's main risk factors include: 1) Supply chain dependencies..."
  - 2 iterations
  - 2 text retrievals, 0 table retrievals
```

### Example 3: Complex Question (Needs Both)

```
User: "What was Microsoft's operating margin and why did it change?"

PHASE 0: Financial keywords detected → Start with TABLE

ITERATION 1:
  Step 1: Retrieve TABLE
  Step 2: LLM selects "Income Statement" table
  Step 3: Generate answer with margin calculation
  Step 4: Evaluate → quality_score = 0.6 (has number, missing explanation)
  Step 5: Next type = TEXT (need "why")

ITERATION 2:
  Step 1: Retrieve TEXT (dynamic switch - need context)
  Step 2: Returns MD&A discussion of margin drivers
  Step 3: Refine with explanation of changes
  Step 4: Evaluate → quality_score = 0.85 (better, still missing YoY comparison)
  Step 5: Next type = TABLE

ITERATION 3:
  Step 1: Retrieve TABLE
  Step 2: LLM selects prior year comparison table
  Step 3: Add YoY comparison
  Step 4: Evaluate → quality_score = 0.91 ≥ 0.9
  → EARLY TERMINATION

Result: "Microsoft's operating margin was X% in FY 2024, up from Y% in FY 2023..."
  - 3 iterations
  - 2 table retrievals, 1 text retrieval
```

---

## Related Documentation

- **[Agent README](../agent/README.md)** - Full agent documentation
- **[One-Pass SEC Service](../agent/rag/sec_filings_service.py)** - Fallback version
- **[Data Ingestion](../agent/rag/data_ingestion/README.md)** - 10-K ingestion pipeline
