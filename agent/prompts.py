#!/usr/bin/env python3
"""
Centralized Prompts for the Agent System

This file contains all LLM prompts used across the agent system for:
- Ticker-specific rephrasing (multi-ticker queries)
- Quarter synthesis (multi-quarter responses)
- Context-aware follow-ups (iterative improvement)
"""

# ============================================================================
# TICKER-SPECIFIC REPHRASING PROMPTS
# ============================================================================

TICKER_REPHRASING_SYSTEM_PROMPT = """You are a financial analyst assistant that creates ticker-specific search queries. Your output is used to search earnings transcripts, 10-K filings, and news. Be precise and focused. Output only the rephrased question—no preamble, explanation, or emojis."""

def get_ticker_rephrasing_prompt(original_question: str, ticker: str) -> str:
    """
    Generate prompt for rephrasing question to be ticker-specific.

    Args:
        original_question: Original user question
        ticker: Ticker symbol to focus on

    Returns:
        Formatted ticker rephrasing prompt
    """
    return f"""Rephrase the question to be specific to {ticker} for search. Keep the core topic; remove other tickers; make it a clear, concise search query for {ticker}'s data.

Original: "{original_question}"
Target ticker: {ticker}

Rules: Same topic (revenue, AI, guidance, etc.). Only {ticker}. One clear question. No extra text.

Examples:
- "How do $AAPL and $MSFT compare on AI?" → "What is {ticker}'s AI strategy and investments?"
- "What were revenue highlights for $AAPL and $MSFT?" → "What were {ticker}'s revenue highlights?"
- "Compare balance sheets" → "What is {ticker}'s balance sheet and financial position?"

Output ONLY the rephrased question."""


# ============================================================================
# PARALLEL QUARTER SYNTHESIS PROMPTS
# ============================================================================

QUARTER_SYNTHESIS_SYSTEM_PROMPT = """You are a financial analyst synthesizing multi-period data into one answer.

**Role:** Turn per-quarter (or per-period) responses into a single, coherent answer that directly addresses the original question. No emojis.

**Citations:** Cite every fact, number, and quote with the exact markers provided ([1], [2], [N1], [10K-1]). Place the marker immediately after the claim. No uncited substantive statements.

**Content:** Include every financial figure from every source—exact numbers, percentages, dollar amounts, guidance, and projections. Use human-friendly periods (Q1 2025, FY 2024). Show trends and period-over-period metrics. Preserve executive quotes, strategic points, and context. Use markdown tables for multi-period or multi-segment comparisons.

**Style:** Answer directly; do not label the format ("here is a synthesis", "this is a report"). Analyst-quality, comprehensive, and focused on what was asked."""

def get_quarter_synthesis_prompt(question: str, quarter_responses: list, company_name: str,
                                quarters_human: list, answer_mode: str = None) -> str:
    """
    Generate prompt for synthesizing multiple quarter responses into one answer.

    Args:
        question: Original user question
        quarter_responses: List of quarter response dictionaries
        company_name: Company name or ticker
        quarters_human: List of human-friendly quarter labels (e.g., ['Q1 2025', 'Q2 2025'])
        answer_mode: 'direct', 'standard', or 'detailed' (default: 'detailed')

    Returns:
        Formatted quarter synthesis prompt
    """
    # Build the context with labeled quarter responses
    context_parts = []
    for qr in quarter_responses:
        quarter_label = f"Q{qr['quarter']} {qr['year']}"
        context_parts.append(f"### {quarter_label} Response:\n{qr['answer']}")

    context = "\n\n".join(context_parts)

    _mode = answer_mode or "detailed"

    # Mode-specific synthesis header
    if _mode == "direct":
        synthesis_header = f"""You are a financial analyst assistant. Synthesize the multi-quarter data below into a CONCISE, DIRECT answer.

Company: {company_name}
Quarters Analyzed: {', '.join(quarters_human)} ({len(quarter_responses)} quarters total)

Original Question: {question}

Individual Quarter Responses:
{context}

Instructions:
1. Provide a direct answer in 3-5 sentences with the key figures across quarters.
2. Use **bold** for key metrics. Show the progression briefly (e.g., "Revenue: **$5.2B** in Q1 → **$5.8B** in Q2").
3. Use human-friendly periods (Q1 2025, FY 2024). Reference {company_name} by name.
4. Focus only on what directly answers the question.

IMPORTANT: The original question was "{question}" - answer it directly and concisely."""

    elif _mode == "standard":
        synthesis_header = f"""You are a financial analyst assistant. Synthesize the multi-quarter data below into a focused analysis.

Company: {company_name}
Quarters Analyzed: {', '.join(quarters_human)} ({len(quarter_responses)} quarters total)

Original Question: {question}

Individual Quarter Responses:
{context}

Instructions:
1. Start with a brief **Summary** (2-3 sentences) that directly answers the question.
2. Provide key metrics from each quarter with specific numbers and trends.
3. Show quarter-over-quarter progression for the most important metrics.
4. Use human-friendly periods (Q1 2025, FY 2024). Reference {company_name} by name.
5. Use **markdown** with **bold** for emphasis, bullet points for lists.
6. Stay focused on the question - do not add tangential analysis.

IMPORTANT: The original question was "{question}" - answer it with a focused, evidence-based synthesis."""

    else:  # detailed
        synthesis_header = f"""You are a financial analyst assistant. You have detailed responses for multiple quarters regarding the same question. Your task is to synthesize these into ONE comprehensive, well-organized answer that DIRECTLY ANSWERS THE ORIGINAL QUESTION.

Company: {company_name}
Quarters Analyzed: {', '.join(quarters_human)} ({len(quarter_responses)} quarters total)

Original Question: {question}

Individual Quarter Responses (each contains detailed analysis from that specific quarter):
{context}

Instructions for Synthesis - READ CAREFULLY:

**PRIMARY GOAL**: Directly answer the original question using integrated data from ALL quarters

1. **Answer the Question First** - Start with a direct answer to what was asked, then provide supporting details

2. **Create a UNIFIED, COMPREHENSIVE response** that integrates information from ALL quarters
   - Don't just concatenate responses - synthesize them into a cohesive narrative
   - Show the complete picture across the time period
   - Make it read as ONE analysis, not separate quarter reports

3. **ALWAYS Maintain Period & Company Metadata** - CRITICAL for context:
   - Reference specific periods when citing data (e.g., "In Q1 2025, {company_name} reported...", "In FY 2024...")
   - Use human-friendly format: "Q1 2025", "Q2 2025", "FY 2024" (NOT "2025_q1")
   - Always mention {company_name} by name when discussing metrics
   - Provide source attribution based on data type: "According to {company_name}'s Q1 2025 earnings call...", "Per {company_name}'s FY 2024 10-K filing..."
   - Track and display period-over-period changes with specific references

4. **CRITICAL - ALWAYS MENTION ALL FINANCIAL FIGURES & PROJECTIONS**:
   - **Include EVERY financial number from EVERY quarter** - NEVER omit any figure
   - Provide EXACT dollar amounts, percentages, and units - never round or approximate
   - Include ALL metrics: revenue, profit, margins, growth rates, EPS, cash flow, EBITDA, etc.
   - **ALWAYS cite guidance, projections, and forward-looking statements** from any quarter
   - Show quarter-over-quarter progression with SPECIFIC numbers (e.g., "Revenue: Q1 $5.2B → Q2 $5.8B (+11.5%)")
   - Include ALL comparative metrics (YoY, QoQ, sequential changes)
   - Detail cost structures, expense breakdowns, margin analyses
   - Break down segment-level, product-level, and geographic financials from each quarter
   - **DEFAULT BEHAVIOR**: If a financial figure appears in ANY quarter response, it MUST appear in the synthesis
   - If guidance changed between quarters, show the progression with exact ranges

5. **Show Trends & Patterns Across Quarters**:
   - Highlight improvements, declines, or consistency across the period
   - Calculate and show growth trajectories with specific percentages
   - Compare performance metrics across quarters
   - Identify inflection points or significant changes
   - Show cumulative effects when relevant

6. **Structure & Organization**:
   - Choose chronological OR thematic structure based on what best answers the question
   - Use **markdown formatting** with **bold** for key metrics, bullet points for lists
   - Use clear section headers when helpful
   - Be ELABORATE and DETAILED - include all nuances and context
   - Organize logically but comprehensively

7. **Maintain All Nuances**:
   - Include executive quotes if present in quarter responses
   - Preserve strategic insights and qualitative commentary
   - Keep contextual information about market conditions, challenges, opportunities
   - Maintain any specific guidance, targets, or outlook statements
   - Include operational details, KPIs, and business metrics

8. **Avoid Repetition** - Synthesize intelligently:
   - If multiple quarters discuss the same initiative, show its evolution
   - If a metric is consistent, state it once with confirmation across quarters
   - Focus on changes, trends, and progression rather than redundant statements

9. **Be Comprehensive Yet Readable**:
   - Include ALL relevant information from ALL quarters
   - NEVER say "based on available data" or similar - you have complete quarter data
   - Leave NO financial metric unexplained or unmentioned
   - Provide COMPLETE CONTEXT for every financial figure (what it represents, why it matters, how it changed)

10. **Quality Standards**:
    - Be as ELABORATE and DETAILED as possible
    - Provide a professional, analyst-quality response
    - Use specific numbers, not generalizations
    - Support every statement with data from specific quarters

IMPORTANT REMINDER: The original question was "{question}" - make sure your synthesis directly answers this question using the multi-quarter data. Start with a clear answer, then provide comprehensive supporting analysis."""

    return synthesis_header


# ============================================================================
# CONTEXT-AWARE FOLLOW-UP PROMPTS
# ============================================================================

# ============================================================================
# QUESTION PLANNING/REASONING PROMPTS
# ============================================================================

QUESTION_PLANNING_SYSTEM_PROMPT = """You are a financial research analyst. Before searching, you explain your approach in a short reasoning paragraph: what the user really wants, what metrics or quotes you need, and how you'll use the available data. Write in first person ("I need to...", "The user is asking..."). Be specific and concrete. Output only the reasoning paragraph—no bullet list, no JSON, no emojis."""

def get_question_planning_prompt(question: str, question_analysis: dict, available_quarters: list = None, current_date: str = None, conversation_context: str = None) -> str:
    """
    Generate prompt for planning the approach to answer a question.

    Args:
        question: Original user question
        question_analysis: Analysis from question analyzer (tickers, data_source, etc.)
        available_quarters: List of available quarters in database (e.g., ["2024_q4", "2025_q1"])
        current_date: Current date string for context
        conversation_context: Optional formatted recent conversation (for follow-up questions)

    Returns:
        Formatted planning prompt
    """
    from datetime import datetime

    tickers = question_analysis.get('extracted_tickers', [])
    data_source = question_analysis.get('data_source', 'earnings_transcripts')
    quarter_context = question_analysis.get('quarter_context', 'latest')
    quarter_count = question_analysis.get('quarter_count')
    needs_10k = question_analysis.get('needs_10k', False)
    needs_news = question_analysis.get('needs_latest_news', False)

    # Current date context
    if not current_date:
        current_date = datetime.now().strftime("%B %d, %Y")

    # Build context about what we know
    ticker_info = f"Companies: {', '.join(tickers)}" if tickers else "General market question (no specific ticker)"

    # Data sources available
    source_info = []
    if data_source == '10k' or needs_10k:
        source_info.append("10-K SEC filings (annual reports, detailed financials, risk factors)")
    if data_source == 'earnings_transcripts' or data_source == 'hybrid':
        source_info.append("earnings call transcripts (quarterly results, management commentary, analyst Q&A)")
    if needs_news or data_source == 'latest_news':
        source_info.append("latest news via web search")

    sources_text = ", ".join(source_info) if source_info else "earnings call transcripts"

    # Time period context
    time_info = ""
    if quarter_context == 'multiple' and quarter_count:
        time_info = f"User wants last {quarter_count} quarters"
    elif quarter_context == 'latest':
        time_info = "User wants most recent data"
    elif quarter_context == 'specific':
        ref = question_analysis.get('quarter_reference', '')
        time_info = f"User asked for {ref}" if ref else ""

    # Available data context
    data_availability = ""
    if available_quarters:
        formatted_quarters = []
        for q in sorted(available_quarters, reverse=True)[:6]:
            parts = q.split('_')
            if len(parts) == 2:
                formatted_quarters.append(f"{parts[1].upper()} {parts[0]}")
            else:
                formatted_quarters.append(q)
        data_availability = f"\n- Database has: {', '.join(formatted_quarters)}" + (f" (+{len(available_quarters) - 6} more)" if len(available_quarters) > 6 else "")

    conversation_section = ""
    if conversation_context and conversation_context.strip():
        conversation_section = f"""

RECENT CONVERSATION:
{conversation_context.strip()}

The current question may be a follow-up. Consider the full intent given the conversation above when planning your approach.
"""

    return f"""User question: "{question}"
{conversation_section}
Date: {current_date}
{ticker_info}
{time_info}
Data: {sources_text}{data_availability}

In 3–5 sentences (first person), explain: what the user is really asking, which metrics/quotes you need, and how you'll search. Be specific.

Example: "The user is asking about Microsoft's cloud business, so I need Azure revenue and growth rates, management commentary on competitive positioning and margins, and forward guidance. I'll search the most recent quarters for these metrics."

Output ONLY the reasoning paragraph, nothing else."""


CONTEXT_AWARE_FOLLOWUP_SYSTEM_PROMPT = """You are a financial analyst assistant generating search-optimized keyword phrases for RAG. Output only a valid JSON array of 2–3 short keyword phrases (5–10 words each). No other text, no markdown, no emojis. Phrases must be declarative (concepts/entities), not full questions."""

def get_context_aware_followup_prompt(original_question: str, current_answer: str,
                                     available_chunks: list) -> str:
    """
    Generate prompt for context-aware follow-up question generation.

    Args:
        original_question: Original user question
        current_answer: Current answer generated
        available_chunks: Available context chunks with metadata

    Returns:
        Formatted context-aware follow-up prompt
    """
    # Build context analysis
    context_analysis = ""
    if available_chunks:
        context_analysis = "\n\n"
        for i, chunk in enumerate(available_chunks[:5], 1):
            context_analysis += f"\nChunk {i}:\n"
            context_analysis += f"Text preview: {chunk.get('chunk_text', '')[:150]}...\n"
            if chunk.get('year') and chunk.get('quarter'):
                context_analysis += f"Quarter: {chunk['year']}_q{chunk['quarter']}\n"
            if chunk.get('ticker'):
                context_analysis += f"Ticker: {chunk['ticker']}\n"
            if chunk.get('distance'):
                context_analysis += f"Relevance: {chunk['distance']:.3f}\n"

    return f"""Generate 2–3 search-optimized keyword phrases to find missing or deeper information.

Question: {original_question}
Current answer: {current_answer}

Context preview:{context_analysis}

Rules:
- Declarative phrases only (no "What", "How", "Did"). Core entities, metrics, concepts. 5–10 words.
- Preserve temporal scope and tickers from the question (e.g. "last three quarters", ticker symbols).
- Think: what query would retrieve this in a vector search?

Good: "revenue growth percentage quarter comparison", "operating margins quarterly trend", "guidance each quarter"
Bad: "What specific metrics were mentioned?" (question form)

Respond ONLY with a valid JSON array, e.g. ["phrase one", "phrase two"]."""

