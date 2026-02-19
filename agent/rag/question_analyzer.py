#!/usr/bin/env python3
"""
Question Analyzer for the RAG system.

This module handles question analysis, validation, and preprocessing for the RAG system.
It uses AI models to analyze questions and extract relevant information like tickers,
quarter references, and question types.
"""

import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
import openai

# Import local modules
from .config import Config
from .conversation_memory import ConversationMemory
from agent.llm import get_llm, LLMClient
from .rag_utils import parse_json_with_repair
from agent.prompts import (
    TICKER_REPHRASING_SYSTEM_PROMPT,
    get_ticker_rephrasing_prompt
)

# Import Logfire for observability (optional)
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

# Configure logging
logger = logging.getLogger(__name__)
rag_logger = logging.getLogger('rag_system')


# ═════════════════════════════════════════════════════════════════════
# STAGE 1: INITIALIZATION & SETUP
# ═════════════════════════════════════════════════════════════════════

class QuestionAnalyzer:
    """Analyzes and preprocesses questions for the RAG system."""
    
    def __init__(self, openai_api_key: Optional[str] = None, config: Config = None, database_manager = None, llm: Optional[LLMClient] = None):
        """Initialize the question analyzer. Uses shared LLM from config if llm not provided."""
        self.config = config or Config()
        self.database_manager = database_manager
        self.llm = llm if llm is not None else get_llm(self.config, openai_api_key=openai_api_key)
        self.conversation_memory = ConversationMemory()
        logger.info(f"✅ QuestionAnalyzer initialized with LLM ({self.llm.provider_name})")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 2: QUESTION ANALYSIS (LLM-based Analysis)
    # ═════════════════════════════════════════════════════════════════════

    def _build_fallback_from_raw_json(self, raw_json: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Build a fallback analysis response from partially-parsed raw JSON."""
        return {
            "is_valid": raw_json.get('is_valid', True),
            "reason": raw_json.get('reason', 'Question analysis completed'),
            "question_type": raw_json.get('question_type', 'multiple_companies'),
            "extracted_ticker": raw_json.get('extracted_ticker'),
            "extracted_tickers": raw_json.get('extracted_tickers', []),
            "topic": raw_json.get('topic', ''),
            "time_refs": raw_json.get('time_refs', []),
            "suggested_improvements": raw_json.get('suggested_improvements', []),
            "confidence": raw_json.get('confidence', 0.5),
            "user_hints": raw_json.get('user_hints', {}),
            "original_question": question
        }

    @staticmethod
    def _build_error_fallback(question: str) -> Dict[str, Any]:
        """Build a generic error fallback when no JSON could be parsed at all."""
        return {
            "is_valid": False,
            "reason": "Unable to analyze your question. Please try rephrasing it.",
            "question_type": "invalid",
            "extracted_ticker": None,
            "extracted_tickers": [],
            "topic": "",
            "time_refs": [],
            "suggested_improvements": ["Try asking about a specific company like $AAPL or $MSFT"],
            "confidence": 0.0,
            "user_hints": {},
            "original_question": question
        }


    async def analyze_question(self, question: str, conversation_id: str = None, db_connection = None) -> Dict[str, Any]:
        """Analyze a question and determine the appropriate data source (earnings transcripts, 10-K filings, or news)."""
        rag_logger.info(f"🔍 Starting question analysis for: '{question}'")

        # Retry configuration - increased for better JSON parsing reliability
        max_retries = 8
        base_delay = 0.5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get conversation context if conversation_id provided
                conversation_context = ""
                has_conversation_context = False
                if conversation_id:
                    # Get conversation context using the conversation memory system
                    conversation_context = await self.conversation_memory.format_context(conversation_id)
                    
                    if conversation_context:
                        has_conversation_context = True
                        rag_logger.info(f"📚 Conversation context retrieved ({len(conversation_context)} chars)")
                        rag_logger.info(f"📚 Context preview: {conversation_context[:200]}...")
                        conversation_context = f"""

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION HISTORY (READ THIS - THEN DECIDE WHETHER TO USE IT OR IGNORE IT)
═══════════════════════════════════════════════════════════════════════════════
Below is recent dialogue. The CURRENT question might refer to it (e.g. "those companies",
"their", "compare them") - OR the user might be asking about something completely
different or a different company. You must decide:

  • USE this history when: The question clearly refers back to the conversation
    (pronouns, "those companies", "all of them", "the companies above", "same companies",
    "compare them", "their revenue", etc.). Then extract tickers FROM THIS HISTORY.

  • IGNORE this history when: The user explicitly asks about a different company
    (e.g. "What about $WMT?", "Now look at Costco", "Same for $DIS?") or asks a new
    standalone question that does not reference the prior discussion. Then use only
    tickers mentioned in the current question (e.g. $TICKER or explicit name).
═══════════════════════════════════════════════════════════════════════════════

{conversation_context}

═══════════════════════════════════════════════════════════════════════════════
END CONVERSATION HISTORY - Use it only when the question references prior context;
ignore it when the user is asking about a different company or a fresh, standalone question.
═══════════════════════════════════════════════════════════════════════════════
"""
                    else:
                        rag_logger.info(f"📚 No conversation context found for conversation_id: {conversation_id}")
                
                # Get comprehensive quarter context for LLM
                quarter_context = self.config.get_quarter_context_for_llm()
                
                # Build context-aware instructions for ticker extraction
                if has_conversation_context:
                    ticker_instructions = """**STEP 0 - WHEN CONVERSATION HISTORY IS PRESENT:**
   - READ the CONVERSATION HISTORY above. Then decide whether the CURRENT question is about that prior context or about something new.

**WHEN TO USE THE CONVERSATION HISTORY (resolve tickers from history):**
   - The question uses pronouns or back-references that clearly point to the prior discussion: "those companies", "these companies", "all companies (mentioned) above", "their", "they", "them", "it" (referring to companies), "compare them", "same companies", "the companies we discussed", "all of them", "each of them", "both companies".
   - In that case: extract EVERY ticker that appears in the conversation history (User or Assistant messages). Do NOT output extracted_tickers: [] when the question clearly refers to prior companies.

**WHEN TO IGNORE THE CONVERSATION HISTORY (do not use it for tickers):**
   - The user explicitly asks about a different company: e.g. "What about $WMT?", "Now do Costco", "Same for $DIS?", "Switch to $PG". Use only the ticker(s) the user is asking about now.
   - The question is a new, standalone question with no reference to the prior dialogue (e.g. a new $TICKER in the question, or a general question that does not say "they", "those", "their", "above", etc.). Then extract only from the current question; conversation context does not matter for ticker extraction.
   - When in doubt: if the question names a specific company (e.g. $TICKER or company name) that is different from or in addition to the conversation, prefer the current question's company/tickers. If the question is vague and refers to "them" or "those companies", use the history.

**FOLLOW-UP "THE SAME ABOUT X" / "WHAT ABOUT X" (critical for memory):**
   - When the user asks "would you say the same about $X?", "what about [company]?", "same for $Y?" they are asking to apply the **same topic or conclusion** from the previous turn to the **new** company. You must use conversation context for the TOPIC, not for tickers.
   - Tickers: set extracted_tickers to the **new** company only (e.g. ["WMT"], ["DIS"], ["PG"]) - the one they are asking about now.
   - Topic: set it to the **SAME CONCEPT/TOPIC** as the previous question or the previous Assistant answer. Read the CONVERSATION HISTORY: if the prior was about "biggest risk" or "risk factors", use "risk factors"; if the prior was about "revenue growth" or "margins", use "revenue growth and margins" or "gross margin and profitability".

**FOLLOW-UP TEMPORAL INHERITANCE (critical for continuity):**
   - When the current question does NOT explicitly mention a year or quarter, but IS clearly a follow-up to the prior turn (e.g. "now show me the income statement", "what about expenses?", "also the cash flow statement"), detect and include the time references from the conversation history.
   - Example: Prior question was "Show me AAPL's balance sheet in 2024" -> follow-up "now income statement" -> time_refs should include ["2024"].
   - Example: Prior question was "What did MSFT say about AI in Q3 2024?" -> follow-up "what about revenue?" -> time_refs should include ["Q3 2024"].
   - Only override if the current question explicitly specifies a DIFFERENT time period (e.g. "now show me 2023").

1. Extract company tickers: from $TICKER format in the current question (e.g. $AAPL -> AAPL), and from the conversation history ONLY when the question clearly references prior context (see above)."""
                else:
                    ticker_instructions = """1. Extract company tickers from $TICKER format (e.g., $AAPL -> AAPL)"""
                
                # Create analysis prompt following JSON output best practices
                analysis_prompt = f"""Analyze this financial/business question and extract key information. Respond with valid JSON only.

QUESTION: "{question}"

{quarter_context}

{conversation_context}

INSTRUCTIONS:
{ticker_instructions}
2. Classify question type based on the nature of the question
3. Extract the main topic/subject of the question (what the user is asking about)
4. Detect temporal references (e.g., "Q4 2024", "2024", "latest quarter") but do NOT resolve them
5. Identify any explicit user requests or hints (e.g., "check 10-K", "from earnings call", "latest news")
6. Assess validity and provide suggestions if needed

REQUIRED JSON STRUCTURE:
{{
  "is_valid": true,
  "reason": "Brief explanation",
  "question_type": "specific_company|multiple_companies|general_market|financial_metrics|guidance|challenges|outlook|industry_analysis|executive_leadership|business_strategy|company_info|latest_news|invalid",
  "extracted_ticker": "TICKER or null",
  "extracted_tickers": ["TICKER1", "TICKER2"],
  "topic": "what the question is about (e.g., 'iPhone sales', 'risk factors', 'revenue growth')",
  "time_refs": ["Q4 2024", "latest"],
  "suggested_improvements": ["Suggestion 1"],
  "confidence": 0.95,
  "user_hints": {{}}
}}

**CRITICAL: DETECTING INVALID QUESTIONS**

**What we CAN answer (mark is_valid=true):**
We have data from PUBLIC COMPANY earnings call transcripts, 10-K SEC filings, and company news.
Valid questions are about:
- Public company financial performance, revenue, earnings, margins, growth
- What management said in earnings calls (guidance, strategy, commentary)
- 10-K filing data (balance sheets, risk factors, executive compensation)
- Company news and recent developments
- Industry trends discussed by public companies
- Comparing public companies

**If the question is NOT about the above, mark is_valid=false.**
This includes: gibberish, greetings, non-finance topics, things we don't have data for, or questions too vague to answer.

When marking as invalid, provide a helpful reason explaining what we CAN help with and suggest example questions.

**USER HINTS** - Intelligently determine answer complexity and data sources:

**Data Source Hints:**
- If user mentions "10-K", "annual report", "SEC filing" -> user_hints: {{"data_source": "10k"}}
- If user mentions "earnings call", "transcript", "earnings" -> user_hints: {{"data_source": "earnings_transcripts"}}
- If user mentions "latest news", "recent news", "news" -> user_hints: {{"data_source": "latest_news"}}

**Answer Mode (ALWAYS SET THIS - REQUIRED):**
You MUST determine the appropriate answer_mode for EVERY question based on its complexity and scope:

- **"direct"** - Simple, single-datapoint lookups that need just a number or brief fact:
  * "What was AAPL revenue in Q4 2024?"
  * "What is Microsoft's EPS?"
  * "What was the profit margin?"
  * Simple yes/no questions
  * Single metric queries with no analysis required

- **"standard"** - Moderate questions requiring some context and explanation:
  * "Tell me about Apple's recent performance"
  * "What did Microsoft announce?"
  * Questions asking about 1-2 related metrics
  * Straightforward comparisons between 2 companies
  * "What did the CEO say about X?"

- **"detailed"** - Complex, analytical questions requiring comprehensive research and multiple data points:
  * Questions asking to "analyze", "comment on", "explain", "evaluate", "assess"
  * Questions about financial statement analysis (balance sheet, cash flow, debt structure, capital allocation)
  * Multi-company comparisons (3+ companies)
  * Questions asking for "breakdown", "details", "comprehensive", "usage", "strategy", "rationale"
  * Questions requiring synthesis of multiple data sources
  * "How" and "Why" questions requiring deep analysis
  * Questions about trends over multiple periods
  * Industry analysis questions
  * Executive leadership/strategy questions

**deep_search:** ONLY when user EXPLICITLY requests exhaustive/thorough/deep search
  * User explicitly says: "search thoroughly", "dig deep", "exhaustive search", "find everything"
  * User explicitly asks to "search harder" or "look more carefully"
  * User asks for "all mentions" or "complete list" of something specific
  * **IMPORTANT:** Do NOT use deep_search by default - only when explicitly requested
  * **Default to "detailed" for complex questions unless user explicitly asks for deeper search**

**Examples of answer_mode:**
- "What was AAPL revenue?" -> {{"answer_mode": "direct"}}
- "Tell me about AAPL's iPhone sales" -> {{"answer_mode": "standard"}}
- "Comment on Oracle's balance sheet and debt usage" -> {{"answer_mode": "detailed"}}
- "Analyze Microsoft's cloud strategy over the past year" -> {{"answer_mode": "detailed"}}
- "Compare revenue growth for AAPL, MSFT, and GOOGL" -> {{"answer_mode": "detailed"}}
- "Search thoroughly for all mentions of AI investments" -> {{"answer_mode": "deep_search"}}
- "Dig deep and find everything about Oracle's subsidiaries" -> {{"answer_mode": "deep_search"}}

**CRITICAL:** Always include "answer_mode" in user_hints - DO NOT leave it out. If you're unsure, default to "standard".

EXAMPLES:

QUESTION: "Apple's revenue in 2024"
OUTPUT: {{"is_valid": true, "reason": "Valid financial question about Apple", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "revenue", "time_refs": ["2024"], "suggested_improvements": ["Specify exact quarters for more precise data"], "confidence": 0.9, "user_hints": {{"answer_mode": "direct"}}}}

QUESTION: "Compare Microsoft and Google cloud revenue for the last 3 years"
OUTPUT: {{"is_valid": true, "reason": "Valid comparison question", "question_type": "multiple_companies", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT", "GOOGL"], "topic": "cloud revenue", "time_refs": ["last 3 years"], "suggested_improvements": [], "confidence": 0.9, "user_hints": {{"answer_mode": "detailed"}}}}

QUESTION: "What did Apple say about iPhone sales in their latest quarter?"
OUTPUT: {{"is_valid": true, "reason": "Valid question about quarterly results", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "iPhone sales", "time_refs": ["latest quarter"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"answer_mode": "standard"}}}}

QUESTION: "How has Microsoft's revenue changed over the last 3 quarters?"
OUTPUT: {{"is_valid": true, "reason": "Valid question about revenue trends", "question_type": "specific_company", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT"], "topic": "revenue changes", "time_refs": ["last 3 quarters"], "suggested_improvements": [], "confidence": 0.9, "user_hints": {{"answer_mode": "standard"}}}}

QUESTION: "Comment on Oracle's balance sheet and their usage of debt"
OUTPUT: {{"is_valid": true, "reason": "Complex financial analysis question", "question_type": "specific_company", "extracted_ticker": "ORCL", "extracted_tickers": ["ORCL"], "topic": "balance sheet and debt analysis", "time_refs": ["latest"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"answer_mode": "detailed", "data_source": "10k"}}}}

QUESTION: "What's the latest news on NVIDIA?"
OUTPUT: {{"is_valid": true, "reason": "Question about latest news", "question_type": "latest_news", "extracted_ticker": "NVDA", "extracted_tickers": ["NVDA"], "topic": "latest news", "time_refs": ["latest"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "latest_news"}}}}

QUESTION: "Find me all latest news on nvidia"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly asking for latest news", "question_type": "latest_news", "extracted_ticker": "NVDA", "extracted_tickers": ["NVDA"], "topic": "latest news", "time_refs": ["latest"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "latest_news"}}}}

QUESTION: "What was Tim Cook's compensation in 2023? Find out from the 10k"
OUTPUT: {{"is_valid": true, "reason": "Question about executive compensation", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "CEO compensation", "time_refs": ["2023"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k"}}}}

QUESTION: "Find out Tim cooks compensation from 10k for 2023"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10k", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "executive compensation", "time_refs": ["2023"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k"}}}}

QUESTION: "find out Tim cooks compensation in 2023"
OUTPUT: {{"is_valid": true, "reason": "Question about executive compensation", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "executive compensation", "time_refs": ["2023"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{}}}}

QUESTION: "What was the CEO's salary at Apple in 2023?"
OUTPUT: {{"is_valid": true, "reason": "Question about CEO salary", "question_type": "executive_leadership", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "CEO salary", "time_refs": ["2023"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{}}}}

QUESTION: "Show me Apple's balance sheet from their annual report"
OUTPUT: {{"is_valid": true, "reason": "Question about balance sheet", "question_type": "financial_metrics", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "balance sheet", "time_refs": ["latest"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k"}}}}

QUESTION: "What are Apple's total assets from their 10-K filing?"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10-K", "question_type": "financial_metrics", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "total assets", "time_refs": ["latest"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k"}}}}

QUESTION: "Get me Microsoft's risk factors from the 10k"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10k", "question_type": "company_info", "extracted_ticker": "MSFT", "extracted_tickers": ["MSFT"], "topic": "risk factors", "time_refs": ["latest"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k"}}}}

QUESTION: "Show me Google's income statement from their 10-K"
OUTPUT: {{"is_valid": true, "reason": "Question explicitly mentions 10-K", "question_type": "financial_metrics", "extracted_ticker": "GOOGL", "extracted_tickers": ["GOOGL"], "topic": "income statement", "time_refs": ["latest"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k"}}}}

QUESTION: "Analyze $ABNB 10k from 2020"
OUTPUT: {{"is_valid": true, "reason": "Question asks for 10-K analysis for a specific year", "question_type": "specific_company", "extracted_ticker": "ABNB", "extracted_tickers": ["ABNB"], "topic": "10-K analysis", "time_refs": ["2020"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k", "answer_mode": "detailed"}}}}

QUESTION: "Compile $ABNB performance from 2020 to 2024 based on its 10k"
OUTPUT: {{"is_valid": true, "reason": "Question asks for multi-year 10-K compilation", "question_type": "specific_company", "extracted_ticker": "ABNB", "extracted_tickers": ["ABNB"], "topic": "performance compilation", "time_refs": ["2020 to 2024"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"data_source": "10k", "answer_mode": "detailed"}}}}

QUESTION: "What did Apple say about their revenue in Q4 2024?"
OUTPUT: {{"is_valid": true, "reason": "Question about quarterly earnings discussion", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "revenue discussion", "time_refs": ["Q4 2024"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{}}}}

QUESTION: "Give me a brief summary of Apple's Q4 revenue"
OUTPUT: {{"is_valid": true, "reason": "Valid question with answer mode hint", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "revenue", "time_refs": ["Q4"], "suggested_improvements": [], "confidence": 0.95, "user_hints": {{"answer_mode": "direct"}}}}

QUESTION: "wrekashfkjbhkl;ahsnhbnsjg"
OUTPUT: {{"is_valid": false, "reason": "I couldn't understand your question. I can help you analyze public company earnings calls, 10-K filings, and news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "topic": "", "time_refs": [], "suggested_improvements": ["What did $AAPL say about revenue in Q4?", "Compare $MSFT and $GOOGL cloud revenue", "What's the latest news on $NVDA?"], "confidence": 0.0, "user_hints": {{}}}}

QUESTION: "hello hi"
OUTPUT: {{"is_valid": false, "reason": "Hi! I'm a financial research assistant. I can help you analyze public company earnings calls, 10-K SEC filings, and company news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "topic": "", "time_refs": [], "suggested_improvements": ["What guidance did $TSLA provide for next quarter?", "Show me $AAPL's executive compensation from 10-K", "What are tech companies saying about AI?"], "confidence": 0.0, "user_hints": {{}}}}

QUESTION: "What's a good recipe for pasta?"
OUTPUT: {{"is_valid": false, "reason": "I can only help with public company financial data. I have access to earnings call transcripts, 10-K filings, and company news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "topic": "", "time_refs": [], "suggested_improvements": ["What did $AAPL report in their latest earnings?", "Compare profit margins across tech companies", "What risks did $META disclose in their 10-K?"], "confidence": 0.0, "user_hints": {{}}}}

QUESTION: "How do I get a home loan?"
OUTPUT: {{"is_valid": false, "reason": "I don't have data on personal finance or loans. I specialize in public company financial analysis using earnings calls, 10-K filings, and news.", "question_type": "invalid", "extracted_ticker": null, "extracted_tickers": [], "topic": "", "time_refs": [], "suggested_improvements": ["What did banks like $JPM say about lending in earnings?", "Compare $BAC and $WFC financial performance", "What's in $GS latest 10-K filing?"], "confidence": 0.0, "user_hints": {{}}}}"""

                # Add conversation context examples and rules when context is present
                if has_conversation_context:
                    analysis_prompt += """

**CONVERSATION CONTEXT - WHEN TO USE IT vs WHEN TO IGNORE IT**

USE conversation history for tickers when the question clearly refers to the prior discussion (no new company named):
- Trigger phrases: "those companies", "these companies", "all companies above", "their", "they", "them", "compare them", "same companies", "the companies we discussed", "all of them", "each of them", "both companies".
- Then: set extracted_tickers to ALL tickers from the conversation history. Do NOT output extracted_tickers: [] when the question clearly refers to prior companies.

IGNORE conversation history when the user is asking about a different company or a fresh question:
- User explicitly names a different company: e.g. "What about $WMT?", "Now do Costco", "Same for $DIS?", "Switch to $PG" -> use only the company they are asking about now (e.g. ["WMT"], ["COST"], ["DIS"], ["PG"]). Do not carry over tickers from the previous conversation.
- Standalone question with a new $TICKER or company name in the current message -> extract only from the current question. Conversation context does not matter.
- When the question could be either "about the same companies" or "about a new company": if a specific new company is named ($TICKER or name), treat it as a new request and use that ticker; if only pronouns/back-references appear, use the history.

EXAMPLES (USE history):
- QUESTION: "Compare latest quarter of all companies mentioned above" + CONVERSATION had $TGT, $WMT, $COST -> extracted_tickers: ["TGT", "WMT", "COST"]
- QUESTION: "What was their revenue?" + CONVERSATION had $NFLX -> extracted_tickers: ["NFLX"]

EXAMPLES (IGNORE history - different or new company):
- QUESTION: "What about Kroger's margins?" or "$KR margins?" + CONVERSATION had $TGT, $WMT -> extracted_tickers: ["KR"] (user switched to a different company)
- QUESTION: "How did Disney do in Q3?" + CONVERSATION had $NFLX -> extracted_tickers: ["DIS"] (user is now asking about Disney, not Netflix)

FOLLOW-UP "THE SAME ABOUT X" - use conversation for TOPIC, new company for TICKERS:
- QUESTION: "Would you say the same about $DIS?" + CONVERSATION: User asked "What is the biggest risk to $NFLX?", Assistant answered about competition/content costs -> extracted_tickers: ["DIS"], topic: "biggest risk, risk factors"
- QUESTION: "What about Costco's operating margins?" + CONVERSATION: User asked "What were Target's operating margins?" -> extracted_tickers: ["COST"], topic: "operating margins"""
                
                analysis_prompt += """

RESPOND WITH VALID JSON ONLY. NO EXPLANATIONS OR ADDITIONAL TEXT."""

                # Add retry-specific instructions for subsequent attempts
                if attempt > 0:
                    analysis_prompt += f"""

RETRY ATTEMPT {attempt + 1}: Previous response was invalid JSON.
CRITICAL: Return ONLY valid JSON matching the exact structure above.
Double-check: no trailing commas, proper quotes, all 7 required fields present.
Example: {{"is_valid": true, "reason": "Valid question", "question_type": "specific_company", "extracted_ticker": "AAPL", "extracted_tickers": ["AAPL"], "topic": "revenue growth", "time_refs": ["Q4 2024"], "suggested_improvements": [], "confidence": 0.9, "user_hints": {{}}}}"""

                # Build context-aware system message
                if has_conversation_context:
                    rag_logger.info(f"🧠 Using CONVERSATION CONTEXT MODE - use history for tickers or for topic on 'same about X' follow-ups")
                    system_message = (
                        "You are a JSON-only response assistant for financial question analysis. Respond with valid JSON only. Do not use emojis. "
                        "CONVERSATION HISTORY may be provided. Read it, then decide: "
                        "(1) USE it for tickers when the question refers to prior companies (e.g. 'those companies', 'their', 'compare them') - set extracted_tickers from history. "
                        "(2) When the user asks 'the same about $X?' or 'what about [company]?' they want the SAME TOPIC applied to the NEW company: set extracted_tickers to the new company only (e.g. ['DIS'], ['COST']), and set topic to the SAME as the previous question or answer (e.g. if prior was about biggest risk -> 'risk factors'; if prior was margins -> 'operating margins'). "
                        "(3) IGNORE history for tickers when the user asks a new standalone question with a new $TICKER - use only current question's tickers. "
                        "Extract topic as the main subject of the question. Detect time_refs (temporal references) but do NOT resolve them. "
                        "Capture user_hints for explicit requests like 'check 10-K', 'from earnings call', 'latest news', 'brief', 'detailed'. "
                        "No explanations or extra text."
                    )
                else:
                    rag_logger.info(f"🎯 Using STANDARD MODE - no conversation context available")
                    system_message = (
                        "You are a JSON-only response assistant for financial question analysis. Respond with valid JSON only. Do not use emojis. "
                        "Extract tickers from $TICKER format. Extract the main topic/subject of the question. "
                        "Detect temporal references (time_refs) but do NOT resolve them. "
                        "Capture user_hints for explicit data source requests ('10-K', 'earnings call', 'latest news') and answer mode hints ('brief', 'detailed'). "
                        "No explanations or additional text."
                    )
                
                start_time = time.time()
                model = self.config.get("cerebras_model", "qwen-3-235b-instruct-2507")
                rag_logger.info(f"🤖 Sending question to LLM ({self.llm.provider_name}) model: {model} (attempt {attempt + 1}/{max_retries})")

                if LOGFIRE_AVAILABLE and logfire:
                    with logfire.span(
                        "llm.question_analysis",
                        model=model,
                        question=question,
                        system_prompt=system_message,
                        user_prompt=analysis_prompt,
                        has_conversation_context=has_conversation_context,
                        attempt=attempt + 1,
                        max_retries=max_retries
                    ):
                        analysis_text = self.llm.complete(
                            [
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": analysis_prompt}
                            ],
                            model=model,
                            max_tokens=1000,
                            temperature=0.1,
                            stream=False,
                        )
                else:
                    analysis_text = self.llm.complete(
                        [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": analysis_prompt}
                        ],
                        model=model,
                        max_tokens=1000,
                        temperature=0.1,
                        stream=False,
                    )
                call_time = time.time() - start_time
                rag_logger.info(f"✅ Received response from LLM in {call_time:.3f}s")
                
                analysis_text = analysis_text.strip()
                rag_logger.info(f"📝 Raw Cerebras response length: {len(analysis_text)} characters")
                rag_logger.info(f"📝 Raw Cerebras response (first 500 chars): {analysis_text[:500]}")
                if len(analysis_text) == 0:
                    rag_logger.error("❌ CRITICAL: Model returned empty response!")
                    raise Exception("Model returned empty response - this indicates a model or prompt issue")
                
                # Clean up the response (remove any markdown formatting)
                if analysis_text.startswith("```json"):
                    analysis_text = analysis_text[7:]
                    rag_logger.info("🧹 Removed ```json prefix from response")
                if analysis_text.endswith("```"):
                    analysis_text = analysis_text[:-3]
                    rag_logger.info("🧹 Removed ``` suffix from response")
                
                # Try to parse JSON with repair attempts
                try:
                    # Local import to avoid circular dependency (app.routers.chat imports agent)
                    from app.schemas.rag import QuestionAnalysisResult
                    analysis_result = parse_json_with_repair(analysis_text, attempt, QuestionAnalysisResult, rag_logger)
                    rag_logger.info(f"✅ Successfully parsed JSON analysis result")
                    rag_logger.info(f"📊 Analysis result: valid={analysis_result.get('is_valid')}, ticker={analysis_result.get('extracted_ticker')}, type={analysis_result.get('question_type')}, topic={analysis_result.get('topic', '')}")

                    # Add original question
                    analysis_result["original_question"] = question
                except Exception as parse_error:
                    # If validation fails but we have the raw JSON, try to extract basic fields
                    rag_logger.warning(f"⚠️ Validation failed but attempting to extract fields from raw response")
                    import json as json_lib
                    try:
                        raw_json = json_lib.loads(analysis_text)
                        topic = raw_json.get('topic', '')
                        rag_logger.info(f"📊 Extracted topic={topic} from raw JSON")
                        # Re-raise to continue with normal error handling
                        raise parse_error
                    except:
                        raise parse_error
                
                # Log ticker extraction results for debugging (especially important for conversation context)
                extracted_tickers = analysis_result.get('extracted_tickers', [])
                topic = analysis_result.get('topic', '')
                time_refs = analysis_result.get('time_refs', [])
                question_type = analysis_result.get('question_type', '')

                if extracted_tickers:
                    rag_logger.info(f"🎯 Extracted tickers: {extracted_tickers}")
                    if conversation_context:
                        rag_logger.info(f"🎯 Tickers were extracted with conversation context available")
                else:
                    rag_logger.warning(f"⚠️ No tickers extracted from question: '{question}'")
                    if conversation_context:
                        rag_logger.warning(f"⚠️ Conversation context was available but no tickers extracted!")

                # Log temporal detection
                if time_refs:
                    rag_logger.info(f"🕒 Detected time references: {time_refs}")

                # Log topic extraction
                if topic:
                    rag_logger.info(f"📋 Extracted topic: {topic}")

                # Log question analysis to Logfire
                if LOGFIRE_AVAILABLE and logfire:
                    logfire.info(
                        "question.analysis.complete",
                        original_question=question,
                        topic=topic,
                        is_valid=analysis_result.get('is_valid', False),
                        question_type=question_type,
                        tickers=extracted_tickers,
                        time_refs=time_refs,
                        user_hints=analysis_result.get('user_hints', {}),
                        confidence=analysis_result.get('confidence', 0)
                    )

                return analysis_result
                
            except json.JSONDecodeError as e:
                rag_logger.error(f"❌ JSON parsing failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay  # Fixed delay
                    rag_logger.info(f"🔄 Retrying in {delay} seconds... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    rag_logger.error(f"💥 All {max_retries} attempts failed. Last error: {e}")
                    rag_logger.error(f"📝 Last response text: {analysis_text[:500]}...")
                    
                    # Try to extract fields from the raw response - PRESERVE is_valid from LLM!
                    rag_logger.warning("🔄 Attempting to extract response from raw JSON despite validation errors")

                    try:
                        import json as json_lib
                        raw_json = json_lib.loads(analysis_text)
                        fallback_response = self._build_fallback_from_raw_json(raw_json, question)
                        rag_logger.info(f"✅ Fallback response created from raw JSON: is_valid={fallback_response['is_valid']}, question_type={fallback_response['question_type']}")
                        return fallback_response

                    except Exception as parse_err:
                        rag_logger.error(f"❌ Could not parse raw JSON: {parse_err}")
                        return self._build_error_fallback(question)
            except Exception as e:
                rag_logger.error(f"❌ Question analysis failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay  # Fixed delay
                    rag_logger.info(f"🔄 Retrying in {delay} seconds... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    rag_logger.error(f"💥 All {max_retries} attempts failed. Last error: {e}")
                    
                    # Try to extract fields from the raw response - PRESERVE is_valid from LLM!
                    rag_logger.warning("🔄 Attempting to extract response from raw JSON despite general errors")

                    try:
                        import json as json_lib
                        raw_json = json_lib.loads(analysis_text)
                        fallback_response = self._build_fallback_from_raw_json(raw_json, question)
                        rag_logger.info(f"✅ Fallback response created from raw JSON: is_valid={fallback_response['is_valid']}, question_type={fallback_response['question_type']}")
                        return fallback_response

                    except Exception as parse_err:
                        rag_logger.error(f"❌ Could not parse raw JSON: {parse_err}")
                        return self._build_error_fallback(question)
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in analyze_question retry loop")

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 3: QUARTER DETERMINATION & RESOLUTION
    # ═════════════════════════════════════════════════════════════════════

    def determine_target_quarter(self, analysis: Dict[str, Any], ticker: str = None) -> str:
        """Determine the target quarter based on question analysis."""
        quarter_reference = analysis.get('quarter_reference')
        quarter_context = analysis.get('quarter_context', 'latest')
        quarter_count = analysis.get('quarter_count')
        
        # Resolve 'latest' quarter references first
        if quarter_reference == 'latest':
            # Use database_manager to resolve latest quarter
            if self.database_manager:
                resolved_quarter = self.database_manager.resolve_latest_quarter_reference(quarter_reference, ticker)
                if resolved_quarter != "NO_QUARTERS_AVAILABLE":
                    return resolved_quarter
            # Fallback to configured latest quarter
            latest_quarter = self.config.get_latest_quarter()
            if latest_quarter:
                return latest_quarter
            else:
                return "NO_QUARTERS_AVAILABLE"
        
        # If specific quarter is mentioned, try to match it
        if quarter_reference:
            available_quarters = self.config.get('available_quarters', [])
            
            # Direct match (should work now since LLM uses database format)
            if quarter_reference in available_quarters:
                return quarter_reference
            
            # Quarter requested but not available - return special marker for clear error
            return f"UNAVAILABLE_QUARTER:{quarter_reference}"
        
        # Handle multiple quarters (e.g., "last 3 quarters" or year-only like "2024")
        if quarter_context == 'multiple' and quarter_count:
            # Special case: quarter_count=4 usually means whole year
            if quarter_count == 4:
                return 'year_all'  # Special marker for full year queries
            return 'multiple'  # General marker for multiple quarter queries
        
        # Handle context-based quarter selection
        if quarter_context == 'previous':
            available_quarters = self.config.get('available_quarters', [])
            if len(available_quarters) >= 2:
                # If we have multiple quarters, previous would be the second-to-last
                return available_quarters[-2]
            elif len(available_quarters) == 1:
                return self.config.get_latest_quarter()
            else:
                return "NO_QUARTERS_AVAILABLE"
        
        # Handle latest quarter (explicit case)
        if quarter_context == 'latest':
            # Check if quarter_reference is specifically "latest"
            if quarter_reference == 'latest':
                available_quarters = self.config.get('available_quarters', [])
                if available_quarters:
                    return self.config.get_latest_quarter()  # Latest quarter is first in the list
                else:
                    return "NO_QUARTERS_AVAILABLE"
            else:
                available_quarters = self.config.get('available_quarters', [])
                if available_quarters:
                    return self.config.get_latest_quarter()  # Latest quarter is first in the list
                else:
                    return "NO_QUARTERS_AVAILABLE"
        
        # Default to latest quarter if available
        available_quarters = self.config.get('available_quarters', [])
        if available_quarters:
            return available_quarters[0]
        else:
            return "NO_QUARTERS_AVAILABLE"

    def add_to_conversation_memory(self, conversation_id: str, message: str, role: str = "user"):
        """Add a message to conversation memory."""
        self.conversation_memory.add_message(conversation_id, message, role)
    
    def get_quarters_to_search(self, analysis: Dict[str, Any]) -> List[str]:
        """Get list of quarters to search based on question analysis."""
        quarter_context = analysis.get('quarter_context', 'latest')
        quarter_count = analysis.get('quarter_count')
        available_quarters = self.config.get('available_quarters', [])
        
        # Debug logging
        rag_logger.info(f"🔍 get_quarters_to_search debug (instance: {getattr(self, 'instance_id', 'unknown')}):")
        rag_logger.info(f"   quarter_context: {quarter_context}")
        rag_logger.info(f"   quarter_count: {quarter_count}")
        rag_logger.info(f"   available_quarters: {available_quarters}")
        rag_logger.info(f"   available_quarters length: {len(available_quarters)}")
        
        # Handle multiple quarters (e.g., "last 3 quarters", "2024", "last 1 year")
        if quarter_context == 'multiple' and quarter_count:
            # Special case: quarter_count=4 could mean:
            # 1. "last 1 year" (last 4 quarters) - should use company-specific quarters
            # 2. Year-only query like "2024" or "2025" - all quarters in that specific year
            if quarter_count == 4:
                quarter_reference = analysis.get('quarter_reference')
                # Check if this is a year-only query (has explicit year reference like "2024_all" or "2025")
                # A specific quarter like "2024_q4" is NOT a year-only query - it's a specific quarter reference
                is_year_only = quarter_reference and (
                    '_all' in quarter_reference or  # e.g., "2024_all"
                    (quarter_reference.isdigit() and len(quarter_reference) == 4)  # e.g., "2024" or "2025"
                ) and '_q' not in quarter_reference  # Exclude specific quarters like "2024_q4"
                
                if is_year_only:
                    # Extract year from quarter_reference
                    if '_all' in quarter_reference:
                        year = quarter_reference.split('_')[0]
                    else:
                        year = quarter_reference
                    rag_logger.info(f"  🗓️ YEAR-ONLY DETECTED: Looking for all quarters in year {year}")
                    # Find all quarters for this specific year
                    year_quarters = [q for q in available_quarters if q.startswith(year + '_')]
                    year_quarters.sort(reverse=True)  # Sort newest first
                    rag_logger.info(f"   year-only quarters result: {year_quarters}")
                    return year_quarters
                else:
                    # This is "last 1 year" (last 4 quarters) - use company-specific quarters if available
                    ticker = analysis.get('extracted_ticker')
                    if ticker and self.database_manager:
                        # Get company-specific last 4 quarters from database
                        company_quarters = self.database_manager.get_last_n_quarters_for_company(ticker, 4)
                        if company_quarters:
                            rag_logger.info(f"   ✅ Company-specific last 4 quarters (1 year) for {ticker}: {company_quarters}")
                            return company_quarters
                        else:
                            rag_logger.warning(f"   ⚠️ No company-specific quarters found for {ticker}, falling back to general quarters")
                    
                    # Fallback to general available quarters (when no ticker or company-specific query failed)
                    result = self._get_last_n_quarters_business_logic(available_quarters, 4)
                    rag_logger.info(f"   general last 4 quarters result: {result}")
                    return result
            else:
                # For multiple quarters (not 4), always try to get company-specific quarters if ticker is available
                ticker = analysis.get('extracted_ticker')
                if ticker and self.database_manager:
                    # Get company-specific last N quarters from database
                    # This finds the latest quarter for this specific company and gets N quarters going back
                    company_quarters = self.database_manager.get_last_n_quarters_for_company(ticker, quarter_count)
                    if company_quarters:
                        rag_logger.info(f"   ✅ Company-specific last {quarter_count} quarters for {ticker}: {company_quarters}")
                        return company_quarters
                    else:
                        rag_logger.warning(f"   ⚠️ No company-specific quarters found for {ticker}, falling back to general quarters")
                
                # Fallback to general available quarters (when no ticker or company-specific query failed)
                # Use the actual latest quarter from available_quarters (already sorted DESC)
                result = self._get_last_n_quarters_business_logic(available_quarters, quarter_count)
                rag_logger.info(f"   general multiple quarters result: {result}")
                return result
        
        # For single quarter queries, return the determined target quarter
        # Extract ticker from analysis for company-specific latest quarter resolution
        ticker = analysis.get('extracted_ticker')
        target_quarter = self.determine_target_quarter(analysis, ticker)
        
        # Handle special error cases
        if target_quarter == 'multiple':
            # Fallback to all available quarters
            return available_quarters
        elif target_quarter.startswith('UNAVAILABLE_QUARTER:'):
            # Return empty list to trigger clear error message
            return []
        elif target_quarter == 'NO_QUARTERS_AVAILABLE':
            # Return empty list to trigger clear error message
            return []
        else:
            return [target_quarter]
    
    def _get_last_n_quarters_business_logic(self, available_quarters: List[str], n: int) -> List[str]:
        """
        Get the last N quarters from available quarters.
        
        This method assumes available_quarters is already sorted in reverse chronological order
        (year DESC, quarter DESC), so the first quarter is the latest. It simply returns
        the first N quarters from the list.
        
        Args:
            available_quarters: List of available quarters (sorted DESC, latest first)
            n: Number of quarters to return
            
        Returns:
            List of the last N quarters (latest first)
        """
        if not available_quarters:
            return []
        
        # available_quarters is already sorted in reverse chronological order (latest first)
        # So we just take the first N quarters
        result = available_quarters[:n] if len(available_quarters) >= n else available_quarters
        rag_logger.info(f"🔄 Last {n} quarters from available quarters: {result}")
        return result

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 4: QUESTION VALIDATION & PROCESSING
    # ═════════════════════════════════════════════════════════════════════

    async def process_question(self, question: str, conversation_id: str = None) -> Dict[str, Any]:
        """Complete question processing pipeline: analyze, validate, and return processed result."""
        analysis = await self.analyze_question(question, conversation_id)

        # Check if question is invalid based on AI's assessment
        is_valid = analysis.get("is_valid", False)

        # If the AI marked it as invalid, reject it
        if not is_valid:
            message = analysis.get('reason', 'I can help you analyze public company earnings calls, 10-K SEC filings, and company news.')
            suggestions = analysis.get('suggested_improvements', [
                "What did $AAPL say about revenue in Q4?",
                "Compare $MSFT and $GOOGL cloud revenue",
                "What's the latest news on $NVDA?"
            ])

            return {
                "status": "rejected",
                "message": message,
                "suggestions": suggestions,
                "question_type": analysis["question_type"],
                "original_question": question
            }

        # Extract tickers
        if analysis.get("extracted_tickers"):
            extracted_tickers = analysis["extracted_tickers"]
            extracted_ticker = analysis["extracted_tickers"][0]
        else:
            extracted_ticker = analysis.get("extracted_ticker")
            extracted_tickers = [extracted_ticker] if extracted_ticker else []

        result = {
            "status": "processed",
            "original_question": question,
            "question_type": analysis["question_type"],
            "extracted_ticker": extracted_ticker,
            "extracted_tickers": extracted_tickers,
            "topic": analysis.get("topic", ""),
            "time_refs": analysis.get("time_refs", []),
            "confidence": analysis.get("confidence", 0.8),
            "user_hints": analysis.get("user_hints", {})
        }

        return result

    # ═════════════════════════════════════════════════════════════════════
    # STAGE 5: TICKER-SPECIFIC QUESTION CREATION
    # ═════════════════════════════════════════════════════════════════════

    def create_ticker_specific_question(self, original_question: str, ticker: str) -> str:
        """Create a ticker-specific rephrased question for better search results."""
        rag_logger.info(f"🎯 Creating ticker-specific question for {ticker}")

        # Get ticker-specific prompt from centralized prompts
        ticker_prompt = get_ticker_rephrasing_prompt(original_question, ticker)

        try:
            rag_logger.info(f"🤖 ===== TICKER REPHRASING LLM CALL =====")
            rag_logger.info(f"🔍 Provider: {self.llm.provider_name}")
            rag_logger.info(f"🎯 Ticker: {ticker}")
            rag_logger.info(f"📝 Original question: {original_question}")

            start_time = time.time()
            ticker_question = self.llm.complete(
                [
                    {"role": "system", "content": TICKER_REPHRASING_SYSTEM_PROMPT},
                    {"role": "user", "content": ticker_prompt}
                ],
                max_tokens=200,
                temperature=0.3,
                stream=False,
            )
            call_time = time.time() - start_time
            
            ticker_question = ticker_question.strip()
            rag_logger.info(f"✅ ===== TICKER REPHRASING LLM RESPONSE ===== (call time: {call_time:.3f}s)")
            rag_logger.info(f"📝 Rephrased question: {ticker_question}")
            return ticker_question
            
        except Exception as e:
            rag_logger.error(f"❌ Error creating ticker-specific question: {e}")
            raise Exception(f"Failed to create ticker-specific question: {e}")
