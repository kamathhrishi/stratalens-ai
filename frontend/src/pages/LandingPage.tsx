import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect, useRef } from 'react'
import { SignedIn, SignedOut, UserButton } from '@clerk/clerk-react'
import StrataLensLogo from '../components/StrataLensLogo'
import AboutModal from '../components/AboutModal'
import { Check, X, Shield, Globe, Send, ArrowRight, ChevronRight, FileText, MessageSquare, Sparkles, BookOpen, Clock } from 'lucide-react'

// Mock data for 10-K Analysis Chat - Minimal card style with data extraction
const filingChatData = [
  {
    question: "What are $NVDA's key risk factors?",
    answer: "NVIDIA's main risks include customer concentration and supply chain dependencies.",
    dataPoints: [
      { label: "Top 10 customers", value: "52% of revenue" },
      { label: "Taiwan manufacturing", value: "100% of GPUs" },
    ],
    source: "10-K FY2024 • Item 1A"
  },
  {
    question: "What is $AAPL's services revenue?",
    answer: "Apple Services continues strong growth, now the second-largest segment.",
    dataPoints: [
      { label: "Services Revenue", value: "$85.2B" },
      { label: "YoY Growth", value: "+14%" },
    ],
    source: "10-K FY2024 • Item 7"
  },
  {
    question: "How did $AMD margins change?",
    answer: "AMD's data center margins expanded significantly with MI300 ramp.",
    dataPoints: [
      { label: "Data Center Margin", value: "52.4%" },
      { label: "vs Prior Year", value: "+6.2 pts" },
    ],
    source: "10-K FY2024 • Item 8"
  },
]

// Mock data for AI Chat - Full conversation with reasoning
const chatData = [
  {
    question: "What is $NVDA's revenue breakdown by segment?",
    thinking: "Searching 10-K filings for segment data...",
    answer: "NVIDIA's FY2024 revenue: **Data Center** $47.5B (78%), **Gaming** $10.4B (17%), **Pro Viz** $1.6B (3%), **Auto** $1.1B (2%).",
    sources: ["10-K Item 7", "Income Statement"]
  },
  {
    question: "Compare $MSFT and $GOOGL cloud growth",
    thinking: "Analyzing both companies' cloud segments...",
    answer: "**Azure** grew 29% to $96.8B. **Google Cloud** grew 26% to $33.1B. Microsoft leads in enterprise AI; Google in infrastructure.",
    sources: ["MSFT 10-K", "GOOGL 10-K"]
  },
  {
    question: "What did $META say about AI capex?",
    thinking: "Searching earnings transcripts for AI commentary...",
    answer: "Zuckerberg: *\"AI is our biggest investment.\"* Meta guides **$37-40B capex** in 2025 for AI infrastructure.",
    sources: ["Q4 Earnings Call", "10-K Guidance"]
  },
]

// Mock data for Transcript Chat - Quote-focused style
const transcriptChatData = [
  {
    question: "What did Jensen Huang say about AI demand?",
    ticker: "NVDA",
    quarter: "Q4 2024",
    speaker: "Jensen Huang",
    role: "CEO",
    quote: "Demand for accelerated computing and generative AI has surged. Data centers are racing to modernize the entire computing stack.",
    timestamp: "08:45"
  },
  {
    question: "Satya's comments on AI adoption?",
    ticker: "MSFT",
    quarter: "Q2 2025",
    speaker: "Satya Nadella",
    role: "CEO",
    quote: "Copilot is now used by over 70% of Fortune 500 companies. Azure AI services have become a $5 billion annual run rate business.",
    timestamp: "24:30"
  },
  {
    question: "Lisa Su on data center momentum?",
    ticker: "AMD",
    quarter: "Q4 2024",
    speaker: "Lisa Su",
    role: "CEO",
    quote: "Our data center GPU revenue grew more than 100% sequentially. MI300X is seeing exceptional demand from cloud and enterprise customers.",
    timestamp: "15:20"
  },
]

const exampleQueries = [
  "What is $NVDA's competitive moat in AI chips?",
  "Compare $MSFT and $GOOGL cloud segment growth",
  "$META AI capex commentary in last 3 quarters",
  "Comment on $ORCL balance sheet and their usage of debt?",
]

// Tech company tickers for the scrolling banner
const techTickers = [
  'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD', 'AVGO', 'CRM',
  'ADBE', 'INTC', 'QCOM', 'NFLX', 'CSCO', 'ORCL', 'NOW', 'UBER', 'ABNB', 'COIN',
  'SQ', 'PLTR', 'SNOW', 'CRWD', 'PANW', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL'
]

const AUTH_DISABLED = import.meta.env.VITE_AUTH_DISABLED === 'true'

export default function LandingPage() {
  const navigate = useNavigate()
  const [filingChatIndex, setFilingChatIndex] = useState(0)
  const [chatIndex, setChatIndex] = useState(0)
  const [transcriptChatIndex, setTranscriptChatIndex] = useState(0)
  const [inputValue, setInputValue] = useState('')
  const [aboutOpen, setAboutOpen] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const interval = setInterval(() => setFilingChatIndex(i => (i + 1) % filingChatData.length), 4000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => setChatIndex(i => (i + 1) % chatData.length), 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const interval = setInterval(() => setTranscriptChatIndex(i => (i + 1) % transcriptChatData.length), 4500)
    return () => clearInterval(interval)
  }, [])

  const handleSubmit = (query: string) => {
    if (!query.trim()) return
    navigate(`/chat?q=${encodeURIComponent(query)}`)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(inputValue)
    }
  }

  return (
    <div className="min-h-screen bg-white font-sans antialiased">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 h-16 bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 h-full flex items-center justify-between">
          <a href="/" className="flex items-center gap-2.5 group">
            <div className="w-9 h-9 bg-[#0a1628] rounded-lg flex items-center justify-center">
              <StrataLensLogo size={17} className="text-white" />
            </div>
            <span className="text-lg font-semibold text-[#0a1628] tracking-tight">StrataLens</span>
          </a>
          <div className="hidden md:flex items-center gap-6">
            <a href="#features" className="text-slate-500 text-sm font-medium hover:text-[#0a1628] transition-colors px-2 py-1">Features</a>
            <a href="#why" className="text-slate-500 text-sm font-medium hover:text-[#0a1628] transition-colors px-2 py-1">Why StrataLens</a>
            <button
              onClick={() => setAboutOpen(true)}
              className="text-slate-500 text-sm font-medium hover:text-[#0a1628] transition-colors px-2 py-1"
            >
              About
            </button>
            {AUTH_DISABLED ? (
              <button
                onClick={() => navigate('/chat')}
                className="px-5 py-2.5 bg-[#0a1628] text-white text-sm font-medium rounded-lg hover:bg-[#1e293b] transition-colors"
              >
                Open Platform
              </button>
            ) : (
              <>
                <SignedOut>
                  <button
                    onClick={() => navigate('/sign-in')}
                    className="text-slate-500 text-sm font-medium hover:text-[#0a1628] transition-colors px-2 py-1"
                  >
                    Sign In
                  </button>
                  <button
                    onClick={() => navigate('/sign-up')}
                    className="px-5 py-2.5 bg-[#0a1628] text-white text-sm font-medium rounded-lg hover:bg-[#1e293b] transition-colors"
                  >
                    Get Started
                  </button>
                </SignedOut>
                <SignedIn>
                  <button
                    onClick={() => navigate('/chat')}
                    className="px-5 py-2.5 bg-[#0a1628] text-white text-sm font-medium rounded-lg hover:bg-[#1e293b] transition-colors"
                  >
                    Open Platform
                  </button>
                  <UserButton afterSignOutUrl="/" />
                </SignedIn>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-16 min-h-screen flex items-center relative overflow-hidden">
        {/* Background - subtle warm tone */}
        <div className="absolute inset-0 bg-[#faf9f7]" />

        {/* Subtle grid pattern */}
        <div className="absolute inset-0 opacity-[0.4]" style={{
          backgroundImage: `linear-gradient(to right, #e2e8f0 1px, transparent 1px), linear-gradient(to bottom, #e2e8f0 1px, transparent 1px)`,
          backgroundSize: '64px 64px'
        }} />

        <div className="max-w-6xl mx-auto px-6 py-20 relative z-10 w-full">
          <div className="max-w-3xl mx-auto text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            >
              {/* Enterprise Label - subtle, no animation */}
              <span className="inline-block text-xs font-medium uppercase tracking-[0.2em] text-slate-400 mb-6">
                Market Intelligence Platform
              </span>

              {/* Headline - Serif for authority */}
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-semibold text-[#0a1628] mb-6 leading-[1.15]" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
                Institutional-Grade{' '}
                <span className="block">Research Intelligence</span>
              </h1>

              {/* Subheadline */}
              <p className="text-lg text-slate-500 leading-relaxed mb-10 max-w-xl mx-auto">
                Transform SEC filings and earnings transcripts into actionable insights.
                Built for analysts who demand accuracy and speed.
              </p>

              {/* Stats row - more muted */}
              <div className="flex items-center justify-center gap-10 mb-8">
                <div className="text-center">
                  <div className="text-2xl font-semibold text-[#0a1628]">500+</div>
                  <div className="text-sm text-slate-400">Tech Companies</div>
                </div>
                <div className="w-px h-12 bg-slate-200" />
                <div className="text-center">
                  <div className="text-2xl font-semibold text-[#0a1628]">3 Years</div>
                  <div className="text-sm text-slate-400">Earnings Data</div>
                </div>
                <div className="w-px h-12 bg-slate-200" />
                <div className="text-center">
                  <div className="text-2xl font-semibold text-[#0a1628]">10-K</div>
                  <div className="text-sm text-slate-400">SEC Filings</div>
                </div>
              </div>

              {/* Coming Soon - subtle */}
              <div className="flex items-center justify-center gap-2 mb-10 text-xs text-slate-400">
                <span className="font-medium">Expanding to:</span>
                <span className="px-2 py-0.5 bg-white border border-slate-200 rounded">8-K</span>
                <span className="px-2 py-0.5 bg-white border border-slate-200 rounded">10-Q</span>
                <span className="px-2 py-0.5 bg-white border border-slate-200 rounded">Private Companies</span>
              </div>
            </motion.div>

            {/* Search Input */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="max-w-2xl mx-auto mb-6"
            >
              <div className="relative bg-white rounded-lg border border-slate-300 shadow-sm hover:border-slate-400 transition-all duration-200 focus-within:border-[#0a1628] focus-within:ring-1 focus-within:ring-[#0a1628]">
                <textarea
                  ref={inputRef}
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Query SEC filings and earnings transcripts..."
                  className="w-full px-5 py-4 pr-14 text-base text-[#0a1628] placeholder:text-slate-400 bg-transparent resize-none focus:outline-none min-h-[56px] max-h-[120px]"
                  rows={1}
                />
                <button
                  onClick={() => handleSubmit(inputValue)}
                  disabled={!inputValue.trim()}
                  className="absolute right-3 bottom-3 w-10 h-10 bg-[#0a1628] text-white rounded-lg flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[#1e293b] transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </motion.div>

            {/* Example Queries */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.4 }}
              className="max-w-2xl mx-auto"
            >
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-3">Example Queries</p>
              <div className="flex flex-col gap-2">
                {exampleQueries.map((query, i) => (
                  <button
                    key={i}
                    onClick={() => handleSubmit(query)}
                    className="flex items-center justify-between px-4 py-3 bg-white border border-slate-200 rounded-lg text-left text-slate-600 hover:border-slate-300 hover:bg-slate-50 transition-all group"
                  >
                    <span className="text-sm">{query}</span>
                    <ChevronRight className="w-4 h-4 text-slate-300 group-hover:text-slate-500 group-hover:translate-x-0.5 transition-all" />
                  </button>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Coverage Banner - Enterprise style */}
      <div className="py-10 bg-white border-y border-slate-200 relative overflow-hidden">
        {/* Fade edges */}
        <div className="absolute left-0 top-0 bottom-0 w-40 bg-gradient-to-r from-white to-transparent z-10" />
        <div className="absolute right-0 top-0 bottom-0 w-40 bg-gradient-to-l from-white to-transparent z-10" />

        {/* Label */}
        <p className="text-center text-xs font-medium text-slate-400 uppercase tracking-[0.15em] mb-5">
          Coverage includes
        </p>

        {/* Tickers - more muted */}
        <div className="flex animate-scroll-right gap-16">
          {[...techTickers, ...techTickers].map((ticker, i) => (
            <span key={i} className="text-lg font-medium text-slate-300 whitespace-nowrap tracking-wide">
              ${ticker}
            </span>
          ))}
        </div>
      </div>

      {/* Features Section */}
      <section id="features" className="py-24 bg-[#faf9f7] relative">
        <div className="max-w-6xl mx-auto px-6">
          {/* Section Header */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <span className="inline-block text-xs font-medium uppercase tracking-[0.2em] text-slate-400 mb-4">
              Capabilities
            </span>
            <h2 className="text-3xl md:text-4xl font-semibold text-[#0a1628] mb-4" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
              Comprehensive Research Suite
            </h2>
            <p className="text-lg text-slate-500 max-w-xl mx-auto">
              Purpose-built tools for institutional-quality analysis of SEC filings, earnings calls, and market data.
            </p>
          </motion.div>

          {/* Feature 01 - 10-K Analysis */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-28"
          >
            <div>
              <span className="inline-block text-xs font-medium uppercase tracking-[0.15em] text-slate-400 mb-3">
                SEC Filings
              </span>
              <h3 className="text-2xl font-semibold text-[#0a1628] mb-4" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
                10-K Filing Analysis
              </h3>
              <p className="text-slate-500 leading-relaxed mb-6">
                Instantly analyze annual reports from NVIDIA, Apple, Microsoft, AMD, and 500+ tech companies. Risk factors, revenue breakdowns, competitive positioning—all extracted and structured.
              </p>
              <ul className="space-y-3">
                {[
                  "Automated financial metric extraction",
                  "Semiconductor, software, fintech coverage",
                  "Risk factors and MD&A analysis"
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="w-5 h-5 bg-slate-100 rounded flex items-center justify-center">
                      <Check className="w-3 h-3 text-[#0a1628]" />
                    </div>
                    <span className="text-slate-600 text-sm">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            {/* 10-K Chat Mock */}
            <div className="relative">
              <div className="bg-white rounded-xl border border-slate-200 shadow-lg overflow-hidden">
                {/* Question header */}
                <AnimatePresence mode="wait">
                  <motion.div
                    key={filingChatIndex}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="px-5 py-4 bg-slate-50 border-b border-slate-200"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 bg-[#0a1628] rounded-lg flex items-center justify-center">
                        <MessageSquare className="w-4 h-4 text-white" />
                      </div>
                      <p className="text-sm font-medium text-[#0a1628]">{filingChatData[filingChatIndex].question}</p>
                    </div>
                  </motion.div>
                </AnimatePresence>

                {/* Answer content */}
                <div className="p-5">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={filingChatIndex}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                    >
                      <p className="text-sm text-slate-600 mb-4">{filingChatData[filingChatIndex].answer}</p>

                      {/* Data point cards */}
                      <div className="grid grid-cols-2 gap-3 mb-4">
                        {filingChatData[filingChatIndex].dataPoints.map((point, i) => (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: i * 0.1 }}
                            className="bg-slate-50 rounded-lg p-4 border border-slate-200"
                          >
                            <p className="text-xs text-slate-400 mb-1">{point.label}</p>
                            <p className="text-lg font-semibold text-[#0a1628]" style={{ fontFamily: "'JetBrains Mono', monospace" }}>{point.value}</p>
                          </motion.div>
                        ))}
                      </div>

                      {/* Source tag */}
                      <div className="flex items-center gap-2">
                        <FileText className="w-3.5 h-3.5 text-slate-400" />
                        <span className="text-xs text-slate-500 font-mono">{filingChatData[filingChatIndex].source}</span>
                      </div>
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Feature 02 - AI Chat */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center mb-28"
          >
            {/* AI Chat Mock - Clean conversation style */}
            <div className="order-2 lg:order-1 relative">
              <div className="bg-white rounded-xl border border-slate-200 shadow-lg overflow-hidden">
                {/* Chat header */}
                <div className="flex items-center gap-3 px-5 py-4 bg-slate-50 border-b border-slate-200">
                  <div className="w-8 h-8 bg-[#0a1628] rounded-lg flex items-center justify-center">
                    <Sparkles className="w-4 h-4 text-white" />
                  </div>
                  <span className="font-semibold text-[#0a1628]">Research Assistant</span>
                  <div className="ml-auto flex items-center gap-1.5">
                    <span className="w-2 h-2 bg-emerald-500 rounded-full" />
                    <span className="text-xs text-slate-500">Active</span>
                  </div>
                </div>

                <div className="p-5 min-h-[260px]">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={chatIndex}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="space-y-4"
                    >
                      {/* User message */}
                      <div className="flex justify-end">
                        <div className="bg-slate-100 text-[#0a1628] px-4 py-2.5 rounded-lg text-sm max-w-[85%]">
                          {chatData[chatIndex].question}
                        </div>
                      </div>

                      {/* Thinking indicator - no bounce */}
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.2 }}
                        className="flex items-center gap-2 text-xs text-slate-400"
                      >
                        <span className="italic">{chatData[chatIndex].thinking}</span>
                      </motion.div>

                      {/* Assistant response */}
                      <motion.div
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="flex justify-start"
                      >
                        <div className="bg-white border-l-2 border-[#0a1628] text-slate-700 px-4 py-3 text-sm max-w-[95%] leading-relaxed">
                          <p dangerouslySetInnerHTML={{ __html: chatData[chatIndex].answer.replace(/\*\*(.*?)\*\*/g, '<strong class="text-[#0a1628]">$1</strong>').replace(/\*(.*?)\*/g, '<em>$1</em>') }} />

                          {/* Source chips */}
                          <div className="flex gap-2 mt-3 pt-3 border-t border-slate-100">
                            {chatData[chatIndex].sources.map((source, i) => (
                              <span key={i} className="text-xs bg-slate-100 text-slate-600 px-2.5 py-1 rounded flex items-center gap-1 font-mono">
                                <BookOpen className="w-3 h-3" />
                                {source}
                              </span>
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
            </div>
            <div className="order-1 lg:order-2">
              <span className="inline-block text-xs font-medium uppercase tracking-[0.15em] text-slate-400 mb-3">
                Natural Language
              </span>
              <h3 className="text-2xl font-semibold text-[#0a1628] mb-4" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
                Query in Plain English
              </h3>
              <p className="text-slate-500 leading-relaxed mb-6">
                Compare cloud growth across $MSFT, $GOOGL, and $AMZN. Analyze semiconductor supply chains. Understand fintech unit economics. All in natural language.
              </p>
              <ul className="space-y-3">
                {[
                  "Cross-company competitive analysis",
                  "Every insight sourced and verifiable",
                  "Follow-up questions supported"
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="w-5 h-5 bg-slate-100 rounded flex items-center justify-center">
                      <Check className="w-3 h-3 text-[#0a1628]" />
                    </div>
                    <span className="text-slate-600 text-sm">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          </motion.div>

          {/* Feature 03 - Earnings Transcripts */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center"
          >
            <div>
              <span className="inline-block text-xs font-medium uppercase tracking-[0.15em] text-slate-400 mb-3">
                Earnings Calls
              </span>
              <h3 className="text-2xl font-semibold text-[#0a1628] mb-4" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
                Executive Commentary Search
              </h3>
              <p className="text-slate-500 leading-relaxed mb-6">
                Find what Jensen Huang said about AI demand. What Lisa Su said about data center momentum. What Satya Nadella said about Copilot adoption. Direct quotes, fully searchable.
              </p>
              <ul className="space-y-3">
                {[
                  "Executive commentary and guidance",
                  "Analyst Q&A insights",
                  "3 years of earnings call history"
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="w-5 h-5 bg-slate-100 rounded flex items-center justify-center">
                      <Check className="w-3 h-3 text-[#0a1628]" />
                    </div>
                    <span className="text-slate-600 text-sm">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            {/* Transcript Chat Mock - Clean Quote Style */}
            <div className="relative">
              <div className="bg-white rounded-xl border border-slate-200 shadow-lg overflow-hidden">
                {/* Question */}
                <div className="px-5 py-4 bg-slate-50 border-b border-slate-200">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={transcriptChatIndex}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex items-center gap-3"
                    >
                      <div className="w-8 h-8 bg-[#0a1628] rounded-lg flex items-center justify-center">
                        <MessageSquare className="w-4 h-4 text-white" />
                      </div>
                      <p className="text-sm font-medium text-[#0a1628]">{transcriptChatData[transcriptChatIndex].question}</p>
                    </motion.div>
                  </AnimatePresence>
                </div>

                {/* Quote response */}
                <div className="p-5">
                  <AnimatePresence mode="wait">
                    <motion.div
                      key={transcriptChatIndex}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                    >
                      {/* Speaker attribution */}
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 bg-slate-200 rounded-lg flex items-center justify-center text-[#0a1628] font-semibold text-sm">
                          {transcriptChatData[transcriptChatIndex].speaker.split(' ').map(n => n[0]).join('')}
                        </div>
                        <div>
                          <p className="font-semibold text-[#0a1628]">{transcriptChatData[transcriptChatIndex].speaker}</p>
                          <p className="text-xs text-slate-500">{transcriptChatData[transcriptChatIndex].role} · ${transcriptChatData[transcriptChatIndex].ticker}</p>
                        </div>
                        <div className="ml-auto text-right">
                          <p className="text-xs font-mono text-slate-500 bg-slate-100 px-2 py-1 rounded">{transcriptChatData[transcriptChatIndex].quarter}</p>
                        </div>
                      </div>

                      {/* Quote block */}
                      <div className="relative pl-4 border-l-2 border-[#0a1628] bg-slate-50 py-3 pr-4 rounded-r">
                        <p className="text-sm text-slate-700 leading-relaxed">
                          "{transcriptChatData[transcriptChatIndex].quote}"
                        </p>
                      </div>

                      {/* Timestamp */}
                      <div className="flex items-center gap-2 mt-4 text-xs text-slate-400 font-mono">
                        <Clock className="w-3.5 h-3.5" />
                        <span>{transcriptChatData[transcriptChatIndex].timestamp}</span>
                      </div>
                    </motion.div>
                  </AnimatePresence>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Why StrataLens Section */}
      <section id="why" className="py-24 bg-white relative">
        <div className="max-w-5xl mx-auto px-6 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="inline-block text-xs font-medium uppercase tracking-[0.2em] text-slate-400 mb-4">
              Differentiation
            </span>
            <h2 className="text-3xl md:text-4xl font-semibold text-[#0a1628] mb-4" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
              Primary Source Intelligence
            </h2>
            <p className="text-lg text-slate-500">
              Built for analysts who demand accuracy and auditability
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* StrataLens Card */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-white rounded-xl p-8 border border-slate-200 shadow-sm"
            >
              <div className="flex items-center mb-8">
                <div className="w-12 h-12 bg-[#0a1628] rounded-lg flex items-center justify-center mr-4">
                  <Shield className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-[#0a1628]">StrataLens</h3>
                  <p className="text-sm text-slate-500">Primary Sources</p>
                </div>
              </div>
              <div className="space-y-4">
                {[
                  { title: "Official SEC Filings", desc: "10-K reports directly from the SEC" },
                  { title: "Earnings Transcripts", desc: "Word-for-word executive commentary" },
                  { title: "Tech Sector Focus", desc: "500+ companies: semis, software, fintech" },
                  { title: "Verifiable & Auditable", desc: "Citation-backed, traceable insights" },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="w-5 h-5 bg-slate-100 rounded flex items-center justify-center flex-shrink-0 mt-0.5">
                      <Check className="w-3 h-3 text-[#0a1628]" />
                    </div>
                    <div>
                      <p className="font-medium text-[#0a1628] text-sm">{item.title}</p>
                      <p className="text-sm text-slate-500">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* LLM + Web Search Card */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="bg-slate-50 rounded-xl p-8 border border-slate-200"
            >
              <div className="flex items-center mb-8">
                <div className="w-12 h-12 bg-slate-300 rounded-lg flex items-center justify-center mr-4">
                  <Globe className="w-6 h-6 text-slate-500" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-slate-400">Generic LLM</h3>
                  <p className="text-sm text-slate-400">Secondary Sources</p>
                </div>
              </div>
              <div className="space-y-4">
                {[
                  { title: "News Articles", desc: "Second-hand interpretations" },
                  { title: "Web Content", desc: "Unverified, potentially outdated" },
                  { title: "No Sector Focus", desc: "Generic coverage across industries" },
                  { title: "No Verification", desc: "Cannot trace to original sources" },
                ].map((item, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <div className="w-5 h-5 bg-slate-200 rounded flex items-center justify-center flex-shrink-0 mt-0.5">
                      <X className="w-3 h-3 text-slate-400" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-400 text-sm">{item.title}</p>
                      <p className="text-sm text-slate-400">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-[#0a1628] relative overflow-hidden">
        <div className="absolute inset-0 opacity-5" style={{
          backgroundImage: `linear-gradient(to right, #fff 1px, transparent 1px), linear-gradient(to bottom, #fff 1px, transparent 1px)`,
          backgroundSize: '48px 48px'
        }} />
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="max-w-2xl mx-auto px-6 text-center relative z-10"
        >
          <h2 className="text-3xl md:text-4xl font-semibold text-white mb-6" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
            Accelerate Your Research Workflow
          </h2>
          <p className="text-lg text-slate-400 mb-10">
            Transform how you analyze SEC filings and earnings transcripts. Built for institutional-quality research.
          </p>
          <button
            onClick={() => navigate('/chat')}
            className="inline-flex items-center gap-3 px-8 py-4 bg-white text-[#0a1628] text-base font-medium rounded-lg hover:bg-slate-100 transition-colors"
          >
            Start Researching
            <ArrowRight className="w-5 h-5" />
          </button>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-6 bg-[#0a1628] border-t border-slate-800">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-slate-800 rounded-lg flex items-center justify-center">
                <StrataLensLogo size={14} className="text-white" />
              </div>
              <span className="text-base font-medium text-white">StrataLens</span>
            </div>
            <p className="text-sm text-slate-500">
              Institutional-grade market intelligence platform
            </p>
          </div>
        </div>
      </footer>

      {/* About Modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />

      {/* Ticker scroll animation */}
      <style>{`
        @keyframes scroll-right {
          0% { transform: translateX(0); }
          100% { transform: translateX(-50%); }
        }
        .animate-scroll-right {
          animation: scroll-right 60s linear infinite;
        }
      `}</style>
    </div>
  )
}
