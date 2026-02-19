import { motion, AnimatePresence } from 'framer-motion'
import { User, ChevronDown, ChevronUp, FileText, Newspaper, Link as LinkIcon, ExternalLink, Table, Expand, Shrink, Eye } from 'lucide-react'
import StrataLensLogo from './StrataLensLogo'
import { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ChatMessage as ChatMessageType, Source } from '../lib/api'
import ReasoningTrace from './ReasoningTrace'
import type { DocumentPanelContent } from './DocumentPanel'

interface ChatMessageProps {
  message: ChatMessageType
  onOpenDocument?: (content: DocumentPanelContent) => void
}

// Citation type detection
function getCitationType(source: Source): 'transcript' | '10k' | 'news' {
  if (source.type === 'news' || source.url?.includes('http')) return 'news'
  if (source.type === '10-K' || source.section || source.fiscal_year) return '10k'
  return 'transcript'
}

// Format quarter display string from source
function formatQuarterDisplay(source: Source): string {
  // If we have both year and quarter, combine them
  if (source.year && source.quarter) {
    return `Q${source.quarter} ${source.year}`
  }
  // If quarter is already formatted (e.g., "2025_Q2" or "2025_q2")
  const qStr = source.quarter != null ? String(source.quarter) : ''
  if (qStr.includes('_')) {
    // "2025_Q2" -> "Q2 2025"
    const parts = qStr.split('_')
    if (parts.length === 2) {
      const yearPart = parts[0]
      const qPart = parts[1].toUpperCase()
      return `${qPart} ${yearPart}`
    }
  }
  // Return as-is if can't parse
  return qStr.replace('_', ' ')
}

// Get display title for citation
function getCitationTitle(source: Source, type: 'transcript' | '10k' | 'news'): string {
  if (type === 'transcript') {
    const company = source.company || source.ticker || 'Unknown Company'
    const quarter = formatQuarterDisplay(source)
    return quarter ? `${company} - ${quarter}` : company
  }
  if (type === '10k') {
    const ticker = source.ticker || 'Unknown'
    const year = source.fiscal_year ? `FY${source.fiscal_year}` : ''
    return year ? `${ticker} ${year} 10-K` : `${ticker} 10-K`
  }
  // News
  return source.title || 'News Article'
}

// Convert citation markers in text to markdown links for clickable citations
// Uses a plain-text marker as link text (no nested brackets) to avoid markdown parsing issues
function preprocessCitationMarkers(text: string, sources: Source[]): string {
  if (!sources || sources.length === 0) return text
  // Build a lookup: marker -> actual source marker (for scroll target)
  const markerMap = new Map<string, string>()
  const all10K = sources.every(s => getCitationType(s) === '10k')
  for (const s of sources) {
    if (s.marker) {
      markerMap.set(s.marker, s.marker)
      // Also map the no-hyphen variant, e.g. [10K1] -> [10K-1]
      const normalized = s.marker.replace(/-/g, '')
      if (normalized !== s.marker) markerMap.set(normalized, s.marker)
    }
  }
  // When all sources are 10-K, map plain [1], [2] to [10K-1], [10K-2] so old or alternate model output stays clickable
  if (all10K && sources.length > 0) {
    sources.forEach((s, idx) => {
      if (s.marker) {
        const num = idx + 1
        markerMap.set(`[${num}]`, s.marker)
      }
    })
  }
  // Match [1], [N1], [10K1], [10K-1], [10Q1], etc.
  return text.replace(/\[(\d+|N\d+|10[KQ]-?\d+)\]/g, (match) => {
    const actual = markerMap.get(match)
    if (actual) {
      const targetInner = actual.slice(1, -1) // strip brackets from actual marker (scroll target)
      const displayInner = match.slice(1, -1)  // keep original so [1] shows as "1", not "10K-1"
      return `[${displayInner}](#cite-${targetInner})`
    }
    return match
  })
}

// Citation badge component - enterprise style
function CitationBadge({ type }: { type: 'transcript' | '10k' | 'news' }) {
  const badges = {
    transcript: { icon: <FileText className="w-3 h-3" />, label: 'Transcript', color: 'bg-slate-100 text-slate-600' },
    '10k': { icon: <Table className="w-3 h-3" />, label: '10-K', color: 'bg-slate-100 text-slate-600' },
    news: { icon: <Newspaper className="w-3 h-3" />, label: 'News', color: 'bg-slate-100 text-slate-600' },
  }
  const badge = badges[type]

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded ${badge.color}`}>
      {badge.icon}
      {badge.label}
    </span>
  )
}

// Render transcript chunk text with bold speaker names and spacing between speakers
function TranscriptChunkText({ text }: { text: string }) {
  const speakerPattern = /^([A-Z][a-zA-Z\s.\-]+[A-Za-z]):\s*/

  type Block = { speaker?: string; lines: string[] }
  const blocks: Block[] = []
  let current: Block = { lines: [] }

  for (const line of text.split('\n')) {
    const match = line.match(speakerPattern)
    if (match && match[1].length >= 3 && match[1].length <= 60 && !/\d/.test(match[1])) {
      if (current.speaker || current.lines.some(l => l.trim())) blocks.push(current)
      current = { speaker: match[1], lines: [line.slice(match[0].length)] }
    } else {
      current.lines.push(line)
    }
  }
  blocks.push(current)

  const hasSpeakers = blocks.some(b => b.speaker)
  if (!hasSpeakers) {
    return <span className="whitespace-pre-wrap">{text}</span>
  }

  return (
    <div>
      {blocks.map((block, i) => (
        <div key={i} className={i > 0 ? 'mt-2.5' : ''}>
          {block.speaker && (
            <div className="font-bold text-slate-800 text-[11px] mb-0.5 uppercase tracking-wide">{block.speaker}:</div>
          )}
          <div className="whitespace-pre-wrap">{block.lines.join('\n')}</div>
        </div>
      ))}
    </div>
  )
}

// Individual citation card - clickable to expand
interface CitationCardProps {
  source: Source
  onViewTranscript?: (source: Source) => void
  onViewSECFiling?: (source: Source) => void
  highlighted?: boolean
}

function CitationCard({ source, onViewTranscript, onViewSECFiling, highlighted }: CitationCardProps) {
  const [expanded, setExpanded] = useState(false)
  const type = getCitationType(source)
  const title = getCitationTitle(source, type)

  // Subtitle based on type
  const subtitle = type === 'transcript'
    ? source.transcript_available ? 'Full transcript available' : ''
    : type === '10k'
      ? source.section || ''
      : source.published_date || ''

  const hasContent = source.chunk_text && source.chunk_text.length > 0
  const canViewTranscript = type === 'transcript' && source.ticker && source.quarter
  const canViewSECFiling = type === '10k' && source.ticker && source.fiscal_year

  return (
    <div
      className={`bg-white border rounded-lg overflow-hidden transition-all ${
        expanded ? 'border-slate-300' : 'border-slate-200'
      } ${highlighted ? 'ring-2 ring-blue-400 ring-offset-1' : ''}`}
    >
      <div className="p-3">
        <div className="flex items-start justify-between gap-2">
          <div
            className="flex-1 min-w-0 cursor-pointer"
            onClick={() => hasContent && setExpanded(!expanded)}
          >
            {/* Badge and marker */}
            <div className="flex items-center gap-2 mb-1.5">
              <CitationBadge type={type} />
              {source.marker && (
                <span className="text-xs font-mono text-slate-400">{source.marker}</span>
              )}
            </div>

            {/* Title - prominent */}
            <h4 className="font-medium text-[#0a1628] text-sm">{title}</h4>

            {/* Subtitle */}
            {subtitle && (
              <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 flex-shrink-0">
            {/* View Full Transcript button - only for transcripts */}
            {canViewTranscript && onViewTranscript && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onViewTranscript(source)
                }}
                className="flex items-center gap-1.5 px-2 py-1 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded transition-colors"
                title="View full transcript with highlighted sections"
              >
                <Eye className="w-3.5 h-3.5" />
                View
              </button>
            )}
            {/* View SEC Filing button - for 10-K, 10-Q, 8-K */}
            {canViewSECFiling && onViewSECFiling && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  onViewSECFiling(source)
                }}
                className="flex items-center gap-1.5 px-2 py-1 text-xs font-medium text-slate-600 bg-slate-100 hover:bg-slate-200 rounded transition-colors"
                title="View full SEC filing with highlighted sections"
              >
                <Eye className="w-3.5 h-3.5" />
                View Filing
              </button>
            )}
            {type === 'news' && source.url && (
              <a
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="p-1.5 text-slate-400 hover:text-[#0a1628] hover:bg-slate-100 rounded transition-colors"
                title="Open article"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
            {hasContent && (
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  setExpanded(!expanded)
                }}
                className="p-1.5 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded transition-colors"
                title={expanded ? 'Show less' : 'Show more'}
              >
                {expanded ? <Shrink className="w-4 h-4" /> : <Expand className="w-4 h-4" />}
              </button>
            )}
          </div>
        </div>

        {/* Preview text - only when collapsed */}
        {hasContent && !expanded && (
          <p
            className="text-xs text-slate-500 mt-2 line-clamp-2 leading-relaxed cursor-pointer"
            onClick={() => setExpanded(true)}
          >
            {source.chunk_text!.substring(0, 150)}{source.chunk_text!.length > 150 ? '...' : ''}
          </p>
        )}
      </div>

      {/* Expanded content */}
      <AnimatePresence>
        {expanded && hasContent && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-slate-100"
          >
            <div className="p-3 bg-slate-50 text-sm text-slate-700 max-h-64 overflow-y-auto leading-relaxed">
              {type === 'transcript'
                ? <TranscriptChunkText text={source.chunk_text!} />
                : <span className="whitespace-pre-wrap">{source.chunk_text}</span>
              }
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// Citations section - COLLAPSED by default
interface CitationsSectionProps {
  sources: Source[]
  onViewTranscript: (source: Source) => void
  onViewSECFiling: (source: Source) => void
  isExpanded?: boolean
  onToggleExpand?: (v: boolean) => void
  highlightedMarker?: string | null
}

function CitationsSection({ sources, onViewTranscript, onViewSECFiling, isExpanded, onToggleExpand, highlightedMarker }: CitationsSectionProps) {
  const [internalExpanded, setInternalExpanded] = useState(false)
  const expanded = isExpanded !== undefined ? isExpanded : internalExpanded
  const toggleExpanded = () => {
    const next = !expanded
    if (onToggleExpand) onToggleExpand(next)
    else setInternalExpanded(next)
  }

  // Group by type
  const transcripts = sources.filter(s => getCitationType(s) === 'transcript')
  const tenKs = sources.filter(s => getCitationType(s) === '10k')
  const news = sources.filter(s => getCitationType(s) === 'news')

  const parts = []
  if (transcripts.length > 0) parts.push(`${transcripts.length} transcript${transcripts.length > 1 ? 's' : ''}`)
  if (tenKs.length > 0) parts.push(`${tenKs.length} 10-K`)
  if (news.length > 0) parts.push(`${news.length} news`)

  return (
    <div className="mt-4 rounded-lg border border-slate-200 overflow-hidden">
      {/* Header - clickable */}
      <button
        onClick={toggleExpanded}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <LinkIcon className="w-4 h-4 text-slate-500" />
          <span className="font-medium text-[#0a1628] text-sm">
            {sources.length} source{sources.length > 1 ? 's' : ''}
          </span>
          <span className="text-sm text-slate-400 font-mono">
            ({parts.join(', ')})
          </span>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        )}
      </button>

      {/* Content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-3 space-y-4 bg-white">
              {/* Transcript sources */}
              {transcripts.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <FileText className="w-3.5 h-3.5" />
                    Document Sources ({transcripts.length})
                  </h5>
                  <div className="space-y-2">
                    {transcripts.map((source, idx) => (
                      <div key={`transcript-${idx}`} id={source.marker ? `cite-card-${source.marker.slice(1, -1)}` : undefined}>
                        <CitationCard
                          source={source}
                          onViewTranscript={onViewTranscript}
                          highlighted={!!highlightedMarker && source.marker === highlightedMarker}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 10-K sources */}
              {tenKs.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Table className="w-3.5 h-3.5" />
                    10-K SEC Filings ({tenKs.length})
                  </h5>
                  <div className="space-y-2">
                    {tenKs.map((source, idx) => (
                      <div key={`10k-${idx}`} id={source.marker ? `cite-card-${source.marker.slice(1, -1)}` : undefined}>
                        <CitationCard
                          source={source}
                          onViewSECFiling={onViewSECFiling}
                          highlighted={!!highlightedMarker && source.marker === highlightedMarker}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* News sources */}
              {news.length > 0 && (
                <div>
                  <h5 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1.5">
                    <Newspaper className="w-3.5 h-3.5" />
                    Web Sources ({news.length})
                  </h5>
                  <div className="space-y-2">
                    {news.map((source, idx) => (
                      <div key={`news-${idx}`} id={source.marker ? `cite-card-${source.marker.slice(1, -1)}` : undefined}>
                        <CitationCard source={source} highlighted={!!highlightedMarker && source.marker === highlightedMarker} />
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function ChatMessage({ message, onOpenDocument }: ChatMessageProps) {
  const [showReasoning, setShowReasoning] = useState(true)
  const [citationsExpanded, setCitationsExpanded] = useState(false)
  const [highlightedMarker, setHighlightedMarker] = useState<string | null>(null)

  const handleCitationClick = useCallback((markerInner: string) => {
    const fullMarker = `[${markerInner}]`
    setCitationsExpanded(true)
    setHighlightedMarker(fullMarker)
    setTimeout(() => {
      const el = document.getElementById(`cite-card-${markerInner}`)
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }, 300)
    setTimeout(() => setHighlightedMarker(null), 2000)
  }, [])

  const isUser = message.role === 'user'
  const hasReasoning = message.reasoning && message.reasoning.length > 0
  const hasSources = message.sources && message.sources.length > 0

  // Get all relevant chunks for highlighting in transcript
  const getRelevantChunks = (ticker: string, quarter: string) => {
    if (!message.sources) return []
    return message.sources
      .filter(s => {
        const sourceType = getCitationType(s)
        if (sourceType !== 'transcript') return false
        const matchesTicker = s.ticker === ticker || s.company === ticker
        // Ensure both quarters are strings before comparing
        const sQuarter = s.quarter != null ? String(s.quarter) : ''
        const qQuarter = quarter != null ? String(quarter) : ''
        const matchesQuarter = sQuarter === qQuarter ||
          sQuarter.replace('_', ' ') === qQuarter.replace('_', ' ')
        return matchesTicker && matchesQuarter
      })
      .map(s => ({
        chunk_text: s.chunk_text || '',
        chunk_id: s.chunk_id,
        relevance_score: s.relevance_score || 0.5
      }))
  }

  const handleViewTranscript = (source: Source) => {
    if (!onOpenDocument) return
    const qStr = String(source.quarter || '')
    let formattedQuarter: string
    // If quarter already contains a year (4 consecutive digits), use it directly
    if (/\d{4}/.test(qStr)) {
      formattedQuarter = qStr
    } else if (source.year && source.quarter) {
      formattedQuarter = `${source.year}_Q${source.quarter}`
    } else {
      formattedQuarter = qStr
    }
    onOpenDocument({
      type: 'transcript',
      company: source.company || source.ticker || '',
      ticker: source.ticker || '',
      quarter: formattedQuarter,
      relevantChunks: source.chunk_text
        ? [{ chunk_text: source.chunk_text, chunk_id: source.chunk_id, relevance_score: source.relevance_score || 0.8 }]
        : getRelevantChunks(source.ticker || '', formattedQuarter),
    })
  }

  const handleViewSECFiling = (source: Source) => {
    if (!onOpenDocument) return
    const fiscalYear = typeof source.fiscal_year === 'number'
      ? source.fiscal_year
      : parseInt(source.fiscal_year || '2023', 10)
    const quarter = typeof source.quarter === 'number'
      ? source.quarter
      : source.quarter ? parseInt(source.quarter, 10) : undefined
    onOpenDocument({
      type: 'sec-filing',
      ticker: source.ticker || '',
      filingType: source.type || '10-K',
      fiscalYear,
      quarter,
      filingDate: source.filing_date,
      relevantChunks: source.chunk_text ? [{
        chunk_text: source.chunk_text,
        chunk_id: source.chunk_id,
        sec_section: source.section,
        relevance_score: source.relevance_score || 0.8,
        char_offset: source.char_offset,
      }] : getSECFilingRelevantChunks(source.ticker || '', fiscalYear, source.type || '10-K'),
      primaryChunkId: source.chunk_id,
    })
  }

  // Get relevant chunks for SEC filing highlighting
  const getSECFilingRelevantChunks = (ticker: string, fiscalYear: string | number, filingType: string) => {
    if (!message.sources) return []
    return message.sources
      .filter(s => {
        const sourceType = getCitationType(s)
        if (sourceType !== '10k') return false
        const sourceFY = String(s.fiscal_year)
        const targetFY = String(fiscalYear)
        return s.ticker === ticker && sourceFY === targetFY && s.type === filingType
      })
      .map(s => ({
        chunk_text: s.chunk_text || '',
        chunk_id: s.chunk_id,
        sec_section: s.section,
        relevance_score: s.relevance_score || 0.8,
        char_offset: s.char_offset,
      }))
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.2 }}
        className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''}`}
      >
        {/* Avatar */}
        <div className={`flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center ${
          isUser
            ? 'bg-slate-200'
            : 'bg-[#0a1628]'
        }`}>
          {isUser ? (
            <User className="w-4 h-4 text-slate-600" />
          ) : (
            <StrataLensLogo size={16} className="text-white" />
          )}
        </div>

        {/* Message content */}
        <div className={`flex-1 min-w-0 ${isUser ? 'flex flex-col items-end' : ''}`}>
          {/* Reasoning trace (for assistant) */}
          {!isUser && hasReasoning && (
            <div className="w-full mb-3">
              <button
                onClick={() => setShowReasoning(!showReasoning)}
                className="flex items-center gap-2 text-sm text-slate-400 hover:text-[#0a1628] mb-2 transition-colors"
              >
                {showReasoning ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
                {showReasoning ? 'Hide' : 'Show'} reasoning
              </button>
              {showReasoning && (
                <ReasoningTrace steps={message.reasoning!} isStreaming={message.isStreaming} onDocumentClick={onOpenDocument} />
              )}
            </div>
          )}

          {/* Message bubble - clean enterprise style */}
          <div
            className={`
              ${isUser
                ? 'bg-slate-100 text-[#0a1628] rounded-lg px-4 py-3 max-w-[85%]'
                : 'w-full'
              }
              ${message.isStreaming && !message.content ? 'min-w-[100px]' : ''}
            `}
          >
            {message.content ? (
              isUser ? (
                <div className="whitespace-pre-wrap">{message.content}</div>
              ) : (
                <div className="prose prose-slate max-w-none
                  prose-headings:text-[#0a1628] prose-headings:font-semibold
                  prose-h1:text-xl prose-h1:mb-4 prose-h1:mt-6
                  prose-h2:text-lg prose-h2:mb-3 prose-h2:mt-5
                  prose-h3:text-base prose-h3:mb-2 prose-h3:mt-4
                  prose-p:text-slate-700 prose-p:leading-relaxed prose-p:mb-3
                  prose-strong:text-[#0a1628] prose-strong:font-semibold
                  prose-ul:my-2 prose-ol:my-2
                  prose-li:text-slate-700 prose-li:my-1
                  prose-code:text-[#0a1628] prose-code:bg-slate-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono prose-code:before:content-[''] prose-code:after:content-['']
                  prose-pre:bg-[#0a1628] prose-pre:text-slate-100 prose-pre:rounded-lg prose-pre:p-4
                  prose-blockquote:border-l-[#0a1628] prose-blockquote:bg-slate-50 prose-blockquote:py-1 prose-blockquote:px-4 prose-blockquote:rounded-r prose-blockquote:text-slate-600
                  prose-table:border-collapse prose-table:w-full
                  prose-th:bg-slate-100 prose-th:text-[#0a1628] prose-th:font-semibold prose-th:text-left prose-th:p-2 prose-th:border prose-th:border-slate-200
                  prose-td:p-2 prose-td:border prose-td:border-slate-200 prose-td:text-slate-700
                  prose-a:text-[#0a1628] prose-a:font-medium prose-a:underline prose-a:underline-offset-2
                ">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ href, children }) => {
                        if (href?.startsWith('#cite-')) {
                          const marker = href.replace('#cite-', '')
                          // Extract just the number for display (e.g. "10K-1" -> "1", "N2" -> "2", "3" -> "3")
                          const displayNum = marker.replace(/^(10[KQ]-?|N)/, '')
                          return (
                            <button
                              onClick={() => handleCitationClick(marker)}
                              className="inline-flex items-center justify-center w-5 h-5 mx-0.5 text-[10px] font-semibold rounded-full bg-slate-200 text-slate-600 hover:bg-blue-500 hover:text-white cursor-pointer transition-colors align-super -translate-y-0.5"
                              title={`Jump to source [${marker}]`}
                            >
                              {displayNum}
                            </button>
                          )
                        }
                        return <a href={href}>{children}</a>
                      }
                    }}
                  >
                    {preprocessCitationMarkers(message.content, message.sources || [])}
                  </ReactMarkdown>
                  {message.isStreaming && (
                    <span className="inline-block w-0.5 h-4 ml-1 bg-[#0a1628] animate-pulse" />
                  )}
                </div>
              )
            ) : null}
          </div>

          {/* Sources - only for assistant messages when done streaming */}
          {!isUser && hasSources && !message.isStreaming && (
            <CitationsSection
              sources={message.sources!}
              onViewTranscript={handleViewTranscript}
              onViewSECFiling={handleViewSECFiling}
              isExpanded={citationsExpanded}
              onToggleExpand={setCitationsExpanded}
              highlightedMarker={highlightedMarker}
            />
          )}
        </div>
      </motion.div>

    </>
  )
}
