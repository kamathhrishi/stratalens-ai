import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Download, Calendar, FileText, ZoomIn, ZoomOut, Eye, Loader2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import { config } from '../lib/config'

interface SECFilingChunk {
  chunk_text: string
  chunk_id?: string
  sec_section?: string
  relevance_score?: number
  char_offset?: number
}

interface HeadingNode {
  level: number
  text: string
  char_offset: number
  children: HeadingNode[]
}

interface TocEntry {
  title: string
  page: string
}

interface SECFilingViewerProps {
  isOpen: boolean
  onClose: () => void
  ticker: string
  filingType: string
  fiscalYear: number
  quarter?: number
  filingDate?: string
  relevantChunks?: SECFilingChunk[]
  primaryChunkId?: string  // The specific citation that was clicked - scroll to this one
  panelMode?: boolean  // When true, renders as embedded panel content (no overlay)
}

interface SECFilingData {
  success: boolean
  ticker: string
  company_name?: string
  filing_type: string
  fiscal_year: number
  filing_date?: string
  document_text: string
  document_markdown?: string
  highlighted_markdown?: string
  document_length?: number
  sections?: string[]
  section_offsets?: Record<string, { start: number; end: number }>
  document_structure?: HeadingNode[]
  table_of_contents?: TocEntry[]
  headings_map?: Record<string, number>
}

// ─── Markdown renderer for structured filings ────────────────────────────────

function MarkdownFilingViewer({
  markdown,
  zoom,
  contentRef,
}: {
  markdown: string
  zoom: number
  contentRef: React.RefObject<HTMLDivElement | null>
}) {
  return (
    <div
      ref={contentRef}
      className="bg-white shadow-lg rounded-lg p-12 sec-filing-document sec-filing-markdown"
      style={{
        fontSize: `${zoom}%`,
        fontFamily: 'Georgia, "Times New Roman", serif',
        lineHeight: 1.6,
      }}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-2xl font-bold text-slate-900 mt-10 mb-4 border-b-2 border-slate-300 pb-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl font-bold text-slate-900 mt-8 mb-3 uppercase tracking-wide">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg font-semibold text-slate-800 mt-6 mb-2">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-base font-semibold text-slate-700 mt-4 mb-2">
              {children}
            </h4>
          ),
          h5: ({ children }) => (
            <h5 className="text-sm font-semibold text-slate-700 mt-3 mb-1">
              {children}
            </h5>
          ),
          h6: ({ children }) => (
            <h6 className="text-sm font-medium text-slate-600 mt-2 mb-1">
              {children}
            </h6>
          ),
          p: ({ children }) => (
            <p className="mb-4 leading-relaxed text-slate-700 text-justify hyphens-auto">
              {children}
            </p>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto my-6 rounded-lg border border-slate-200">
              <table className="min-w-full border-collapse text-sm">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-slate-50 border-b-2 border-slate-200">
              {children}
            </thead>
          ),
          th: ({ children }) => (
            <th className="px-4 py-2 text-left font-semibold text-slate-700 whitespace-nowrap text-xs uppercase tracking-wide">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-2 border-b border-slate-100 text-slate-700">
              {children}
            </td>
          ),
          tr: ({ children }) => (
            <tr className="even:bg-slate-50/50 hover:bg-blue-50/30 transition-colors">
              {children}
            </tr>
          ),
          // Highlight marks injected by backend
          mark: ({ children, ...props }) => (
            <mark className="highlighted-chunk" {...props}>
              {children}
            </mark>
          ),
          // Sub tags from datamule (used for smaller text)
          sub: ({ children }) => (
            <span className="text-sm text-slate-600">{children}</span>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-slate-300 pl-4 my-4 text-slate-600 italic">
              {children}
            </blockquote>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-inside mb-4 space-y-1 text-slate-700">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside mb-4 space-y-1 text-slate-700">
              {children}
            </ol>
          ),
          hr: () => <hr className="my-6 border-slate-200" />,
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  )
}

// ─── Plain-text fallback renderer ────────────────────────────────────────────

function formatSECFiling(documentText: string): string {
  if (!documentText) return '<div class="text-slate-500 p-8 text-center">No document text available</div>'

  let formatted = documentText
    .split('\n\n')
    .map(para => {
      const trimmed = para.trim()
      if (!trimmed) return ''
      if (trimmed === trimmed.toUpperCase() && trimmed.length < 100 && !trimmed.includes('.')) {
        return `<h3 class="text-lg font-bold text-slate-900 mt-8 mb-4 uppercase">${trimmed}</h3>`
      }
      return `<p class="mb-4 leading-relaxed text-slate-700">${trimmed.replace(/\n/g, '<br>')}</p>`
    })
    .filter(Boolean)
    .join('\n')

  return `<div class="sec-filing-content">${formatted}</div>`
}

// ─── TOC Sidebar ─────────────────────────────────────────────────────────────

function TOCSidebar({
  toc,
  structure,
  onJumpTo,
}: {
  toc: TocEntry[]
  structure?: HeadingNode[]
  onJumpTo: (text: string) => void
}) {
  const items = toc.length > 0 ? toc : (structure || []).slice(0, 30).map(n => ({ title: n.text, page: '' }))

  if (items.length === 0) return null

  return (
    <div className="w-64 shrink-0 border-r border-slate-200 bg-slate-50 overflow-y-auto">
      <div className="p-4 border-b border-slate-200">
        <h3 className="text-sm font-semibold text-slate-700 uppercase tracking-wide">Contents</h3>
      </div>
      <nav className="p-2">
        {items.map((item, i) => (
          <button
            key={i}
            onClick={() => onJumpTo(item.title)}
            className="w-full text-left px-3 py-1.5 text-sm text-slate-600 hover:text-blue-700 hover:bg-blue-50 rounded transition-colors truncate flex items-center justify-between group"
            title={item.title}
          >
            <span className="truncate">{item.title}</span>
            {item.page && (
              <span className="text-xs text-slate-400 group-hover:text-blue-400 ml-2 shrink-0">
                {item.page}
              </span>
            )}
          </button>
        ))}
      </nav>
    </div>
  )
}

// ─── Main Viewer Component ────────────────────────────────────────────────────

export default function SECFilingViewer({
  isOpen,
  onClose,
  ticker,
  filingType,
  fiscalYear,
  quarter,
  filingDate,
  relevantChunks = [],
  primaryChunkId,
  panelMode = false,
}: SECFilingViewerProps) {
  const [filing, setFiling] = useState<SECFilingData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [zoom, setZoom] = useState(100)
  const [showTOC, setShowTOC] = useState(false)
  const contentRef = useRef<HTMLDivElement>(null)

  // Stable key derived from chunk identity — avoids refetch when parent re-renders with new array reference
  const chunkKey = relevantChunks.map(c => c.chunk_id || c.chunk_text.slice(0, 20)).join('|')

  // Fetch filing when modal opens
  useEffect(() => {
    if (!isOpen) return

    const fetchFiling = async () => {
      setLoading(true)
      setError(null)

      try {
        if (relevantChunks && relevantChunks.length > 0) {
          // Use with-highlights endpoint for rich highlighting
          const response = await fetch(`${config.apiBaseUrl}/sec-filings/with-highlights`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              ticker,
              filing_type: filingType,
              fiscal_year: fiscalYear,
              quarter,
              filing_date: filingDate,
              relevant_chunks: relevantChunks.map(chunk => ({
                chunk_text: chunk.chunk_text,
                chunk_id: chunk.chunk_id,
                sec_section: chunk.sec_section,
                relevance_score: chunk.relevance_score,
                char_offset: chunk.char_offset,
              })),
            }),
          })

          if (!response.ok) {
            const err = await response.json().catch(() => ({}))
            throw new Error(err.detail || `Filing not found`)
          }

          const data = await response.json()
          // Prefer markdown over plain text; prefer highlighted over raw
          setFiling({
            ...data,
            // Use highlighted_markdown if available, else raw markdown, else plain
            document_markdown: data.highlighted_markdown || data.document_markdown,
            document_text: data.highlighted_document || data.document_text,
          })
        } else {
          // No chunks - plain fetch
          let url = `${config.apiBaseUrl}/sec-filings/${ticker}/${filingType}/${fiscalYear}`
          const params = new URLSearchParams()
          if (filingType === '10-Q' && quarter) params.append('quarter', quarter.toString())
          if (filingType === '8-K' && filingDate) params.append('filing_date', filingDate)
          if (params.toString()) url += `?${params.toString()}`

          const response = await fetch(url)
          if (!response.ok) {
            const err = await response.json().catch(() => ({}))
            throw new Error(err.detail || `Filing not found`)
          }
          const data = await response.json()
          setFiling(data)
        }
      } catch (err) {
        console.error('Failed to fetch SEC filing:', err)
        setError(err instanceof Error ? err.message : 'Failed to load SEC filing')
      } finally {
        setLoading(false)
      }
    }

    fetchFiling()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, ticker, filingType, fiscalYear, quarter, filingDate, chunkKey])

  // Scroll to the clicked citation's mark, falling back to the first mark
  useEffect(() => {
    if (filing && contentRef.current && relevantChunks.length > 0) {
      setTimeout(() => {
        const el = contentRef.current
        if (!el) return
        // Try to find the specific mark for the clicked citation first
        const targetMark = primaryChunkId
          ? el.querySelector(`mark[data-chunk-id="${primaryChunkId}"]`) as HTMLElement | null
          : null
        const markToScrollTo = targetMark || el.querySelector('mark.highlighted-chunk') as HTMLElement | null
        if (markToScrollTo) {
          markToScrollTo.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
      }, 600)
    }
  }, [filing, relevantChunks, primaryChunkId])

  const handleDownload = () => {
    if (!filing) return
    const content = filing.document_markdown || filing.document_text
    const ext = filing.document_markdown ? 'md' : 'txt'
    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${ticker}_${filingType}_FY${fiscalYear}.${ext}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleJumpToSection = (heading: string) => {
    if (!contentRef.current) return
    // Try to find the heading in the rendered markdown
    const allHeadings = contentRef.current.querySelectorAll('h1,h2,h3,h4,h5,h6')
    for (const el of allHeadings) {
      if (el.textContent?.toLowerCase().includes(heading.toLowerCase().slice(0, 20))) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
        return
      }
    }
  }

  const handleZoomIn = () => setZoom(z => Math.min(z + 10, 200))
  const handleZoomOut = () => setZoom(z => Math.max(z - 10, 50))

  const isMarkdown = !!(filing?.document_markdown)
  const hasTOC = !!(filing?.table_of_contents?.length || filing?.document_structure?.length)

  const displayTitle = filing
    ? `${filing.company_name || ticker} — ${filingType} FY${fiscalYear}${quarter ? ` Q${quarter}` : ''}`
    : `${ticker} ${filingType} FY${fiscalYear}`

  if (!isOpen && !panelMode) return null

  const viewerContent = (
    <>
      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white px-5 py-3.5 flex items-center justify-between border-b border-slate-700 shrink-0">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <div className="p-2 bg-white/10 rounded-lg">
            <FileText className="w-5 h-5" />
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-base font-semibold truncate">{displayTitle}</h2>
            {filing?.filing_date && (
              <div className="flex items-center gap-2 text-xs text-slate-300 mt-0.5">
                <Calendar className="w-3 h-3" />
                <span>Filed: {new Date(filing.filing_date).toLocaleDateString()}</span>

                {isMarkdown && (
                  <>
                    <span className="text-slate-500">•</span>
                    <span className="text-emerald-400">Structured</span>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
        <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg transition-colors ml-2">
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* ── Toolbar ── */}
      <div className="bg-slate-50 border-b border-slate-200 px-4 py-2.5 flex items-center justify-between gap-3 shrink-0">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1 bg-white rounded-lg border border-slate-200 px-1">
            <button onClick={handleZoomOut} className="p-1.5 hover:bg-slate-100 rounded transition-colors" title="Zoom out">
              <ZoomOut className="w-4 h-4 text-slate-600" />
            </button>
            <span className="px-2 text-xs font-medium text-slate-600 min-w-[2.5rem] text-center">{zoom}%</span>
            <button onClick={handleZoomIn} className="p-1.5 hover:bg-slate-100 rounded transition-colors" title="Zoom in">
              <ZoomIn className="w-4 h-4 text-slate-600" />
            </button>
          </div>
          {hasTOC && (
            <button
              onClick={() => setShowTOC(v => !v)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
                showTOC ? 'bg-slate-900 text-white border-slate-900' : 'bg-white text-slate-600 border-slate-200 hover:border-slate-400'
              }`}
            >
              Contents
            </button>
          )}
          {relevantChunks.length > 0 && (
            <div className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-50 rounded-lg border border-blue-200">
              <Eye className="w-3.5 h-3.5 text-blue-600" />
              <span className="text-xs font-medium text-blue-900">
                {relevantChunks.length} citation{relevantChunks.length !== 1 ? 's' : ''} highlighted
              </span>
            </div>
          )}
        </div>
        <button
          onClick={handleDownload}
          disabled={!filing}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors text-xs font-medium disabled:opacity-50"
        >
          <Download className="w-3.5 h-3.5" />
          Download
        </button>
      </div>

      {/* ── Body ── */}
      <div className="flex flex-1 overflow-hidden">
        {showTOC && filing && (
          <TOCSidebar
            toc={filing.table_of_contents || []}
            structure={filing.document_structure}
            onJumpTo={handleJumpToSection}
          />
        )}
        <div className="flex-1 overflow-auto bg-slate-100">
          {loading && (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-8 h-8 text-slate-400 animate-spin mx-auto mb-3" />
                <p className="text-slate-600 text-sm">Loading SEC filing…</p>
              </div>
            </div>
          )}
          {error && (
            <div className="h-full flex items-center justify-center">
              <div className="text-center max-w-sm">
                <div className="w-14 h-14 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <FileText className="w-7 h-7 text-red-600" />
                </div>
                <h3 className="text-base font-semibold text-slate-900 mb-2">Filing Not Available</h3>
                <p className="text-sm text-slate-600 mb-4">{error}</p>
                <button onClick={onClose} className="px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors text-sm">
                  Close
                </button>
              </div>
            </div>
          )}
          {filing && !loading && !error && (
            <div className="max-w-5xl mx-auto p-8">
              {isMarkdown ? (
                <MarkdownFilingViewer markdown={filing.document_markdown!} zoom={zoom} contentRef={contentRef} />
              ) : (
                <div
                  ref={contentRef}
                  className="bg-white shadow-lg rounded-lg p-12 sec-filing-document"
                  style={{ fontSize: `${zoom}%`, fontFamily: 'Georgia, "Times New Roman", serif', lineHeight: 1.6 }}
                  dangerouslySetInnerHTML={{ __html: formatSECFiling(filing.document_text) }}
                />
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Footer ── */}
      <div className="bg-slate-50 border-t border-slate-200 px-5 py-2 flex items-center justify-between text-xs text-slate-400 shrink-0">
        <span>SEC EDGAR • {ticker} {filingType} FY{fiscalYear}</span>
        <span>{isMarkdown ? 'Structured Markdown' : 'Plain Text'} • SEC EDGAR Database</span>
      </div>

      <style>{`
        .sec-filing-document { counter-reset: page; }
        .sec-filing-markdown { line-height: 1.65; }
        .highlighted-chunk, mark.highlighted-chunk {
          background: linear-gradient(135deg, rgba(59,130,246,0.22) 0%, rgba(59,130,246,0.12) 100%);
          padding: 1px 3px; border-radius: 3px; border-left: 3px solid rgb(59,130,246);
          cursor: help; transition: all 0.15s ease; display: inline; color: inherit;
        }
        .highlighted-chunk:hover, mark.highlighted-chunk:hover {
          background: linear-gradient(135deg, rgba(59,130,246,0.35) 0%, rgba(59,130,246,0.22) 100%);
          box-shadow: 0 2px 8px rgba(59,130,246,0.18);
        }
        .sec-filing-content { color: #1e293b; }
        .sec-filing-content p { text-align: justify; hyphens: auto; }
        .sec-filing-markdown table tr:nth-child(even) td { background: rgba(241,245,249,0.4); }
        .sec-filing-markdown table tr:has(mark.highlighted-chunk) td,
        .sec-filing-markdown table tr:has(mark.highlighted-chunk) th {
          background: rgba(59,130,246,0.08) !important; outline: 1px solid rgba(59,130,246,0.2);
        }
        @media print {
          .sec-filing-document { box-shadow: none; padding: 0; }
          .highlighted-chunk { background: rgba(255,255,0,0.3) !important; border: none !important; }
        }
      `}</style>
    </>
  )

  // Panel mode: content fills the parent DocumentPanel (no overlay)
  if (panelMode) {
    return <div className="flex flex-col h-full overflow-hidden bg-white">{viewerContent}</div>
  }

  // Modal mode: full-screen overlay
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          onClick={e => e.stopPropagation()}
          className="bg-white rounded-xl shadow-2xl w-full max-w-7xl h-[92vh] flex flex-col overflow-hidden"
        >
          {viewerContent}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}
