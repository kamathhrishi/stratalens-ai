import { useState, useRef, useEffect } from 'react'
import { useAuth } from '@clerk/clerk-react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Search,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  Download,
  X,
  ChevronDown,
  ChevronUp,
  FileText,
  Table,
  LinkIcon,
} from 'lucide-react'
import Sidebar from '../components/Sidebar'
import { config } from '../lib/config'
import type { ReasoningStep } from '../lib/api'
import ReasoningTrace from '../components/ReasoningTrace'

interface Source {
  type: string
  ticker: string
  chunk_text: string
  similarity: number
  marker: string
  year?: number
  quarter?: number
  fiscal_year?: number
  section?: string
}

interface ScreenerResult {
  success: boolean
  columns: string[]
  friendly_columns: Record<string, string>
  data_rows: Record<string, unknown>[]
  sql_query_generated?: string
  message?: string
  error?: string
  tables_used?: string[]
  execution_time?: number
  pagination_info?: {
    page: number
    page_size: number
    total_rows: number
    total_pages: number
  }
  query_id?: string
}

interface SortState {
  column: string | null
  direction: 'asc' | 'desc'
}

const EXAMPLE_QUERIES = [
  'Top 10 tech stocks by revenue growth',
  'Companies with P/E under 15 and dividend yield over 3%',
  'Companies investing heavily in AI infrastructure',
  'Tech stocks with revenue > $50B that are investing in AI',
]

// Citation display component
function CitationsSection({ sources }: { sources: Source[] }) {
  const [expanded, setExpanded] = useState(false)
  const [expandedChunk, setExpandedChunk] = useState<number | null>(null)

  if (!sources || sources.length === 0) return null

  const sourceType = sources[0]?.type === 'transcript' ? 'Transcript' : '10-K Filing'
  const sourceIcon = sources[0]?.type === 'transcript' ? <FileText className="w-3.5 h-3.5" /> : <Table className="w-3.5 h-3.5" />

  return (
    <div className="mt-2 rounded-lg border border-slate-200 overflow-hidden bg-white">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-slate-50 hover:bg-slate-100 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <LinkIcon className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs font-medium text-[#0a1628]">
            {sources.length} citation{sources.length > 1 ? 's' : ''}
          </span>
          <span className="text-xs text-slate-400">({sourceType})</span>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        )}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: 'auto' }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-2 space-y-1.5">
              {sources.map((source, idx) => {
                const isExpanded = expandedChunk === idx
                const text = source.chunk_text || ''
                const preview = text.substring(0, 120)

                return (
                  <div key={idx} className="border border-slate-200 rounded-lg overflow-hidden">
                    <div className="p-2 bg-slate-50">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            {sourceIcon}
                            <span className="text-xs font-medium text-[#0a1628]">
                              {source.type === 'transcript'
                                ? `Q${source.quarter} ${source.year}`
                                : `FY${source.fiscal_year}`}
                            </span>
                            {source.section && (
                              <span className="text-xs text-slate-500">{source.section}</span>
                            )}
                          </div>
                          <p className="text-xs text-slate-600 leading-relaxed">
                            {isExpanded ? text : preview + (text.length > 120 ? '...' : '')}
                          </p>
                        </div>
                        {text.length > 120 && (
                          <button
                            onClick={() => setExpandedChunk(isExpanded ? null : idx)}
                            className="text-xs text-[#0083f1] hover:text-[#0066cc] font-medium"
                          >
                            {isExpanded ? 'Less' : 'More'}
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default function ScreenerPage() {
  const { getToken } = useAuth()
  const navigate = useNavigate()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [query, setQuery] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [reasoningSteps, setReasoningSteps] = useState<ReasoningStep[]>([])
  const [result, setResult] = useState<ScreenerResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sortState, setSortState] = useState<SortState>({ column: null, direction: 'asc' })
  const [currentPage, setCurrentPage] = useState(1)
  const [topN, setTopN] = useState(20)
  const abortControllerRef = useRef<AbortController | null>(null)

  const pageSize = 50

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  const executeQuery = async () => {
    if (!query.trim() || isLoading) return

    // Reset state
    setIsLoading(true)
    setReasoningSteps([])
    setResult(null)
    setError(null)
    setSortState({ column: null, direction: 'asc' })
    setCurrentPage(1)

    // Abort any existing request
    abortControllerRef.current?.abort()
    abortControllerRef.current = new AbortController()

    try {
      const token = await getToken()

      const params = new URLSearchParams({
        question: query,
        page: '1',
        page_size: pageSize.toString(),
        top_n: topN.toString(),
      })

      const response = await fetch(
        `${config.apiBaseUrl}/screener/smart/stream?${params}`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        signal: abortControllerRef.current.signal,
      })

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim()
            if (jsonStr && jsonStr !== '[DONE]') {
              try {
                const event = JSON.parse(jsonStr)
                handleSSEEvent(event)
              } catch (e) {
                console.warn('Failed to parse SSE event:', e)
              }
            }
          }
        }
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        // Request was cancelled
        return
      }
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSSEEvent = (event: Record<string, unknown>) => {
    const eventType = event.type as string
    switch (eventType) {
      case 'reasoning':
        if (event.event) {
          const reasoningEvent = event.event as { message: string; details?: { step?: string } }
          setReasoningSteps(prev => [
            ...prev,
            {
              message: reasoningEvent.message,
              step: reasoningEvent.details?.step || 'reasoning',
              data: reasoningEvent.details,
            },
          ])
        }
        break
      case 'intent_result':
        // Intent analysis complete
        break
      case 'partial_result':
        // Progressive streaming - update table as data comes in
        const partialData = event.data as ScreenerResult | undefined
        if (partialData && partialData.data_rows && partialData.columns) {
          setResult(partialData)
          setError(null)
        }
        break
      case 'result':
        // Final result data
        const resultData = event.data as ScreenerResult | undefined
        console.log('Result event received:', resultData)
        if (resultData) {
          // Check if this is a valid result with data
          if (resultData.data_rows && resultData.columns) {
            setResult(resultData)
            setError(null) // Clear any previous error
          } else if (resultData.error) {
            setError(resultData.error)
          } else if (!resultData.success) {
            setError(resultData.message || 'Query failed')
          }
        }
        break
      case 'error':
        setError((event.message as string) || 'An error occurred')
        break
    }
  }

  const isQualitativeResult = result?.columns?.includes('evidence_summary')

  const handleSort = async (column: string) => {
    if (!result || isLoading) return

    // For qualitative results (no server-side sort), sort client-side
    if (isQualitativeResult) {
      const newDirection =
        sortState.column === column && sortState.direction === 'asc' ? 'desc' : 'asc'
      setSortState({ column, direction: newDirection })

      const sortedRows = [...result.data_rows].sort((a, b) => {
        const aVal = a[column]
        const bVal = b[column]
        if (aVal === null || aVal === undefined) return 1
        if (bVal === null || bVal === undefined) return -1
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return newDirection === 'asc' ? aVal - bVal : bVal - aVal
        }
        const aStr = String(aVal)
        const bStr = String(bVal)
        return newDirection === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr)
      })
      setResult(prev => (prev ? { ...prev, data_rows: sortedRows } : null))
      return
    }

    const newDirection =
      sortState.column === column && sortState.direction === 'asc' ? 'desc' : 'asc'

    setSortState({ column, direction: newDirection })

    try {
      const token = await getToken()
      const response = await fetch(`${config.apiBaseUrl}/screener/query/sort`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          column,
          direction: newDirection,
          query_id: result.query_id,
        }),
      })

      if (!response.ok) throw new Error('Sort failed')

      const data = await response.json()
      if (data.success) {
        setResult(prev =>
          prev
            ? {
                ...prev,
                data_rows: data.data_rows,
                pagination_info: data.pagination_info,
              }
            : null
        )
      }
    } catch (err) {
      console.error('Sort error:', err)
    }
  }

  const handlePageChange = async (newPage: number) => {
    if (!result || isLoading) return

    setCurrentPage(newPage)

    try {
      const token = await getToken()
      const response = await fetch(`${config.apiBaseUrl}/screener/query/paginate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          question: query,
          page: newPage,
          page_size: pageSize,
        }),
      })

      if (!response.ok) throw new Error('Pagination failed')

      const data = await response.json()
      if (data.success) {
        setResult(prev =>
          prev
            ? {
                ...prev,
                data_rows: data.data_rows,
                pagination_info: data.pagination_info,
              }
            : null
        )
      }
    } catch (err) {
      console.error('Pagination error:', err)
    }
  }

  const cancelQuery = async () => {
    abortControllerRef.current?.abort()
    try {
      const token = await getToken()
      await fetch(`${config.apiBaseUrl}/screener/cancel`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })
    } catch (e) {
      // Ignore cancel errors
    }
    setIsLoading(false)
  }

  const exportCSV = () => {
    if (!result) return

    const headers = result.columns.map(col => result.friendly_columns[col] || col)
    const rows = result.data_rows.map(row =>
      result.columns.map(col => {
        const val = row[col]
        if (val === null || val === undefined) return ''
        const str = String(val)
        if (str.includes(',') || str.includes('"') || str.includes('\n')) return `"${str.replace(/"/g, '""')}"`
        return str
      })
    )

    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `screener-${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const formatCellValue = (value: unknown, column?: string): string => {
    if (value === null || value === undefined) return '\u2014'
    if (column === 'relevance_score' && typeof value === 'number') {
      return `${(value * 100).toFixed(1)}%`
    }
    if (typeof value === 'number') {
      if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}B`
      if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`
      if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(2)}K`
      if (Number.isInteger(value)) return value.toLocaleString()
      return value.toFixed(2)
    }
    return String(value)
  }

  // Get background color based on relevance score (0-1)
  const getRelevanceColor = (score: number): string => {
    // Convert to 0-100 scale
    const percentage = score * 100

    if (percentage >= 70) {
      // High relevance: Strong green
      return 'bg-emerald-100 border-emerald-300 text-emerald-900'
    } else if (percentage >= 50) {
      // Medium-high: Light green
      return 'bg-green-50 border-green-200 text-green-900'
    } else if (percentage >= 30) {
      // Medium: Yellow/amber
      return 'bg-amber-50 border-amber-200 text-amber-900'
    } else {
      // Low: Gray
      return 'bg-slate-50 border-slate-200 text-slate-700'
    }
  }

  const isQualitativeTextColumn = (col: string) =>
    col === 'evidence_summary' || col === 'citations'

  const totalPages = result?.pagination_info?.total_pages || 1
  const totalRows = result?.pagination_info?.total_rows || result?.data_rows?.length || 0

  return (
    <div className="min-h-screen bg-[#faf9f7]">
      <Sidebar
        isCollapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main
        className={`transition-all duration-200 ${
          sidebarCollapsed ? 'lg:ml-[72px]' : 'lg:ml-[240px]'
        }`}
      >
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-[#0a1628] mb-1">Stock Screener</h1>
            <p className="text-slate-500">
              Screen stocks on financials, qualitative themes, or both at once
            </p>
          </div>

          {/* Search Input - unified card */}
          <div className="bg-white rounded-xl border border-slate-200 p-5 mb-6">
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="text"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && executeQuery()}
                  placeholder="e.g., Tech stocks with revenue > $50B that are investing in AI"
                  className="w-full pl-10 pr-4 py-3 bg-slate-50 border border-slate-200 rounded-lg text-[#0a1628] placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#0083f1]/20 focus:border-[#0083f1] transition-colors"
                  disabled={isLoading}
                />
              </div>
              <select
                value={topN}
                onChange={e => setTopN(Number(e.target.value))}
                className="px-4 py-3 bg-slate-50 border border-slate-200 rounded-lg text-[#0a1628] focus:outline-none focus:ring-2 focus:ring-[#0083f1]/20 focus:border-[#0083f1] transition-colors font-medium"
                disabled={isLoading}
              >
                <option value={10}>Top 10</option>
                <option value={20}>Top 20</option>
                <option value={30}>Top 30</option>
                <option value={50}>Top 50</option>
              </select>
              {isLoading ? (
                <button
                  onClick={cancelQuery}
                  className="px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors flex items-center gap-2 font-medium"
                >
                  <X className="w-5 h-5" />
                  Cancel
                </button>
              ) : (
                <button
                  onClick={executeQuery}
                  disabled={!query.trim()}
                  className="px-6 py-3 bg-[#0a1628] text-white rounded-lg hover:bg-[#1e293b] transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 font-medium"
                >
                  <Search className="w-5 h-5" />
                  Search
                </button>
              )}
            </div>

            {/* Example queries - show when idle */}
            {!isLoading && !result && reasoningSteps.length === 0 && (
              <div className="mt-5">
                <p className="text-sm text-slate-500 mb-3">Example queries:</p>
                <div className="grid sm:grid-cols-2 gap-2">
                  {EXAMPLE_QUERIES.map((example, i) => (
                    <button
                      key={i}
                      onClick={() => setQuery(example)}
                      className="p-4 text-left bg-white border border-slate-200 rounded-lg hover:border-slate-300 hover:bg-slate-50 transition-all text-sm text-slate-600"
                    >
                      {example}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Reasoning Trace - inside the same card with light background */}
            <AnimatePresence>
              {(isLoading || reasoningSteps.length > 0) && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-5 pt-5 border-t border-slate-200"
                >
                  <ReasoningTrace steps={reasoningSteps} isStreaming={isLoading} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6">
              <p className="text-red-600">{error}</p>
            </div>
          )}

          {/* Results Table */}
          {result && result.data_rows && result.data_rows.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-xl border border-slate-200 overflow-hidden"
            >
              {/* Table Header */}
              <div className="flex items-center justify-between px-5 py-3.5 border-b border-slate-200 bg-slate-50">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold text-[#0a1628]">
                    {totalRows.toLocaleString()} {totalRows === 1 ? 'result' : 'results'}
                  </span>
                  {result.execution_time && (
                    <span className="text-xs text-slate-500 font-mono">
                      {result.execution_time.toFixed(2)}s
                    </span>
                  )}
                </div>
                <button
                  onClick={exportCSV}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-600 hover:text-[#0a1628] hover:bg-slate-100 rounded-lg transition-colors font-medium"
                >
                  <Download className="w-4 h-4" />
                  Export CSV
                </button>
              </div>

              {/* Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200">
                      {result.columns.map(col => (
                        <th
                          key={col}
                          onClick={() => handleSort(col)}
                          className={`px-5 py-3 text-left text-xs font-semibold text-slate-600 uppercase tracking-wide cursor-pointer hover:bg-slate-100 transition-colors ${
                            isQualitativeTextColumn(col) ? 'min-w-[300px]' : 'whitespace-nowrap'
                          }`}
                        >
                          <div className="flex items-center gap-1.5">
                            {result.friendly_columns[col] || col}
                            {sortState.column === col ? (
                              sortState.direction === 'asc' ? (
                                <ArrowUp className="w-4 h-4 text-[#0083f1]" />
                              ) : (
                                <ArrowDown className="w-4 h-4 text-[#0083f1]" />
                              )
                            ) : (
                              <ArrowUpDown className="w-4 h-4 text-slate-300" />
                            )}
                          </div>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {result.data_rows.map((row, rowIdx) => (
                      <tr
                        key={rowIdx}
                        className="hover:bg-slate-50 transition-colors"
                      >
                        {result.columns.map(col => {
                          const isSymbolColumn = col.toLowerCase() === 'symbol' || col.toLowerCase() === 'ticker'
                          const isCompanyColumn = col.toLowerCase().includes('company') || col.toLowerCase().includes('name')
                          const symbol = row['symbol'] || row['Symbol'] || row['ticker'] || row['Ticker']
                          const isClickable = (isSymbolColumn || isCompanyColumn) && symbol

                          // Special rendering for evidence_summary column
                          if (col === 'evidence_summary') {
                            return (
                              <td key={col} className="px-5 py-3.5 text-sm max-w-md">
                                <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                                  <span className="whitespace-pre-wrap text-[#0a1628] leading-relaxed">
                                    {String(row[col] || '\u2014')}
                                  </span>
                                </div>
                              </td>
                            )
                          }

                          // Citations column - separate from evidence
                          if (col === 'citations') {
                            const citations = row[col] as Source[] | undefined
                            return (
                              <td key={col} className="px-5 py-3.5 text-sm">
                                {citations && citations.length > 0 ? (
                                  <CitationsSection sources={citations} />
                                ) : (
                                  <span className="text-slate-400">\u2014</span>
                                )}
                              </td>
                            )
                          }

                          // Special styling for relevance_score
                          if (col === 'relevance_score') {
                            // If score is null/undefined, show blurred placeholder
                            if (row[col] === null || row[col] === undefined) {
                              return (
                                <td key={col} className="px-5 py-3.5 text-sm whitespace-nowrap">
                                  <span className="inline-flex items-center px-3 py-1.5 rounded-lg border border-slate-200 bg-slate-50 text-slate-400 font-semibold blur-sm select-none">
                                    85.0%
                                  </span>
                                </td>
                              )
                            }
                            // Show actual score
                            if (typeof row[col] === 'number') {
                              const score = row[col] as number
                              return (
                                <td key={col} className="px-5 py-3.5 text-sm whitespace-nowrap">
                                  <span className={`inline-flex items-center px-3 py-1.5 rounded-lg border font-semibold ${getRelevanceColor(score)}`}>
                                    {formatCellValue(row[col], col)}
                                  </span>
                                </td>
                              )
                            }
                          }

                          return (
                            <td
                              key={col}
                              className="px-5 py-3.5 text-sm whitespace-nowrap"
                            >
                              {isClickable ? (
                                <button
                                  onClick={() => {
                                    navigate(`/companies/${symbol}`)
                                  }}
                                  className="text-[#0083f1] hover:text-[#0066cc] font-medium hover:underline transition-colors"
                                >
                                  {formatCellValue(row[col], col)}
                                </button>
                              ) : (
                                <span className="text-[#0a1628]">{formatCellValue(row[col], col)}</span>
                              )}
                            </td>
                          )
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination (only for financial results with pagination info) */}
              {!isQualitativeResult && totalPages > 1 && (
                <div className="flex items-center justify-between px-5 py-3.5 border-t border-slate-200 bg-slate-50">
                  <span className="text-sm text-slate-500">
                    Page {currentPage} of {totalPages}
                  </span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => handlePageChange(currentPage - 1)}
                      disabled={currentPage === 1}
                      className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </button>
                    {/* Page numbers */}
                    {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                      let pageNum: number
                      if (totalPages <= 5) {
                        pageNum = i + 1
                      } else if (currentPage <= 3) {
                        pageNum = i + 1
                      } else if (currentPage >= totalPages - 2) {
                        pageNum = totalPages - 4 + i
                      } else {
                        pageNum = currentPage - 2 + i
                      }
                      return (
                        <button
                          key={pageNum}
                          onClick={() => handlePageChange(pageNum)}
                          className={`min-w-[32px] h-8 px-2 text-sm rounded-lg transition-colors font-medium ${
                            currentPage === pageNum
                              ? 'bg-[#0a1628] text-white'
                              : 'text-slate-600 hover:bg-slate-100'
                          }`}
                        >
                          {pageNum}
                        </button>
                      )
                    })}
                    <button
                      onClick={() => handlePageChange(currentPage + 1)}
                      disabled={currentPage === totalPages}
                      className="p-2 text-slate-600 hover:bg-slate-100 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Empty State */}
          {!isLoading && !result && !error && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-xl border border-slate-200 p-12 text-center"
            >
              <div className="w-14 h-14 bg-slate-100 rounded-xl flex items-center justify-center mx-auto mb-6">
                <Search className="w-7 h-7 text-slate-400" />
              </div>
              <h3 className="text-xl font-semibold text-[#0a1628] mb-2">
                Start screening stocks
              </h3>
              <p className="text-slate-500 max-w-lg mx-auto leading-relaxed">
                Screen stocks on financial metrics, qualitative themes from earnings transcripts
                and 10-K filings, or combine both in a single query.
              </p>
            </motion.div>
          )}
        </div>
      </main>
    </div>
  )
}
