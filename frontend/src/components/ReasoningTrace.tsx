import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, ChevronDown, ChevronRight, FileText, Globe } from 'lucide-react'
import type { ReasoningStep } from '../lib/api'
import type { DocumentPanelContent } from './DocumentPanel'

interface ReasoningTraceProps {
  steps: ReasoningStep[]
  isStreaming?: boolean
  defaultCollapsed?: boolean
  onDocumentClick?: (content: DocumentPanelContent) => void
}

interface DocRef {
  // SEC / transcript fields
  ticker?: string
  fiscal_year?: number
  filing_type?: string
  year?: number | string
  quarter?: number | string
  // News fields
  title?: string
  url?: string
}

// Steps to filter out (noise)
const FILTERED_STEPS = [
  'Refining answer with additional context',
  'Analyzing answer quality',
  'Preparing comprehensive response',
  'Generating response...',
  'Generating response',
]

function shouldFilter(message: string): boolean {
  return FILTERED_STEPS.some(filter =>
    message.toLowerCase().includes(filter.toLowerCase())
  )
}

function cleanMessage(message: string): string {
  return message
    .replace(/[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]|[\u{1F600}-\u{1F64F}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]/gu, '')
    .replace(/[📊📈📋✅❌⚠️🔍🔄💹🚀💎🐘🏗️👥🔐⚡🤖📚🖥️]/g, '')
    .trim()
}

function isSearchStep(step: ReasoningStep): boolean {
  return step.step === 'iteration_search' ||
         step.step === 'iteration_transcript_search' ||
         step.step === 'iteration_news_search' ||
         step.message.startsWith('Searching:')
}

function parseSearchQueries(message: string): string[] {
  const queries: string[] = []
  const lines = message.split('\n')
  for (const line of lines) {
    const match = line.match(/Searching:\s*"([^"]+)"/)
    if (match) queries.push(match[1])
  }
  return queries
}

function extractDocuments(step: ReasoningStep): DocRef[] {
  if (step.step === 'news_search') {
    return (step.data?.articles as DocRef[] | undefined) ?? []
  }
  return (step.data?.documents as DocRef[] | undefined) ?? []
}

function hasDocuments(step: ReasoningStep): boolean {
  if (step.step === 'news_search') {
    const articles = step.data?.articles as DocRef[] | undefined
    return Array.isArray(articles) && articles.length > 0
  }
  const docs = step.data?.documents as DocRef[] | undefined
  return Array.isArray(docs) && docs.length > 0
}

function processSteps(steps: ReasoningStep[]): ReasoningStep[] {
  const filtered = steps.filter(step => !shouldFilter(step.message))
  const processed: ReasoningStep[] = []
  let pendingSearchQueries: string[] = []

  for (const step of filtered) {
    // 10k_search, transcript 'search', and news_search steps with documents — never group, always keep
    if ((step.step === '10k_search' || step.step === 'search' || step.step === 'news_search') && hasDocuments(step)) {
      if (pendingSearchQueries.length > 0) {
        processed.push({ message: pendingSearchQueries.join('\n'), step: 'search_group', data: { queries: pendingSearchQueries } })
        pendingSearchQueries = []
      }
      processed.push(step)
    } else if (isSearchStep(step)) {
      const queries = parseSearchQueries(step.message)
      if (queries.length > 0) {
        pendingSearchQueries.push(...queries)
      } else if (step.message.startsWith('Searching:')) {
        const query = step.message.replace('Searching:', '').trim().replace(/^"|"$/g, '')
        if (query) pendingSearchQueries.push(query)
      }
    } else {
      if (pendingSearchQueries.length > 0) {
        processed.push({ message: pendingSearchQueries.join('\n'), step: 'search_group', data: { queries: pendingSearchQueries } })
        pendingSearchQueries = []
      }
      processed.push(step)
    }
  }

  if (pendingSearchQueries.length > 0) {
    processed.push({ message: pendingSearchQueries.join('\n'), step: 'search_group', data: { queries: pendingSearchQueries } })
  }

  return processed
}

function DocumentChips({ docs, onDocumentClick }: { docs: DocRef[], onDocumentClick?: (content: DocumentPanelContent) => void }) {
  if (docs.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1 mt-1.5">
      {docs.map((doc, i) => {
        // News article chip — opens URL in new tab
        if (doc.url) {
          const label = doc.title
            ? doc.title.length > 40 ? doc.title.slice(0, 40) + '…' : doc.title
            : new URL(doc.url).hostname
          return (
            <a
              key={i}
              href={doc.url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-slate-200 text-slate-800 border border-slate-300 hover:bg-slate-300 transition-colors"
            >
              <Globe className="w-3 h-3 flex-shrink-0" />
              {label}
            </a>
          )
        }

        // SEC filing / transcript chip — opens document viewer
        if (!onDocumentClick || !doc.ticker) return null
        const isTranscript = doc.quarter !== undefined
        const label = isTranscript
          ? `${doc.ticker} Q${doc.quarter} ${doc.year}`
          : `${doc.ticker} ${doc.fiscal_year} ${doc.filing_type ?? '10-K'}`

        const content: DocumentPanelContent = isTranscript
          ? { type: 'transcript', company: doc.ticker, ticker: doc.ticker, quarter: `Q${doc.quarter} ${doc.year}` }
          : { type: 'sec-filing', ticker: doc.ticker, filingType: doc.filing_type ?? '10-K', fiscalYear: doc.fiscal_year! }

        return (
          <button
            key={i}
            onClick={() => onDocumentClick(content)}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-slate-200 text-slate-800 border border-slate-300 hover:bg-slate-300 transition-colors"
          >
            <FileText className="w-3 h-3" />
            {label}
          </button>
        )
      })}
    </div>
  )
}

export default function ReasoningTrace({ steps, isStreaming, defaultCollapsed = false, onDocumentClick }: ReasoningTraceProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed)
  const processedSteps = processSteps(steps)

  if (processedSteps.length === 0) return null

  const lastStep = processedSteps[processedSteps.length - 1]
  const summaryText = isStreaming
    ? 'Processing...'
    : cleanMessage(lastStep.message).slice(0, 60) + (lastStep.message.length > 60 ? '...' : '')

  return (
    <div className="bg-slate-50/80 rounded-lg border border-slate-200/60 overflow-hidden">
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-slate-100/50 transition-colors"
      >
        {isCollapsed ? (
          <ChevronRight className="w-4 h-4 text-slate-400 flex-shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400 flex-shrink-0" />
        )}
        <span className="text-sm text-slate-500 flex-1 truncate">
          {isCollapsed ? summaryText : 'Reasoning'}
        </span>
        {isStreaming && (
          <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-400 flex-shrink-0" />
        )}
        <span className="text-xs text-slate-400">
          {processedSteps.length} step{processedSteps.length !== 1 ? 's' : ''}
        </span>
      </button>

      <AnimatePresence initial={false}>
        {!isCollapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 space-y-1.5 border-t border-slate-200/60">
              {processedSteps.map((step, index) => {
                const stableKey = `${step.step}-${step.message.slice(0, 50)}-${index}`
                const isLatest = index === processedSteps.length - 1
                const cleanedMessage = cleanMessage(step.message)
                const docs = extractDocuments(step)

                return (
                  <motion.div
                    key={stableKey}
                    initial={isLatest ? { opacity: 0, x: -4 } : false}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.15 }}
                    className="flex items-start gap-2"
                  >
                    <span className="flex-shrink-0 w-1 h-1 rounded-full bg-slate-300 mt-[7px]" />

                    {step.step === 'search_group' ? (
                      <div className="flex-1">
                        <span className="text-slate-500 text-sm">Searching for context</span>
                        <div className="mt-1 ml-2 space-y-0.5">
                          {(step.data?.queries as string[] || step.message.split('\n')).map((query, qIdx) => (
                            <p key={qIdx} className="text-slate-400 text-xs italic">
                              "{query}"
                            </p>
                          ))}
                        </div>
                      </div>
                    ) : cleanedMessage.includes('\n') ? (
                      <div className="flex-1">
                        {cleanedMessage.split('\n').map((line, lineIdx) => {
                          const trimmedLine = line.trim()
                          const isBullet = trimmedLine.startsWith('-') || trimmedLine.startsWith('•')
                          const cleanLine = isBullet ? trimmedLine.substring(1).trim() : line
                          return (
                            <p key={lineIdx} className={`text-slate-500 text-sm leading-relaxed ${isBullet ? 'ml-2' : ''}`}>
                              {isBullet && <span className="text-slate-400 mr-1">-</span>}
                              {cleanLine}
                            </p>
                          )
                        })}
                        <DocumentChips docs={docs} onDocumentClick={onDocumentClick} />
                      </div>
                    ) : (
                      <div className="flex-1">
                        <span className="text-slate-500 text-sm leading-relaxed">{cleanedMessage}</span>
                        <DocumentChips docs={docs} onDocumentClick={onDocumentClick} />
                      </div>
                    )}
                  </motion.div>
                )
              })}
              {isStreaming && (
                <div className="flex items-center gap-2 text-slate-400 mt-1.5 pt-1.5 border-t border-slate-100">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span className="text-xs">Processing...</span>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
