import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import type { ReasoningStep } from '../lib/api'

interface ReasoningTraceProps {
  steps: ReasoningStep[]
  isStreaming?: boolean
}

// Steps to filter out (noise)
const FILTERED_STEPS = [
  'Refining answer with additional context',
  'Analyzing answer quality',
  'Preparing comprehensive response',
  'Generating response...',
  'Generating response',
]

// Check if message should be filtered
function shouldFilter(message: string): boolean {
  return FILTERED_STEPS.some(filter =>
    message.toLowerCase().includes(filter.toLowerCase())
  )
}

// Check if this is a search step with queries
function isSearchStep(step: ReasoningStep): boolean {
  return step.step === 'iteration_search' ||
         step.step === 'iteration_transcript_search' ||
         step.step === 'iteration_news_search' ||
         step.step === 'search' ||
         step.message.startsWith('Searching:')
}

// Parse search queries from message
function parseSearchQueries(message: string): string[] {
  // Handle messages like 'Searching: "query1"\nSearching: "query2"'
  const queries: string[] = []
  const lines = message.split('\n')

  for (const line of lines) {
    const match = line.match(/Searching:\s*"([^"]+)"/)
    if (match) {
      queries.push(match[1])
    }
  }

  return queries.length > 0 ? queries : []
}

// Group consecutive search steps
function processSteps(steps: ReasoningStep[]): ReasoningStep[] {
  const filtered = steps.filter(step => !shouldFilter(step.message))
  const processed: ReasoningStep[] = []
  let pendingSearchQueries: string[] = []

  for (const step of filtered) {
    if (isSearchStep(step)) {
      const queries = parseSearchQueries(step.message)
      if (queries.length > 0) {
        pendingSearchQueries.push(...queries)
      } else if (step.message.startsWith('Searching:')) {
        // Single search query without quotes
        const query = step.message.replace('Searching:', '').trim().replace(/^"|"$/g, '')
        if (query) {
          pendingSearchQueries.push(query)
        }
      }
    } else {
      // Flush pending search queries as a single step
      if (pendingSearchQueries.length > 0) {
        processed.push({
          message: pendingSearchQueries.join('\n'),
          step: 'search_group',
          data: { queries: pendingSearchQueries }
        })
        pendingSearchQueries = []
      }
      processed.push(step)
    }
  }

  // Flush any remaining search queries
  if (pendingSearchQueries.length > 0) {
    processed.push({
      message: pendingSearchQueries.join('\n'),
      step: 'search_group',
      data: { queries: pendingSearchQueries }
    })
  }

  return processed
}

export default function ReasoningTrace({ steps, isStreaming }: ReasoningTraceProps) {
  const processedSteps = processSteps(steps)

  if (processedSteps.length === 0) return null

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      transition={{ duration: 0.2 }}
      className="overflow-hidden bg-slate-50/80 rounded-xl p-4 border border-slate-200/60"
    >
      <div className="space-y-2">
        {processedSteps.map((step, index) => {
          // Use stable key based on message content to prevent re-animation
          const stableKey = `${step.step}-${step.message.slice(0, 50)}-${index}`
          const isLatest = index === processedSteps.length - 1

          return (
            <motion.div
              key={stableKey}
              initial={isLatest ? { opacity: 0, x: -8 } : false}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.15 }}
              className="flex items-start gap-2.5"
            >
              {/* Simple circular bullet - lighter blue */}
              <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-[#0083f1]/60 mt-[7px]" />

              {step.step === 'search_group' ? (
                // Search group with queries on separate lines
                <div className="flex-1">
                  <span className="text-slate-500 text-sm">Searching for additional context</span>
                  <div className="mt-1.5 ml-1 space-y-1">
                    {(step.data?.queries as string[] || step.message.split('\n')).map((query, qIdx) => (
                      <p key={qIdx} className="text-slate-400 text-sm italic leading-relaxed">
                        "{query}"
                      </p>
                    ))}
                  </div>
                </div>
              ) : step.message.includes('\n') ? (
                // Multi-line message (e.g., bullet point lists)
                <div className="flex-1">
                  {step.message.split('\n').map((line, lineIdx) => {
                    const trimmedLine = line.trim()
                    const isBullet = trimmedLine.startsWith('-') || trimmedLine.startsWith('•')
                    const cleanLine = isBullet ? trimmedLine.substring(1).trim() : line
                    return (
                      <p
                        key={lineIdx}
                        className={`text-slate-500 text-sm leading-relaxed ${isBullet ? 'ml-2 flex items-start gap-1.5' : ''}`}
                      >
                        {isBullet && <span className="text-slate-400 flex-shrink-0">-</span>}
                        <span>{cleanLine}</span>
                      </p>
                    )
                  })}
                </div>
              ) : (
                // Regular step - lighter text for less prominence
                <span className="text-slate-500 text-sm leading-relaxed">{step.message}</span>
              )}
            </motion.div>
          )
        })}
        {isStreaming && processedSteps.length > 0 && (
          <div className="flex items-center gap-2 text-slate-400 mt-2 pt-2 border-t border-slate-200/60">
            <Loader2 className="w-4 h-4 animate-spin text-[#0083f1]/70" />
            <span className="text-sm">Processing...</span>
          </div>
        )}
      </div>
    </motion.div>
  )
}
