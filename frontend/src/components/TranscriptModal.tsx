import { useState, useEffect, useCallback, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Download, Building2, Calendar, FileText, Highlighter } from 'lucide-react'
import { config } from '../lib/config'

interface TranscriptChunk {
  chunk_text: string
  chunk_id?: string
  relevance_score?: number
}

interface TranscriptModalProps {
  isOpen: boolean
  onClose: () => void
  company: string
  ticker: string
  quarter: string
  relevantChunks: TranscriptChunk[]
  panelMode?: boolean  // When true, renders as embedded panel content (no overlay)
}

interface TranscriptData {
  success: boolean
  transcript_text: string
  metadata?: {
    date?: string
    title?: string
  }
}

// Parse quarter string to year and quarter number
function parseQuarter(quarter: string | number | null | undefined): { year: number; quarterNum: number } | null {
  if (quarter == null) return null
  const quarterStr = String(quarter)

  const patterns = [
    /(\d{4})[-_\s]?[Qq](\d)/,
    /[Qq](\d)[-_\s]?(\d{4})/,
  ]

  for (const pattern of patterns) {
    const match = quarterStr.match(pattern)
    if (match) {
      if (pattern === patterns[0]) {
        return { year: parseInt(match[1]), quarterNum: parseInt(match[2]) }
      } else {
        return { year: parseInt(match[2]), quarterNum: parseInt(match[1]) }
      }
    }
  }
  return null
}

// Format transcript text with speaker sections (matches old vanilla JS approach)
function formatTranscriptWithSpeakers(transcriptText: string, relevantChunks: TranscriptChunk[] = []): string {
  if (!transcriptText) return 'No transcript available'

  // Enhanced speaker patterns to better match earnings call transcripts
  const speakerPatterns = [
    // Pattern for full names like "Kenneth J. Dorell:" or "Mark Elliot Zuckerberg:"
    /^([A-Z][a-zA-Z\s]+[A-Za-z]):\s/gm,
    // Pattern for names with middle initials like "Kenneth J. Dorell:"
    /^([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+):\s/gm,
    // Pattern for names with periods like "Mr. John Pitzer:"
    /^([A-Z][a-z]*\.?\s*[A-Za-z\s]+[A-Za-z]):\s/gm,
    // Pattern for "Operator:" style single words
    /^([A-Z][a-z]+):\s/gm,
    // Pattern for names with hyphens like "Lip-Bu Tan:"
    /^([A-Za-z]+-[A-Za-z\s]+[A-Za-z]):\s/gm,
  ]

  let formattedText = transcriptText
  let hasSpeakers = false
  const commonWords = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'all', 'this', 'that']
  let speakerIndex = 0

  // Apply formatting for each speaker pattern
  speakerPatterns.forEach(pattern => {
    formattedText = formattedText.replace(pattern, (match, speaker) => {
      const cleanSpeaker = speaker.trim()
      // Skip if it's too short, contains numbers, or is a common word
      if (cleanSpeaker.length < 3 || /\d/.test(cleanSpeaker) || commonWords.includes(cleanSpeaker.toLowerCase())) {
        return match
      }
      hasSpeakers = true
      const separatorClass = speakerIndex > 0 ? 'mt-7 pt-5 border-t border-slate-200' : ''
      speakerIndex++
      return `<div class="speaker-section mb-3 ${separatorClass}"><div class="speaker-name font-bold text-slate-900 mb-2 text-[15px] tracking-tight">${cleanSpeaker}:</div>`
    })
  })

  // Close any unclosed speaker sections and wrap content
  if (hasSpeakers) {
    formattedText = formattedText.replace(
      /(<div class="speaker-section mb-6"><div class="speaker-name font-semibold text-\[#0083f1\] mb-2 text-base">[^<]*<\/div>)([^<]*?)(?=<div class="speaker-section mb-6">|$)/gs,
      (_match, speakerTag, content) => {
        const cleanContent = content.trim().replace(/\n\s*\n/g, '\n')
        return speakerTag + `<div class="speaker-content text-slate-700 leading-relaxed whitespace-pre-wrap">${cleanContent}</div></div>`
      }
    )
  } else {
    formattedText = `<div class="speaker-section mb-4"><div class="speaker-content text-slate-700 leading-relaxed whitespace-pre-wrap">${formattedText}</div></div>`
  }

  // Apply chunk highlighting if relevant chunks are provided
  if (relevantChunks && relevantChunks.length > 0) {
    formattedText = highlightRelevantChunks(formattedText, relevantChunks)
  }

  return formattedText
}

// Highlight relevant chunks in the transcript
function highlightRelevantChunks(formattedText: string, relevantChunks: TranscriptChunk[]): string {
  let highlightedText = formattedText
  let highlightCount = 0

  relevantChunks.forEach((chunk, index) => {
    if (chunk.chunk_text && chunk.chunk_text.trim().length > 20) {
      // Try multiple strategies to find and highlight the chunk
      const strategies = [
        // Strategy 1: Full chunk text (first 200 chars)
        chunk.chunk_text.substring(0, 200),
        // Strategy 2: First 100 chars
        chunk.chunk_text.substring(0, 100),
        // Strategy 3: First 50 chars
        chunk.chunk_text.substring(0, 50),
      ]

      for (const textToMatch of strategies) {
        if (textToMatch.length < 20) continue

        try {
          // Escape special regex characters
          const escapedChunkText = textToMatch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
          // Create regex with flexible whitespace
          const chunkRegex = new RegExp(escapedChunkText.replace(/\s+/g, '\\s+'), 'gi')

          if (chunkRegex.test(highlightedText)) {
            const relevanceScore = chunk.relevance_score || 0.5
            const highlightIntensity = Math.min(Math.max(relevanceScore * 100, 20), 60) / 100
            const isFirst = highlightCount === 0

            highlightedText = highlightedText.replace(chunkRegex, (match) => {
              return `<mark ${isFirst ? 'id="first-highlight"' : ''} class="highlighted-chunk" style="background: linear-gradient(135deg, rgba(59, 130, 246, ${highlightIntensity}) 0%, rgba(59, 130, 246, ${highlightIntensity * 0.6}) 100%); padding: 4px 8px; border-radius: 4px; border-left: 3px solid rgb(59, 130, 246); display: inline; cursor: pointer; transition: all 0.2s ease;" data-chunk-id="${chunk.chunk_id || index}" title="Cited passage - ${(relevanceScore * 100).toFixed(0)}% relevant">${match}</mark>`
            })
            highlightCount++
            break // Stop trying strategies once one succeeds
          }
        } catch (e) {
          console.warn('Failed to highlight chunk:', e)
        }
      }
    }
  })

  console.log(`Highlighted ${highlightCount} of ${relevantChunks.length} chunks`)
  return highlightedText
}

export default function TranscriptModal({
  isOpen,
  onClose,
  company,
  ticker,
  quarter,
  relevantChunks,
  panelMode = false,
}: TranscriptModalProps) {
  const [transcript, setTranscript] = useState<TranscriptData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [highlightCount, setHighlightCount] = useState(0)
  const contentRef = useRef<HTMLDivElement>(null)

  // Fetch transcript when modal opens
  useEffect(() => {
    if (!isOpen) return

    const fetchTranscript = async () => {
      setLoading(true)
      setError(null)

      const parsed = parseQuarter(quarter)
      if (!parsed) {
        setError(`Invalid quarter format: ${quarter}`)
        setLoading(false)
        return
      }

      try {
        const response = await fetch(
          `${config.apiBaseUrl}/transcript/${ticker}/${parsed.year}/${parsed.quarterNum}`
        )

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.detail || `Transcript not found for ${ticker} Q${parsed.quarterNum} ${parsed.year}`)
        }

        const data = await response.json()
        setTranscript(data)
      } catch (err) {
        console.error('Failed to fetch transcript:', err)
        setError(err instanceof Error ? err.message : 'Failed to load transcript')
      } finally {
        setLoading(false)
      }
    }

    fetchTranscript()
  }, [isOpen, ticker, quarter])

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      document.body.style.overflow = 'hidden'
    }

    return () => {
      document.removeEventListener('keydown', handleEscape)
      document.body.style.overflow = ''
    }
  }, [isOpen, onClose])

  // Scroll to first highlight and count highlights after render
  useEffect(() => {
    if (!loading && transcript && contentRef.current) {
      setTimeout(() => {
        const highlights = contentRef.current?.querySelectorAll('.highlighted-chunk')
        setHighlightCount(highlights?.length || 0)

        const firstHighlight = document.getElementById('first-highlight')
        if (firstHighlight) {
          firstHighlight.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }
      }, 100)
    }
  }, [loading, transcript])

  // Download transcript
  const handleDownload = useCallback(() => {
    if (!transcript?.transcript_text) return

    const blob = new Blob([transcript.transcript_text], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${ticker}_${quarter}_transcript.txt`
    a.click()
    URL.revokeObjectURL(url)
  }, [transcript, ticker, quarter])

  // Process transcript with formatting and highlighting
  const processedTranscript = transcript?.transcript_text
    ? formatTranscriptWithSpeakers(transcript.transcript_text, relevantChunks)
    : ''

  const parsed = parseQuarter(quarter)
  const displayQuarter = parsed ? `Q${parsed.quarterNum} ${parsed.year}` : quarter

  const modalContent = (
    <>
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white">
              <div>
                <h2 className="text-xl font-bold text-slate-900">
                  {company || ticker} {displayQuarter} Earnings Transcript
                </h2>
                {relevantChunks.length > 0 && (
                  <p className="text-sm text-slate-500 mt-0.5">
                    {highlightCount > 0 ? `${highlightCount} section${highlightCount !== 1 ? 's' : ''} highlighted` : `${relevantChunks.length} relevant section${relevantChunks.length !== 1 ? 's' : ''}`}
                  </p>
                )}
              </div>
              <button
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Metadata bar */}
            <div className="flex items-center gap-4 px-6 py-3 bg-slate-50 border-b border-slate-100 text-sm text-slate-600">
              <div className="flex items-center gap-1.5">
                <Building2 className="w-4 h-4 text-[#0083f1]" />
                <span className="font-medium">{ticker}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Calendar className="w-4 h-4 text-[#0083f1]" />
                <span>{displayQuarter}</span>
              </div>
              {transcript?.metadata?.date && (
                <div className="flex items-center gap-1.5">
                  <FileText className="w-4 h-4 text-[#0083f1]" />
                  <span>{transcript.metadata.date}</span>
                </div>
              )}
              {highlightCount > 0 && (
                <div className="flex items-center gap-1.5 ml-auto">
                  <Highlighter className="w-4 h-4 text-blue-500" />
                  <span className="text-blue-600 font-medium">
                    {highlightCount} highlighted
                  </span>
                </div>
              )}
            </div>

            {/* Highlight summary banner */}
            {!loading && !error && highlightCount > 0 && (
              <div className="px-6 py-3 bg-blue-50 border-b border-blue-100">
                <div className="flex items-center gap-2">
                  <Highlighter className="w-4 h-4 text-blue-600" />
                  <span className="font-semibold text-blue-800">
                    {highlightCount} Relevant Section{highlightCount !== 1 ? 's' : ''} Found
                  </span>
                </div>
                <p className="text-sm text-blue-700 mt-1">
                  Highlighted sections show the passages cited in the AI response. Scroll down or click highlights for details.
                </p>
              </div>
            )}

            {/* Content */}
            <div ref={contentRef} className="flex-1 overflow-y-auto px-6 py-4">
              {loading ? (
                <div className="flex flex-col items-center justify-center py-12">
                  <div className="w-8 h-8 border-3 border-[#0083f1] border-t-transparent rounded-full animate-spin" />
                  <p className="mt-4 text-slate-500">Loading transcript...</p>
                </div>
              ) : error ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mb-4">
                    <FileText className="w-8 h-8 text-slate-400" />
                  </div>
                  <p className="text-slate-700 font-medium">Full earnings transcripts coming soon</p>
                  <p className="text-slate-400 text-sm mt-2">
                    We're expanding our transcript coverage. Check back shortly.
                  </p>
                </div>
              ) : (
                <div
                  className="prose prose-slate max-w-none"
                  dangerouslySetInnerHTML={{ __html: processedTranscript }}
                />
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-slate-200 bg-slate-50">
              <p className="text-sm text-slate-500">
                {highlightCount > 0 ? 'Click highlighted sections for details' : 'Full earnings call transcript'}
              </p>
              <button
                onClick={handleDownload}
                disabled={!transcript?.transcript_text}
                className="flex items-center gap-2 px-4 py-2 bg-[#0083f1] text-white rounded-lg hover:bg-[#0070d8] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Download className="w-4 h-4" />
                Download
              </button>
            </div>
    </>
  )

  // Panel mode: fills the parent DocumentPanel
  if (panelMode) {
    return <div className="flex flex-col h-full overflow-hidden bg-white">{modalContent}</div>
  }

  // Modal mode: full-screen overlay
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          onClick={onClose}
        >
          <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
          <motion.div
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="relative w-full max-w-4xl max-h-[90vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {modalContent}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
