import { motion, AnimatePresence } from 'framer-motion'
import SECFilingViewer from './SECFilingViewer'
import TranscriptModal from './TranscriptModal'

interface SECFilingPanelContent {
  type: 'sec-filing'
  ticker: string
  filingType: string
  fiscalYear: number
  quarter?: number
  filingDate?: string
  relevantChunks?: Array<{
    chunk_text: string
    chunk_id?: string
    sec_section?: string
    relevance_score?: number
    char_offset?: number
  }>
  primaryChunkId?: string
}

interface TranscriptPanelContent {
  type: 'transcript'
  company: string
  ticker: string
  quarter: string
  relevantChunks?: Array<{
    chunk_text: string
    chunk_id?: string
    relevance_score?: number
  }>
}

export type DocumentPanelContent = SECFilingPanelContent | TranscriptPanelContent

interface DocumentPanelProps {
  isOpen: boolean
  onClose: () => void
  content: DocumentPanelContent | null
}

const PANEL_WIDTH = 680

export default function DocumentPanel({ isOpen, onClose, content }: DocumentPanelProps) {
  return (
    <AnimatePresence>
      {isOpen && content && (
        <motion.div
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ type: 'spring', damping: 28, stiffness: 260 }}
          className="fixed top-0 right-0 h-full z-40 shadow-2xl border-l border-slate-200 flex flex-col"
          style={{ width: PANEL_WIDTH }}
        >
          {content.type === 'sec-filing' ? (
            <SECFilingViewer
              key={`${content.ticker}-${content.filingType}-${content.fiscalYear}`}
              isOpen={true}
              onClose={onClose}
              ticker={content.ticker}
              filingType={content.filingType}
              fiscalYear={content.fiscalYear}
              quarter={content.quarter}
              filingDate={content.filingDate}
              relevantChunks={content.relevantChunks}
              primaryChunkId={content.primaryChunkId}
              panelMode={true}
            />
          ) : (
            <TranscriptModal
              key={`${content.ticker}-${content.quarter}`}
              isOpen={true}
              onClose={onClose}
              company={content.company}
              ticker={content.ticker}
              quarter={content.quarter}
              relevantChunks={content.relevantChunks || []}
              panelMode={true}
            />
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export const DOCUMENT_PANEL_WIDTH = PANEL_WIDTH
