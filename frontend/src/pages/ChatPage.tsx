import { useEffect, useRef, useState, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { MessageSquare, Info } from 'lucide-react'
import ChatInput from '../components/ChatInput'
import ChatMessage from '../components/ChatMessage'
import Sidebar from '../components/Sidebar'
import AboutModal from '../components/AboutModal'
import DocumentPanel, { DOCUMENT_PANEL_WIDTH } from '../components/DocumentPanel'
import type { DocumentPanelContent } from '../components/DocumentPanel'
import { useChat } from '../hooks/useChat'

export default function ChatPage() {
  const [searchParams] = useSearchParams()
  const {
    messages,
    isLoading,
    sendMessage,
    conversations,
    currentConversationId,
    loadConversation,
    startNewConversation,
  } = useChat()
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const hasExecutedInitialQuery = useRef(false)
  const shouldScrollRef = useRef(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)
  const [documentPanel, setDocumentPanel] = useState<{ open: boolean; content: DocumentPanelContent | null }>({
    open: false,
    content: null,
  })

  const handleOpenDocument = useCallback((content: DocumentPanelContent) => {
    setDocumentPanel({ open: true, content })
  }, [])

  const handleCloseDocument = useCallback(() => {
    setDocumentPanel(prev => ({ ...prev, open: false }))
  }, [])
  // Scroll after React renders the new message into the DOM
  useEffect(() => {
    if (!shouldScrollRef.current) return
    shouldScrollRef.current = false
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
      })
    })
  }, [messages])

  const handleSendMessage = useCallback((content: string) => {
    shouldScrollRef.current = true
    sendMessage(content)
  }, [sendMessage])

  // Auto-execute query from URL parameter
  useEffect(() => {
    const query = searchParams.get('q')
    if (query && !hasExecutedInitialQuery.current && messages.length === 0) {
      hasExecutedInitialQuery.current = true
      handleSendMessage(decodeURIComponent(query))
    }
  }, [searchParams, handleSendMessage, messages.length])

  const isEmpty = messages.length === 0

  const activePanelWidth = documentPanel.open ? DOCUMENT_PANEL_WIDTH : 0

  return (
    <div className="min-h-screen bg-[#faf9f7]">
      {/* Left Sidebar */}
      <Sidebar
        isCollapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onLoadConversation={loadConversation}
        onNewConversation={startNewConversation}
      />

      {/* Main content area - shifts based on both sidebars */}
      <div
        className="min-h-screen flex flex-col transition-all duration-300"
        style={{
          paddingLeft: sidebarCollapsed ? '60px' : '220px',
          paddingRight: `${activePanelWidth}px`,
        }}
      >
        {/* Header bar */}
        <header className="sticky top-0 z-20 bg-white border-b border-slate-200">
          <div className="flex items-center justify-between h-14 px-4 lg:px-6">
            <div className="flex items-center gap-3">
              <div className="lg:hidden w-10" /> {/* Spacer for mobile menu button */}
              <h1 className="text-lg font-semibold text-[#0a1628]">Research</h1>
              {messages.length > 0 && (
                <span className="text-sm text-slate-400 font-mono">
                  {messages.filter(m => m.role === 'user').length} queries
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              {messages.length > 0 && (
                <button
                  onClick={startNewConversation}
                  className="text-sm text-slate-500 hover:text-[#0a1628] font-medium transition-colors px-3 py-1.5 hover:bg-slate-100 rounded-lg"
                >
                  New session
                </button>
              )}
              <button
                onClick={() => setAboutOpen(true)}
                className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-[#0a1628] font-medium transition-colors px-3 py-1.5 hover:bg-slate-100 rounded-lg"
              >
                <Info className="w-4 h-4" />
                About
              </button>
            </div>
          </div>
        </header>

        {/* Chat area */}
        <main ref={messagesContainerRef} className="flex-1 pb-32 overflow-y-auto">
          <div className={`mx-auto px-4 lg:px-6 transition-all duration-300 ${documentPanel.open ? 'max-w-2xl' : 'max-w-4xl'}`}>
            {isEmpty ? (
              // Empty state - enterprise style
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)] text-center py-12"
              >
                <div className="w-14 h-14 bg-[#0a1628] rounded-xl flex items-center justify-center mb-6">
                  <MessageSquare className="w-7 h-7 text-white" />
                </div>
                <h1 className="text-2xl font-semibold text-[#0a1628] mb-2" style={{ fontFamily: "'Playfair Display', Georgia, serif" }}>
                  Research Query
                </h1>
                <p className="text-slate-500 max-w-md mb-8">
                  Query SEC filings and earnings transcripts for 500+ tech companies.
                  Semiconductors, software, and fintech coverage.
                </p>

                {/* Data Coverage Badges - more muted */}
                <div className="flex flex-wrap justify-center gap-3 mb-8">
                  <span className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded text-xs text-slate-500 font-mono">
                    500+ Companies
                  </span>
                  <span className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded text-xs text-slate-500 font-mono">
                    3Y Earnings
                  </span>
                  <span className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-slate-200 rounded text-xs text-slate-500 font-mono">
                    10-K Filings
                  </span>
                </div>

                {/* Example queries - clean */}
                <div className="grid sm:grid-cols-2 gap-2 w-full max-w-2xl">
                  {[
                    "What is $AAPL's revenue breakdown by segment?",
                    "How has $NVDA's gross margin changed over time?",
                    "What are the main risks mentioned in $TSLA's 10-K?",
                    "What did $MSFT's CEO say about AI in the last earnings call?",
                  ].map((query, index) => (
                    <button
                      key={index}
                      onClick={() => handleSendMessage(query)}
                      className="p-4 text-left bg-white border border-slate-200 rounded-lg hover:border-slate-300 hover:bg-slate-50 transition-all text-sm text-slate-600"
                    >
                      {query}
                    </button>
                  ))}
                </div>
              </motion.div>
            ) : (
              // Messages
              <div className="py-6 space-y-6">
                {messages.map((message) => (
                  <ChatMessage key={message.id} message={message} onOpenDocument={handleOpenDocument} />
                ))}
                <div />
              </div>
            )}
          </div>
        </main>

        {/* Fixed input at bottom */}
        <div
          className="fixed bottom-0 bg-gradient-to-t from-[#faf9f7] via-[#faf9f7] to-transparent pt-6 pb-4 transition-all duration-300"
          style={{
            left: sidebarCollapsed ? '60px' : '200px',
            right: `${activePanelWidth}px`,
          }}
        >
          <div className={`mx-auto px-4 lg:px-6 transition-all duration-300 ${documentPanel.open ? 'max-w-2xl' : 'max-w-4xl'}`}>
            <ChatInput
              onSubmit={handleSendMessage}
              isLoading={isLoading}
              placeholder="Query SEC filings and earnings transcripts..."
              autoFocus={!searchParams.get('q')}
            />
            <p className="text-center text-xs text-slate-400 mt-3">
              Results derived from primary SEC filings and earnings transcripts.
              Always verify for investment decisions.
            </p>
          </div>
        </div>
      </div>

      {/* Document Panel (right sidebar) */}
      <DocumentPanel
        isOpen={documentPanel.open}
        onClose={handleCloseDocument}
        content={documentPanel.content}
      />

      {/* About Modal */}
      <AboutModal isOpen={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  )
}
