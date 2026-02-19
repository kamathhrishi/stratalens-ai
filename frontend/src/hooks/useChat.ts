import { useState, useCallback, useEffect } from 'react'
import { useAuth } from '@clerk/clerk-react'
import {
  streamChat,
  generateMessageId,
  fetchConversations,
  fetchConversation,
  type ChatMessage,
  type ReasoningStep,
  type Source,
  type SSEEvent,
  type Conversation,
} from '../lib/api'

interface UseChatReturn {
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null
  sendMessage: (content: string) => Promise<void>
  clearMessages: () => void
  currentReasoning: ReasoningStep[]
  // Conversation management
  conversations: Conversation[]
  currentConversationId: string | null
  loadConversation: (conversationId: string) => Promise<void>
  startNewConversation: () => void
  refreshConversations: () => Promise<void>
}

export function useChat(): UseChatReturn {
  const { getToken, isSignedIn } = useAuth()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentReasoning, setCurrentReasoning] = useState<ReasoningStep[]>([])
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null)

  // Fetch conversations on mount if signed in
  useEffect(() => {
    if (isSignedIn) {
      refreshConversations()
    }
  }, [isSignedIn])

  const refreshConversations = useCallback(async () => {
    if (!isSignedIn) return
    try {
      const token = await getToken()
      if (token) {
        const convs = await fetchConversations(token)
        setConversations(convs)
      }
    } catch (err) {
      console.error('Failed to fetch conversations:', err)
    }
  }, [isSignedIn, getToken])

  const loadConversation = useCallback(async (conversationId: string) => {
    if (!isSignedIn) return
    setIsLoading(true)
    setError(null)
    setMessages([])  // Clear immediately for clean transition
    try {
      const token = await getToken()
      if (token) {
        const conversation = await fetchConversation(conversationId, token)
        setCurrentConversationId(conversationId)
        // Convert backend messages to frontend format
        const messages = conversation.messages || []
        const loadedMessages: ChatMessage[] = messages.map((msg) => {
          return {
            id: msg.id,
            role: msg.role as 'user' | 'assistant',
            content: msg.content || '',
            sources: msg.citations || [],
            reasoning: msg.reasoning || [],
            timestamp: new Date(msg.created_at),
            isStreaming: false,
          }
        })
        setMessages(loadedMessages)
        setCurrentReasoning([]) // Clear global reasoning when loading saved conversation
      }
    } catch (err) {
      console.error('Failed to load conversation:', err)
      setError(err instanceof Error ? err.message : 'Failed to load conversation')
    } finally {
      setIsLoading(false)
    }
  }, [isSignedIn, getToken])

  const startNewConversation = useCallback(() => {
    setCurrentConversationId(null)
    setMessages([])
    setError(null)
    setCurrentReasoning([])
  }, [])

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return

    setError(null)
    setIsLoading(true)
    setCurrentReasoning([])

    // Get auth token if signed in
    let authToken: string | null = null
    if (isSignedIn) {
      authToken = await getToken()
    }

    // Add user message
    const userMessage: ChatMessage = {
      id: generateMessageId(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date(),
    }

    // Add placeholder assistant message
    const assistantMessageId = generateMessageId()
    const assistantMessage: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      reasoning: [],
      sources: [],
      timestamp: new Date(),
      isStreaming: true,
    }

    setMessages((prev) => [...prev, userMessage, assistantMessage])

    try {
      let accumulatedContent = ''
      const accumulatedReasoning: ReasoningStep[] = []
      let accumulatedSources: Source[] = []
      let newConversationId = currentConversationId

      for await (const event of streamChat(content, {
        conversationId: currentConversationId || undefined,
        authToken,
      })) {
        // Capture conversation_id from first response
        if (event.conversation_id && !newConversationId) {
          newConversationId = event.conversation_id
          setCurrentConversationId(newConversationId)
        }

        handleSSEEvent(
          event,
          assistantMessageId,
          accumulatedContent,
          accumulatedReasoning,
          accumulatedSources,
          (newContent) => {
            accumulatedContent = newContent
          },
          (newSources) => {
            accumulatedSources = newSources
          }
        )
      }

      // Mark streaming as complete
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, isStreaming: false }
            : msg
        )
      )

      // Refresh conversations list after sending a message
      if (isSignedIn) {
        refreshConversations()
      }
    } catch (err) {
      console.error('Chat error:', err)
      const errorMessage = err instanceof Error ? err.message : 'An error occurred'
      setError(errorMessage)

      // Update assistant message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: 'Sorry, an error occurred while processing your request. Please try again.',
                isStreaming: false,
              }
            : msg
        )
      )
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, isSignedIn, getToken, currentConversationId, refreshConversations])

  const handleSSEEvent = useCallback(
    (
      event: SSEEvent,
      messageId: string,
      currentContent: string,
      currentReasoningSteps: ReasoningStep[],
      _currentSources: Source[],
      setContent: (content: string) => void,
      setSources: (sources: Source[]) => void
    ) => {
      switch (event.type) {
        case 'token':
          // Backend sends token in 'content' field, not 'token'
          const tokenContent = event.content || event.token
          if (tokenContent) {
            const newContent = currentContent + tokenContent
            setContent(newContent)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId ? { ...msg, content: newContent } : msg
              )
            )
          }
          break

        // All reasoning/progress event types
        case 'reasoning':
        case 'progress':
        case 'analysis':
        case 'search':
        case 'news_search':
        case '10k_search':
        case 'iteration_start':
        case 'iteration_search':
        case 'iteration_transcript_search':
        case 'iteration_news_search':
        case 'iteration_followup':
        case 'iteration_complete':
        case 'iteration_final':
        case 'agent_decision':
        case 'planning_start':
        case 'planning_complete':
        case 'retrieval_complete':
        case 'evaluation_complete':
        case 'search_complete':
        case '10k_planning':
        case '10k_retrieval':
        case '10k_evaluation':
        case 'api_retry':
          if (event.message) {
            const newStep: ReasoningStep = {
              message: event.message,
              step: event.step || event.type,
              data: event.data,
            }
            currentReasoningSteps.push(newStep)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId
                  ? { ...msg, reasoning: [...currentReasoningSteps] }
                  : msg
              )
            )
          }
          break

        case 'result':
        case '10k_answer': {
          // Handle answer - could be in various nested locations
          const data = event.data as Record<string, unknown> | undefined
          const response = data?.response as Record<string, unknown> | undefined

          // Try multiple paths: event.answer -> event.data.answer -> event.data.response.answer
          const answerContent = event.answer ||
            data?.answer as string ||
            response?.answer as string

          if (answerContent) {
            setContent(answerContent)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId ? { ...msg, content: answerContent } : msg
              )
            )
          }

          // Handle citations - could be in various nested locations
          // Backend sends citations with fields: company, ticker, quarter, chunk_text, chunk_id, etc.
          const citationsData = event.citations ||
            data?.citations ||
            response?.citations
          if (citationsData && Array.isArray(citationsData)) {
            // Pass through ALL fields from backend - cast to any to avoid type stripping
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const newSources: Source[] = citationsData.map((c: any) => ({
              // Common
              title: c.title,
              type: c.type,
              marker: c.marker,

              // Transcript fields
              company: c.company,
              ticker: c.ticker,
              quarter: c.quarter,
              year: c.year,
              chunk_text: c.chunk_text,
              chunk_id: c.chunk_id,
              relevance_score: c.relevance_score,
              transcript_available: c.transcript_available,

              // 10-K fields
              fiscal_year: c.fiscal_year,
              section: c.section,
              chunk_type: c.chunk_type,
              path: c.path,

              // News fields
              url: c.url,
              published_date: c.published_date,

              // Legacy
              page: c.page,
            }))
            setSources(newSources)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId ? { ...msg, sources: newSources } : msg
              )
            )
          }
          break
        }

        case 'error':
          setError(event.message || 'An error occurred')
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === messageId
                ? { ...msg, content: event.message || 'An error occurred', isStreaming: false }
                : msg
            )
          )
          break

        case 'done':
          // Stream complete - no action needed
          break

        default:
          // Handle any unknown event types that have a message
          if (event.message) {
            const newStep: ReasoningStep = {
              message: event.message,
              step: event.step || event.type,
              data: event.data,
            }
            currentReasoningSteps.push(newStep)
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === messageId
                  ? { ...msg, reasoning: [...currentReasoningSteps] }
                  : msg
              )
            )
          }
      }
    },
    []
  )

  const clearMessages = useCallback(() => {
    setMessages([])
    setError(null)
    setCurrentReasoning([])
    setCurrentConversationId(null)
  }, [])

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages,
    currentReasoning,
    conversations,
    currentConversationId,
    loadConversation,
    startNewConversation,
    refreshConversations,
  }
}
