import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Send, Loader2 } from 'lucide-react'

interface ChatInputProps {
  onSubmit: (message: string) => void
  isLoading?: boolean
  placeholder?: string
  autoFocus?: boolean
  initialValue?: string
  size?: 'default' | 'large'
}

export default function ChatInput({
  onSubmit,
  isLoading = false,
  placeholder = 'Ask about any company...',
  autoFocus = false,
  initialValue = '',
  size = 'default',
}: ChatInputProps) {
  const [value, setValue] = useState(initialValue)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus()
    }
  }, [autoFocus])

  useEffect(() => {
    if (initialValue) {
      setValue(initialValue)
    }
  }, [initialValue])

  const handleSubmit = () => {
    if (value.trim() && !isLoading) {
      onSubmit(value.trim())
      setValue('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  // Auto-resize textarea up to ~300px, then scroll
  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    const newHeight = Math.min(el.scrollHeight, 300)
    el.style.height = `${newHeight}px`
    el.style.overflowY = el.scrollHeight > 300 ? 'auto' : 'hidden'
  }, [value])

  const isLarge = size === 'large'

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`relative flex items-end gap-3 bg-white border border-slate-300 rounded-lg shadow-sm hover:border-slate-400 focus-within:border-[#0a1628] focus-within:ring-1 focus-within:ring-[#0a1628] transition-all ${
        isLarge ? 'p-4' : 'p-3'
      }`}
    >
      <textarea
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={isLoading}
        rows={1}
        className={`flex-1 resize-none bg-transparent border-none outline-none placeholder:text-slate-400 text-[#0a1628] overflow-hidden ${
          isLarge ? 'text-lg' : 'text-base'
        }`}
      />
      <button
        onClick={handleSubmit}
        disabled={!value.trim() || isLoading}
        className={`flex-shrink-0 flex items-center justify-center rounded-lg transition-colors ${
          isLarge ? 'w-11 h-11' : 'w-9 h-9'
        } ${
          value.trim() && !isLoading
            ? 'bg-[#0a1628] hover:bg-[#1e293b] text-white'
            : 'bg-slate-100 text-slate-400 cursor-not-allowed'
        }`}
      >
        {isLoading ? (
          <Loader2 className={`${isLarge ? 'w-5 h-5' : 'w-4 h-4'} animate-spin`} />
        ) : (
          <Send className={`${isLarge ? 'w-5 h-5' : 'w-4 h-4'}`} />
        )}
      </button>
    </motion.div>
  )
}
