import { motion } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { FileText, AlertTriangle, Bot } from 'lucide-react'
import type { ChatMessage } from '../lib/api'

interface Props {
  message: ChatMessage
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
        className="mb-6 flex justify-end"
      >
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-gradient-to-br from-cyan-600 to-violet-600
          px-4 py-3 text-[15px] leading-relaxed text-white shadow-lg shadow-cyan-950/40">
          {message.content}
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="group mb-6 flex gap-3"
    >
      <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-xl
        border border-edge bg-panel shadow-inner">
        {message.error ? (
          <AlertTriangle className="h-4 w-4 text-red-400" />
        ) : (
          <Bot className="h-4 w-4 text-cyan-400" />
        )}
      </div>

      <div className="min-w-0 flex-1">
        {message.pending && !message.content ? (
          <Thinking />
        ) : (
          <div
            className={`md ${
              message.error ? 'rounded-2xl border border-red-500/30 bg-red-500/5 px-4 py-3 text-red-200' : ''
            } ${message.pending && message.content ? 'caret' : ''}`}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* citations */}
        {!message.pending && message.citations && message.citations.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {message.citations.map((c) => (
              <span
                key={c.n}
                title={`Cited from page ${c.page}`}
                className="inline-flex items-center gap-1 rounded-lg border border-cyan-500/25
                  bg-cyan-500/10 px-2 py-0.5 text-[11px] font-medium text-cyan-300"
              >
                <FileText className="h-3 w-3" />
                p.{c.page}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}

function Thinking() {
  return (
    <div className="flex items-center gap-2 py-2">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          animate={{ opacity: [0.25, 1, 0.25], y: [0, -3, 0] }}
          transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
          className="h-1.5 w-1.5 rounded-full bg-cyan-400"
        />
      ))}
      <span className="ml-1 text-xs text-slate-500">Reading the document…</span>
    </div>
  )
}
