import { useCallback, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowUp, Paperclip, Square, UploadCloud } from 'lucide-react'
import { api, type SessionSummary } from '../lib/api'

interface Props {
  hasSession: boolean
  disabled: boolean
  onSend: (text: string) => void
  onStop: () => void
  onUploaded: (summary: SessionSummary) => void
}

export default function Composer({ hasSession, disabled, onSend, onStop, onUploaded }: Props) {
  const [text, setText] = useState('')
  const [uploading, setUploading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [uploadName, setUploadName] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  const fileRef = useRef<HTMLInputElement>(null)
  const taRef = useRef<HTMLTextAreaElement>(null)

  const doUpload = useCallback(
    async (file: File) => {
      setError(null)
      setUploading(true)
      setProgress(0)
      setUploadName(file.name)
      try {
        const summary = await api.upload(file, setProgress)
        onUploaded(summary)
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setUploading(false)
        setProgress(0)
        setUploadName('')
      }
    },
    [onUploaded],
  )

  const send = () => {
    const t = text.trim()
    if (!t || disabled || uploading) return
    onSend(t)
    setText('')
    // reset auto-grown height so the box collapses back to one line
    if (taRef.current) taRef.current.style.height = 'auto'
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) void doUpload(f)
  }

  return (
    <div
      className="relative z-10 px-6 pb-6 pt-2"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <div className="mx-auto w-full max-w-3xl">
        {/* upload progress / error banner */}
        <AnimatePresence>
          {(uploading || error) && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
              className={`mb-2 rounded-xl border px-4 py-2.5 text-xs ${
                error
                  ? 'border-red-500/30 bg-red-500/10 text-red-300'
                  : 'border-cyan-500/25 bg-cyan-500/5 text-cyan-300'
              }`}
            >
              {error ? (
                error
              ) : (
                <div className="flex items-center gap-3">
                  <UploadCloud className="h-4 w-4 shrink-0 animate-pulse" />
                  <span className="min-w-0 flex-1 truncate">
                    Indexing <b>{uploadName}</b>…
                  </span>
                  <span className="font-mono">{progress}%</span>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        <div
          className={`relative flex items-end gap-2 rounded-2xl border p-2 shadow-2xl transition-all
            ${dragging
              ? 'border-cyan-400 bg-cyan-500/10 shadow-cyan-500/20'
              : 'border-edge bg-panel/80 shadow-black/40 backdrop-blur-xl focus-within:border-cyan-500/50'}`}
        >
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf,.pdf"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) void doUpload(f)
              e.target.value = ''
            }}
          />

          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            title="Upload a PDF"
            className="btn-ghost h-10 w-10 justify-center !px-0"
          >
            <Paperclip className="h-[18px] w-[18px]" />
          </button>

          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder={
              uploading
                ? 'Processing document…'
                : hasSession
                  ? 'Ask anything about your document…'
                  : 'Upload a PDF first, then ask anything…'
            }
            className="max-h-40 min-h-[40px] flex-1 resize-none bg-transparent py-2.5 text-[15px]
              text-slate-100 placeholder:text-slate-600 outline-none"
            onInput={(e) => {
              const el = e.currentTarget
              el.style.height = 'auto'
              el.style.height = `${Math.min(el.scrollHeight, 160)}px`
            }}
          />

          {disabled ? (
            <button
              onClick={onStop}
              title="Stop generating"
              className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10
                text-slate-300 transition-colors hover:bg-white/20"
            >
              <Square className="h-4 w-4 fill-current" />
            </button>
          ) : (
            <button
              onClick={send}
              disabled={!text.trim() || uploading}
              title="Send message"
              className="btn-primary h-10 w-10 justify-center !px-0"
            >
              <ArrowUp className="h-5 w-5" />
            </button>
          )}
        </div>

        <div className="mt-2 text-center text-[11px] text-slate-600">
          {dragging
            ? 'Drop your PDF to index it'
            : 'Enter to send · Shift+Enter for new line · Attach or drop a PDF'}
        </div>
      </div>
    </div>
  )
}
