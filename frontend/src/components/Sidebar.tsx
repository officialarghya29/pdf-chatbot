import { useState } from 'react'
import { MessageSquare, Plus, Trash2, FileText, Sparkles } from 'lucide-react'
import type { SessionSummary } from '../lib/api'
import { timeAgo } from '../lib/utils'

interface Props {
  sessions: SessionSummary[]
  activeId: string | null
  onNew: () => void
  onOpen: (id: string) => void
  onDelete: (id: string) => void
}

export default function Sidebar({ sessions, activeId, onNew, onOpen, onDelete }: Props) {
  const [confirmId, setConfirmId] = useState<string | null>(null)

  return (
    <aside className="glass-strong z-20 flex h-full w-72 shrink-0 flex-col border-r">
      {/* brand */}
      <div className="flex items-center gap-3 px-5 pb-4 pt-5">
        <div className="relative">
          <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-cyan-500 to-violet-500 blur-md opacity-50" />
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-panel border border-edge">
            <Sparkles className="h-5 w-5 text-cyan-400" />
          </div>
        </div>
        <div>
          <div className="text-[15px] font-bold tracking-tight text-white">Unfold</div>
          <div className="text-[11px] font-medium uppercase tracking-widest text-cyan-400/80">
            PDF · AI
          </div>
        </div>
      </div>

      <div className="px-4 pb-3">
        <button onClick={onNew} className="btn-primary w-full justify-center">
          <Plus className="h-4 w-4" />
          New Chat
        </button>
      </div>

      {/* sessions */}
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-2">
        <div className="px-2 pb-2 pt-1 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
          Documents
        </div>
        {sessions.length === 0 ? (
          <div className="px-2 py-6 text-xs text-slate-600">No documents yet.</div>
        ) : (
          <ul className="space-y-1">
            {sessions.map((s) => (
              <li key={s.session_id}>
                <button
                  onClick={() => onOpen(s.session_id)}
                  className={`group flex w-full items-center gap-2.5 rounded-xl px-3 py-2.5 text-left transition-all ${
                    activeId === s.session_id
                      ? 'bg-gradient-to-r from-cyan-500/15 to-violet-500/10 ring-1 ring-cyan-500/30'
                      : 'hover:bg-white/5'
                  }`}
                >
                  <FileText
                    className={`h-4 w-4 shrink-0 ${
                      activeId === s.session_id ? 'text-cyan-400' : 'text-slate-500'
                    }`}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] font-medium text-slate-200">
                      {s.title}
                    </span>
                    <span className="block text-[11px] text-slate-500">
                      {s.pages} pages · {timeAgo(s.created_at)}
                    </span>
                  </span>
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation()
                      setConfirmId((v) => (v === s.session_id ? null : s.session_id))
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && setConfirmId(s.session_id)}
                    className={`shrink-0 rounded-md p-1 text-slate-500 opacity-0 transition-all hover:text-red-400 group-hover:opacity-100 ${
                      confirmId === s.session_id ? 'opacity-100' : ''
                    }`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </span>
                </button>
                {confirmId === s.session_id && (
                  <div className="mx-2 mb-1 flex items-center justify-between rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-1.5 text-[11px]">
                    <span className="text-red-300">Delete this chat?</span>
                    <span className="flex gap-1">
                      <button
                        onClick={() => setConfirmId(null)}
                        className="rounded-md px-2 py-0.5 text-slate-400 hover:text-slate-200"
                      >
                        No
                      </button>
                      <button
                        onClick={() => {
                          onDelete(s.session_id)
                          setConfirmId(null)
                        }}
                        className="rounded-md bg-red-500/20 px-2 py-0.5 font-semibold text-red-300 hover:bg-red-500/30"
                      >
                        Delete
                      </button>
                    </span>
                    </div>
                  )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-t border-edge/60 px-5 py-3 text-[11px] text-slate-600">
        <div className="flex items-center gap-1.5">
          <MessageSquare className="h-3 w-3" />
          {sessions.length} document{sessions.length === 1 ? '' : 's'} stored
        </div>
      </div>
    </aside>
  )
}
