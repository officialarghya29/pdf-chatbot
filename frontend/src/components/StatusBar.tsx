import { FileText, Wifi, Cpu, RefreshCw } from 'lucide-react'
import type { Health, SessionSummary } from '../lib/api'

interface Props {
  connection: 'online' | 'offline'
  session: SessionSummary | null
  health: Health | null
  onRetry: () => void
}

export default function StatusBar({ connection, session, health, onRetry }: Props) {
  const offline = connection === 'offline' && !health

  return (
    <header className="relative z-10 flex items-center justify-between border-b border-edge/60
      bg-panel/40 px-6 py-3 backdrop-blur-xl">
      <div className="flex min-w-0 items-center gap-3">
        {session ? (
          <>
            <FileText className="h-4 w-4 shrink-0 text-cyan-400" />
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-white">{session.title}</div>
              <div className="text-[11px] text-slate-500">
                {session.pages} pages · {session.chunks} chunks indexed
              </div>
            </div>
          </>
        ) : (
          <div className="text-sm font-medium text-slate-400">No document selected</div>
        )}
      </div>

      <div className="flex items-center gap-3 text-[11px]">
        {health && (
          <span className="hidden items-center gap-1.5 rounded-full border border-edge bg-white/[0.03]
            px-3 py-1 text-slate-400 sm:flex">
            <Cpu className="h-3 w-3 text-violet-400" />
            <span className="font-mono">{health.version}</span>
          </span>
        )}

        {offline ? (
          <button
            onClick={onRetry}
            className="flex items-center gap-1.5 rounded-full border border-red-500/30 bg-red-500/10
              px-3 py-1 font-medium text-red-300 transition-colors hover:bg-red-500/20"
          >
            <RefreshCw className="h-3 w-3" />
            Backend offline
          </button>
        ) : (
          <span className="flex items-center gap-1.5 rounded-full border border-emerald-500/25
            bg-emerald-500/10 px-3 py-1 font-medium text-emerald-300">
            <Wifi className="h-3 w-3" />
            Connected
          </span>
        )}
      </div>
    </header>
  )
}
