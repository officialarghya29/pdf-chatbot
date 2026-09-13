import { useState } from 'react'
import { AlertTriangle, X } from 'lucide-react'

interface Props {
  configured: boolean
}

export default function ApiWarning({ configured }: Props) {
  const [dismissed, setDismissed] = useState(false)

  if (configured || dismissed) return null

  return (
    <div className="relative z-50 border-b border-amber-500/30 bg-amber-500/10 backdrop-blur-md">
      <div className="mx-auto flex max-w-3xl items-center gap-3 px-6 py-3">
        <AlertTriangle className="h-5 w-5 shrink-0 text-amber-400" />
        <div className="flex-1 text-sm text-amber-200">
          <span className="font-semibold text-amber-100">API key not configured.</span>{' '}
          Set <code className="rounded bg-black/20 px-1.5 py-0.5 text-xs font-mono text-amber-300">OPENAI_API_KEY</code> in{' '}
          <code className="rounded bg-black/20 px-1.5 py-0.5 text-xs font-mono text-amber-300">backend/.env</code>{' '}
          and restart the server to unlock PDF chat.
        </div>
        <button
          onClick={() => setDismissed(true)}
          className="shrink-0 rounded-lg p-1.5 text-amber-400/60 hover:text-amber-200 hover:bg-white/5"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}
