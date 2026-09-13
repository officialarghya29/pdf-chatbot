import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown, FileSearch, X } from 'lucide-react'
import type { Source } from '../lib/api'

interface Props {
  sources: Source[]
}

export default function SourcePanel({ sources }: Props) {
  const [open, setOpen] = useState(false)

  if (sources.length === 0) return null

  return (
    <div className="relative z-10 mx-auto w-full max-w-3xl px-6 pb-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between rounded-xl border border-edge
          bg-panel/80 px-4 py-2 text-xs backdrop-blur transition-colors
          hover:border-cyan-500/30"
      >
        <span className="flex items-center gap-2 text-slate-400">
          <FileSearch className="h-3.5 w-3.5 text-cyan-400" />
          <span className="font-medium text-slate-300">{sources.length} source{sources.length === 1 ? '' : 's'}</span>
          <span className="text-slate-600">· pages {sources.map((s) => s.page).join(', ')}</span>
        </span>
        {open ? (
          <X className="h-3.5 w-3.5 text-slate-500" />
        ) : (
          <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8, height: 0 }}
            animate={{ opacity: 1, y: 0, height: 'auto' }}
            exit={{ opacity: 0, y: -8, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 max-h-48 space-y-2 overflow-y-auto rounded-xl border border-edge
              bg-panel/90 p-3 backdrop-blur">
              {sources.map((s, i) => (
                <div key={i} className="rounded-lg bg-white/[0.03] p-3">
                  <div className="mb-1 flex items-center gap-2 text-[11px] font-semibold text-cyan-300">
                    <span className="rounded-md bg-cyan-500/15 px-1.5 py-0.5">p.{s.page}</span>
                    <span className="text-slate-600">chunk #{i + 1}</span>
                  </div>
                  <p className="line-clamp-3 text-[12px] leading-relaxed text-slate-400">
                    {s.snippet}
                  </p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
