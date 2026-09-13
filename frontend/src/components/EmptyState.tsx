import { motion } from 'framer-motion'
import { Upload, Zap, Shield, Quote } from 'lucide-react'

interface Props {
  onPick: (text: string) => void
}

const SUGGESTIONS = [
  'Summarize the key points of this document',
  'What are the main conclusions?',
  'Explain the methodology in simple terms',
  'List any risks or limitations mentioned',
]

const FEATURES = [
  { icon: Zap, label: 'Streaming answers', desc: 'Tokens appear in real time' },
  { icon: Quote, label: 'Cited sources', desc: 'Every answer maps to pages' },
  { icon: Shield, label: 'Local sessions', desc: 'Your docs stay on your server' },
]

export default function EmptyState({ onPick }: Props) {
  return (
    <div className="flex min-h-full flex-col items-center justify-center px-6 py-12">
      {/* hero */}
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="flex flex-col items-center text-center"
      >
        <div className="relative mb-8">
          <div className="absolute -inset-6 rounded-full bg-gradient-to-br from-cyan-500/20 to-violet-500/20 blur-2xl animate-pulse-slow" />
          <motion.div
            animate={{ y: [0, -8, 0] }}
            transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
            className="relative flex h-24 w-24 items-center justify-center rounded-3xl border border-edge bg-panel shadow-2xl shadow-cyan-500/10"
          >
            <Upload className="h-10 w-10 text-cyan-400" />
          </motion.div>
        </div>

        <h1 className="text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
          Chat with any{' '}
          <span className="bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
            PDF
          </span>
        </h1>
        <p className="mt-3 max-w-md text-[15px] leading-relaxed text-slate-400">
          Upload a document and ask questions. NeoChat reads it, cites its pages
          and answers in real time.
        </p>
      </motion.div>

      {/* suggestions */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.5 }}
        className="mt-10 grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2"
      >
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="group rounded-2xl border border-edge bg-panel/60 p-4 text-left backdrop-blur
              transition-all hover:border-cyan-500/40 hover:bg-panel hover:shadow-lg hover:shadow-cyan-500/5
              active:scale-[0.98]"
          >
            <div className="text-[13px] font-medium text-slate-300 group-hover:text-white">
              {s}
            </div>
            <div className="mt-1 text-[11px] text-slate-600 group-hover:text-cyan-400/70">
              Ask after uploading a PDF →
            </div>
          </button>
        ))}
      </motion.div>

      {/* features */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4, duration: 0.6 }}
        className="mt-12 flex flex-wrap items-center justify-center gap-x-8 gap-y-3"
      >
        {FEATURES.map((f) => (
          <div key={f.label} className="flex items-center gap-2.5">
            <f.icon className="h-4 w-4 text-cyan-400/80" />
            <div>
              <div className="text-xs font-semibold text-slate-300">{f.label}</div>
              <div className="text-[10px] text-slate-600">{f.desc}</div>
            </div>
          </div>
        ))}
      </motion.div>
    </div>
  )
}
