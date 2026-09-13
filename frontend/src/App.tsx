import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Health, type ChatMessage, type SessionSummary, type Source } from './lib/api'
import { uid } from './lib/utils'
import Sidebar from './components/Sidebar'
import EmptyState from './components/EmptyState'
import MessageBubble from './components/MessageBubble'
import SourcePanel from './components/SourcePanel'
import Composer from './components/Composer'
import StatusBar from './components/StatusBar'
import ApiWarning from './components/ApiWarning'

type Connection = 'online' | 'offline'

export default function App() {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sources, setSources] = useState<Source[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [connection, setConnection] = useState<Connection>('online')

  const scrollRef = useRef<HTMLDivElement>(null)
  const pinnedRef = useRef(true)
  const abortRef = useRef<(() => void) | null>(null)
  const activeIdRef = useRef<string | null>(null)
  activeIdRef.current = activeId

  const activeSession = sessions.find((s) => s.session_id === activeId) ?? null

  const refreshSessions = useCallback(async () => {
    try {
      const list = await api.listSessions()
      setSessions(list)
    } catch {
      setConnection('offline')
    }
  }, [])

  // boot: health check + session list
  useEffect(() => {
    ;(async () => {
      try {
        const h = await api.health()
        setConnection('online')
        setHealth(h)
        await refreshSessions()
      } catch {
        setConnection('offline')
        setHealth(null)
      }
    })()
    return () => abortRef.current?.()
  }, [refreshSessions])

  // open a session: fetch its history
  const openSession = useCallback(async (id: string) => {
    abortRef.current?.()
    setActiveId(id)
    setSources([])
    try {
      const { messages: history } = await api.getMessages(id)
      setMessages(
        history.map((m) => ({
          id: uid(),
          role: m.role === 'assistant' ? 'assistant' : 'user',
          content: m.content,
        })),
      )
    } catch {
      setMessages([])
    }
  }, [])

  const newChat = useCallback(() => {
    abortRef.current?.()
    setActiveId(null)
    setMessages([])
    setSources([])
  }, [])

  const deleteSession = useCallback(
    async (id: string) => {
      try {
        await api.deleteSession(id)
      } catch {
        /* ignore */
      }
      if (activeIdRef.current === id) newChat()
      await refreshSessions()
    },
    [newChat, refreshSessions],
  )

  const handleUploaded = useCallback(
    (summary: SessionSummary) => {
      setSessions((prev) => [summary, ...prev.filter((s) => s.session_id !== summary.session_id)])
      setActiveId(summary.session_id)
      setMessages([])
      setSources([])
      setHealth((prev) => prev ? { ...prev, sessions: (prev.sessions ?? 0) + 1 } : prev)
    },
    [],
  )

  const sendMessage = useCallback(
    async (text: string) => {
      let sessionId = activeIdRef.current

      // Optimistic user bubble
      const userMsg: ChatMessage = { id: uid(), role: 'user', content: text }
      setMessages((m) => [...m, userMsg])

      if (!sessionId) {
        setMessages((m) => [
          ...m,
          {
            id: uid(),
            role: 'assistant',
            content:
              'You have no document loaded yet. **Attach a PDF** with the paperclip button (or drag & drop it onto the composer), then ask your question.',
            error: true,
          },
        ])
        return
      }

      const assistantId = uid()
      setMessages((m) => [
        ...m,
        { id: assistantId, role: 'assistant', content: '', pending: true },
      ])
      setSources([])

      const patch = (fn: (m: ChatMessage) => ChatMessage) =>
        setMessages((all) => all.map((m) => (m.id === assistantId ? fn(m) : m)))

      abortRef.current = api.ask(sessionId, text, {
        onStart: (srcs) => setSources(srcs),
        onDelta: (t) =>
          patch((m) => ({ ...m, pending: false, content: m.content + t })),
        onDone: ({ answer, citations }) => {
          patch((m) => ({ ...m, pending: false, content: answer, citations }))
          refreshSessions()
        },
        onError: (msg) =>
          patch((m) => ({
            ...m,
            pending: false,
            error: true,
            content: m.content || msg,
          })),
      })
    },
    [refreshSessions],
  )

  const stopStreaming = useCallback(() => abortRef.current?.(), [])

  // autoscroll: follow new tokens only when the user is already near the bottom
  useEffect(() => {
    const el = scrollRef.current
    if (el && pinnedRef.current) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
    }
  }, [messages])

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }, [])

  return (
    <div className="flex h-full bg-void text-slate-200">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        onNew={newChat}
        onOpen={openSession}
        onDelete={deleteSession}
      />

      <main className="relative flex min-w-0 flex-1 flex-col">
        {/* ambient background */}
        <div className="pointer-events-none absolute inset-0 bg-grid" />
        <div className="pointer-events-none absolute -top-40 left-1/2 h-96 w-[42rem] -translate-x-1/2 rounded-full bg-cyan-500/10 blur-[120px]" />
        <div className="pointer-events-none absolute bottom-0 right-0 h-80 w-96 rounded-full bg-violet-500/10 blur-[120px]" />

        <ApiWarning configured={health?.llm_configured ?? false} />
        <StatusBar
          connection={connection}
          session={activeSession}
          health={health}
          onRetry={refreshSessions}
        />

        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="relative z-10 flex-1 overflow-y-auto"
        >
          {activeSession ? (
            <div className="mx-auto w-full max-w-3xl px-6 py-8">
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}
            </div>
          ) : (
            <EmptyState onPick={sendMessage} />
          )}
        </div>

        <SourcePanel sources={sources} />

        <Composer
          disabled={!!activeSession && messages.some((m) => m.pending)}
          hasSession={!!activeSession}
          onSend={sendMessage}
          onStop={stopStreaming}
          onUploaded={handleUploaded}
        />
      </main>
    </div>
  )
}
