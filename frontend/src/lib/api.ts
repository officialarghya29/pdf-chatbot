export interface Citation {
  n: number
  page: number
}

export interface Source {
  page: number
  snippet: string
}

export interface SessionSummary {
  session_id: string
  title: string
  pages: number
  chunks: number
  created_at: number
  message_count: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  sources?: Source[]
  error?: boolean
  pending?: boolean
}

export interface Health {
  status: string
  version: string
  llm_configured: boolean
  sessions: number
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch {
      /* keep default */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  async health(): Promise<Health> {
    return handle(await fetch('/api/health'))
  },

  async listSessions(): Promise<SessionSummary[]> {
    return handle(await fetch('/api/sessions'))
  },

  async getSession(id: string): Promise<SessionSummary> {
    return handle(await fetch(`/api/sessions/${id}`))
  },

  async getMessages(id: string): Promise<{ messages: { role: string; content: string }[] }> {
    return handle(await fetch(`/api/sessions/${id}/messages`))
  },

  async deleteSession(id: string): Promise<void> {
    await handle(await fetch(`/api/sessions/${id}`, { method: 'DELETE' }))
  },

  async clearMessages(id: string): Promise<void> {
    await handle(await fetch(`/api/sessions/${id}/messages`, { method: 'DELETE' }))
  },

  async upload(
    file: File,
    onProgress?: (pct: number) => void,
  ): Promise<SessionSummary> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      const form = new FormData()
      form.append('file', file)

      xhr.upload.addEventListener('progress', (e) => {
        if (e.lengthComputable && onProgress) {
          onProgress(Math.round((e.loaded / e.total) * 100))
        }
      })
      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText))
          } catch {
            reject(new Error('Invalid server response'))
          }
        } else {
          let msg = `Upload failed (${xhr.status})`
          try {
            const body = JSON.parse(xhr.responseText)
            if (body?.detail) msg = body.detail
          } catch {
            /* ignore */
          }
          reject(new Error(msg))
        }
      })
      xhr.addEventListener('error', () => reject(new Error('Network error during upload')))
      xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')))

      xhr.open('POST', '/api/upload')
      xhr.send(form)
    })
  },

  /** Streaming chat over SSE. Returns an abort function. */
  ask(
    sessionId: string,
    query: string,
    handlers: {
      onStart?: (sources: Source[]) => void
      onDelta?: (text: string) => void
      onDone?: (result: { answer: string; citations: Citation[] }) => void
      onError?: (message: string) => void
    },
  ): () => void {
    const controller = new AbortController()

    fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, query }),
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok || !res.body) {
          let detail = `Chat failed (${res.status})`
          try {
            const b = await res.json()
            if (b?.detail) detail = b.detail
          } catch {
            /* ignore */
          }
          handlers.onError?.(detail)
          return
        }

        let sawDone = false

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })

          const events = buffer.split('\n\n')
          buffer = events.pop() ?? ''

          for (const evt of events) {
            const line = evt.split('\n').find((l) => l.startsWith('data: '))
            if (!line) continue
            try {
              const payload = JSON.parse(line.slice(6))
              switch (payload.type) {
                case 'start':
                  handlers.onStart?.(payload.sources ?? [])
                  break
                case 'delta':
                  handlers.onDelta?.(payload.text)
                  break
                case 'done':
                  sawDone = true
                  handlers.onDone?.({
                    answer: payload.answer,
                    citations: payload.citations ?? [],
                  })
                  break
                case 'error':
                  handlers.onError?.(payload.detail ?? 'Unknown error')
                  break
              }
            } catch {
              /* malformed event — skip */
            }
          }
        }

        if (!sawDone) {
          // stream ended without a done event (server restart, proxy cut, …)
          handlers.onError?.('The response stream ended unexpectedly. Please try again.')
        }
      })
      .catch((err: unknown) => {
        if ((err as Error).name !== 'AbortError') {
          handlers.onError?.((err as Error).message || 'Network error')
        }
      })

    return () => controller.abort()
  },
}
