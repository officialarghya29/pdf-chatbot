import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { api } from './api'

// Helper: build a fetch Response whose body is a ReadableStream of SSE text
function sseResponse(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c))
      controller.close()
    },
  })
  return new Response(body, {
    status,
    headers: { 'content-type': 'text/event-stream' },
  }) as Response
}

// Build an SSE "data: {...}\n\n" frame from a JS object
function frame(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`
}

const okJson = (obj: unknown, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })

describe('api.ask SSE parsing', () => {
  const sessionId = 'session123'
  const handlers = () => ({
    onStart: vi.fn(),
    onDelta: vi.fn(),
    onDone: vi.fn(),
    onError: vi.fn(),
  })

  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('dispatches start then deltas then done', async () => {
    const h = handlers()
    vi.mocked(fetch).mockResolvedValue(
      sseResponse([
        frame({ type: 'start', sources: [{ page: 3, snippet: 'snip' }] }),
        frame({ type: 'delta', text: 'Hel' }),
        frame({ type: 'delta', text: 'lo world' }),
        frame({ type: 'done', answer: 'Hello world', citations: [{ n: 1, page: 3 }] }),
      ]),
    )

    api.ask(sessionId, 'q', h)
    await vi.waitFor(() => expect(h.onDone).toHaveBeenCalledTimes(1))

    expect(h.onStart).toHaveBeenCalledWith([{ page: 3, snippet: 'snip' }])
    expect(h.onDelta).toHaveBeenNthCalledWith(1, 'Hel')
    expect(h.onDelta).toHaveBeenNthCalledWith(2, 'lo world')
    expect(h.onDone).toHaveBeenCalledWith({
      answer: 'Hello world',
      citations: [{ n: 1, page: 3 }],
    })
    expect(h.onError).not.toHaveBeenCalled()
  })

  it('dispatches events split across TCP chunk boundaries', async () => {
    const h = handlers()
    const full = frame({ type: 'delta', text: 'split payload' }) + frame({ type: 'done', answer: 'split payload', citations: [] })
    const mid = Math.floor(full.length / 2)
    vi.mocked(fetch).mockResolvedValue(sseResponse([full.slice(0, mid), full.slice(mid)]))

    api.ask(sessionId, 'q', h)
    await vi.waitFor(() => expect(h.onDone).toHaveBeenCalledTimes(1))
    expect(h.onDelta).toHaveBeenCalledWith('split payload')
  })

  it('handles text containing newlines and quotes inside JSON', async () => {
    const h = handlers()
    vi.mocked(fetch).mockResolvedValue(
      sseResponse([
        frame({ type: 'delta', text: 'line1\n"quoted" }{' }),
        frame({ type: 'done', answer: 'x', citations: [] }),
      ]),
    )

    api.ask(sessionId, 'q', h)
    await vi.waitFor(() => expect(h.onDone).toHaveBeenCalledTimes(1))
    expect(h.onDelta).toHaveBeenCalledWith('line1\n"quoted" }{')
  })

  it('emits onError when stream ends without a done event', async () => {
    const h = handlers()
    vi.mocked(fetch).mockResolvedValue(
      sseResponse([frame({ type: 'start' }), frame({ type: 'delta', text: 'partial' })]),
    )

    api.ask(sessionId, 'q', h)
    await vi.waitFor(() => expect(h.onError).toHaveBeenCalledTimes(1))
    expect(h.onError).toHaveBeenCalledWith('The response stream ended unexpectedly. Please try again.')
    expect(h.onDone).not.toHaveBeenCalled()
  })

  it('skips malformed events without crashing and still processes the rest', async () => {
    const h = handlers()
    vi.mocked(fetch).mockResolvedValue(
      sseResponse(['data: {broken json\n\n', frame({ type: 'delta', text: 'ok' }), frame({ type: 'done', answer: 'ok', citations: [] })]),
    )

    api.ask(sessionId, 'q', h)
    await vi.waitFor(() => expect(h.onDone).toHaveBeenCalledTimes(1))
    expect(h.onDelta).toHaveBeenCalledWith('ok')
  })

  it('reports HTTP errors from the endpoint', async () => {
    const h = handlers()
    vi.mocked(fetch).mockResolvedValue(okJson({ detail: 'Session not found.' }, 404))

    api.ask(sessionId, 'q', h)
    await vi.waitFor(() => expect(h.onError).toHaveBeenCalledWith('Session not found.'))
  })

  it('reports network failures', async () => {
    const h = handlers()
    vi.mocked(fetch).mockRejectedValue(new TypeError('Failed to fetch'))

    api.ask(sessionId, 'q', h)
    await vi.waitFor(() => expect(h.onError).toHaveBeenCalledWith('Failed to fetch'))
  })
})
