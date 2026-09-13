import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import Composer from './Composer'

const base = {
  hasSession: true,
  onSend: vi.fn(),
  onStop: vi.fn(),
  onUploaded: vi.fn(),
}

describe('Composer', () => {
  it('sends trimmed text, clears it, and resets textarea height', () => {
    const onSend = vi.fn()
    render(<Composer {...base} disabled={false} onSend={onSend} />)

    const ta = screen.getByPlaceholderText('Ask anything about your document…') as HTMLTextAreaElement
    ta.style.height = '120px' // simulate auto-grow
    fireEvent.change(ta, { target: { value: '  hello world  ' } })
    fireEvent.keyDown(ta, { key: 'Enter' })

    expect(onSend).toHaveBeenCalledWith('hello world')
    expect(ta.value).toBe('')
    expect(ta.style.height).toBe('auto')
  })

  it('Shift+Enter inserts a newline instead of sending', () => {
    const onSend = vi.fn()
    render(<Composer {...base} disabled={false} onSend={onSend} />)

    const ta = screen.getByPlaceholderText('Ask anything about your document…')
    fireEvent.change(ta, { target: { value: 'line' } })
    fireEvent.keyDown(ta, { key: 'Enter', shiftKey: true })

    expect(onSend).not.toHaveBeenCalled()
  })

  it('shows the stop button while streaming and calls onStop', () => {
    const onStop = vi.fn()
    render(<Composer {...base} disabled onStop={onStop} />)

    const stop = screen.getByTitle('Stop generating')
    expect(stop).toBeInTheDocument()
    fireEvent.click(stop)
    expect(onStop).toHaveBeenCalledTimes(1)
    expect(screen.queryByTitle('Send message')).not.toBeInTheDocument()
  })

  it('send button is disabled when input is empty', () => {
    render(<Composer {...base} disabled={false} />)
    expect(screen.getByTitle('Send message')).toBeDisabled()
  })
})
