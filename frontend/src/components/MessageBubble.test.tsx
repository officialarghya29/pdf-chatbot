import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import MessageBubble from './MessageBubble'
import type { ChatMessage } from '../lib/api'

describe('MessageBubble', () => {
  it('renders user messages right-aligned bubble with content', () => {
    const m: ChatMessage = { id: '1', role: 'user', content: 'What is flux?' }
    render(<MessageBubble message={m} />)
    expect(screen.getByText('What is flux?')).toBeInTheDocument()
  })

  it('renders assistant markdown including bold and code', () => {
    const m: ChatMessage = {
      id: '2',
      role: 'assistant',
      content: 'The **flux** is `42.0`',
    }
    const { container } = render(<MessageBubble message={m} />)
    expect(screen.getByText('flux')).toBeInTheDocument()
    expect(container.querySelector('strong')).toBeInTheDocument()
    expect(container.querySelector('code')).toBeInTheDocument()
  })

  it('shows citation page chips after done', () => {
    const m: ChatMessage = {
      id: '3',
      role: 'assistant',
      content: 'Answer [1].',
      citations: [
        { n: 1, page: 4 },
        { n: 2, page: 7 },
      ],
    }
    render(<MessageBubble message={m} />)
    expect(screen.getByTitle('Cited from page 4')).toBeInTheDocument()
    expect(screen.getByTitle('Cited from page 7')).toBeInTheDocument()
  })

  it('styles error messages with the alert icon and error box', () => {
    const m: ChatMessage = {
      id: '4',
      role: 'assistant',
      content: 'Provider exploded',
      error: true,
    }
    const { container } = render(<MessageBubble message={m} />)
    expect(container.querySelector('.border-red-500\\/30')).not.toBeNull()
    // error avatar must not show the normal bot icon
    expect(container.querySelector('svg.lucide-bot')).toBeNull()
  })

  it('shows the thinking dots while pending with no content', () => {
    const m: ChatMessage = { id: '5', role: 'assistant', content: '', pending: true }
    render(<MessageBubble message={m} />)
    expect(screen.getByText('Reading the document…')).toBeInTheDocument()
  })
})
