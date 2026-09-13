import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import Sidebar from './Sidebar'
import type { SessionSummary } from '../lib/api'

const sessions: SessionSummary[] = [
  { session_id: 's1', title: 'Test PDF', pages: 5, chunks: 10, created_at: Date.now() / 1000 - 60, message_count: 4 },
  { session_id: 's2', title: 'Another Doc', pages: 3, chunks: 5, created_at: Date.now() / 1000 - 3600, message_count: 0 },
]

const base = {
  activeId: null,
  onNew: vi.fn(),
  onOpen: vi.fn(),
  onDelete: vi.fn(),
  onClear: vi.fn(),
}

describe('Sidebar', () => {
  it('renders session titles', () => {
    render(<Sidebar {...base} sessions={sessions} />)
    expect(screen.getByText('Test PDF')).toBeInTheDocument()
    expect(screen.getByText('Another Doc')).toBeInTheDocument()
  })

  it('shows document count in footer', () => {
    render(<Sidebar {...base} sessions={sessions} />)
    expect(screen.getByText(/2 documents stored/)).toBeInTheDocument()
  })

  it('calls onOpen when session clicked', () => {
    render(<Sidebar {...base} sessions={sessions} />)
    screen.getByText('Test PDF').click()
    expect(base.onOpen).toHaveBeenCalledWith('s1')
  })

  it('calls onNew when New Chat clicked', () => {
    render(<Sidebar {...base} sessions={sessions} />)
    screen.getByText('New Chat').click()
    expect(base.onNew).toHaveBeenCalledTimes(1)
  })

  it('shows empty state when no sessions', () => {
    render(<Sidebar {...base} sessions={[]} />)
    expect(screen.getByText('No documents yet.')).toBeInTheDocument()
    expect(screen.getByText(/0 documents stored/)).toBeInTheDocument()
  })
})
