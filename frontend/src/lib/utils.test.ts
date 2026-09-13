import { describe, it, expect } from 'vitest'
import { uid, formatBytes, timeAgo } from './utils'

describe('uid', () => {
  it('produces unique, non-empty ids', () => {
    const ids = new Set(Array.from({ length: 200 }, () => uid()))
    expect(ids.size).toBe(200)
    for (const id of ids) expect(id.length).toBeGreaterThan(0)
  })
})

describe('formatBytes', () => {
  it('formats bytes, KB and MB', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })
})

describe('timeAgo', () => {
  it('covers all buckets', () => {
    const now = Math.floor(Date.now() / 1000)
    expect(timeAgo(now - 5)).toBe('just now')
    expect(timeAgo(now - 300)).toBe('5m ago')
    expect(timeAgo(now - 7200)).toBe('2h ago')
    expect(timeAgo(now - 3 * 86400)).toBe('3d ago')
  })
})
