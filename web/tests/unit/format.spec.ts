import { describe, expect, it } from 'vitest'
import {
  formatBytes,
  formatDuration,
  formatNumber,
  formatPercent,
  formatRelative,
  formatValue,
} from '@/utils/format'
import { ema } from '@/utils/ema'

describe('format', () => {
  it('prints bytes in binary units', () => {
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(1536)).toBe('1.5 KiB')
    expect(formatBytes(3 * 1024 ** 3)).toBe('3.0 GiB')
    expect(formatBytes(null)).toBe('–')
  })

  it('prints numbers, small and large', () => {
    expect(formatNumber(12345)).toBe('12,345')
    expect(formatNumber(0.12345, 3)).toBe('0.123')
    expect(formatNumber(0.00001234)).toBe('1.23e-5')
    expect(formatNumber(undefined)).toBe('–')
    expect(formatPercent(0.125)).toBe('12.5 %')
  })

  it('prints durations and relative times', () => {
    expect(formatDuration(42)).toBe('42 s')
    expect(formatDuration(125)).toBe('2 min 05 s')
    expect(formatDuration(3720)).toBe('1 h 02 min')
    expect(formatRelative(1_000_000 - 120, 1_000_000 * 1000)).toBe('2 minutes ago')
  })

  it('prints configuration values on one line', () => {
    expect(formatValue(undefined)).toBe('–')
    expect(formatValue(null)).toBe('null')
    expect(formatValue(0.0002)).toBe('0.0002')
    expect(formatValue(['q', 'v'])).toBe('["q","v"]')
  })
})

describe('ema', () => {
  it('returns the input with no smoothing and keeps gaps', () => {
    expect(ema([1, null, 3], 0)).toEqual([1, null, 3])
    const smoothed = ema([1, null, 1, 1], 0.5)
    expect(smoothed[1]).toBeNull()
    expect(smoothed[0]).toBeCloseTo(1)
    expect(smoothed[3]).toBeCloseTo(1)
  })
})
