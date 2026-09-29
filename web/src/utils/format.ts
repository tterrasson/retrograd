// Formatting only: every number shown here was computed by the server.

const BYTE_UNITS = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']

export function formatBytes(bytes: number | null | undefined, digits = 1): string {
  if (bytes === null || bytes === undefined || !Number.isFinite(bytes)) return '–'
  let value = Math.abs(bytes)
  let unit = 0
  while (value >= 1024 && unit < BYTE_UNITS.length - 1) {
    value /= 1024
    unit++
  }
  const sign = bytes < 0 ? '-' : ''
  return `${sign}${unit === 0 ? value : value.toFixed(digits)} ${BYTE_UNITS[unit]}`
}

/** Integers as they are, others with `digits` decimals; `–` for nothing. */
export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  if (!Number.isFinite(value)) return value > 0 ? '∞' : '-∞'
  if (Number.isInteger(value)) return value.toLocaleString('en-US')
  const magnitude = Math.abs(value)
  if (magnitude !== 0 && (magnitude < 1e-3 || magnitude >= 1e6)) return value.toExponential(2)
  return value.toFixed(digits)
}

export function formatPercent(fraction: number | null | undefined, digits = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return '–'
  return `${(fraction * 100).toFixed(digits)} %`
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '–'
  const total = Math.max(0, Math.round(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h) return `${h} h ${String(m).padStart(2, '0')} min`
  if (m) return `${m} min ${String(s).padStart(2, '0')} s`
  return `${s} s`
}

export function formatRate(perSecond: number | null | undefined, unit = 'tok/s'): string {
  if (perSecond === null || perSecond === undefined || !Number.isFinite(perSecond)) return '–'
  return `${perSecond >= 100 ? Math.round(perSecond).toLocaleString('en-US') : perSecond.toFixed(1)} ${unit}`
}

/** A Unix time in seconds as a local date and time. */
export function formatDate(unixSeconds: number | null | undefined): string {
  if (!unixSeconds) return '–'
  return new Date(unixSeconds * 1000).toLocaleString()
}

/** "3 min ago", "in 2 h". `now` is injectable for tests. */
export function formatRelative(unixSeconds: number | null | undefined, now = Date.now()): string {
  if (!unixSeconds) return '–'
  const delta = Math.round(unixSeconds - now / 1000)
  const magnitude = Math.abs(delta)
  const units: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, 'second'],
    [3600, 'minute'],
    [86_400, 'hour'],
    [604_800, 'day'],
    [2_629_800, 'week'],
    [31_557_600, 'month'],
    [Infinity, 'year'],
  ]
  const divisors: Record<string, number> = {
    second: 1,
    minute: 60,
    hour: 3600,
    day: 86_400,
    week: 604_800,
    month: 2_629_800,
    year: 31_557_600,
  }
  const format = new Intl.RelativeTimeFormat('en', { numeric: 'auto' })
  for (const [limit, unit] of units) {
    if (magnitude < limit) return format.format(Math.round(delta / (divisors[unit] ?? 1)), unit)
  }
  return formatDate(unixSeconds)
}

/** A JSON value on one line, for table cells. */
export function formatValue(value: unknown): string {
  if (value === undefined) return '–'
  if (value === null) return 'null'
  if (typeof value === 'number') return String(value)
  if (typeof value === 'string') return value
  return JSON.stringify(value)
}

export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count.toLocaleString('en-US')} ${count === 1 ? singular : plural}`
}
