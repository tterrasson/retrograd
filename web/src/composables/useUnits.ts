import {
  formatBytes,
  formatDate,
  formatDuration,
  formatNumber,
  formatPercent,
  formatRate,
  formatRelative,
} from '@/utils/format'

/** The formatters, as one object for templates. */
export function useUnits() {
  return {
    bytes: formatBytes,
    number: formatNumber,
    percent: formatPercent,
    duration: formatDuration,
    rate: formatRate,
    date: formatDate,
    relative: formatRelative,
  }
}
