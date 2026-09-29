import { computed, type Ref } from 'vue'
import type { MetricSeries } from '@/api/events'

export interface MetricGroup {
  prefix: string
  names: string[]
}

/** Metric names grouped by their prefix (`train/`, `eval/`, `agent/`…). */
export function groupMetricNames(names: readonly string[]): MetricGroup[] {
  const groups = new Map<string, string[]>()
  for (const name of names) {
    const slash = name.indexOf('/')
    const prefix = slash > 0 ? name.slice(0, slash) : 'other'
    const list = groups.get(prefix) ?? []
    list.push(name)
    groups.set(prefix, list)
  }
  return [...groups]
    .map(([prefix, list]) => ({ prefix, names: list.sort() }))
    .sort((a, b) => a.prefix.localeCompare(b.prefix))
}

/** The metric names a series holds, recomputed when it grows. */
export function useMetricNames(series: MetricSeries, revision: Ref<number>) {
  return computed(() => {
    void revision.value
    return series.names()
  })
}
