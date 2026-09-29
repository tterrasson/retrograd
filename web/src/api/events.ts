import {
  onScopeDispose,
  shallowRef,
  triggerRef,
  type InjectionKey,
  type Ref,
  type ShallowRef,
} from 'vue'
import { useQueryClient, type QueryClient } from '@tanstack/vue-query'
import { keys } from './keys'
import { applyRunEvent, statusInfo } from './runs'
import { connectSse, type SseMessage, type SseState } from './sse'
import type { EventOf, RunEvent, RunProgress, RunStatus } from './types'

/**
 * Metrics as columns, append-only: one array per axis and one per metric name,
 * all the same length, `null` where a sample did not carry a name.
 */
export class MetricSeries {
  readonly seq: number[] = []
  readonly at: number[] = []
  readonly iteration: number[] = []
  readonly globalStep: number[] = []
  readonly values = new Map<string, (number | null)[]>()

  get length(): number {
    return this.seq.length
  }

  names(): string[] {
    return [...this.values.keys()].sort()
  }

  push(sample: {
    seq: number
    at: number
    iteration: number
    global_step: number
    values: Record<string, number>
  }): void {
    // Samples arrive in `seq` order; one already held (a backfill that
    // overlapped the stream) is not appended twice.
    const last = this.seq[this.seq.length - 1]
    if (last !== undefined && sample.seq <= last) {
      this.insert(sample)
      return
    }
    const index = this.seq.length
    this.seq.push(sample.seq)
    this.at.push(sample.at)
    this.iteration.push(sample.iteration)
    this.globalStep.push(sample.global_step)
    for (const [name, value] of Object.entries(sample.values)) {
      let column = this.values.get(name)
      if (!column) {
        column = new Array<number | null>(index).fill(null)
        this.values.set(name, column)
      }
      column.push(Number.isFinite(value) ? value : null)
    }
    for (const column of this.values.values()) {
      if (column.length < index + 1) column.push(null)
    }
  }

  /** A sample older than the newest one: placed by `seq`, skipped if present. */
  private insert(sample: Parameters<MetricSeries['push']>[0]): void {
    let low = 0
    let high = this.seq.length
    while (low < high) {
      const mid = (low + high) >> 1
      if ((this.seq[mid] ?? 0) < sample.seq) low = mid + 1
      else high = mid
    }
    if (this.seq[low] === sample.seq) return
    this.seq.splice(low, 0, sample.seq)
    this.at.splice(low, 0, sample.at)
    this.iteration.splice(low, 0, sample.iteration)
    this.globalStep.splice(low, 0, sample.global_step)
    const length = this.seq.length
    for (const name of Object.keys(sample.values)) {
      if (!this.values.has(name))
        this.values.set(name, new Array<number | null>(length - 1).fill(null))
    }
    for (const [name, column] of this.values) {
      const value = sample.values[name]
      column.splice(low, 0, value !== undefined && Number.isFinite(value) ? value : null)
    }
  }

  /** The last finite value of a metric. */
  latest(name: string): number | null {
    const column = this.values.get(name)
    if (!column) return null
    for (let index = column.length - 1; index >= 0; index--) {
      const value = column[index]
      if (value !== null && value !== undefined) return value
    }
    return null
  }
}

/** A bounded journal: the oldest line leaves when a new one arrives past the cap. */
export class RingBuffer<T> {
  private items: T[] = []
  private dropped = 0
  constructor(readonly capacity: number) {}

  push(item: T): void {
    this.items.push(item)
    if (this.items.length > this.capacity) {
      const excess = this.items.length - this.capacity
      this.items.splice(0, excess)
      this.dropped += excess
    }
  }

  toArray(): readonly T[] {
    return this.items
  }

  get size(): number {
    return this.items.length
  }

  get droppedCount(): number {
    return this.dropped
  }
}

export type JournalEvent = Exclude<RunEvent, { type: 'metrics' } | { type: 'progress' }>

export interface Marker {
  kind: 'checkpoint' | 'evaluation'
  seq: number
  iteration: number
  globalStep: number
  label: string
}

export interface RunEvents {
  status: ShallowRef<RunStatus | null>
  progress: ShallowRef<RunProgress | null>
  /** Mutated in place; `revision` changes whenever it does. */
  series: MetricSeries
  journal: RingBuffer<JournalEvent>
  markers: ShallowRef<Marker[]>
  evaluations: ShallowRef<EventOf<'evaluation'>[]>
  checkpoints: ShallowRef<EventOf<'checkpoint'>[]>
  error: ShallowRef<string | null>
  connection: ShallowRef<SseState>
  lastSeq: ShallowRef<number>
  /** Bumped (at most once per frame) when `series` or `journal` changed. */
  revision: ShallowRef<number>
  /** A gap was reported and reloaded from the metrics route. */
  lagged: ShallowRef<number>
}

export const RunEventsKey: InjectionKey<Ref<RunEvents | null>> = Symbol('run-events')

export const JOURNAL_CAPACITY = 5_000

/** The state a run's event stream builds, independent of any connection. */
export function createRunEventState(): RunEvents {
  return {
    status: shallowRef(null),
    progress: shallowRef(null),
    series: new MetricSeries(),
    journal: new RingBuffer<JournalEvent>(JOURNAL_CAPACITY),
    markers: shallowRef([]),
    evaluations: shallowRef([]),
    checkpoints: shallowRef([]),
    error: shallowRef(null),
    connection: shallowRef('connecting'),
    lastSeq: shallowRef(0),
    revision: shallowRef(0),
    lagged: shallowRef(0),
  }
}

/**
 * Folds one event into the state. Answers `false` for an event already seen
 * (its `seq` is not above the last one), which a reconnect can deliver.
 */
export function reduceRunEvent(state: RunEvents, event: RunEvent): boolean {
  if (event.seq <= state.lastSeq.value) return false
  state.lastSeq.value = event.seq
  const lastIteration = state.series.iteration[state.series.length - 1] ?? 0
  const lastStep = state.series.globalStep[state.series.length - 1] ?? 0
  switch (event.type) {
    case 'metrics':
      state.series.push(event)
      return true
    case 'progress': {
      const { type: _type, seq: _seq, at: _at, ...progress } = event
      state.progress.value = progress
      return true
    }
    case 'status':
      state.status.value = event.status
      break
    case 'terminal':
      state.status.value = event.status
      state.error.value = event.error ?? null
      break
    case 'checkpoint':
      state.checkpoints.value = [...state.checkpoints.value, event]
      state.markers.value = [
        ...state.markers.value,
        {
          kind: 'checkpoint',
          seq: event.seq,
          iteration: lastIteration,
          globalStep: lastStep,
          label: event.path.split(/[\\/]/).pop() ?? event.path,
        },
      ]
      break
    case 'evaluation':
      state.evaluations.value = [...state.evaluations.value, event]
      state.markers.value = [
        ...state.markers.value,
        {
          kind: 'evaluation',
          seq: event.seq,
          iteration: event.iteration,
          globalStep: lastStep,
          label: `eval ${event.iteration}`,
        },
      ]
      break
    default:
      break
  }
  state.journal.push(event)
  return true
}

export function parseRunEvent(message: SseMessage): RunEvent | null {
  try {
    const value = JSON.parse(message.data) as RunEvent
    return typeof value === 'object' && value !== null && typeof value.seq === 'number'
      ? value
      : null
  } catch {
    return null
  }
}

/**
 * The event stream of one run, from its first event: a finished run is
 * replayed from its journal, a live one continues from there, so there is one
 * code path for both. Mirrors status and progress into the query cache.
 */
export function useRunEvents(
  id: string,
  options: { observed?: () => boolean; queryClient?: QueryClient } = {},
): RunEvents {
  const queryClient = options.queryClient ?? useQueryClient()
  const state = createRunEventState()

  let frame: number | undefined
  const bump = () => {
    if (frame !== undefined) return
    const schedule =
      typeof requestAnimationFrame === 'function'
        ? requestAnimationFrame
        : (callback: () => void) => setTimeout(callback, 16) as unknown as number
    frame = schedule(() => {
      frame = undefined
      state.revision.value++
      triggerRef(state.markers)
    })
  }

  let trajectoryTimer: ReturnType<typeof setTimeout> | undefined
  const refreshTrajectories = () => {
    if (trajectoryTimer) return
    trajectoryTimer = setTimeout(() => {
      trajectoryTimer = undefined
      void queryClient.invalidateQueries({ queryKey: keys.trajectoryUpdates(id) })
    }, 2_000)
  }

  const connection = connectSse({
    url: `/v1/runs/${encodeURIComponent(id)}/events?since=0`,
    onState: (next) => (state.connection.value = next),
    onFatal: (problem) => (state.error.value = problem.detail || problem.title),
    shouldReconnect: () => !statusInfo(state.status.value).terminal,
    onMessage(message) {
      if (message.event === 'lagged') {
        state.lagged.value++
        // The per-run stream replays every event type from Last-Event-ID,
        // including status, journal entries and checkpoints.
        return 'reconnect'
      }
      const event = parseRunEvent(message)
      if (!event || !reduceRunEvent(state, event)) return
      switch (event.type) {
        case 'status':
        case 'terminal':
        case 'progress':
          applyRunEvent(queryClient, id, event)
          if (event.type === 'terminal') {
            void queryClient.invalidateQueries({ queryKey: keys.run(id) })
            void queryClient.invalidateQueries({ queryKey: keys.artifacts(id) })
          }
          break
        case 'checkpoint':
          void queryClient.invalidateQueries({ queryKey: keys.checkpoints(id) })
          break
        case 'metrics':
          if (options.observed?.()) refreshTrajectories()
          break
        default:
          break
      }
      bump()
    },
  })

  onScopeDispose(() => {
    connection.close()
    if (trajectoryTimer) clearTimeout(trajectoryTimer)
    if (frame !== undefined && typeof cancelAnimationFrame === 'function')
      cancelAnimationFrame(frame)
  })

  return state
}
