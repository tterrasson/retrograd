import { computed, onScopeDispose, ref, toValue, type MaybeRefOrGetter } from 'vue'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import type { QueryClient } from '@tanstack/vue-query'
import { client, settle, unwrap } from './client'
import { keys, type RunFilters } from './keys'
import { connectSse, type SseState } from './sse'
import type {
  CancelRequest,
  CommandAccepted,
  GenerateRequest,
  PatchRequest,
  EventOf,
  RunEvent,
  RunListing,
  RunStatus,
  RunSummary,
  RunView,
  TaggedRunEvent,
} from './types'

/** How a status reads on screen. Display only: no action is refused from it. */
export interface StatusInfo {
  label: string
  color: string
  /** The run will not change again. */
  terminal: boolean
  /** Something is happening right now (spinner-worthy). */
  busy: boolean
}

const STATUS: Record<RunStatus, StatusInfo> = {
  queued: { label: 'queued', color: 'info', terminal: false, busy: false },
  resolving: { label: 'resolving', color: 'info', terminal: false, busy: true },
  starting: { label: 'starting', color: 'info', terminal: false, busy: true },
  running: { label: 'running', color: 'primary', terminal: false, busy: true },
  paused: { label: 'paused', color: 'warning', terminal: false, busy: false },
  pausing: { label: 'pausing', color: 'warning', terminal: false, busy: true },
  cancelling: { label: 'cancelling', color: 'warning', terminal: false, busy: true },
  completed: { label: 'completed', color: 'success', terminal: true, busy: false },
  failed: { label: 'failed', color: 'error', terminal: true, busy: false },
  cancelled: { label: 'cancelled', color: 'secondary', terminal: true, busy: false },
  interrupted: { label: 'interrupted', color: 'error', terminal: true, busy: false },
}

export function statusInfo(status: RunStatus | null | undefined): StatusInfo {
  return (
    (status && STATUS[status]) || {
      label: status ?? 'unknown',
      color: 'default',
      terminal: false,
      busy: false,
    }
  )
}

/** The command buttons a status shows. The server still decides each one. */
export interface CommandSet {
  pause: boolean
  resume: boolean
  cancel: boolean
  checkpoint: boolean
  evaluate: boolean
  remove: boolean
}

export function commandsFor(status: RunStatus | null | undefined): CommandSet {
  const info = statusInfo(status)
  const live = status === 'running' || status === 'paused' || status === 'pausing'
  return {
    pause: status === 'running',
    resume: status === 'paused',
    cancel: !info.terminal && status !== 'cancelling',
    checkpoint: live,
    evaluate: live,
    remove: info.terminal,
  }
}

export function shortId(id: string): string {
  return id.length > 8 ? id.slice(0, 8) : id
}

export function runLabel(run: Pick<RunSummary, 'id' | 'name'>): string {
  return run.name || shortId(run.id)
}

export function useRuns(filters: MaybeRefOrGetter<RunFilters>) {
  return useQuery({
    queryKey: computed(() => keys.runs(toValue(filters))),
    queryFn: ({ signal }) => {
      const { status, algorithm, cursor, limit } = toValue(filters)
      return unwrap(
        client.GET('/v1/runs', {
          params: {
            query: {
              status: (status || undefined) as RunStatus | undefined,
              algorithm: algorithm || undefined,
              cursor: cursor || undefined,
              limit,
            },
          },
          signal,
        }),
      )
    },
    placeholderData: keepPreviousData,
  })
}

export function useRun(id: MaybeRefOrGetter<string>) {
  return useQuery({
    queryKey: computed(() => keys.run(toValue(id))),
    queryFn: ({ signal }) =>
      unwrap(client.GET('/v1/runs/{id}', { params: { path: { id: toValue(id) } }, signal })),
  })
}

export function useCheckpoints(id: MaybeRefOrGetter<string>) {
  return useQuery({
    queryKey: computed(() => keys.checkpoints(toValue(id))),
    queryFn: ({ signal }) =>
      unwrap(
        client.GET('/v1/runs/{id}/checkpoints', { params: { path: { id: toValue(id) } }, signal }),
      ),
  })
}

/** Writes what a command answered into the cached run, then lets the lists refetch. */
function applyAccepted(queryClient: QueryClient, accepted: CommandAccepted) {
  queryClient.setQueryData<RunView>(keys.run(accepted.id), (old) =>
    old ? { ...old, status: accepted.status } : old,
  )
  patchListRows(queryClient, accepted.id, (row) => ({ ...row, status: accepted.status }))
  void queryClient.invalidateQueries({ queryKey: keys.runsAll() })
}

export function useRunCommands(id: MaybeRefOrGetter<string>) {
  const queryClient = useQueryClient()
  const path = () => ({ params: { path: { id: toValue(id) } } })
  const onSuccess = (accepted: CommandAccepted) => applyAccepted(queryClient, accepted)

  const pause = useMutation({
    mutationFn: () => unwrap(client.POST('/v1/runs/{id}/pause', path())),
    onSuccess,
  })
  const resume = useMutation({
    mutationFn: () => unwrap(client.POST('/v1/runs/{id}/resume', path())),
    onSuccess,
  })
  const cancel = useMutation({
    mutationFn: (body: CancelRequest) =>
      unwrap(client.POST('/v1/runs/{id}/cancel', { ...path(), body })),
    onSuccess,
  })
  const checkpoint = useMutation({
    mutationFn: () => unwrap(client.POST('/v1/runs/{id}/checkpoints', path())),
    onSuccess: (accepted: CommandAccepted) => {
      onSuccess(accepted)
      void queryClient.invalidateQueries({ queryKey: keys.checkpoints(toValue(id)) })
    },
  })
  const patch = useMutation({
    mutationFn: (body: PatchRequest) => unwrap(client.PATCH('/v1/runs/{id}', { ...path(), body })),
    onSuccess,
  })
  const remove = useMutation({
    mutationFn: () => settle(client.DELETE('/v1/runs/{id}', path())),
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: keys.run(toValue(id)) })
      void queryClient.invalidateQueries({ queryKey: keys.runsAll() })
    },
  })
  const evaluate = useMutation({
    mutationFn: () => unwrap(client.POST('/v1/runs/{id}/evaluate', { ...path(), body: {} })),
  })
  const generate = useMutation({
    mutationFn: (body: GenerateRequest) =>
      unwrap(client.POST('/v1/runs/{id}/generate', { ...path(), body })),
  })
  return { pause, resume, cancel, checkpoint, patch, remove, evaluate, generate }
}

function patchListRows(
  queryClient: QueryClient,
  id: string,
  patch: (row: RunSummary) => RunSummary,
): boolean {
  let found = false
  queryClient.setQueriesData<RunListing>({ queryKey: keys.runsAll() }, (old) => {
    if (!old) return old
    let changed = false
    const runs = old.runs.map((row) => {
      if (row.id !== id) return row
      changed = true
      return patch(row)
    })
    if (changed) found = true
    return changed ? { ...old, runs } : old
  })
  return found
}

/** Applies one event of a run to every cached copy of that run. */
export function applyRunEvent(queryClient: QueryClient, id: string, event: RunEvent): boolean {
  switch (event.type) {
    case 'status':
    case 'terminal': {
      const status = event.status
      const error = event.type === 'terminal' ? event.error : undefined
      queryClient.setQueryData<RunView>(keys.run(id), (old) =>
        old ? { ...old, status, ...(error !== undefined ? { error } : {}) } : old,
      )
      return patchListRows(queryClient, id, (row) => ({
        ...row,
        status,
        ...(error !== undefined ? { error } : {}),
      }))
    }
    case 'progress': {
      const {
        type: _type,
        seq: _seq,
        at: _at,
        run: _run,
        ...progress
      } = event as EventOf<'progress'> & {
        run?: string
      }
      queryClient.setQueryData<RunView>(keys.run(id), (old) => (old ? { ...old, progress } : old))
      return patchListRows(queryClient, id, (row) => ({ ...row, progress }))
    }
    default:
      return true
  }
}

/**
 * Keeps every cached run list current from `GET /v1/events` while the calling
 * component is mounted: rows are patched in place, and an event from a run no
 * list knows about refetches the lists.
 */
export function useLiveRunList() {
  const queryClient = useQueryClient()
  const state = ref<SseState>('connecting')
  let pending: ReturnType<typeof setTimeout> | undefined
  const refetchSoon = () => {
    if (pending) return
    pending = setTimeout(() => {
      pending = undefined
      void queryClient.invalidateQueries({ queryKey: keys.runsAll() })
    }, 500)
  }
  const connection = connectSse({
    url: '/v1/events',
    onState: (next) => (state.value = next),
    onMessage(message) {
      if (message.event === 'lagged') {
        refetchSoon()
        return
      }
      if (!['status', 'progress', 'terminal'].includes(message.event)) return
      let event: TaggedRunEvent
      try {
        event = JSON.parse(message.data) as TaggedRunEvent
      } catch {
        return
      }
      const known = applyRunEvent(queryClient, event.run, event)
      if (!known || event.type === 'terminal') refetchSoon()
    },
  })
  onScopeDispose(() => {
    connection.close()
    if (pending) clearTimeout(pending)
  })
  return { state }
}
