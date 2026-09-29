import { onScopeDispose, ref, shallowRef, watch } from 'vue'
import { useMutation, useQueryClient } from '@tanstack/vue-query'
import { toProblem } from './problem'
import { authorizedFetch, client, unwrap } from './client'
import { keys } from './keys'
import type {
  ConfigDocument,
  ForkFrom,
  Limits,
  PlanRequest,
  PlanResponse,
  PreflightResponse,
  Recipe,
  RunView,
} from './types'

/** A partial configuration tree: what the client pins, over what the server resolves. */
export type Params = Record<string, unknown>

/**
 * The four ways to say what to train. The server reads the same three JSON
 * forms (recipe, config, fork) and a TOML document; the UI only chooses which
 * one it sends.
 */
export type PlanInput =
  | { form: 'recipe'; recipe: Recipe; params: Params; name?: string }
  | { form: 'config'; config: ConfigDocument; params: Params; name?: string }
  | { form: 'fork'; fork_from: ForkFrom; params: Params; name?: string }
  | { form: 'toml'; toml: string }

export function planRequest(input: Exclude<PlanInput, { form: 'toml' }>): PlanRequest {
  const request: PlanRequest = {}
  if (input.name) request.name = input.name
  if (Object.keys(input.params).length) request.params = input.params as PlanRequest['params']
  switch (input.form) {
    case 'recipe':
      request.recipe = input.recipe
      break
    case 'config':
      request.config = input.config
      break
    case 'fork':
      request.fork_from = input.fork_from
      break
  }
  return request
}

async function tomlCall<T>(path: string, toml: string, signal?: AbortSignal): Promise<T> {
  const response = await authorizedFetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/toml', Accept: 'application/json' },
    body: toml,
    signal,
  })
  return (await response.json()) as T
}

/** `POST /v1/plan`: resolves, estimates and answers without creating anything. */
export function postPlan(input: PlanInput, signal?: AbortSignal): Promise<PlanResponse> {
  if (input.form === 'toml') return tomlCall<PlanResponse>('/v1/plan', input.toml, signal)
  return unwrap(client.POST('/v1/plan', { body: planRequest(input), signal }))
}

/** `POST /v1/runs` with the same body the plan was computed from. */
export function createRun(input: PlanInput, idempotencyKey: string): Promise<RunView> {
  if (input.form === 'toml') {
    return authorizedFetch('/v1/runs', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/toml',
        Accept: 'application/json',
        'Idempotency-Key': idempotencyKey,
      },
      body: input.toml,
    }).then((response) => response.json() as Promise<RunView>)
  }
  return unwrap(
    client.POST('/v1/runs', {
      body: planRequest(input),
      headers: { 'Idempotency-Key': idempotencyKey },
    }),
  )
}

export function useCreateRun() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ input, key }: { input: PlanInput; key: string }) => createRun(input, key),
    onSuccess: (run) => {
      queryClient.setQueryData(keys.run(run.id), run)
      void queryClient.invalidateQueries({ queryKey: keys.runsAll() })
    },
  })
}

export function usePreflight() {
  return useMutation({
    mutationFn: ({ model, signal }: { model: string; signal?: AbortSignal }) =>
      unwrap(client.POST('/v1/preflight', { body: { model }, signal })) as Promise<
        PreflightResponse & PreflightReport
      >,
  })
}

/**
 * The engine's preflight report, which the contract publishes as an open
 * object. Every field is optional here: the view shows what is present.
 */
export interface PreflightReport {
  schema_version?: number
  profile_fingerprint?: string
  graph_fingerprint?: string
  memory?: Record<string, unknown>
  warnings?: { code: string; message: string }[]
  kernel_summary?: {
    ggml_op: string
    backend: string
    nodes: number
    bytes: number
    implementation: string
  }[]
  placements?: unknown[]
}

/**
 * Replans whenever the input settles for `delay` ms, cancelling the request in
 * flight: the answer shown is always the answer to the latest input.
 */
export function useLivePlan(input: () => PlanInput | null, delay = 800) {
  const data = shallowRef<PlanResponse | null>(null)
  const error = shallowRef<unknown>(null)
  const pending = ref(false)
  let controller: AbortController | null = null
  let timer: ReturnType<typeof setTimeout> | undefined

  async function run(value: PlanInput) {
    controller?.abort()
    const current = new AbortController()
    controller = current
    pending.value = true
    try {
      const answer = await postPlan(value, current.signal)
      if (controller !== current) return
      data.value = answer
      error.value = null
    } catch (cause) {
      if (controller !== current) return
      const problem = toProblem(cause)
      if (problem.type === 'aborted') return
      error.value = problem
    } finally {
      if (controller === current) pending.value = false
    }
  }

  function schedule(immediate = false) {
    if (timer) clearTimeout(timer)
    const value = input()
    if (!value) {
      controller?.abort()
      pending.value = false
      return
    }
    if (immediate) void run(value)
    else timer = setTimeout(() => void run(value), delay)
  }

  watch(
    () => JSON.stringify(input()),
    () => schedule(),
    { immediate: true },
  )
  onScopeDispose(() => {
    if (timer) clearTimeout(timer)
    controller?.abort()
  })

  return { data, error, pending, refresh: () => schedule(true) }
}

/**
 * A memory limit as the user typed it (`all`, `0.8`, `12GiB`, a byte count).
 * The contract publishes the field as an open object; the server parses the
 * spelling and refuses what it cannot read.
 */
export function budgetRequest(text: string): NonNullable<Limits['vram']> {
  const numeric = Number(text)
  const value: string | number = text !== '' && Number.isFinite(numeric) ? numeric : text
  return value as unknown as NonNullable<Limits['vram']>
}
