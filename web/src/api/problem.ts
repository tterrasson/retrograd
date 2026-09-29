import type { ErrorCode, FieldError } from './types'

/** The slug at the end of a problem `type` URI, which is what a client branches on. */
export type ProblemKind =
  | 'invalid-request'
  | 'server-declared'
  | 'unknown-catalog-id'
  | 'insufficient-memory'
  | 'forbidden-path'
  | 'not-found'
  | 'conflict'
  | 'payload-too-large'
  | 'unsupported-media-type'
  | 'method-not-allowed'
  | 'device-busy'
  | 'timeout'
  | 'unauthorized'
  | 'not-implemented'
  | 'internal'
  | 'network'
  | 'openai'
  | (string & {})

/**
 * The only error shape of the application. Everything that fails - a problem
 * document, an OpenAI error envelope, a dropped connection - becomes one.
 */
export class ApiProblem extends Error {
  readonly status: number
  readonly type: string
  readonly title: string
  readonly detail: string
  readonly traceId?: string
  readonly errors: FieldError[]
  readonly meta?: unknown

  constructor(init: {
    status: number
    type: string
    title: string
    detail?: string
    traceId?: string
    errors?: FieldError[]
    meta?: unknown
  }) {
    super(init.detail ? `${init.title}: ${init.detail}` : init.title)
    this.name = 'ApiProblem'
    this.status = init.status
    this.type = init.type
    this.title = init.title
    this.detail = init.detail ?? ''
    this.traceId = init.traceId
    this.errors = init.errors ?? []
    this.meta = init.meta
  }

  get kind(): ProblemKind {
    const slash = this.type.lastIndexOf('/')
    return slash >= 0 ? this.type.slice(slash + 1) : this.type
  }

  /** The field errors whose pointer starts with `prefix`. */
  fieldErrors(prefix = ''): FieldError[] {
    return this.errors.filter((error) => error.pointer.startsWith(prefix))
  }
}

export function isProblem(value: unknown): value is ApiProblem {
  return value instanceof ApiProblem
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function fieldErrorsOf(value: unknown): FieldError[] {
  if (!Array.isArray(value)) return []
  return value.filter(isRecord).map((item) => {
    const error: FieldError = {
      pointer: asString(item.pointer) ?? '',
      code: (asString(item.code) ?? 'invalid_value') as ErrorCode,
      message: asString(item.message) ?? '',
    }
    const hint = asString(item.hint)
    if (hint !== undefined) error.hint = hint
    return error
  })
}

/** A parsed body: RFC 9457 problem document, OpenAI envelope, or anything else. */
export function problemFromBody(status: number, body: unknown, statusText = ''): ApiProblem {
  if (isRecord(body) && typeof body.type === 'string' && typeof body.title === 'string') {
    return new ApiProblem({
      status: typeof body.status === 'number' ? body.status : status,
      type: body.type,
      title: body.title,
      detail: asString(body.detail),
      traceId: asString(body.trace_id),
      errors: fieldErrorsOf(body.errors),
      meta: body.meta,
    })
  }
  if (isRecord(body) && isRecord(body.error)) {
    const error = body.error
    return new ApiProblem({
      status,
      type: 'openai',
      title: asString(error.type) ?? 'error',
      detail: asString(error.message),
      errors: [],
      meta: { code: error.code, param: error.param },
    })
  }
  return new ApiProblem({
    status,
    type: status === 0 ? 'network' : `http-${status}`,
    title: statusText || (status === 0 ? 'network error' : `HTTP ${status}`),
    detail: typeof body === 'string' ? body.slice(0, 500) : undefined,
  })
}

/** Reads a failed response into an `ApiProblem`. Never throws. */
export async function problemFromResponse(response: Response): Promise<ApiProblem> {
  let text = ''
  try {
    text = await response.text()
  } catch {
    // The body was already read or the connection dropped: the status is all there is.
  }
  let body: unknown = text
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }
  return problemFromBody(response.status, body, response.statusText)
}

/** Wraps anything thrown into an `ApiProblem`, so a caller only knows one shape. */
export function toProblem(error: unknown): ApiProblem {
  if (isProblem(error)) return error
  if (error instanceof DOMException && error.name === 'AbortError') {
    return new ApiProblem({ status: 0, type: 'aborted', title: 'cancelled' })
  }
  const detail = error instanceof Error ? error.message : String(error)
  return new ApiProblem({ status: 0, type: 'network', title: 'network error', detail })
}

/** `insufficient-memory` carries the dominant posts in `meta`; read them defensively. */
export interface MemoryShortfall {
  overflowBytes?: number
  vramBudgetBytes?: number
  ramBudgetBytes?: number
  measuredDeviceBytes?: number
  dominantPosts: { post: string; bytes: number }[]
  leversApplied: string[]
  unlocks: string[]
}

export function memoryShortfall(problem: ApiProblem): MemoryShortfall | null {
  if (problem.kind !== 'insufficient-memory' || !isRecord(problem.meta)) return null
  const meta = problem.meta
  const number = (value: unknown) => (typeof value === 'number' ? value : undefined)
  const strings = (value: unknown) =>
    Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
  const posts = Array.isArray(meta.dominant_posts)
    ? meta.dominant_posts
        .filter(isRecord)
        .map((item) => ({ post: asString(item.post) ?? '?', bytes: number(item.bytes) ?? 0 }))
    : []
  return {
    overflowBytes: number(meta.overflow_bytes),
    vramBudgetBytes: number(meta.vram_budget_bytes),
    ramBudgetBytes: number(meta.ram_budget_bytes),
    measuredDeviceBytes: number(meta.measured_device_bytes),
    dominantPosts: posts,
    leversApplied: strings(meta.levers_applied),
    unlocks: strings(meta.unlocks),
  }
}
