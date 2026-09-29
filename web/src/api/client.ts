import createClient, { type Middleware } from 'openapi-fetch'
import type { paths } from './schema'
import { ApiProblem, problemFromResponse, toProblem } from './problem'

/** What the client needs from the session, without importing it. */
export interface ClientHooks {
  token(): string | null
  /** A request came back `401`: the token is gone or wrong. */
  onUnauthorized(): void
}

let hooks: ClientHooks = { token: () => null, onUnauthorized: () => undefined }

export function configureClient(next: ClientHooks): void {
  hooks = next
}

export function authHeaders(token: string | null = hooks.token()): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {}
}

const auth: Middleware = {
  onRequest({ request }) {
    const token = hooks.token()
    if (token && !request.headers.has('Authorization')) {
      request.headers.set('Authorization', `Bearer ${token}`)
    }
    return request
  },
}

function problemMiddleware(reportUnauthorized: boolean): Middleware {
  return {
    async onResponse({ response }) {
      if (response.ok) return response
      const problem = await problemFromResponse(response.clone())
      if (reportUnauthorized && response.status === 401) hooks.onUnauthorized()
      throw problem
    },
    onError({ error }) {
      return toProblem(error)
    },
  }
}

/** The typed client every query goes through. Same origin: no base URL. */
export const client = createClient<paths>({ baseUrl: '' })
client.use(auth, problemMiddleware(true))

/**
 * A client that neither sends the session token nor reacts to a `401`: the
 * probe that decides whether a token is needed at all, and the check of a
 * token typed on the login screen.
 */
export const probeClient = createClient<paths>({ baseUrl: '' })
probeClient.use(problemMiddleware(false))

/** The body of a successful call. The middleware already threw on anything else. */
export async function unwrap<T>(call: Promise<{ data?: T; response: Response }>): Promise<T> {
  const { data, response } = await call
  if (data === undefined) {
    throw new ApiProblem({
      status: response.status,
      type: 'empty-body',
      title: 'empty response',
      detail: `${response.url} answered ${response.status} without a body`,
    })
  }
  return data
}

/** For operations that answer `204`. */
export async function settle(call: Promise<{ response: Response }>): Promise<void> {
  await call
}

/**
 * `fetch` with the session's token, for what the typed client does not carry:
 * event streams, the OpenAI routes, a TOML body. A failed response is thrown
 * as an `ApiProblem`, as everywhere else.
 */
export async function authorizedFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers)
  for (const [name, value] of Object.entries(authHeaders())) {
    if (!headers.has(name)) headers.set(name, value)
  }
  let response: Response
  try {
    response = await fetch(input, { ...init, headers })
  } catch (error) {
    throw toProblem(error)
  }
  if (!response.ok) {
    const problem = await problemFromResponse(response)
    if (response.status === 401) hooks.onUnauthorized()
    throw problem
  }
  return response
}

export function reportUnauthorized(): void {
  hooks.onUnauthorized()
}
