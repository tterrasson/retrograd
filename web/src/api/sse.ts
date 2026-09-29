// A `text/event-stream` client over `fetch`. `EventSource` cannot send an
// `Authorization` header, so it cannot talk to a server that wants a token.

import { authHeaders, reportUnauthorized } from './client'
import { ApiProblem, problemFromResponse, toProblem } from './problem'

export interface SseMessage {
  /** The `id:` field, when the frame carried one. */
  id?: string
  /** `message` when the frame named no event. */
  event: string
  data: string
}

/**
 * Incremental parser for the event-stream format (WHATWG HTML, "server-sent
 * events"): lines end with CRLF, LF or CR; a blank line dispatches; `:` starts
 * a comment (the keep-alive); `data:` lines accumulate, joined by LF.
 */
export class SseParser {
  private buffer = ''
  private data: string[] = []
  private event = ''
  private id: string | undefined
  private sawData = false

  /** Feeds a decoded chunk; answers the messages it completed. */
  push(chunk: string): SseMessage[] {
    this.buffer += chunk
    const out: SseMessage[] = []
    let start = 0
    for (let index = 0; index < this.buffer.length; index++) {
      const char = this.buffer[index]
      if (char !== '\n' && char !== '\r') continue
      // A CR at the very end may be the first half of a CRLF split across chunks.
      if (char === '\r' && index === this.buffer.length - 1) break
      const line = this.buffer.slice(start, index)
      if (char === '\r' && this.buffer[index + 1] === '\n') index++
      start = index + 1
      const message = this.line(line)
      if (message) out.push(message)
    }
    this.buffer = this.buffer.slice(start)
    return out
  }

  /** Forgets a partial frame, as a reconnect must. */
  reset(): void {
    this.buffer = ''
    this.data = []
    this.event = ''
    this.id = undefined
    this.sawData = false
  }

  private line(line: string): SseMessage | null {
    if (line === '') return this.dispatch()
    if (line.startsWith(':')) return null
    const colon = line.indexOf(':')
    const field = colon < 0 ? line : line.slice(0, colon)
    let value = colon < 0 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    switch (field) {
      case 'data':
        this.data.push(value)
        this.sawData = true
        break
      case 'event':
        this.event = value
        break
      case 'id':
        if (!value.includes('\0')) this.id = value
        break
      default:
        // `retry:` and unknown fields: the reconnect policy is ours.
        break
    }
    return null
  }

  private dispatch(): SseMessage | null {
    const message: SseMessage | null = this.sawData
      ? { event: this.event || 'message', data: this.data.join('\n') }
      : null
    if (message && this.id !== undefined) message.id = this.id
    this.data = []
    this.event = ''
    this.sawData = false
    // The last event id persists across frames, per the specification; it is
    // tracked by the connection, not here.
    this.id = undefined
    return message
  }
}

export type SseState = 'connecting' | 'open' | 'reconnecting' | 'closed'

export interface SseOptions {
  url: string
  /** Resume after this id (`Last-Event-ID`). Updated from every `id:` seen. */
  lastEventId?: string
  /** Return `reconnect` to replay a gap from the last event id. */
  onMessage(message: SseMessage): void | 'reconnect'
  onState?(state: SseState): void
  /** A failure that ends the stream for good (a 4xx, most often). */
  onFatal?(problem: ApiProblem): void
  /**
   * Asked when the server ends the stream by itself. `false` stops there - a
   * finished run's stream ends because it has nothing more to say.
   */
  shouldReconnect?(): boolean
  /** Longest wait between two attempts. */
  maxBackoffMs?: number
  fetch?: (input: string, init?: RequestInit) => Promise<Response>
  /** Injected in tests: resolves after `ms`, or early when `signal` aborts. */
  sleep?(ms: number, signal: AbortSignal): Promise<void>
}

export interface SseConnection {
  close(): void
  readonly lastEventId: string | undefined
}

function defaultSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve()
    const timer = setTimeout(done, ms)
    function done() {
      clearTimeout(timer)
      signal.removeEventListener('abort', done)
      resolve()
    }
    signal.addEventListener('abort', done)
  })
}

/** 500 ms, doubling, capped. */
export function backoff(attempt: number, max = 10_000): number {
  return Math.min(max, 500 * 2 ** Math.max(0, attempt))
}

/**
 * Opens a stream and keeps it open: a dropped connection reconnects with
 * `Last-Event-ID` after an exponential backoff, so a server that numbers its
 * events resumes exactly where the client stopped.
 */
export function connectSse(options: SseOptions): SseConnection {
  const controller = new AbortController()
  const signal = controller.signal
  const doFetch = options.fetch ?? ((input: string, init?: RequestInit) => fetch(input, init))
  const sleep = options.sleep ?? defaultSleep
  const max = options.maxBackoffMs ?? 10_000
  let lastEventId = options.lastEventId
  let attempt = 0

  const setState = (state: SseState) => options.onState?.(state)

  async function run(): Promise<void> {
    const parser = new SseParser()
    const decoder = new TextDecoder()
    while (!signal.aborted) {
      setState(attempt === 0 ? 'connecting' : 'reconnecting')
      parser.reset()
      let receivedAny = false
      try {
        const headers: Record<string, string> = {
          Accept: 'text/event-stream',
          ...authHeaders(),
        }
        if (lastEventId !== undefined) headers['Last-Event-ID'] = lastEventId
        const response = await doFetch(options.url, { headers, signal, cache: 'no-store' })
        if (!response.ok) {
          const problem = await problemFromResponse(response)
          if (response.status === 401) reportUnauthorized()
          // A refusal is an answer: asking again gets the same one. Only an
          // overloaded or failing server is worth retrying.
          if (response.status < 500 && response.status !== 429) {
            options.onFatal?.(problem)
            break
          }
          throw problem
        }
        if (!response.body) throw new ApiProblem({ status: 0, type: 'network', title: 'no body' })
        setState('open')
        const reader = response.body.getReader()
        let replay = false
        for (;;) {
          const { value, done } = await reader.read()
          if (done) break
          for (const message of parser.push(decoder.decode(value, { stream: true }))) {
            receivedAny = true
            if (message.id !== undefined) lastEventId = message.id
            if (options.onMessage(message) === 'reconnect') {
              replay = true
              break
            }
            if (signal.aborted) break
          }
          if (replay) {
            await reader.cancel()
            break
          }
          if (signal.aborted) break
        }
        if (signal.aborted) break
        if (options.shouldReconnect && !options.shouldReconnect()) break
      } catch (error) {
        if (signal.aborted) break
        const problem = toProblem(error)
        if (problem.type === 'aborted') break
      }
      if (receivedAny) attempt = 0
      setState('reconnecting')
      await sleep(backoff(attempt, max), signal)
      attempt++
    }
    setState('closed')
  }

  void run()

  return {
    close: () => controller.abort(),
    get lastEventId() {
      return lastEventId
    },
  }
}
