import { describe, expect, it } from 'vitest'
import { backoff, connectSse, SseParser, type SseMessage } from '@/api/sse'

describe('SseParser', () => {
  it('reads id, event and multi-line data, and dispatches on a blank line', () => {
    const parser = new SseParser()
    const out = parser.push('id: 7\nevent: metrics\ndata: {"a":\ndata: 1}\n\n')
    expect(out).toEqual([{ id: '7', event: 'metrics', data: '{"a":\n1}' }])
  })

  it('ignores comments, which is what a keep-alive is', () => {
    const parser = new SseParser()
    expect(parser.push(':\n\n: keep-alive\n\n')).toEqual([])
    expect(parser.push('data: x\n\n')).toEqual([{ event: 'message', data: 'x' }])
  })

  it('assembles frames split anywhere across chunks, CRLF included', () => {
    const parser = new SseParser()
    const text = 'id: 1\r\nevent: log\r\ndata: hello\r\n\r\nid: 2\r\ndata: world\r\n\r\n'
    const out: SseMessage[] = []
    for (const char of text) out.push(...parser.push(char))
    expect(out).toEqual([
      { id: '1', event: 'log', data: 'hello' },
      { id: '2', event: 'message', data: 'world' },
    ])
  })

  it('keeps a value without its single leading space, and a field without a colon', () => {
    const parser = new SseParser()
    expect(parser.push('data:  two spaces\ndata\n\n')).toEqual([
      { event: 'message', data: ' two spaces\n' },
    ])
  })

  it('does not dispatch a frame without data', () => {
    const parser = new SseParser()
    expect(parser.push('event: lonely\nid: 3\n\n')).toEqual([])
  })
})

describe('backoff', () => {
  it('doubles from 500 ms and stops at 10 s', () => {
    expect(backoff(0)).toBe(500)
    expect(backoff(1)).toBe(1000)
    expect(backoff(4)).toBe(8000)
    expect(backoff(5)).toBe(10_000)
    expect(backoff(30)).toBe(10_000)
  })
})

function streamOf(text: string): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(text))
      controller.close()
    },
  })
  return new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
}

describe('connectSse', () => {
  it('replays every event after a lag from the last event id', async () => {
    const seen: string[] = []
    const lastIds: (string | null)[] = []
    const answers = [
      streamOf(
        'id: 1\nevent: status\ndata: running\n\nevent: lagged\ndata: {"since":1}\n\nid: 3\nevent: checkpoint\ndata: too late\n\n',
      ),
      streamOf(
        'id: 2\nevent: checkpoint\ndata: saved\n\nid: 3\nevent: terminal\ndata: completed\n\n',
      ),
    ]
    let done: () => void = () => undefined
    const finished = new Promise<void>((resolve) => (done = resolve))
    const connection = connectSse({
      url: '/v1/runs/x/events',
      fetch: async (_input, init) => {
        lastIds.push(new Headers(init?.headers).get('Last-Event-ID'))
        const next = answers.shift()
        if (!next) throw new Error('no more answers')
        return next
      },
      sleep: async () => undefined,
      onMessage: (message) => {
        if (message.event === 'lagged') return 'reconnect'
        seen.push(`${message.event}:${message.data}`)
        if (message.event === 'terminal') done()
      },
      shouldReconnect: () => seen.length < 3,
    })
    await finished
    connection.close()
    expect(seen).toEqual(['status:running', 'checkpoint:saved', 'terminal:completed'])
    expect(lastIds).toEqual([null, '1'])
  })

  it('reconnects after a cut with the last id it saw', async () => {
    const seen: string[] = []
    const lastIds: (string | null)[] = []
    const answers = [
      streamOf('id: 1\ndata: a\n\nid: 2\ndata: b\n\n'),
      streamOf('id: 3\nevent: terminal\ndata: c\n\n'),
    ]
    let done: () => void = () => undefined
    const finished = new Promise<void>((resolve) => (done = resolve))
    const connection = connectSse({
      url: '/v1/runs/x/events',
      fetch: async (_input, init) => {
        const headers = new Headers(init?.headers)
        lastIds.push(headers.get('Last-Event-ID'))
        const next = answers.shift()
        if (!next) throw new Error('no more answers')
        return next
      },
      sleep: async () => undefined,
      onMessage: (message) => {
        seen.push(message.data)
        if (message.event === 'terminal') done()
      },
      shouldReconnect: () => seen.length < 3,
    })
    await finished
    connection.close()
    expect(seen).toEqual(['a', 'b', 'c'])
    expect(lastIds).toEqual([null, '2'])
    expect(connection.lastEventId).toBe('3')
  })

  it('stops on a refusal instead of asking again', async () => {
    let calls = 0
    const fatal = new Promise<number>((resolve) => {
      connectSse({
        url: '/v1/runs/missing/events',
        fetch: async () => {
          calls++
          return new Response(
            JSON.stringify({
              type: 'https://retrograd.dev/problems/not-found',
              title: 'not found',
              status: 404,
              detail: 'no run',
            }),
            { status: 404, headers: { 'Content-Type': 'application/problem+json' } },
          )
        },
        sleep: async () => undefined,
        onMessage: () => undefined,
        onFatal: (problem) => resolve(problem.status),
      })
    })
    expect(await fatal).toBe(404)
    expect(calls).toBe(1)
  })
})
