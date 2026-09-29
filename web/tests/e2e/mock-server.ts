// A stand-in `retrograd-server` for the end-to-end tests.
//
// It serves the production build from `dist/` with the same fallback the real
// server applies (an extensionless path is the application, a missing file is
// a plain 404), and answers `/v1` from the fixtures: bodies recorded from the
// real server where one can be recorded without a model, typed ones otherwise.
//
// One behaviour is deliberate: a run's event stream opened without
// `Last-Event-ID` is cut half-way, so every page that follows a run has to
// resume - and the tests can check that it did so without a gap or a
// duplicate. `GET /__mock/streams` lists every stream opened, with the id it
// resumed from.
//
// Environment: `MOCK_PORT` (default 4174), `MOCK_TOKEN` (a bearer token the
// API then requires), `MOCK_MODE=viewer` (answer like `retrograd-server view`).
import { existsSync, readFileSync } from 'node:fs'
import { extname, join, normalize } from 'node:path'
import { AGENT_RUN, artifacts, checkpoints, events, listing, runs } from '../fixtures/api'

const root = new URL('../..', import.meta.url).pathname
const dist = join(root, 'dist')
const recorded = join(root, 'tests/fixtures/recorded')

const port = Number(process.env.MOCK_PORT ?? 4174)
const token = process.env.MOCK_TOKEN ?? null
const viewer = process.env.MOCK_MODE === 'viewer'

/** The run whose trajectories the fixtures hold. */
const OBSERVED = viewer ? 'local' : AGENT_RUN

const streams: { run: string; lastEventId: string | null }[] = []

function recordedJson(name: string): unknown {
  return JSON.parse(readFileSync(join(recorded, name), 'utf8'))
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

function problem(status: number, slug: string, detail: string): Response {
  return new Response(
    JSON.stringify({
      type: `https://retrograd.dev/problems/${slug}`,
      title: slug.replaceAll('-', ' '),
      status,
      detail,
      trace_id: 'mock',
    }),
    { status, headers: { 'content-type': 'application/problem+json' } },
  )
}

function capabilities(): unknown {
  const body = recordedJson(viewer ? 'capabilities-viewer.json' : 'capabilities.json') as {
    features: Record<string, unknown>
  }
  body.features.auth = token !== null
  body.features.ui = true
  return body
}

const MEDIA: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.woff2': 'font/woff2',
}

function asset(pathname: string): Response {
  const relative = normalize(pathname).replace(/^\/+/, '')
  if (relative.startsWith('..')) return new Response('not found', { status: 404 })
  let file = join(dist, relative || 'index.html')
  if (!existsSync(file) || relative === '') {
    if (extname(relative)) return new Response('not found', { status: 404 })
    file = join(dist, 'index.html')
  }
  return new Response(Bun.file(file), {
    headers: { 'content-type': MEDIA[extname(file)] ?? 'application/octet-stream' },
  })
}

/** Replays a run's stream after `since`, cutting the first connection short. */
function eventStream(run: string, since: number, resumed: boolean): Response {
  const stream = events[run] ?? []
  const pending = stream.filter((event) => event.seq > since)
  const cut = resumed ? pending.length : Math.ceil(pending.length / 2)
  const encoder = new TextEncoder()
  const body = new ReadableStream({
    async start(controller) {
      controller.enqueue(encoder.encode(': connected\n\n'))
      for (const event of pending.slice(0, cut)) {
        const frame = `id: ${event.seq}\nevent: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`
        controller.enqueue(encoder.encode(frame))
        await Bun.sleep(5)
      }
      // A cut stream ends like a dropped connection; a complete one ends with
      // its terminal event, after which a client stops on its own.
      controller.close()
    },
  })
  return new Response(body, {
    headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' },
  })
}

/** The aggregate stream: open, quiet, kept alive. */
function quietStream(request: Request): Response {
  const encoder = new TextEncoder()
  let timer: ReturnType<typeof setInterval> | undefined
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(': connected\n\n'))
      timer = setInterval(() => controller.enqueue(encoder.encode(': keep-alive\n\n')), 15_000)
      request.signal.addEventListener('abort', () => {
        clearInterval(timer)
        controller.close()
      })
    },
    cancel() {
      clearInterval(timer)
    },
  })
  return new Response(body, { headers: { 'content-type': 'text/event-stream' } })
}

function commandAccepted(run: string): Response {
  const status = runs[run]?.status ?? 'completed'
  return json({ id: run, status })
}

function trajectories(parts: string[], url: URL): Response {
  // parts: [runs, id, trajectories, updates?, u?, groups?, g?]
  const [, id, , , update, , group] = parts
  if (id !== OBSERVED) {
    return id && runs[id]
      ? json({ observed: false, segments: 0, skipped: 0, updates: [] })
      : problem(404, 'not-found', `no run with id ${id}`)
  }
  if (update === undefined) return json(recordedJson('trajectories.json'))
  const name =
    group === undefined ? `update-${update}.json` : `update-${update}-group-${group}.json`
  if (!existsSync(join(recorded, name))) {
    return problem(404, 'not-found', `update ${update} has no group ${group ?? ''}`)
  }
  const body = recordedJson(name) as { members?: { summary: { member: number } }[] }
  const member = url.searchParams.get('member')
  if (group !== undefined && member !== null && body.members) {
    body.members = body.members.filter((entry) => entry.summary.member === Number(member))
  }
  return json(body)
}

function api(request: Request, url: URL): Response {
  const path = url.pathname.replace(/^\/v1\/?/, '')
  const parts = path.split('/').filter(Boolean)
  const method = request.method

  if (path === 'health') return json(recordedJson('health.json'))
  if (token !== null) {
    const header = request.headers.get('authorization')
    const signed = parts[2] === 'artifacts' && url.searchParams.has('sig')
    if (header !== `Bearer ${token}` && !signed) {
      return problem(401, 'unauthorized', 'this server requires a bearer token')
    }
  }
  if (path === 'capabilities') return json(capabilities())

  if (parts[0] === 'runs' && parts[2] === 'trajectories' && method === 'GET') {
    return trajectories(parts, url)
  }
  if (viewer) return problem(404, 'not-found', `no route for ${url.pathname}`)

  for (const name of [
    'defaults',
    'config-schema',
    'rewards',
    'judges',
    'mcp-servers',
    'environments',
  ]) {
    if (path === name) return json(recordedJson(`${name}.json`))
  }
  if (path === 'events') return quietStream(request)
  if (path === 'model-files') return json({ roots: ['/models'], files: [], truncated: false })
  if (path === 'datasets' && method === 'GET') return json({ datasets: [] })
  if (path === 'models') return json({ object: 'list', data: [] })

  if (parts[0] === 'runs') {
    const id = parts[1]
    if (!id) {
      const status = url.searchParams.get('status')
      const algorithm = url.searchParams.get('algorithm')
      return json({
        runs: listing.filter(
          (run) =>
            (!status || run.status === status) && (!algorithm || run.algorithm === algorithm),
        ),
      })
    }
    const run = runs[id]
    if (!run) return problem(404, 'not-found', `no run with id ${id}`)
    const tail = parts.slice(2).join('/')
    if (tail === '' && method === 'GET') return json(run)
    if (tail === 'events') {
      const lastEventId = request.headers.get('last-event-id')
      streams.push({ run: id, lastEventId })
      const since = Number(lastEventId ?? url.searchParams.get('since') ?? 0)
      return eventStream(id, since, lastEventId !== null)
    }
    if (tail === 'metrics') {
      const since = Number(url.searchParams.get('since') ?? 0)
      const samples = (events[id] ?? [])
        .filter((event) => event.type === 'metrics' && event.seq > since)
        .map((event) => ({
          seq: event.seq,
          at: event.at,
          iteration: 'iteration' in event ? event.iteration : 0,
          global_step: 'global_step' in event ? event.global_step : 0,
          values: 'values' in event ? event.values : {},
        }))
      const last = samples.at(-1)?.seq ?? since
      return json({ metrics: samples, next_since: last })
    }
    if (tail === 'checkpoints' && method === 'GET') return json(checkpoints)
    if (tail === 'artifacts') return json(artifacts)
    if (/^artifacts\/[^/]+\/link$/.test(tail) && method === 'POST') {
      const name = parts[3]
      return json({
        href: `/v1/runs/${id}/artifacts/${name}?sig=mock&exp=9999999999`,
        expires_at: 9_999_999_999,
      })
    }
    if (/^artifacts\/[^/]+$/.test(tail)) {
      return new Response('not a real adapter', {
        headers: {
          'content-type': 'application/octet-stream',
          'content-disposition': `attachment; filename="${parts[3]}.gguf"`,
        },
      })
    }
    if (['pause', 'resume', 'cancel', 'checkpoints'].includes(tail) && method === 'POST') {
      return commandAccepted(id)
    }
  }
  return problem(404, 'not-found', `no route for ${url.pathname}`)
}

const server = Bun.serve({
  port,
  hostname: '127.0.0.1',
  idleTimeout: 0,
  fetch(request) {
    const url = new URL(request.url)
    if (url.pathname === '/__mock/streams') return json(streams)
    if (url.pathname === '/v1' || url.pathname.startsWith('/v1/')) return api(request, url)
    return asset(url.pathname)
  },
})

console.log(
  `mock retrograd-server on ${server.url} (${viewer ? 'viewer' : 'server'}${token ? ', token' : ''})`,
)
