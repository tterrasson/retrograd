// The OpenAI-compatible routes: their own wire format and their own error
// envelope, so they are not called through the typed client.
import { useQuery } from '@tanstack/vue-query'
import { authorizedFetch } from './client'
import { keys } from './keys'
import { problemFromBody } from './problem'
import { SseParser } from './sse'

export interface ModelCard {
  id: string
  object: string
  created: number
  owned_by: string
  /** Served by a run that is still training. */
  live?: boolean
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatOptions {
  model: string
  messages: ChatMessage[]
  temperature?: number
  top_p?: number
  max_tokens?: number
  seed?: number
}

export interface ChatUsage {
  prompt_tokens?: number
  completion_tokens?: number
}

export async function listModels(signal?: AbortSignal): Promise<ModelCard[]> {
  const response = await authorizedFetch('/v1/models', { signal })
  const body = (await response.json()) as { data?: ModelCard[] }
  return body.data ?? []
}

export function useOpenAiModels(enabled: () => boolean) {
  return useQuery({
    queryKey: keys.openaiModels(),
    queryFn: ({ signal }) => listModels(signal),
    enabled,
    staleTime: 10_000,
  })
}

/** Targets grouped by the run they come from: `<run>`, `<run>@final`, `<run>@base`… */
export function groupTargets(
  models: readonly ModelCard[],
): { run: string; targets: ModelCard[] }[] {
  const groups = new Map<string, ModelCard[]>()
  for (const model of models) {
    const at = model.id.indexOf('@')
    const run = at < 0 ? model.id : model.id.slice(0, at)
    const list = groups.get(run) ?? []
    list.push(model)
    groups.set(run, list)
  }
  return [...groups].map(([run, targets]) => ({ run, targets }))
}

/**
 * `POST /v1/chat/completions` with `stream: true`. Calls `onDelta` with every
 * piece of text; resolves with the usage the last chunk reported, if any.
 */
export async function streamChat(
  options: ChatOptions,
  onDelta: (text: string) => void,
  signal?: AbortSignal,
): Promise<ChatUsage | null> {
  const response = await authorizedFetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ ...options, stream: true }),
    signal,
  })
  if (!response.body) return null
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  const parser = new SseParser()
  let usage: ChatUsage | null = null
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    for (const message of parser.push(decoder.decode(value, { stream: true }))) {
      if (message.data === '[DONE]') return usage
      let chunk: {
        choices?: { delta?: { content?: string | null } }[]
        usage?: ChatUsage
        error?: unknown
      }
      try {
        chunk = JSON.parse(message.data)
      } catch {
        continue
      }
      if (chunk.error) throw problemFromBody(response.status, chunk)
      const text = chunk.choices?.[0]?.delta?.content
      if (text) onDelta(text)
      if (chunk.usage) usage = chunk.usage
    }
  }
  return usage
}
