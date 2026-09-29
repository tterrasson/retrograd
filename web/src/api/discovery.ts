import { computed, type MaybeRefOrGetter, toValue } from 'vue'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { client, probeClient, unwrap } from './client'
import { isProblem } from './problem'
import { keys, type CatalogKind } from './keys'
import type {
  Capabilities,
  EnvironmentEntry,
  JudgeEntry,
  McpServerEntry,
  ModelFileListing,
  RewardEntry,
} from './types'

export type ProbeResult =
  | { kind: 'open'; capabilities: Capabilities }
  | { kind: 'token-required' }
  | { kind: 'accepted'; capabilities: Capabilities }

/**
 * `GET /v1/capabilities` with the given token, or none. A `200` without a token
 * means the server does not ask for one; a `401` means it does.
 */
export async function probeCapabilities(token: string | null): Promise<ProbeResult> {
  try {
    const capabilities = await unwrap(
      probeClient.GET('/v1/capabilities', {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      }),
    )
    return token ? { kind: 'accepted', capabilities } : { kind: 'open', capabilities }
  } catch (error) {
    if (isProblem(error) && error.status === 401) return { kind: 'token-required' }
    throw error
  }
}

export function useCapabilities(enabled: MaybeRefOrGetter<boolean> = true) {
  return useQuery({
    queryKey: keys.capabilities(),
    enabled: computed(() => toValue(enabled)),
    queryFn: ({ signal }) => unwrap(client.GET('/v1/capabilities', { signal })),
    staleTime: Infinity,
  })
}

/** Seeds the cache with the capabilities the session probe already read. */
export function useSeedCapabilities() {
  const queryClient = useQueryClient()
  return (capabilities: Capabilities) => queryClient.setQueryData(keys.capabilities(), capabilities)
}

export function useDefaults() {
  return useQuery({
    queryKey: keys.defaults(),
    queryFn: ({ signal }) => unwrap(client.GET('/v1/defaults', { signal })),
    staleTime: Infinity,
  })
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: ({ signal }) => unwrap(client.GET('/v1/health', { signal })),
    staleTime: 60_000,
  })
}

export function useRewards() {
  return useQuery({
    queryKey: keys.catalog('rewards'),
    queryFn: async ({ signal }): Promise<RewardEntry[]> =>
      (await unwrap(client.GET('/v1/rewards', { signal }))).rewards,
    staleTime: 60_000,
  })
}

export function useJudges() {
  return useQuery({
    queryKey: keys.catalog('judges'),
    queryFn: async ({ signal }): Promise<JudgeEntry[]> =>
      (await unwrap(client.GET('/v1/judges', { signal }))).judges,
    staleTime: 60_000,
  })
}

export function useMcpServers() {
  return useQuery({
    queryKey: keys.catalog('mcp-servers'),
    queryFn: async ({ signal }): Promise<McpServerEntry[]> =>
      (await unwrap(client.GET('/v1/mcp-servers', { signal }))).mcp_servers,
    staleTime: 60_000,
  })
}

export function useEnvironments() {
  return useQuery({
    queryKey: keys.catalog('environments'),
    queryFn: async ({ signal }): Promise<EnvironmentEntry[]> =>
      (await unwrap(client.GET('/v1/environments', { signal }))).environments,
    staleTime: 60_000,
  })
}

export function isCatalogKind(value: string): value is CatalogKind {
  return ['rewards', 'judges', 'mcp-servers', 'environments'].includes(value)
}

/** `GET /v1/model-files`. `refresh()` asks the server to walk its roots again. */
export function useModelFiles() {
  const queryClient = useQueryClient()
  const query = useQuery({
    queryKey: keys.modelFiles(),
    queryFn: ({ signal }): Promise<ModelFileListing> =>
      unwrap(client.GET('/v1/model-files', { signal })),
    staleTime: 30_000,
  })
  async function refresh() {
    const listing = await unwrap(
      client.GET('/v1/model-files', { params: { query: { refresh: true } } }),
    )
    queryClient.setQueryData(keys.modelFiles(), listing)
  }
  return { ...query, refresh }
}

export function useFeature<K extends keyof Capabilities['features']>(name: MaybeRefOrGetter<K>) {
  const capabilities = useCapabilities()
  return computed(() => capabilities.data.value?.features[toValue(name)])
}
