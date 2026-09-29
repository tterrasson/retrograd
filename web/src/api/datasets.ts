import { computed, toValue, type MaybeRefOrGetter } from 'vue'
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/vue-query'
import { authHeaders, client, reportUnauthorized, settle, unwrap } from './client'
import { keys } from './keys'
import { ApiProblem, problemFromBody } from './problem'
import type { DatasetListing, DatasetView } from './types'

export function useDatasets() {
  const query = useInfiniteQuery({
    queryKey: keys.datasets(),
    initialPageParam: undefined as string | undefined,
    queryFn: ({ pageParam, signal }): Promise<DatasetListing> =>
      unwrap(
        client.GET('/v1/datasets', {
          params: { query: { limit: 100, ...(pageParam ? { cursor: pageParam } : {}) } },
          signal,
        }),
      ),
    getNextPageParam: (last) => last.next_cursor ?? undefined,
  })
  const datasets = computed<DatasetView[]>(
    () => query.data.value?.pages.flatMap((page) => page.datasets) ?? [],
  )
  return { ...query, datasets }
}

export function useDatasetPreview(id: MaybeRefOrGetter<string | null | undefined>, limit = 10) {
  return useQuery({
    queryKey: computed(() => keys.datasetPreview(toValue(id) ?? '')),
    enabled: computed(() => !!toValue(id)),
    queryFn: ({ signal }) =>
      unwrap(
        client.GET('/v1/datasets/{id}/preview', {
          params: { path: { id: toValue(id) ?? '' }, query: { limit } },
          signal,
        }),
      ),
    staleTime: Infinity,
  })
}

export function useTokenize() {
  return useMutation({
    mutationFn: ({ id, model }: { id: string; model: string }) =>
      unwrap(
        client.POST('/v1/datasets/{id}/tokenize', {
          params: { path: { id } },
          body: { model },
        }),
      ),
  })
}

export function useDeleteDataset() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      settle(client.DELETE('/v1/datasets/{id}', { params: { path: { id } } })),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.datasets() }),
  })
}

/** The media type the upload route reads a file as, from its name. */
export function uploadMediaType(file: File): string {
  const name = file.name.toLowerCase()
  if (name.endsWith('.jsonl') || name.endsWith('.ndjson')) return 'application/x-ndjson'
  if (name.endsWith('.json')) return 'application/json'
  return 'text/plain'
}

export interface UploadOptions {
  name?: string
  format?: string
  onProgress?(loaded: number, total: number): void
  signal?: AbortSignal
}

/**
 * `POST /v1/datasets` with the file as the raw body. `XMLHttpRequest` rather
 * than `fetch`, because only it reports upload progress.
 */
export function uploadDataset(file: File, options: UploadOptions = {}): Promise<DatasetView> {
  const query = new URLSearchParams()
  if (options.name) query.set('name', options.name)
  if (options.format) query.set('format', options.format)
  const url = `/v1/datasets${query.size ? `?${query}` : ''}`
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', url)
    request.setRequestHeader('Content-Type', uploadMediaType(file))
    for (const [name, value] of Object.entries(authHeaders())) request.setRequestHeader(name, value)
    request.upload.onprogress = (event) => options.onProgress?.(event.loaded, event.total)
    request.onload = () => {
      let body: unknown = request.responseText
      try {
        body = JSON.parse(request.responseText)
      } catch {
        // Not JSON: kept as text for the problem's detail.
      }
      if (request.status >= 200 && request.status < 300) {
        resolve(body as DatasetView)
        return
      }
      if (request.status === 401) reportUnauthorized()
      reject(problemFromBody(request.status, body, request.statusText))
    }
    request.onerror = () =>
      reject(new ApiProblem({ status: 0, type: 'network', title: 'upload failed' }))
    request.onabort = () =>
      reject(new ApiProblem({ status: 0, type: 'aborted', title: 'cancelled' }))
    options.signal?.addEventListener('abort', () => request.abort())
    request.send(file)
  })
}

export function useUploadDataset() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ file, ...options }: UploadOptions & { file: File }) =>
      uploadDataset(file, options),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: keys.datasets() }),
  })
}
