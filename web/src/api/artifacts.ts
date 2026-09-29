import { computed, toValue, type MaybeRefOrGetter } from 'vue'
import { useMutation, useQuery } from '@tanstack/vue-query'
import { client, unwrap } from './client'
import { keys } from './keys'

export function useArtifacts(id: MaybeRefOrGetter<string>) {
  return useQuery({
    queryKey: computed(() => keys.artifacts(toValue(id))),
    queryFn: ({ signal }) =>
      unwrap(
        client.GET('/v1/runs/{id}/artifacts', { params: { path: { id: toValue(id) } }, signal }),
      ),
  })
}

/**
 * Asks for a short-lived signed link, then lets the browser fetch it: the
 * download streams to disk instead of through a `Blob` in memory.
 */
export function useDownloadArtifact(id: MaybeRefOrGetter<string>) {
  return useMutation({
    mutationFn: async (name: string) => {
      const link = await unwrap(
        client.POST('/v1/runs/{id}/artifacts/{name}/link', {
          params: { path: { id: toValue(id), name } },
        }),
      )
      const anchor = document.createElement('a')
      anchor.href = link.href
      anchor.rel = 'noopener'
      anchor.download = ''
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      return link
    },
  })
}
