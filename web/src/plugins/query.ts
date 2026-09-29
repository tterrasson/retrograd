import { QueryClient, type VueQueryPluginOptions } from '@tanstack/vue-query'
import { isProblem } from '@/api/problem'

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 5_000,
        refetchOnWindowFocus: false,
        // A 4xx is an answer, not an accident: retrying it asks the same
        // question and gets the same refusal.
        retry: (count, error) => {
          if (isProblem(error) && error.status >= 400 && error.status < 500) return false
          return count < 2
        },
      },
      mutations: { retry: false },
    },
  })
}

export function queryPluginOptions(client: QueryClient): VueQueryPluginOptions {
  return { queryClient: client }
}
