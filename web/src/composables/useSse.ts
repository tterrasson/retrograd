import { onScopeDispose, ref, watch, type MaybeRefOrGetter, toValue } from 'vue'
import { connectSse, type SseMessage, type SseState } from '@/api/sse'

/**
 * A stream bound to the calling component: opened when `url` is set, reopened
 * when it changes, closed on unmount.
 */
export function useSse(
  url: MaybeRefOrGetter<string | null>,
  onMessage: (message: SseMessage) => void,
) {
  const state = ref<SseState>('closed')
  let close: (() => void) | null = null
  watch(
    () => toValue(url),
    (next) => {
      close?.()
      close = null
      if (!next) {
        state.value = 'closed'
        return
      }
      const connection = connectSse({
        url: next,
        onMessage,
        onState: (value) => (state.value = value),
      })
      close = () => connection.close()
    },
    { immediate: true },
  )
  onScopeDispose(() => close?.())
  return { state }
}
