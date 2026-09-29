import { ref } from 'vue'
import { toProblem } from '@/api/problem'

export interface Notice {
  id: number
  text: string
  color: 'success' | 'error' | 'info' | 'warning'
  detail?: string
}

const queue = ref<Notice[]>([])
let next = 1

/** The global snackbar's queue. */
export function useNotify() {
  function push(text: string, color: Notice['color'] = 'info', detail?: string) {
    queue.value = [...queue.value, { id: next++, text, color, detail }]
  }
  function dismiss(id: number) {
    queue.value = queue.value.filter((notice) => notice.id !== id)
  }
  function error(cause: unknown, prefix?: string) {
    const problem = toProblem(cause)
    const text = problem.detail || problem.title
    const trace = problem.traceId ? `trace ${problem.traceId}` : undefined
    push(prefix ? `${prefix}: ${text}` : text, 'error', trace)
  }
  return { queue, push, dismiss, error, success: (text: string) => push(text, 'success') }
}
