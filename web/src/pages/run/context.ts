import { inject, type ComputedRef, type InjectionKey, type Ref } from 'vue'
import { RunEventsKey, type RunEvents } from '@/api/events'
import type { RunView } from '@/api/types'

export const RunKey: InjectionKey<ComputedRef<RunView | null>> = Symbol('run')

/** The run of the enclosing run page. */
export function useRunContext(): ComputedRef<RunView | null> {
  const run = inject(RunKey)
  if (!run) throw new Error('a run tab must be rendered inside the run page')
  return run
}

/** The event stream of the enclosing run page. */
export function useRunEventsContext(): Ref<RunEvents | null> {
  const events = inject(RunEventsKey)
  if (!events) throw new Error('a run tab must be rendered inside the run page')
  return events
}
