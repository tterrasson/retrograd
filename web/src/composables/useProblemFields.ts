import { computed, toValue, type MaybeRefOrGetter } from 'vue'
import { isProblem, type ApiProblem } from '@/api/problem'
import type { FieldError } from '@/api/types'
import { pointerToPath } from '@/utils/pointer'

export interface ProblemFields {
  /** Messages per dotted path under the prefix, for `:error-messages`. */
  fields: Record<string, string[]>
  /** The errors no field of this form claims: shown in a banner. */
  unclaimed: FieldError[]
}

function message(error: FieldError): string {
  return error.hint ? `${error.message} (${error.hint})` : error.message
}

/**
 * Splits a problem's field errors into the ones a form can show beside a
 * field (their pointer is under `prefix` and names a known field, when a list
 * is given) and the rest.
 */
export function problemFields(
  problem: ApiProblem | null | undefined,
  prefix: string,
  known?: (path: string) => boolean,
): ProblemFields {
  const fields: Record<string, string[]> = {}
  const unclaimed: FieldError[] = []
  for (const error of problem?.errors ?? []) {
    const path = pointerToPath(error.pointer, prefix)
    if (path === null || (known && !known(path))) {
      unclaimed.push(error)
      continue
    }
    ;(fields[path] ??= []).push(message(error))
  }
  return { fields, unclaimed }
}

export function useProblemFields(
  problem: MaybeRefOrGetter<unknown>,
  prefix: MaybeRefOrGetter<string>,
  known?: (path: string) => boolean,
) {
  return computed(() => {
    const value = toValue(problem)
    return problemFields(isProblem(value) ? value : null, toValue(prefix), known)
  })
}

/** The messages of the errors whose pointer is exactly `pointer`, or below it. */
export function messagesAt(problem: unknown, pointer: string): string[] {
  if (!isProblem(problem)) return []
  return problem.errors
    .filter((error) => error.pointer === pointer || error.pointer.startsWith(`${pointer}/`))
    .map(message)
}
