import { stringify } from 'smol-toml'

function prune(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(prune)
  if (typeof value === 'object' && value !== null) {
    const out: Record<string, unknown> = {}
    for (const [key, child] of Object.entries(value)) {
      if (child === null || child === undefined) continue
      out[key] = prune(child)
    }
    return out
  }
  return value
}

/**
 * A JSON document rendered as TOML, for reading and copying. TOML has no null:
 * absent and null fields are dropped, which is what a TOML file would say.
 */
export function toToml(document: unknown): string {
  const pruned = prune(document)
  if (typeof pruned !== 'object' || pruned === null || Array.isArray(pruned)) return ''
  return stringify(pruned as Record<string, unknown>)
}
