// JSON Pointers (RFC 6901), dotted paths (`training.lr`), and the partial
// trees they address.

export function escapeSegment(segment: string): string {
  return segment.replace(/~/g, '~0').replace(/\//g, '~1')
}

export function unescapeSegment(segment: string): string {
  return segment.replace(/~1/g, '/').replace(/~0/g, '~')
}

/** `/a/b~1c` → `['a', 'b/c']`. The empty pointer is the whole document. */
export function parsePointer(pointer: string): string[] {
  if (pointer === '' || pointer === '#') return []
  const body = pointer.startsWith('#') ? pointer.slice(1) : pointer
  if (!body.startsWith('/')) return [unescapeSegment(body)]
  return body.slice(1).split('/').map(unescapeSegment)
}

export function toPointer(segments: readonly string[]): string {
  return segments.map((segment) => `/${escapeSegment(segment)}`).join('')
}

/**
 * `/params/training/lr` with prefix `/params` → `training.lr`; `null` when the
 * pointer is not under the prefix.
 */
export function pointerToPath(pointer: string, prefix = ''): string | null {
  if (prefix && pointer !== prefix && !pointer.startsWith(`${prefix}/`)) return null
  return parsePointer(pointer.slice(prefix.length)).join('.')
}

export function pathToPointer(path: string, prefix = ''): string {
  return prefix + (path ? toPointer(path.split('.')) : '')
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function getAt(tree: unknown, segments: readonly string[]): unknown {
  let node = tree
  for (const segment of segments) {
    if (Array.isArray(node)) node = node[Number(segment)]
    else if (isPlainObject(node)) node = node[segment]
    else return undefined
  }
  return node
}

export function hasAt(tree: unknown, segments: readonly string[]): boolean {
  let node = tree
  for (const segment of segments) {
    if (!isPlainObject(node) || !(segment in node)) return false
    node = node[segment]
  }
  return true
}

/** A copy of `tree` with `value` at `segments`, creating the tables on the way. */
export function setAt(
  tree: Record<string, unknown>,
  segments: readonly string[],
  value: unknown,
): Record<string, unknown> {
  if (segments.length === 0) return isPlainObject(value) ? value : tree
  const [head, ...rest] = segments as [string, ...string[]]
  const child = tree[head]
  const next = rest.length ? setAt(isPlainObject(child) ? child : {}, rest, value) : value
  return { ...tree, [head]: next }
}

/** A copy of `tree` without `segments`, dropping the tables it leaves empty. */
export function removeAt(
  tree: Record<string, unknown>,
  segments: readonly string[],
): Record<string, unknown> {
  if (segments.length === 0) return {}
  const [head, ...rest] = segments as [string, ...string[]]
  if (!(head in tree)) return tree
  if (rest.length === 0) {
    const { [head]: _removed, ...kept } = tree
    return kept
  }
  const child = tree[head]
  if (!isPlainObject(child)) return tree
  const pruned = removeAt(child, rest)
  if (Object.keys(pruned).length === 0) {
    const { [head]: _removed, ...kept } = tree
    return kept
  }
  return { ...tree, [head]: pruned }
}

/** Every leaf of a tree as a dotted path. Arrays and empty tables are leaves. */
export function leafPaths(tree: unknown, prefix: string[] = []): string[][] {
  if (!isPlainObject(tree) || (Object.keys(tree).length === 0 && prefix.length > 0)) {
    return prefix.length ? [prefix] : []
  }
  return Object.entries(tree).flatMap(([key, value]) => leafPaths(value, [...prefix, key]))
}
