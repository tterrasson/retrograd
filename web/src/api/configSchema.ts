// The configuration document's JSON Schema, crossed with a plan answer: what
// can be set (the schema), what is set and why (the effective configuration
// and its provenance), and what the client pinned (params).

import { useQuery } from '@tanstack/vue-query'
import { client, unwrap } from './client'
import { keys } from './keys'
import type { DerivedField, Origin, Provenance } from './types'
import { getAt, hasAt } from '@/utils/pointer'

export interface JsonSchema {
  $ref?: string
  $defs?: Record<string, JsonSchema>
  type?: string | string[]
  properties?: Record<string, JsonSchema>
  required?: string[]
  additionalProperties?: boolean | JsonSchema
  items?: JsonSchema
  enum?: unknown[]
  const?: unknown
  oneOf?: JsonSchema[]
  anyOf?: JsonSchema[]
  allOf?: JsonSchema[]
  title?: string
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  exclusiveMaximum?: number
  minItems?: number
  maxItems?: number
  format?: string
  'x-retrograd-server-declared'?: boolean
  'x-retrograd-applies-to'?: string[]
  'x-retrograd-patchable'?: boolean
  [extension: string]: unknown
}

/** `GET /v1/config-schema`: the schema of the server that answers, loaded once. */
export function useConfigSchema() {
  return useQuery({
    queryKey: keys.configSchema(),
    queryFn: async ({ signal }) =>
      (await unwrap(client.GET('/v1/config-schema', { signal }))) as unknown as JsonSchema,
    staleTime: Infinity,
    retry: false,
  })
}

const MAX_DEPTH = 8

/** Follows `$ref` (`#/$defs/…` or `#/components/schemas/…`) and merges `allOf`. */
export function resolveSchema(node: JsonSchema | undefined, root: JsonSchema): JsonSchema {
  let current: JsonSchema = node ?? {}
  for (let hops = 0; current.$ref && hops < 16; hops++) {
    const name = current.$ref.split('/').pop() ?? ''
    const target =
      root.$defs?.[name] ??
      ((root as { components?: { schemas?: Record<string, JsonSchema> } }).components?.schemas?.[
        name
      ] as JsonSchema | undefined)
    const { $ref: _ref, ...siblings } = current
    current = target ? { ...target, ...siblings } : siblings
  }
  if (current.allOf?.length) {
    const { allOf, ...rest } = current
    return allOf.reduce<JsonSchema>((merged, part) => {
      const resolved = resolveSchema(part, root)
      return {
        ...merged,
        ...resolved,
        properties: { ...merged.properties, ...resolved.properties },
        required: [...(merged.required ?? []), ...(resolved.required ?? [])],
      }
    }, rest)
  }
  return current
}

function isNullSchema(node: JsonSchema): boolean {
  return node.type === 'null' || (Array.isArray(node.type) && node.type.every((t) => t === 'null'))
}

/** Strips `null` from a type union: `Option<T>` in the schema is `T | null`. */
export function unwrapNullable(
  node: JsonSchema,
  root: JsonSchema,
): { schema: JsonSchema; nullable: boolean } {
  const resolved = resolveSchema(node, root)
  const variants = resolved.oneOf ?? resolved.anyOf
  if (variants) {
    const kept = variants.filter((variant) => !isNullSchema(resolveSchema(variant, root)))
    const nullable = kept.length < variants.length
    if (kept.length === 1 && kept[0]) {
      const { oneOf: _o, anyOf: _a, ...rest } = resolved
      const inner = resolveSchema(kept[0], root)
      return {
        schema: { ...inner, ...rest, description: rest.description ?? inner.description },
        nullable,
      }
    }
    return { schema: { ...resolved, oneOf: kept, anyOf: undefined }, nullable }
  }
  if (Array.isArray(resolved.type)) {
    const types = resolved.type.filter((type) => type !== 'null')
    const nullable = types.length < resolved.type.length
    return { schema: { ...resolved, type: types.length === 1 ? types[0] : types }, nullable }
  }
  return { schema: resolved, nullable: false }
}

export type EditorKind =
  'select' | 'switch' | 'integer' | 'number' | 'text' | 'chips' | 'variant' | 'json'

function primaryType(schema: JsonSchema): string | undefined {
  return Array.isArray(schema.type) ? schema.type[0] : schema.type
}

/** The editor a schema node calls for. Bounds only help typing; the server decides. */
export function editorFor(schema: JsonSchema): EditorKind {
  if (schema.enum?.length) return 'select'
  if ((schema.oneOf?.length ?? 0) > 1) {
    const allConst = schema.oneOf?.every(
      (variant) => variant.const !== undefined || variant.enum?.length === 1,
    )
    return allConst ? 'select' : 'variant'
  }
  switch (primaryType(schema)) {
    case 'boolean':
      return 'switch'
    case 'integer':
      return 'integer'
    case 'number':
      return 'number'
    case 'string':
      return 'text'
    case 'array': {
      const itemType = schema.items ? primaryType(schema.items) : undefined
      return itemType === 'string' || itemType === 'integer' || itemType === 'number'
        ? 'chips'
        : 'json'
    }
    default:
      return 'json'
  }
}

/** The choices of a `select`: `enum`, or a `oneOf` of constants. */
export function enumOptions(schema: JsonSchema): unknown[] {
  if (schema.enum) return schema.enum
  return (schema.oneOf ?? []).map((variant) => variant.const ?? variant.enum?.[0])
}

function isServerDeclared(schema: JsonSchema): boolean {
  return schema['x-retrograd-server-declared'] === true
}

/** `false` when the node is marked for other algorithms only. */
export function appliesTo(schema: JsonSchema, algorithm: string | null | undefined): boolean {
  const list = schema['x-retrograd-applies-to']
  if (!Array.isArray(list) || !list.length || !algorithm) return true
  return list.includes(algorithm)
}

export interface ParamRow {
  /** Dotted, the spelling of provenance and defaults: `training.lr`. */
  path: string
  segments: string[]
  section: string
  label: string
  schema: JsonSchema
  nullable: boolean
  editor: EditorKind
  description?: string
  /** The resolved value; `undefined` when the plan does not configure it. */
  value: unknown
  present: boolean
  origin?: Origin
  /** Pinned by the client in `params`. */
  override: boolean
  paramValue?: unknown
  rule?: DerivedField
  patchable: boolean
  required: boolean
}

export interface ParamSection {
  name: string
  description?: string
  /** The section exists in the effective configuration. */
  present: boolean
  /** Fields a newly enabled section must carry, with the value to start from. */
  required: { segments: string[]; initial: unknown }[]
  rows: ParamRow[]
}

export interface CrossInput {
  schema: JsonSchema | null
  effective: unknown
  provenance?: Provenance | null
  params?: Record<string, unknown>
  defaults?: readonly DerivedField[]
  algorithm?: string | null
}

/** A starting value for a required field of a section being enabled. */
export function initialValue(schema: JsonSchema, root: JsonSchema): unknown {
  const { schema: node } = unwrapNullable(schema, root)
  if (node.default !== undefined) return node.default
  if (node.enum?.length) return node.enum[0]
  switch (primaryType(node)) {
    case 'boolean':
      return false
    case 'integer':
    case 'number':
      return node.minimum ?? 0
    case 'string':
      return ''
    case 'array':
      return []
    case 'object': {
      const out: Record<string, unknown> = {}
      for (const key of node.required ?? []) {
        const child = node.properties?.[key]
        if (child) out[key] = initialValue(child, root)
      }
      return out
    }
    default:
      return null
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Walks the schema section by section and answers one row per settable leaf,
 * each carrying its resolved value, its origin and whether the client pinned
 * it. Without a schema, the effective configuration alone gives the rows.
 */
export function crossSchema(input: CrossInput): ParamSection[] {
  const { schema, effective, provenance, params = {}, defaults = [], algorithm } = input
  const rules = new Map(defaults.map((rule) => [rule.path, rule]))
  const sections = new Map<string, ParamSection>()

  const sectionOf = (name: string, description?: string): ParamSection => {
    let section = sections.get(name)
    if (!section) {
      section = {
        name,
        description,
        present: name === 'general' || hasAt(effective, [name]),
        required: [],
        rows: [],
      }
      sections.set(name, section)
    }
    return section
  }

  const addRow = (
    segments: string[],
    node: JsonSchema,
    nullable: boolean,
    required: boolean,
    section: string,
  ) => {
    const path = segments.join('.')
    sectionOf(section).rows.push({
      path,
      segments,
      section,
      label: section === 'general' ? path : segments.slice(1).join('.') || path,
      schema: node,
      nullable,
      editor: editorFor(node),
      description: node.description,
      value: getAt(effective, segments),
      present: hasAt(effective, segments),
      origin: provenance?.[path],
      override: hasAt(params, segments),
      paramValue: getAt(params, segments),
      rule: rules.get(path),
      patchable: node['x-retrograd-patchable'] === true,
      required,
    })
  }

  if (schema) {
    const root = resolveSchema(schema, schema)
    const walk = (
      node: JsonSchema,
      segments: string[],
      section: string,
      required: boolean,
      depth: number,
    ) => {
      const { schema: resolved, nullable } = unwrapNullable(node, schema)
      if (isServerDeclared(resolved) || !appliesTo(resolved, algorithm)) return
      const properties = resolved.properties
      const isTable = properties && Object.keys(properties).length > 0
      if (isTable && depth < MAX_DEPTH) {
        const requiredKeys = new Set(resolved.required ?? [])
        for (const [key, child] of Object.entries(properties)) {
          walk(child, [...segments, key], section, requiredKeys.has(key), depth + 1)
        }
        return
      }
      addRow(segments, resolved, nullable, required, section)
    }
    const requiredTop = new Set(root.required ?? [])
    for (const [name, child] of Object.entries(root.properties ?? {})) {
      const { schema: resolved } = unwrapNullable(child, schema)
      if (isServerDeclared(resolved) || !appliesTo(resolved, algorithm)) continue
      const isTable = resolved.properties && Object.keys(resolved.properties).length > 0
      if (!isTable) {
        sectionOf('general')
        addRow([name], resolved, false, requiredTop.has(name), 'general')
        continue
      }
      const section = sectionOf(name, resolved.description)
      section.required = (resolved.required ?? []).flatMap((key) => {
        const field = resolved.properties?.[key]
        return field ? [{ segments: [key], initial: initialValue(field, schema) }] : []
      })
      walk(resolved, [name], name, requiredTop.has(name), 1)
    }
  } else {
    const walkValue = (value: unknown, segments: string[], section: string) => {
      if (isPlainObject(value) && Object.keys(value).length > 0 && segments.length < MAX_DEPTH) {
        for (const [key, child] of Object.entries(value))
          walkValue(child, [...segments, key], section)
        return
      }
      const node: JsonSchema = inferSchema(value)
      addRow(segments, node, value === null, false, section)
    }
    if (isPlainObject(effective)) {
      for (const [name, value] of Object.entries(effective)) {
        if (isPlainObject(value)) {
          sectionOf(name)
          walkValue(value, [name], name)
        } else {
          sectionOf('general')
          walkValue(value, [name], 'general')
        }
      }
    }
  }

  return [...sections.values()].filter((section) => section.rows.length > 0)
}

function inferSchema(value: unknown): JsonSchema {
  if (typeof value === 'boolean') return { type: 'boolean' }
  if (typeof value === 'number') return { type: Number.isInteger(value) ? 'integer' : 'number' }
  if (typeof value === 'string') return { type: 'string' }
  if (Array.isArray(value)) {
    const first = value[0]
    return { type: 'array', items: first === undefined ? { type: 'string' } : inferSchema(first) }
  }
  return {}
}

/** Keeps the rows a search and a "pinned only" view let through. */
export function filterSections(
  sections: readonly ParamSection[],
  options: { search?: string; overridesOnly?: boolean; presentOnly?: boolean },
): ParamSection[] {
  const needle = options.search?.trim().toLowerCase() ?? ''
  return sections
    .map((section) => ({
      ...section,
      rows: section.rows.filter((row) => {
        if (options.overridesOnly && !row.override) return false
        if (options.presentOnly && !row.present) return false
        if (!needle) return true
        return (
          row.path.toLowerCase().includes(needle) ||
          (row.description ?? '').toLowerCase().includes(needle)
        )
      }),
    }))
    .filter((section) => section.rows.length > 0)
}
