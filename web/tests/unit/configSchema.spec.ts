import { describe, expect, it } from 'vitest'
import {
  appliesTo,
  crossSchema,
  editorFor,
  enumOptions,
  filterSections,
  initialValue,
  unwrapNullable,
  type JsonSchema,
} from '@/api/configSchema'

const schema: JsonSchema = {
  $ref: '#/$defs/ConfigDocument',
  $defs: {
    ConfigDocument: {
      type: 'object',
      required: ['run'],
      properties: {
        run: { $ref: '#/$defs/RunToml' },
        training: { oneOf: [{ type: 'null' }, { $ref: '#/$defs/TrainingToml' }] },
        grpo: {
          oneOf: [{ type: 'null' }, { $ref: '#/$defs/GrpoToml' }],
          'x-retrograd-applies-to': ['grpo'],
        },
        evaluation: { oneOf: [{ type: 'null' }, { $ref: '#/$defs/EvaluationToml' }] },
      },
    },
    RunToml: {
      type: 'object',
      properties: { algorithm: { type: 'string', enum: ['sft', 'grpo'] } },
    },
    TrainingToml: {
      type: 'object',
      description: 'How to train.',
      properties: {
        lr: {
          type: ['number', 'null'],
          minimum: 0,
          description: 'Learning rate.',
          'x-retrograd-patchable': true,
        },
        epochs: { type: 'integer' },
        packing: { type: 'boolean' },
        targets: { type: 'array', items: { type: 'string' } },
      },
    },
    GrpoToml: {
      type: 'object',
      properties: {
        group_size: { type: 'integer', minimum: 2 },
        reward_command: { type: 'string', 'x-retrograd-server-declared': true },
      },
    },
    EvaluationToml: {
      type: 'object',
      required: ['dataset', 'every_iterations'],
      properties: {
        dataset: { type: 'string' },
        every_iterations: { type: 'integer', minimum: 1 },
        patience: { type: ['integer', 'null'] },
      },
    },
  },
}

const effective = {
  run: { algorithm: 'sft' },
  training: { lr: 0.0002, epochs: 3, packing: true, targets: ['q', 'v'] },
}

describe('schema walking', () => {
  it('resolves $ref and strips null from unions', () => {
    const { schema: node, nullable } = unwrapNullable(
      { oneOf: [{ type: 'null' }, { $ref: '#/$defs/TrainingToml' }] },
      schema,
    )
    expect(nullable).toBe(true)
    expect(Object.keys(node.properties ?? {})).toEqual(['lr', 'epochs', 'packing', 'targets'])
    expect(unwrapNullable({ type: ['integer', 'null'] }, schema)).toEqual({
      schema: { type: 'integer' },
      nullable: true,
    })
  })

  it('chooses the editor from the schema', () => {
    expect(editorFor({ type: 'string', enum: ['a', 'b'] })).toBe('select')
    expect(editorFor({ type: 'boolean' })).toBe('switch')
    expect(editorFor({ type: 'integer', minimum: 1 })).toBe('integer')
    expect(editorFor({ type: 'number' })).toBe('number')
    expect(editorFor({ type: 'string' })).toBe('text')
    expect(editorFor({ type: 'array', items: { type: 'string' } })).toBe('chips')
    expect(editorFor({ type: 'array', items: { type: 'object' } })).toBe('json')
    expect(editorFor({ oneOf: [{ type: 'integer' }, { type: 'string' }] })).toBe('variant')
    expect(editorFor({ oneOf: [{ const: 'a' }, { const: 'b' }] })).toBe('select')
    expect(enumOptions({ oneOf: [{ const: 'a' }, { const: 'b' }] })).toEqual(['a', 'b'])
    expect(editorFor({ type: 'object' })).toBe('json')
  })

  it('reads applies-to against the algorithm', () => {
    expect(appliesTo({ 'x-retrograd-applies-to': ['grpo'] }, 'sft')).toBe(false)
    expect(appliesTo({ 'x-retrograd-applies-to': ['grpo'] }, 'grpo')).toBe(true)
    expect(appliesTo({}, 'sft')).toBe(true)
  })

  it('starts a new section with its required fields', () => {
    const evaluation = unwrapNullable({ $ref: '#/$defs/EvaluationToml' }, schema).schema
    expect(initialValue(evaluation, schema)).toEqual({ dataset: '', every_iterations: 1 })
  })
})

describe('crossSchema', () => {
  const sections = crossSchema({
    schema,
    effective,
    provenance: {
      'training.lr': { source: 'derived', reason: 'scaled to the batch' },
      'training.epochs': { source: 'override' },
    },
    params: { training: { epochs: 3 } },
    defaults: [
      { path: 'training.lr', rule: 'lr by rank', applies_to: 'all', client_can_set: true },
    ],
    algorithm: 'sft',
  })
  const byName = new Map(sections.map((section) => [section.name, section]))

  it('hides sections for other algorithms and fields the operator declares', () => {
    expect(byName.has('grpo')).toBe(false)
    const grpo = crossSchema({
      schema,
      effective: { run: { algorithm: 'grpo' } },
      algorithm: 'grpo',
    })
    const rows = grpo.find((section) => section.name === 'grpo')?.rows.map((row) => row.path)
    expect(rows).toEqual(['grpo.group_size'])
  })

  it('crosses each row with its value, origin, rule and override', () => {
    const lr = byName.get('training')?.rows.find((row) => row.path === 'training.lr')
    expect(lr).toMatchObject({
      value: 0.0002,
      present: true,
      origin: { source: 'derived', reason: 'scaled to the batch' },
      override: false,
      patchable: true,
      editor: 'number',
      nullable: true,
      label: 'lr',
    })
    expect(lr?.rule?.rule).toBe('lr by rank')
    const epochs = byName.get('training')?.rows.find((row) => row.path === 'training.epochs')
    expect(epochs).toMatchObject({ override: true, paramValue: 3 })
  })

  it('lists a section the plan did not configure, as not present', () => {
    const evaluation = byName.get('evaluation')
    expect(evaluation?.present).toBe(false)
    expect(evaluation?.rows.every((row) => !row.present)).toBe(true)
    expect(evaluation?.required.map((field) => field.segments.join('.'))).toEqual([
      'dataset',
      'every_iterations',
    ])
  })

  it('works from the effective configuration alone when there is no schema', () => {
    const rows = crossSchema({ schema: null, effective }).flatMap((section) =>
      section.rows.map((row) => row.path),
    )
    expect(rows).toEqual([
      'run.algorithm',
      'training.lr',
      'training.epochs',
      'training.packing',
      'training.targets',
    ])
  })

  it('filters by search, pinned rows and configured rows', () => {
    expect(
      filterSections(sections, { search: 'learning' }).flatMap((s) => s.rows.map((r) => r.path)),
    ).toEqual(['training.lr'])
    expect(
      filterSections(sections, { overridesOnly: true }).flatMap((s) => s.rows.map((r) => r.path)),
    ).toEqual(['training.epochs'])
    expect(
      filterSections(sections, { presentOnly: true }).some((s) => s.name === 'evaluation'),
    ).toBe(false)
  })
})
