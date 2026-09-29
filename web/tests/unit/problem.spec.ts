import { describe, expect, it } from 'vitest'
import { ApiProblem, memoryShortfall, problemFromBody, problemFromResponse } from '@/api/problem'
import { messagesAt, problemFields } from '@/composables/useProblemFields'

const invalid = {
  type: 'https://retrograd.dev/problems/invalid-request',
  title: 'invalid request',
  status: 422,
  detail: 'the request does not validate',
  trace_id: 'abc123',
  errors: [
    { pointer: '/recipe/model', code: 'path_not_found', message: 'no such file' },
    {
      pointer: '/params/training/lr',
      code: 'out_of_range',
      message: 'must be positive',
      hint: 'try 1e-4',
    },
    { pointer: '/fork_from', code: 'override_conflict', message: 'cannot fork' },
  ],
}

describe('ApiProblem', () => {
  it('reads a problem document, its slug and its trace id', () => {
    const problem = problemFromBody(422, invalid)
    expect(problem).toBeInstanceOf(ApiProblem)
    expect(problem.kind).toBe('invalid-request')
    expect(problem.traceId).toBe('abc123')
    expect(problem.errors).toHaveLength(3)
    expect(problem.fieldErrors('/params')).toHaveLength(1)
  })

  it('reads an OpenAI error envelope', () => {
    const problem = problemFromBody(404, {
      error: { message: 'no model x', type: 'invalid_request_error', code: 'model_not_found' },
    })
    expect(problem.kind).toBe('openai')
    expect(problem.detail).toBe('no model x')
    expect(problem.status).toBe(404)
  })

  it('keeps a body that is neither as its detail', async () => {
    const problem = await problemFromResponse(
      new Response('gateway exploded', { status: 502, statusText: 'Bad Gateway' }),
    )
    expect(problem.status).toBe(502)
    expect(problem.title).toBe('Bad Gateway')
    expect(problem.detail).toBe('gateway exploded')
  })

  it('reads the dominant posts of insufficient-memory from meta', () => {
    const problem = problemFromBody(422, {
      type: 'https://retrograd.dev/problems/insufficient-memory',
      title: 'insufficient memory',
      status: 422,
      detail: '10 bytes over',
      meta: {
        overflow_bytes: 10,
        vram_budget_bytes: 100,
        dominant_posts: [{ post: 'activations', bytes: 80 }],
        levers_applied: ['gradient_checkpointing'],
        unlocks: ['truncate_context'],
      },
    })
    expect(memoryShortfall(problem)).toEqual({
      overflowBytes: 10,
      vramBudgetBytes: 100,
      ramBudgetBytes: undefined,
      measuredDeviceBytes: undefined,
      dominantPosts: [{ post: 'activations', bytes: 80 }],
      leversApplied: ['gradient_checkpointing'],
      unlocks: ['truncate_context'],
    })
    expect(memoryShortfall(problemFromBody(422, invalid))).toBeNull()
  })
})

describe('pointer to field', () => {
  it('maps pointers under a prefix to dotted fields and leaves the rest unclaimed', () => {
    const problem = problemFromBody(422, invalid)
    const params = problemFields(problem, '/params')
    expect(params.fields).toEqual({ 'training.lr': ['must be positive (try 1e-4)'] })
    expect(params.unclaimed.map((error) => error.pointer)).toEqual(['/recipe/model', '/fork_from'])
    expect(problemFields(problem, '/recipe').fields).toEqual({ model: ['no such file'] })
  })

  it('lets a form claim only the fields it shows', () => {
    const problem = problemFromBody(422, invalid)
    const fields = problemFields(problem, '/params', (path) => path === 'lora.rank')
    expect(fields.fields).toEqual({})
    expect(fields.unclaimed).toHaveLength(3)
  })

  it('answers the messages at a pointer and below it', () => {
    const problem = problemFromBody(422, invalid)
    expect(messagesAt(problem, '/params/training')).toEqual(['must be positive (try 1e-4)'])
    expect(messagesAt(new Error('x'), '/recipe')).toEqual([])
  })
})
