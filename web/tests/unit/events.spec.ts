import { describe, expect, it } from 'vitest'
import { createRunEventState, MetricSeries, reduceRunEvent, RingBuffer } from '@/api/events'
import type { RunEvent } from '@/api/types'

const metrics = (seq: number, values: Record<string, number>): RunEvent => ({
  type: 'metrics',
  seq,
  at: seq * 1000,
  iteration: seq,
  global_step: seq * 10,
  values,
})

describe('MetricSeries', () => {
  it('keeps one column per name, aligned, with null where a sample lacks it', () => {
    const series = new MetricSeries()
    series.push({ seq: 1, at: 0, iteration: 1, global_step: 10, values: { loss: 2 } })
    series.push({ seq: 2, at: 0, iteration: 2, global_step: 20, values: { loss: 1.5, lr: 0.1 } })
    series.push({ seq: 3, at: 0, iteration: 3, global_step: 30, values: { lr: 0.05 } })
    expect(series.values.get('loss')).toEqual([2, 1.5, null])
    expect(series.values.get('lr')).toEqual([null, 0.1, 0.05])
    expect(series.names()).toEqual(['loss', 'lr'])
    expect(series.latest('loss')).toBe(1.5)
  })

  it('places a late sample by seq and never twice', () => {
    const series = new MetricSeries()
    series.push({ seq: 1, at: 0, iteration: 1, global_step: 1, values: { loss: 3 } })
    series.push({ seq: 5, at: 0, iteration: 5, global_step: 5, values: { loss: 1 } })
    series.push({ seq: 3, at: 0, iteration: 3, global_step: 3, values: { loss: 2 } })
    series.push({ seq: 3, at: 0, iteration: 3, global_step: 3, values: { loss: 2 } })
    expect(series.seq).toEqual([1, 3, 5])
    expect(series.values.get('loss')).toEqual([3, 2, 1])
  })
})

describe('RingBuffer', () => {
  it('drops the oldest past its capacity and counts them', () => {
    const buffer = new RingBuffer<number>(3)
    for (let i = 0; i < 5; i++) buffer.push(i)
    expect(buffer.toArray()).toEqual([2, 3, 4])
    expect(buffer.droppedCount).toBe(2)
  })
})

describe('reduceRunEvent', () => {
  it('folds each seq once, so a replay after a reconnect duplicates nothing', () => {
    const state = createRunEventState()
    expect(reduceRunEvent(state, metrics(1, { loss: 1 }))).toBe(true)
    expect(reduceRunEvent(state, { type: 'log', seq: 2, at: 0, message: 'hello' })).toBe(true)
    expect(reduceRunEvent(state, metrics(1, { loss: 1 }))).toBe(false)
    expect(reduceRunEvent(state, { type: 'log', seq: 2, at: 0, message: 'hello' })).toBe(false)
    expect(state.series.length).toBe(1)
    expect(state.journal.size).toBe(1)
    expect(state.lastSeq.value).toBe(2)
  })

  it('records status, checkpoints and evaluations, with markers at the last step', () => {
    const state = createRunEventState()
    reduceRunEvent(state, metrics(1, { loss: 1 }))
    reduceRunEvent(state, { type: 'status', seq: 2, at: 0, status: 'running' })
    reduceRunEvent(state, { type: 'checkpoint', seq: 3, at: 0, path: '/runs/x/ckpt/step-10' })
    reduceRunEvent(state, {
      type: 'evaluation',
      seq: 4,
      at: 0,
      iteration: 1,
      improved: true,
      best: 0.5,
      stale: 0,
      keep_training: true,
    })
    reduceRunEvent(state, { type: 'terminal', seq: 5, at: 0, status: 'failed', error: 'boom' })
    expect(state.status.value).toBe('failed')
    expect(state.error.value).toBe('boom')
    expect(
      state.markers.value.map((marker) => [marker.kind, marker.globalStep, marker.label]),
    ).toEqual([
      ['checkpoint', 10, 'step-10'],
      ['evaluation', 10, 'eval 1'],
    ])
    expect(state.journal.toArray().map((event) => event.type)).toEqual([
      'status',
      'checkpoint',
      'evaluation',
      'terminal',
    ])
  })
})
