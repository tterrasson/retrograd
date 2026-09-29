import { describe, expect, it } from 'vitest'
import { groupParam, memberView, segmentStarts, timelineSeries } from '@/api/trajectories'
import type { MemberSummary, Message, ToolCall, UpdateSummary } from '@/api/types'
import {
  acceptsSummary,
  byReward,
  compactLine,
  duplicates,
  emptyFilters,
  finalAnswer,
  formatMemberRef,
  memberText,
  parseMemberRef,
} from '@/features/trajectories'

const summary = (member: number, fields: Partial<MemberSummary> = {}): MemberSummary => ({
  member,
  seed: member,
  tokens: 10,
  truncated: false,
  turns: 0,
  tool_calls: 0,
  tool_errors: 0,
  ...fields,
})

const update = (
  n: number,
  segment: number,
  metrics: Record<string, number>,
  texts = true,
): UpdateSummary => ({
  update: n,
  segment,
  status: 'completed',
  metrics,
  groups: 1,
  rollouts: 2,
  texts,
})

describe('timeline', () => {
  it('charts only the metrics updates carry, with the reward band', () => {
    const series = timelineSeries([
      update(1, 0, { 'reward/mean': 0.5, 'reward/std': 0.1 }),
      update(2, 0, { 'reward/mean': 0.7, 'reward/std': 0.2, 'policy/kl': 0.01 }, false),
    ])
    expect(series.map((item) => item.key)).toEqual(['reward/mean', 'policy/kl'])
    expect(series[0]?.points).toEqual([
      { update: 1, value: 0.5, spread: 0.1, texts: true },
      { update: 2, value: 0.7, spread: 0.2, texts: false },
    ])
  })

  it('marks where a resumed segment starts', () => {
    expect(segmentStarts([update(1, 0, {}), update(2, 0, {}), update(3, 1, {})])).toEqual([3])
  })
})

describe('members', () => {
  it('flattens a detail into one view', () => {
    const view = memberView({
      summary: summary(1, { reward: 1 }),
      conversation: {
        messages: [{ role: 'assistant', content: 'hi', is_error: false }],
        prefix: true,
        step_rewards: [],
        metadata: {},
      },
    })
    expect(view.member).toBe(1)
    expect(view.messages?.[0]?.content).toBe('hi')
    expect(view.prefix).toBe(true)
  })

  it('names a group path segment, PPO included', () => {
    expect(groupParam(3)).toBe('3')
    expect(groupParam(null)).toBe('-')
    expect(groupParam(undefined)).toBe('-')
  })

  it('reads the texts of a conversation', () => {
    const args = { q: 'x' } as unknown as ToolCall['arguments']
    const messages: Message[] = [
      { role: 'user', content: 'find it', is_error: false },
      {
        role: 'assistant',
        content: '',
        is_error: false,
        tool_calls: [{ id: 'c1', name: 'search', arguments: args }],
      },
      { role: 'tool', content: 'nothing', tool_call_id: 'c1', is_error: true },
      { role: 'assistant', content: 'done', is_error: false },
    ]
    const member = { ...summary(0), messages }
    expect(compactLine(member.messages)).toBe(
      'user → assistant[search] → tool(search)! → assistant',
    )
    expect(finalAnswer(member)).toBe('done')
    expect(memberText(member)).toContain('search {"q":"x"}')
    expect(finalAnswer({ completion: 'plain' })).toBe('plain')
  })

  it('finds members that said the same thing', () => {
    const members = [
      { ...summary(0), completion: 'a' },
      { ...summary(1), completion: 'b' },
      { ...summary(2), completion: 'a' },
    ]
    expect([...duplicates(members)]).toEqual([0, 2])
  })
})

describe('filters', () => {
  it('filters on what the summary says', () => {
    const filters = emptyFilters()
    expect(acceptsSummary(filters, summary(0))).toBe(true)
    expect(acceptsSummary({ ...filters, trainedOnly: true }, summary(0, { trained: null }))).toBe(
      false,
    )
    expect(acceptsSummary({ ...filters, flaggedOnly: true }, summary(0, { truncated: true }))).toBe(
      true,
    )
    expect(acceptsSummary({ ...filters, flaggedOnly: true }, summary(0))).toBe(false)
    expect(
      acceptsSummary(
        { ...filters, skipReason: 'zero_signal' },
        summary(0, { skip_reason: 'zero_signal' }),
      ),
    ).toBe(true)
    expect(
      acceptsSummary({ ...filters, toolErrorsOnly: true }, summary(0, { tool_errors: 1 })),
    ).toBe(true)
    expect(acceptsSummary({ ...filters, rewardMin: 0.5 }, summary(0, { reward: 0.4 }))).toBe(false)
    expect(acceptsSummary({ ...filters, rewardMax: 0.5 }, summary(0))).toBe(false)
  })

  it('sorts by reward, a missing one last', () => {
    expect(
      byReward([summary(0, { reward: 0.1 }), summary(1), summary(2, { reward: 0.9 })]).map(
        (m) => m.member,
      ),
    ).toEqual([2, 0, 1])
  })

  it('writes and reads a member reference', () => {
    const ref = { update: 12, group: '-', member: 5 }
    expect(parseMemberRef(formatMemberRef(ref))).toEqual(ref)
    expect(parseMemberRef('nonsense')).toBeNull()
  })
})
