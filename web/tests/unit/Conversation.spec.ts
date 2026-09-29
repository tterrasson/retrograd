import { describe, expect, it } from 'vitest'
import { Conversation } from '@/features/trajectories'
import type { MemberView } from '@/api/trajectories'
import type { ToolCall } from '@/api/types'
import { mountWithPlugins } from './mount'

const args = { query: 'weather' } as unknown as ToolCall['arguments']

const member: MemberView = {
  member: 3,
  seed: 7,
  tokens: 120,
  truncated: false,
  turns: 2,
  tool_calls: 1,
  tool_errors: 1,
  reward: 0.25,
  prefix: true,
  messages: [
    { role: 'user', content: 'What is the weather?', is_error: false },
    {
      role: 'assistant',
      content: 'Let me look.',
      is_error: false,
      tool_calls: [{ id: 'call-1', name: 'search', arguments: args }],
    },
    { role: 'tool', content: 'service unavailable', tool_call_id: 'call-1', is_error: true },
    { role: 'assistant', content: 'I could not find out.', is_error: false },
  ],
  step_rewards: [
    { step_index: 0, kind: 'tool', reward: -0.5, message_indices: [1] },
    { step_index: 1, kind: 'format', reward: 0.25, message_indices: [] },
  ],
  terminal_reward_raw: 0.75,
  judge_explanation: 'Honest about the failure.',
  metadata: { summary: 'no answer' },
}

describe('Conversation', () => {
  const wrapper = mountWithPlugins(Conversation, { props: { member } })

  it('pairs each tool result with its call, by tool_call_id', () => {
    const call = wrapper.find('.tool-call')
    expect(call.exists()).toBe(true)
    expect(call.text()).toContain('search')
    expect(call.text()).toContain('service unavailable')
    expect(call.text()).toContain('"query": "weather"')
    expect(call.classes()).toContain('error')
  })

  it('draws a result once, inside its call, not again as a message', () => {
    expect(wrapper.findAll('.bubble')).toHaveLength(3)
  })

  it('puts step rewards beside the messages they name, and lists the others', () => {
    const bubbles = wrapper.findAll('.bubble')
    expect(bubbles[1]?.text()).toMatch(/step 0\s*-0\.500/)
    expect(bubbles[0]?.text()).not.toMatch(/step 0/)
    expect(wrapper.text()).toMatch(/step rewards without a message:\s*step 1 \(format\) 0\.250/)
  })

  it('shows the terminal reward, the judge and the prefix note', () => {
    expect(wrapper.text()).toContain('terminal reward')
    expect(wrapper.text()).toContain('0.750')
    expect(wrapper.text()).toContain('Honest about the failure.')
    expect(wrapper.text()).toContain("starts after the scenario's prefix")
  })
})
