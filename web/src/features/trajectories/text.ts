import type { MemberView } from '@/api/trajectories'
import type { Message } from '@/api/types'

/** Everything a member said, as one string: what a text search looks through. */
export function memberText(member: Pick<MemberView, 'completion' | 'messages'>): string {
  if (member.completion !== undefined && member.completion !== null) return member.completion
  return (member.messages ?? [])
    .map((message) =>
      [message.content ?? '']
        .concat(
          (message.tool_calls ?? []).map(
            (call) => `${call.name} ${JSON.stringify(call.arguments)}`,
          ),
        )
        .join('\n'),
    )
    .join('\n')
}

/** The final answer of a member: its completion, or the last assistant message. */
export function finalAnswer(member: Pick<MemberView, 'completion' | 'messages'>): string {
  if (member.completion !== undefined && member.completion !== null) return member.completion
  const messages = member.messages ?? []
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]
    if (message?.role === 'assistant' && message.content) return message.content
  }
  return ''
}

/** `system → user → assistant[search] → tool(search)! → assistant`. */
export function compactLine(messages: readonly Message[]): string {
  const names = new Map<string, string>()
  const parts: string[] = []
  for (const message of messages) {
    if (message.role === 'tool') {
      const name = (message.tool_call_id && names.get(message.tool_call_id)) || '?'
      parts.push(`tool(${name})${message.is_error ? '!' : ''}`)
    } else if (message.role === 'assistant') {
      for (const call of message.tool_calls ?? []) names.set(call.id, call.name)
      const calls = (message.tool_calls ?? []).map((call) => call.name)
      parts.push(calls.length ? `assistant[${calls.join(', ')}]` : 'assistant')
    } else {
      parts.push(message.role)
    }
  }
  return parts.join(' → ')
}

/** Tool names called anywhere in these members. */
export function toolNames(members: readonly Pick<MemberView, 'messages'>[]): string[] {
  const names = new Set<string>()
  for (const member of members) {
    for (const message of member.messages ?? []) {
      for (const call of message.tool_calls ?? []) names.add(call.name)
    }
  }
  return [...names].sort()
}

/** Members whose full text another member of the same group also produced. */
export function duplicates(members: readonly MemberView[]): Set<number> {
  const counts = new Map<string, number>()
  for (const member of members) {
    const text = memberText(member)
    counts.set(text, (counts.get(text) ?? 0) + 1)
  }
  return new Set(
    members
      .filter((member) => (counts.get(memberText(member)) ?? 0) > 1)
      .map((member) => member.member),
  )
}
