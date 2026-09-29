<script setup lang="ts">
import { computed } from 'vue'
import type { MemberView } from '@/api/trajectories'
import type { Message, StepReward } from '@/api/types'
import { formatNumber } from '@/utils/format'
import { JsonTree, MarkdownText } from '@/features/shared'
import MessageBubble from './MessageBubble.vue'

const props = withDefaults(defineProps<{ member: MemberView; markdown?: boolean }>(), {
  markdown: false,
})

const messages = computed<Message[]>(() => props.member.messages ?? [])
const callNames = computed(() => {
  const names = new Map<string, string>()
  for (const message of messages.value)
    for (const call of message.tool_calls ?? []) names.set(call.id, call.name)
  return names
})
/** Each tool result, by the id of the call it answers. */
const results = computed(() => {
  const map = new Map<string, Message>()
  for (const message of messages.value) {
    if (message.role === 'tool' && message.tool_call_id) map.set(message.tool_call_id, message)
  }
  return map
})
const stepsAt = computed(() => {
  const map = new Map<number, StepReward[]>()
  for (const step of props.member.step_rewards ?? []) {
    for (const index of step.message_indices) {
      const list = map.get(index) ?? []
      list.push(step)
      map.set(index, list)
    }
  }
  return map
})
/** A tool result is drawn inside the call it answers, not a second time. */
function shownWithCall(message: Message): boolean {
  return (
    message.role === 'tool' && !!message.tool_call_id && callNames.value.has(message.tool_call_id)
  )
}
const unattributed = computed(() =>
  (props.member.step_rewards ?? []).filter((step) => !step.message_indices.length),
)
const metadata = computed(() => props.member.metadata ?? null)
const hasMetadata = computed(() => !!metadata.value && Object.keys(metadata.value).length > 0)
</script>

<template>
  <div class="conversation">
    <v-alert
      v-if="member.prefix === false"
      type="info"
      variant="tonal"
      density="compact"
      class="mb-2"
    >
      Full conversation: the scenario prefix did not match.
    </v-alert>
    <v-alert
      v-else-if="member.prefix === true"
      variant="tonal"
      density="compact"
      class="mb-2"
      color="secondary"
    >
      The conversation starts after the scenario's prefix.
    </v-alert>

    <template v-if="member.completion !== undefined && member.completion !== null">
      <MarkdownText v-if="markdown" :text="member.completion" />
      <pre v-else class="text-pre rg-code">{{ member.completion }}</pre>
    </template>
    <template v-for="(message, index) in messages" :key="index">
      <MessageBubble
        v-if="!shownWithCall(message) || stepsAt.has(index)"
        :message="message"
        :steps="stepsAt.get(index) ?? []"
        :results="results"
        :call-names="callNames"
        :markdown="markdown"
      />
    </template>

    <div v-if="unattributed.length" class="text-body-2 mt-1">
      step rewards without a message:
      <span v-for="step in unattributed" :key="step.step_index" class="mono mr-2">
        step {{ step.step_index }} ({{ step.kind }}) {{ formatNumber(step.reward) }}
      </span>
    </div>
    <div
      v-if="member.terminal_reward_raw !== null && member.terminal_reward_raw !== undefined"
      class="text-body-2 mt-1"
    >
      terminal reward <span class="mono">{{ formatNumber(member.terminal_reward_raw) }}</span>
    </div>
    <div v-if="member.judge_explanation" class="mt-2">
      <div class="text-caption rg-muted">judge</div>
      <pre class="text-pre rg-code">{{ member.judge_explanation }}</pre>
    </div>
    <details v-if="hasMetadata" class="mt-2">
      <summary class="text-body-2 rg-clickable">trajectory metadata</summary>
      <JsonTree :value="metadata" :open-depth="2" />
    </details>
  </div>
</template>
