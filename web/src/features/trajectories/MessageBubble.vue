<script setup lang="ts">
import { computed } from 'vue'
import type { Message, StepReward } from '@/api/types'
import { MarkdownText } from '@/features/shared'
import StepRewardBar from './StepRewardBar.vue'
import ToolCallBlock from './ToolCallBlock.vue'

const FOLD_LINES = 20

const props = withDefaults(
  defineProps<{
    message: Message
    steps?: StepReward[]
    /** Tool results by call id, so a call shows its answer. */
    results?: Map<string, Message>
    /** Tool call names by id, for a tool message's header. */
    callNames?: Map<string, string>
    markdown?: boolean
  }>(),
  { steps: () => [], results: undefined, callNames: undefined, markdown: false },
)

const content = computed(() => props.message.content ?? '')
const foldable = computed(
  () => props.message.role === 'tool' && content.value.split('\n').length > FOLD_LINES,
)
const toolName = computed(() =>
  props.message.tool_call_id
    ? (props.callNames?.get(props.message.tool_call_id) ?? 'unknown call')
    : null,
)
const color = computed(() => {
  switch (props.message.role) {
    case 'assistant':
      return 'primary'
    case 'tool':
      return props.message.is_error ? 'error' : 'warning'
    case 'system':
      return undefined
    default:
      return 'secondary'
  }
})
</script>

<template>
  <div class="bubble mb-2" :class="[`role-${message.role}`, { error: message.is_error }]">
    <div class="d-flex align-center flex-wrap ga-1 mb-1">
      <v-chip :color="color" size="x-small" label class="text-uppercase">{{ message.role }}</v-chip>
      <span v-if="message.role === 'tool'" class="mono text-caption">
        {{ toolName }}<template v-if="message.tool_call_id"> ({{ message.tool_call_id }})</template>
      </span>
      <v-chip v-if="message.is_error" color="error" size="x-small">error</v-chip>
      <StepRewardBar v-if="steps.length" :steps="steps" />
    </div>
    <details v-if="foldable">
      <summary class="text-caption rg-clickable">
        {{ content.split('\n').slice(0, 2).join(' ⏎ ') }} … ({{ content.split('\n').length }} lines)
      </summary>
      <pre class="text-pre">{{ content }}</pre>
    </details>
    <template v-else-if="content">
      <MarkdownText v-if="markdown && message.role !== 'tool'" :text="content" />
      <pre v-else class="text-pre" :class="{ 'text-error': message.is_error }">{{ content }}</pre>
    </template>
    <ToolCallBlock
      v-for="call in message.tool_calls ?? []"
      :key="call.id"
      :call="call"
      :result="results ? (results.get(call.id) ?? null) : undefined"
    />
  </div>
</template>

<style scoped>
.bubble {
  border-radius: 8px;
  padding: 8px 10px;
  background: rgba(var(--v-theme-on-surface), 0.035);
  border-left: 3px solid transparent;
}
.role-assistant {
  border-left-color: rgb(var(--v-theme-primary));
}
.role-user {
  border-left-color: rgb(var(--v-theme-secondary));
}
.role-tool {
  border-left-color: rgb(var(--v-theme-warning));
}
.bubble.error {
  border-left-color: rgb(var(--v-theme-error));
  background: rgba(var(--v-theme-error), 0.06);
}
</style>
