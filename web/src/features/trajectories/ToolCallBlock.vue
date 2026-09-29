<script setup lang="ts">
import { computed } from 'vue'
import type { Message, ToolCall } from '@/api/types'

const props = defineProps<{ call: ToolCall; result?: Message | null }>()
const args = computed(() => JSON.stringify(props.call.arguments, null, 2))
const lines = computed(() => (props.result?.content ?? '').split('\n').length)
</script>

<template>
  <details class="tool-call my-1" :class="{ error: result?.is_error }">
    <summary>
      <span class="mono font-weight-medium">{{ call.name }}</span>
      <span class="mono text-caption rg-muted ml-1">{{ call.id }}</span>
      <v-chip v-if="result?.is_error" color="error" size="x-small" class="ml-2">error</v-chip>
      <span v-else-if="result === null" class="text-caption rg-muted ml-2">no result</span>
    </summary>
    <div class="text-caption rg-muted mt-1">arguments</div>
    <pre class="text-pre rg-code">{{ args }}</pre>
    <template v-if="result">
      <div class="text-caption mt-1" :class="result.is_error ? 'text-error' : 'rg-muted'">
        result<template v-if="lines > 1"> · {{ lines }} lines</template>
      </div>
      <pre class="text-pre rg-code" :class="{ 'text-error': result.is_error }">{{
        result.content
      }}</pre>
    </template>
  </details>
</template>

<style scoped>
.tool-call {
  border-left: 3px solid rgba(var(--v-theme-primary), 0.5);
  padding-left: 8px;
}
.tool-call.error {
  border-left-color: rgb(var(--v-theme-error));
}
.tool-call summary {
  cursor: pointer;
}
</style>
