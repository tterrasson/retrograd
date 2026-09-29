<script setup lang="ts">
import { computed } from 'vue'
import { useDatasetPreview } from '@/api/datasets'
import { JsonTree, ProblemAlert } from '@/features/shared'

const props = defineProps<{ id: string }>()
const preview = useDatasetPreview(() => props.id)

type Example = unknown
interface ChatMessage {
  role?: string
  content?: unknown
}

function messagesOf(example: Example): ChatMessage[] | null {
  if (typeof example !== 'object' || example === null) return null
  const messages = (example as { messages?: unknown }).messages
  return Array.isArray(messages) ? (messages as ChatMessage[]) : null
}
function pairOf(example: Example): { prompt: unknown; chosen: unknown; rejected: unknown } | null {
  if (typeof example !== 'object' || example === null) return null
  const record = example as Record<string, unknown>
  return 'chosen' in record && 'rejected' in record
    ? { prompt: record.prompt, chosen: record.chosen, rejected: record.rejected }
    : null
}
function text(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) {
    return value
      .map((item) =>
        typeof item === 'object' && item && 'content' in item
          ? `${(item as ChatMessage).role ?? ''}: ${text((item as ChatMessage).content)}`
          : text(item),
      )
      .join('\n')
  }
  return JSON.stringify(value, null, 2)
}
const examples = computed<Example[]>(() => (preview.data.value?.examples ?? []) as Example[])
</script>

<template>
  <div>
    <ProblemAlert :error="preview.error.value" />
    <v-progress-linear v-if="preview.isPending.value" indeterminate color="primary" />
    <div v-if="preview.data.value" class="text-caption rg-muted mb-2">
      format <span class="mono">{{ preview.data.value.format }}</span> · first
      {{ examples.length }} examples, as read from disk
    </div>
    <v-expansion-panels variant="accordion">
      <v-expansion-panel v-for="(example, index) in examples" :key="index">
        <v-expansion-panel-title>
          <span class="mono text-caption mr-2">#{{ index + 1 }}</span>
          <span class="text-truncate text-body-2" style="max-width: 48rem">{{
            text(messagesOf(example)?.[0]?.content ?? pairOf(example)?.prompt ?? example).slice(
              0,
              160,
            )
          }}</span>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <div v-if="messagesOf(example)">
            <div v-for="(message, m) in messagesOf(example)" :key="m" class="mb-2">
              <div class="text-caption text-uppercase rg-muted">{{ message.role }}</div>
              <pre class="text-pre rg-code">{{ text(message.content) }}</pre>
            </div>
          </div>
          <v-row v-else-if="pairOf(example)" dense>
            <v-col cols="12"
              ><div class="text-caption rg-muted">prompt</div>
              <pre class="text-pre rg-code">{{ text(pairOf(example)?.prompt) }}</pre>
            </v-col>
            <v-col cols="12" md="6"
              ><div class="text-caption text-success">chosen</div>
              <pre class="text-pre rg-code">{{ text(pairOf(example)?.chosen) }}</pre>
            </v-col>
            <v-col cols="12" md="6"
              ><div class="text-caption text-error">rejected</div>
              <pre class="text-pre rg-code">{{ text(pairOf(example)?.rejected) }}</pre>
            </v-col>
          </v-row>
          <pre v-else-if="typeof example === 'string'" class="text-pre rg-code">{{ example }}</pre>
          <JsonTree v-else :value="example" :open-depth="2" />
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </div>
</template>
