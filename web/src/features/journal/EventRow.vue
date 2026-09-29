<script setup lang="ts">
import { computed } from 'vue'
import type { JournalEvent } from '@/api/events'
import { formatNumber } from '@/utils/format'

const props = defineProps<{ event: JournalEvent }>()

const time = computed(() => new Date(props.event.at).toLocaleTimeString())
const tone = computed(() => {
  switch (props.event.type) {
    case 'terminal':
      return props.event.error ? 'text-error' : 'text-success'
    case 'memory':
      return 'text-warning'
    case 'checkpoint':
      return 'text-secondary'
    case 'evaluation':
      return 'text-primary'
    default:
      return ''
  }
})
const text = computed(() => {
  const event = props.event
  switch (event.type) {
    case 'log':
      return event.message
    case 'memory':
      return event.note
    case 'status':
      return `status → ${event.status}`
    case 'terminal':
      return `${event.status}${event.error ? `: ${event.error}` : ''}`
    case 'checkpoint':
      return `checkpoint ${event.path}`
    case 'evaluation': {
      const parts = [`iteration ${event.iteration}`]
      if (event.loss != null) parts.push(`loss ${formatNumber(event.loss, 4)}`)
      if (event.perplexity != null) parts.push(`ppl ${formatNumber(event.perplexity, 3)}`)
      if (event.accuracy != null) parts.push(`acc ${formatNumber(event.accuracy, 4)}`)
      if (event.mean_reward != null) parts.push(`reward ${formatNumber(event.mean_reward, 4)}`)
      parts.push(
        `best ${formatNumber(event.best, 4)}`,
        event.improved ? 'improved' : `stale ${event.stale}`,
      )
      if (!event.keep_training) parts.push('stopping')
      return `evaluation ${parts.join(' · ')}`
    }
    default:
      return JSON.stringify(event)
  }
})
</script>

<template>
  <div class="event-row d-flex ga-2 px-2 mono">
    <span class="rg-muted text-no-wrap">{{ time }}</span>
    <span class="kind text-no-wrap" :class="tone">{{ event.type }}</span>
    <span class="message" :class="tone">{{ text }}</span>
  </div>
</template>

<style scoped>
.event-row {
  font-size: 0.8rem;
  line-height: 1.5;
  padding-block: 1px;
}
.kind {
  width: 6.5rem;
  flex: none;
}
.message {
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
