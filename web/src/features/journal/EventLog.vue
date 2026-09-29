<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { mdiArrowCollapseDown } from '@mdi/js'
import type { JournalEvent, RingBuffer } from '@/api/events'
import { CopyButton, EmptyState } from '@/features/shared'
import EventRow from './EventRow.vue'

const props = defineProps<{ journal: RingBuffer<JournalEvent>; revision: number }>()
const follow = defineModel<boolean>('follow', { default: true })

const kinds = ['log', 'memory', 'status', 'checkpoint', 'evaluation', 'terminal'] as const
const enabled = ref<string[]>([...kinds])

const events = computed(() => {
  void props.revision
  const wanted = new Set(enabled.value)
  return props.journal.toArray().filter((event) => wanted.has(event.type))
})

const scroller = ref<{ scrollToIndex(index: number): void } | null>(null)
watch(
  () => [events.value.length, follow.value] as const,
  async ([length, following]) => {
    if (!following || !length) return
    await nextTick()
    scroller.value?.scrollToIndex(length - 1)
  },
)

const asText = computed(() =>
  events.value
    .map((event) => `${new Date(event.at).toISOString()} ${event.type} ${JSON.stringify(event)}`)
    .join('\n'),
)
</script>

<template>
  <v-card>
    <div class="d-flex align-center flex-wrap ga-2 pa-2">
      <v-chip-group v-model="enabled" multiple aria-label="Event types">
        <v-chip
          v-for="kind in kinds"
          :key="kind"
          :value="kind"
          filter
          size="small"
          variant="outlined"
          >{{ kind }}</v-chip
        >
      </v-chip-group>
      <v-spacer />
      <span class="text-caption rg-muted">
        {{ events.length }} lines<template v-if="journal.droppedCount">
          · {{ journal.droppedCount }} older lines dropped</template
        >
      </span>
      <v-btn
        :icon="mdiArrowCollapseDown"
        size="small"
        :variant="follow ? 'tonal' : 'text'"
        :color="follow ? 'primary' : undefined"
        :aria-pressed="follow"
        aria-label="Follow the latest line"
        title="Follow the latest line"
        @click="follow = !follow"
      />
      <CopyButton :text="asText" label="Copy the journal" />
    </div>
    <v-divider />
    <v-virtual-scroll
      v-if="events.length"
      ref="scroller"
      :items="events"
      height="calc(100vh - 320px)"
      item-key="seq"
      role="log"
      aria-live="off"
    >
      <template #default="{ item }">
        <EventRow :event="item" />
      </template>
    </v-virtual-scroll>
    <EmptyState v-else title="No events yet" text="Lines appear here as the run emits them." />
  </v-card>
</template>
