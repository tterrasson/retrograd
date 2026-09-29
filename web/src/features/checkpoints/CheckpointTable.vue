<script setup lang="ts">
import { mdiCompareHorizontal, mdiSourceFork } from '@mdi/js'
import type { CheckpointEntry } from '@/api/types'
import { formatBytes } from '@/utils/format'
import { CopyButton, RelativeTime } from '@/features/shared'

defineProps<{ checkpoints: CheckpointEntry[]; latest?: string | null; loading?: boolean }>()
const emit = defineEmits<{ fork: [CheckpointEntry]; compare: [CheckpointEntry] }>()

const headers = [
  { title: 'Checkpoint', key: 'id' },
  { title: 'Step', key: 'global_step', align: 'end' as const },
  { title: 'Kind', key: 'kind' },
  { title: 'Size', key: 'bytes', align: 'end' as const },
  { title: 'Written', key: 'written_at' },
  { title: 'Adapter', key: 'adapter' },
  { title: 'Complete', key: 'complete' },
  { title: '', key: 'actions', sortable: false, align: 'end' as const },
]
</script>

<template>
  <v-data-table
    :headers="headers"
    :items="checkpoints"
    :loading="loading"
    item-value="id"
    :sort-by="[{ key: 'global_step', order: 'desc' }]"
    items-per-page="25"
  >
    <template #[`item.id`]="{ item }">
      <span class="mono">{{ item.id }}</span>
      <v-chip v-if="item.id === latest" color="primary" class="ml-2">latest</v-chip>
      <CopyButton :text="item.path" label="Copy the server path" />
    </template>
    <template #[`item.global_step`]="{ item }"
      ><span class="mono">{{ item.global_step }}</span></template
    >
    <template #[`item.bytes`]="{ item }"
      ><span class="mono">{{ formatBytes(item.bytes) }}</span></template
    >
    <template #[`item.written_at`]="{ item }"><RelativeTime :at="item.written_at" /></template>
    <template #[`item.adapter`]="{ item }"
      ><span class="mono text-caption">{{ item.adapter ?? '–' }}</span></template
    >
    <template #[`item.complete`]="{ item }">
      <v-chip :color="item.complete ? 'success' : 'warning'">{{
        item.complete ? 'complete' : 'partial'
      }}</v-chip>
    </template>
    <template #[`item.actions`]="{ item }">
      <v-btn size="small" variant="text" :prepend-icon="mdiSourceFork" @click="emit('fork', item)"
        >Fork</v-btn
      >
      <v-btn
        size="small"
        variant="text"
        :prepend-icon="mdiCompareHorizontal"
        @click="emit('compare', item)"
        >Compare</v-btn
      >
    </template>
    <template #no-data>
      <div class="pa-6 rg-muted">No checkpoint written yet.</div>
    </template>
  </v-data-table>
</template>
