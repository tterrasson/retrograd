<script setup lang="ts">
import type { DatasetView } from '@/api/types'
import { formatBytes, formatNumber } from '@/utils/format'
import { RelativeTime } from '@/features/shared'

defineProps<{ datasets: DatasetView[]; loading?: boolean; selectable?: boolean }>()
const selected = defineModel<string | null>('selected', { default: null })
const emit = defineEmits<{ open: [DatasetView] }>()

const headers = [
  { title: 'Name', key: 'name' },
  { title: 'Format', key: 'format' },
  { title: 'Examples', key: 'examples', align: 'end' as const },
  { title: 'Size', key: 'bytes', align: 'end' as const },
  { title: 'Tokens p50 / p99', key: 'stats', sortable: false },
  { title: 'Uploaded', key: 'created_at' },
]

function click(_event: unknown, row: { item: DatasetView }) {
  selected.value = row.item.id
  emit('open', row.item)
}
</script>

<template>
  <v-data-table
    :headers="headers"
    :items="datasets"
    :loading="loading"
    item-value="id"
    :row-props="
      ({ item }: { item: DatasetView }) => ({
        class: item.id === selected ? 'bg-surface-light' : '',
      })
    "
    :sort-by="[{ key: 'created_at', order: 'desc' }]"
    items-per-page="10"
    class="rg-datasets"
    @click:row="click"
  >
    <template #[`item.name`]="{ item }">
      <v-radio-group
        v-if="selectable"
        :model-value="selected"
        hide-details
        density="compact"
        class="d-inline-flex"
      >
        <v-radio :value="item.id" :aria-label="`select ${item.name ?? item.id}`" />
      </v-radio-group>
      <span class="font-weight-medium">{{ item.name ?? '–' }}</span>
      <div class="mono text-caption rg-muted">{{ item.id.slice(0, 12) }}</div>
    </template>
    <template #[`item.format`]="{ item }"
      ><span class="mono">{{ item.format }}</span></template
    >
    <template #[`item.examples`]="{ item }"
      ><span class="mono">{{ formatNumber(item.examples) }}</span></template
    >
    <template #[`item.bytes`]="{ item }"
      ><span class="mono">{{ formatBytes(item.bytes) }}</span></template
    >
    <template #[`item.stats`]="{ item }">
      <span class="mono"
        >{{ formatNumber(item.stats.p50) }} / {{ formatNumber(item.stats.p99) }}</span
      >
      <span v-if="!item.stats.measured" class="text-caption rg-muted"> est.</span>
    </template>
    <template #[`item.created_at`]="{ item }"><RelativeTime :at="item.created_at" /></template>
    <template #no-data><div class="pa-6 rg-muted">No dataset uploaded yet.</div></template>
  </v-data-table>
</template>

<style scoped>
.rg-datasets :deep(tbody tr) {
  cursor: pointer;
}
</style>
