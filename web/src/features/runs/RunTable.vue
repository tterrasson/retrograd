<script setup lang="ts">
import { useRouter } from 'vue-router'
import { runLabel, shortId } from '@/api/runs'
import type { RunSummary } from '@/api/types'
import { formatNumber, formatRate } from '@/utils/format'
import { RelativeTime } from '@/features/shared'
import RunStatusChip from './RunStatusChip.vue'
import RunProgressBar from './RunProgressBar.vue'
import RunControls from './RunControls.vue'

defineProps<{ runs: RunSummary[]; loading?: boolean; itemsPerPage: number }>()

const router = useRouter()
const headers = [
  { title: 'Run', key: 'name', sortable: false },
  { title: 'Algorithm', key: 'algorithm', sortable: false },
  { title: 'Model', key: 'model', sortable: false },
  { title: 'Status', key: 'status', sortable: false },
  { title: 'Progress', key: 'progress', sortable: false, minWidth: '160px' },
  { title: 'Loss / reward', key: 'last', sortable: false },
  { title: 'Throughput', key: 'throughput', sortable: false },
  { title: 'Created', key: 'created_at', sortable: false },
  { title: 'Started', key: 'started_at', sortable: false },
  { title: 'Finished', key: 'finished_at', sortable: false },
  { title: '', key: 'actions', sortable: false, align: 'end' as const },
]

function open(_event: unknown, row: { item: RunSummary }) {
  void router.push(`/runs/${row.item.id}`)
}
</script>

<template>
  <v-data-table-server
    :headers="headers"
    :items="runs"
    :items-length="runs.length"
    :items-per-page="itemsPerPage"
    :loading="loading"
    item-value="id"
    hide-default-footer
    class="rg-runs"
    @click:row="open"
  >
    <template #[`item.name`]="{ item }">
      <router-link :to="`/runs/${item.id}`" class="font-weight-medium" @click.stop>
        {{ runLabel(item) }}
      </router-link>
      <div class="mono text-caption rg-muted">{{ shortId(item.id) }}</div>
    </template>
    <template #[`item.algorithm`]="{ item }">
      <span class="mono">{{ item.algorithm }}</span>
      <div v-if="item.objective" class="text-caption rg-muted">{{ item.objective }}</div>
    </template>
    <template #[`item.model`]="{ item }">
      <span class="mono text-caption" :title="item.model">{{ item.model }}</span>
    </template>
    <template #[`item.status`]="{ item }">
      <RunStatusChip :status="item.status" />
      <div v-if="item.queue_position" class="text-caption rg-muted">
        #{{ item.queue_position }} in queue
      </div>
      <div
        v-if="item.error"
        class="text-caption text-error text-truncate"
        style="max-width: 16rem"
        :title="item.error"
      >
        {{ item.error }}
      </div>
    </template>
    <template #[`item.progress`]="{ item }">
      <RunProgressBar :progress="item.progress" />
    </template>
    <template #[`item.last`]="{ item }">
      <span class="mono">
        <template
          v-if="item.progress.train_loss !== null && item.progress.train_loss !== undefined"
        >
          {{ formatNumber(item.progress.train_loss, 4) }}
        </template>
        <template v-else-if="item.progress.reward !== null && item.progress.reward !== undefined">
          {{ formatNumber(item.progress.reward, 3) }}
        </template>
        <template v-else>–</template>
      </span>
    </template>
    <template #[`item.throughput`]="{ item }">
      <span class="mono">{{ formatRate(item.progress.tokens_per_second) }}</span>
    </template>
    <template #[`item.created_at`]="{ item }"><RelativeTime :at="item.created_at" /></template>
    <template #[`item.started_at`]="{ item }"><RelativeTime :at="item.started_at" /></template>
    <template #[`item.finished_at`]="{ item }"><RelativeTime :at="item.finished_at" /></template>
    <template #[`item.actions`]="{ item }">
      <RunControls :id="item.id" :status="item.status" compact />
    </template>
    <template #no-data>
      <slot name="empty" />
    </template>
  </v-data-table-server>
</template>

<style scoped>
.rg-runs :deep(tbody tr) {
  cursor: pointer;
}
</style>
