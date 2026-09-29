<script setup lang="ts">
import { mdiDownload } from '@mdi/js'
import type { ArtifactEntry } from '@/api/types'
import { formatBytes } from '@/utils/format'
import { CopyButton } from '@/features/shared'

defineProps<{ artifacts: ArtifactEntry[]; loading?: boolean; downloading?: string | null }>()
const emit = defineEmits<{ download: [string]; checkpoints: [] }>()

const headers = [
  { title: 'Name', key: 'name' },
  { title: 'Kind', key: 'kind' },
  { title: 'Present', key: 'present' },
  { title: 'Size', key: 'bytes', align: 'end' as const },
  { title: '', key: 'actions', sortable: false, align: 'end' as const },
]
</script>

<template>
  <v-data-table
    :headers="headers"
    :items="artifacts"
    :loading="loading"
    item-value="name"
    items-per-page="-1"
    hide-default-footer
  >
    <template #[`item.name`]="{ item }"
      ><span class="mono">{{ item.name }}</span></template
    >
    <template #[`item.present`]="{ item }">
      <v-chip :color="item.present ? 'success' : undefined">{{
        item.present ? 'present' : 'absent'
      }}</v-chip>
    </template>
    <template #[`item.bytes`]="{ item }"
      ><span class="mono">{{ formatBytes(item.bytes) }}</span></template
    >
    <template #[`item.actions`]="{ item }">
      <div class="d-flex align-center justify-end ga-1">
        <v-btn
          v-if="item.downloadable && item.present"
          size="small"
          color="primary"
          variant="tonal"
          :prepend-icon="mdiDownload"
          :loading="downloading === item.name"
          @click="emit('download', item.name)"
        >
          Download
        </v-btn>
        <template v-else-if="item.present">
          <v-btn
            v-if="item.name === 'checkpoints'"
            size="small"
            variant="text"
            @click="emit('checkpoints')"
          >
            Open checkpoints
          </v-btn>
          <span
            class="mono text-caption rg-muted text-truncate"
            style="max-width: 22rem"
            :title="item.path"
            >{{ item.path }}</span
          >
          <CopyButton :text="item.path" label="Copy the server path" />
        </template>
      </div>
    </template>
    <template #no-data><div class="pa-6 rg-muted">This run has no artifact.</div></template>
  </v-data-table>
</template>
