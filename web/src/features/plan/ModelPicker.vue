<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { mdiMagnify, mdiRefresh } from '@mdi/js'
import { useModelFiles } from '@/api/discovery'
import { usePreflight, type PreflightReport } from '@/api/plan'
import type { ModelFile } from '@/api/types'
import { formatBytes, formatNumber } from '@/utils/format'
import { JsonTree, ProblemAlert, RelativeTime } from '@/features/shared'

defineProps<{ errors?: string[] }>()
const model = defineModel<string>({ required: true })

const listing = useModelFiles()
const preflight = usePreflight()
const report = ref<PreflightReport | null>(null)
const preflightError = ref<unknown>(null)
const search = ref('')
const showAll = ref(false)
const manual = ref(false)
const refreshing = ref(false)

const files = computed<ModelFile[]>(() =>
  (listing.data.value?.files ?? []).filter((file) => showAll.value || file.role === 'model'),
)
const headers = [
  { title: 'File', key: 'relative' },
  { title: 'Root', key: 'root' },
  { title: 'Size', key: 'bytes', align: 'end' as const },
  { title: 'Modified', key: 'modified_at' },
  { title: 'Role', key: 'role' },
]

let controller: AbortController | null = null
watch(
  model,
  async (path) => {
    controller?.abort()
    report.value = null
    preflightError.value = null
    if (!path) return
    controller = new AbortController()
    try {
      report.value = await preflight.mutateAsync({ model: path, signal: controller.signal })
    } catch (error) {
      if ((error as { type?: string }).type !== 'aborted') preflightError.value = error
    }
  },
  { immediate: true },
)

async function refresh() {
  refreshing.value = true
  try {
    await listing.refresh()
  } finally {
    refreshing.value = false
  }
}

function pick(_event: unknown, row: { item: ModelFile }) {
  model.value = row.item.path
}
</script>

<template>
  <div>
    <div class="d-flex flex-wrap align-center ga-2 mb-2">
      <v-text-field
        v-model="search"
        :prepend-inner-icon="mdiMagnify"
        label="Filter by name"
        hide-details
        density="compact"
        style="max-width: 360px"
        clearable
      />
      <v-switch
        v-model="showAll"
        label="Projectors and vocab-only files too"
        color="primary"
        hide-details
        density="compact"
      />
      <v-spacer />
      <v-btn :prepend-icon="mdiRefresh" variant="text" :loading="refreshing" @click="refresh"
        >Scan again</v-btn
      >
      <v-btn variant="text" @click="manual = !manual">{{
        manual ? 'Pick from the list' : 'Type a path'
      }}</v-btn>
    </div>
    <ProblemAlert :error="listing.error.value" />
    <v-alert
      v-if="listing.data.value?.truncated"
      type="info"
      variant="tonal"
      density="compact"
      class="mb-2"
    >
      The listing stopped at its limit: some files under
      {{ listing.data.value.roots.join(', ') }} are not shown. Type the path if yours is missing.
    </v-alert>

    <v-text-field
      v-if="manual"
      v-model="model"
      label="Model path on the server"
      class="mono"
      :error-messages="errors"
    />
    <template v-else>
      <v-data-table
        :headers="headers"
        :items="files"
        :search="search"
        :loading="listing.isFetching.value"
        item-value="path"
        density="compact"
        items-per-page="10"
        :row-props="
          ({ item }: { item: ModelFile }) => ({
            class: item.path === model ? 'bg-surface-light' : '',
          })
        "
        class="rg-models"
        @click:row="pick"
      >
        <template #[`item.relative`]="{ item }">
          <v-radio-group :model-value="model" hide-details density="compact" class="d-inline-flex">
            <v-radio
              :value="item.path"
              :aria-label="`use ${item.relative}`"
              @click.stop="model = item.path"
            />
          </v-radio-group>
          <span class="mono">{{ item.relative }}</span>
        </template>
        <template #[`item.root`]="{ item }"
          ><span class="mono text-caption rg-muted">{{ item.root }}</span></template
        >
        <template #[`item.bytes`]="{ item }"
          ><span class="mono">{{ formatBytes(item.bytes) }}</span></template
        >
        <template #[`item.modified_at`]="{ item }"
          ><RelativeTime :at="item.modified_at"
        /></template>
        <template #[`item.role`]="{ item }"
          ><v-chip :color="item.role === 'model' ? 'primary' : undefined">{{
            item.role
          }}</v-chip></template
        >
        <template #no-data>
          <div class="pa-4 rg-muted">
            No GGUF file under {{ listing.data.value?.roots.join(', ') || "the server's roots" }}.
          </div>
        </template>
      </v-data-table>
      <div v-if="errors?.length" class="text-error text-body-2 mt-1" role="alert">
        {{ errors.join('; ') }}
      </div>
    </template>

    <v-card v-if="model" class="mt-3" variant="outlined">
      <v-card-title class="text-subtitle-1"
        >Preflight <span class="mono text-body-2 rg-muted ml-2">{{ model }}</span></v-card-title
      >
      <v-card-text>
        <v-progress-linear v-if="preflight.isPending.value" indeterminate color="primary" />
        <ProblemAlert :error="preflightError" />
        <template v-if="report">
          <v-alert
            v-for="warning in report.warnings ?? []"
            :key="warning.code + warning.message"
            type="warning"
            variant="tonal"
            density="compact"
            class="mb-2"
          >
            {{ warning.message }} <span class="text-caption">({{ warning.code }})</span>
          </v-alert>
          <v-alert
            v-if="!(report.warnings ?? []).length"
            type="success"
            variant="tonal"
            density="compact"
            class="mb-2"
          >
            The engine can load this model on this server.
          </v-alert>
          <v-table v-if="report.kernel_summary?.length" density="compact" class="mb-2">
            <thead>
              <tr>
                <th>op</th>
                <th>backend</th>
                <th>implementation</th>
                <th class="text-end">nodes</th>
                <th class="text-end">bytes</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="kernel in report.kernel_summary"
                :key="kernel.ggml_op + kernel.backend + kernel.implementation"
              >
                <td class="mono">{{ kernel.ggml_op }}</td>
                <td class="mono">{{ kernel.backend }}</td>
                <td class="mono">{{ kernel.implementation }}</td>
                <td class="mono text-end">{{ formatNumber(kernel.nodes) }}</td>
                <td class="mono text-end">{{ formatBytes(kernel.bytes) }}</td>
              </tr>
            </tbody>
          </v-table>
          <details v-if="report.memory">
            <summary class="text-body-2 rg-clickable">Memory report</summary>
            <JsonTree :value="report.memory" :open-depth="1" />
          </details>
        </template>
      </v-card-text>
    </v-card>
  </div>
</template>

<style scoped>
.rg-models :deep(tbody tr) {
  cursor: pointer;
}
</style>
