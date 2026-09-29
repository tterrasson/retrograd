<script setup lang="ts">
import { computed, ref } from 'vue'
import { useDatasets } from '@/api/datasets'
import type { DataSpec, DatasetView } from '@/api/types'
import { DatasetPanel, DatasetTable, DatasetUpload } from '@/features/datasets'
import { ProblemAlert } from '@/features/shared'

const props = defineProps<{
  model: string
  /** Messages for `/recipe/data/…` and `/recipe/eval/…`, by field. */
  dataErrors?: Record<string, string[]>
  evalErrors?: Record<string, string[]>
}>()
const data = defineModel<DataSpec>('data', { required: true })
const evalSet = defineModel<DataSpec | null | undefined>('eval', { required: true })

const datasets = useDatasets()
const source = ref<'existing' | 'upload' | 'path'>(data.value.path ? 'path' : 'existing')
const formats = computed(() => datasets.datasets.value.map((item) => item.format))
const selected = computed(
  () => datasets.datasets.value.find((item) => item.id === data.value.dataset) ?? null,
)

function useDataset(id: string | null) {
  data.value = id ? { dataset: id } : {}
}
function uploaded(dataset: DatasetView) {
  useDataset(dataset.id)
  source.value = 'existing'
}
const trainingId = computed({
  get: () => data.value.dataset ?? null,
  set: (id: string | null) => useDataset(id),
})
const path = computed({
  get: () => data.value.path ?? '',
  set: (value: string) =>
    (data.value = value
      ? { path: value, ...(data.value.format ? { format: data.value.format } : {}) }
      : {}),
})
const pathFormat = computed({
  get: () => data.value.format ?? '',
  set: (value: string) => (data.value = { ...data.value, format: value || null }),
})

const evalOptions = computed(() => [
  { title: 'None', value: null },
  ...datasets.datasets.value.map((item) => ({
    title: `${item.name ?? item.id.slice(0, 12)} (${item.format})`,
    value: item.id,
  })),
])
const evalId = computed({
  get: () => evalSet.value?.dataset ?? null,
  set: (id: string | null) => (evalSet.value = id ? { dataset: id } : null),
})
const evalSelected = computed(
  () => datasets.datasets.value.find((item) => item.id === evalId.value) ?? null,
)
const errorsFor = (errors: Record<string, string[]> | undefined, field: string) =>
  errors?.[field] ?? []
const anyDataError = computed(() => Object.values(props.dataErrors ?? {}).flat())
</script>

<template>
  <div>
    <v-tabs v-model="source" color="primary" density="compact" class="mb-3">
      <v-tab value="existing">Uploaded datasets</v-tab>
      <v-tab value="upload">Upload</v-tab>
      <v-tab value="path">Path on the server</v-tab>
    </v-tabs>
    <ProblemAlert :error="datasets.error.value" />
    <v-window v-model="source">
      <v-window-item value="existing">
        <v-card variant="outlined">
          <DatasetTable
            v-model:selected="trainingId"
            :datasets="datasets.datasets.value"
            :loading="datasets.isFetching.value"
            selectable
          />
        </v-card>
        <div
          v-if="errorsFor(dataErrors, 'dataset').length || anyDataError.length"
          class="text-error text-body-2 mt-1"
          role="alert"
        >
          {{
            (errorsFor(dataErrors, 'dataset').length
              ? errorsFor(dataErrors, 'dataset')
              : anyDataError
            ).join('; ')
          }}
        </div>
      </v-window-item>
      <v-window-item value="upload">
        <DatasetUpload :formats="formats" @uploaded="uploaded" />
      </v-window-item>
      <v-window-item value="path">
        <p class="text-body-2 mb-2">
          A file the server can read under its allowed roots. The server says so if it cannot.
        </p>
        <v-row dense>
          <v-col cols="12" md="8">
            <v-text-field
              v-model="path"
              label="data.path"
              class="mono"
              :error-messages="errorsFor(dataErrors, 'path')"
            />
          </v-col>
          <v-col cols="12" md="4">
            <v-combobox
              v-model="pathFormat"
              :items="[...new Set(formats)]"
              label="data.format"
              :error-messages="errorsFor(dataErrors, 'format')"
            />
          </v-col>
        </v-row>
      </v-window-item>
    </v-window>

    <v-card v-if="selected" variant="outlined" class="mt-3">
      <v-card-title class="text-subtitle-1">{{ selected.name ?? selected.id }}</v-card-title>
      <v-card-text><DatasetPanel :dataset="selected" :model="model || null" /></v-card-text>
    </v-card>

    <h3 class="text-subtitle-1 mt-5 mb-2">
      Evaluation set <span class="text-caption rg-muted">(optional)</span>
    </h3>
    <v-select
      v-model="evalId"
      :items="evalOptions"
      label="eval.dataset"
      :error-messages="Object.values(evalErrors ?? {}).flat()"
      style="max-width: 520px"
    />
    <v-card v-if="evalSelected" variant="outlined" class="mt-2">
      <v-card-text><DatasetPanel :dataset="evalSelected" :model="model || null" /></v-card-text>
    </v-card>
  </div>
</template>
