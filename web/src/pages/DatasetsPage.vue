<script setup lang="ts">
import { computed, ref } from 'vue'
import { mdiDatabaseOutline, mdiDeleteOutline } from '@mdi/js'
import { useDatasets, useDeleteDataset } from '@/api/datasets'
import { useModelFiles } from '@/api/discovery'
import type { DatasetView } from '@/api/types'
import { useNotify } from '@/composables/useNotify'
import { DatasetPanel, DatasetTable, DatasetUpload } from '@/features/datasets'
import { EmptyState, ProblemAlert } from '@/features/shared'

const datasets = useDatasets()
const models = useModelFiles()
const remove = useDeleteDataset()
const notify = useNotify()

const selectedId = ref<string | null>(null)
const selected = computed(
  () => datasets.datasets.value.find((item) => item.id === selectedId.value) ?? null,
)
const formats = computed(() => datasets.datasets.value.map((item) => item.format))
const modelPaths = computed(() =>
  (models.data.value?.files ?? []).filter((file) => file.role === 'model').map((file) => file.path),
)
const confirm = ref(false)

function uploaded(dataset: DatasetView) {
  selectedId.value = dataset.id
  notify.success(`Uploaded ${dataset.name ?? dataset.id}`)
}

async function destroy() {
  if (!selected.value) return
  try {
    await remove.mutateAsync(selected.value.id)
    notify.success('Dataset deleted')
    selectedId.value = null
    confirm.value = false
  } catch (error) {
    notify.error(error, 'Delete')
  }
}
</script>

<template>
  <div>
    <h1 class="text-h5 mb-3">Datasets</h1>
    <v-row>
      <v-col cols="12" lg="8">
        <ProblemAlert :error="datasets.error.value" />
        <v-card>
          <DatasetTable
            v-model:selected="selectedId"
            :datasets="datasets.datasets.value"
            :loading="datasets.isFetching.value"
          />
          <div v-if="datasets.hasNextPage.value" class="pa-2 text-center">
            <v-btn
              variant="text"
              :loading="datasets.isFetchingNextPage.value"
              @click="datasets.fetchNextPage()"
              >Load more</v-btn
            >
          </div>
        </v-card>
      </v-col>
      <v-col cols="12" lg="4">
        <v-card>
          <v-card-title class="text-subtitle-1">Upload</v-card-title>
          <v-card-text><DatasetUpload :formats="formats" @uploaded="uploaded" /></v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <v-card v-if="selected" class="mt-4">
      <v-card-title class="d-flex align-center">
        {{ selected.name ?? selected.id }}
        <v-spacer />
        <v-btn color="error" variant="text" :prepend-icon="mdiDeleteOutline" @click="confirm = true"
          >Delete</v-btn
        >
      </v-card-title>
      <v-card-text>
        <DatasetPanel :dataset="selected" :models="modelPaths" />
      </v-card-text>
    </v-card>
    <EmptyState
      v-else-if="!datasets.isPending.value && datasets.datasets.value.length"
      :icon="mdiDatabaseOutline"
      title="Pick a dataset"
      text="Its first examples and its token lengths show here."
    />

    <v-dialog v-model="confirm" max-width="460">
      <v-card>
        <v-card-title>Delete this dataset?</v-card-title>
        <v-card-text>
          The server keeps no record of which runs used it: a run that still names it will fail to
          resume or fork.
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="confirm = false">Keep</v-btn>
          <v-btn color="error" :loading="remove.isPending.value" @click="destroy">Delete</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
