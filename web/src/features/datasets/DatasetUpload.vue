<script setup lang="ts">
import { computed, ref } from 'vue'
import { mdiUpload } from '@mdi/js'
import { useUploadDataset } from '@/api/datasets'
import type { DatasetView } from '@/api/types'
import { formatBytes } from '@/utils/format'
import { ProblemAlert } from '@/features/shared'

const props = withDefaults(defineProps<{ formats?: string[] }>(), { formats: () => [] })
const emit = defineEmits<{ uploaded: [DatasetView] }>()

const upload = useUploadDataset()
const file = ref<File | null>(null)
const name = ref('')
const format = ref<string | null>(null)
const loaded = ref(0)
const total = ref(0)
const failure = ref<unknown>(null)
let controller: AbortController | null = null

const percent = computed(() => (total.value ? (loaded.value / total.value) * 100 : 0))
const suggestions = computed(() => [...new Set(props.formats)].sort())

function stop() {
  controller?.abort()
}

async function submit() {
  if (!file.value) return
  failure.value = null
  loaded.value = 0
  total.value = file.value.size
  controller = new AbortController()
  try {
    const dataset = await upload.mutateAsync({
      file: file.value,
      name: name.value.trim() || undefined,
      format: format.value?.trim() || undefined,
      signal: controller.signal,
      onProgress: (done, size) => {
        loaded.value = done
        total.value = size
      },
    })
    emit('uploaded', dataset)
    file.value = null
    name.value = ''
  } catch (error) {
    failure.value = error
  } finally {
    controller = null
  }
}
</script>

<template>
  <form @submit.prevent="submit">
    <v-file-input
      v-model="file"
      label="Dataset file"
      accept=".jsonl,.ndjson,.json,.txt"
      :prepend-icon="mdiUpload"
      show-size
      :multiple="false"
    />
    <v-row dense>
      <v-col cols="12" sm="6"><v-text-field v-model="name" label="Name (optional)" /></v-col>
      <v-col cols="12" sm="6">
        <v-combobox
          v-model="format"
          :items="suggestions"
          label="Format"
          placeholder="detected by the server"
          clearable
        />
      </v-col>
    </v-row>
    <div v-if="upload.isPending.value" class="mb-2">
      <v-progress-linear
        :model-value="percent"
        color="primary"
        height="8"
        rounded
        :aria-label="`uploaded ${Math.round(percent)} percent`"
      />
      <div class="text-caption rg-muted mt-1">
        {{ formatBytes(loaded) }} of {{ formatBytes(total) }}
      </div>
    </div>
    <ProblemAlert :error="failure" />
    <div class="d-flex ga-2">
      <v-btn type="submit" color="primary" :disabled="!file" :loading="upload.isPending.value"
        >Upload</v-btn
      >
      <v-btn v-if="upload.isPending.value" variant="text" @click="stop">Stop</v-btn>
    </div>
  </form>
</template>
