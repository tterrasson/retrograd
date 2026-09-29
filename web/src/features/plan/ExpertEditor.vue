<script setup lang="ts">
import { ref, watch } from 'vue'
import { mdiFileUploadOutline } from '@mdi/js'

const params = defineModel<Record<string, unknown>>('params', { required: true })
const toml = defineModel<string>('toml', { required: true })
const useToml = defineModel<boolean>('useToml', { required: true })

const text = ref('')
const parseError = ref<string | null>(null)
watch(
  params,
  (value) => {
    const current = (() => {
      try {
        return JSON.parse(text.value) as unknown
      } catch {
        return undefined
      }
    })()
    if (JSON.stringify(current) !== JSON.stringify(value))
      text.value = JSON.stringify(value, null, 2)
  },
  { immediate: true, deep: true },
)

function apply() {
  try {
    const value: unknown = JSON.parse(text.value || '{}')
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
      parseError.value = 'params is a JSON object'
      return
    }
    parseError.value = null
    params.value = value as Record<string, unknown>
  } catch (error) {
    parseError.value = error instanceof Error ? error.message : 'invalid JSON'
  }
}

async function load(files: File | File[] | null) {
  const file = Array.isArray(files) ? files[0] : files
  if (!file) return
  toml.value = await file.text()
  useToml.value = true
}
</script>

<template>
  <div>
    <v-switch
      v-model="useToml"
      label="Send a complete TOML document instead"
      color="primary"
      hide-details
      class="mb-2"
    />
    <template v-if="!useToml">
      <p class="text-body-2 mb-2">
        <span class="mono">params</span> as sent to the server: a partial tree over what it
        resolves.
      </p>
      <v-textarea
        v-model="text"
        class="mono"
        rows="12"
        auto-grow
        :error-messages="parseError ? [parseError] : []"
        @blur="apply"
      />
      <v-btn variant="tonal" @click="apply">Apply</v-btn>
    </template>
    <template v-else>
      <p class="text-body-2 mb-2">
        Sent as it is (<span class="mono">application/toml</span>): the server reads it like a run
        file.
      </p>
      <v-file-input
        label="Load a .toml file"
        accept=".toml,text/plain"
        :prepend-icon="mdiFileUploadOutline"
        density="compact"
        @update:model-value="load"
      />
      <v-textarea v-model="toml" class="mono" rows="16" auto-grow label="TOML" />
    </template>
  </div>
</template>
