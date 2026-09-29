<script setup lang="ts">
import { ref, watch } from 'vue'
import { useTokenize } from '@/api/datasets'
import type { DatasetTokenization, DatasetView } from '@/api/types'
import { ProblemAlert } from '@/features/shared'
import DatasetPreview from './DatasetPreview.vue'
import TokenizationCard from './TokenizationCard.vue'

// One dataset: its first examples, and its token lengths under a model.
const props = defineProps<{ dataset: DatasetView; model?: string | null; models?: string[] }>()

const tokenize = useTokenize()
const tokenization = ref<DatasetTokenization | null>(null)
const failure = ref<unknown>(null)
const chosen = ref<string | null>(props.model ?? null)

watch(
  () => [props.dataset.id, props.model] as const,
  ([, model]) => {
    tokenization.value = null
    failure.value = null
    chosen.value = model ?? chosen.value
    if (model) void run()
  },
  { immediate: true },
)

async function run() {
  if (!chosen.value) return
  failure.value = null
  try {
    tokenization.value = await tokenize.mutateAsync({ id: props.dataset.id, model: chosen.value })
  } catch (error) {
    failure.value = error
  }
}
</script>

<template>
  <v-row>
    <v-col cols="12" md="7">
      <div class="text-subtitle-2 mb-2">Preview</div>
      <DatasetPreview :id="dataset.id" />
    </v-col>
    <v-col cols="12" md="5">
      <div class="text-subtitle-2 mb-2">Token lengths</div>
      <div v-if="!model" class="d-flex ga-2 align-center mb-2">
        <v-combobox
          v-model="chosen"
          :items="models ?? []"
          label="Model to tokenize with"
          hide-details
          density="compact"
        />
        <v-btn :disabled="!chosen" :loading="tokenize.isPending.value" @click="run">Tokenize</v-btn>
      </div>
      <v-progress-linear
        v-if="tokenize.isPending.value"
        indeterminate
        color="primary"
        class="mb-2"
      />
      <ProblemAlert :error="failure" />
      <TokenizationCard :tokenization="tokenization" :stats="dataset.stats" />
    </v-col>
  </v-row>
</template>
