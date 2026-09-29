<script setup lang="ts">
import { ref } from 'vue'
import { useRunCommands } from '@/api/runs'
import type { EvaluationResult, GenerationResult } from '@/api/types'
import { formatNumber } from '@/utils/format'
import { ProblemAlert } from '@/features/shared'

const props = defineProps<{ id: string }>()
const commands = useRunCommands(() => props.id)

const evaluation = ref<EvaluationResult | null>(null)
const evaluationError = ref<unknown>(null)
async function evaluate() {
  evaluationError.value = null
  try {
    evaluation.value = await commands.evaluate.mutateAsync()
  } catch (error) {
    evaluationError.value = error
  }
}

const prompt = ref('')
const chat = ref(true)
const includeBase = ref(true)
const maxTokens = ref('256')
const temperature = ref('')
const seed = ref('0')
const generation = ref<GenerationResult | null>(null)
const generationError = ref<unknown>(null)
function optional(text: string): number | null {
  return text.trim() === '' ? null : Number(text)
}
async function generate() {
  generationError.value = null
  try {
    generation.value = await commands.generate.mutateAsync({
      prompt: prompt.value,
      chat: chat.value,
      include_base: includeBase.value,
      max_new_tokens: optional(maxTokens.value),
      temperature: optional(temperature.value),
      seed: optional(seed.value),
    })
  } catch (error) {
    generationError.value = error
  }
}
</script>

<template>
  <v-row>
    <v-col cols="12" md="5">
      <v-card class="h-100">
        <v-card-title class="text-subtitle-1">Evaluate now</v-card-title>
        <v-card-text>
          <p class="text-body-2 mb-3">
            Runs the evaluation set at the next callback. The result is shown here only: it does not
            enter the run's history, its best score or its patience.
          </p>
          <v-btn color="primary" :loading="commands.evaluate.isPending.value" @click="evaluate"
            >Evaluate</v-btn
          >
          <ProblemAlert :error="evaluationError" class="mt-3">
            <div
              v-if="(evaluationError as { kind?: string })?.kind === 'timeout'"
              class="text-body-2"
            >
              The run did not reach its next callback in time.
            </div>
          </ProblemAlert>
          <v-table v-if="evaluation" density="compact" class="mt-3">
            <tbody>
              <tr>
                <td>iteration</td>
                <td class="mono">{{ evaluation.iteration }}</td>
              </tr>
              <tr>
                <td>examples</td>
                <td class="mono">{{ evaluation.examples }}</td>
              </tr>
              <tr v-if="evaluation.loss != null">
                <td>loss</td>
                <td class="mono">{{ formatNumber(evaluation.loss, 4) }}</td>
              </tr>
              <tr v-if="evaluation.perplexity != null">
                <td>perplexity</td>
                <td class="mono">{{ formatNumber(evaluation.perplexity, 3) }}</td>
              </tr>
              <tr v-if="evaluation.accuracy != null">
                <td>accuracy</td>
                <td class="mono">{{ formatNumber(evaluation.accuracy, 4) }}</td>
              </tr>
              <tr v-if="evaluation.mean_reward != null">
                <td>mean reward</td>
                <td class="mono">{{ formatNumber(evaluation.mean_reward, 4) }}</td>
              </tr>
              <tr v-if="evaluation.reward_min != null">
                <td>reward range</td>
                <td class="mono">
                  {{ formatNumber(evaluation.reward_min) }} …
                  {{ formatNumber(evaluation.reward_max) }}
                </td>
              </tr>
            </tbody>
          </v-table>
        </v-card-text>
      </v-card>
    </v-col>
    <v-col cols="12" md="7">
      <v-card class="h-100">
        <v-card-title class="text-subtitle-1">Generate</v-card-title>
        <v-card-text>
          <form @submit.prevent="generate">
            <v-textarea v-model="prompt" label="Prompt" rows="3" auto-grow />
            <v-row dense>
              <v-col cols="4"
                ><v-text-field v-model="maxTokens" label="max new tokens" inputmode="numeric"
              /></v-col>
              <v-col cols="4"
                ><v-text-field
                  v-model="temperature"
                  label="temperature"
                  inputmode="decimal"
                  placeholder="server default"
              /></v-col>
              <v-col cols="4"
                ><v-text-field v-model="seed" label="seed" inputmode="numeric"
              /></v-col>
            </v-row>
            <div class="d-flex ga-4 flex-wrap">
              <v-switch
                v-model="chat"
                label="chat template"
                color="primary"
                hide-details
                density="compact"
              />
              <v-switch
                v-model="includeBase"
                label="also answer with the base model"
                color="primary"
                hide-details
                density="compact"
              />
            </div>
            <v-btn
              type="submit"
              color="primary"
              class="mt-2"
              :disabled="!prompt.trim()"
              :loading="commands.generate.isPending.value"
            >
              Generate
            </v-btn>
          </form>
          <ProblemAlert :error="generationError" class="mt-3" />
          <v-row v-if="generation" class="mt-2">
            <v-col :cols="generation.base_text != null ? 6 : 12">
              <div class="text-caption rg-muted">
                adapter · {{ generation.tokens }} tokens · iteration {{ generation.iteration }}
              </div>
              <pre class="text-pre rg-code">{{ generation.text }}</pre>
            </v-col>
            <v-col v-if="generation.base_text != null" cols="6">
              <div class="text-caption rg-muted">base</div>
              <pre class="text-pre rg-code">{{ generation.base_text }}</pre>
            </v-col>
          </v-row>
        </v-card-text>
      </v-card>
    </v-col>
  </v-row>
</template>
