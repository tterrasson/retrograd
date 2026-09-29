<script setup lang="ts">
import { computed } from 'vue'
import { commandsFor, statusInfo } from '@/api/runs'
import { formatNumber, formatRate } from '@/utils/format'
import { EvaluatePanel, PatchForm } from '@/features/runs'
import { StatTile } from '@/features/shared'
import { useRunContext, useRunEventsContext } from './context'

const run = useRunContext()
const events = useRunEventsContext()

const status = computed(() => events.value?.status.value ?? run.value?.status ?? null)
const progress = computed(() => events.value?.progress.value ?? run.value?.progress ?? null)
const evaluations = computed(() => events.value?.evaluations.value ?? [])
const lastEvaluation = computed(() => evaluations.value[evaluations.value.length - 1] ?? null)
const terminal = computed(() => statusInfo(status.value).terminal)
const commands = computed(() => commandsFor(status.value))

const tiles = computed(() => {
  const p = progress.value
  if (!p) return []
  return [
    { label: 'iteration', value: `${p.iteration ?? 0} / ${p.iterations ?? 0}` },
    { label: 'global step', value: formatNumber(p.global_step ?? 0) },
    { label: 'train loss', value: formatNumber(p.train_loss, 4) },
    { label: 'eval loss', value: formatNumber(p.eval_loss, 4) },
    { label: 'reward', value: formatNumber(p.reward, 4) },
    {
      label: 'learning rate',
      value: p.learning_rate != null ? p.learning_rate.toExponential(2) : '–',
    },
    { label: 'throughput', value: formatRate(p.tokens_per_second) },
  ]
})
</script>

<template>
  <div>
    <v-row dense>
      <v-col v-for="tile in tiles" :key="tile.label" cols="6" sm="4" md="3" lg="2">
        <StatTile :label="tile.label" :value="tile.value" />
      </v-col>
    </v-row>

    <v-row class="mt-2">
      <v-col cols="12" md="6">
        <v-card class="h-100">
          <v-card-title class="text-subtitle-1">Last evaluation</v-card-title>
          <v-card-text v-if="lastEvaluation">
            <v-table density="compact">
              <tbody>
                <tr>
                  <td>iteration</td>
                  <td class="mono">{{ lastEvaluation.iteration }}</td>
                </tr>
                <tr v-if="lastEvaluation.loss != null">
                  <td>loss</td>
                  <td class="mono">{{ formatNumber(lastEvaluation.loss, 4) }}</td>
                </tr>
                <tr v-if="lastEvaluation.perplexity != null">
                  <td>perplexity</td>
                  <td class="mono">{{ formatNumber(lastEvaluation.perplexity, 3) }}</td>
                </tr>
                <tr v-if="lastEvaluation.accuracy != null">
                  <td>accuracy</td>
                  <td class="mono">{{ formatNumber(lastEvaluation.accuracy, 4) }}</td>
                </tr>
                <tr v-if="lastEvaluation.mean_reward != null">
                  <td>mean reward</td>
                  <td class="mono">{{ formatNumber(lastEvaluation.mean_reward, 4) }}</td>
                </tr>
                <tr>
                  <td>best score</td>
                  <td class="mono">{{ formatNumber(lastEvaluation.best, 4) }}</td>
                </tr>
                <tr>
                  <td>patience</td>
                  <td>
                    <v-chip v-if="lastEvaluation.improved" color="success">improved</v-chip>
                    <span v-else class="mono"
                      >{{ lastEvaluation.stale }} evaluation(s) without improvement</span
                    >
                    <v-chip v-if="!lastEvaluation.keep_training" color="warning" class="ml-2"
                      >stopping early</v-chip
                    >
                  </td>
                </tr>
              </tbody>
            </v-table>
          </v-card-text>
          <v-card-text v-else class="rg-muted">No evaluation yet.</v-card-text>
        </v-card>
      </v-col>
      <v-col cols="12" md="6">
        <PatchForm v-if="run && !terminal" :id="run.id" />
        <v-card v-else-if="run" class="h-100">
          <v-card-title class="text-subtitle-1">Outcome</v-card-title>
          <v-card-text>
            <p>The run is {{ statusInfo(status).label }}.</p>
            <p v-if="run.error" class="text-error mt-2">{{ run.error }}</p>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>

    <div v-if="run && commands.evaluate" class="mt-4">
      <EvaluatePanel :id="run.id" />
    </div>
  </div>
</template>
