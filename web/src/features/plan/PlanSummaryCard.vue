<script setup lang="ts">
import { computed } from 'vue'
import type { Budgets, PhaseResources, PlanSummary } from '@/api/types'
import { formatBytes, formatNumber, formatPercent } from '@/utils/format'
import MemoryBreakdown from './MemoryBreakdown.vue'
import type { MemorySegment } from './memory'

const props = defineProps<{ plan: PlanSummary; budgets?: Budgets | null }>()

function segments(phase: PhaseResources): MemorySegment[] {
  return phase.posts.map((post) => ({
    name: post.name,
    bytes: post.bytes,
    detail: `${post.origin}, ${post.bound}`,
  }))
}

const resources = computed(() => props.plan.memory.resources)
const budgets = computed(() => props.budgets ?? props.plan.memory.budgets)
const phases = computed(() => [
  { label: 'Device, persistent', phase: resources.value.persistent_device },
  { label: 'Optimizer, transient', phase: resources.value.optimizer_transient },
  { label: 'Generation, transient', phase: resources.value.generation_transient },
  { label: 'Kernels', phase: resources.value.kernel },
  { label: 'Host, persistent', phase: resources.value.persistent_host },
])
</script>

<template>
  <v-card>
    <v-card-title class="text-subtitle-1">Plan</v-card-title>
    <v-card-text>
      <v-row dense>
        <v-col cols="6" sm="3"
          ><div class="text-caption rg-muted">iterations</div>
          <div class="mono">{{ formatNumber(plan.iterations) }}</div></v-col
        >
        <v-col cols="6" sm="3"
          ><div class="text-caption rg-muted">optimizer steps</div>
          <div class="mono">{{ formatNumber(plan.total_steps) }}</div></v-col
        >
        <v-col cols="6" sm="3"
          ><div class="text-caption rg-muted">truncated examples</div>
          <div class="mono">{{ formatPercent(plan.truncation_fraction) }}</div></v-col
        >
        <v-col v-if="plan.eval_truncation_fraction != null" cols="6" sm="3">
          <div class="text-caption rg-muted">truncated eval examples</div>
          <div class="mono">{{ formatPercent(plan.eval_truncation_fraction) }}</div>
        </v-col>
        <v-col v-if="plan.judge_calls_expected" cols="6" sm="3">
          <div class="text-caption rg-muted">judge calls</div>
          <div class="mono">{{ formatNumber(plan.judge_calls_expected) }}</div>
        </v-col>
        <v-col cols="6" sm="3">
          <div class="text-caption rg-muted">backend</div>
          <div class="mono">
            {{ plan.execution.expected_backend }}
            <span class="rg-muted">({{ plan.execution.confidence }})</span>
          </div>
        </v-col>
      </v-row>

      <v-alert
        v-for="warning in plan.warnings"
        :key="warning.code + (warning.field ?? '')"
        type="warning"
        variant="tonal"
        density="compact"
        class="mt-2"
      >
        <code v-if="warning.field" class="mono">{{ warning.field }}</code> {{ warning.message }}
        <span class="text-caption rg-muted">({{ warning.code }})</span>
      </v-alert>

      <h3 class="text-subtitle-2 mt-4 mb-2">Memory</h3>
      <MemoryBreakdown
        label="Device peak"
        :segments="[{ name: 'device peak', bytes: resources.device_peak_bytes }]"
        :total="resources.device_peak_bytes"
        :budget="budgets.vram.effective_bytes"
      />
      <div class="text-caption rg-muted mb-3">
        bounds {{ formatBytes(resources.lower_bound_device_bytes) }} –
        {{ formatBytes(resources.upper_bound_device_bytes) }}; host peak
        {{ formatBytes(resources.host_peak_bytes) }} of
        {{ formatBytes(budgets.ram.effective_bytes) }}
      </div>
      <details>
        <summary class="text-body-2 rg-clickable">Every post</summary>
        <div class="mt-2">
          <MemoryBreakdown
            v-for="entry in phases"
            :key="entry.label"
            :label="entry.label"
            :segments="segments(entry.phase)"
            :total="entry.phase.bytes"
          />
        </div>
      </details>

      <template v-if="plan.levers.length || plan.defaults_applied.length">
        <h3 class="text-subtitle-2 mt-4 mb-1">Applied to fit</h3>
        <v-table density="compact">
          <tbody>
            <tr v-for="rule in [...plan.levers, ...plan.defaults_applied]" :key="rule.id + rule.to">
              <td class="mono">{{ rule.id }}</td>
              <td class="mono">{{ rule.from }} → {{ rule.to }}</td>
              <td>{{ rule.note }}</td>
              <td class="rg-muted">{{ rule.cost }}</td>
            </tr>
          </tbody>
        </v-table>
      </template>

      <template v-if="plan.execution.cpu_fallbacks.length">
        <h3 class="text-subtitle-2 mt-4 mb-1">CPU fallbacks</h3>
        <v-table density="compact">
          <tbody>
            <tr
              v-for="fallback in plan.execution.cpu_fallbacks"
              :key="fallback.ggml_op + fallback.reason"
            >
              <td class="mono">{{ fallback.ggml_op }}</td>
              <td class="mono">{{ fallback.from_backend }} → {{ fallback.to_backend }}</td>
              <td>{{ fallback.reason }}</td>
            </tr>
          </tbody>
        </v-table>
      </template>
    </v-card-text>
  </v-card>
</template>
