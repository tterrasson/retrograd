<script setup lang="ts">
import type { StepReward } from '@/api/types'
import { formatNumber } from '@/utils/format'

defineProps<{ steps: StepReward[] }>()
</script>

<template>
  <span class="d-inline-flex flex-wrap ga-1">
    <v-chip
      v-for="step in steps"
      :key="`${step.step_index}-${step.kind}`"
      size="x-small"
      :color="step.reward < 0 ? 'error' : 'success'"
      :title="`${step.kind} reward of step ${step.step_index}`"
    >
      step {{ step.step_index }} {{ step.reward >= 0 ? '+' : '' }}{{ formatNumber(step.reward) }}
    </v-chip>
  </span>
</template>
