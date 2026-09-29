<script setup lang="ts">
import { computed } from 'vue'
import type { RunProgress } from '@/api/types'

const props = defineProps<{ progress: RunProgress | null | undefined; height?: number }>()
const value = computed(() => {
  const iterations = props.progress?.iterations ?? 0
  return iterations > 0 ? ((props.progress?.iteration ?? 0) / iterations) * 100 : 0
})
const label = computed(() => {
  const p = props.progress
  return p && p.iterations ? `${p.iteration ?? 0} / ${p.iterations}` : '–'
})
</script>

<template>
  <div class="d-flex align-center ga-2" style="min-width: 120px">
    <v-progress-linear
      :model-value="value"
      :height="height ?? 6"
      rounded
      color="primary"
      :aria-label="`iteration ${label}`"
      class="flex-grow-1"
    />
    <span class="mono text-caption text-no-wrap">{{ label }}</span>
  </div>
</template>
