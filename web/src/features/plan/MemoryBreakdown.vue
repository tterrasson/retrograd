<script setup lang="ts">
import { computed } from 'vue'
import { formatBytes } from '@/utils/format'
import type { MemorySegment } from './memory'

const props = defineProps<{
  label: string
  segments: MemorySegment[]
  budget?: number | null
  total?: number | null
}>()

const PALETTE = ['primary', 'secondary', 'warning', 'info', 'success', 'error']
const sum = computed(
  () => props.total ?? props.segments.reduce((acc, segment) => acc + segment.bytes, 0),
)
const scale = computed(() => Math.max(sum.value, props.budget ?? 0, 1))
const over = computed(() => props.budget != null && sum.value > props.budget)
</script>

<template>
  <div class="memory mb-3">
    <div class="d-flex text-body-2 mb-1">
      <span class="font-weight-medium">{{ label }}</span>
      <v-spacer />
      <span class="mono" :class="{ 'text-error': over }">
        {{ formatBytes(sum)
        }}<template v-if="budget != null"> / {{ formatBytes(budget) }}</template>
      </span>
    </div>
    <div
      class="bar"
      role="img"
      :aria-label="`${label}: ${formatBytes(sum)}${budget != null ? ` of ${formatBytes(budget)}` : ''}`"
    >
      <v-tooltip v-for="(segment, index) in segments" :key="segment.name" location="top">
        <template #activator="{ props: activator }">
          <div
            v-bind="activator"
            class="segment"
            :class="`bg-${PALETTE[index % PALETTE.length]}`"
            :style="{ width: `${(segment.bytes / scale) * 100}%` }"
          />
        </template>
        {{ segment.name }}: {{ formatBytes(segment.bytes)
        }}<template v-if="segment.detail"> ({{ segment.detail }})</template>
      </v-tooltip>
      <div
        v-if="budget != null"
        class="budget"
        :style="{ left: `${(budget / scale) * 100}%` }"
        title="budget"
      />
    </div>
    <div class="d-flex flex-wrap ga-2 mt-1">
      <span
        v-for="(segment, index) in segments"
        :key="segment.name"
        class="text-caption d-flex align-center ga-1"
      >
        <span class="swatch" :class="`bg-${PALETTE[index % PALETTE.length]}`" />
        {{ segment.name }} <span class="mono rg-muted">{{ formatBytes(segment.bytes) }}</span>
      </span>
    </div>
  </div>
</template>

<style scoped>
.bar {
  position: relative;
  display: flex;
  height: 14px;
  border-radius: 4px;
  overflow: visible;
  background: rgba(var(--v-theme-on-surface), 0.08);
}
.segment {
  height: 100%;
  min-width: 1px;
}
.segment:first-child {
  border-radius: 4px 0 0 4px;
}
.budget {
  position: absolute;
  top: -3px;
  bottom: -3px;
  width: 2px;
  background: rgb(var(--v-theme-on-surface));
}
.swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  display: inline-block;
}
</style>
