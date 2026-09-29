<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { UpdateSummary } from '@/api/types'
import { formatNumber } from '@/utils/format'

const props = defineProps<{ updates: UpdateSummary[]; selected: number | null }>()
const emit = defineEmits<{ select: [number] }>()

const range = computed(() => {
  const values = props.updates
    .map((entry) => entry.metrics['reward/mean'])
    .filter((value): value is number => value !== undefined && Number.isFinite(value))
  return values.length ? { low: Math.min(...values), high: Math.max(...values) } : null
})
function width(entry: UpdateSummary): string {
  const value = entry.metrics['reward/mean']
  const r = range.value
  if (value === undefined || !Number.isFinite(value) || !r) return '0%'
  return `${r.high > r.low ? ((value - r.low) / (r.high - r.low)) * 100 : 100}%`
}
function title(entry: UpdateSummary): string {
  if (!entry.status) return 'incomplete update, or incomplete export'
  const reward = entry.metrics['reward/mean']
  const parts = [`reward/mean ${formatNumber(reward)}`]
  if (!entry.texts) parts.push('no texts exported')
  return parts.join(' · ')
}
function segmentChanged(index: number): boolean {
  const previous = props.updates[index - 1]
  const current = props.updates[index]
  return !!previous && !!current && previous.segment !== current.segment
}

const list = ref<HTMLElement | null>(null)
watch(
  () => props.selected,
  async () => {
    await nextTick()
    list.value?.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: 'nearest' })
  },
)
</script>

<template>
  <ol ref="list" class="updates" role="listbox" aria-label="Updates">
    <li
      v-for="(entry, index) in updates"
      :key="`${entry.segment}:${entry.update}`"
      role="option"
      :aria-selected="entry.update === selected"
      :title="title(entry)"
      :class="{
        selected: entry.update === selected,
        incomplete: !entry.status,
        skipped: entry.status === 'skipped',
        textless: !entry.texts,
        resumed: segmentChanged(index),
      }"
      @click="emit('select', entry.update)"
    >
      <span class="label mono">#{{ entry.update }}</span>
      <span class="bar"><span :style="{ width: width(entry) }" /></span>
    </li>
  </ol>
</template>

<style scoped>
.updates {
  list-style: none;
  padding: 0;
  margin: 0;
  max-height: calc(100vh - 260px);
  overflow-y: auto;
}
.updates li {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 2px 6px;
  cursor: pointer;
  border-radius: 4px;
  font-size: 0.8rem;
}
.updates li:hover {
  background: rgba(var(--v-theme-on-surface), 0.05);
}
.updates li.selected {
  background: rgba(var(--v-theme-primary), 0.16);
}
.updates li.textless {
  opacity: 0.55;
}
.updates li.incomplete .label {
  font-style: italic;
}
.updates li.skipped .label {
  text-decoration: line-through;
}
.updates li.resumed {
  border-top: 2px dashed rgb(var(--v-theme-warning));
}
.label {
  width: 3.5rem;
  flex: none;
}
.bar {
  flex: 1;
  height: 6px;
  background: rgba(var(--v-theme-on-surface), 0.06);
  border-radius: 3px;
  overflow: hidden;
}
.bar span {
  display: block;
  height: 100%;
  background: rgb(var(--v-theme-primary));
}
</style>
