<script setup lang="ts">
import { computed } from 'vue'
import { mdiFilterRemoveOutline, mdiMagnify } from '@mdi/js'
import {
  activeFilterCount,
  emptyFilters,
  type GroupSort,
  type TrajectoryFilterState,
} from './filters'

const props = defineProps<{ skipReasons: string[]; tools: string[] }>()
const filters = defineModel<TrajectoryFilterState>({ required: true })
const sort = defineModel<GroupSort>('sort', { required: true })

function set<K extends keyof TrajectoryFilterState>(key: K, value: TrajectoryFilterState[K]) {
  filters.value = { ...filters.value, [key]: value }
}
function number(text: string | null): number | null {
  if (text === null || text.trim() === '') return null
  const value = Number(text)
  return Number.isFinite(value) ? value : null
}
const active = computed(() => activeFilterCount(filters.value))
const reasons = computed(() => props.skipReasons)
</script>

<template>
  <div class="d-flex flex-wrap align-center ga-2" role="search" aria-label="Filter the members">
    <v-text-field
      :model-value="filters.text"
      :prepend-inner-icon="mdiMagnify"
      label="Search the texts"
      density="compact"
      hide-details
      clearable
      style="min-width: 200px; max-width: 280px"
      @update:model-value="set('text', $event ?? '')"
    />
    <v-text-field
      :model-value="filters.rewardMin ?? ''"
      label="reward ≥"
      type="number"
      density="compact"
      hide-details
      style="max-width: 110px"
      @update:model-value="set('rewardMin', number($event))"
    />
    <v-text-field
      :model-value="filters.rewardMax ?? ''"
      label="reward ≤"
      type="number"
      density="compact"
      hide-details
      style="max-width: 110px"
      @update:model-value="set('rewardMax', number($event))"
    />
    <v-select
      :model-value="filters.skipReason"
      :items="reasons"
      label="skip reason"
      density="compact"
      hide-details
      clearable
      style="max-width: 180px"
      @update:model-value="set('skipReason', $event)"
    />
    <v-combobox
      :model-value="filters.tool"
      :items="tools"
      label="tool"
      density="compact"
      hide-details
      clearable
      style="max-width: 180px"
      @update:model-value="set('tool', $event || null)"
    />
    <v-checkbox
      :model-value="filters.trainedOnly"
      label="trained only"
      density="compact"
      hide-details
      @update:model-value="set('trainedOnly', !!$event)"
    />
    <v-checkbox
      :model-value="filters.flaggedOnly"
      label="truncated or skipped"
      density="compact"
      hide-details
      @update:model-value="set('flaggedOnly', !!$event)"
    />
    <v-checkbox
      :model-value="filters.toolErrorsOnly"
      label="tool errors"
      density="compact"
      hide-details
      @update:model-value="set('toolErrorsOnly', !!$event)"
    />
    <v-checkbox
      :model-value="filters.compact"
      label="compact"
      density="compact"
      hide-details
      @update:model-value="set('compact', !!$event)"
    />
    <v-checkbox
      :model-value="filters.markdown"
      label="markdown"
      density="compact"
      hide-details
      @update:model-value="set('markdown', !!$event)"
    />
    <v-select
      v-model="sort"
      :items="[
        { title: 'group id', value: 'id' },
        { title: 'mean reward', value: 'reward' },
        { title: 'reward spread', value: 'variance' },
      ]"
      label="sort groups by"
      density="compact"
      hide-details
      style="max-width: 180px"
    />
    <v-btn
      v-if="active"
      variant="text"
      size="small"
      :prepend-icon="mdiFilterRemoveOutline"
      @click="filters = { ...emptyFilters(), compact: filters.compact, markdown: filters.markdown }"
    >
      Clear {{ active }}
    </v-btn>
  </div>
</template>
