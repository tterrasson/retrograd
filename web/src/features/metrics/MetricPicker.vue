<script setup lang="ts">
import { computed } from 'vue'
import { groupMetricNames } from './useMetricSeries'

const props = defineProps<{ names: string[] }>()
const selected = defineModel<string[]>({ required: true })
const groups = computed(() => groupMetricNames(props.names))

function toggleGroup(names: string[]) {
  const all = names.every((name) => selected.value.includes(name))
  selected.value = all
    ? selected.value.filter((name) => !names.includes(name))
    : [...new Set([...selected.value, ...names])]
}
</script>

<template>
  <div>
    <div v-for="group in groups" :key="group.prefix" class="mb-2">
      <button
        type="button"
        class="text-caption text-uppercase rg-muted rg-clickable bg-transparent"
        @click="toggleGroup(group.names)"
      >
        {{ group.prefix }}
      </button>
      <v-chip-group
        v-model="selected"
        multiple
        column
        selected-class="text-primary"
        :aria-label="`${group.prefix} metrics`"
      >
        <v-chip
          v-for="name in group.names"
          :key="name"
          :value="name"
          filter
          variant="outlined"
          size="small"
          class="mono"
        >
          {{ name.slice(group.prefix === 'other' ? 0 : group.prefix.length + 1) }}
        </v-chip>
      </v-chip-group>
    </div>
  </div>
</template>
