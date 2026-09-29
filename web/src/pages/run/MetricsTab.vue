<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { mdiChartLine } from '@mdi/js'
import { usePreferencesStore } from '@/stores/preferences'
import { MetricChart, MetricPicker, groupMetricNames } from '@/features/metrics'
import { EmptyState } from '@/features/shared'
import { useRunEventsContext } from './context'

const events = useRunEventsContext()
const preferences = usePreferencesStore()
const revision = computed(() => events.value?.revision.value ?? 0)
const names = computed<string[]>(() => {
  void revision.value
  return events.value ? events.value.series.names() : []
})

const selected = ref<string[]>([])
let seeded = false
watch(
  names,
  (list) => {
    if (seeded || !list.length) return
    seeded = true
    // A first view: the first group, a handful of curves.
    selected.value = groupMetricNames(list)[0]?.names.slice(0, 6) ?? []
  },
  { immediate: true },
)
const shown = computed(() => selected.value.filter((name) => names.value.includes(name)))
</script>

<template>
  <div>
    <EmptyState
      v-if="!names.length"
      :icon="mdiChartLine"
      title="No metric yet"
      text="Curves appear as soon as the run reports its first step."
    />
    <v-row v-else>
      <v-col cols="12" md="3">
        <v-card class="pa-3">
          <div class="text-subtitle-2 mb-2">Axis</div>
          <v-btn-toggle
            v-model="preferences.axis"
            mandatory
            density="compact"
            color="primary"
            class="mb-3"
          >
            <v-btn value="global_step" size="small">step</v-btn>
            <v-btn value="iteration" size="small">iteration</v-btn>
          </v-btn-toggle>
          <v-slider
            v-model="preferences.smoothing"
            :min="0"
            :max="0.99"
            :step="0.01"
            label="Smoothing"
            thumb-label
            hide-details
            color="primary"
            class="mb-3"
          />
          <MetricPicker v-model="selected" :names="names" />
        </v-card>
      </v-col>
      <v-col cols="12" md="9">
        <v-row dense>
          <v-col v-for="name in shown" :key="name" cols="12" lg="6">
            <v-card class="pa-2">
              <MetricChart
                v-if="events"
                :title="name"
                :series="events.series"
                :revision="revision"
                :names="[name]"
                :axis="preferences.axis"
                :smoothing="preferences.smoothing"
                :markers="events.markers.value"
              />
            </v-card>
          </v-col>
        </v-row>
        <p class="text-caption rg-muted mt-2">
          Dashed lines mark checkpoints, dotted lines evaluations. Scroll or drag the bar under a
          chart to zoom.
        </p>
      </v-col>
    </v-row>
  </div>
</template>
