<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { TrajectoryViewer } from '@/features/trajectories'
import { useRunEventsContext } from './context'
import { useSessionStore } from '@/stores/session'

const route = useRoute()
const session = useSessionStore()
const events = useRunEventsContext()
const runId = computed(() => String(route.params.id))
const dropped = computed(() => {
  const value = events.value
  if (!value) return null
  void value.revision.value
  return value.series.latest('observe/dropped_batches')
})
</script>

<template>
  <TrajectoryViewer :run-id="runId" :dropped-batches="dropped" :poll="session.viewer" />
</template>
