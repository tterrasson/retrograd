<script setup lang="ts">
import { computed } from 'vue'
import { usePreferencesStore } from '@/stores/preferences'
import { EventLog } from '@/features/journal'
import { useRunEventsContext } from './context'

const events = useRunEventsContext()
const preferences = usePreferencesStore()
const revision = computed(() => events.value?.revision.value ?? 0)
</script>

<template>
  <EventLog
    v-if="events"
    v-model:follow="preferences.followJournal"
    :journal="events.journal"
    :revision="revision"
  />
</template>
