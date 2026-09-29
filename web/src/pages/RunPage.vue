<script setup lang="ts">
import {
  computed,
  effectScope,
  onUnmounted,
  provide,
  shallowRef,
  watch,
  type EffectScope,
} from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useQueryClient } from '@tanstack/vue-query'
import { mdiCompareHorizontal, mdiContentDuplicate, mdiSourceFork } from '@mdi/js'
import { RunEventsKey, useRunEvents, type RunEvents } from '@/api/events'
import { runLabel, shortId, statusInfo, useRun } from '@/api/runs'
import { RunKey } from '@/pages/run/context'
import { useSessionStore } from '@/stores/session'
import { useDraftStore } from '@/stores/draft'
import { RunControls, RunProgressBar, RunStatusChip } from '@/features/runs'
import { CopyButton, EmptyState, ProblemAlert } from '@/features/shared'
import { isProblem } from '@/api/problem'

const props = defineProps<{ id: string }>()
const route = useRoute()
const router = useRouter()
const session = useSessionStore()
const draft = useDraftStore()
const queryClient = useQueryClient()

const viewer = computed(() => session.viewer)
const run = useRun(() => props.id)
const view = computed(() => (viewer.value ? null : (run.data.value ?? null)))
provide(RunKey, view)

// One event stream per run, for as long as its page is open; every tab reads it.
const events = shallowRef<RunEvents | null>(null)
provide(RunEventsKey, events)
let scope: EffectScope | null = null
watch(
  () => [props.id, viewer.value] as const,
  ([id, isViewer]) => {
    scope?.stop()
    scope = null
    events.value = null
    if (isViewer) return
    scope = effectScope()
    events.value =
      scope.run(() =>
        useRunEvents(id, { queryClient, observed: () => view.value?.observed === true }),
      ) ?? null
  },
  { immediate: true },
)
onUnmounted(() => scope?.stop())

const status = computed(() => events.value?.status.value ?? view.value?.status ?? null)
const progress = computed(() => events.value?.progress.value ?? view.value?.progress ?? null)
const notFound = computed(() => isProblem(run.error.value) && run.error.value.status === 404)

const tabs = computed(() => {
  const list = [
    { name: 'run-overview', title: 'Overview' },
    { name: 'run-metrics', title: 'Curves' },
    { name: 'run-journal', title: 'Journal' },
    { name: 'run-checkpoints', title: 'Checkpoints' },
  ]
  if (view.value?.observed) list.push({ name: 'run-trajectories', title: 'Trajectories' })
  list.push(
    { name: 'run-artifacts', title: 'Artifacts' },
    { name: 'run-config', title: 'Configuration' },
  )
  return list
})

function duplicate() {
  if (!view.value) return
  draft.duplicate(view.value.id, view.value.name ?? '', view.value.effective_config)
  void router.push('/runs/new')
}
</script>

<template>
  <div v-if="viewer">
    <h1 class="text-h5 mb-3">Trajectories</h1>
    <router-view />
  </div>
  <EmptyState v-else-if="notFound" title="No such run" :text="`The server knows no run ${id}.`">
    <v-btn to="/runs" color="primary">All runs</v-btn>
  </EmptyState>
  <div v-else>
    <ProblemAlert v-if="!notFound" :error="run.error.value" />
    <div class="d-flex flex-wrap align-center ga-3 mb-2">
      <h1 class="text-h5">{{ view ? runLabel(view) : shortId(id) }}</h1>
      <RunStatusChip :status="status" live />
      <span class="mono text-caption rg-muted">{{ id }}</span>
      <CopyButton :text="id" label="Copy the run id" />
      <v-spacer />
      <RunControls :id="id" :status="status" />
    </div>
    <div v-if="view" class="d-flex flex-wrap align-center ga-2 mb-2">
      <v-chip class="mono">{{ view.algorithm }}</v-chip>
      <v-chip v-if="view.objective">{{ view.objective }}</v-chip>
      <v-chip class="mono" :title="view.model">{{ view.model }}</v-chip>
      <v-chip v-if="view.holds_device" color="primary">holds the device</v-chip>
      <v-chip v-if="view.queue_position" color="info">#{{ view.queue_position }} in queue</v-chip>
      <v-chip
        v-if="events && events.connection.value === 'reconnecting' && !statusInfo(status).terminal"
        color="warning"
      >
        reconnecting
      </v-chip>
      <v-spacer />
      <v-btn size="small" variant="text" :prepend-icon="mdiContentDuplicate" @click="duplicate"
        >Duplicate</v-btn
      >
      <v-btn
        size="small"
        variant="text"
        :prepend-icon="mdiSourceFork"
        :to="{ name: 'run-checkpoints', params: { id } }"
      >
        Fork from a checkpoint
      </v-btn>
      <v-btn
        size="small"
        variant="text"
        :prepend-icon="mdiCompareHorizontal"
        :to="{ path: '/compare', query: { run: id } }"
      >
        Compare
      </v-btn>
    </div>
    <RunProgressBar :progress="progress" :height="8" class="mb-3" />
    <v-alert
      v-if="view?.error || events?.error.value"
      type="error"
      variant="tonal"
      class="mb-3"
      role="alert"
    >
      {{ events?.error.value ?? view?.error }}
    </v-alert>

    <v-tabs :model-value="route.name" color="primary" class="mb-3" show-arrows>
      <v-tab
        v-for="tab in tabs"
        :key="tab.name"
        :value="tab.name"
        :to="{ name: tab.name, params: { id } }"
      >
        {{ tab.title }}
      </v-tab>
    </v-tabs>
    <router-view />
  </div>
</template>
