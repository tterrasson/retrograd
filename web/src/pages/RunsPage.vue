<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { mdiChevronLeft, mdiChevronRight, mdiPlus, mdiRocketLaunchOutline } from '@mdi/js'
import { useCapabilities } from '@/api/discovery'
import { useLiveRunList, useRuns } from '@/api/runs'
import { RUN_STATUSES, type RunStatus } from '@/api/types'
import { RunTable } from '@/features/runs'
import { EmptyState, ProblemAlert } from '@/features/shared'

const route = useRoute()
const router = useRouter()

const status = computed(() =>
  typeof route.query.status === 'string' ? (route.query.status as RunStatus) : null,
)
const algorithm = computed(() =>
  typeof route.query.algorithm === 'string' ? route.query.algorithm : '',
)
const pageSize = ref(25)
const cursors = ref<(string | undefined)[]>([undefined])
const page = computed(() => cursors.value.length - 1)

watch([status, algorithm, pageSize], () => (cursors.value = [undefined]))

const filters = computed(() => ({
  status: status.value || undefined,
  algorithm: algorithm.value || undefined,
  cursor: cursors.value[cursors.value.length - 1],
  limit: pageSize.value,
}))
const runs = useRuns(filters)
const live = useLiveRunList()
const capabilities = useCapabilities()

const rows = computed(() => runs.data.value?.runs ?? [])
const nextCursor = computed(() => runs.data.value?.next_cursor ?? null)
const running = computed(() => rows.value.filter((run) => run.status === 'running').length)
const queued = computed(() => rows.value.filter((run) => run.queue_position != null))
const filtered = computed(() => !!status.value || !!algorithm.value)

function setQuery(name: 'status' | 'algorithm', value: string | null) {
  const query = { ...route.query }
  if (value) query[name] = value
  else delete query[name]
  void router.replace({ query })
}

let algorithmTimer: ReturnType<typeof setTimeout> | undefined
const algorithmInput = ref(algorithm.value)
watch(algorithmInput, (value) => {
  if (algorithmTimer) clearTimeout(algorithmTimer)
  algorithmTimer = setTimeout(() => setQuery('algorithm', value.trim() || null), 400)
})
</script>

<template>
  <div>
    <div class="d-flex align-center flex-wrap ga-3 mb-3">
      <h1 class="text-h5">Runs</h1>
      <v-chip v-if="live.state.value === 'open'" color="success" size="x-small" aria-live="polite"
        >live</v-chip
      >
      <v-chip
        v-else-if="live.state.value === 'reconnecting'"
        color="warning"
        size="x-small"
        aria-live="polite"
        >reconnecting</v-chip
      >
      <v-spacer />
      <v-btn color="primary" :prepend-icon="mdiPlus" to="/runs/new">New run</v-btn>
    </div>

    <v-alert
      v-if="capabilities.data.value"
      variant="tonal"
      density="compact"
      class="mb-3"
      color="info"
    >
      Device: {{ running }} running on this page,
      {{ capabilities.data.value.features.max_concurrent_runs }} at once at most.
      <template v-if="queued.length">
        Queued:
        <span v-for="run in queued" :key="run.id" class="mr-2">
          <router-link :to="`/runs/${run.id}`">{{ run.name || run.id.slice(0, 8) }}</router-link>
          (#{{ run.queue_position }})
        </span>
      </template>
    </v-alert>

    <v-row dense class="mb-1">
      <v-col cols="12" sm="4" md="3">
        <v-select
          :model-value="status"
          :items="[...RUN_STATUSES]"
          label="Status"
          clearable
          hide-details
          @update:model-value="setQuery('status', $event)"
        />
      </v-col>
      <v-col cols="12" sm="4" md="3">
        <v-text-field
          v-model="algorithmInput"
          label="Algorithm"
          placeholder="sft, grpo, ppo…"
          clearable
          hide-details
        />
      </v-col>
    </v-row>

    <ProblemAlert :error="runs.error.value" />

    <v-card>
      <RunTable :runs="rows" :loading="runs.isFetching.value" :items-per-page="pageSize">
        <template #empty>
          <EmptyState
            v-if="!runs.isPending.value"
            :icon="mdiRocketLaunchOutline"
            :title="filtered ? 'No run matches these filters' : 'No run yet'"
            :text="
              filtered ? 'Clear a filter to see more.' : 'Start one from a model and a dataset.'
            "
          >
            <v-btn v-if="!filtered" color="primary" to="/runs/new">New run</v-btn>
          </EmptyState>
        </template>
      </RunTable>
      <v-divider />
      <div class="d-flex align-center justify-end ga-2 pa-2">
        <span class="text-caption rg-muted">Per page</span>
        <v-select
          v-model="pageSize"
          :items="[10, 25, 50, 100]"
          density="compact"
          hide-details
          variant="plain"
          style="max-width: 80px"
          aria-label="Runs per page"
        />
        <span class="text-caption rg-muted">page {{ page + 1 }}</span>
        <v-btn
          :icon="mdiChevronLeft"
          size="small"
          variant="text"
          :disabled="page === 0"
          aria-label="Previous page"
          @click="cursors.pop()"
        />
        <v-btn
          :icon="mdiChevronRight"
          size="small"
          variant="text"
          :disabled="!nextCursor"
          aria-label="Next page"
          @click="nextCursor && cursors.push(nextCursor)"
        />
      </div>
    </v-card>
  </div>
</template>
