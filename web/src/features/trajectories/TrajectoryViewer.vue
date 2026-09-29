<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter, type LocationQuery } from 'vue-router'
import { useQueryClient } from '@tanstack/vue-query'
import { mdiCompareHorizontal, mdiEyeOffOutline, mdiPageLast } from '@mdi/js'
import { groupParam, groupStats, useTrajectoryOverview, useUpdateDetail } from '@/api/trajectories'
import { keys } from '@/api/keys'
import { isProblem } from '@/api/problem'
import type { GroupSummary } from '@/api/types'
import { useShortcuts } from '@/composables/useShortcuts'
import { usePreferencesStore } from '@/stores/preferences'
import { formatNumber, pluralize } from '@/utils/format'
import { EmptyState, ProblemAlert } from '@/features/shared'
import GroupPanel from './GroupPanel.vue'
import RolloutDiff from './RolloutDiff.vue'
import TrajectoryFilters from './TrajectoryFilters.vue'
import UpdateList from './UpdateList.vue'
import UpdateTimeline from './UpdateTimeline.vue'
import {
  emptyFilters,
  formatMemberRef,
  parseMemberRef,
  sameMember,
  type GroupSort,
  type MemberRef,
  type TrajectoryFilterState,
} from './filters'

const props = defineProps<{
  runId: string
  /** The standalone viewer has no SSE stream to announce new updates. */
  poll?: boolean
  /** Export batches the run dropped (`observe/dropped_batches`), when known. */
  droppedBatches?: number | null
}>()

const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()
const preferences = usePreferencesStore()

const overview = useTrajectoryOverview(
  () => props.runId,
  true,
  () => props.poll === true,
)
const updates = computed(() => overview.data.value?.updates ?? [])
const latest = computed(() => updates.value[updates.value.length - 1]?.update ?? null)

// State that names a case lives in the URL, so a link opens the same view.
function queryNumber(query: LocationQuery, key: string): number | null {
  const value = query[key]
  if (typeof value !== 'string' || value === '') return null
  const number = Number(value)
  return Number.isInteger(number) ? number : null
}
const urlUpdate = computed(() => queryNumber(route.query, 'update'))
const follow = ref(urlUpdate.value === null)
const selected = computed(() => (follow.value ? latest.value : (urlUpdate.value ?? latest.value)))
const seenCount = ref(0)
watch(
  [follow, () => updates.value.length],
  ([following, length]) => {
    if (following) seenCount.value = length
  },
  { immediate: true },
)
const fresh = computed(() => Math.max(0, updates.value.length - seenCount.value))

function replaceQuery(patch: Record<string, string | null>) {
  const query: Record<string, string> = {}
  for (const [key, value] of Object.entries(route.query))
    if (typeof value === 'string') query[key] = value
  for (const [key, value] of Object.entries(patch)) {
    if (value === null) delete query[key]
    else query[key] = value
  }
  void router.replace({ query })
}

function select(update: number) {
  if (follow.value) seenCount.value = updates.value.length
  follow.value = false
  replaceQuery({ update: String(update), group: null, member: null })
}
function setFollow(value: boolean) {
  follow.value = value
  if (value) replaceQuery({ update: null, group: null, member: null })
}
function step(delta: number) {
  const list = updates.value
  if (!list.length) return
  const index = list.findIndex((entry) => entry.update === selected.value)
  const next = Math.min(list.length - 1, Math.max(0, (index < 0 ? list.length : index) + delta))
  const target = list[next]
  if (target) select(target.update)
}

useShortcuts({
  j: () => step(1),
  ArrowDown: () => step(1),
  k: () => step(-1),
  ArrowUp: () => step(-1),
  f: () => setFollow(!follow.value),
})

const summary = computed(
  () => updates.value.find((entry) => entry.update === selected.value) ?? null,
)
const detail = useUpdateDetail(() => props.runId, selected)
watch(summary, (next, previous) => {
  if (next && previous && next.update === previous.update) {
    // The same update may have gained groups, members or its final outcome.
    // This key also covers every open group and individual member query.
    void queryClient.invalidateQueries({
      queryKey: keys.trajectoryUpdate(props.runId, next.update),
    })
  }
})

const filters = ref<TrajectoryFilterState>({
  ...emptyFilters(),
  compact: preferences.compactTrajectories,
})
watch(
  () => filters.value.compact,
  (value) => (preferences.compactTrajectories = value),
)
const sort = ref<GroupSort>('id')

const sortedGroups = computed<GroupSummary[]>(() => {
  const list = [...detail.groups.value]
  if (sort.value === 'id') return list.sort((a, b) => (a.group ?? -1) - (b.group ?? -1))
  const key = (group: GroupSummary) => {
    const stats = groupStats(group)
    return (sort.value === 'reward' ? stats.mean : stats.spread) ?? -Infinity
  }
  return list.sort((a, b) => key(b) - key(a))
})
const skipReasons = computed(() => [
  ...new Set(
    detail.groups.value.flatMap((group) =>
      group.members
        .map((member) => member.skip_reason)
        .filter((reason): reason is string => !!reason),
    ),
  ),
])

const urlGroup = computed(() => (typeof route.query.group === 'string' ? route.query.group : null))
const urlMember = computed(() => queryNumber(route.query, 'member'))
const openGroups = ref<Set<string>>(new Set())
watch(
  [selected, urlGroup],
  ([, group]) => {
    openGroups.value = new Set(group ? [group] : [])
  },
  { immediate: true },
)
function setOpen(group: GroupSummary, open: boolean) {
  const key = groupParam(group.group)
  const next = new Set(openGroups.value)
  if (open) next.add(key)
  else next.delete(key)
  openGroups.value = next
  if (!follow.value && selected.value !== null) {
    replaceQuery({ update: String(selected.value), group: open ? key : null, member: null })
  }
}
function focusMember(group: GroupSummary, member: number | null) {
  if (follow.value || selected.value === null) return
  replaceQuery({
    update: String(selected.value),
    group: groupParam(group.group),
    member: member === null ? null : String(member),
  })
}

const compare = computed<MemberRef[]>(() => {
  const raw = typeof route.query.compare === 'string' ? route.query.compare : ''
  if (!raw) return []
  if (/^\d+$/.test(raw) && selected.value !== null && urlGroup.value) {
    const refs: MemberRef[] = []
    if (urlMember.value !== null)
      refs.push({ update: selected.value, group: urlGroup.value, member: urlMember.value })
    refs.push({ update: selected.value, group: urlGroup.value, member: Number(raw) })
    return refs.slice(-2)
  }
  return raw
    .split(',')
    .map(parseMemberRef)
    .filter((ref): ref is MemberRef => ref !== null)
    .slice(-2)
})
function toggleCompare(ref: MemberRef, on: boolean) {
  let next = compare.value.filter((item) => !sameMember(item, ref))
  if (on) next = [...next, ref].slice(-2)
  replaceQuery({ compare: next.length ? next.map(formatMemberRef).join(',') : null })
}
const compareOpen = ref(false)
useShortcuts({ Escape: () => (compareOpen.value = false) })

const notFound = computed(
  () => isProblem(overview.error.value) && overview.error.value.status === 404,
)
const HEADLINE = ['reward/mean', 'reward/std', 'batch/trained_fraction', 'policy/kl']
</script>

<template>
  <div>
    <ProblemAlert v-if="!notFound" :error="overview.error.value" />
    <v-progress-linear v-if="overview.isPending.value" indeterminate color="primary" />
    <EmptyState
      v-else-if="overview.data.value && !overview.data.value.observed"
      :icon="mdiEyeOffOutline"
      title="No trajectories"
      text="This run does not export its rollouts: it has no [observe] section."
    />
    <template v-else-if="overview.data.value">
      <div class="d-flex flex-wrap align-center ga-2 mb-2">
        <v-chip class="mono">{{ overview.data.value.algorithm ?? '–' }}</v-chip>
        <span v-if="overview.data.value.model" class="mono text-caption">{{
          overview.data.value.model
        }}</span>
        <span v-if="overview.data.value.every" class="text-caption rg-muted">
          texts every
          {{ overview.data.value.every === 1 ? 'update' : `${overview.data.value.every} updates` }}
        </span>
        <span class="text-caption rg-muted">{{ pluralize(updates.length, 'update') }}</span>
        <v-spacer />
        <v-btn
          v-if="fresh && !follow"
          size="small"
          color="secondary"
          variant="tonal"
          :prepend-icon="mdiPageLast"
          aria-live="polite"
          @click="setFollow(true)"
        >
          {{ pluralize(fresh, 'new update') }}
        </v-btn>
        <v-switch
          :model-value="follow"
          label="follow the latest"
          color="primary"
          density="compact"
          hide-details
          @update:model-value="setFollow(!!$event)"
        />
      </div>
      <v-alert v-if="droppedBatches" type="warning" variant="tonal" density="compact" class="mb-2">
        {{ pluralize(droppedBatches, 'export batch', 'export batches') }} were dropped: some updates
        are missing data.
      </v-alert>

      <v-card class="pa-2 mb-3">
        <UpdateTimeline :updates="updates" :selected="selected" @select="select" />
      </v-card>

      <EmptyState
        v-if="!updates.length"
        title="Waiting for the first update"
        text="Updates appear as the run finishes them."
      />
      <v-row v-else>
        <v-col cols="12" md="2">
          <v-card class="pa-1">
            <UpdateList :updates="updates" :selected="selected" @select="select" />
          </v-card>
          <p class="text-caption rg-muted mt-1"><kbd>j</kbd>/<kbd>k</kbd> move between updates</p>
        </v-col>
        <v-col cols="12" md="10">
          <div v-if="summary" class="d-flex flex-wrap align-center ga-2 mb-2">
            <h2 class="text-h6">Update {{ summary.update }}</h2>
            <span class="text-caption rg-muted">segment {{ summary.segment }}</span>
            <v-chip
              v-if="summary.status"
              :color="summary.status === 'skipped' ? 'warning' : undefined"
              >{{ summary.status }}</v-chip
            >
            <v-chip v-else color="warning">incomplete update, or incomplete export</v-chip>
            <span v-for="name in HEADLINE" :key="name" class="mono text-caption">
              <template v-if="summary.metrics[name] !== undefined"
                >{{ name }} {{ formatNumber(summary.metrics[name]) }}</template
              >
            </span>
            <v-spacer />
            <v-btn
              size="small"
              :disabled="compare.length !== 2"
              :prepend-icon="mdiCompareHorizontal"
              color="primary"
              variant="tonal"
              @click="compareOpen = true"
            >
              compare {{ compare.length }}/2
            </v-btn>
          </div>
          <TrajectoryFilters
            v-model="filters"
            v-model:sort="sort"
            :skip-reasons="skipReasons"
            :tools="[]"
            class="mb-2"
          />
          <ProblemAlert :error="detail.error.value" />
          <v-progress-linear
            v-if="detail.isFetching.value && !detail.isFetchingNextPage.value"
            indeterminate
            color="primary"
            class="mb-2"
          />
          <div v-if="summary && !summary.texts" class="rg-muted text-body-2 pa-2">
            No rollout was exported for this update<template
              v-if="overview.data.value.every && overview.data.value.every > 1"
            >
              (texts are kept for one update out of {{ overview.data.value.every }})</template
            >.
          </div>
          <GroupPanel
            v-for="group in sortedGroups"
            :key="`${selected}:${groupParam(group.group)}`"
            :run-id="runId"
            :update="selected ?? 0"
            :group="group"
            :filters="filters"
            :compare="compare"
            :focus-member="urlGroup === groupParam(group.group) ? urlMember : null"
            :open="openGroups.has(groupParam(group.group))"
            @update:open="setOpen(group, $event)"
            @compare="toggleCompare"
            @focus="focusMember(group, $event)"
          />
          <div v-if="detail.hasNextPage.value" class="text-center">
            <v-btn
              variant="tonal"
              :loading="detail.isFetchingNextPage.value"
              @click="detail.fetchNextPage()"
              >More groups</v-btn
            >
          </div>
        </v-col>
      </v-row>
    </template>

    <v-dialog v-model="compareOpen" max-width="1400" scrollable>
      <v-card>
        <v-card-title>Compare two members</v-card-title>
        <v-card-text>
          <RolloutDiff
            v-if="compare.length === 2 && compare[0] && compare[1]"
            :run-id="runId"
            :left="compare[0]"
            :right="compare[1]"
          />
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="compareOpen = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
