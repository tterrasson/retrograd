<script setup lang="ts">
import { computed, ref } from 'vue'
import { mdiChevronDown, mdiChevronUp } from '@mdi/js'
import { groupParam, useGroupDetail, type MemberView } from '@/api/trajectories'
import type { GroupSummary, PromptView } from '@/api/types'
import { formatNumber } from '@/utils/format'
import { JsonTree } from '@/features/shared'
import MemberCard from './MemberCard.vue'
import MessageBubble from './MessageBubble.vue'
import { acceptsSummary, byReward, type MemberRef, type TrajectoryFilterState } from './filters'
import { duplicates, memberText } from './text'

const props = defineProps<{
  runId: string
  update: number
  group: GroupSummary
  filters: TrajectoryFilterState
  compare: MemberRef[]
  /** Member to open on arrival, from the URL. */
  focusMember?: number | null
}>()
const open = defineModel<boolean>('open', { default: false })
const emit = defineEmits<{ compare: [MemberRef, boolean]; focus: [number | null] }>()

const param = computed(() => groupParam(props.group.group))
const flat = computed(() => props.group.group === null || props.group.group === undefined)
const openMembers = ref<Set<number>>(
  new Set(props.focusMember !== null && props.focusMember !== undefined ? [props.focusMember] : []),
)
const ppoMember = computed(() => (flat.value ? ([...openMembers.value][0] ?? null) : null))

const searching = computed(() => !!props.filters.text.trim() || !!props.filters.tool)
const wantDetail = computed(() =>
  flat.value ? ppoMember.value !== null : open.value || searching.value,
)
const detail = useGroupDetail(
  () => props.runId,
  () => props.update,
  () => (wantDetail.value ? param.value : null),
  () => ppoMember.value,
)
const details = computed(() => {
  const map = new Map<number, MemberView>()
  for (const member of detail.data.value?.members ?? []) map.set(member.member, member)
  return map
})
const dupes = computed(() =>
  flat.value ? new Set<number>() : duplicates(detail.data.value?.members ?? []),
)

function acceptsText(member: number): boolean {
  if (!searching.value) return true
  const full = details.value.get(member)
  if (!full) return !detail.isSuccess.value
  if (props.filters.tool) {
    const called = (full.messages ?? []).some((message) =>
      (message.tool_calls ?? []).some((call) => call.name === props.filters.tool),
    )
    if (!called) return false
  }
  const needle = props.filters.text.trim().toLowerCase()
  return !needle || memberText(full).toLowerCase().includes(needle)
}

const members = computed(() =>
  byReward(props.group.members).filter(
    (member) => acceptsSummary(props.filters, member) && acceptsText(member.member),
  ),
)
const scale = computed(() =>
  Math.max(1e-9, ...props.group.members.map((member) => Math.abs(member.advantage ?? 0))),
)
const rewards = computed(() => byReward(props.group.members).map((member) => member.reward ?? null))
const rewardRange = computed(() => {
  const values = rewards.value.filter(
    (value): value is number => value !== null && Number.isFinite(value),
  )
  return values.length ? { low: Math.min(...values), high: Math.max(...values) } : null
})
const zeroSignal = computed(() =>
  props.group.members.some((member) => member.skip_reason === 'zero_signal'),
)
const trained = computed(() => props.group.trained)
const prompt = computed<PromptView>(() => detail.data.value?.prompt ?? props.group.prompt)
const promptPreview = computed(() => {
  const messages = props.group.prompt?.messages ?? []
  const users = messages.filter((message) => message.role === 'user')
  return (users[users.length - 1]?.content ?? messages[messages.length - 1]?.content ?? '').slice(
    0,
    200,
  )
})
const lastUser = computed(() => {
  const users = (prompt.value?.messages ?? []).filter((message) => message.role === 'user')
  return users[users.length - 1]?.content
})

function barHeight(value: number | null): string {
  const range = rewardRange.value
  if (value === null || !range) return '2px'
  const span = range.high - range.low
  return `${span > 0 ? 3 + ((value - range.low) / span) * 21 : 24}px`
}
function isComparing(member: number): boolean {
  return props.compare.some(
    (item) => item.update === props.update && item.group === param.value && item.member === member,
  )
}
function toggleMember(member: number) {
  const next = new Set(flat.value ? [] : openMembers.value)
  if (openMembers.value.has(member)) next.delete(member)
  else next.add(member)
  openMembers.value = next
  emit('focus', next.has(member) ? member : null)
}
</script>

<template>
  <v-card class="group mb-2" :class="{ open }">
    <div
      class="d-flex align-center flex-wrap ga-2 pa-2 rg-clickable"
      role="button"
      tabindex="0"
      :aria-expanded="open"
      @click="open = !open"
      @keydown.enter.prevent="open = !open"
      @keydown.space.prevent="open = !open"
    >
      <span class="font-weight-medium text-no-wrap">{{
        flat ? 'rollouts' : `group ${group.group}`
      }}</span>
      <span class="minibars" aria-hidden="true">
        <span
          v-for="(value, index) in rewards"
          :key="index"
          class="minibar"
          :class="{ empty: value === null }"
          :style="{ height: barHeight(value) }"
          :title="formatNumber(value)"
        />
      </span>
      <span v-if="rewardRange" class="mono text-caption">
        {{ formatNumber(rewardRange.low) }} … {{ formatNumber(rewardRange.high) }}
      </span>
      <v-chip v-if="zeroSignal" color="warning" size="x-small">zero_signal</v-chip>
      <span class="text-caption rg-muted">{{ trained }}/{{ group.members.length }} trained</span>
      <span v-if="members.length !== group.members.length" class="text-caption rg-muted">
        · {{ members.length }} shown
      </span>
      <span class="prompt-preview text-body-2 rg-muted text-truncate">{{ promptPreview }}</span>
      <v-spacer />
      <v-icon :icon="open ? mdiChevronUp : mdiChevronDown" aria-hidden="true" />
    </div>
    <div v-if="open" class="px-2 pb-2">
      <details v-if="prompt" class="prompt mb-2">
        <summary class="text-body-2 rg-clickable">
          prompt <span class="mono">{{ prompt.key }}</span>
        </summary>
        <template v-for="(message, index) in prompt.messages" :key="index">
          <details v-if="message.role === 'system'" class="mb-1">
            <summary class="text-caption rg-clickable">system</summary>
            <pre class="text-pre rg-code">{{ message.content }}</pre>
          </details>
          <MessageBubble v-else :message="message" :markdown="filters.markdown" />
        </template>
        <template v-if="prompt.reward_text && prompt.reward_text !== lastUser">
          <div class="text-caption rg-muted">reward text</div>
          <pre class="text-pre rg-code">{{ prompt.reward_text }}</pre>
        </template>
        <details v-if="prompt.metadata && Object.keys(prompt.metadata).length">
          <summary class="text-caption rg-clickable">scenario metadata</summary>
          <JsonTree :value="prompt.metadata" :open-depth="2" />
        </details>
      </details>
      <v-progress-linear
        v-if="detail.isFetching.value && !flat"
        indeterminate
        color="primary"
        class="mb-2"
      />
      <template v-for="member in members" :key="member.member">
        <MemberCard
          :member="member"
          :detail="details.get(member.member) ?? null"
          :loading="openMembers.has(member.member) && detail.isFetching.value"
          :open="openMembers.has(member.member)"
          :comparing="isComparing(member.member)"
          :duplicate="dupes.has(member.member)"
          :compact="filters.compact"
          :markdown="filters.markdown"
          :scale="scale"
          @toggle="toggleMember(member.member)"
          @compare="emit('compare', { update, group: param, member: member.member }, $event)"
        />
      </template>
      <div v-if="!members.length" class="text-body-2 rg-muted pa-2">
        No member matches the filters.
      </div>
    </div>
  </v-card>
</template>

<style scoped>
.minibars {
  display: inline-flex;
  align-items: flex-end;
  gap: 2px;
  height: 24px;
}
.minibar {
  width: 6px;
  background: rgb(var(--v-theme-primary));
  border-radius: 1px;
}
.minibar.empty {
  background: rgba(var(--v-theme-on-surface), 0.25);
}
.prompt-preview {
  max-width: 36rem;
}
.group.open {
  border-color: rgba(var(--v-theme-primary), 0.6);
}
</style>
