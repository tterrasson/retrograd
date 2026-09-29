<script setup lang="ts">
import { computed } from 'vue'
import { diffWords } from 'diff'
import { useGroupDetail, type MemberView } from '@/api/trajectories'
import { formatNumber } from '@/utils/format'
import MessageBubble from './MessageBubble.vue'
import type { MemberRef } from './filters'
import { finalAnswer } from './text'

const props = defineProps<{ runId: string; left: MemberRef; right: MemberRef }>()

function load(side: () => MemberRef) {
  return useGroupDetail(
    () => props.runId,
    () => side().update,
    () => side().group,
    () => (side().group === '-' ? side().member : null),
  )
}
const leftQuery = load(() => props.left)
const rightQuery = load(() => props.right)
const pick = (members: MemberView[] | undefined, ref: MemberRef) =>
  members?.find((member) => member.member === ref.member) ?? null
const leftMember = computed(() => pick(leftQuery.data.value?.members, props.left))
const rightMember = computed(() => pick(rightQuery.data.value?.members, props.right))

/** Turns side by side: the same index on both sides shares a row. */
const rows = computed(() => {
  const a = leftMember.value?.messages ?? []
  const b = rightMember.value?.messages ?? []
  return Array.from({ length: Math.max(a.length, b.length) }, (_, index) => ({
    index,
    left: a[index],
    right: b[index],
  }))
})

function summaryOf(member: MemberView | null): string | null {
  const summary = member?.metadata?.summary
  return typeof summary === 'string' ? summary : null
}
const answerDiff = computed(() =>
  leftMember.value && rightMember.value
    ? diffWords(finalAnswer(leftMember.value), finalAnswer(rightMember.value))
    : [],
)
const summaryDiff = computed(() => {
  const a = summaryOf(leftMember.value)
  const b = summaryOf(rightMember.value)
  return a !== null && b !== null ? diffWords(a, b) : null
})
const label = (ref: MemberRef) =>
  `update ${ref.update} · ${ref.group === '-' ? 'rollouts' : `group ${ref.group}`} · #${ref.member}`
const loading = computed(() => leftQuery.isFetching.value || rightQuery.isFetching.value)
</script>

<template>
  <div>
    <v-progress-linear v-if="loading" indeterminate color="primary" class="mb-2" />
    <v-row dense class="mb-2">
      <v-col
        v-for="(side, index) in [
          { ref: left, member: leftMember },
          { ref: right, member: rightMember },
        ]"
        :key="index"
        cols="6"
      >
        <div class="font-weight-medium">{{ label(side.ref) }}</div>
        <div v-if="side.member" class="text-caption mono">
          reward {{ formatNumber(side.member.reward) }} · adv
          {{ formatNumber(side.member.advantage) }} · {{ side.member.tokens }} tok
        </div>
      </v-col>
    </v-row>

    <h3 class="text-subtitle-2 mb-1">Final answers, word by word</h3>
    <pre class="text-pre rg-code diff mb-3"><span
      v-for="(part, index) in answerDiff"
      :key="index"
      :class="{ added: part.added, removed: part.removed }"
    >{{ part.value }}</span></pre>

    <template v-if="summaryDiff">
      <h3 class="text-subtitle-2 mb-1">summary</h3>
      <pre class="text-pre rg-code diff mb-3"><span
        v-for="(part, index) in summaryDiff"
        :key="index"
        :class="{ added: part.added, removed: part.removed }"
      >{{ part.value }}</span></pre>
    </template>

    <template v-if="rows.length">
      <h3 class="text-subtitle-2 mb-1">Turn by turn</h3>
      <v-row v-for="row in rows" :key="row.index" dense>
        <v-col cols="6"><MessageBubble v-if="row.left" :message="row.left" /></v-col>
        <v-col cols="6"><MessageBubble v-if="row.right" :message="row.right" /></v-col>
      </v-row>
    </template>
    <v-row v-else-if="leftMember && rightMember" dense>
      <v-col cols="6">
        <pre class="text-pre rg-code">{{ leftMember.completion }}</pre>
      </v-col>
      <v-col cols="6">
        <pre class="text-pre rg-code">{{ rightMember.completion }}</pre>
      </v-col>
    </v-row>
  </div>
</template>

<style scoped>
.diff .added {
  background: rgba(var(--v-theme-success), 0.25);
}
.diff .removed {
  background: rgba(var(--v-theme-error), 0.22);
  text-decoration: line-through;
}
</style>
