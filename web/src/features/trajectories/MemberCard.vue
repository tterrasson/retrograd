<script setup lang="ts">
import { computed } from 'vue'
import { mdiChevronDown, mdiChevronUp } from '@mdi/js'
import type { MemberView } from '@/api/trajectories'
import type { MemberSummary } from '@/api/types'
import { formatNumber } from '@/utils/format'
import Conversation from './Conversation.vue'
import { compactLine } from './text'

const props = withDefaults(
  defineProps<{
    member: MemberSummary
    detail?: MemberView | null
    loading?: boolean
    open?: boolean
    comparing?: boolean
    duplicate?: boolean
    compact?: boolean
    markdown?: boolean
    selectable?: boolean
    /** Largest |advantage| of the group, for the bar's scale. */
    scale?: number
  }>(),
  {
    detail: null,
    loading: false,
    open: false,
    comparing: false,
    duplicate: false,
    compact: false,
    markdown: false,
    selectable: true,
    scale: 2,
  },
)
const emit = defineEmits<{ toggle: []; compare: [boolean] }>()

const advantage = computed(() => props.member.advantage ?? null)
const barWidth = computed(() =>
  advantage.value === null ? 0 : Math.min(Math.abs(advantage.value) / (props.scale || 1), 1) * 50,
)
const hasRange = computed(
  () => props.member.advantage_min !== null && props.member.advantage_min !== undefined,
)
</script>

<template>
  <v-card variant="outlined" class="member mb-2" :class="{ comparing }">
    <div class="d-flex align-center flex-wrap ga-2 px-2 py-1">
      <v-checkbox-btn
        v-if="selectable"
        :model-value="comparing"
        density="compact"
        :aria-label="`compare member ${member.member}`"
        @update:model-value="emit('compare', !!$event)"
      />
      <span class="mono font-weight-medium">#{{ member.member }}</span>
      <span class="mono reward" :title="'reward'">{{ formatNumber(member.reward) }}</span>
      <span
        v-if="
          member.reward_raw !== null &&
          member.reward_raw !== undefined &&
          member.reward_raw !== member.reward
        "
        class="mono text-caption rg-muted"
        >raw {{ formatNumber(member.reward_raw) }}</span
      >
      <span v-if="member.judge_term" class="mono text-caption rg-muted"
        >judge {{ formatNumber(member.judge_term) }}</span
      >
      <span class="advantage" :title="`advantage ${formatNumber(advantage, 4)}`" aria-hidden="true">
        <span :class="(advantage ?? 0) >= 0 ? 'pos' : 'neg'" :style="{ width: `${barWidth}%` }" />
      </span>
      <span class="mono text-caption">adv {{ formatNumber(advantage) }}</span>
      <span v-if="hasRange" class="mono text-caption rg-muted">
        [{{ formatNumber(member.advantage_min) }}, {{ formatNumber(member.advantage_max) }}]
      </span>
      <span class="mono text-caption rg-muted">{{ member.tokens }} tok</span>
      <span
        v-if="member.seed !== null && member.seed !== undefined"
        class="mono text-caption rg-muted"
        >seed {{ member.seed }}</span
      >
      <span v-if="member.turns" class="mono text-caption rg-muted">{{ member.turns }} turns</span>
      <span v-if="member.tool_calls" class="mono text-caption rg-muted"
        >{{ member.tool_calls }} calls</span
      >
      <v-chip v-if="member.tool_errors" color="error" size="x-small"
        >{{ member.tool_errors }} tool errors</v-chip
      >
      <v-chip v-if="member.truncated" color="warning" size="x-small">truncated</v-chip>
      <v-chip v-if="member.skip_reason" color="error" size="x-small"
        >skipped: {{ member.skip_reason }}</v-chip
      >
      <v-chip v-else-if="member.eligible === false" color="error" size="x-small">skipped</v-chip>
      <v-chip v-if="member.trained === true" color="success" size="x-small">trained</v-chip>
      <v-chip v-else-if="member.trained === null || member.trained === undefined" size="x-small"
        >unknown execution</v-chip
      >
      <v-chip v-if="duplicate" size="x-small">duplicate</v-chip>
      <v-spacer />
      <v-btn
        :icon="open ? mdiChevronUp : mdiChevronDown"
        size="small"
        variant="text"
        :aria-expanded="open"
        :aria-label="open ? 'Hide the conversation' : 'Show the conversation'"
        @click="emit('toggle')"
      />
    </div>
    <div v-if="open" class="px-3 pb-3">
      <v-progress-linear v-if="loading" indeterminate color="primary" class="mb-2" />
      <template v-if="detail">
        <div v-if="compact && detail.messages" class="mono text-caption">
          {{ compactLine(detail.messages) }}
        </div>
        <Conversation v-else :member="detail" :markdown="markdown" />
      </template>
      <div v-else-if="!loading" class="rg-muted text-body-2">
        This member's content was not exported.
      </div>
    </div>
  </v-card>
</template>

<style scoped>
.member.comparing {
  border-color: rgb(var(--v-theme-primary));
}
.reward {
  min-width: 3.5rem;
  font-weight: 600;
}
.advantage {
  position: relative;
  display: inline-block;
  width: 64px;
  height: 8px;
  background: rgba(var(--v-theme-on-surface), 0.08);
  border-radius: 4px;
}
.advantage::after {
  content: '';
  position: absolute;
  left: 50%;
  top: -2px;
  bottom: -2px;
  width: 1px;
  background: rgba(var(--v-theme-on-surface), 0.4);
}
.advantage .pos,
.advantage .neg {
  position: absolute;
  top: 0;
  bottom: 0;
}
.advantage .pos {
  left: 50%;
  background: rgb(var(--v-theme-success));
  border-radius: 0 4px 4px 0;
}
.advantage .neg {
  right: 50%;
  background: rgb(var(--v-theme-error));
  border-radius: 4px 0 0 4px;
}
</style>
