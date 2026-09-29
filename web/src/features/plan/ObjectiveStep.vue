<script setup lang="ts">
import { computed } from 'vue'
import { useDefaults, useEnvironments, useJudges, useMcpServers, useRewards } from '@/api/discovery'
import {
  ALLOWS,
  OBJECTIVES,
  type Allow,
  type Limits,
  type Objective,
  type Recipe,
} from '@/api/types'
import { budgetRequest } from '@/api/plan'

const props = defineProps<{
  /** Messages per field under `/recipe`. */
  errors?: Record<string, string[]>
  /** The resolved `observe` table, when the plan configures one. */
  observe?: { every?: unknown } | null
  /** Unlocks the last refusal asked for (`needs_opt_in`). */
  unlocks?: string[]
  everyErrors?: string[]
}>()
const recipe = defineModel<Recipe>('recipe', { required: true })
const observeEvery = defineModel<number | null>('observeEvery', { default: null })
/** `false` declines the export; `null` leaves it to the server. */
const observeEnabled = defineModel<boolean | null>('observeEnabled', { default: null })

const rewards = useRewards()
const judges = useJudges()
const mcpServers = useMcpServers()
const environments = useEnvironments()
const defaults = useDefaults()

// Copy for people; the list of objectives itself comes from the contract.
const OBJECTIVE_TEXT: Record<Objective, string> = {
  'instruction-tuning':
    'Supervised fine-tuning on conversations or texts: the model learns to answer like the examples.',
  'reasoning-rl': 'Reinforcement learning on prompts with a reward that checks the answer (GRPO).',
  'preference-rl': 'Reinforcement learning against a judge that compares answers.',
  agentic: 'Multi-turn trajectories with tools, scored by a reward or a judge.',
  'preference-tuning': 'Offline preference optimization on chosen/rejected pairs.',
}

const err = (field: string) => props.errors?.[field] ?? []
function patch(fields: Partial<Recipe>) {
  recipe.value = { ...recipe.value, ...fields }
}

const objective = computed({
  get: () => recipe.value.objective ?? null,
  set: (value: Objective | null) => patch({ objective: value ?? undefined }),
})
const reward = computed({
  get: () => recipe.value.reward?.id ?? null,
  set: (id: string | null) => patch({ reward: id ? { id } : null }),
})
const judge = computed({
  get: () => recipe.value.judge?.id ?? null,
  set: (id: string | null) => patch({ judge: id ? { id } : null }),
})
const judgeEntry = computed(
  () => judges.data.value?.find((entry) => entry.id === judge.value) ?? null,
)
function judgeField(name: 'rubric' | 'mode' | 'anchor' | 'max_pairs', value: string) {
  if (!recipe.value.judge) return
  const numeric = name === 'anchor' || name === 'max_pairs'
  const next = {
    ...recipe.value.judge,
    [name]: value === '' ? null : numeric ? Number(value) : value,
  }
  patch({ judge: next })
}
const tools = computed({
  get: () => recipe.value.tools ?? [],
  set: (value: string[]) => patch({ tools: value }),
})
const toolOptions = computed(() => [
  ...(mcpServers.data.value ?? []).map((server) => ({
    title: `${server.id} (MCP${server.status ? `, ${server.status}` : ''})`,
    value: server.id,
  })),
  ...(environments.data.value ?? []).map((environment) => ({
    title: `${environment.id} (${environment.kind})`,
    value: environment.id,
  })),
])

function budgetField(name: 'epochs' | 'updates' | 'minutes', value: string) {
  const next = { ...(recipe.value.budget ?? {}), [name]: value === '' ? null : Number(value) }
  const empty = Object.values(next).every((item) => item === null || item === undefined)
  patch({ budget: empty ? null : next })
}
function limitField(name: keyof Limits, value: string) {
  const next: Limits = {
    ...(recipe.value.limits ?? {}),
    [name]: value.trim() ? budgetRequest(value.trim()) : null,
  }
  patch({ limits: next })
}
function limitText(name: keyof Limits): string {
  const value = recipe.value.limits?.[name]
  return value === null || value === undefined ? '' : String(value)
}

const allow = computed({
  get: () => recipe.value.allow ?? [],
  set: (value: Allow[]) => patch({ allow: value }),
})
function allowLabel(id: string): { title: string; note?: string } {
  const entry = defaults.data.value?.active_defaults.find((item) => item.id === id)
  return entry ? { title: entry.note || id, note: entry.cost } : { title: id }
}

const seed = computed({
  get: () =>
    recipe.value.seed === null || recipe.value.seed === undefined ? '' : String(recipe.value.seed),
  set: (value: string) => patch({ seed: value === '' ? null : Number(value) }),
})

const everyText = computed({
  get: () => (observeEvery.value === null ? '' : String(observeEvery.value)),
  set: (value: string) => (observeEvery.value = value.trim() === '' ? null : Number(value)),
})
const resolvedEvery = computed(() => {
  const every = props.observe?.every
  return typeof every === 'number' ? every : null
})
</script>

<template>
  <div>
    <v-radio-group v-model="objective" :error-messages="err('objective')" aria-label="Objective">
      <v-row dense>
        <v-col v-for="value in OBJECTIVES" :key="value" cols="12" md="6" lg="4">
          <v-card
            :color="objective === value ? 'primary' : undefined"
            :variant="objective === value ? 'tonal' : 'outlined'"
            class="pa-3 h-100 rg-clickable"
            @click="objective = value"
          >
            <v-radio :value="value" :label="value" class="font-weight-medium" />
            <div class="text-body-2 rg-muted">{{ OBJECTIVE_TEXT[value] }}</div>
          </v-card>
        </v-col>
      </v-row>
    </v-radio-group>

    <v-row dense class="mt-2">
      <v-col cols="12" md="4">
        <v-select
          v-model="reward"
          :items="
            (rewards.data.value ?? []).map((entry) => ({
              title: entry.id,
              value: entry.id,
              props: { subtitle: entry.description },
            }))
          "
          label="Reward"
          clearable
          :loading="rewards.isPending.value"
          :error-messages="err('reward').concat(err('reward.id'))"
          :hint="rewards.data.value?.length === 0 ? 'The server declares no reward.' : undefined"
          persistent-hint
        />
      </v-col>
      <v-col cols="12" md="4">
        <v-select
          v-model="judge"
          :items="
            (judges.data.value ?? []).map((entry) => ({
              title: entry.id,
              value: entry.id,
              props: { subtitle: entry.description },
            }))
          "
          label="Judge"
          clearable
          :loading="judges.isPending.value"
          :error-messages="err('judge').concat(err('judge.id'))"
          :hint="judges.data.value?.length === 0 ? 'The server declares no judge.' : undefined"
          persistent-hint
        />
      </v-col>
      <v-col cols="12" md="4">
        <v-select
          v-model="tools"
          :items="toolOptions"
          label="Tools"
          multiple
          chips
          closable-chips
          :error-messages="err('tools')"
        />
      </v-col>
    </v-row>
    <v-row v-if="judgeEntry?.client_settings.length" dense>
      <v-col v-for="setting in judgeEntry.client_settings" :key="setting" cols="12" md="3">
        <v-text-field
          v-if="
            setting === 'rubric' ||
            setting === 'mode' ||
            setting === 'anchor' ||
            setting === 'max_pairs'
          "
          :model-value="String(recipe.judge?.[setting] ?? '')"
          :label="`judge.${setting}`"
          :error-messages="err(`judge.${setting}`)"
          @change="judgeField(setting, ($event.target as HTMLInputElement).value)"
        />
      </v-col>
    </v-row>

    <h3 class="text-subtitle-1 mt-4 mb-2">Budget</h3>
    <v-row dense>
      <v-col
        v-for="field in ['epochs', 'updates', 'minutes'] as const"
        :key="field"
        cols="4"
        md="2"
      >
        <v-text-field
          :model-value="recipe.budget?.[field] ?? ''"
          :label="field"
          type="number"
          min="0"
          :error-messages="err(`budget.${field}`)"
          @change="budgetField(field, ($event.target as HTMLInputElement).value)"
        />
      </v-col>
      <v-col cols="6" md="3">
        <v-text-field
          :model-value="limitText('vram')"
          label="VRAM limit"
          placeholder="all, 0.8 or 12GiB"
          :error-messages="err('limits.vram')"
          @change="limitField('vram', ($event.target as HTMLInputElement).value)"
        />
      </v-col>
      <v-col cols="6" md="3">
        <v-text-field
          :model-value="limitText('ram')"
          label="RAM limit"
          placeholder="all, 0.8 or 24GiB"
          :error-messages="err('limits.ram')"
          @change="limitField('ram', ($event.target as HTMLInputElement).value)"
        />
      </v-col>
      <v-col cols="6" md="2">
        <v-text-field
          v-model.lazy="seed"
          label="seed"
          type="number"
          min="0"
          :error-messages="err('seed')"
        />
      </v-col>
    </v-row>

    <h3 class="text-subtitle-1 mt-2 mb-1">Allowed degradations</h3>
    <div class="d-flex flex-wrap ga-4">
      <div v-for="id in ALLOWS" :key="id">
        <v-checkbox
          v-model="allow"
          :value="id"
          :label="allowLabel(id).title"
          :hint="allowLabel(id).note"
          persistent-hint
          :color="unlocks?.includes(id) ? 'warning' : 'primary'"
          :class="{ 'needs-opt-in': unlocks?.includes(id) }"
          density="compact"
        >
          <template v-if="unlocks?.includes(id)" #append>
            <v-chip color="warning" size="x-small">needed to fit</v-chip>
          </template>
        </v-checkbox>
        <div class="mono text-caption rg-muted ml-10">{{ id }}</div>
      </div>
    </div>
    <div v-if="err('allow').length" class="text-error text-body-2" role="alert">
      {{ err('allow').join('; ') }}
    </div>

    <template v-if="observe || observeEnabled === false">
      <h3 class="text-subtitle-1 mt-4 mb-1">Exported trajectories</h3>
      <v-switch
        :model-value="observeEnabled !== false"
        label="Export rollouts for the trajectories view"
        color="primary"
        density="compact"
        hide-details
        @update:model-value="observeEnabled = $event ? null : false"
      />
      <div v-if="observeEnabled !== false" class="d-flex align-center ga-2 flex-wrap">
        <span>One update out of</span>
        <v-text-field
          v-model.lazy="everyText"
          type="number"
          min="0"
          density="compact"
          hide-details="auto"
          style="max-width: 120px"
          :placeholder="resolvedEvery !== null ? String(resolvedEvery) : undefined"
          :error-messages="everyErrors ?? []"
          aria-label="observe.every"
        />
        <span class="text-body-2 rg-muted">
          <template v-if="observeEvery === null && resolvedEvery !== null"
            >server's choice: {{ resolvedEvery }}</template
          >
          <template v-else>1 exports every update</template>
        </span>
        <v-btn v-if="observeEvery !== null" size="small" variant="text" @click="observeEvery = null"
          >Let the server choose</v-btn
        >
      </div>
    </template>
  </div>
</template>
