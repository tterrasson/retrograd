<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { mdiRefresh, mdiRocketLaunch } from '@mdi/js'
import { useConfigSchema } from '@/api/configSchema'
import { useCapabilities, useDefaults } from '@/api/discovery'
import { useCreateRun, useLivePlan } from '@/api/plan'
import { isProblem, memoryShortfall } from '@/api/problem'
import type { Provenance } from '@/api/types'
import { problemFields } from '@/composables/useProblemFields'
import { useNotify } from '@/composables/useNotify'
import { useDraftStore } from '@/stores/draft'
import { getAt, removeAt, setAt } from '@/utils/pointer'
import { formatBytes } from '@/utils/format'
import {
  DatasetStep,
  ExpertEditor,
  MemoryBreakdown,
  ModelPicker,
  ObjectiveStep,
  ParamsTable,
  PlanSummaryCard,
} from '@/features/plan'
import { ProblemAlert } from '@/features/shared'

const draft = useDraftStore()
const router = useRouter()
const notify = useNotify()
const capabilities = useCapabilities()
const defaults = useDefaults()
const schema = useConfigSchema()
const create = useCreateRun()

const recipeForm = computed(() => draft.form === 'recipe')
/** A recipe is worth planning once it names a model and some data. */
const plannable = computed(() => {
  if (draft.form !== 'recipe') return true
  const data = draft.recipe.data
  return !!draft.recipe.model && !!(data.dataset || data.path)
})
const plan = useLivePlan(() => (plannable.value ? draft.input : null))
const createError = ref<unknown>(null)
const problem = computed(() => {
  const error = createError.value ?? plan.error.value
  return isProblem(error) ? error : null
})

const recipeErrors = computed(() => problemFields(problem.value, '/recipe'))
const paramErrors = computed(() => problemFields(problem.value, '/params'))
const unclaimed = computed(() =>
  (problem.value?.errors ?? []).filter(
    (error) => !error.pointer.startsWith('/recipe') && !error.pointer.startsWith('/params'),
  ),
)
function scoped(prefix: string): Record<string, string[]> {
  const out: Record<string, string[]> = {}
  for (const [path, messages] of Object.entries(recipeErrors.value.fields)) {
    if (path === prefix) out[''] = messages
    else if (path.startsWith(`${prefix}.`)) out[path.slice(prefix.length + 1)] = messages
  }
  return out
}
const modelErrors = computed(() => recipeErrors.value.fields.model ?? [])
const dataErrors = computed(() => scoped('data'))
const evalErrors = computed(() => scoped('eval'))
const objectiveErrors = computed(() => {
  const out: Record<string, string[]> = {}
  for (const [path, messages] of Object.entries(recipeErrors.value.fields)) {
    if (path === 'model' || path.startsWith('data') || path.startsWith('eval')) continue
    out[path] = messages
  }
  return out
})
const shortfall = computed(() => (problem.value ? memoryShortfall(problem.value) : null))

const effective = computed(() => plan.data.value?.effective_config ?? null)
const planProvenance = computed<Provenance | null>(() => plan.data.value?.provenance ?? null)
const algorithm = computed(() => {
  const value = getAt(effective.value, ['run', 'algorithm'])
  return typeof value === 'string' ? value : null
})
const observe = computed(() => {
  const value = getAt(effective.value, ['observe'])
  return typeof value === 'object' && value !== null ? (value as { every?: unknown }) : null
})
const observeEvery = computed({
  get: () => {
    const value = getAt(draft.params, ['observe', 'every'])
    return typeof value === 'number' ? value : null
  },
  set: (value: number | null) => {
    draft.params =
      value === null
        ? removeAt(draft.params, ['observe', 'every'])
        : setAt(draft.params, ['observe', 'every'], value)
  },
})
const observeEnabled = computed({
  get: () => (getAt(draft.params, ['observe', 'enabled']) === false ? false : null),
  set: (value: boolean | null) => {
    draft.params =
      value === false
        ? setAt(draft.params, ['observe', 'enabled'], false)
        : removeAt(draft.params, ['observe', 'enabled'])
  },
})
const everyErrors = computed(() => [
  ...(paramErrors.value.fields['observe.every'] ?? []),
  ...(paramErrors.value.fields['observe.enabled'] ?? []),
])
const useToml = computed({
  get: () => draft.form === 'toml',
  set: (value: boolean) => {
    draft.form = value ? 'toml' : draft.config ? 'config' : draft.forkFrom ? 'fork' : 'recipe'
  },
})

const steps = computed(() => [
  { title: 'Model', value: 1, error: modelErrors.value.length > 0 },
  {
    title: 'Data',
    value: 2,
    error: Object.keys(dataErrors.value).length + Object.keys(evalErrors.value).length > 0,
  },
  { title: 'Objective', value: 3, error: Object.keys(objectiveErrors.value).length > 0 },
  { title: 'Parameters', value: 4, error: Object.keys(paramErrors.value.fields).length > 0 },
  { title: 'Plan and launch', value: 5, error: !!problem.value && !!shortfall.value },
])

let idempotency: { input: string; key: string } | null = null
async function launch() {
  const input = draft.input
  if (!input) return
  // A response may be lost after the server has created the run. Reuse the
  // key for the same body, but never for a draft the user has since edited.
  const body = JSON.stringify(input)
  if (idempotency?.input !== body) idempotency = { input: body, key: crypto.randomUUID() }
  createError.value = null
  try {
    const run = await create.mutateAsync({ input, key: idempotency.key })
    idempotency = null
    notify.success(`Run ${run.name ?? run.id.slice(0, 8)} created`)
    draft.reset()
    await router.push(`/runs/${run.id}`)
  } catch (error) {
    createError.value = error
  }
}

function startOver() {
  draft.reset()
  idempotency = null
  createError.value = null
}
</script>

<template>
  <div>
    <div class="d-flex align-center flex-wrap ga-2 mb-3">
      <h1 class="text-h5">New run</h1>
      <v-chip v-if="draft.origin" color="secondary">{{ draft.origin }}</v-chip>
      <v-spacer />
      <v-progress-circular
        v-if="plan.pending.value"
        indeterminate
        size="20"
        width="2"
        color="primary"
        aria-label="Planning"
      />
      <v-btn variant="text" :prepend-icon="mdiRefresh" @click="startOver">Start over</v-btn>
    </div>

    <v-stepper v-model="draft.step" :items="steps" editable alt-labels hide-actions flat>
      <template #[`item.1`]>
        <template v-if="recipeForm">
          <ModelPicker v-model="draft.recipe.model" :errors="modelErrors" />
        </template>
        <v-alert v-else type="info" variant="tonal">
          The model comes from
          {{
            draft.form === 'toml'
              ? 'the TOML document'
              : (draft.origin ?? 'the copied configuration')
          }}.
          <v-btn variant="text" size="small" @click="startOver">Start from a recipe instead</v-btn>
        </v-alert>
      </template>

      <template #[`item.2`]>
        <DatasetStep
          v-if="recipeForm"
          v-model:data="draft.recipe.data"
          v-model:eval="draft.recipe.eval"
          :model="draft.recipe.model"
          :data-errors="dataErrors"
          :eval-errors="evalErrors"
        />
        <v-alert v-else type="info" variant="tonal"
          >The data comes from {{ draft.origin ?? 'the document' }}.</v-alert
        >
      </template>

      <template #[`item.3`]>
        <ObjectiveStep
          v-if="recipeForm"
          v-model:recipe="draft.recipe"
          v-model:observe-every="observeEvery"
          v-model:observe-enabled="observeEnabled"
          :errors="objectiveErrors"
          :observe="observe"
          :unlocks="shortfall?.unlocks"
          :every-errors="everyErrors"
        />
        <v-alert v-else type="info" variant="tonal"
          >The objective comes from {{ draft.origin ?? 'the document' }}.</v-alert
        >
      </template>

      <template #[`item.4`]>
        <v-expansion-panels class="mb-3">
          <v-expansion-panel title="Expert mode: params as JSON, or a TOML document">
            <v-expansion-panel-text>
              <ExpertEditor
                v-model:params="draft.params"
                v-model:toml="draft.toml"
                v-model:use-toml="useToml"
              />
            </v-expansion-panel-text>
          </v-expansion-panel>
        </v-expansion-panels>
        <v-text-field v-model="draft.name" label="Run name (optional)" style="max-width: 420px" />
        <ProblemAlert
          v-if="Object.keys(paramErrors.fields).length"
          :error="problem"
          :errors="problem?.fieldErrors('/params')"
          title="Some parameters were refused"
        />
        <div v-if="!plannable" class="pa-4 rg-muted">
          Choose a model and a dataset first: the table lists what the server resolves for them.
        </div>
        <ParamsTable
          v-else-if="draft.form !== 'toml'"
          v-model:params="draft.params"
          :schema="schema.data.value ?? null"
          :effective="effective ?? {}"
          :provenance="planProvenance"
          :defaults="defaults.data.value?.derived ?? []"
          :algorithm="algorithm"
          :errors="paramErrors.fields"
        />
      </template>

      <template #[`item.5`]>
        <ProblemAlert
          v-if="problem && !shortfall"
          :error="problem"
          :errors="unclaimed.length ? unclaimed : problem.errors"
        />
        <v-card v-if="shortfall" variant="outlined" color="error" class="mb-3">
          <v-card-title class="text-subtitle-1">Does not fit in memory</v-card-title>
          <v-card-text>
            <p class="mb-2">{{ problem?.detail }}</p>
            <MemoryBreakdown
              v-if="shortfall.dominantPosts.length"
              label="Largest posts"
              :segments="
                shortfall.dominantPosts.map((post) => ({ name: post.post, bytes: post.bytes }))
              "
              :budget="shortfall.vramBudgetBytes ?? null"
            />
            <div v-if="shortfall.overflowBytes" class="text-body-2">
              over by {{ formatBytes(shortfall.overflowBytes) }}
            </div>
            <div v-if="shortfall.leversApplied.length" class="text-body-2 mt-1">
              already applied: <span class="mono">{{ shortfall.leversApplied.join(', ') }}</span>
            </div>
            <div v-if="shortfall.unlocks.length" class="mt-2">
              Allowing <span class="mono">{{ shortfall.unlocks.join(', ') }}</span> may make it fit.
              <v-btn size="small" variant="tonal" color="warning" @click="draft.step = 3"
                >Review at the objective step</v-btn
              >
            </div>
          </v-card-text>
        </v-card>
        <PlanSummaryCard
          v-if="plan.data.value && !problem"
          :plan="plan.data.value.plan"
          :budgets="capabilities.data.value?.budgets"
        />
        <div v-else-if="!plan.data.value && !problem" class="pa-4 rg-muted">
          {{ plannable ? 'Planning…' : 'Choose a model and a dataset first.' }}
        </div>
        <div class="d-flex ga-2 mt-4">
          <v-btn
            color="primary"
            size="large"
            :prepend-icon="mdiRocketLaunch"
            :loading="create.isPending.value"
            :disabled="!draft.input || !plannable"
            @click="launch"
          >
            Launch
          </v-btn>
          <v-btn variant="text" :loading="plan.pending.value" @click="plan.refresh()"
            >Plan again</v-btn
          >
        </div>
      </template>
    </v-stepper>

    <div class="d-flex ga-2 mt-3">
      <v-btn variant="text" :disabled="draft.step <= 1" @click="draft.step--">Back</v-btn>
      <v-btn v-if="draft.step < 5" color="primary" variant="tonal" @click="draft.step++"
        >Next</v-btn
      >
    </div>
  </div>
</template>
