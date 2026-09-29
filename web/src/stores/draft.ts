import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { PlanInput, Params } from '@/api/plan'
import type { ConfigDocument, ForkFrom, Recipe } from '@/api/types'

const STORAGE_KEY = 'retrograd.draft'

export type DraftForm = PlanInput['form']

interface Stored {
  form: DraftForm
  step: number
  name: string
  recipe: Recipe
  config: ConfigDocument | null
  forkFrom: ForkFrom | null
  params: Params
  toml: string
  origin: string | null
}

function emptyRecipe(): Recipe {
  return { model: '', data: {} }
}

function blank(): Stored {
  return {
    form: 'recipe',
    step: 1,
    name: '',
    recipe: emptyRecipe(),
    config: null,
    forkFrom: null,
    params: {},
    toml: '',
    origin: null,
  }
}

function load(): Stored {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? { ...blank(), ...(JSON.parse(raw) as Partial<Stored>) } : blank()
  } catch {
    return blank()
  }
}

/** The "new run" form, kept across reloads until the run is created or discarded. */
export const useDraftStore = defineStore('draft', () => {
  const initial = load()
  const form = ref<DraftForm>(initial.form)
  const step = ref(initial.step)
  const name = ref(initial.name)
  const recipe = ref<Recipe>(initial.recipe)
  const config = ref<ConfigDocument | null>(initial.config)
  const forkFrom = ref<ForkFrom | null>(initial.forkFrom)
  const params = ref<Params>(initial.params)
  const toml = ref(initial.toml)
  /** A human note of where the draft came from ("duplicate of …"). */
  const origin = ref<string | null>(initial.origin)

  watch(
    [form, step, name, recipe, config, forkFrom, params, toml, origin],
    () => {
      try {
        const value: Stored = {
          form: form.value,
          step: step.value,
          name: name.value,
          recipe: recipe.value,
          config: config.value,
          forkFrom: forkFrom.value,
          params: params.value,
          toml: toml.value,
          origin: origin.value,
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(value))
      } catch {
        // Not persisted.
      }
    },
    { deep: true },
  )

  /** The body the plan and the creation are sent, from the current form. */
  const input = computed<PlanInput | null>(() => {
    const base = { params: params.value, ...(name.value ? { name: name.value } : {}) }
    switch (form.value) {
      case 'recipe':
        return { form: 'recipe', recipe: recipe.value, ...base }
      case 'config':
        return config.value ? { form: 'config', config: config.value, ...base } : null
      case 'fork':
        return forkFrom.value ? { form: 'fork', fork_from: forkFrom.value, ...base } : null
      case 'toml':
        return toml.value.trim() ? { form: 'toml', toml: toml.value } : null
      default:
        return null
    }
  })

  function reset() {
    const fresh = blank()
    form.value = fresh.form
    step.value = fresh.step
    name.value = fresh.name
    recipe.value = fresh.recipe
    config.value = fresh.config
    forkFrom.value = fresh.forkFrom
    params.value = fresh.params
    toml.value = fresh.toml
    origin.value = fresh.origin
  }

  /** Starts from a run's own configuration. */
  function duplicate(runId: string, runName: string, effective: ConfigDocument) {
    reset()
    form.value = 'config'
    config.value = effective
    name.value = runName ? `${runName} (copy)` : ''
    origin.value = `duplicate of ${runName || runId}`
    step.value = 4
  }

  /** Continues a run from one of its checkpoints. */
  function fork(runId: string, runName: string, checkpoint?: string) {
    reset()
    form.value = 'fork'
    forkFrom.value = checkpoint ? { run: runId, checkpoint } : { run: runId }
    name.value = runName ? `${runName} (fork)` : ''
    origin.value = `fork of ${runName || runId}${checkpoint ? ` at ${checkpoint}` : ''}`
    step.value = 4
  }

  return {
    form,
    step,
    name,
    recipe,
    config,
    forkFrom,
    params,
    toml,
    origin,
    input,
    reset,
    duplicate,
    fork,
  }
})
