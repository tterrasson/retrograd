<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { mdiDeleteSweepOutline } from '@mdi/js'
import { useFeature } from '@/api/discovery'
import { streamChat, useOpenAiModels, type ChatMessage } from '@/api/openai'
import { runLabel, statusInfo, useRunCommands, useRuns } from '@/api/runs'
import { ChatPane, PromptBar, TargetPicker, type PaneTurn } from '@/features/compare'
import { ProblemAlert } from '@/features/shared'

const route = useRoute()
const serving = useFeature('serving_enabled')
const models = useOpenAiModels(() => serving.value === true)

const MAX_COLUMNS = 6
const mode = ref<'pair' | 'grid'>('pair')
const pairRun = ref<string | null>(typeof route.query.run === 'string' ? route.query.run : null)
const gridTargets = ref<string[]>(
  typeof route.query.targets === 'string' ? route.query.targets.split(',').filter(Boolean) : [],
)
if (gridTargets.value.length) mode.value = 'grid'

const system = ref('')
const temperature = ref(0.7)
const topP = ref(1)
const maxTokens = ref(512)
const seed = ref<number | null>(0)
const markdown = ref(true)

const runIds = computed(() => {
  const ids = new Set<string>()
  for (const model of models.data.value ?? []) ids.add(model.id.split('@')[0] ?? model.id)
  return [...ids]
})
const targets = computed<string[]>(() => {
  if (mode.value === 'grid') return gridTargets.value.slice(0, MAX_COLUMNS)
  return pairRun.value ? [`${pairRun.value}@base`, pairRun.value] : []
})

interface Column {
  target: string
  turns: PaneTurn[]
  state: 'idle' | 'waiting' | 'streaming' | 'error'
  error: unknown
  usage: string | null
}
const columns = ref<Column[]>([])
watch(
  targets,
  (list) => {
    columns.value = list.map(
      (target) =>
        columns.value.find((column) => column.target === target) ?? {
          target,
          turns: [],
          state: 'idle',
          error: null,
          usage: null,
        },
    )
  },
  { immediate: true },
)

let controller: AbortController | null = null
const busy = computed(() =>
  columns.value.some((column) => column.state === 'waiting' || column.state === 'streaming'),
)

async function sendTo(column: Column, message: string, signal: AbortSignal) {
  const history: ChatMessage[] = [
    ...(system.value.trim() ? [{ role: 'system' as const, content: system.value.trim() }] : []),
    ...column.turns.map((turn) => ({ role: turn.role, content: turn.content })),
  ]
  column.turns.push({ role: 'user', content: message })
  column.turns.push({ role: 'assistant', content: '' })
  // The reactive copy, so each delta repaints the column.
  const reply = column.turns[column.turns.length - 1] as PaneTurn
  column.state = 'waiting'
  column.error = null
  column.usage = null
  try {
    const usage = await streamChat(
      {
        model: column.target,
        messages: [...history, { role: 'user', content: message }],
        temperature: temperature.value,
        top_p: topP.value,
        max_tokens: maxTokens.value,
        ...(seed.value !== null ? { seed: seed.value } : {}),
      },
      (delta) => {
        column.state = 'streaming'
        reply.content += delta
      },
      signal,
    )
    column.state = 'idle'
    if (usage?.completion_tokens !== undefined) {
      column.usage = `${usage.prompt_tokens ?? '?'} prompt tokens, ${usage.completion_tokens} generated`
    }
  } catch (error) {
    column.state = 'error'
    column.error = error
  }
}

function send(message: string) {
  controller?.abort()
  controller = new AbortController()
  const signal = controller.signal
  // One stream per column, all at once: the server serializes them on the
  // device, and each column says so while it waits.
  for (const column of columns.value) void sendTo(column, message, signal)
}
function stop() {
  controller?.abort()
  for (const column of columns.value) if (column.state !== 'error') column.state = 'idle'
}
function clear() {
  for (const column of columns.value) {
    column.turns = []
    column.state = 'idle'
    column.error = null
    column.usage = null
  }
}
onBeforeUnmount(() => controller?.abort())

// Without serving: an active run answers through its own generate route, with the base beside it.
const runs = useRuns(() => ({ limit: 100 }))
const activeRuns = computed(() =>
  (runs.data.value?.runs ?? []).filter((run) => !statusInfo(run.status).terminal),
)
const generateRun = ref<string | null>(typeof route.query.run === 'string' ? route.query.run : null)
const commands = useRunCommands(() => generateRun.value ?? '')
const generatePrompt = ref('')
const generation = ref<{ text: string; base: string | null } | null>(null)
const generateError = ref<unknown>(null)
async function generate() {
  if (!generateRun.value) return
  generateError.value = null
  try {
    const result = await commands.generate.mutateAsync({
      prompt: generatePrompt.value,
      chat: true,
      include_base: true,
      max_new_tokens: maxTokens.value,
      temperature: temperature.value,
      top_p: topP.value,
      seed: seed.value,
    })
    generation.value = { text: result.text, base: result.base_text ?? null }
  } catch (error) {
    generateError.value = error
  }
}
</script>

<template>
  <div>
    <h1 class="text-h5 mb-3">Compare</h1>

    <template v-if="serving === false">
      <v-alert type="info" variant="tonal" class="mb-3">
        This server does not serve models (<span class="mono">serving</span> is off), so only a run
        that is training can answer - through its own generation route, with the base model beside
        it.
      </v-alert>
      <v-card class="pa-3">
        <v-select
          v-model="generateRun"
          :items="activeRuns.map((run) => ({ title: runLabel(run), value: run.id }))"
          label="Active run"
          :loading="runs.isFetching.value"
          no-data-text="No run is active."
        />
        <v-textarea v-model="generatePrompt" label="Prompt" rows="3" auto-grow />
        <v-btn
          color="primary"
          :disabled="!generateRun || !generatePrompt.trim()"
          :loading="commands.generate.isPending.value"
          @click="generate"
        >
          Generate
        </v-btn>
        <ProblemAlert :error="generateError" class="mt-3" />
        <v-row v-if="generation" class="mt-2">
          <v-col cols="12" md="6">
            <ChatPane
              title="base"
              :turns="[{ role: 'assistant', content: generation.base ?? '' }]"
              state="idle"
              :markdown="markdown"
            />
          </v-col>
          <v-col cols="12" md="6">
            <ChatPane
              title="adapter"
              :turns="[{ role: 'assistant', content: generation.text }]"
              state="idle"
              :markdown="markdown"
            />
          </v-col>
        </v-row>
      </v-card>
    </template>

    <template v-else>
      <ProblemAlert :error="models.error.value" />
      <v-card class="pa-3 mb-3">
        <v-btn-toggle v-model="mode" mandatory color="primary" density="compact" class="mb-3">
          <v-btn value="pair">Base vs adapter</v-btn>
          <v-btn value="grid">One prompt, several targets</v-btn>
        </v-btn-toggle>
        <v-row dense>
          <v-col cols="12" md="6">
            <v-select
              v-if="mode === 'pair'"
              v-model="pairRun"
              :items="runIds"
              label="Run"
              :loading="models.isFetching.value"
              no-data-text="No run has weights to serve."
            />
            <TargetPicker
              v-else
              v-model="gridTargets"
              :models="models.data.value ?? []"
              :max="MAX_COLUMNS"
              :loading="models.isFetching.value"
            />
          </v-col>
          <v-col cols="12" md="6">
            <v-textarea v-model="system" label="System prompt (optional)" rows="1" auto-grow />
          </v-col>
        </v-row>
        <v-row dense>
          <v-col cols="6" sm="3"
            ><v-text-field
              v-model.number="temperature"
              type="number"
              step="0.1"
              min="0"
              label="temperature"
          /></v-col>
          <v-col cols="6" sm="3"
            ><v-text-field
              v-model.number="topP"
              type="number"
              step="0.05"
              min="0"
              max="1"
              label="top_p"
          /></v-col>
          <v-col cols="6" sm="3"
            ><v-text-field v-model.number="maxTokens" type="number" min="1" label="max_tokens"
          /></v-col>
          <v-col cols="6" sm="3">
            <v-text-field
              :model-value="seed ?? ''"
              type="number"
              min="0"
              label="seed"
              hint="fixed so that columns differ by their weights only"
              @update:model-value="seed = $event === '' ? null : Number($event)"
            />
          </v-col>
        </v-row>
        <div class="d-flex align-center ga-2">
          <v-switch
            v-model="markdown"
            label="render markdown"
            color="primary"
            density="compact"
            hide-details
          />
          <v-spacer />
          <v-btn variant="text" :prepend-icon="mdiDeleteSweepOutline" @click="clear"
            >Clear the conversation</v-btn
          >
        </div>
      </v-card>

      <v-row v-if="columns.length" dense class="mb-3">
        <v-col
          v-for="column in columns"
          :key="column.target"
          cols="12"
          :md="columns.length > 2 ? 4 : 6"
          :lg="columns.length > 3 ? 4 : 12 / columns.length"
        >
          <ChatPane
            :title="column.target"
            :turns="column.turns"
            :state="column.state"
            :error="column.error"
            :usage="column.usage"
            :markdown="markdown"
          />
        </v-col>
      </v-row>
      <div v-else class="pa-4 rg-muted">Pick a run, or at least two targets.</div>

      <PromptBar :busy="busy" :disabled="!columns.length" @send="send" @stop="stop" />
    </template>
  </div>
</template>
