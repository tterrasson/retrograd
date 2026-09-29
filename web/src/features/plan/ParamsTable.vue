<script setup lang="ts">
import { computed, ref } from 'vue'
import { mdiBackupRestore, mdiInformationOutline, mdiMagnify, mdiPlus } from '@mdi/js'
import {
  crossSchema,
  filterSections,
  type JsonSchema,
  type ParamRow,
  type ParamSection,
} from '@/api/configSchema'
import type { DerivedField, Provenance } from '@/api/types'
import { formatValue } from '@/utils/format'
import { removeAt, setAt } from '@/utils/pointer'
import ParamEditor from './ParamEditor.vue'
import SourceChip from './SourceChip.vue'

const props = withDefaults(
  defineProps<{
    schema: JsonSchema | null
    effective: unknown
    provenance?: Provenance | null
    defaults?: DerivedField[]
    algorithm?: string | null
    readonly?: boolean
    /** Messages per dotted path, from a problem's `/params/…` pointers. */
    errors?: Record<string, string[]>
  }>(),
  { provenance: null, defaults: () => [], algorithm: null, readonly: false, errors: () => ({}) },
)
const params = defineModel<Record<string, unknown>>('params', { default: () => ({}) })

const search = ref('')
const view = ref<'all' | 'configured' | 'overrides'>(props.readonly ? 'configured' : 'all')

const sections = computed<ParamSection[]>(() =>
  crossSchema({
    schema: props.schema,
    effective: props.effective,
    provenance: props.provenance,
    params: params.value,
    defaults: props.defaults,
    algorithm: props.algorithm,
  }),
)
const visible = computed(() =>
  filterSections(sections.value, {
    search: search.value,
    overridesOnly: view.value === 'overrides',
    presentOnly: view.value === 'configured',
  }),
)
const overrideCount = computed(() =>
  sections.value.reduce(
    (count, section) => count + section.rows.filter((row) => row.override).length,
    0,
  ),
)
const opened = ref<string[]>([])
const panels = computed({
  get: () =>
    search.value || view.value === 'overrides'
      ? visible.value.map((section) => section.name)
      : opened.value,
  set: (value: string[]) => (opened.value = value),
})

function commit(row: ParamRow, value: unknown) {
  params.value = setAt(params.value, row.segments, value)
}
function restore(row: ParamRow) {
  params.value = removeAt(params.value, row.segments)
}
function enable(section: ParamSection) {
  let next = setAt(params.value, [section.name], {})
  for (const field of section.required)
    next = setAt(next, [section.name, ...field.segments], field.initial)
  params.value = next
  if (!opened.value.includes(section.name)) opened.value = [...opened.value, section.name]
}
function restoreSection(section: ParamSection) {
  params.value = removeAt(params.value, [section.name])
}
function sectionOverrides(section: ParamSection): number {
  return section.rows.filter((row) => row.override).length
}
</script>

<template>
  <div>
    <div class="d-flex flex-wrap align-center ga-2 mb-2">
      <v-text-field
        v-model="search"
        :prepend-inner-icon="mdiMagnify"
        label="Search a path or a description"
        clearable
        hide-details
        density="compact"
        style="max-width: 420px"
      />
      <v-btn-toggle v-model="view" mandatory density="compact" color="primary">
        <v-btn value="all" size="small">All</v-btn>
        <v-btn value="configured" size="small">Configured</v-btn>
        <v-btn v-if="!readonly" value="overrides" size="small">Pinned ({{ overrideCount }})</v-btn>
      </v-btn-toggle>
    </div>
    <v-alert v-if="!schema" variant="tonal" density="compact" color="info" class="mb-2">
      The server publishes no configuration schema: only what the plan configured is listed.
    </v-alert>

    <v-expansion-panels v-model="panels" multiple variant="accordion">
      <v-expansion-panel v-for="section in visible" :key="section.name" :value="section.name">
        <v-expansion-panel-title>
          <div class="d-flex align-center ga-2 flex-wrap">
            <span class="mono font-weight-medium">[{{ section.name }}]</span>
            <v-chip v-if="!section.present" size="x-small">not enabled</v-chip>
            <v-chip v-if="sectionOverrides(section)" size="x-small" color="secondary">
              {{ sectionOverrides(section) }} pinned
            </v-chip>
            <span
              v-if="section.description"
              class="text-caption rg-muted text-truncate"
              style="max-width: 40rem"
            >
              {{ section.description }}
            </span>
          </div>
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <div v-if="!section.present && !readonly" class="mb-2">
            <v-btn
              size="small"
              variant="tonal"
              color="primary"
              :prepend-icon="mdiPlus"
              @click="enable(section)"
            >
              Enable [{{ section.name }}]
            </v-btn>
          </div>
          <v-table density="compact" class="params">
            <thead>
              <tr>
                <th scope="col">Path</th>
                <th scope="col">Value</th>
                <th scope="col">Source</th>
                <th v-if="!readonly" scope="col">Edit</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="row in section.rows"
                :key="row.path"
                :class="{ 'row-error': errors[row.path]?.length }"
              >
                <td class="path">
                  <span class="mono">{{ row.label }}</span>
                  <v-tooltip v-if="row.description || row.rule" location="top" max-width="420">
                    <template #activator="{ props: activator }">
                      <v-icon
                        v-bind="activator"
                        :icon="mdiInformationOutline"
                        size="14"
                        class="ml-1"
                        :aria-label="`about ${row.path}`"
                      />
                    </template>
                    <div v-if="row.description">{{ row.description }}</div>
                    <div v-if="row.rule" class="mt-1">
                      <strong>rule:</strong> {{ row.rule.rule }}
                      <span v-if="row.rule.thresholds">
                        ({{
                          Object.entries(row.rule.thresholds)
                            .map(([k, v]) => `${k} ${v}`)
                            .join(', ')
                        }})
                      </span>
                    </div>
                  </v-tooltip>
                  <v-chip v-if="row.patchable" size="x-small" class="ml-1">live</v-chip>
                  <div v-if="readonly && errors[row.path]" class="text-error text-caption">
                    {{ errors[row.path]?.join('; ') }}
                  </div>
                </td>
                <td class="value mono">
                  <span v-if="row.present">{{ formatValue(row.value) }}</span>
                  <span v-else class="rg-muted">not enabled</span>
                </td>
                <td>
                  <SourceChip :origin="row.origin" :override="row.override && !readonly" />
                </td>
                <td v-if="!readonly" class="edit">
                  <div class="d-flex align-center ga-1">
                    <ParamEditor
                      :row="row"
                      :root="schema"
                      :errors="errors[row.path]"
                      @commit="commit(row, $event)"
                    />
                    <v-btn
                      v-if="row.override"
                      :icon="mdiBackupRestore"
                      size="small"
                      variant="text"
                      :aria-label="`Give ${row.path} back to the server`"
                      title="Give back to the server"
                      @click="restore(row)"
                    />
                  </div>
                </td>
              </tr>
            </tbody>
          </v-table>
          <div v-if="!readonly && sectionOverrides(section)" class="mt-2">
            <v-btn size="small" variant="text" @click="restoreSection(section)"
              >Give the whole section back</v-btn
            >
          </div>
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
    <div v-if="!visible.length" class="pa-4 rg-muted">No parameter matches.</div>
  </div>
</template>

<style scoped>
.params .path {
  white-space: nowrap;
  vertical-align: top;
  padding-top: 10px;
}
.params .value {
  max-width: 22rem;
  word-break: break-word;
  vertical-align: top;
  padding-top: 10px;
}
.params .edit {
  min-width: 260px;
  padding-block: 4px;
}
.row-error {
  background: rgba(var(--v-theme-error), 0.06);
}
</style>
