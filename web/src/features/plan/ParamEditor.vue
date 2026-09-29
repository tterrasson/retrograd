<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import {
  enumOptions,
  editorFor,
  unwrapNullable,
  type JsonSchema,
  type ParamRow,
} from '@/api/configSchema'

const props = defineProps<{ row: ParamRow; root: JsonSchema | null; errors?: string[] }>()
const emit = defineEmits<{ commit: [unknown] }>()

/** What the editor shows: the pinned value, else the resolved one. */
const current = computed(() => (props.row.override ? props.row.paramValue : props.row.value))
const label = computed(() => props.row.label)
const hint = computed(() => {
  const s = props.row.schema
  const bounds: string[] = []
  if (s.minimum !== undefined) bounds.push(`≥ ${s.minimum}`)
  if (s.exclusiveMinimum !== undefined) bounds.push(`> ${s.exclusiveMinimum}`)
  if (s.maximum !== undefined) bounds.push(`≤ ${s.maximum}`)
  if (s.exclusiveMaximum !== undefined) bounds.push(`< ${s.exclusiveMaximum}`)
  return bounds.join(', ')
})

const text = ref('')
const jsonError = ref<string | null>(null)
const reset = () => {
  const value = current.value
  if (props.row.editor === 'json' || props.row.editor === 'variant') {
    text.value = value === undefined ? '' : JSON.stringify(value, null, 2)
  } else {
    text.value = value === undefined || value === null ? '' : String(value)
  }
  jsonError.value = null
}
watch(current, reset, { immediate: true })

const options = computed(() => {
  const items = enumOptions(props.row.schema).map((value) => ({ title: String(value), value }))
  return props.row.nullable ? [{ title: 'null', value: null }, ...items] : items
})

function commitNumber(integer: boolean) {
  const trimmed = text.value.trim()
  if (trimmed === '') {
    if (props.row.nullable) emit('commit', null)
    return
  }
  const value = Number(trimmed)
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    jsonError.value = integer ? 'an integer' : 'a number'
    return
  }
  jsonError.value = null
  if (value !== current.value) emit('commit', value)
}

function commitText() {
  if (text.value === (current.value ?? '')) return
  emit('commit', text.value === '' && props.row.nullable ? null : text.value)
}

function commitJson() {
  const trimmed = text.value.trim()
  if (trimmed === '') {
    if (props.row.nullable) emit('commit', null)
    return
  }
  try {
    const value: unknown = JSON.parse(trimmed)
    jsonError.value = null
    emit('commit', value)
  } catch (error) {
    jsonError.value = error instanceof Error ? error.message : 'invalid JSON'
  }
}

const chips = computed<string[]>(() =>
  Array.isArray(current.value) ? (current.value as unknown[]).map((item) => String(item)) : [],
)
function commitChips(values: string[]) {
  const itemType = props.row.schema.items?.type
  const numeric = itemType === 'integer' || itemType === 'number'
  emit('commit', numeric ? values.map(Number) : values)
}

/** The variants of a `oneOf`, each with the editor it would call for. */
const variants = computed(() =>
  (props.row.schema.oneOf ?? []).map((variant, index) => {
    const { schema } = props.root ? unwrapNullable(variant, props.root) : { schema: variant }
    const type = Array.isArray(schema.type) ? schema.type.join(' | ') : schema.type
    return {
      index,
      schema,
      title: schema.title ?? schema.description?.split('\n')[0] ?? type ?? `variant ${index + 1}`,
      editor: editorFor(schema),
    }
  }),
)
const variantIndex = ref(0)
const errorMessages = computed(() => [
  ...(props.errors ?? []),
  ...(jsonError.value ? [jsonError.value] : []),
])
</script>

<template>
  <div class="param-editor">
    <v-select
      v-if="row.editor === 'select'"
      :model-value="current"
      :items="options"
      :label="label"
      density="compact"
      hide-details="auto"
      :error-messages="errorMessages"
      @update:model-value="emit('commit', $event)"
    />
    <v-switch
      v-else-if="row.editor === 'switch'"
      :model-value="current === true"
      :label="String(current ?? '–')"
      color="primary"
      density="compact"
      hide-details="auto"
      :error-messages="errorMessages"
      @update:model-value="emit('commit', $event)"
    />
    <v-text-field
      v-else-if="row.editor === 'integer' || row.editor === 'number'"
      v-model="text"
      type="number"
      :step="row.editor === 'integer' ? 1 : 'any'"
      :min="row.schema.minimum"
      :max="row.schema.maximum"
      :hint="hint"
      :placeholder="row.nullable ? 'null' : undefined"
      density="compact"
      hide-details="auto"
      :error-messages="errorMessages"
      @change="commitNumber(row.editor === 'integer')"
      @keydown.enter="commitNumber(row.editor === 'integer')"
    />
    <v-text-field
      v-else-if="row.editor === 'text'"
      v-model="text"
      density="compact"
      hide-details="auto"
      :placeholder="row.nullable ? 'null' : undefined"
      :error-messages="errorMessages"
      @change="commitText"
      @keydown.enter="commitText"
    />
    <v-combobox
      v-else-if="row.editor === 'chips'"
      :model-value="chips"
      multiple
      chips
      closable-chips
      density="compact"
      hide-details="auto"
      :error-messages="errorMessages"
      @update:model-value="commitChips($event as string[])"
    />
    <div v-else-if="row.editor === 'variant'">
      <v-select
        v-model="variantIndex"
        :items="variants"
        item-title="title"
        item-value="index"
        label="variant"
        density="compact"
        hide-details
        class="mb-1"
      />
      <v-textarea
        v-model="text"
        rows="2"
        auto-grow
        class="mono"
        density="compact"
        hide-details="auto"
        :hint="`JSON for ${variants[variantIndex]?.title ?? 'this variant'}`"
        persistent-hint
        :error-messages="errorMessages"
        @change="commitJson"
      />
    </div>
    <v-textarea
      v-else
      v-model="text"
      rows="2"
      auto-grow
      class="mono"
      density="compact"
      hide-details="auto"
      hint="JSON"
      :error-messages="errorMessages"
      @change="commitJson"
    />
  </div>
</template>

<style scoped>
.param-editor {
  min-width: 180px;
}
</style>
