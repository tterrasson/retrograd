<script setup lang="ts">
import { computed } from 'vue'

defineOptions({ name: 'JsonTree' })

const props = withDefaults(
  defineProps<{ value: unknown; name?: string; depth?: number; openDepth?: number }>(),
  { name: undefined, depth: 0, openDepth: 1 },
)

const isObject = computed(
  () => typeof props.value === 'object' && props.value !== null && !Array.isArray(props.value),
)
const isArray = computed(() => Array.isArray(props.value))
const entries = computed<[string, unknown][]>(() => {
  if (isArray.value) return (props.value as unknown[]).map((item, index) => [String(index), item])
  if (isObject.value) return Object.entries(props.value as Record<string, unknown>)
  return []
})
const summary = computed(() =>
  isArray.value ? `[${entries.value.length}]` : `{${entries.value.length}}`,
)
const scalar = computed(() => {
  const value = props.value
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  return String(value)
})
const scalarClass = computed(() => {
  const value = props.value
  if (value === null) return 'text-disabled'
  if (typeof value === 'string') return 'text-secondary'
  if (typeof value === 'number') return 'text-primary'
  return 'text-warning'
})
</script>

<template>
  <div class="json-tree mono">
    <details v-if="isObject || isArray" :open="depth < openDepth">
      <summary>
        <span v-if="name !== undefined" class="font-weight-medium">{{ name }}: </span>
        <span class="rg-muted">{{ summary }}</span>
      </summary>
      <div class="pl-4">
        <JsonTree
          v-for="[key, child] in entries"
          :key="key"
          :name="key"
          :value="child"
          :depth="depth + 1"
          :open-depth="openDepth"
        />
      </div>
    </details>
    <div v-else>
      <span v-if="name !== undefined" class="font-weight-medium">{{ name }}: </span>
      <span :class="scalarClass">{{ scalar }}</span>
    </div>
  </div>
</template>

<style scoped>
.json-tree summary {
  cursor: pointer;
}
.json-tree {
  line-height: 1.5;
}
</style>
