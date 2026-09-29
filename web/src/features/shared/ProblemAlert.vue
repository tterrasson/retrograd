<script setup lang="ts">
import { computed } from 'vue'
import { toProblem } from '@/api/problem'
import type { FieldError } from '@/api/types'

const props = withDefaults(
  defineProps<{
    error: unknown
    /** Only these field errors are listed; by default, all of them. */
    errors?: FieldError[]
    title?: string
    closable?: boolean
  }>(),
  { errors: undefined, title: undefined, closable: false },
)
defineEmits<{ close: [] }>()

const problem = computed(() => (props.error ? toProblem(props.error) : null))
const listed = computed(() => props.errors ?? problem.value?.errors ?? [])
const color = computed(() => {
  const status = problem.value?.status ?? 0
  return status >= 500 || status === 0 ? 'error' : 'warning'
})
</script>

<template>
  <v-alert
    v-if="problem"
    :color="color"
    variant="tonal"
    :closable="closable"
    role="alert"
    class="mb-3"
    @click:close="$emit('close')"
  >
    <div class="font-weight-medium">{{ title ?? problem.title }}</div>
    <div v-if="problem.detail" class="text-body-2">{{ problem.detail }}</div>
    <ul v-if="listed.length" class="mt-1 text-body-2 pl-4">
      <li v-for="(item, index) in listed" :key="index">
        <code v-if="item.pointer" class="mono">{{ item.pointer }}</code>
        {{ item.message }}
        <span v-if="item.hint" class="rg-muted">— {{ item.hint }}</span>
      </li>
    </ul>
    <div v-if="problem.traceId" class="text-caption mt-1">
      trace id <code class="mono">{{ problem.traceId }}</code>
    </div>
    <slot />
  </v-alert>
</template>
