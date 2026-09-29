<script setup lang="ts">
import { computed } from 'vue'
import type { Origin } from '@/api/types'

const props = defineProps<{ origin?: Origin; override?: boolean }>()
const source = computed(() => (props.override ? 'override' : props.origin?.source))
const color = computed(() => {
  switch (source.value) {
    case 'override':
      return 'secondary'
    case 'derived':
      return 'primary'
    case 'measured':
      return 'success'
    default:
      return undefined
  }
})
</script>

<template>
  <v-tooltip v-if="source" :disabled="!origin?.reason" location="top" max-width="360">
    <template #activator="{ props: activator }">
      <v-chip
        v-bind="activator"
        :color="color"
        label
        :aria-label="`source ${source}${origin?.reason ? `: ${origin.reason}` : ''}`"
      >
        {{ source }}
      </v-chip>
    </template>
    {{ origin?.reason }}
  </v-tooltip>
</template>
