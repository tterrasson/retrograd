<script setup lang="ts">
import { computed } from 'vue'
import { statusInfo } from '@/api/runs'
import type { RunStatus } from '@/api/types'

const props = defineProps<{ status: RunStatus | null | undefined; live?: boolean }>()
const info = computed(() => statusInfo(props.status))
</script>

<template>
  <v-chip :color="info.color" :aria-live="live ? 'polite' : undefined" role="status" label>
    <v-progress-circular
      v-if="info.busy"
      indeterminate
      size="10"
      width="2"
      class="mr-1"
      aria-hidden="true"
    />
    {{ info.label }}
  </v-chip>
</template>
