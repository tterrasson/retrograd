<script setup lang="ts">
import { formatDate, formatRelative } from '@/utils/format'

defineProps<{ at: number | null | undefined }>()
</script>

<template>
  <span v-if="at" :title="formatDate(at)">{{ formatRelative(at) }}</span>
  <span v-else class="rg-muted">–</span>
</template>
