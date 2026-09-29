<script setup lang="ts">
import { toProblem } from '@/api/problem'
import { MarkdownText } from '@/features/shared'
import type { PaneTurn } from './types'

defineProps<{
  title: string
  turns: PaneTurn[]
  state: 'idle' | 'waiting' | 'streaming' | 'error'
  error?: unknown
  markdown?: boolean
  usage?: string | null
}>()
</script>

<template>
  <v-card class="pane h-100 d-flex flex-column">
    <v-card-title class="text-subtitle-2 mono d-flex align-center">
      <span class="text-truncate">{{ title }}</span>
      <v-spacer />
      <v-chip v-if="state === 'waiting'" size="x-small" color="info">waiting for the device</v-chip>
      <v-chip v-else-if="state === 'streaming'" size="x-small" color="primary">streaming</v-chip>
    </v-card-title>
    <v-card-text class="flex-grow-1" aria-live="polite">
      <div v-for="(turn, index) in turns" :key="index" class="mb-3">
        <div class="text-caption text-uppercase rg-muted">{{ turn.role }}</div>
        <MarkdownText v-if="markdown && turn.role === 'assistant'" :text="turn.content" />
        <pre v-else class="text-pre">{{ turn.content }}</pre>
      </div>
      <v-alert v-if="state === 'error' && error" type="error" variant="tonal" density="compact">
        {{ toProblem(error).detail || toProblem(error).title }}
      </v-alert>
      <div v-if="usage" class="text-caption rg-muted">{{ usage }}</div>
    </v-card-text>
  </v-card>
</template>
