<script setup lang="ts">
defineProps<{ label: string; value: string; hint?: string }>()
</script>

<template>
  <v-card class="pa-3 h-100">
    <div class="text-caption rg-muted">{{ label }}</div>
    <div class="text-h6 mono" :title="hint">{{ value }}</div>
    <slot />
  </v-card>
</template>
