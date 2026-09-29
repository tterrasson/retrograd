<script setup lang="ts">
defineProps<{ icon?: string; title: string; text?: string }>()
</script>

<template>
  <div class="d-flex flex-column align-center text-center py-10 px-4 rg-muted" role="status">
    <v-icon v-if="icon" :icon="icon" size="48" class="mb-3" />
    <div class="text-h6">{{ title }}</div>
    <div v-if="text" class="text-body-2 mt-1" style="max-width: 36rem">{{ text }}</div>
    <div class="mt-4"><slot /></div>
  </div>
</template>
