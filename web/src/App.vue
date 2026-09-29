<script setup lang="ts">
import AppShell from '@/layouts/AppShell.vue'
import { useThemeSync } from '@/composables/useThemeSync'
import { useNotify } from '@/composables/useNotify'
import { computed } from 'vue'

useThemeSync()
const notify = useNotify()
const current = computed(() => notify.queue.value[0] ?? null)
const open = computed({
  get: () => current.value !== null,
  set: (value: boolean) => {
    if (!value && current.value) notify.dismiss(current.value.id)
  },
})
</script>

<template>
  <v-app>
    <AppShell>
      <router-view />
    </AppShell>
    <v-snackbar
      v-model="open"
      :color="current?.color"
      :timeout="current?.color === 'error' ? 8000 : 4000"
      location="bottom end"
      role="status"
      aria-live="polite"
    >
      <div>{{ current?.text }}</div>
      <div v-if="current?.detail" class="mono text-caption">{{ current.detail }}</div>
      <template #actions>
        <v-btn variant="text" @click="open = false">Close</v-btn>
      </template>
    </v-snackbar>
  </v-app>
</template>
