<script setup lang="ts">
import { ref } from 'vue'
import { mdiSend, mdiStop } from '@mdi/js'

defineProps<{ busy?: boolean; disabled?: boolean }>()
const emit = defineEmits<{ send: [string]; stop: [] }>()
const text = ref('')

function send() {
  const value = text.value.trim()
  if (!value) return
  emit('send', value)
  text.value = ''
}
</script>

<template>
  <form class="d-flex align-end ga-2" @submit.prevent="send">
    <v-textarea
      v-model="text"
      label="Message"
      rows="2"
      auto-grow
      hide-details
      :disabled="disabled"
      @keydown.enter.exact.prevent="send"
    />
    <v-btn
      v-if="busy"
      :icon="mdiStop"
      color="error"
      variant="tonal"
      aria-label="Stop"
      @click="emit('stop')"
    />
    <v-btn
      v-else
      type="submit"
      :icon="mdiSend"
      color="primary"
      :disabled="disabled || !text.trim()"
      aria-label="Send"
    />
  </form>
</template>
