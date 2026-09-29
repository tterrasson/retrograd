<script setup lang="ts">
import { ref } from 'vue'
import { mdiCheck, mdiContentCopy } from '@mdi/js'

const props = withDefaults(defineProps<{ text: string; label?: string; size?: string }>(), {
  label: 'Copy',
  size: 'small',
})
const done = ref(false)

async function copy() {
  try {
    await navigator.clipboard.writeText(props.text)
    done.value = true
    setTimeout(() => (done.value = false), 1500)
  } catch {
    done.value = false
  }
}
</script>

<template>
  <v-btn
    :icon="done ? mdiCheck : mdiContentCopy"
    :size="size"
    variant="text"
    :aria-label="label"
    :title="label"
    @click.stop="copy"
  />
</template>
