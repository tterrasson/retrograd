<script setup lang="ts">
import { ref, watch } from 'vue'
import type { CancelRequest } from '@/api/types'

const open = defineModel<boolean>({ required: true })
defineProps<{ busy?: boolean }>()
const emit = defineEmits<{ confirm: [CancelRequest] }>()
const at = ref<'boundary' | 'now'>('boundary')
const checkpoint = ref(false)
watch(open, (value) => {
  if (value) {
    at.value = 'boundary'
    checkpoint.value = false
  }
})
</script>

<template>
  <v-dialog v-model="open" max-width="460">
    <v-card>
      <v-card-title>Cancel the run</v-card-title>
      <v-card-text>
        <v-radio-group v-model="at" label="When">
          <v-radio value="boundary" label="At the end of the current iteration" />
          <v-radio value="now" label="Now" />
        </v-radio-group>
        <v-switch
          v-model="checkpoint"
          label="Write a checkpoint first"
          color="primary"
          hide-details
        />
      </v-card-text>
      <v-card-actions>
        <v-spacer />
        <v-btn variant="text" @click="open = false">Keep running</v-btn>
        <v-btn color="error" :loading="busy" @click="emit('confirm', { at, checkpoint })"
          >Cancel run</v-btn
        >
      </v-card-actions>
    </v-card>
  </v-dialog>
</template>
