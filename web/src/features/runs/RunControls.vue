<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { mdiContentSaveOutline, mdiDeleteOutline, mdiPause, mdiPlay, mdiStop } from '@mdi/js'
import { commandsFor, useRunCommands } from '@/api/runs'
import type { CancelRequest, RunStatus } from '@/api/types'
import { useNotify } from '@/composables/useNotify'
import CancelDialog from './CancelDialog.vue'

const props = withDefaults(
  defineProps<{ id: string; status: RunStatus | null | undefined; compact?: boolean }>(),
  { compact: false },
)
const emit = defineEmits<{ deleted: [] }>()

const router = useRouter()
const notify = useNotify()
const commands = useRunCommands(() => props.id)
// Which buttons show. A button shown is still only a request: the server decides.
const shown = computed(() => commandsFor(props.status))
const cancelOpen = ref(false)
const deleteOpen = ref(false)

async function run<T>(label: string, action: () => Promise<T>) {
  try {
    await action()
    notify.success(label)
  } catch (error) {
    notify.error(error, label)
  }
}

const pause = () => run('Pause requested', () => commands.pause.mutateAsync())
const resume = () => run('Resume requested', () => commands.resume.mutateAsync())
const checkpoint = () => run('Checkpoint requested', () => commands.checkpoint.mutateAsync())
async function cancel(body: CancelRequest) {
  await run('Cancel requested', () => commands.cancel.mutateAsync(body))
  cancelOpen.value = false
}
async function remove() {
  try {
    await commands.remove.mutateAsync()
    notify.success('Run deleted')
    deleteOpen.value = false
    emit('deleted')
    if (router.currentRoute.value.params.id === props.id) void router.push('/runs')
  } catch (error) {
    notify.error(error, 'Delete')
  }
}
const size = computed(() => (props.compact ? 'small' : 'default'))
</script>

<template>
  <div class="d-flex ga-1 align-center flex-wrap" role="group" aria-label="Run controls">
    <v-btn
      v-if="shown.pause"
      :size="size"
      variant="tonal"
      :prepend-icon="compact ? undefined : mdiPause"
      :icon="compact ? mdiPause : undefined"
      :loading="commands.pause.isPending.value"
      aria-label="Pause"
      title="Pause"
      @click.stop="pause"
      >{{ compact ? '' : 'Pause' }}</v-btn
    >
    <v-btn
      v-if="shown.resume"
      :size="size"
      variant="tonal"
      color="primary"
      :prepend-icon="compact ? undefined : mdiPlay"
      :icon="compact ? mdiPlay : undefined"
      :loading="commands.resume.isPending.value"
      aria-label="Resume"
      title="Resume"
      @click.stop="resume"
      >{{ compact ? '' : 'Resume' }}</v-btn
    >
    <v-btn
      v-if="shown.checkpoint"
      :size="size"
      variant="tonal"
      :prepend-icon="compact ? undefined : mdiContentSaveOutline"
      :icon="compact ? mdiContentSaveOutline : undefined"
      :loading="commands.checkpoint.isPending.value"
      aria-label="Checkpoint now"
      title="Checkpoint now"
      @click.stop="checkpoint"
      >{{ compact ? '' : 'Checkpoint' }}</v-btn
    >
    <v-btn
      v-if="shown.cancel"
      :size="size"
      variant="tonal"
      color="error"
      :prepend-icon="compact ? undefined : mdiStop"
      :icon="compact ? mdiStop : undefined"
      aria-label="Cancel"
      title="Cancel"
      @click.stop="cancelOpen = true"
      >{{ compact ? '' : 'Cancel' }}</v-btn
    >
    <v-btn
      v-if="shown.remove"
      :size="size"
      variant="text"
      color="error"
      :prepend-icon="compact ? undefined : mdiDeleteOutline"
      :icon="compact ? mdiDeleteOutline : undefined"
      aria-label="Delete"
      title="Delete"
      @click.stop="deleteOpen = true"
      >{{ compact ? '' : 'Delete' }}</v-btn
    >
    <CancelDialog v-model="cancelOpen" :busy="commands.cancel.isPending.value" @confirm="cancel" />
    <v-dialog v-model="deleteOpen" max-width="440">
      <v-card>
        <v-card-title>Delete this run?</v-card-title>
        <v-card-text>
          Its record and every artifact it wrote under the server's run directory - adapter,
          checkpoints, logs, trajectories - are removed. This cannot be undone.
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" @click="deleteOpen = false">Keep</v-btn>
          <v-btn color="error" :loading="commands.remove.isPending.value" @click="remove"
            >Delete</v-btn
          >
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
