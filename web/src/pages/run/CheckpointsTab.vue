<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { mdiContentSaveOutline } from '@mdi/js'
import { commandsFor, useCheckpoints, useRunCommands } from '@/api/runs'
import type { CheckpointEntry } from '@/api/types'
import { useDraftStore } from '@/stores/draft'
import { useNotify } from '@/composables/useNotify'
import { CheckpointTable } from '@/features/checkpoints'
import { ProblemAlert } from '@/features/shared'
import { useRunContext, useRunEventsContext } from './context'

const run = useRunContext()
const events = useRunEventsContext()
const router = useRouter()
const draft = useDraftStore()
const notify = useNotify()
const id = computed(() => run.value?.id ?? '')
const listing = useCheckpoints(id)
const commands = useRunCommands(id)
const status = computed(() => events.value?.status.value ?? run.value?.status ?? null)

async function checkpointNow() {
  try {
    await commands.checkpoint.mutateAsync()
    notify.success('Checkpoint requested')
  } catch (error) {
    notify.error(error, 'Checkpoint')
  }
}

function fork(entry: CheckpointEntry) {
  if (!run.value) return
  draft.fork(run.value.id, run.value.name ?? '', entry.id)
  void router.push('/runs/new')
}

function compare(entry: CheckpointEntry) {
  void router.push({
    path: '/compare',
    query: { run: id.value, targets: `${id.value}@${entry.id},${id.value}@base` },
  })
}
</script>

<template>
  <div>
    <div class="d-flex align-center mb-2 ga-2">
      <span v-if="listing.data.value?.directory" class="mono text-caption rg-muted">{{
        listing.data.value.directory
      }}</span>
      <v-spacer />
      <v-btn
        v-if="commandsFor(status).checkpoint"
        color="primary"
        variant="tonal"
        :prepend-icon="mdiContentSaveOutline"
        :loading="commands.checkpoint.isPending.value"
        @click="checkpointNow"
      >
        Checkpoint now
      </v-btn>
    </div>
    <ProblemAlert :error="listing.error.value" />
    <v-card>
      <CheckpointTable
        :checkpoints="listing.data.value?.checkpoints ?? []"
        :latest="listing.data.value?.latest"
        :loading="listing.isFetching.value"
        @fork="fork"
        @compare="compare"
      />
    </v-card>
  </div>
</template>
