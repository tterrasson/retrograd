<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRunCommands } from '@/api/runs'
import type { PatchRequest } from '@/api/types'
import { messagesAt } from '@/composables/useProblemFields'
import { useNotify } from '@/composables/useNotify'
import { ProblemAlert } from '@/features/shared'

// Exactly the fields `PATCH /v1/runs/{id}` accepts.
const props = defineProps<{ id: string }>()
const commands = useRunCommands(() => props.id)
const notify = useNotify()

const lr = ref('')
const everyIterations = ref('')
const patience = ref('')
const everySteps = ref('')
const mode = ref('')
const failure = ref<unknown>(null)

function numberOrNull(text: string): number | null {
  const trimmed = text.trim()
  return trimmed === '' ? null : Number(trimmed)
}

// Absent and null mean the same to the server: "leave it as it is".
const body = computed<PatchRequest>(() => {
  const lrValue = numberOrNull(lr.value)
  const every = numberOrNull(everyIterations.value)
  const wait = numberOrNull(patience.value)
  const steps = numberOrNull(everySteps.value)
  const checkpointMode = mode.value.trim() || null
  return {
    training: lrValue !== null ? { lr: lrValue } : null,
    evaluation:
      every !== null || wait !== null ? { every_iterations: every, patience: wait } : null,
    checkpoint:
      steps !== null || checkpointMode ? { every_steps: steps, mode: checkpointMode } : null,
  }
})
const empty = computed(
  () => !body.value.training && !body.value.evaluation && !body.value.checkpoint,
)
const numeric = [
  (value: string) => value.trim() === '' || Number.isFinite(Number(value)) || 'a number',
]

async function submit() {
  failure.value = null
  try {
    const accepted = await commands.patch.mutateAsync(body.value)
    notify.success(
      accepted.applies_at_iteration !== null && accepted.applies_at_iteration !== undefined
        ? `Applies at iteration ${accepted.applies_at_iteration}`
        : 'Change accepted',
    )
    lr.value = everyIterations.value = patience.value = everySteps.value = mode.value = ''
  } catch (error) {
    failure.value = error
  }
}
</script>

<template>
  <v-card>
    <v-card-title class="text-subtitle-1">Adjust while running</v-card-title>
    <v-card-text>
      <form @submit.prevent="submit">
        <v-row dense>
          <v-col cols="12" sm="4">
            <v-text-field
              v-model="lr"
              label="training.lr"
              inputmode="decimal"
              :rules="numeric"
              :error-messages="messagesAt(failure, '/training/lr')"
            />
          </v-col>
          <v-col cols="6" sm="4">
            <v-text-field
              v-model="everyIterations"
              label="evaluation.every_iterations"
              inputmode="numeric"
              :rules="numeric"
              :error-messages="messagesAt(failure, '/evaluation/every_iterations')"
            />
          </v-col>
          <v-col cols="6" sm="4">
            <v-text-field
              v-model="patience"
              label="evaluation.patience"
              inputmode="numeric"
              :rules="numeric"
              :error-messages="messagesAt(failure, '/evaluation/patience')"
            />
          </v-col>
          <v-col cols="6" sm="4">
            <v-text-field
              v-model="everySteps"
              label="checkpoint.every_steps"
              inputmode="numeric"
              :rules="numeric"
              :error-messages="messagesAt(failure, '/checkpoint/every_steps')"
            />
          </v-col>
          <v-col cols="6" sm="4">
            <v-text-field
              v-model="mode"
              label="checkpoint.mode"
              :error-messages="messagesAt(failure, '/checkpoint/mode')"
            />
          </v-col>
          <v-col cols="12" sm="4" class="d-flex align-center">
            <v-btn
              type="submit"
              color="primary"
              :disabled="empty"
              :loading="commands.patch.isPending.value"
            >
              Apply
            </v-btn>
          </v-col>
        </v-row>
        <ProblemAlert :error="failure" />
      </form>
    </v-card-text>
  </v-card>
</template>
