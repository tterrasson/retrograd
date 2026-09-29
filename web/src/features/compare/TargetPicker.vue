<script setup lang="ts">
import { computed } from 'vue'
import { groupTargets, type ModelCard } from '@/api/openai'

const props = defineProps<{ models: ModelCard[]; max?: number; loading?: boolean }>()
const targets = defineModel<string[]>({ required: true })

const items = computed(() =>
  groupTargets(props.models).flatMap(({ run, targets: list }) => [
    { type: 'subheader' as const, title: run },
    ...list.map((model) => ({
      title: model.id,
      value: model.id,
      props: { subtitle: model.live ? 'served by a run that is training' : undefined },
    })),
  ]),
)
</script>

<template>
  <v-autocomplete
    v-model="targets"
    :items="items"
    :loading="loading"
    label="Targets"
    multiple
    chips
    closable-chips
    :counter="max"
    :rules="[(value: string[]) => !max || value.length <= max || `at most ${max}`]"
    hint="A run answers with its current adapter; @final, @base and @<checkpoint> pin one."
    persistent-hint
  />
</template>
