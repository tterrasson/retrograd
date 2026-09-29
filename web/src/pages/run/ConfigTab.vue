<script setup lang="ts">
import { computed, ref } from 'vue'
import { mdiFileDocumentOutline } from '@mdi/js'
import { useConfigSchema } from '@/api/configSchema'
import { useDefaults } from '@/api/discovery'
import type { PlanSummary, Provenance } from '@/api/types'
import { toToml } from '@/utils/toml'
import { ParamsTable, PlanSummaryCard } from '@/features/plan'
import { CopyButton } from '@/features/shared'
import { useRunContext } from './context'

const run = useRunContext()
const schema = useConfigSchema()
const defaults = useDefaults()

const effective = computed(() => run.value?.effective_config ?? {})
const provenance = computed(() => (run.value?.provenance ?? {}) as Provenance)
const plan = computed(() => (run.value?.plan ?? null) as PlanSummary | null)
const hasPlan = computed(
  () => !!plan.value && typeof plan.value === 'object' && 'memory' in plan.value,
)
const algorithm = computed(() => run.value?.algorithm ?? null)

const tomlOpen = ref(false)
const toml = computed(() => (tomlOpen.value ? toToml(effective.value) : ''))
</script>

<template>
  <div v-if="run">
    <div class="d-flex mb-2">
      <v-spacer />
      <v-btn variant="tonal" :prepend-icon="mdiFileDocumentOutline" @click="tomlOpen = true"
        >Export as TOML</v-btn
      >
    </div>
    <v-row>
      <v-col cols="12" lg="7">
        <v-card class="pa-3">
          <ParamsTable
            :schema="schema.data.value ?? null"
            :effective="effective"
            :provenance="provenance"
            :defaults="defaults.data.value?.derived ?? []"
            :algorithm="algorithm"
            readonly
          />
        </v-card>
      </v-col>
      <v-col cols="12" lg="5">
        <PlanSummaryCard v-if="hasPlan && plan" :plan="plan" />
      </v-col>
    </v-row>
    <v-dialog v-model="tomlOpen" max-width="820" scrollable>
      <v-card>
        <v-card-title class="d-flex align-center">
          Effective configuration
          <v-spacer />
          <CopyButton :text="toml" label="Copy the TOML" />
        </v-card-title>
        <v-card-text>
          <pre class="text-pre rg-code">{{ toml }}</pre>
        </v-card-text>
        <v-card-actions>
          <span class="text-caption rg-muted px-2"
            >For reading and copying: null values have no TOML spelling and are left out.</span
          >
          <v-spacer />
          <v-btn variant="text" @click="tomlOpen = false">Close</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>
