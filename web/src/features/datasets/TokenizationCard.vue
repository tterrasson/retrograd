<script setup lang="ts">
import { computed } from 'vue'
import type { DatasetStats, DatasetTokenization } from '@/api/types'
import { formatNumber, formatPercent } from '@/utils/format'

const props = defineProps<{
  tokenization?: DatasetTokenization | null
  stats?: DatasetStats | null
}>()
const stats = computed(() => props.tokenization?.stats ?? props.stats ?? null)
const truncation = computed(() =>
  Object.entries(props.tokenization?.truncation ?? {}).sort((a, b) => Number(a[0]) - Number(b[0])),
)
</script>

<template>
  <div v-if="stats">
    <div class="text-caption rg-muted mb-1">
      <template v-if="tokenization"
        >tokenized with <span class="mono">{{ tokenization.tokenizer }}</span></template
      >
      <template v-else-if="!stats.measured">estimated, not measured with a tokenizer</template>
    </div>
    <v-table density="compact">
      <thead>
        <tr>
          <th>p50</th>
          <th>p90</th>
          <th>p99</th>
          <th>max</th>
          <th>total tokens</th>
        </tr>
      </thead>
      <tbody>
        <tr class="mono">
          <td>{{ formatNumber(stats.p50) }}</td>
          <td>{{ formatNumber(stats.p90) }}</td>
          <td>{{ formatNumber(stats.p99) }}</td>
          <td>{{ formatNumber(stats.max) }}</td>
          <td>{{ formatNumber(stats.total_tokens) }}</td>
        </tr>
      </tbody>
    </v-table>
    <template v-if="truncation.length">
      <div class="text-subtitle-2 mt-3">Truncated at each context</div>
      <v-table density="compact">
        <thead>
          <tr>
            <th>context</th>
            <th>examples truncated</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="[context, fraction] in truncation" :key="context">
            <td class="mono">{{ formatNumber(Number(context)) }}</td>
            <td class="mono">{{ formatPercent(fraction) }}</td>
          </tr>
        </tbody>
      </v-table>
    </template>
  </div>
</template>
