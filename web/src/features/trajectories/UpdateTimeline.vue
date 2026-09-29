<script setup lang="ts">
import { computed } from 'vue'
import type { EChartsOption, LineSeriesOption } from 'echarts'
import { segmentStarts, timelineSeries, type TimelineSeries } from '@/api/trajectories'
import type { UpdateSummary } from '@/api/types'
import { formatNumber } from '@/utils/format'
import { VChart, useChartColors } from '@/features/metrics'

const props = defineProps<{ updates: UpdateSummary[]; selected: number | null }>()
const emit = defineEmits<{ select: [number] }>()

const colors = useChartColors()
const series = computed(() => timelineSeries(props.updates))
const starts = computed(() => segmentStarts(props.updates))

function option(chart: TimelineSeries): EChartsOption {
  const c = colors.value
  const lines: LineSeriesOption[] = []
  const banded = chart.points.filter((point) => point.spread !== null)
  if (banded.length > 1) {
    lines.push(
      {
        type: 'line',
        data: banded.map((point) => [point.update, point.value - (point.spread ?? 0)]),
        stack: 'band',
        lineStyle: { opacity: 0 },
        symbol: 'none',
        silent: true,
        tooltip: { show: false },
      },
      {
        type: 'line',
        data: banded.map((point) => [point.update, 2 * (point.spread ?? 0)]),
        stack: 'band',
        lineStyle: { opacity: 0 },
        areaStyle: { color: c.primary, opacity: 0.12 },
        symbol: 'none',
        silent: true,
        tooltip: { show: false },
      },
    )
  }
  lines.push({
    type: 'line',
    name: chart.title,
    data: chart.points.map((point) => ({
      value: [point.update, point.value],
      symbolSize: point.update === props.selected ? 9 : 5,
      itemStyle: {
        color: point.update === props.selected ? c.secondary : point.texts ? c.primary : c.axis,
      },
    })),
    showSymbol: true,
    lineStyle: { width: 1.5, color: c.primary },
    markLine: starts.value.length
      ? {
          symbol: 'none',
          silent: true,
          label: { show: false },
          lineStyle: { color: c.warning, type: 'dashed' },
          data: starts.value.map((update) => ({ xAxis: update })),
        }
      : undefined,
  })
  return {
    animation: false,
    textStyle: { color: c.text, fontSize: 10 },
    grid: { left: 44, right: 8, top: 8, bottom: 20 },
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const item = Array.isArray(params) ? params[0] : params
        const value = (item?.value ?? []) as [number, number]
        return `update ${value[0]}: ${formatNumber(value[1], 4)}`
      },
    },
    xAxis: {
      type: 'value',
      min: 'dataMin',
      max: 'dataMax',
      axisLine: { lineStyle: { color: c.axis } },
      splitLine: { show: false },
      axisLabel: { fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      scale: true,
      splitLine: { lineStyle: { color: c.grid } },
      axisLabel: { fontSize: 10, formatter: (value: number) => formatNumber(value, 2) },
    },
    series: lines,
  }
}

function click(params: { value?: unknown; data?: unknown }) {
  const value = (params.value ?? (params.data as { value?: unknown } | undefined)?.value) as unknown
  if (Array.isArray(value) && typeof value[0] === 'number') emit('select', value[0])
}
</script>

<template>
  <div class="timeline">
    <figure v-for="chart in series" :key="chart.key" class="chart ma-0">
      <figcaption class="mono text-caption">{{ chart.title }}</figcaption>
      <VChart
        :option="option(chart)"
        autoresize
        style="height: 110px; width: 100%"
        role="img"
        :aria-label="`${chart.title} per update; click a point to open its update`"
        @click="click"
      />
    </figure>
    <div v-if="!series.length" class="text-body-2 rg-muted pa-2">No update summary yet.</div>
  </div>
</template>

<style scoped>
.timeline {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 8px;
}
</style>
