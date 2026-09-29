<script setup lang="ts">
import { computed } from 'vue'
import type { EChartsOption, LineSeriesOption } from 'echarts'
import type { Marker, MetricSeries } from '@/api/events'
import type { MetricAxis } from '@/stores/preferences'
import { ema } from '@/utils/ema'
import { formatNumber } from '@/utils/format'
import { VChart } from './echarts'
import { useChartColors } from './useChartColors'

const props = withDefaults(
  defineProps<{
    series: MetricSeries
    revision: number
    names: string[]
    axis: MetricAxis
    smoothing: number
    markers?: Marker[]
    height?: string
    title?: string
  }>(),
  { markers: () => [], height: '240px', title: undefined },
)

const colors = useChartColors()

const option = computed<EChartsOption>(() => {
  void props.revision
  const x = props.axis === 'iteration' ? props.series.iteration : props.series.globalStep
  const palette = colors.value.palette
  const lines = props.names.flatMap((name, index): LineSeriesOption[] => {
    const column = props.series.values.get(name)
    if (!column) return []
    const color = palette[index % palette.length]
    const raw: [number, number | null][] = column.map((value, i) => [x[i] ?? i, value])
    const present = raw.filter((point) => point[1] !== null)
    const out: LineSeriesOption[] = []
    if (props.smoothing > 0) {
      const smoothed = ema(
        present.map((point) => point[1]),
        props.smoothing,
      )
      out.push({
        type: 'line',
        name: `${name} (raw)`,
        data: present,
        showSymbol: false,
        sampling: 'lttb',
        lineStyle: { width: 1, opacity: 0.25, color },
        itemStyle: { color },
        emphasis: { disabled: true },
        silent: true,
        tooltip: { show: false },
      })
      out.push({
        type: 'line',
        name,
        data: present.map((point, i) => [point[0], smoothed[i] ?? null]),
        showSymbol: false,
        sampling: 'lttb',
        lineStyle: { width: 2, color },
        itemStyle: { color },
      })
    } else {
      out.push({
        type: 'line',
        name,
        data: present,
        showSymbol: false,
        sampling: 'lttb',
        lineStyle: { width: 2, color },
        itemStyle: { color },
      })
    }
    return out
  })
  const marks = props.markers.map((marker) => ({
    xAxis: props.axis === 'iteration' ? marker.iteration : marker.globalStep,
    name: marker.label,
    lineStyle: {
      color: marker.kind === 'checkpoint' ? colors.value.secondary : colors.value.warning,
      type: marker.kind === 'checkpoint' ? ('dashed' as const) : ('dotted' as const),
    },
    label: { show: false },
  }))
  const first = lines[0]
  if (first && marks.length) {
    Object.assign(first, {
      markLine: { symbol: 'none', silent: false, data: marks, animation: false },
    })
  }
  return {
    animation: false,
    textStyle: { color: colors.value.text },
    grid: { left: 56, right: 16, top: 16, bottom: 56 },
    tooltip: {
      trigger: 'axis',
      valueFormatter: (value) => formatNumber(typeof value === 'number' ? value : Number(value), 5),
    },
    xAxis: {
      type: 'value',
      name: props.axis === 'iteration' ? 'iteration' : 'step',
      nameLocation: 'middle',
      nameGap: 24,
      axisLine: { lineStyle: { color: colors.value.axis } },
      splitLine: { lineStyle: { color: colors.value.grid } },
      min: 'dataMin',
      max: 'dataMax',
    },
    yAxis: {
      type: 'value',
      scale: true,
      axisLine: { lineStyle: { color: colors.value.axis } },
      splitLine: { lineStyle: { color: colors.value.grid } },
      axisLabel: { formatter: (value: number) => formatNumber(value, 3) },
    },
    dataZoom: [
      { type: 'inside', filterMode: 'none' },
      { type: 'slider', height: 16, bottom: 8, filterMode: 'none' },
    ],
    series: lines,
  }
})
</script>

<template>
  <figure class="ma-0">
    <figcaption v-if="title" class="text-body-2 font-weight-medium mono mb-1">
      {{ title }}
    </figcaption>
    <VChart
      :option="option"
      :style="{ height, width: '100%' }"
      autoresize
      :aria-label="`chart of ${names.join(', ')}`"
      role="img"
    />
  </figure>
</template>
