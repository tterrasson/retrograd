import { computed } from 'vue'
import { useTheme } from 'vuetify'

/** Chart colors taken from the active theme, so both themes keep their contrast. */
export function useChartColors() {
  const theme = useTheme()
  return computed(() => {
    const colors = theme.current.value.colors
    const dark = theme.current.value.dark
    return {
      text: dark ? '#cdc2dc' : '#4d4159',
      axis: dark ? '#524369' : '#bda9d6',
      grid: dark ? 'rgba(160,140,200,0.14)' : 'rgba(122,90,160,0.12)',
      primary: colors.primary ?? '#7a2fd4',
      secondary: colors.secondary ?? '#a8126f',
      warning: colors.warning ?? '#8a560f',
      error: colors.error ?? '#b4234a',
      background: colors.surface ?? '#ffffff',
      palette: dark
        ? ['#d4a4f4', '#f78fb0', '#f9c676', '#7fd4a6', '#8fc4ff', '#ffab91']
        : ['#7a2fd4', '#a8126f', '#986014', '#1f7a4d', '#1c5fa8', '#b4234a'],
    }
  })
}
