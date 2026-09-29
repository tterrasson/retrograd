import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

const STORAGE_KEY = 'retrograd.preferences'

export type ThemePreference = 'system' | 'light' | 'dark'
export type MetricAxis = 'global_step' | 'iteration'

interface Stored {
  theme: ThemePreference
  smoothing: number
  axis: MetricAxis
  followJournal: boolean
  compactTrajectories: boolean
}

const DEFAULTS: Stored = {
  theme: 'system',
  smoothing: 0.6,
  axis: 'global_step',
  followJournal: true,
  compactTrajectories: false,
}

function load(): Stored {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? { ...DEFAULTS, ...(JSON.parse(raw) as Partial<Stored>) } : { ...DEFAULTS }
  } catch {
    return { ...DEFAULTS }
  }
}

/** What this browser prefers: theme, curve smoothing, axis. Never sent to the server. */
export const usePreferencesStore = defineStore('preferences', () => {
  const initial = load()
  const theme = ref<ThemePreference>(initial.theme)
  const smoothing = ref(initial.smoothing)
  const axis = ref<MetricAxis>(initial.axis)
  const followJournal = ref(initial.followJournal)
  const compactTrajectories = ref(initial.compactTrajectories)

  watch(
    [theme, smoothing, axis, followJournal, compactTrajectories],
    () => {
      try {
        const value: Stored = {
          theme: theme.value,
          smoothing: smoothing.value,
          axis: axis.value,
          followJournal: followJournal.value,
          compactTrajectories: compactTrajectories.value,
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify(value))
      } catch {
        // Not persisted: the choice lasts for this page.
      }
    },
    { deep: true },
  )

  return { theme, smoothing, axis, followJournal, compactTrajectories }
})
