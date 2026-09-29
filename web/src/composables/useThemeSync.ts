import { onScopeDispose, watch } from 'vue'
import { useTheme } from 'vuetify'
import { THEMES } from '@/plugins/vuetify'
import { usePreferencesStore } from '@/stores/preferences'

/** Applies the theme preference, following the system when asked to. */
export function useThemeSync() {
  const theme = useTheme()
  const preferences = usePreferencesStore()
  const media = window.matchMedia?.('(prefers-color-scheme: dark)')
  const apply = () => {
    const dark =
      preferences.theme === 'dark' || (preferences.theme === 'system' && !!media?.matches)
    theme.global.name.value = dark ? THEMES.dark : THEMES.light
    document.documentElement.dataset.theme = dark ? 'dark' : 'light'
  }
  watch(() => preferences.theme, apply, { immediate: true })
  media?.addEventListener('change', apply)
  onScopeDispose(() => media?.removeEventListener('change', apply))
}
