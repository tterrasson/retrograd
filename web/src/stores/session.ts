import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { probeCapabilities } from '@/api/discovery'
import type { Capabilities } from '@/api/types'

const STORAGE_KEY = 'retrograd.token'

export type AuthMode = 'unknown' | 'open' | 'token'

function readStoredToken(): { token: string | null; remember: boolean } {
  try {
    const local = localStorage.getItem(STORAGE_KEY)
    if (local) return { token: local, remember: true }
    return { token: sessionStorage.getItem(STORAGE_KEY), remember: false }
  } catch {
    return { token: null, remember: false }
  }
}

/**
 * Who the client is to the server: whether a token is needed (found by asking
 * without one), the token itself, and the capabilities read on the way.
 */
export const useSessionStore = defineStore('session', () => {
  const stored = readStoredToken()
  const token = ref<string | null>(stored.token)
  const remember = ref(stored.remember)
  const mode = ref<AuthMode>('unknown')
  const capabilities = ref<Capabilities | null>(null)
  const unreachable = ref<string | null>(null)
  let started: Promise<void> | null = null

  const authenticated = computed(
    () =>
      mode.value === 'open' || (mode.value === 'token' && !!token.value && !!capabilities.value),
  )
  const viewer = computed(() => capabilities.value?.features.mode === 'viewer')

  function persist() {
    try {
      localStorage.removeItem(STORAGE_KEY)
      sessionStorage.removeItem(STORAGE_KEY)
      if (!token.value) return
      ;(remember.value ? localStorage : sessionStorage).setItem(STORAGE_KEY, token.value)
    } catch {
      // Storage refused (private mode): the token lives for this page only.
    }
  }

  /** Asks the server, once, whether it wants a token; checks a stored one. */
  function init(): Promise<void> {
    started ??= (async () => {
      try {
        const open = await probeCapabilities(null)
        if (open.kind === 'open') {
          mode.value = 'open'
          capabilities.value = open.capabilities
          return
        }
        mode.value = 'token'
        if (token.value) {
          const checked = await probeCapabilities(token.value)
          if (checked.kind === 'accepted') capabilities.value = checked.capabilities
          else forget()
        }
        unreachable.value = null
      } catch (error) {
        unreachable.value = error instanceof Error ? error.message : String(error)
        started = null
      }
    })()
    return started
  }

  /** Checks a typed token against the server before keeping it. */
  async function login(candidate: string, keep: boolean): Promise<boolean> {
    const result = await probeCapabilities(candidate)
    if (result.kind !== 'accepted') return false
    token.value = candidate
    remember.value = keep
    capabilities.value = result.capabilities
    mode.value = 'token'
    persist()
    return true
  }

  function forget() {
    token.value = null
    capabilities.value = mode.value === 'open' ? capabilities.value : null
    persist()
  }

  return {
    token,
    remember,
    mode,
    capabilities,
    unreachable,
    authenticated,
    viewer,
    init,
    login,
    forget,
  }
})
