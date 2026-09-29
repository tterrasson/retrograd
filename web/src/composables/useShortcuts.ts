import { onScopeDispose } from 'vue'

export type ShortcutMap = Record<string, (event: KeyboardEvent) => void>

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
}

/**
 * Single-key shortcuts while the calling component is mounted. Ignored while
 * the focus is in a field, and with a modifier held.
 */
export function useShortcuts(map: ShortcutMap) {
  const handler = (event: KeyboardEvent) => {
    if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return
    if (isTyping(event.target)) return
    const action = map[event.key]
    if (!action) return
    event.preventDefault()
    action(event)
  }
  window.addEventListener('keydown', handler)
  onScopeDispose(() => window.removeEventListener('keydown', handler))
}
