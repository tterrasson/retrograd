import { describe, expect, it } from 'vitest'
import { RunControls } from '@/features/runs'
import { mountWithPlugins } from './mount'

function buttons(status: 'running' | 'paused' | 'completed' | 'queued') {
  const wrapper = mountWithPlugins(RunControls, { props: { id: 'run-1', status } })
  return ['Pause', 'Resume', 'Checkpoint now', 'Cancel', 'Delete'].filter((label) =>
    wrapper.find(`button[aria-label="${label}"]`).exists(),
  )
}

describe('RunControls', () => {
  // Shown by status, never refused here: pressing one is a request the server decides.
  it('shows the commands of a running run', () => {
    expect(buttons('running')).toEqual(['Pause', 'Checkpoint now', 'Cancel'])
  })

  it('shows resume for a paused run', () => {
    expect(buttons('paused')).toEqual(['Resume', 'Checkpoint now', 'Cancel'])
  })

  it('shows only delete for a finished run', () => {
    expect(buttons('completed')).toEqual(['Delete'])
  })

  it('lets a queued run be cancelled', () => {
    expect(buttons('queued')).toEqual(['Cancel'])
  })
})
