import { describe, expect, it } from 'vitest'
import { commandsFor, statusInfo } from '@/api/runs'
import { RUN_STATUSES } from '@/api/types'

describe('run statuses', () => {
  it('labels every status the contract has', () => {
    for (const status of RUN_STATUSES) expect(statusInfo(status).label).toBe(status)
  })

  it('shows the commands a status calls for', () => {
    expect(commandsFor('running')).toEqual({
      pause: true,
      resume: false,
      cancel: true,
      checkpoint: true,
      evaluate: true,
      remove: false,
    })
    expect(commandsFor('paused')).toMatchObject({ pause: false, resume: true, cancel: true })
    expect(commandsFor('completed')).toMatchObject({
      cancel: false,
      remove: true,
      checkpoint: false,
    })
    expect(commandsFor('queued')).toMatchObject({ cancel: true, pause: false, remove: false })
  })
})
