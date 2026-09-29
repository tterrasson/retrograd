import { describe, expect, it } from 'vitest'
import { nextTick } from 'vue'
import { ParamsTable } from '@/features/plan'
import type { JsonSchema } from '@/api/configSchema'
import type { Provenance } from '@/api/types'
import { mountWithPlugins } from './mount'

const schema: JsonSchema = {
  type: 'object',
  properties: {
    training: {
      type: 'object',
      properties: {
        lr: { type: 'number', minimum: 0, description: 'Learning rate.' },
        packing: { type: 'boolean' },
      },
    },
  },
}
const effective = { training: { lr: 0.0002, packing: true } }
const provenance: Provenance = { 'training.lr': { source: 'derived', reason: 'from the rank' } }

async function mountTable(props: Record<string, unknown> = {}) {
  const wrapper = mountWithPlugins(ParamsTable, {
    props: { schema, effective, provenance, params: {}, ...props },
  })
  // A search opens every section that matches.
  await wrapper.find('input[type="text"]').setValue('lr')
  await nextTick()
  return wrapper
}

describe('ParamsTable', () => {
  it('shows the resolved value and where it came from', async () => {
    const wrapper = await mountTable()
    expect(wrapper.text()).toContain('0.0002')
    expect(wrapper.text()).toContain('derived')
  })

  it('pins an edited value into params', async () => {
    const wrapper = await mountTable()
    const input = wrapper.find('input[type="number"]')
    await input.setValue('0.001')
    await input.trigger('change')
    const updates = wrapper.emitted('update:params') ?? []
    expect(updates[updates.length - 1]).toEqual([{ training: { lr: 0.001 } }])
  })

  it('gives a pinned value back to the server', async () => {
    const wrapper = await mountTable({ params: { training: { lr: 0.001 } } })
    expect(wrapper.text()).toContain('override')
    await wrapper.find('button[aria-label="Give training.lr back to the server"]').trigger('click')
    const updates = wrapper.emitted('update:params') ?? []
    expect(updates[updates.length - 1]).toEqual([{}])
  })

  it('shows a refusal on the row it names', async () => {
    const wrapper = await mountTable({ errors: { 'training.lr': ['must be positive'] } })
    expect(wrapper.text()).toContain('must be positive')
    expect(wrapper.find('tr.row-error').exists()).toBe(true)
  })

  it('edits nothing in read-only mode', async () => {
    const wrapper = mountWithPlugins(ParamsTable, {
      props: { schema, effective, provenance, readonly: true },
    })
    await wrapper.find('input[type="text"]').setValue('lr')
    await nextTick()
    expect(wrapper.find('input[type="number"]').exists()).toBe(false)
    expect(wrapper.text()).toContain('0.0002')
  })
})
