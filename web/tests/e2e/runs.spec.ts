import { expect, test } from '@playwright/test'
import { AGENT_RUN, agentEvents, journalLines, SFT_RUN } from '../fixtures/api'

test('the run list shows every run with what it trains', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveURL(/\/runs$/)
  await expect(page.getByText('search-agent')).toBeVisible()
  await expect(page.getByText('chat-sft')).toBeVisible()
  await expect(page.getByText('agent_grpo').first()).toBeVisible()
})

test('a run filtered out of the URL stays filtered', async ({ page }) => {
  await page.goto('/runs?algorithm=sft')
  await expect(page.getByText('chat-sft')).toBeVisible()
  await expect(page.getByText('search-agent')).toHaveCount(0)
})

test('a cut event stream resumes without a gap or a duplicate', async ({ page, request }) => {
  const before = ((await (await request.get('/__mock/streams')).json()) as unknown[]).length
  await page.goto(`/runs/${AGENT_RUN}/journal`)

  const lines = journalLines(agentEvents)
  await expect(page.getByText(`${lines} lines`)).toBeVisible({ timeout: 15_000 })
  // Every log line exactly once, in order.
  for (const update of [1, 2, 3]) {
    await expect(page.getByText(`update ${update} trained`, { exact: true })).toHaveCount(1)
  }

  const streams = (await (await request.get('/__mock/streams')).json()) as {
    run: string
    lastEventId: string | null
  }[]
  const mine = streams.slice(before).filter((stream) => stream.run === AGENT_RUN)
  expect(mine[0]?.lastEventId).toBeNull()
  const resumed = mine.find((stream) => stream.lastEventId !== null)
  expect(Number(resumed?.lastEventId)).toBe(Math.ceil(agentEvents.length / 2))
})

test('a finished run shows its configuration and where each value came from', async ({ page }) => {
  await page.goto(`/runs/${SFT_RUN}/config`)
  await expect(page.getByText('training.lr').first()).toBeVisible()
  await expect(page.getByText('derived').first()).toBeVisible()
})

test('an unknown path of the application is the application', async ({ page }) => {
  await page.goto('/no/such/page')
  await expect(page.getByText(/not found/i).first()).toBeVisible()
})
