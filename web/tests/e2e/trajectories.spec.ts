import { expect, test } from '@playwright/test'
import { AGENT_RUN } from '../fixtures/api'

const URL = `/runs/${AGENT_RUN}/trajectories`

test('the latest update opens, and j/k move between updates', async ({ page }) => {
  await page.goto(URL)
  await expect(page.getByText('3 updates')).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Update 3' })).toBeVisible()

  await page.keyboard.press('k')
  await expect(page.getByRole('heading', { name: 'Update 2' })).toBeVisible()
  await page.keyboard.press('k')
  await expect(page.getByRole('heading', { name: 'Update 1' })).toBeVisible()
  await expect(page.getByText('No rollout was exported for this update')).toBeVisible()
  await page.keyboard.press('j')
  await expect(page.getByRole('heading', { name: 'Update 2' })).toBeVisible()
})

test('a group without signal says so, and a trajectory shows its tool calls', async ({ page }) => {
  await page.goto(`${URL}?update=2`)
  await expect(page.getByRole('heading', { name: 'Update 2' })).toBeVisible()
  await expect(page.getByText('zero_signal').first()).toBeVisible()

  await page.getByRole('button', { name: /group 0/ }).click()
  await expect(page.getByText('#1')).toBeVisible()
  await expect(page.getByText('1 tool errors')).toBeVisible()
  await page.getByRole('button', { name: 'Show the conversation' }).nth(1).click()
  await expect(page.getByText('search').first()).toBeVisible()
  await expect(page.getByText(/Final answer from member 1/)).toBeVisible()
})

test('a run without an export says it has none', async ({ page }) => {
  await page.goto('/runs/3a7c9e10-5b2d-4c8f-9e61-0d4b2a1f7c35/trajectories')
  await expect(page.getByText('No trajectories')).toBeVisible()
})
