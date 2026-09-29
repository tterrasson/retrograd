import { expect, test } from '@playwright/test'
import { SERVERS, TOKEN } from '../../playwright.config'

const token = `http://127.0.0.1:${SERVERS.token}`
const viewer = `http://127.0.0.1:${SERVERS.viewer}`

test('a server that wants a token asks for it, refuses a wrong one, keeps a right one', async ({
  page,
}) => {
  await page.goto(`${token}/runs`)
  await expect(page).toHaveURL(/\/login\?next=/)
  await page.getByLabel('Token').fill('wrong')
  await page.getByRole('button', { name: 'Connect' }).click()
  await expect(page.getByText('The server refused this token.')).toBeVisible()

  await page.getByLabel('Token').fill(TOKEN)
  await page.getByRole('button', { name: 'Connect' }).click()
  await expect(page).toHaveURL(/\/runs$/)
  await expect(page.getByText('search-agent')).toBeVisible()
})

test('a viewer serves one run and no navigation', async ({ page }) => {
  await page.goto(`${viewer}/`)
  await expect(page).toHaveURL(/\/runs\/local\/trajectories/)
  await expect(page.getByText('3 updates')).toBeVisible()
  await expect(page.getByRole('navigation', { name: 'Main navigation' })).toHaveCount(0)
  await page.goto(`${viewer}/runs`)
  await expect(page).toHaveURL(/\/runs\/local\/trajectories/)
})
