// End-to-end tests against the production build, served by a stand-in server
// (`tests/e2e/mock-server.ts`) that replays recorded API bodies. Three
// instances: an open server, one that wants a token, and a viewer. Build first:
// `bun run build`.
import { defineConfig, devices } from '@playwright/test'

export const SERVERS = {
  open: 4174,
  token: 4175,
  viewer: 4176,
} as const

export const TOKEN = 'e2e-test-token'

const mock = (port: number, env: Record<string, string> = {}) => ({
  command: 'bun tests/e2e/mock-server.ts',
  url: `http://127.0.0.1:${port}/v1/health`,
  env: { MOCK_PORT: String(port), ...env },
  reuseExistingServer: !process.env.CI,
  timeout: 30_000,
})

export default defineConfig({
  testDir: 'tests/e2e',
  // The stand-in keeps a log of the streams it served; one worker keeps it
  // readable per test.
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${SERVERS.open}`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    mock(SERVERS.open),
    mock(SERVERS.token, { MOCK_TOKEN: TOKEN }),
    mock(SERVERS.viewer, { MOCK_MODE: 'viewer' }),
  ],
})
