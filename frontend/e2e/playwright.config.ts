import { defineConfig, devices } from 'playwright/test';

/**
 * Browser boundary suite for the public chat stream (Task 2).
 *
 * All network is mocked at the browser boundary (`page.route` on
 * `/api/v1/**`); no live backend is required or contacted. Auth is seeded
 * through the production localStorage keys (`auth_token`, `auth_user`) via
 * `addInitScript` — no production test hooks exist or are needed.
 *
 * Run: `npx playwright test --config e2e/playwright.config.ts`
 * (resolves the `playwright` library from the repo-root install; no
 * frontend dependency was added). Browsers are NOT vendored in this
 * environment — see the task report for the exact blocker.
 */
export default defineConfig({
  // NOTE: the Playwright specs use the `.e2e.ts` suffix (not `.spec.ts`)
  // so the Vitest runner (default include `**/*.{test,spec}.*`) never
  // collects them — the two runners stay disjoint without config coupling.
  testDir: '.',
  testMatch: '**/*.e2e.ts',
  fullyParallel: true,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    command: 'npm run preview -- --port 4173 --strictPort',
    port: 4173,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
