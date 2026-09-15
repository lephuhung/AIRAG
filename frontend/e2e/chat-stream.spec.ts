/**
 * Browser-boundary coverage for the public chat stream (Task 2).
 *
 * Every scenario runs against the REAL production bundle (`vite preview`)
 * with ALL network mocked at the browser boundary. Selectors use only
 * production DOM (message text, the existing
 * `data-testid="clarification-options"` banner, sonner toasts, the
 * `aria-label` stop button) — no production test hooks were added.
 *
 * Covered: streamed token → complete; clarification render + server-issued
 * resume selection; citation render after history reload; cancellation with
 * no stale retractable artifacts; public error render; unknown-event
 * tolerance.
 */
import { test, expect, type Page } from 'playwright/test';

const SESSION_ID = 'sess-e2e-1';

function sseEvent(type: string, data: unknown): string {
  return `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
}

function b64(obj: unknown): string {
  return Buffer.from(JSON.stringify(obj), 'utf8').toString('base64');
}

interface ApiMocks {
  history: unknown[];
  /** Static SSE body, or a responder keyed off the posted JSON envelope. */
  streamBody: string | ((postBody: unknown) => string);
}

async function installApiMocks(page: Page, mocks: ApiMocks): Promise<void> {
  const token = `h.${b64({ exp: 9_999_999_999 })}.s`;
  // Production localStorage keys (stores/authStore.ts). The token middle
  // segment is JWT-shaped with a far-future exp so the refresh flow never
  // triggers; isAuthenticated derives from token+user presence.
  await page.addInitScript(
    ({ t }: { t: string }) => {
      window.localStorage.setItem('auth_token', t);
      window.localStorage.setItem(
        'auth_user',
        JSON.stringify({ id: 'u-e2e', email: 'e2e@test.local', full_name: 'E2E' }),
      );
    },
    { t: token },
  );

  await page.route('**/api/v1/**', async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    if (url.includes(`/sessions/${SESSION_ID}/stream`) && method === 'POST') {
      const body =
        typeof mocks.streamBody === 'function'
          ? mocks.streamBody(route.request().postDataJSON())
          : mocks.streamBody;
      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body,
      });
      return;
    }
    if (url.includes('/stream/cancel')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
      return;
    }
    if (url.includes(`/sessions/${SESSION_ID}/history`)) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          session_id: SESSION_ID,
          messages: mocks.history,
          total: mocks.history.length,
        }),
      });
      return;
    }
    if (url.endsWith('/rag/chat/sessions')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{ id: SESSION_ID, title: 'E2E session' }]),
      });
      return;
    }
    if (url.includes('/workspaces')) {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([{ id: 'ws-e2e', name: 'Personal' }]),
      });
      return;
    }
    if (url.includes('documents')) {
      await route.fulfill({ status: 200, contentType: 'application/json', body: '[]' });
      return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
}

async function openChat(page: Page): Promise<void> {
  await page.goto(`/chat/${SESSION_ID}`);
  // Composer is production DOM (ChatInputArea textarea).
  await expect(page.locator('textarea').first()).toBeVisible({ timeout: 15_000 });
}

async function send(page: Page, text: string): Promise<void> {
  await page.locator('textarea').first().fill(text);
  await page.locator('textarea').first().press('Enter');
}

// --- scenarios -----------------------------------------------------------

test('streamed token followed by complete renders the answer', async ({ page }) => {
  await installApiMocks(page, {
    history: [],
    streamBody:
      sseEvent('status', { step: 'generating', detail: 'Streaming…' }) +
      sseEvent('token', { text: 'Chào bạn, ' }) +
      sseEvent('token', { text: 'đây là đáp án E2E.' }) +
      sseEvent('complete', { answer: 'Chào bạn, đây là đáp án E2E.' }),
  });
  await openChat(page);
  await send(page, 'xin chào');
  await expect(page.getByText('Chào bạn, đây là đáp án E2E.').first()).toBeVisible({ timeout: 15_000 });
});

test('clarification renders and submits the server-issued selection', async ({ page }) => {
  let postedSelection: unknown = null;
  // Resume turn: capture the posted envelope, then answer through it.
  await installApiMocks(page, {
    history: [],
    streamBody: (postBody: unknown) => {
      const body = postBody as Record<string, unknown>;
      postedSelection = body['clarification_selection'] ?? null;
      const asked = postedSelection !== null;
      return asked
        ? sseEvent('complete', { answer: 'Đã rõ, trả lời theo Thông tư B.' })
        : sseEvent('clarification_required', {
            clarification_id: 'clr-e2e',
            reason: 'semantic_ambiguity',
            question: 'Bạn muốn hỏi văn bản nào?',
            options: [
              { option_id: 'opt-a', label: 'Nghị định A' },
              { option_id: 'opt-b', label: 'Thông tư B' },
            ],
            resume: { thread_id: SESSION_ID },
          });
    },
  });
  await openChat(page);
  await send(page, 'văn bản này quy định gì?');
  const banner = page.getByTestId('clarification-options');
  await expect(banner).toBeVisible({ timeout: 15_000 });
  await expect(banner.getByText('Bạn muốn hỏi văn bản nào?')).toBeVisible();
  await banner.getByRole('button', { name: 'Thông tư B' }).click();
  await expect(page.getByText('Đã rõ, trả lời theo Thông tư B.').first()).toBeVisible({ timeout: 15_000 });
  // Only the server-issued triple is submitted — never fabricated identity.
  expect(postedSelection).toEqual({ clarification_id: 'clr-e2e', selected_option_id: 'opt-b' });
});

test('citation renders after history reload', async ({ page }) => {
  await installApiMocks(page, {
    history: [
      { id: 'u-1', role: 'user', content: 'Điều 5 nói gì?', timestamp: '2026-09-15T00:00:00Z' },
      {
        id: 'a-1',
        role: 'assistant',
        content: 'Điều 5 quy định về E2E [1].',
        sources: [],
        citations: [{ citation_id: 'c1', label: '[1] Điều 5', document_id: 'doc-e2e-1' }],
        timestamp: '2026-09-15T00:00:01Z',
      },
    ],
    streamBody: '',
  });
  await openChat(page);
  // Reloaded answer content renders…
  await expect(page.getByText('Điều 5 quy định về E2E [1].').first()).toBeVisible({ timeout: 15_000 });
  // …and the persisted citation resolves to a badge beyond the raw marker
  // (content marker + rendered citation badge share the handle text).
  expect(await page.getByText('[1]').count()).toBeGreaterThanOrEqual(2);
});

test('cancellation leaves no stale retractable artifacts', async ({ page }) => {
  await installApiMocks(page, {
    history: [],
    // Rollback clears the speculative token; the tail token streams after;
    // the connection then hangs until the test cancels it.
    streamBody:
      sseEvent('token', { text: 'nội dung suy đoán…' }) +
      sseEvent('token_rollback', {}) +
      sseEvent('token', { text: 'sau rollback.' }),
  });
  await openChat(page);
  await send(page, 'câu hỏi dài');
  await expect(page.getByText('sau rollback.').first()).toBeVisible({ timeout: 15_000 });
  // Pre-rollback speculation was retracted, not kept.
  await expect(page.getByText('nội dung suy đoán…')).toHaveCount(0);
  // Stop the hanging run via the production stop button (aria-label
  // `chat.cancel`, fallback "Stop"); cancel is quiet — no error banner.
  await page
    .getByRole('button', { name: /stop|cancel|dừng|hủy/i })
    .first()
    .click({ timeout: 15_000 });
  await expect(page.locator('textarea').first()).toBeEnabled({ timeout: 15_000 });
});

test('public error renders without crashing the panel', async ({ page }) => {
  await installApiMocks(page, {
    history: [],
    streamBody: sseEvent('error', { message: 'E2E boom (public)' }),
  });
  await openChat(page);
  await send(page, 'gây lỗi');
  await expect(page.getByText('E2E boom (public)').first()).toBeVisible({ timeout: 15_000 });
  // Composer stays usable after the error.
  await expect(page.locator('textarea').first()).toBeEnabled();
});

test('unknown future event types are tolerated', async ({ page }) => {
  await installApiMocks(page, {
    history: [],
    streamBody:
      sseEvent('frobnicator_v9', { anything: [1, 2, 3] }) +
      sseEvent('token', { text: 'vẫn ổn.' }) +
      sseEvent('complete', { answer: 'vẫn ổn.' }),
  });
  await openChat(page);
  await send(page, 'kiểm tra tương thích');
  await expect(page.getByText('vẫn ổn.').first()).toBeVisible({ timeout: 15_000 });
});
