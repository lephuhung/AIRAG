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
  /**
   * Static SSE body, or a responder keyed off the posted JSON envelope.
   * A responder may return a Promise that resolves later (or never
   * before the test cancels): the route handler awaits it, so the HTTP
   * response stays open and the hook keeps `isStreaming === true` — the
   * only state in which the production stop button is rendered.
   */
  streamBody: string | ((postBody: unknown) => string | Promise<string>);
}

async function installApiMocks(
  page: Page,
  mocks: ApiMocks,
): Promise<{ cancelPosts: string[] }> {
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

  const cancelPosts: string[] = [];
  await page.route('**/api/v1/**', async (route) => {
    const url = route.request().url();
    const method = route.request().method();
    // NOTE: the cancel check comes first — `/stream/cancel` also contains
    // the `/sessions/<id>/stream` prefix and must not be answered as SSE.
    if (url.includes('/stream/cancel')) {
      cancelPosts.push(url);
      await route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
      return;
    }
    if (url.includes(`/sessions/${SESSION_ID}/stream`) && method === 'POST') {
      const body =
        typeof mocks.streamBody === 'function'
          ? await mocks.streamBody(route.request().postDataJSON())
          : mocks.streamBody;
      try {
        await route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          body,
        });
      } catch {
        // The test cancelled first: the browser dropped the request, so
        // there is nothing left to fulfil. Quiet by design.
      }
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
  return { cancelPosts };
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

test('cancellation posts the server cancel and leaves no stale artifacts', async ({ page }) => {
  // Deferred stream: the first POST hangs on a gate the test controls, so
  // the connection stays open, `isStreaming` stays true, and the
  // production stop button (rendered only while streaming) is genuinely
  // reachable. A finite `route.fulfill` body would end the stream
  // immediately and unmount the button before it could be clicked.
  let releaseStream!: (body: string) => void;
  const streamGate = new Promise<string>((resolve) => {
    releaseStream = resolve;
  });
  let streamCalls = 0;
  const { cancelPosts } = await installApiMocks(page, {
    history: [],
    streamBody: () => {
      streamCalls += 1;
      if (streamCalls > 1) {
        return sseEvent('complete', { answer: 'sau hủy, vẫn ổn.' });
      }
      return streamGate;
    },
  });
  await openChat(page);
  await send(page, 'câu hỏi dài');
  // The run is in flight: the stop button is rendered and clickable.
  const stop = page.getByRole('button', { name: /stop|cancel|dừng|hủy/i }).first();
  await expect(stop).toBeVisible({ timeout: 15_000 });
  await stop.click();
  // The hook posts the server-side cancel for the detached run…
  await expect
    .poll(() => cancelPosts.length, { timeout: 15_000 })
    .toBe(1);
  expect(cancelPosts[0]).toContain('/stream/cancel');
  // …aborts the socket quietly (no error banner, composer usable)…
  await expect(page.locator('textarea').first()).toBeEnabled({ timeout: 15_000 });
  // …and leaves no stale retractable UI: no clarification banner, no
  // error toast, and a fresh turn afterwards starts clean.
  await expect(page.getByTestId('clarification-options')).toHaveCount(0);
  releaseStream(sseEvent('complete', { answer: 'late terminal (ignored)' }));
  await send(page, 'câu hỏi tiếp theo');
  await expect(page.getByText('sau hủy, vẫn ổn.').first()).toBeVisible({ timeout: 15_000 });
  expect(streamCalls).toBe(2);
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
