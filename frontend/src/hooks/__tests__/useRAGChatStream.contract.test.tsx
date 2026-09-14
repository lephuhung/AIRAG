/**
 * Phase 4D (Task 9) — versioned public chat/SSE contract hook tests.
 *
 * Covers the frontend presentation contract owned by
 * backend/app/services/agents/v2/transport.py ("v2.chat/1"):
 *  1. clarification_required with real `event:` + `data:` framing
 *     (not data-only) → pendingClarification + clarifying status.
 *  2. Legacy v1 `clarification` event normalizes to structured options.
 *  3. submitClarification resolves server-issued option ids, rejects
 *     fabricated document/workspace/binding identity.
 *  4. Fragmented SSE frames split mid-line still parse.
 *  5. Data-only frames are ignored (never misattributed to a stale type).
 *  6. Unknown future event types never crash the stream.
 *  7. citation events land in pendingCitations + final message (reload metadata).
 *  8. cancelled terminates quietly; complete carries clarification resume block.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useRAGChatStream } from '../useRAGChatStream';
import { useAuthStore } from '@/stores/authStore';
import type { ChatMessage } from '@/types';

function mockFetchFrames(frames: string[]) {
  const encoder = new TextEncoder();
  const chunks = frames.map((f) => encoder.encode(f));
  let readIndex = 0;
  (global.fetch as any) = vi.fn(() =>
    Promise.resolve({
      ok: true,
      body: {
        getReader: () => ({
          read: async () => {
            if (readIndex >= chunks.length) {
              return { done: true, value: undefined };
            }
            return { done: false, value: chunks[readIndex++] };
          },
          cancel: async () => {},
        }),
      },
    }),
  );
}

function renderStreamHook(sessionId = 'test-session-t9') {
  const queryClient = new QueryClient();
  const utils = renderHook(() => useRAGChatStream(sessionId), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
  return { ...utils, queryClient };
}

describe('public chat contract (Task 9)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.setState({
      token: 'test-token',
      user: { id: 'user-1', email: 'test@test.com' } as any,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('handles clarification_required with event+data framing and resolves issued options', async () => {
    mockFetchFrames([
      'event: status\ndata: {"step":"analyzing","detail":"Running v2 graph..."}\n\n',
      'event: clarification_required\ndata: {"clarification_id":"clr-1","question":"Which document?","options":[{"option_id":"opt-1","label":"Doc A"},{"option_id":"opt-2","label":"Doc B"}],"resume":{"thread_id":"thread-1"}}\n\n',
      'event: complete\ndata: {"answer":"Which document?","clarification":{"clarification_id":"clr-1","options":[{"option_id":"opt-1","label":"Doc A"}],"resume":{"thread_id":"thread-1"}}}\n\n',
    ]);
    const { result } = renderStreamHook();

    let final: ChatMessage | null = null;
    await act(async () => {
      final = await result.current.sendMessage('ambiguous query', [], false);
    });

    await waitFor(() => {
      expect(result.current.pendingClarification?.clarification_id).toBe('clr-1');
    });
    expect(result.current.pendingClarification?.options).toHaveLength(2);
    expect(result.current.status).toBe('idle');
    // Server-issued option resolves to its label for the resume reply.
    expect(result.current.submitClarification('opt-2')).toBe('Doc B');
    // Fabricated document UUID / unknown id is refused.
    expect(
      result.current.submitClarification('33333333-3333-3333-3333-333333333333'),
    ).toBeNull();
    // Terminal carries the resume block for reload rebuilds.
    expect(final?.clarification?.clarification_id).toBe('clr-1');
  });

  it('normalizes the legacy v1 clarification event to structured options', async () => {
    mockFetchFrames([
      'event: clarification\ndata: {"message":"Pick one","options":["Alpha","Beta"],"context":{}}\n\n',
      'event: complete\ndata: {"answer":"Pick one"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('summarize that document', [], false);
    });

    await waitFor(() => {
      expect(result.current.pendingClarification?.question).toBe('Pick one');
    });
    expect(result.current.pendingClarification?.options).toEqual([
      { option_id: 'Alpha', label: 'Alpha' },
      { option_id: 'Beta', label: 'Beta' },
    ]);
    // Legacy labels are server-issued values: selectable, nothing else is.
    expect(result.current.submitClarification('Alpha')).toBe('Alpha');
    expect(result.current.submitClarification('Gamma')).toBeNull();
  });

  it('parses fragmented frames split mid-line and ignores data-only frames', async () => {
    mockFetchFrames([
      'event: tok',
      'en\ndata: {"text": "hel',
      'lo"}\n\n',
      // Data-only frame: no event line — must NOT inherit the token type.
      'data: {"text": "orphan"}\n\n',
      'event: complete\ndata: {"answer": "hello"}\n\n',
    ]);
    const { result } = renderStreamHook();

    let final: ChatMessage | null = null;
    await act(async () => {
      final = await result.current.sendMessage('hi', [], false);
    });

    expect(final?.content).toBe('hello');
  });

  it('ignores unknown future event types without crashing', async () => {
    mockFetchFrames([
      'event: future_shiny\ndata: {"anything": 1}\n\n',
      'event: token\ndata: {"text": "ok"}\n\n',
      'event: complete\ndata: {"answer": "ok"}\n\n',
    ]);
    const { result } = renderStreamHook();

    let final: ChatMessage | null = null;
    await act(async () => {
      final = await result.current.sendMessage('hi', [], false);
    });

    expect(final?.content).toBe('ok');
    expect(result.current.error).toBeNull();
  });

  it('projects citation events into reload-safe message metadata', async () => {
    mockFetchFrames([
      'event: citation\ndata: {"citations":[{"citation_id":"cit-abc","label":"\\u0110i\\u1ec1u 5 \\u2014 Lu\\u1eadt ANM","document_id":"11111111-1111-1111-1111-111111111111","chunk_id":"c1"}]}\n\n',
      'event: complete\ndata: {"answer": "done", "citations": [{"citation_id":"cit-abc","label":"x"}]}\n\n',
    ]);
    const { result } = renderStreamHook();

    let final: ChatMessage | null = null;
    await act(async () => {
      final = await result.current.sendMessage('section query', [], false);
    });

    await waitFor(() => {
      expect(result.current.pendingCitations).toHaveLength(1);
    });
    expect(result.current.pendingCitations[0].citation_id).toBe('cit-abc');
    expect(final?.citations?.[0].citation_id).toBe('cit-abc');
  });

  it('treats cancelled as a quiet terminal, not an error', async () => {
    mockFetchFrames(['event: cancelled\ndata: {"reason": "user_stop"}\n\n']);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('hi', [], false);
    });

    expect(result.current.status).toBe('idle');
    expect(result.current.error).toBeNull();
  });
});
