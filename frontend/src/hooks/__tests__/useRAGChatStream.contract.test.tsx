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
    // Server-issued option resolves to the selection triple for the resume reply.
    expect(result.current.submitClarification('opt-2')).toEqual({
      clarification_id: 'clr-1',
      selected_option_id: 'opt-2',
      label: 'Doc B',
    });
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
    expect(result.current.submitClarification('Alpha')).toEqual({
      clarification_id: 'clr-1',
      selected_option_id: 'Alpha',
      label: 'Alpha',
    });
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

describe('normalized relay vocabulary (fix round 1, C1/I4)', () => {
  it('projects normalized citations into the sources presentation model', async () => {
    mockFetchFrames([
      'event: status\ndata: {"contract_version":"v2.chat/1","step":"retrieving","phase":"executing","detail":"Searching"}\n\n',
      'event: citation\ndata: {"contract_version":"v2.chat/1","citations":[{"citation_id":"cit-1","label":"L1","document_id":"d1","chunk_id":"c1","content":"excerpt"}],"image_refs":[],"people":[]}\n\n',
      'event: token\ndata: {"contract_version":"v2.chat/1","text":"answer"}\n\n',
      'event: complete\ndata: {"contract_version":"v2.chat/1","answer":"answer","citations":[],"image_refs":[],"people":[],"potential_abbreviations":[]}\n\n',
    ]);
    const { result } = renderStreamHook();

    let final: ChatMessage | null = null;
    await act(async () => {
      final = await result.current.sendMessage('rag query', [], false);
    });

    // Existing citation panels keep rendering via the compat projection.
    expect(result.current.pendingSources.length).toBeGreaterThan(0);
    expect(result.current.pendingSources[0].chunk_id).toBe('c1');
    expect(final?.sources?.[0].chunk_id).toBe('c1');
    expect(
      result.current.agentSteps.some((s) => s.step === 'sources_found'),
    ).toBe(true);
  });

  it('merges people from citation frames and abbreviations from status', async () => {
    mockFetchFrames([
      'event: citation\ndata: {"citations":[],"image_refs":[],"people":[{"ho_ten":"Nguyen Van A","_source_schema":"lg"}]}\n\n',
      'event: status\ndata: {"step":"abbreviations","phase":"evaluating","abbreviations":["ABC"]}\n\n',
      'event: complete\ndata: {"answer":"done"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('who is X', [], false);
    });

    await waitFor(() => {
      expect(result.current.pendingPeople).toHaveLength(1);
    });
    expect(result.current.potentialAbbreviations).toEqual(['ABC']);
  });

  it('clears speculative citations on public status-rollback', async () => {
    mockFetchFrames([
      'event: citation\ndata: {"citations":[{"citation_id":"cit-1","label":"L1","document_id":"d1","chunk_id":"c1"}]}\n\n',
      'event: status\ndata: {"step":"rollback","phase":"executing","detail":""}\n\n',
      'event: complete\ndata: {"answer":"fresh"}\n\n',
    ]);
    const { result } = renderStreamHook();

    let final: ChatMessage | null = null;
    await act(async () => {
      final = await result.current.sendMessage('q', [], false);
    });

    expect(result.current.pendingCitations).toHaveLength(0);
    expect(result.current.pendingSources).toHaveLength(0);
    expect(final?.sources ?? []).toHaveLength(0);
    // Fix round 2 (N-I2): the discarded draft's citations must not survive
    // into the completed message either.
    expect(final?.citations ?? []).toHaveLength(0);
  });

  it('sends the structured selection envelope with clarification replies', async () => {
    mockFetchFrames([
      'event: clarification_required\ndata: {"clarification_id":"clr-1","question":"Which?","options":[{"option_id":"opt-1","label":"Doc A"}],"resume":{"thread_id":"t"}}\n\n',
      'event: complete\ndata: {"answer":"Which?"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('ambiguous', [], false);
    });

    const triple = result.current.submitClarification('opt-1');
    expect(triple?.selected_option_id).toBe('opt-1');

    mockFetchFrames(['event: complete\ndata: {"answer":"ok"}\n\n']);
    await act(async () => {
      await result.current.sendMessage('Doc A', [], false, false, undefined, undefined, {
        clarification_id: triple!.clarification_id,
        selected_option_id: triple!.selected_option_id,
      });
    });

    const currentFetch = global.fetch as any;
    const lastCall = currentFetch.mock.calls[currentFetch.mock.calls.length - 1];
    const body = JSON.parse(lastCall[1].body);
    expect(body.clarification_selection).toEqual({
      clarification_id: 'clr-1',
      selected_option_id: 'opt-1',
    });
  });
});

describe('citation handle preservation (fix round 2, N-C1)', () => {
  it('keeps the answer index and KG provenance, and markers resolve', async () => {
    const { injectCitations } = await import(
      '@/components/rag/chat/markdown/citations'
    );
    const { Children, isValidElement } = await import('react');
    mockFetchFrames([
      'event: citation\ndata: {"contract_version":"v2.chat/1","citations":[{"citation_id":"cit-1","label":"L1","index":"a3x9","document_id":"d1","chunk_id":"c1","content":"excerpt","source_type":"vector","score":0.91},{"citation_id":"cit-2","label":"L2","index":"k7q2","document_id":"d2","chunk_id":"k1","content":"kg","source_type":"kg","score":0.77}],"image_refs":[],"people":[]}\n\n',
      'event: complete\ndata: {"answer":"done"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('rag query', [], false);
    });

    // Compat projection preserves the handle the answer cites by.
    expect(result.current.pendingSources[0].index).toBe('a3x9');
    expect(result.current.pendingSources[1].index).toBe('k7q2');
    expect(result.current.pendingSources[1].source_type).toBe('kg');

    // In-text markers resolve against the projected sources (incl. KG).
    const nodes = Children.toArray(
      injectCitations(
        'Theo quy định [a3x9] và đồ thị tri thức [k7q2].',
        result.current.pendingSources,
        [],
      ),
    );
    const indexes = nodes
      .filter((n) => isValidElement(n))
      .map((n) => (n as any).props.index);
    expect(indexes).toContain('a3x9');
    expect(indexes).toContain('k7q2');
    expect(
      nodes.some((n) => typeof n === 'string' && String(n).includes('[a3x9]')),
    ).toBe(false);
  });
});

describe('required scenarios (fix round 2, I1)', () => {
  it('preserves exact-section locators through normalized citations', async () => {
    mockFetchFrames([
      'event: citation\ndata: {"citations":[{"citation_id":"cit-s","label":"\\u0110i\\u1ec1u 5 \\u2014 24/2018/QH14","index":"d5f1","document_id":"d9","chunk_id":"sec-5","document_number":"24/2018/QH14","article_label":"\\u0110i\\u1ec1u 5"}]}\n\n',
      'event: complete\ndata: {"answer":"ok"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('section query', [], false);
    });

    expect(result.current.pendingSources[0].article_label).toBe('Điều 5');
    expect(result.current.pendingSources[0].document_number).toBe('24/2018/QH14');
  });

  it('surfaces not-found/error terminals without crashing', async () => {
    mockFetchFrames([
      'event: error\ndata: {"contract_version":"v2.chat/1","message":"Không tìm thấy văn bản phù hợp"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('unknown doc query', [], false);
    });

    expect(result.current.error).toContain('Không tìm thấy');
    expect(result.current.isStreaming).toBe(false);
  });
});

describe('validity badges + clarification symmetry (fix round 3)', () => {
  it('preserves validity_status/superseded_by for destructive badges', async () => {
    mockFetchFrames([
      'event: citation\ndata: {"citations":[{"citation_id":"cit-v","label":"L","index":"e8f2","document_id":"d1","chunk_id":"c1","validity_status":"superseded","superseded_by":"VB 99/2024"}]}\n\n',
      'event: complete\ndata: {"answer":"ok"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('validity query', [], false);
    });

    expect(result.current.pendingSources[0].validity_status).toBe('superseded');
    expect(result.current.pendingSources[0].superseded_by).toBe('VB 99/2024');
  });

  it('clears pending clarification on public status-rollback', async () => {
    mockFetchFrames([
      'event: clarification_required\ndata: {"clarification_id":"clr-1","question":"Which?","options":[{"option_id":"opt-1","label":"Doc A"}],"resume":{"thread_id":"t"}}\n\n',
      'event: status\ndata: {"step":"rollback","phase":"executing","detail":""}\n\n',
      'event: complete\ndata: {"answer":"fresh"}\n\n',
    ]);
    const { result } = renderStreamHook();

    await act(async () => {
      await result.current.sendMessage('q', [], false);
    });

    expect(result.current.pendingClarification).toBeNull();
  });
});
