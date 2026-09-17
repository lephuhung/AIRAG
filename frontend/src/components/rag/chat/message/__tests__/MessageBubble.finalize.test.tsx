/**
 * B2 — speculative → grounded answer settle animation.
 *
 * The assistant bubble uses ONE prose container for streaming and finished
 * answers; when `isStreaming` flips true→false with content it gains
 * `answer-finalizing` (900ms) and citation chips pop in with a stagger.
 * Messages that mount already finished (history reload) never animate.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, act, cleanup } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
vi.mock('katex/dist/katex.min.css', () => ({}));
import { MessageBubble } from '../MessageBubble';
import { useAuthStore } from '@/stores/authStore';
import type { ChatMessage, ChatSourceChunk } from '@/types';

function makeMessage(overrides: Partial<ChatMessage>): ChatMessage {
  return {
    id: 'm-1',
    role: 'assistant',
    content: '',
    timestamp: new Date().toISOString(),
    ...overrides,
  } as ChatMessage;
}

const SOURCES = [
  {
    index: 'a3z9',
    document_id: 'd1',
    chunk_id: 'c1',
    page_no: 2,
    content: 'excerpt',
    source_type: 'vector',
  },
] as unknown as ChatSourceChunk[];

function renderBubble(message: ChatMessage) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MessageBubble message={message} onAddAbbreviation={() => {}} />
    </QueryClientProvider>,
  );
}

describe('MessageBubble answer-finalize animation', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.setState({
      token: 'test-token',
      user: { id: 'user-1', email: 'test@test.com' } as any,
    });
    // useDocument() inside CitationLink resolves the chip label.
    (global.fetch as any) = vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: async () => ({
          id: 'd1',
          filename: 'Luat.pdf',
          original_filename: 'Luat.pdf',
          file_type: 'pdf',
          status: 'indexed',
        }),
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('settles in place: answer-finalizing + chip, no cursor, class clears after ~1s', async () => {
    const streaming = makeMessage({
      isStreaming: true,
      content: 'Mức phạt là 5 triệu.',
    });
    const { container, rerender } = renderBubble(streaming);

    expect(container.querySelector('.streaming-cursor')).toBeTruthy();
    expect(container.querySelector('.answer-finalizing')).toBeNull();

    // Fake timers before the transition so the 900ms settle timer is ours.
    vi.useFakeTimers();
    rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MessageBubble
          message={makeMessage({
            isStreaming: false,
            content: 'Mức phạt là 5 triệu [a3z9].',
            sources: SOURCES,
          })}
          onAddAbbreviation={() => {}}
        />
      </QueryClientProvider>,
    );

    const prose = container.querySelector('.answer-finalizing');
    expect(prose).toBeTruthy();
    expect(container.querySelector('.citation-chip')).toBeTruthy();
    expect(container.querySelector('.streaming-cursor')).toBeNull();

    // A second non-streaming rerender inside the 900ms window (e.g.
    // ChatPanel's finalMsg replacement) must NOT cancel the settle timer.
    rerender(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <MessageBubble
          message={makeMessage({
            isStreaming: false,
            content: 'Mức phạt là 5 triệu [a3z9]. ',
            sources: SOURCES,
          })}
          onAddAbbreviation={() => {}}
        />
      </QueryClientProvider>,
    );
    expect(container.querySelector('.answer-finalizing')).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(container.querySelector('.answer-finalizing')).toBeNull();
  });

  it('never fires for messages that mount already finished (history reload)', () => {
    const { container } = renderBubble(
      makeMessage({
        isStreaming: false,
        content: 'Mức phạt là 5 triệu [a3z9].',
        sources: SOURCES,
      }),
    );
    expect(container.querySelector('.answer-finalizing')).toBeNull();
    expect(container.querySelector('.citation-chip')).toBeTruthy();
  });
});
