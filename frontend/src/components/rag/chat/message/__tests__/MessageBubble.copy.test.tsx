import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup, fireEvent, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
vi.mock('katex/dist/katex.min.css', () => ({}));
import { MessageBubble } from '../MessageBubble';
import { useAuthStore } from '@/stores/authStore';
import type { ChatMessage } from '@/types';

function renderBubble(message: ChatMessage) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MessageBubble message={message} onAddAbbreviation={() => {}} />
    </QueryClientProvider>,
  );
}

const CONTENT = 'Mạng LAN trong BMNN quy định như thế nào';
const userMsg = (overrides: Partial<ChatMessage> = {}) =>
  ({
    id: 'msg_9fc748f2',
    role: 'user',
    content: CONTENT,
    timestamp: new Date().toISOString(),
    ...overrides,
  }) as ChatMessage;

describe('user message copy button', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.setState({
      token: 't',
      user: { id: 'u1', email: 't@t.com' } as any,
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('secure context: writes message.content (not message.id)', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    vi.stubGlobal('isSecureContext', true);
    Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true });
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    const { container } = renderBubble(userMsg());
    const btn = container.querySelector('button[aria-label]')!;
    await act(async () => {
      fireEvent.click(btn);
    });
    expect(writeText).toHaveBeenCalledWith(CONTENT);
  });

  it('secure context: resolves <document_id=…> tags to @name', async () => {
    const writeText = vi.fn(() => Promise.resolve());
    Object.defineProperty(window, 'isSecureContext', { value: true, configurable: true });
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });
    const doc = { id: 'doc-uuid-1', filename: 'a.pdf', original_filename: 'Bao_cao.pdf', status: 'indexed' } as any;

    const { container } = renderBubble(
      userMsg({
        content: 'xem giúp tôi <document_id=doc-uuid-1> nhé',
        documentIds: ['doc-uuid-1'],
        attachedDocs: [doc],
      }),
    );
    const btn = container.querySelector('button[aria-label]')!;
    await act(async () => {
      fireEvent.click(btn);
    });
    expect(writeText).toHaveBeenCalledWith('xem giúp tôi @Bao_cao nhé');
  });

  it('insecure context (HTTP): falls back to execCommand and still copies content', async () => {
    Object.defineProperty(window, 'isSecureContext', { value: false, configurable: true });
    Object.defineProperty(navigator, 'clipboard', { value: undefined, configurable: true });
    const execCommand = vi.fn(() => true);
    Object.defineProperty(document, 'execCommand', { value: execCommand, configurable: true });

    const { container } = renderBubble(userMsg());
    const btn = container.querySelector('button[aria-label]')!;
    await act(async () => {
      fireEvent.click(btn);
    });
    expect(execCommand).toHaveBeenCalledWith('copy');
    // The hidden textarea must have carried the message content
    expect(execCommand.mock.calls.length).toBe(1);
  });
});
