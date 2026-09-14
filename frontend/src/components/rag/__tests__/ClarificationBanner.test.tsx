/**
 * Phase 4D (Task 9 fix round 1, I1/I2) — clarification banner component tests.
 *
 * - `selectActiveClarification` shows the live request while streaming;
 *   after reload it re-activates the persisted resume block ONLY when no
 *   user message follows it (answered turns never pin a dead banner).
 * - `ClarificationBanner` renders server-issued options and submits ONLY
 *   the issued option_id; buttons disable when inactive.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import {
  ClarificationBanner,
  selectActiveClarification,
} from '../ClarificationBanner';
import type { ChatMessage, PublicClarificationRequest } from '@/types';

function req(overrides: Partial<PublicClarificationRequest> = {}): PublicClarificationRequest {
  return {
    clarification_id: 'clr-1',
    question: 'Which document?',
    options: [
      { option_id: 'opt-1', label: 'Doc A' },
      { option_id: 'opt-2', label: 'Doc B' },
    ],
    resume: { thread_id: 'thread-1' },
    ...overrides,
  };
}

function msg(overrides: Partial<ChatMessage>): ChatMessage {
  return {
    id: `m-${Math.random().toString(36).slice(2)}`,
    role: 'assistant',
    content: '',
    timestamp: new Date().toISOString(),
    ...overrides,
  } as ChatMessage;
}

afterEach(() => cleanup());

describe('selectActiveClarification', () => {
  it('prefers the live request while streaming', () => {
    const live = req();
    const out = selectActiveClarification([msg({})], live);
    expect(out).toEqual({ request: live, active: true });
  });

  it('reactivates the persisted block after reload when unanswered', () => {
    const persisted = req();
    const out = selectActiveClarification(
      [msg({ role: 'user', content: 'q' }), msg({ clarification: persisted })],
      null,
    );
    expect(out).toEqual({ request: persisted, active: true });
  });

  it('hides the banner once a user message follows the clarified turn', () => {
    const persisted = req();
    const out = selectActiveClarification(
      [
        msg({ role: 'user', content: 'q' }),
        msg({ clarification: persisted }),
        msg({ role: 'user', content: 'Doc A' }),
      ],
      null,
    );
    expect(out).toBeNull();
  });

  it('returns null when nothing is pending', () => {
    expect(selectActiveClarification([msg({})], null)).toBeNull();
  });
});

describe('ClarificationBanner', () => {
  it('renders options and submits the issued option_id', () => {
    const onSelect = vi.fn();
    render(<ClarificationBanner request={req()} active={true} onSelect={onSelect} />);
    expect(screen.getByTestId('clarification-options')).toBeTruthy();
    fireEvent.click(screen.getByText('Doc B'));
    expect(onSelect).toHaveBeenCalledWith('opt-2');
  });

  it('disables every option when inactive and never submits', () => {
    const onSelect = vi.fn();
    render(<ClarificationBanner request={req()} active={false} onSelect={onSelect} />);
    const btn = screen.getByText('Doc A') as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    fireEvent.click(btn);
    expect(onSelect).not.toHaveBeenCalled();
  });
});
