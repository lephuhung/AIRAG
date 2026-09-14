/**
 * Phase 4D (Task 9) — structured clarification options banner.
 *
 * Renders the versioned public `clarification_required` request: question +
 * server-issued options. Selection submits ONLY the issued `option_id`
 * (resolved upstream via `submitClarification`); the component itself never
 * fabricates document/workspace/binding identity.
 *
 * `selectActiveClarification` picks what to show: the live in-stream
 * request while streaming, else the persisted resume block from the last
 * assistant message — but ONLY when no user message follows it, so an
 * answered/resumed turn never pins a dead banner (fix round 1, I2).
 */
import type { ChatMessage, PublicClarificationRequest } from "@/types";

export interface ActiveClarification {
  request: PublicClarificationRequest;
  /** Live requests are interactive; persisted reload resumes are too. */
  active: boolean;
}

export function selectActiveClarification(
  messages: ChatMessage[],
  live: PublicClarificationRequest | null,
): ActiveClarification | null {
  if (live && live.options.length > 0) {
    return { request: live, active: true };
  }
  // Last assistant message carrying a resume block.
  let lastIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === "assistant" && (m.clarification?.options?.length ?? 0) > 0) {
      lastIdx = i;
      break;
    }
  }
  if (lastIdx === -1) return null;
  // A user reply after the clarified turn means it was answered — hide.
  for (let i = lastIdx + 1; i < messages.length; i++) {
    if (messages[i].role === "user") return null;
  }
  return { request: messages[lastIdx].clarification!, active: true };
}

export function ClarificationBanner({
  request,
  active,
  onSelect,
}: {
  request: PublicClarificationRequest;
  active: boolean;
  onSelect: (optionId: string) => void;
}) {
  if (request.options.length === 0) return null;
  return (
    <div
      data-testid="clarification-options"
      className="mb-3 rounded-xl border border-primary/25 bg-primary/5 p-3"
    >
      <p className="text-sm font-medium mb-2">{request.question}</p>
      <div className="flex flex-wrap gap-2">
        {request.options.map((opt) => (
          <button
            key={opt.option_id}
            type="button"
            disabled={!active}
            onClick={() => onSelect(opt.option_id)}
            className="px-3 py-1.5 rounded-lg border border-primary/30 bg-background text-sm hover:bg-primary/10 disabled:opacity-50"
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
