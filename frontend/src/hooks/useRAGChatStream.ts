/**
 * useRAGChatStream — SSE streaming hook for HRAG chat.
 *
 * Handles Server-Sent Events from the /chat/{workspace_id}/stream endpoint,
 * with rAF-buffered token rendering, AgentStep tracking, and AbortController cleanup.
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { useAuthStore } from "@/stores/authStore";
import { generateId } from "@/lib/utils";
import type {
  ChatSourceChunk,
  ChatImageRef,
  ChatStreamStatus,
  ChatMessage,
  AgentStep,
  AgentStepType,
  PeopleRecord,
  PublicCitation,
  PublicClarificationRequest,
  ClarificationOption,
} from "@/types";

const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

export interface RAGStreamResult {
  /** Current stream status */
  status: ChatStreamStatus;
  /** Accumulated streaming content (answer text so far) */
  streamingContent: string;
  /** Accumulated thinking text */
  thinkingText: string;
  /** Sources received from retrieval */
  pendingSources: ChatSourceChunk[];
  /** Image refs received from retrieval */
  pendingImages: ChatImageRef[];
  /** People records from MongoDB people search */
  pendingPeople: PeopleRecord[];
  /** Public citations normalized from sources/citation events */
  pendingCitations: PublicCitation[];
  /** Structured clarification request awaiting user selection (or null) */
  pendingClarification: PublicClarificationRequest | null;
  /** Tick to force ChatPanel useEffect re-run after streaming complete */
  streamCompleteTick: number;
  /** Error message if any */
  error: string | null;
  /** Whether currently streaming */
  isStreaming: boolean;
  /** Agent processing steps for ThinkingTimeline */
  agentSteps: AgentStep[];
  /** Potential abbreviations identified by backend but missing from DB */
  potentialAbbreviations: string[];
  /** Server-assigned ID for the assistant message current streaming */
  aiMessageId: string | null;
  /** Server-assigned ID for the user message that started this stream */
  userMessageId: string | null;
  /** Callback invoked when backend updates session title via topic_label (SSE event) */
  onSessionTitleUpdated?: (title: string) => void;
  /** Send a message — returns the finalized ChatMessage on complete */
  sendMessage: (
    message: string,
    history: { role: string; content: string }[],
    enableThinking: boolean,
    forceSearch?: boolean,
    overrideSessionId?: string,
    documentIds?: string[],
  ) => Promise<ChatMessage | null>;
  /** Cancel ongoing stream */
  cancel: () => void;
  /**
   * Resolve a clarification selection to the server-issued label to send.
   * Returns null unless `optionId` was issued in the pending request —
   * the UI can never submit fabricated document/workspace/binding identity.
   */
  submitClarification: (optionId: string) => string | null;
  /** Reset all state */
  reset: () => void;
}

// ---------------------------------------------------------------------------
// AgentStep helpers
// ---------------------------------------------------------------------------

function createStep(
  step: AgentStepType,
  detail: string,
  status: "active" | "completed" | "error" = "active",
): AgentStep {
  return {
    id: generateId(),
    step,
    detail,
    status,
    timestamp: Date.now(),
  };
}

function completeActiveStep(steps: AgentStep[]): AgentStep[] {
  const now = Date.now();
  return steps.map((s) =>
    s.status === "active"
      ? { ...s, status: "completed" as const, durationMs: now - s.timestamp }
      : s,
  );
}

function markActiveError(steps: AgentStep[]): AgentStep[] {
  return steps.map((s) =>
    s.status === "active" ? { ...s, status: "error" as const } : s,
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useRAGChatStream(
  sessionId: string | null,
  onSessionTitleUpdated?: (title: string) => void,
): RAGStreamResult {
  const [status, setStatus] = useState<ChatStreamStatus>("idle");
  const [streamingContent, setStreamingContent] = useState("");
  const [thinkingText, setThinkingText] = useState("");
  const [pendingSources, setPendingSources] = useState<ChatSourceChunk[]>([]);
  const [pendingImages, setPendingImages] = useState<ChatImageRef[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const [potentialAbbreviations, setPotentialAbbreviations] = useState<string[]>([]);
  const [aiMessageId, setAiMessageId] = useState<string | null>(null);
  const [userMessageId, setUserMessageId] = useState<string | null>(null);
  const [pendingPeople, setPendingPeople] = useState<PeopleRecord[]>([]);
  const [pendingCitations, setPendingCitations] = useState<PublicCitation[]>([]);
  const [pendingClarification, setPendingClarification] = useState<PublicClarificationRequest | null>(null);
  // Ref mirror so submitClarification can resolve synchronously (state
  // updaters must stay pure — no side effects inside setState).
  const pendingClarificationRef = useRef<PublicClarificationRequest | null>(null);
  useEffect(() => {
    pendingClarificationRef.current = pendingClarification;
  }, [pendingClarification]);
  // Tick to force ChatPanel useEffect to re-run after complete
  const [streamCompleteTick, setStreamCompleteTick] = useState(0);

  const abortRef = useRef<AbortController | null>(null);
  // Session of the in-flight stream — target for the server-side cancel call.
  const activeSessionRef = useRef<string | null>(null);
  const bufferRef = useRef("");
  const rafRef = useRef<number | undefined>(undefined);

  // Separate thinking text buffer for AgentStep thinkingText updates
  const thinkingBufferRef = useRef("");
  const thinkingRafRef = useRef<number | undefined>(undefined);

  // Track start time for total duration
  const streamStartRef = useRef(0);

  // Persist people data across resets (for final message after streaming completes)
  const peopleDataRef = useRef<PeopleRecord[]>([]);

  // Cleanup on unmount — only closes the socket. The backend run is detached
  // from the connection: it keeps generating and persists the answer, which
  // the remounted panel then picks up from history. Do NOT call the server
  // cancel endpoint here; that is reserved for the explicit stop button.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (thinkingRafRef.current) cancelAnimationFrame(thinkingRafRef.current);
    };
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setStreamingContent("");
    setThinkingText("");
    setPendingSources([]);
    setPendingImages([]);
    setError(null);
    setIsStreaming(false);
    setAgentSteps([]);
    setPotentialAbbreviations([]);
    setPendingCitations([]);
    setPendingClarification(null);
    setAiMessageId(null);
    setUserMessageId(null);
    // Keep pendingPeople - it persists across streaming sessions for card display
    bufferRef.current = "";
    thinkingBufferRef.current = "";
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = undefined;
    }
    if (thinkingRafRef.current) {
      cancelAnimationFrame(thinkingRafRef.current);
      thinkingRafRef.current = undefined;
    }
  }, []);

  const cancel = useCallback(() => {
    // The run is detached server-side: closing the socket alone no longer
    // stops generation, so ask the backend to cancel the run too (it persists
    // whatever partial answer was generated).
    const sid = activeSessionRef.current;
    if (sid) {
      const token = useAuthStore.getState().token;
      fetch(`${BASE_URL}/rag/chat/sessions/${sid}/stream/cancel`, {
        method: "POST",
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      }).catch(() => {});
    }
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
    setIsStreaming(false);
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = undefined;
    }
    if (thinkingRafRef.current) {
      cancelAnimationFrame(thinkingRafRef.current);
      thinkingRafRef.current = undefined;
    }
    // Flush any remaining token buffer
    if (bufferRef.current) {
      const remaining = bufferRef.current;
      bufferRef.current = "";
      setStreamingContent((prev) => prev + remaining);
    }
  }, []);

  const onToken = useCallback((text: string) => {
    // Check for delimiter to separate thinking from answer (safety net for leaked tags)
    const fullText = bufferRef.current + text;
    const delimiter = "</think>\n\n";
    const delimIndex = fullText.indexOf(delimiter);

    if (delimIndex !== -1) {
      // Transition point found in the middle of a token stream!
      const thinkingPart = fullText.slice(0, delimIndex);
      const answerPart = fullText.slice(delimIndex + delimiter.length);

      if (thinkingPart) onThinkingToken(thinkingPart);
      
      // Flush answer chunk
      bufferRef.current = answerPart;
      if (answerPart && !rafRef.current) {
        rafRef.current = requestAnimationFrame(() => {
          const chunk = bufferRef.current;
          bufferRef.current = "";
          rafRef.current = undefined;
          setStreamingContent((prev) => prev + chunk);
        });
      }
    } else {
      // No delimiter found. 
      // If the text starts with "<think" but hasn't finished, we might want to treat it as thinking.
      // But for now, we'll assume that if it's sent as a 'token' event, it's answer content 
      // UNLESS it's very clearly thinking (which we'd detect above).
      bufferRef.current += text;
      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(() => {
          const chunk = bufferRef.current;
          bufferRef.current = "";
          rafRef.current = undefined;
          setStreamingContent((prev) => prev + chunk);
        });
      }
    }
  }, []);

  // Buffered thinking text update for the analyzing AgentStep
  const onThinkingToken = useCallback((text: string) => {
    // Update flat thinkingText state (existing behavior)
    setThinkingText((prev) => prev + text);

    // Buffer thinking text for AgentStep update
    thinkingBufferRef.current += text;
    if (!thinkingRafRef.current) {
      thinkingRafRef.current = requestAnimationFrame(() => {
        const chunk = thinkingBufferRef.current;
        thinkingBufferRef.current = "";
        thinkingRafRef.current = undefined;

        setAgentSteps((prev) => {
          // Find the analyzing step regardless of status — thinking can
          // arrive during both the first iteration (analyzing=active) and
          // the second iteration after tool call (analyzing=completed).
          const idx = prev.findIndex((s) => s.step === "analyzing");
          if (idx === -1) return prev;
          const updated = [...prev];
          updated[idx] = {
            ...updated[idx],
            thinkingText: (updated[idx].thinkingText || "") + chunk,
          };
          return updated;
        });
      });
    }
  }, []);

  const sendMessage = useCallback(
    async (
      message: string,
      history: { role: string; content: string }[],
      enableThinking: boolean,
      forceSearch: boolean = false,
      overrideSessionId?: string,
      documentIds?: string[],
    ): Promise<ChatMessage | null> => {
      // Reset state for new message
      setStreamingContent("");
      setThinkingText("");
      setPendingSources([]);
      setPendingImages([]);
      setError(null);
      setStatus("analyzing");
      setIsStreaming(true);
      setAgentSteps([]);
      setPotentialAbbreviations([]);
      setPendingCitations([]);
      setPendingClarification(null);
      setAiMessageId(null);
      setUserMessageId(null);
      setPendingPeople([]);
      bufferRef.current = "";
      thinkingBufferRef.current = "";
      streamStartRef.current = Date.now();

      // Synchronous local tracker — avoids React 18 batching race condition
      // where agentStepsRef in ChatPanel may be stale when sendMessage resolves
      let localSteps: AgentStep[] = [];
      let localSources: ChatSourceChunk[] = [];
      let localImages: ChatImageRef[] = [];
      let localPeople: PeopleRecord[] = [];
      let localCitations: PublicCitation[] = [];
      let localClarification: PublicClarificationRequest | null = null;
      let localAiMessageId: string | null = null;
      let localUserMessageId: string | null = null;
      // Accumulate all thinking text in this scope so it can be flushed into
      // localSteps at complete time (onThinkingToken only updates setAgentSteps
      // via RAF, which never syncs back to localSteps)
      let thinkingAccumulator = "";
      function syncUpdateSteps(updater: AgentStep[] | ((prev: AgentStep[]) => AgentStep[])): void {
        const next = typeof updater === "function" ? updater(localSteps) : updater;
        localSteps = next;
        setAgentSteps(next);
      }

      abortRef.current = new AbortController();

      try {
        const sid = overrideSessionId || sessionId;
        if (!sid) throw new Error("No active chat session.");
        activeSessionRef.current = sid;
        const token = useAuthStore.getState().token;
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (token) {
          headers["Authorization"] = `Bearer ${token}`;
        }

        const response = await fetch(
          `${BASE_URL}/rag/chat/sessions/${sid}/stream`,
          {
            method: "POST",
            headers,
            body: JSON.stringify({
              message,
              history,
              enable_thinking: enableThinking,
              force_search: forceSearch,
              document_ids: documentIds,
            }),
            signal: abortRef.current.signal,
          },
        );

        if (!response.ok) {
          const err = await response
            .json()
            .catch(() => ({ detail: "Stream request failed" }));
          throw new Error(err.detail || `Error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let sseBuffer = "";
        let currentEventType = "unknown";
        let finalMessage: ChatMessage | null = null;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          sseBuffer += decoder.decode(value, { stream: true });
          const lines = sseBuffer.split("\n");
          sseBuffer = lines.pop() || "";

          for (const line of lines) {
            // Skip heartbeat comments
            if (line.startsWith(":")) continue;
            if (line === "") {
              // Blank line ends one SSE dispatch — a following data-only
              // frame must not inherit the previous event type.
              currentEventType = "unknown";
              continue;
            }

            // Tolerate both "event: x" and "event:x" framing.
            if (line.startsWith("event:")) {
              currentEventType = line.slice(6).trim() || "unknown";
              continue;
            }

            if (line.startsWith("data:")) {
              const jsonStr = line.slice(5).trim();
              if (!jsonStr) continue;

              // Capture the dispatch type, then reset so a later data-only
              // frame is ignored instead of misattributed.
              const dispatchType = currentEventType;
              currentEventType = "unknown";

              try {
                const data = JSON.parse(jsonStr);

                switch (dispatchType) {
                  case "status": {
                    const step = data.step as string;
                    const detail = (data.detail as string) || "";

                    if (step === "analyzing") {
                      setStatus("analyzing");
                      syncUpdateSteps((prev) => [
                        ...prev,
                        createStep("analyzing", detail || "Analyzing your question..."),
                      ]);
                    } else if (step === "searching") {
                      setStatus("retrieving");
                      syncUpdateSteps((prev) => [
                        ...completeActiveStep(prev),
                        createStep("understood", "Understood query", "completed"),
                        createStep("retrieving", detail || "Searching documents..."),
                      ]);
                    } else if (step === "retrieved") {
                      setStatus("retrieving");
                    } else if (step === "retrieving") {
                      setStatus("retrieving");
                      syncUpdateSteps((prev) => [
                        ...completeActiveStep(prev),
                        createStep("understood", "Understood query", "completed"),
                        createStep("retrieving", detail || "Searching documents..."),
                      ]);
                    } else if (step === "generating") {
                      setStatus("generating");
                      syncUpdateSteps((prev) => [
                        ...completeActiveStep(prev),
                        createStep("generating", detail || "Generating answer..."),
                      ]);
                    } else if (step === "rollback") {
                      // Public-contract form of token_rollback: clear
                      // speculative content the same way.
                      bufferRef.current = "";
                      setStreamingContent("");
                      localSources = [];
                      localImages = [];
                      localPeople = [];
                      setPendingSources([]);
                      setPendingImages([]);
                      setPendingPeople([]);
                      peopleDataRef.current = [];
                      setPendingCitations([]);
                      setPotentialAbbreviations([]);
                    }
                    // Public UI phases (Task 9): clarifying/planning surfaced
                    // without exposing chain-of-thought.
                    const phase = data.phase as string | undefined;
                    if (phase === "clarifying") {
                      setStatus("clarifying");
                    } else if (phase === "planning" && step !== "analyzing") {
                      setStatus("analyzing");
                    }
                    break;
                  }

                  case "ai_message_id":
                    localAiMessageId = data.message_id || null;
                    setAiMessageId(localAiMessageId);
                    break;
                  case "user_id":
                    localUserMessageId = data.id || null;
                    setUserMessageId(localUserMessageId);
                    break;

                  case "thinking":
                    onThinkingToken(data.text || "");
                    thinkingAccumulator += data.text || "";
                    break;

                  case "sources": {
                    const sources = (data.sources || []) as ChatSourceChunk[];
                    localSources = sources;
                    setPendingSources([...sources]);

                    // Add sources_found step with badges
                    const badges = sources.map((s) => String(s.index));
                    syncUpdateSteps((prev) => [
                      ...completeActiveStep(prev),
                      createStep("sources_found", `Found ${sources.length} source${sources.length > 1 ? "s" : ""}`, "completed"),
                    ].map((s) =>
                      s.step === "sources_found" && s.status === "completed" && !s.sourceBadges
                        ? { ...s, sourceBadges: badges, sourceCount: sources.length }
                        : s,
                    ));
                    break;
                  }

                  case "images": {
                    const imgs = (data.image_refs || []) as ChatImageRef[];
                    localImages = imgs;
                    setPendingImages([...imgs]);

                    // Update sources_found step with image count
                    if (imgs.length > 0) {
                      syncUpdateSteps((prev) => {
                        let lastSourcesIdx = -1;
                        for (let i = prev.length - 1; i >= 0; i--) {
                          if (prev[i].step === "sources_found") {
                            lastSourcesIdx = i;
                            break;
                          }
                        }
                        if (lastSourcesIdx === -1) return prev;
                        const updated = [...prev];
                        const existing = updated[lastSourcesIdx];
                        updated[lastSourcesIdx] = {
                          ...existing,
                          imageCount: (existing.imageCount || 0) + imgs.length,
                          detail: `Found ${existing.sourceCount || 0} source${(existing.sourceCount || 0) > 1 ? "s" : ""} + ${(existing.imageCount || 0) + imgs.length} image${(existing.imageCount || 0) + imgs.length > 1 ? "s" : ""}`,
                        };
                        return updated;
                      });
                    }
                    break;
                  }

                  case "people_data": {
                    const people = (data.people || []) as PeopleRecord[];
                    localPeople = people;
                    peopleDataRef.current = people;
                    setPendingPeople([...people]);

                    // Add people_found step
                    syncUpdateSteps((prev) => [
                      ...completeActiveStep(prev),
                      createStep("sources_found", `Found ${people.length} people record${people.length > 1 ? "s" : ""}`, "completed"),
                    ]);
                    break;
                  }

                  case "token":
                    onToken(data.text || "");
                    break;

                  // ── Phase 4D (Task 9): versioned public contract events ──
                  case "citation": {
                    const citations = (data.citations || []) as PublicCitation[];
                    localCitations = citations;
                    setPendingCitations([...citations]);
                    if (Array.isArray(data.image_refs) && data.image_refs.length > 0) {
                      localImages = data.image_refs as ChatImageRef[];
                      setPendingImages([...localImages]);
                    }
                    if (Array.isArray(data.people) && data.people.length > 0) {
                      localPeople = data.people as PeopleRecord[];
                      peopleDataRef.current = localPeople;
                      setPendingPeople([...localPeople]);
                    }
                    break;
                  }

                  case "clarification_required":
                  case "clarification": {
                    // Structured request (public) or legacy v1 shape
                    // {message, options: string[], context}. Legacy labels are
                    // server-issued values: presented as options whose id is
                    // the label itself, so selection still submits only
                    // server-issued values.
                    const rawOptions: unknown[] = Array.isArray(data.options)
                      ? data.options
                      : [];
                    const options: ClarificationOption[] = rawOptions.map(
                      (o: unknown, i: number) =>
                        typeof o === "string"
                          ? { option_id: o, label: o }
                          : {
                              option_id: String(
                                (o as Record<string, unknown>).option_id ||
                                (o as Record<string, unknown>).candidate_id ||
                                `opt-${i + 1}`,
                              ),
                              label: String(
                                (o as Record<string, unknown>).label ||
                                (o as Record<string, unknown>).option_id ||
                                `opt-${i + 1}`,
                              ),
                            },
                    );
                    const req: PublicClarificationRequest = {
                      clarification_id: String(
                        data.clarification_id || data.clarificationId || "clr-1",
                      ),
                      reason: String(data.reason || "semantic_ambiguity"),
                      question: String(
                        data.question || data.message || "",
                      ),
                      options,
                      resume: {
                        thread_id: String(
                          data.resume?.thread_id ||
                            data.context?.thread_id ||
                            "",
                        ),
                        message_id: data.resume?.message_id
                          ? String(data.resume.message_id)
                          : undefined,
                      },
                    };
                    localClarification = req;
                    setPendingClarification(req);
                    setStatus("clarifying");
                    break;
                  }

                  case "clarification_resolved": {
                    setPendingClarification(null);
                    localClarification = null;
                    break;
                  }

                  case "cancelled": {
                    // Server-initiated cancel terminal: finalize quietly
                    // without an error banner.
                    setStatus("idle");
                    setIsStreaming(false);
                    break;
                  }

                  default:
                    // Forward-compatible: unknown future event types never
                    // crash the stream.
                    break;

                  case "potential_abbreviations":
                    setPotentialAbbreviations(data.abbreviations || []);
                    break;

                  case "token_rollback":
                    // Clear speculative tokens and all retractable artifacts
                    // Per B5: token_rollback must clear localSources, localImages,
                    // pendingSources, pendingImages, peopleData, potentialAbbreviations
                    bufferRef.current = "";
                    if (rafRef.current) {
                      cancelAnimationFrame(rafRef.current);
                      rafRef.current = undefined;
                    }
                    setStreamingContent("");
                    // Clear all retractable artifacts
                    localSources = [];
                    localImages = [];
                    localPeople = [];
                    localCitations = [];
                    setPendingSources([]);
                    setPendingImages([]);
                    setPendingPeople([]);
                    setPendingCitations([]);
                    peopleDataRef.current = [];
                    setPotentialAbbreviations([]);
                    break;

                  case "complete": {
                    // Strip  markers from streamed answer (defensive — in case backend didn't strip)
                    const cleanAnswer = (streamingContent || data.answer || "").replace(/<\/think>\s*/g, "").trim();
                    // Flush remaining buffer
                    bufferRef.current = "";
                    if (rafRef.current) {
                      cancelAnimationFrame(rafRef.current);
                      rafRef.current = undefined;
                    }
                    // Flush accumulated thinking into localSteps so finalMessage.agentSteps has thinkingText
                    if (thinkingAccumulator) {
                      syncUpdateSteps((prev) =>
                        prev.map((s) =>
                          s.step === "analyzing"
                            ? { ...s, thinkingText: (s.thinkingText || "") + thinkingAccumulator }
                            : s,
                        ),
                      );
                      thinkingAccumulator = "";
                    }
                    // Flush thinking buffer (cancel pending RAF)
                    if (thinkingBufferRef.current) {
                      thinkingBufferRef.current = "";
                      if (thinkingRafRef.current) {
                        cancelAnimationFrame(thinkingRafRef.current);
                        thinkingRafRef.current = undefined;
                      }
                    }

                    // Complete active step + add done step (sync localSteps too)
                    const totalMs = Date.now() - streamStartRef.current;
                    syncUpdateSteps((prev) => [
                      ...completeActiveStep(prev),
                      {
                        ...createStep("done", `Done in ${totalMs >= 1000 ? `${(totalMs / 1000).toFixed(1)}s` : `${totalMs}ms`}`, "completed"),
                        durationMs: totalMs
                      },
                    ]);

                    finalMessage = {
                      id: localAiMessageId || generateId(),
                      role: "assistant",
                      content: cleanAnswer,
                      sources: localSources, // use accumulated sources, backend complete event omits them
                      relatedEntities: data.related_entities || [],
                      imageRefs: localImages,
                      peopleData: localPeople,
                      thinking: data.thinking || null,
                      agentSteps: localSteps, // include synced steps directly in finalMessage
                      potential_abbreviations: data.potential_abbreviations || potentialAbbreviations,
                      // Phase 4D (Task 9): public-contract reload metadata.
                      citations: localCitations.length > 0
                        ? localCitations
                        : (Array.isArray(data.citations) ? data.citations : undefined),
                      clarification: localClarification
                        || (data.clarification as PublicClarificationRequest | undefined)
                        || undefined,
                      timestamp: new Date().toISOString(),
                    };

                    // Immediately clear loading UI state on backend 'complete' event,
                    // without waiting for the underlying HTTP connection to close.
                    setStatus("idle");
                    setIsStreaming(false);
                    setThinkingText(""); // Clear thinking state to prevent ghost thinking panel
                    // Force ChatPanel useEffect to re-run with final peopleData
                    setStreamCompleteTick((t) => t + 1);

                    break;
                  }




                  case "session_title_updated":
                    if (data.Title && onSessionTitleUpdated) {
                      onSessionTitleUpdated(data.Title);
                    }
                    break;

                  case "error":
                    setError(data.message || "Unknown error");
                    setStatus("error");
                    syncUpdateSteps((prev) => markActiveError(prev));
                    break;
                }
              } catch {
                // Ignore malformed JSON
              }
            }
          }
        }

        // Stream ended — if we never got a 'complete' event (connection dropped),
        // finalize with whatever content we have to prevent stuck UI state
        if (!finalMessage && (streamingContent || thinkingAccumulator)) {
          console.warn("[stream] Stream ended without 'complete' event — finalizing with buffered content");
          finalMessage = {
            id: localAiMessageId || generateId(),
            role: "assistant",
            content: (streamingContent || "").replace(/<\/think>\s*/g, "").trim(),
            sources: localSources,
            relatedEntities: [],
            imageRefs: localImages,
            peopleData: localPeople,
            thinking: thinkingAccumulator || null,
            agentSteps: localSteps,
            timestamp: new Date().toISOString(),
          };
        }

        setStatus("idle");
        setIsStreaming(false);
        setThinkingText(""); // Clear thinking state

        return finalMessage;
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          // User cancelled — don't set error
          return null;
        }
        const msg = (err as Error).message || "Stream failed";
        setError(msg);
        setStatus("error");
        setIsStreaming(false);
        syncUpdateSteps((prev) => markActiveError(prev));
        return null;
      }
    },
    [sessionId, onToken, onThinkingToken],
  );

  const submitClarification = useCallback((optionId: string): string | null => {
    // Only a server-issued option_id from the pending request resolves.
    // Returns the server-issued label for the caller to send as the resume
    // reply (the suspended turn auto-resumes on this thread); null refuses
    // fabricated identity.
    const pending = pendingClarificationRef.current;
    if (!pending) return null;
    const match = pending.options.find((o) => o.option_id === optionId);
    return match ? match.label : null;
  }, []);

  return {
    status,
    streamingContent,
    thinkingText,
    pendingSources,
    pendingImages,
    pendingPeople,
    pendingCitations,
    pendingClarification,
    streamCompleteTick,
    error,
    isStreaming,
    agentSteps,
    potentialAbbreviations,
    aiMessageId,
    userMessageId,
    sendMessage,
    cancel,
    reset,
    submitClarification,
  };
}
