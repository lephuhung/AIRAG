# Public chat SSE event inventory (frontend browser boundary)

Source of truth for what the browser accepts from the chat stream. Wire
framing is `event: <type>\ndata: <json>\n\n` per dispatch; data-only frames
are ignored by `useRAGChatStream` (never misattributed). Unknown event
types are ignored (forward-compatible). Malformed JSON frames are skipped.

Backend producers: `backend/app/services/agent/streaming.py`
(`stream_agent_events` = v1 arm, `stream_v2_turn_events` /
`stream_v2_turn_to_sse` = v2 arm). Parser/hook: `frontend/src/hooks/useRAGChatStream.ts`.
Mount: `ChatPanel` (`frontend/src/components/rag/ChatPanel.tsx`) on
`ChatPage` (`/chat`, `/chat/:sessionId`).

| SSE event | Backend DTO (data payload) | Hook transition | Visible UI outcome | Reload behavior | Automated test |
|---|---|---|---|---|---|
| `status` | `{step, detail?, phase?, abbreviations?}` steps: analyzing/searching/retrieving/retrieved/generating/rollback; phases: clarifying/planning (Task 9, no chain-of-thought) | status state + `agentSteps` timeline; `phase=clarifying` → status clarifying; `step=rollback` clears speculative content + retractable artifacts incl. citations | ThinkingTimeline steps; status label | ephemeral (timeline rebuilt from persisted `agent_steps`) | contract test (status phases); rollback integration |
| `ai_message_id` | `{message_id}` server-assigned assistant id | `aiMessageId` state + local | final message `id` | persisted as message id | contract/v2 additive test |
| `user_id` | `{id}` server-assigned user id | `userMessageId` state + local | — (identity plumbing) | persisted as message id | untested (code-reading only; no dedicated assertion) |
| `thinking` | `{text}` | rAF-buffered `thinkingText` + analyzing step | thinking panel while generating; cleared on complete | persisted `thinking` re-renders collapsed | contract test |
| `token` | `{text}` (or `{type,text}`) | rAF-buffered `streamingContent`; `</think>` delimiter safety net reroutes leaked thinking | streamed answer text | persisted answer content | e2e token→complete |
| `sources` | `{sources: ChatSourceChunk[]}` (`document_id`, `chunk_id`, `index`, …) | `pendingSources` + `sources_found` step badges | Sources panel / citation fallback links | persisted `sources` re-render | rollback control; e2e citation reload |
| `images` | `{image_refs: ChatImageRef[]}` (also `images` legacy key in tests) | `pendingImages`; augments `sources_found` step | ImageGallery | persisted `imageRefs` | rollback integration |
| `people_data` | `{people: PeopleRecord[]}` | `pendingPeople` (+ survives `reset()` for card display) | people cards | persisted `peopleData` | contract people test; rollback |
| `potential_abbreviations` | `{abbreviations: string[]}` (+ `status.abbreviations` advisory) | `potentialAbbreviations` | abbreviation modal prompt | ephemeral | contract abbreviations test |
| `citation` (Task 9 public) | `{citations: PublicCitation[]}` (`citation_id`, `document_id`, `chunk_id`, `index`, provenance, rank) + optional `image_refs`, `people` | `pendingCitations`; locatable citations projected into sources model preserving `index`/provenance/score (never synthesized) | citation markers `[n]` + sources; final message `citations` metadata | persisted `citations` → reload-safe (history `GET …/history`) | contract citation tests; e2e reload |
| `clarification` / `clarification_required` | `{clarification_id, reason, question, options[{option_id,label} or strings], resume{thread_id,message_id?}, context?}` — all ids server-issued | `pendingClarification`; status clarifying | `ClarificationBanner` (`data-testid="clarification-options"`) question + option buttons | persisted `clarification` block rebuilds banner with resume | contract clarification tests; e2e resume selection |
| `clarification_resolved` | `{clarification_id}` | clears `pendingClarification` | banner dismissed, stream continues | — (terminal turn persists answer) | streaming tests |
| `token_rollback` / status-rollback | `{}` | clears token buffer + ALL retractable artifacts (sources/images/people/citations/clarification/abbreviations) | speculative content vanishes; no stale badges | rolled-back turn persists only post-rollback content | Vitest rollback integration (genuine framing) + status-rollback contract tests; e2e asserts a clean post-cancel restart, not rollback frames |
| `complete` | `{answer, thinking?, related_entities?, potential_abbreviations?, citations?, clarification?, status?}` (v2 additive: `status`, `citations`) | builds final `ChatMessage` from locals (sources/citations/clarification), clears loading state, bumps `streamCompleteTick` | assistant bubble finalized; timeline `done` | persisted via history save | contract + rollback + e2e |
| `cancelled` | `{}` server-initiated cancel terminal | status idle, streaming false — quiet, no error banner | stream stops, partial content kept | partial persisted server-side | Vitest quiet-terminal test; e2e covers the stop-button path instead (hanging stream → `POST …/stream/cancel` observed → quiet UI + clean next turn), not the `cancelled` wire event |
| `session_title_updated` | `{Title}` | `onSessionTitleUpdated` callback | sidebar session title | persisted server-side | — (callback seam) |
| `error` | `{message}` (+ v2 typed terminals) | `error` state, status error, timeline marked | `toast.error` banner | not persisted as answer | contract error test; e2e error |
| `heartbeat` (`:comment` lines) | — | skipped at parse (`:` prefix) | none (keeps connection alive) | n/a | untested (code-reading only; covered implicitly by fragmented/data-only frame tests, not by a `:`-comment case) |
| unknown future types | any | ignored (`default:` branch) | none — stream continues | n/a | contract unknown-event test |

## Notes

- `sendMessage(message, history, enableThinking, forceSearch?, overrideSessionId?, documentIds?, clarificationSelection?)` posts
  `/rag/chat/sessions/{id}/stream`; clarification replies send only the
  server-issued `{clarification_id, selected_option_id}` triple built by
  `submitClarification` (fabricated option ids are refused).
- `cancel()` aborts the socket AND posts `…/stream/cancel`; unmount aborts
  the socket only (run is server-detached).
- History reload (`GET …/history` → `ChatHistoryResponse{messages: PersistedChatMessage[]}`)
  restores content/sources/citations/clarification; all `pending*` hook
  state is ephemeral and starts empty.
