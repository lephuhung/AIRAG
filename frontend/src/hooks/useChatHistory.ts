import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { MutableRefObject } from "react";
import { api } from "@/lib/api";
import type { ChatHistoryResponse, SessionDocumentsResponse } from "@/types";

export function useChatHistory(
  sessionId: string | null,
  opts?: {
    /** When this ref is true, history polling pauses — used to avoid refetching
     *  over an in-flight web stream. */
    isStreamingRef?: MutableRefObject<boolean>;
    /** Poll interval in ms (default 10s). */
    pollMs?: number;
  },
) {
  return useQuery({
    queryKey: ["chat-history", sessionId],
    queryFn: () =>
      api.get<ChatHistoryResponse>(`/rag/chat/sessions/${sessionId}/history`),
    enabled: !!sessionId,
    // Poll so messages added on another channel (e.g. Telegram) appear in the
    // open session. Pause while a web stream is in flight so we don't refetch
    // the half-written turn over the live one. Read the ref at tick time.
    refetchInterval: () =>
      opts?.isStreamingRef?.current ? false : (opts?.pollMs ?? 10000),
    refetchOnWindowFocus: () => !opts?.isStreamingRef?.current,
    staleTime: 5000,
  });
}

export function useClearChatHistory(sessionId: string | null) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: () => api.delete(`/rag/chat/sessions/${sessionId}/history`),
    onSuccess: () => {
      queryClient.setQueryData<ChatHistoryResponse>(
        ["chat-history", sessionId],
        {
          session_id: sessionId ?? undefined,
          messages: [],
          total: 0,
        },
      );
    },
  });
}

export function useSessionDocuments(sessionId: string | null) {
  return useQuery({
    queryKey: ["session-documents", sessionId],
    queryFn: () =>
      api.get<SessionDocumentsResponse>(`/rag/chat/sessions/${sessionId}/documents`),
    enabled: !!sessionId,
    staleTime: 30000, // Cache for 30 seconds
  });
}
