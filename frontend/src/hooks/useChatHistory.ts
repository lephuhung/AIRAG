import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatHistoryResponse } from "@/types";

export function useChatHistory(sessionId: number | null) {
  return useQuery({
    queryKey: ["chat-history", sessionId],
    queryFn: () =>
      api.get<ChatHistoryResponse>(`/rag/chat/sessions/${sessionId}/history`),
    enabled: !!sessionId,
    staleTime: Infinity, // Don't auto-refetch — we invalidate manually after chat
  });
}

export function useClearChatHistory(sessionId: number | null) {
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
