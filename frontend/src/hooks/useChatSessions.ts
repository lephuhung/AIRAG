import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatSession } from "@/types";

export function useChatSessions() {
  return useQuery({
    queryKey: ["chat-sessions"],
    queryFn: () => api.get<ChatSession[]>("/rag/chat/sessions"),
    // Auto-refresh so sessions started/updated on another channel (e.g. the
    // Telegram bot) surface in the sidebar without a manual reload. The list is
    // ordered by updated_at desc, so an active Telegram chat climbs to the top.
    refetchInterval: 10000,
    refetchOnWindowFocus: true,
  });
}

export function useUpdateSessionTitle() {
  const queryClient = useQueryClient();

  return (sessionId: string, title: string) => {
    queryClient.setQueryData<ChatSession[]>(
      ["chat-sessions"],
      (old) =>
        old?.map((s) =>
          String(s.id) === sessionId ? { ...s, title } : s,
        ),
    );
  };
}

export function useCreateChatSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: { title: string }) =>
      api.post<ChatSession>("/rag/chat/sessions", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
    },
  });
}

export function useDeleteChatSession() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (sessionId: string) =>
      api.delete(`/rag/chat/sessions/${sessionId}`),
    onSuccess: (_, sessionId) => {
      queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      queryClient.removeQueries({ queryKey: ["chat-history", sessionId] });
    },
  });
}
