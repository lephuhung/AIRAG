import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { WorkerOverview, PipelineDocument } from "@/types";

export function useWorkerOverview() {
  return useQuery({
    queryKey: ["worker-overview"],
    queryFn: () => api.get<WorkerOverview>("/workers/overview"),
    refetchInterval: 5000,
  });
}

export function usePipelineDocuments(workspaceId?: string) {
  return useQuery({
    queryKey: ["worker-pipeline", workspaceId],
    queryFn: () =>
      api.get<{ documents: PipelineDocument[] }>(
        `/workers/pipeline${workspaceId ? `?workspace_id=${workspaceId}` : ""}`
      ),
    refetchInterval: 5000,
  });
}

export function usePurgeQueue() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (queueName: string) =>
      api.post(`/workers/queues/${queueName}/purge`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["worker-overview"] });
    },
  });
}

export function useRetryFailed() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (workspaceId?: number) =>
      api.post<{ retried_count: number }>(
        `/workers/retry-failed${workspaceId ? `?workspace_id=${workspaceId}` : ""}`
      ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["worker-overview"] });
      queryClient.invalidateQueries({ queryKey: ["worker-pipeline"] });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}

export function useRetryDocument() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (documentId: number) =>
      api.post(`/workers/retry-failed/${documentId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["worker-overview"] });
      queryClient.invalidateQueries({ queryKey: ["worker-pipeline"] });
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });
}
