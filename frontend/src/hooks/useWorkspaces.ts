import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  KnowledgeBase,
  CreateWorkspace,
  UpdateWorkspace,
  WorkspaceSummary,
} from "@/types";

export function useWorkspaces(options?: { pollWhileIndexing?: boolean }) {
  return useQuery({
    queryKey: ["workspaces"],
    queryFn: () => api.get<KnowledgeBase[]>("/workspaces"),
    // While any workspace still has documents being indexed (indexed_count <
    // document_count), poll so the doc/indexed counts update live and flip to
    // "indexed" without a manual refresh. Stops once everything is indexed.
    // (Deletes/uploads already invalidate ["workspaces"] via useDocuments.)
    refetchInterval: options?.pollWhileIndexing
      ? (query) => {
          const data = query.state.data as KnowledgeBase[] | undefined;
          const indexing = data?.some(
            (ws) => (ws.indexed_count ?? 0) < (ws.document_count ?? 0),
          );
          return indexing ? 4000 : false;
        }
      : undefined,
  });
}

export function useWorkspace(workspaceId: string | null) {
  return useQuery({
    queryKey: ["workspaces", workspaceId],
    queryFn: () => api.get<KnowledgeBase>(`/workspaces/${workspaceId}`),
    enabled: !!workspaceId,
  });
}

export function useWorkspaceSummaries() {
  return useQuery({
    queryKey: ["workspaces", "summary"],
    queryFn: () => api.get<WorkspaceSummary[]>("/workspaces/summary"),
  });
}

export function useCreateWorkspace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateWorkspace) =>
      api.post<KnowledgeBase>("/workspaces", data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}

export function useUpdateWorkspace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateWorkspace }) =>
      api.put<KnowledgeBase>(`/workspaces/${id}`, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
      queryClient.invalidateQueries({ queryKey: ["workspaces", variables.id] });
    },
  });
}

export function useDeleteWorkspace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => api.delete(`/workspaces/${id}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}

export function useSetDefaultWorkspace() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) =>
      api.post<KnowledgeBase>(`/workspaces/${id}/set-default`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
    },
  });
}
