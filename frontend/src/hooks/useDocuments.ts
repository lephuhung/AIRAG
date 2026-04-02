/**
 * Reusable React Query hooks for document CRUD operations.
 *
 * Uses the same query keys as WorkspacePage so React Query's cache is shared —
 * navigating between WorkspacePage and FilesPage never re-fetches redundantly.
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { Document, DocumentStatus } from "@/types";

// ---------------------------------------------------------------------------
// Processing status helpers
// ---------------------------------------------------------------------------
export const PROCESSING_STATUSES = new Set<DocumentStatus>([
  "parsing",
  "ocring",
  "chunking",
  "embedding",
  "building_kg",
]);

export function needsPolling(docs: Document[] | undefined): boolean {
  if (!docs) return false;
  return docs.some((d) => PROCESSING_STATUSES.has(d.status));
}

// ---------------------------------------------------------------------------
// Query hook — list documents for a workspace (auto-polls while processing)
// ---------------------------------------------------------------------------
export function useDocuments(workspaceId: string | undefined) {
  return useQuery({
    queryKey: ["documents", workspaceId],
    queryFn: () =>
      api.get<Document[]>(`/documents/workspace/${workspaceId}`),
    enabled: !!workspaceId,
    refetchInterval: (query) => {
      if (needsPolling(query.state.data)) return 3000;
    },
  });
}

// ---------------------------------------------------------------------------
// Query hook — get a single document by ID
// ---------------------------------------------------------------------------
export function useDocument(documentId: string | undefined) {
  return useQuery({
    queryKey: ["documents", "single", String(documentId)],
    queryFn: () => api.get<Document>(`/documents/${documentId}`),
    enabled: !!documentId,
  });
}


// ---------------------------------------------------------------------------
// Mutation hooks
// ---------------------------------------------------------------------------
export function useDeleteDocument(workspaceId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (docId: string) => api.delete(`/documents/${docId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
      toast.success("Document deleted");
    },
    onError: () => toast.error("Failed to delete document"),
  });
}

export function useProcessDocument(workspaceId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (docId: string) => api.post(`/rag/process/${docId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      toast.info("Analyzing document...", {
        description: "Parsing content and building search index.",
      });
    },
    onError: () => toast.error("Failed to start analysis"),
  });
}

export function useReindexDocument(workspaceId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (docId: string) => api.post(`/rag/reindex/${docId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      toast.success("Document re-processing started");
    },
    onError: () => toast.error("Failed to re-process document"),
  });
}

export function useUpdateDocument(workspaceId: string | undefined) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ docId, data }: { docId: string; data: Partial<Document> }) =>
      api.patch(`/documents/${docId}`, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      toast.success("Document updated successfully");
    },
    onError: () => toast.error("Failed to update document metadata"),
  });
}
