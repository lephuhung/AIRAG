import { useTranslation } from "@/hooks/useTranslation";
import { useMemo, useCallback, useEffect, useRef } from "react";
import { useParams } from "react-router-dom";
import { toast } from "sonner";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { DataPanel } from "@/components/rag/DataPanel";
import { VisualPanel } from "@/components/rag/VisualPanel";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useWorkspace, useUpdateWorkspace } from "@/hooks/useWorkspaces";
import { useUploadDocuments, useReindexDocument } from "@/hooks/useDocuments";
import { api } from "@/lib/api";
import type { Document, RAGStats, DocumentStatus } from "@/types";

const PROCESSING_STATUSES = new Set<DocumentStatus>([
  "parsing",
  "ocring",
  "chunking",
  "embedding",
  "building_kg",
]);

function needsPolling(docs: Document[] | undefined): boolean {
  if (!docs) return false;
  return docs.some((d) => PROCESSING_STATUSES.has(d.status));
}

export function WorkspacePage() {
  const { t } = useTranslation();
  const { workspaceId } = useParams<{ workspaceId: string }>();
  const queryClient = useQueryClient();

  // -- Workspace data --
  const { data: workspace } = useWorkspace(workspaceId ?? null);
  const updateWorkspace = useUpdateWorkspace();

  // -- Uploading (direct-to-MinIO, per-file progress) --
  const { upload, uploadingFiles, isUploading } = useUploadDocuments(workspaceId);

  // -- Store --
  const { selectedDoc, selectDoc, reset: resetStore } = useWorkspaceStore();

  // Reset store when switching between workspaces
  useEffect(() => {
    resetStore();
  }, [workspaceId, resetStore]);

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------
  const { data: documents, isLoading: docsLoading } = useQuery({
    queryKey: ["documents", workspaceId],
    queryFn: () =>
      api.get<Document[]>(`/documents/workspace/${workspaceId}`),
    enabled: !!workspaceId,
    refetchInterval: (query) => {
      if (needsPolling(query.state.data)) return 3000;
      return false;
    },
  });

  const { data: ragStats } = useQuery({
    queryKey: ["rag-stats", workspaceId],
    queryFn: () => api.get<RAGStats>(`/rag/stats/${workspaceId}`),
    enabled: !!workspaceId,
  });

  // Invalidate every query that derives from the workspace's indexed content
  // (chunk stats + the whole KG visual: graph, entities, relationships,
  // analytics). Called when indexing finishes or a document is deleted so the
  // diagram / counts refresh without a manual reload.
  const invalidateDerived = useCallback(() => {
    if (!workspaceId) return;
    for (const key of [
      "rag-stats",
      "kg-graph",
      "kg-entities",
      "kg-relationships",
      "project-analytics",
    ]) {
      queryClient.invalidateQueries({ queryKey: [key, workspaceId] });
    }
  }, [queryClient, workspaceId]);

  // -----------------------------------------------------------------------
  // Refresh derived data when processing finishes
  // -----------------------------------------------------------------------
  const processingCount = useMemo(
    () => documents?.filter((d) => needsPolling([d])).length ?? 0,
    [documents]
  );

  const prevProcessingRef = useRef(processingCount);
  useEffect(() => {
    if (prevProcessingRef.current > 0 && processingCount === 0) {
      invalidateDerived();
    }
    prevProcessingRef.current = processingCount;
  }, [processingCount, invalidateDerived]);

  // Keep selectedDoc in sync with latest document data
  useEffect(() => {
    if (selectedDoc && documents) {
      const updated = documents.find((d) => d.id === selectedDoc.id);
      if (updated && updated.status !== selectedDoc.status) {
        selectDoc(updated);
      }
    }
  }, [documents, selectedDoc, selectDoc]);

  const hasDeepragDocs = (ragStats?.hrag_documents ?? 0) > 0;

  const deleteDoc = useMutation({
    mutationFn: (docId: string) => api.delete(`/documents/${docId}`),
    onSuccess: (_, docId) => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["workspaces"] });
      invalidateDerived();
      if (selectedDoc?.id === docId) selectDoc(null);
      toast.success(t("workspace.delete_success"));
    },
    onError: () => toast.error(t("workspace.delete_failed")),
  });

  const reindexDoc = useReindexDocument(workspaceId);

  const handleDownload = useCallback((doc: Document) => {
    api.downloadFile(`/documents/${doc.id}/download`, doc.original_filename);
  }, []);

  const processDoc = useMutation({
    mutationFn: (docId: string) => api.post(`/rag/process/${docId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
      queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
      toast.info(t("rag.analyzing"), {
        description: t("rag.analyzing_desc"),
      });
    },
    onError: () => toast.error(t("workspace.process_failed")),
  });



  // -----------------------------------------------------------------------
  // Handlers
  // -----------------------------------------------------------------------
  const handleSelectDoc = useCallback(
    (doc: Document) => {
      if (doc.status !== "indexed" && doc.status !== "building_kg") return;
      if (selectedDoc?.id === doc.id) {
        selectDoc(null);
      } else {
        selectDoc(doc);
      }
    },
    [selectedDoc, selectDoc]
  );

  const handleUpdateWorkspace = useCallback(
    async (data: { name: string; description?: string }) => {
      if (!workspaceId) return;
      await updateWorkspace.mutateAsync({ id: workspaceId, data });
    },
    [workspaceId, updateWorkspace]
  );

  // -----------------------------------------------------------------------
  // Render — 2-column layout
  // -----------------------------------------------------------------------
  return (
    <div className="h-full overflow-hidden grid grid-cols-[minmax(350px,30%)_minmax(400px,70%)]">
      {/* Column 1: Data Area */}
      <DataPanel
        workspace={workspace}
        documents={documents}
        docsLoading={docsLoading}
        ragStats={ragStats}
        selectedDocId={selectedDoc?.id ?? null}
        onSelectDoc={handleSelectDoc}
        onUpload={upload}
        uploadingFiles={uploadingFiles}
        isUploading={isUploading}
        onDelete={(id) => deleteDoc.mutate(id)}
        onProcess={(id) => processDoc.mutate(id)}
        onReindex={(id) => reindexDoc.mutate(id)}
        onDownload={handleDownload}
        isProcessing={processDoc.isPending}
        onUpdateWorkspace={handleUpdateWorkspace}
      />

      {/* Column 2: Visual Area */}
      <VisualPanel
        workspaceId={workspaceId || ""}
        hasDeepragDocs={hasDeepragDocs}
      />
    </div>
  );
}
