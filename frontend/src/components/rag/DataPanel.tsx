import { useState, useMemo, useCallback, memo } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { useTranslation } from "@/hooks/useTranslation";
import { AnimatePresence } from "framer-motion";
import {
  FileText,
  Sparkles,
  Loader2,
  ArrowLeft,
  Check,
  X,
  Pencil,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { UploadZone } from "./UploadZone";
import { StatsBar } from "./StatsBar";
import { DocumentFilters, type FilterStatus } from "./DocumentFilters";
import { DocumentCard } from "./DocumentCard";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Document, RAGStats, DocumentStatus } from "@/types";

const PROCESSING_STATUSES = new Set<DocumentStatus>([
  "parsing",
  "ocring",
  "chunking",
  "embedding",
  "building_kg",
]);
const PROCESSABLE_STATUSES = new Set<DocumentStatus>(["pending", "failed"]);

interface DataPanelProps {
  workspace: { id: number; name: string; description?: string | null } | undefined;
  documents: Document[] | undefined;
  docsLoading: boolean;
  ragStats: RAGStats | undefined;
  selectedDocId: number | null;
  onSelectDoc: (doc: Document) => void;
  onUpload: (file: File) => void;
  isUploading: boolean;
  onDelete: (id: number) => void;
  onProcess: (id: number) => void;
  isProcessing: boolean;
  onUpdateWorkspace: (data: { name: string; description?: string }) => Promise<void>;
}

export const DataPanel = memo(function DataPanel({
  workspace,
  documents,
  docsLoading,
  ragStats,
  selectedDocId,
  onSelectDoc,
  onUpload,
  isUploading,
  onDelete,
  onProcess,
  isProcessing,
  onUpdateWorkspace,
}: DataPanelProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [deleteDocConfirm, setDeleteDocConfirm] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<FilterStatus>("all");
  const [isEditingName, setIsEditingName] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [batchProcessing, setBatchProcessing] = useState(false);

  const processingCount = useMemo(
    () => documents?.filter((d) => PROCESSING_STATUSES.has(d.status)).length ?? 0,
    [documents]
  );

  const pendingCount = useMemo(
    () => documents?.filter((d) => PROCESSABLE_STATUSES.has(d.status)).length ?? 0,
    [documents]
  );

  const filteredDocs = useMemo(() => {
    if (!documents) return [];
    let result = documents;
    if (statusFilter !== "all") {
      if (statusFilter === "parsing") {
        result = result.filter((d) => PROCESSING_STATUSES.has(d.status));
      } else {        result = result.filter((d) => d.status === statusFilter);
      }
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter((d) =>
        d.original_filename.toLowerCase().includes(q)
      );
    }
    return result;
  }, [documents, statusFilter, searchQuery]);

  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = { all: 0 };
    documents?.forEach((d) => {
      counts.all = (counts.all || 0) + 1;
      counts[d.status] = (counts[d.status] || 0) + 1;
    });
    return counts as Record<FilterStatus, number>;
  }, [documents]);

  const handleBatchProcess = useCallback(async () => {
    if (!documents || batchProcessing) return;
    const processable = documents.filter((d) => PROCESSABLE_STATUSES.has(d.status));
    if (processable.length === 0) return;

    setBatchProcessing(true);
    const count = processable.length;
    toast.info(t("workspace.analyzing_batch", { count }), {
      description: t("workspace.analyzing_batch_desc"),
    });

    try {
      await api.post("/rag/process-batch", {
        document_ids: processable.map((d) => d.id),
      });
    } catch {
      toast.error(t("workspace.batch_failed"));
    } finally {
      setBatchProcessing(false);
    }
  }, [documents, batchProcessing, t]);

  const handleStartEdit = () => {
    if (workspace) {
      setEditName(workspace.name);
      setEditDesc(workspace.description || "");
      setIsEditingName(true);
    }
  };

  const handleSaveEdit = async () => {
    if (!editName.trim()) return;
    await onUpdateWorkspace({
      name: editName.trim(),
      description: editDesc.trim() || undefined,
    });
    setIsEditingName(false);
  };

  return (
    <div className="h-full flex flex-col border-r overflow-hidden">
      {/* Header — workspace name */}
      <div className="flex-shrink-0 px-3 pt-3 pb-2 border-b space-y-1.5">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-3 h-3" />
          {t("nav.dashboard")}
        </button>

        {isEditingName ? (
          <div className="space-y-1.5">
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSaveEdit()}
              placeholder={t("common.name")}
              autoFocus
              className="text-sm font-semibold h-8"
            />
            <Input
              value={editDesc}
              onChange={(e) => setEditDesc(e.target.value)}
              placeholder={t("common.description")}
              className="text-xs h-7"
            />
            <div className="flex items-center gap-1">
              <Button size="sm" onClick={handleSaveEdit} disabled={!editName.trim()} className="h-6 text-[10px] px-2">
                <Check className="w-3 h-3 mr-0.5" /> {t("common.save")}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setIsEditingName(false)} className="h-6 text-[10px] px-2">
                <X className="w-3 h-3 mr-0.5" /> {t("common.cancel")}
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-1.5">
            <div className="flex-1 min-w-0">
              <h1 className="text-sm font-bold truncate">
                {workspace?.name || t("nav.knowledge_bases")}
              </h1>
              {workspace?.description && (
                <p className="text-[10px] text-muted-foreground truncate">
                  {workspace.description}
                </p>
              )}
            </div>
            <Button
              size="icon"
              variant="ghost"
              onClick={handleStartEdit}
              className="h-6 w-6 flex-shrink-0"
            >
              <Pencil className="w-3 h-3" />
            </Button>
          </div>
        )}
      </div>

      {/* Upload strip — compact single row */}
      <div className="flex-shrink-0 px-3 pt-2 pb-1.5">
        <UploadZone onUpload={onUpload} isUploading={isUploading} compact />
      </div>

      {/* Stats bar */}
      <div className="flex-shrink-0 px-3 py-1.5 border-b space-y-1.5">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold flex items-center gap-1.5">
            <FileText className="w-3.5 h-3.5" />
            {t("nav.files")}
          </h2>
          <div className="flex items-center gap-2">
            {workspace && (
              <button
                onClick={() => navigate(`/knowledge-bases/${workspace.id}/files`)}
                className="text-[10px] text-primary hover:underline transition-colors"
              >
                {t("common.view_all")} &rarr;
              </button>
            )}
            <span className="text-[10px] text-muted-foreground">
              {t("kb.docs_count", { count: documents?.length ?? 0 })}
            </span>
          </div>
        </div>
        <StatsBar stats={ragStats} processingCount={processingCount} />

        {/* Analyze All banner — compact for narrow panel */}
        {pendingCount > 0 && (
          <button
            onClick={handleBatchProcess}
            disabled={batchProcessing || processingCount > 0}
            className={cn(
              "w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-md",
              "border border-blue-400/20 bg-blue-400/[0.06]",
              "hover:bg-blue-400/10 transition-colors",
              (batchProcessing || processingCount > 0) && "opacity-50 pointer-events-none",
            )}
          >
            <div className="flex items-center gap-2 min-w-0">
              <Sparkles className={cn("w-3.5 h-3.5 text-blue-400 flex-shrink-0", batchProcessing && "animate-spin")} />
              <span className="text-[11px] font-medium text-blue-400 truncate">
                {batchProcessing ? t("common.starting") : `${t("workspace.analyze_all")} (${pendingCount})`}
              </span>
            </div>
            <span className="text-[10px] text-muted-foreground flex-shrink-0">
              {t("common.pending_count", { count: pendingCount })}
            </span>
          </button>
        )}
      </div>

      {/* Document list — ~80% */}
      <div className="flex-1 overflow-hidden flex flex-col">
        {docsLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-4 h-4 animate-spin text-muted-foreground mr-2" />
            <span className="text-xs text-muted-foreground">{t("common.loading")}</span>
          </div>
        ) : !documents || documents.length === 0 ? (
          <div className="flex-1 flex items-center justify-center px-3">
            <p className="text-xs text-muted-foreground text-center">
              {t("workspace.no_docs")}
            </p>
          </div>
        ) : (
          <>
            <div className="px-3 pt-2 flex-shrink-0">
              <DocumentFilters
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
                statusFilter={statusFilter}
                onStatusChange={setStatusFilter}
                counts={statusCounts}
              />
            </div>

            <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5">
              <AnimatePresence mode="popLayout">
                {filteredDocs.map((doc) => (
                  <DocumentCard
                    key={doc.id}
                    doc={doc}
                    selected={doc.id === selectedDocId}
                    onDelete={(id) => setDeleteDocConfirm(id)}
                    onProcess={onProcess}
                    isProcessing={isProcessing}
                    onClick={onSelectDoc}
                  />
                ))}
              </AnimatePresence>
              {filteredDocs.length === 0 && documents.length > 0 && (
                <div className="text-center py-4 text-[11px] text-muted-foreground">
                  {t("workspace.no_match")}
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* Delete confirmation */}
      <ConfirmDialog
        open={deleteDocConfirm !== null}
        onConfirm={async () => {
          if (deleteDocConfirm !== null) {
            onDelete(deleteDocConfirm);
            setDeleteDocConfirm(null);
          }
        }}
        onCancel={() => setDeleteDocConfirm(null)}
        title={t("files.delete_confirm_title")}
        message={t("files.delete_confirm_msg")}
        confirmLabel={t("common.delete")}
        variant="danger"
      />
    </div>
  );
});
