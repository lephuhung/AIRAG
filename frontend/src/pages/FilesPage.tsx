import { useState, useMemo, useCallback, useEffect } from "react";
import { useTranslation } from "@/hooks/useTranslation";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Search,
  FolderOpen,
  ArrowUpDown,
  Database,
  X,
  ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FileCard } from "@/components/rag/FileCard";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useDocuments, useDeleteDocument, useProcessDocument, useReindexDocument, PROCESSING_STATUSES } from "@/hooks/useDocuments";
import { useWorkspaces, useWorkspace } from "@/hooks/useWorkspaces";
import { DocumentViewer } from "@/components/rag/DocumentViewer";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { Document, RAGStats } from "@/types";

// ---------------------------------------------------------------------------
// Filter tabs
// ---------------------------------------------------------------------------
type FilterTab = "all" | "indexed" | "processing" | "failed" | "pending";

const FILTER_TABS: { value: FilterTab; label: string; key: string }[] = [
  { value: "all", label: "All", key: "common.all" },
  { value: "indexed", label: "Indexed", key: "files.tabs.indexed" },
  { value: "processing", label: "Processing", key: "files.tabs.processing" },
  { value: "failed", label: "Failed", key: "files.tabs.failed" },
  { value: "pending", label: "Pending", key: "files.tabs.pending" },
];

// ---------------------------------------------------------------------------
// Sort options
// ---------------------------------------------------------------------------
type SortKey = "newest" | "oldest" | "name" | "size";

const SORT_OPTIONS: { value: SortKey; label: string; key: string }[] = [
  { value: "newest", label: "Newest first", key: "files.sort.newest" },
  { value: "oldest", label: "Oldest first", key: "files.sort.oldest" },
  { value: "name", label: "Name A-Z", key: "files.sort.name" },
  { value: "size", label: "Size", key: "files.sort.size" },
];

function sortDocs(docs: Document[], key: SortKey): Document[] {
  const sorted = [...docs];
  switch (key) {
    case "newest":
      return sorted.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    case "oldest":
      return sorted.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
    case "name":
      return sorted.sort((a, b) => a.original_filename.localeCompare(b.original_filename));
    case "size":
      return sorted.sort((a, b) => b.file_size - a.file_size);
    default:
      return sorted;
  }
}


// ---------------------------------------------------------------------------
// Workspace Selector — grid of workspaces
// ---------------------------------------------------------------------------
function WorkspaceSelector({
  onSelect,
}: {
  onSelect: (wsId: number) => void;
}) {
  const { t } = useTranslation();
  const { data: workspaces, isLoading } = useWorkspaces();
  const [wsSearch, setWsSearch] = useState("");

  const filtered = useMemo(() => {
    if (!workspaces) return [];
    if (!wsSearch.trim()) return workspaces;
    const q = wsSearch.toLowerCase();
    return workspaces.filter(
      (ws) =>
        ws.name.toLowerCase().includes(q) ||
        (ws.description && ws.description.toLowerCase().includes(q)),
    );
  }, [workspaces, wsSearch]);

  const navigate = useNavigate();

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 border-b px-6 py-4">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
          <button onClick={() => navigate("/")} className="hover:text-foreground transition-colors">
            {t("nav.dashboard")}
          </button>
          <span>/</span>
          <span className="text-foreground font-medium">{t("files.title")}</span>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-primary" />
              {t("files.title")}
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              {t("files.select_workspace")}
            </p>
          </div>
        </div>
      </div>

      {/* Search */}
      <div className="flex-shrink-0 px-6 py-3 border-b">
        <div className="relative max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
          <Input
            placeholder={t("files.search_ws")}
            value={wsSearch}
            onChange={(e) => setWsSearch(e.target.value)}
            className="pl-9 h-9"
          />
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="rounded-xl border bg-card animate-pulse p-5 h-28" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Database className="w-10 h-10 text-muted-foreground/30 mb-3" />
            <h3 className="text-sm font-medium text-muted-foreground mb-1">
              {workspaces && workspaces.length > 0 ? t("files.no_ws_match") : t("files.no_ws_yet")}
            </h3>
            <p className="text-xs text-muted-foreground/70">
              {workspaces && workspaces.length > 0
                ? t("files.search_ws_hint")
                : t("files.create_ws_hint")}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filtered.map((ws) => (
              <button
                key={ws.id}
                onClick={() => onSelect(ws.id)}
                className="group rounded-xl border bg-card p-5 text-left transition-all hover:shadow-lg hover:-translate-y-0.5 hover:border-primary/30"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <Database className="w-4 h-4 text-primary" />
                    </div>
                    <div className="min-w-0">
                      <p className="font-medium text-sm truncate">{ws.name}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {t("kb.docs_count", { count: ws.document_count })}
                        {" · "}
                        {t("kb.indexed_count", { count: ws.indexed_count })}
                      </p>
                    </div>
                  </div>
                  <ChevronRight className="w-4 h-4 text-muted-foreground/40 group-hover:text-primary transition-colors flex-shrink-0 mt-2" />
                </div>
                {ws.description && (
                  <p className="text-xs text-muted-foreground/70 mt-2 line-clamp-2">
                    {ws.description}
                  </p>
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// FilesPage
// ---------------------------------------------------------------------------
export function FilesPage() {
  const { t } = useTranslation();
  const { workspaceId: urlWorkspaceId } = useParams<{ workspaceId: string }>();
  const navigate = useNavigate();

  // Internal state for selected workspace (when accessed via /files)
  const [selectedWsId, setSelectedWsId] = useState<string | undefined>(urlWorkspaceId);

  // Sync from URL when navigating between routes
  useEffect(() => {
    if (urlWorkspaceId) setSelectedWsId(urlWorkspaceId);
  }, [urlWorkspaceId]);

  const workspaceId = selectedWsId;
  const wsId = workspaceId ? Number(workspaceId) : null;

  // Data
  const { data: workspace } = useWorkspace(wsId);
  const { data: documents, isLoading: docsLoading } = useDocuments(workspaceId);
  const { data: ragStats } = useQuery({
    queryKey: ["rag-stats", workspaceId],
    queryFn: () => api.get<RAGStats>(`/rag/stats/${workspaceId}`),
    enabled: !!workspaceId,
  });

  // Mutations
  const deleteDoc = useDeleteDocument(workspaceId);
  const processDoc = useProcessDocument(workspaceId);
  const reindexDoc = useReindexDocument(workspaceId);

  // UI state
  const [searchQuery, setSearchQuery] = useState("");
  const [filterTab, setFilterTab] = useState<FilterTab>("all");
  const [sortKey, setSortKey] = useState<SortKey>("newest");
  const [deleteDocConfirm, setDeleteDocConfirm] = useState<number | null>(null);
  const [sortMenuOpen, setSortMenuOpen] = useState(false);

  // -- Store (for unified preview) --
  const {
    selectedDoc,
    selectDoc,
    reset: resetStore,
    highlightChunks,
    scrollToPage,
    scrollToHeading,
    scrollToImageSrc,
    clearScrollTarget,
  } = useWorkspaceStore();

  // Reset store when navigating away or switching workspace
  useEffect(() => {
    return () => resetStore();
  }, [workspaceId, resetStore]);

  // Filter + sort
  const filteredDocs = useMemo(() => {
    if (!documents) return [];
    let result = documents;

    // Status filter
    if (filterTab === "indexed") {
      result = result.filter((d) => d.status === "indexed");
    } else if (filterTab === "processing") {
      result = result.filter((d) => PROCESSING_STATUSES.has(d.status));
    } else if (filterTab === "failed") {
      result = result.filter((d) => d.status === "failed");
    } else if (filterTab === "pending") {
      result = result.filter((d) => d.status === "pending");
    }

    // Search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter((d) =>
        d.original_filename.toLowerCase().includes(q),
      );
    }

    return sortDocs(result, sortKey);
  }, [documents, filterTab, searchQuery, sortKey]);

  // Tab counts
  const tabCounts = useMemo(() => {
    const counts: Record<FilterTab, number> = { all: 0, indexed: 0, processing: 0, failed: 0, pending: 0 };
    documents?.forEach((d) => {
      counts.all++;
      if (d.status === "indexed") counts.indexed++;
      else if (PROCESSING_STATUSES.has(d.status)) counts.processing++;
      else if (d.status === "failed") counts.failed++;
      else if (d.status === "pending") counts.pending++;
    });
    return counts;
  }, [documents]);

  // Download handler
  const handleDownload = useCallback((doc: Document) => {
    api.downloadFile(`/documents/${doc.id}/download`, doc.original_filename);
  }, []);

  // Workspace selection handler
  const handleSelectWorkspace = useCallback((wsIdNum: number) => {
    setSelectedWsId(String(wsIdNum));
    // Reset filters when switching workspace
    setSearchQuery("");
    setFilterTab("all");
  }, []);

  // Back to workspace list
  const handleBackToList = useCallback(() => {
    setSelectedWsId(undefined);
    // If we came from a URL with workspaceId, navigate to /files
    if (urlWorkspaceId) {
      navigate("/files");
    }
  }, [urlWorkspaceId, navigate]);

  // ── Show workspace selector when no workspace is selected ──
  if (!workspaceId) {
    return <WorkspaceSelector onSelect={handleSelectWorkspace} />;
  }

  // ── Show files for selected workspace ──
  return (
    <div className="h-full flex overflow-hidden relative">
      {/* Left side: Search, Filters, Grid */}
      <motion.div 
        layout
        initial={false}
        className={cn(
          "flex-1 flex flex-col h-full min-w-0 relative z-10 transition-all duration-300 ease-in-out",
          selectedDoc ? "hidden md:flex md:w-[35%] lg:w-[30%] xl:w-[25%]" : "w-full"
        )}
        transition={{ 
          type: "spring", 
          stiffness: 300, 
          damping: 34,
          mass: 0.8
        }}
      >
        {/* ── Header ── */}
      <div className={cn(
        "flex-shrink-0 border-b transition-all duration-300",
        selectedDoc ? "px-4 py-3" : "px-6 py-4"
      )}>
        {/* Breadcrumb */}
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
          <button
            onClick={handleBackToList}
            className="hover:text-foreground transition-colors underline-offset-4 hover:underline"
          >
            {t("files.title")}
          </button>
          <span>/</span>
          <span className="text-foreground font-medium truncate max-w-[200px]">
            {workspace?.name || t("common.pending")}
          </span>
        </div>

        {/* Title */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className={cn(
              "font-bold flex items-center gap-2 transition-all",
              selectedDoc ? "text-base" : "text-lg"
            )}>
              <FolderOpen className={cn("text-primary", selectedDoc ? "w-4 h-4" : "w-5 h-5")} />
              {workspace?.name || t("files.title")}
            </h1>
            <p className="text-xs text-muted-foreground">
              {t("kb.docs_count", { count: documents?.length ?? 0 })}
              {ragStats && ` \u00b7 ${ragStats.total_chunks} chunks`}
            </p>
          </div>
        </div>
      </div>

      {/* ── Toolbar ── */}
      <div className={cn(
        "flex-shrink-0 border-b flex items-center gap-3 flex-wrap transition-all duration-300",
        selectedDoc ? "px-4 py-2" : "px-6 py-3"
      )}>
        {/* Search */}
        <div className={cn(
          "relative flex-1 min-w-[200px] transition-all",
          selectedDoc ? "max-w-[240px]" : "max-w-sm"
        )}>
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
          <Input
            placeholder={t("files.search_files")}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        {/* Filter tabs */}
        <div className="flex items-center gap-1 bg-muted/40 rounded-lg p-0.5">
          {FILTER_TABS.map((tab) => {
            const isActive = filterTab === tab.value;
            const count = tabCounts[tab.value];
            return (
              <button
                key={tab.value}
                onClick={() => setFilterTab(tab.value)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-md transition-colors",
                  isActive
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {t(tab.key)}
                {count > 0 && (
                  <span
                    className={cn(
                      "ml-1 text-[10px]",
                      isActive ? "text-primary" : "text-muted-foreground/60",
                    )}
                  >
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Sort dropdown */}
        <div className="relative">
          <Button
            variant="outline"
            size="sm"
            className="h-9 gap-1.5 text-xs"
            onClick={() => setSortMenuOpen((v) => !v)}
          >
            <ArrowUpDown className="w-3.5 h-3.5" />
            {t(SORT_OPTIONS.find((o) => o.value === sortKey)?.key || "")}
          </Button>
          {sortMenuOpen && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setSortMenuOpen(false)} />
              <div className="absolute right-0 top-10 z-20 min-w-[150px] rounded-lg border bg-popover shadow-lg py-1">
                {SORT_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    className={cn(
                      "w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 transition-colors",
                      sortKey === opt.value && "text-primary font-medium",
                    )}
                    onClick={() => {
                      setSortKey(opt.value);
                      setSortMenuOpen(false);
                    }}
                  >
                    {t(opt.key)}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {/* ── Grid content ── */}
      <div className="flex-1 overflow-y-auto px-6 py-5">
        {docsLoading ? (
          // Loading skeletons
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5">
            {Array.from({ length: 8 }).map((_, i) => (
              <div
                key={i}
                className="rounded-xl border bg-card animate-pulse"
              >
                <div className="px-4 pt-4 pb-3 flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-muted" />
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-muted rounded w-3/4" />
                    <div className="h-3 bg-muted rounded w-1/2" />
                  </div>
                </div>
                <div className="px-4 pb-3">
                  <div className="h-5 bg-muted rounded w-20" />
                </div>
                <div className="px-4 pb-3 grid grid-cols-2 gap-2">
                  <div className="h-3 bg-muted rounded" />
                  <div className="h-3 bg-muted rounded" />
                </div>
                <div className="px-4 pb-3">
                  <div className="h-3 bg-muted rounded w-2/3" />
                </div>
              </div>
            ))}
          </div>
        ) : !documents || documents.length === 0 ? (
          // Empty state
          <div className="flex flex-col items-center justify-center h-full text-center">
            <FolderOpen className="w-12 h-12 text-muted-foreground/30 mb-3" />
            <h3 className="text-sm font-medium text-muted-foreground mb-1">
              {t("files.no_files_yet")}
            </h3>
            <p className="text-xs text-muted-foreground/70">
              {t("files.upload_hint")}
            </p>
          </div>
        ) : filteredDocs.length === 0 ? (
          // No results state
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Search className="w-10 h-10 text-muted-foreground/30 mb-3" />
            <h3 className="text-sm font-medium text-muted-foreground mb-1">
              {t("files.no_files_match")}
            </h3>
            <p className="text-xs text-muted-foreground/70">
              {t("files.adjust_search_hint")}
            </p>
          </div>
        ) : (
          // File grid
          <div className={cn(
            "grid gap-5 transition-all duration-300",
            selectedDoc 
              ? "grid-cols-1 lg:grid-cols-2" 
              : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
          )}>
            <AnimatePresence mode="popLayout">
              {filteredDocs.map((doc) => (
                <FileCard
                  key={doc.id}
                  doc={doc}
                  onDelete={setDeleteDocConfirm}
                  onReindex={(id) => reindexDoc.mutate(id)}
                  onProcess={(id) => processDoc.mutate(id)}
                  onDownload={handleDownload}
                  onPreview={selectDoc}
                  isProcessing={processDoc.isPending}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Delete confirmation */}
      <ConfirmDialog
        open={deleteDocConfirm !== null}
        onConfirm={async () => {
          if (deleteDocConfirm !== null) {
            deleteDoc.mutate(deleteDocConfirm);
            setDeleteDocConfirm(null);
          }
        }}
        onCancel={() => setDeleteDocConfirm(null)}
        title={t("files.delete_confirm_title")}
        message={t("files.delete_confirm_msg")}
        confirmLabel={t("common.delete")}
        variant="danger"
      />
      </motion.div>

      {/* Right side: Document Viewer (conditionally rendered) */}
      <AnimatePresence mode="popLayout" initial={false}>
        {selectedDoc && (
          <motion.div 
            key={selectedDoc.id}
            initial={{ x: "100%", opacity: 0.5 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: "100%", opacity: 0, transition: { duration: 0.2, ease: "easeInOut" } }}
            transition={{ 
              type: "spring", 
              stiffness: 350, 
              damping: 38,
              mass: 0.8
            }}
            className="w-full md:w-[65%] lg:w-[70%] xl:w-[75%] h-full border-l bg-background flex flex-col z-20 shadow-2xl relative"
          >
            <div className="absolute inset-y-0 -left-6 w-6 bg-gradient-to-r from-transparent to-black/[0.03] pointer-events-none" />
            
            {/* Header with close button */}
            <div className="h-10 border-b flex items-center justify-between px-3 bg-muted/20 shrink-0">
              <span className="text-xs font-semibold truncate text-foreground/70">
                {t("files.preview")}
              </span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => selectDoc(null)}
                className="h-7 w-7 rounded-full text-muted-foreground transition-all duration-200 hover:rotate-90 active:scale-95"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
            
            {/* The actual viewer, taking remaining height */}
            <div className="flex-1 overflow-hidden relative">
              <DocumentViewer
                doc={selectedDoc}
                highlightChunks={highlightChunks}
                scrollToPage={scrollToPage}
                scrollToHeading={scrollToHeading}
                scrollToImageSrc={scrollToImageSrc}
                onScrolled={clearScrollTarget}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
