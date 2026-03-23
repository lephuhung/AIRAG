import { memo, useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useTranslation } from "@/hooks/useTranslation";
import {
  Trash2,
  RefreshCw,
  CheckCircle2,
  Loader2,
  Layers,
  ImageIcon,
  Network,
  FileText,
  MoreHorizontal,
  Download,
  Clock,
  Tag,
  ShieldCheck,
  Eye,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  STATUS_CONFIG,
  getFileConfig,
} from "@/components/rag/DocumentCard";
import { formatFileSize, formatDate, formatProcessingTime } from "@/lib/format";
import type { Document, DocumentStatus } from "@/types";

// ---------------------------------------------------------------------------
// Status badge (reused pattern from DocumentCard)
// ---------------------------------------------------------------------------
function StatusBadge({ status }: { status: DocumentStatus }) {
  const { t } = useTranslation();
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  const Icon = config.icon;
  const isAnimated =
    status === "parsing" ||
    status === "ocring" ||
    status === "chunking" ||
    status === "embedding" ||
    status === "building_kg";

  const showIcon = status !== "indexed";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full",
        config.className,
      )}
    >
      {showIcon && <Icon className={cn("w-3 h-3", isAnimated && "animate-spin")} />}
      {status !== "indexed" && t(config.labelKey)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Sub-task progress pills
// ---------------------------------------------------------------------------
function SubTaskProgress({
  embed_done,
  captions_done,
  kg_done,
}: {
  embed_done?: boolean;
  captions_done?: boolean;
  kg_done?: boolean;
}) {
  const { t } = useTranslation();
  const tasks = [
    { done: embed_done, label: t("files.tasks.embed"), Icon: Layers },
    { done: captions_done, label: t("files.tasks.captions"), Icon: ImageIcon },
    { done: kg_done, label: t("files.tasks.kg"), Icon: Network },
  ];

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {tasks.map(({ done, label, Icon }) => (
        <span
          key={label}
          className={cn(
            "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border",
            done
              ? "bg-green-500/10 text-green-400 border-green-500/20"
              : "bg-amber-400/10 text-amber-400/70 border-amber-400/20",
          )}
        >
          {done ? (
            <CheckCircle2 className="w-2.5 h-2.5" />
          ) : (
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
          )}
          <Icon className="w-2.5 h-2.5" />
          {label}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Active processing statuses
// ---------------------------------------------------------------------------
const ACTIVE_STATUSES: DocumentStatus[] = [
  "parsing",
  "ocring",
  "chunking",
  "embedding",
  "building_kg",
];

// ---------------------------------------------------------------------------
// FileCard component — vertical card for grid display
// ---------------------------------------------------------------------------
interface FileCardProps {
  doc: Document;
  onDelete: (id: number) => void;
  onReindex: (id: number) => void;
  onProcess: (id: number) => void;
  onDownload: (doc: Document) => void;
  onPreview?: (doc: Document) => void;
  isProcessing?: boolean;
}

export const FileCard = memo(function FileCard({
  doc,
  onDelete,
  onReindex,
  onProcess,
  onDownload,
  onPreview,
  isProcessing,
}: FileCardProps) {
  const { t } = useTranslation();
  const fileConfig = getFileConfig(doc.file_type);
  const FileIcon = fileConfig.icon;
  const isActive = ACTIVE_STATUSES.includes(doc.status);
  const showSubTasks =
    doc.status === "chunking" || doc.status === "embedding" || doc.status === "building_kg";

  // Menu state
  const [menuOpen, setMenuOpen] = useState(false);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const close = () => setMenuOpen(false);
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [menuOpen]);

  // Flash animation when user just clicked "Analyze"
  const [justTriggered, setJustTriggered] = useState(false);
  useEffect(() => {
    if (justTriggered) {
      const t = setTimeout(() => setJustTriggered(false), 1200);
      return () => clearTimeout(t);
    }
  }, [justTriggered]);

  const handleProcess = (e: React.MouseEvent) => {
    e.stopPropagation();
    setJustTriggered(true);
    onProcess(doc.id);
  };

  // Stats data
  const stats = [
    { icon: FileText, label: "pages", value: doc.page_count },
    { icon: Layers, label: "chunks", value: doc.chunk_count },
    { icon: ImageIcon, label: "images", value: doc.image_count },
    { icon: FileText, label: "tables", value: doc.table_count },
  ].filter((s) => s.value && s.value > 0);

  const hasSigs = doc.digital_signatures && doc.digital_signatures.length > 0;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 12 }}
      animate={{
        opacity: 1,
        y: 0,
        ...(justTriggered ? { scale: [1, 0.98, 1.01, 1] } : {}),
      }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={justTriggered ? { duration: 0.4 } : undefined}
      className={cn(
        "group relative rounded-xl border bg-card flex flex-col transition-all duration-200",
        isActive
          ? "border-blue-400/50 shadow-[0_0_16px_-3px_rgba(96,165,250,0.3)]"
          : "border-border hover:shadow-lg hover:-translate-y-0.5",
        justTriggered && "ring-2 ring-blue-400/60",
      )}
    >
      {/* Shimmer overlay for active processing */}
      {isActive && (
        <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none">
          <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-blue-400/[0.07] to-transparent" />
        </div>
      )}

      {/* ── Header ── */}
      <div className="relative px-4 pt-4 pb-3 flex items-start gap-3">
        {/* File icon */}
        <div
          className={cn(
            "w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors",
            isActive ? "bg-blue-400/10" : "bg-muted/50",
          )}
        >
          {isActive ? (
            <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
          ) : (
            <FileIcon className={cn("w-5 h-5", fileConfig.color)} />
          )}
        </div>

        {/* Name + meta */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5 min-w-0">
            <p className="font-medium text-sm truncate" title={doc.original_filename}>
              {doc.original_filename}
            </p>
            {doc.status === "indexed" && (
              <CheckCircle2 className="w-4 h-4 text-emerald-500 fill-emerald-500/10 flex-shrink-0" />
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-0.5">
            {doc.file_type.toUpperCase()}
            <span className="mx-1.5 text-muted-foreground/40">&middot;</span>
            {formatFileSize(doc.file_size)}
            {doc.parser_version && (
              <>
                <span className="mx-1.5 text-muted-foreground/40">&middot;</span>
                <span className="text-muted-foreground/60">{doc.parser_version}</span>
              </>
            )}
          </p>
        </div>

        {/* Action Buttons */}
        <div className="relative flex-shrink-0 flex items-center gap-1">
          {/* Quick Analyze button for pending/failed docs */}
          {(doc.status === "pending" || doc.status === "failed") && (
            <Button
              variant="default"
              size="sm"
              onClick={handleProcess}
              disabled={isProcessing}
              className="h-7 px-2 text-[10px] items-center gap-1"
            >
              <RefreshCw className={cn("w-3 h-3", isProcessing && "animate-spin")} />
              {t("files.analyze")}
            </Button>
          )}

          <div className="relative">
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "h-7 w-7 transition-opacity",
                menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100"
              )}
              onClick={(e) => {
                e.stopPropagation();
                setMenuOpen((v) => !v);
              }}
            >
              <MoreHorizontal className="w-4 h-4" />
            </Button>

            {menuOpen && (
              <div className="absolute right-0 top-8 z-20 min-w-[160px] rounded-xl border bg-card/95 backdrop-blur-md shadow-xl py-1.5 animate-in fade-in zoom-in-95 duration-100">
                <button
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-accent transition-colors"
                  onClick={() => {
                    onDownload(doc);
                    setMenuOpen(false);
                  }}
                >
                  <Download className="w-4 h-4 text-muted-foreground" />
                  {t("common.download")}
                </button>
                {onPreview && doc.status === "indexed" && (
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-accent transition-colors"
                    onClick={() => {
                      onPreview(doc);
                      setMenuOpen(false);
                    }}
                  >
                    <Eye className="w-4 h-4 text-muted-foreground" />
                    {t("files.preview")}
                  </button>
                )}
                {(doc.status === "indexed" || doc.status === "building_kg") && (
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-accent transition-colors"
                    onClick={() => {
                      onReindex(doc.id);
                      setMenuOpen(false);
                    }}
                  >
                    <RefreshCw className="w-4 h-4 text-muted-foreground" />
                    {t("files.re_analyze")}
                  </button>
                )}
                <div className="h-px bg-border my-1" />
                <button
                  className="w-full flex items-center gap-2 px-3 py-2 text-xs text-destructive hover:bg-destructive/10 transition-colors"
                  onClick={() => {
                    onDelete(doc.id);
                    setMenuOpen(false);
                  }}
                >
                  <Trash2 className="w-4 h-4" />
                  {t("common.delete")}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Status + processing time ── */}
      <div className="px-4 pb-3 flex items-center justify-between">
        <StatusBadge status={doc.status} />
        {doc.processing_time_ms != null && doc.processing_time_ms > 0 && (
          <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <Clock className="w-3 h-3" />
            {formatProcessingTime(doc.processing_time_ms)}
          </span>
        )}
      </div>

      {/* ── Stats grid ── */}
      {stats.length > 0 && (
        <div className="px-4 pb-3 grid grid-cols-2 sm:grid-cols-3 gap-x-3 gap-y-1">
          {stats.map((s) => {
            const StatIcon = s.icon;
            return (
              <div key={s.label} className="flex items-center gap-1.5 text-[11px] text-muted-foreground min-w-0">
                <StatIcon className="w-3 h-3 flex-shrink-0" />
                <span className="tabular-nums font-medium">{s.value}</span>
                <span className="truncate">{s.label}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Sub-task progress ── */}
      {showSubTasks && (
        <div className="px-4 pb-3">
          <SubTaskProgress
            embed_done={doc.embed_done}
            captions_done={doc.captions_done}
            kg_done={doc.kg_done}
          />
        </div>
      )}

      {/* ── Document type ── */}
      <div className="px-4 pb-3 flex items-center gap-1.5 text-xs text-muted-foreground">
        <Tag className="w-3.5 h-3.5 flex-shrink-0" />
        <span>{doc.document_type?.name || t("common.unknown")}</span>
      </div>

      {/* ── Digital signatures (always shown if not failed) ── */}
      {doc.status !== "failed" && (
        <div className="px-4 pb-3 space-y-1 text-emerald-600 dark:text-emerald-400">
          {!hasSigs ? (
            <div className="flex items-center gap-1.5 text-xs">
              <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
              <span className="truncate">{t("files.metadata.signed_by")}: {t("common.unknown")}</span>
            </div>
          ) : (
            Array.from(
              new Map(
                doc.digital_signatures!.map((sig) => [
                  sig.signer_name?.toLowerCase() || "unknown",
                  sig,
                ])
              ).values()
            ).map((sig, i) => (
              <div
                key={i}
                className="flex items-center gap-1.5 text-xs"
              >
                <ShieldCheck className="w-3.5 h-3.5 flex-shrink-0" />
                <span className="truncate">
                  {t("files.metadata.signed_by")}: {sig.signer_name || t("common.unknown")}
                  {sig.organization && ` (${sig.organization})`}
                </span>
              </div>
            ))
          )}
        </div>
      )}

      {/* ── Error message ── */}
      {doc.error_message && (
        <div className="px-4 pb-3">
          <p className="text-xs text-destructive truncate" title={doc.error_message}>
            {doc.error_message}
          </p>
        </div>
      )}

      {/* ── Timestamps ── */}
      <div className="px-4 pb-3 mt-auto flex items-center gap-3 text-[11px] text-muted-foreground/70">
        <span>Uploaded: {formatDate(doc.created_at)}</span>
        {doc.updated_at !== doc.created_at && (
          <span>Updated: {formatDate(doc.updated_at)}</span>
        )}
      </div>

      {/* ── Action buttons (removed and consolidated into menu to avoid overlap) ── */}
    </motion.div>
  );
});
