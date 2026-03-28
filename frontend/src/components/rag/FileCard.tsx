import { memo, useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
  Tag,
  ShieldCheck,
  Eye,
  AlertCircle,
  Clock
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  STATUS_CONFIG,
  getFileConfig,
} from "@/components/rag/DocumentCard";
import { formatFileSize, formatDate } from "@/lib/format";
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

  // Don't show redundant badge for indexed status (already has checkmark next to name)
  if (status === "indexed") return null;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded-md border shadow-sm whitespace-nowrap min-w-fit",
        config.className,
        status === "failed" ? "border-destructive/20" : "border-border/50"
      )}
    >
      <Icon className={cn("w-3 h-3 shrink-0", isAnimated && "animate-spin")} />
      {t(config.labelKey)}
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
    <div className="flex items-center gap-1 flex-wrap">
      {tasks.map(({ done, label, Icon }) => (
        <span
          key={label}
          className={cn(
            "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold border transition-colors",
            done
              ? "bg-green-500/10 text-green-500 border-green-500/20"
              : "bg-amber-500/10 text-amber-600 border-amber-500/20",
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
// FileCard component — vertical card for grid display
// ---------------------------------------------------------------------------
interface FileCardProps {
  doc: Document;
  onDelete: (docId: number) => void;
  onReindex: (docId: number) => void;
  onProcess: (docId: number) => void;
  onDownload: (doc: Document) => void;
  onPreview: (doc: Document) => void;
  onClickEdit?: (doc: Document) => void;
  isProcessing?: boolean;
  className?: string;
}

export const FileCard = memo(function FileCard({
  doc,
  onDelete,
  onReindex,
  onProcess,
  onDownload,
  onPreview,
  onClickEdit,
  isProcessing,
  className,
}: FileCardProps) {
  const { t } = useTranslation();
  const { status, filename, file_type, file_size, original_filename, created_at, signer_name, digital_signatures } = doc;
  
  const displaySigner = signer_name || (digital_signatures && digital_signatures.length > 0 ? digital_signatures[0].signer_name : null);
  const fileConfig = getFileConfig(file_type || original_filename?.split(".").pop() || "");
  const FileIcon = fileConfig.icon;

  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isMenuOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [isMenuOpen]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "group relative flex flex-col bg-card/60 backdrop-blur-md border rounded-xl shadow-sm hover:shadow-xl hover:-translate-y-0.5 transition-all duration-300",
        status === "failed" ? "border-destructive/30" : "border-border/60",
        className
      )}
    >

      {/* Document Content */}
      <div className="p-4 flex flex-col gap-2.5">
        {/* Main Info */}
        <div className="flex items-start gap-3">
          <div className={cn(
            "p-3 rounded-lg bg-background/80 border shadow-sm",
            fileConfig.color.replace("text-", "bg-").replace("-400", "-400/10")
          )}>
            <FileIcon className={cn("w-6 h-6", fileConfig.color)} />
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="mb-2">
              <StatusBadge status={status} />
            </div>
            <div className="flex items-center gap-1.5 min-w-0">
              <div className="flex items-center gap-1.5 min-w-0 flex-1">
                <h3 className="font-semibold text-sm truncate leading-tight" title={filename || original_filename}>
                  {filename || original_filename}
                </h3>
                {status === "indexed" && (
                  <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
                )}
              </div>

              {/* Action Menu moved inline */}
              <div className="relative shrink-0">
                <Button
                  variant="ghost"
                  size="icon"
                  className="w-7 h-7 rounded-full hover:bg-muted/80 ml-auto"
                  onClick={(e) => {
                    e.stopPropagation();
                    setIsMenuOpen(!isMenuOpen);
                  }}
                >
                  <MoreHorizontal className="w-3.5 h-3.5" />
                </Button>

                <AnimatePresence>
                  {isMenuOpen && (
                    <>
                      <motion.div 
                        ref={menuRef}
                        initial={{ opacity: 0, scale: 0.95, y: -10 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: -10 }}
                        className="absolute right-0 top-full mt-1 w-44 bg-card border rounded-lg shadow-xl z-20 py-1 overflow-hidden"
                      >
                        {status === "pending" && (
                          <button
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left font-semibold text-blue-500"
                            onClick={() => {
                              onProcess(doc.id);
                              setIsMenuOpen(false);
                            }}
                            disabled={isProcessing}
                          >
                            <RefreshCw className={cn("w-3.5 h-3.5", isProcessing && "animate-spin")} />
                            {t("files.analyze")}
                          </button>
                        )}
                        <button
                          className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left"
                          onClick={() => {
                            onDownload(doc);
                            setIsMenuOpen(false);
                          }}
                        >
                          <Download className="w-3.5 h-3.5 text-muted-foreground" />
                          {t("common.download")}
                        </button>
                        <button
                          className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left"
                          onClick={() => {
                            onPreview(doc);
                            setIsMenuOpen(false);
                          }}
                        >
                          <Eye className="w-3.5 h-3.5 text-muted-foreground" />
                          {t("files.preview")}
                        </button>
                        {onClickEdit && (
                          <button
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left font-medium text-foreground/80"
                            onClick={() => {
                              onClickEdit(doc);
                              setIsMenuOpen(false);
                            }}
                          >
                            <Tag className="w-3.5 h-3.5" />
                            {t("files.edit_metadata")}
                          </button>
                        )}
                        <button
                          className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left"
                          onClick={() => {
                            onReindex(doc.id);
                            setIsMenuOpen(false);
                          }}
                        >
                          <RefreshCw className="w-3.5 h-3.5 text-muted-foreground/70" />
                          {t("files.re_analyze")}
                        </button>
                        <div className="h-px bg-border my-1" />
                        <button
                          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-destructive hover:bg-destructive/10 transition-colors text-left"
                          onClick={() => {
                            onDelete(doc.id);
                            setIsMenuOpen(false);
                          }}
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                          {t("common.delete")}
                        </button>
                      </motion.div>
                    </>
                  )}
                </AnimatePresence>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-1 text-[10px] text-muted-foreground uppercase font-medium tracking-wider">
              <span className="shrink-0">{file_type?.replace(".", "") || "FILE"}</span>
              <span className="w-1 h-1 rounded-full bg-border shrink-0" />
              <span className="shrink-0">{formatFileSize(file_size)}</span>
              {doc.parser_version && (
                <div className="hidden sm:flex items-center gap-2 shrink-0">
                  <span className="w-1 h-1 rounded-full bg-border" />
                  <span>{doc.parser_version}</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Status indicator for failed OR processing info */}
        {doc.error_message && (
          <div className={cn(
            "p-2 border rounded flex items-start gap-2",
            status === "failed" 
              ? "bg-destructive/5 border-destructive/10" 
              : "bg-amber-500/5 border-amber-500/10"
          )}>
            <AlertCircle className={cn(
              "w-3.5 h-3.5 shrink-0 mt-0.5",
              status === "failed" ? "text-destructive" : "text-amber-500"
            )} />
            <p className={cn(
              "text-[11px] leading-relaxed font-semibold",
              status === "failed" ? "text-destructive" : "text-amber-600"
            )}>
              {doc.error_message}
            </p>
          </div>
        )}

        {/* Metadata Chips */}
        <div className="flex flex-col gap-1.5 pt-2 mt-0.5 border-t border-border/30">
          {/* Counts row with better layout */}
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] uppercase font-bold tracking-tight text-muted-foreground/80">
            {doc.page_count !== undefined && doc.page_count > 0 && (
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-muted/40 border border-border/10">
                <FileText className="w-3 h-3 text-muted-foreground/60" />
                <span>{t("common.count", { count: doc.page_count, unit: t("common.pages") })}</span>
              </div>
            )}
            {doc.chunk_count > 0 && (
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-muted/40 border border-border/10">
                <Layers className="w-3 h-3 text-muted-foreground/60" />
                <span>{t("common.count", { count: doc.chunk_count, unit: t("common.chunks") })}</span>
              </div>
            )}
            {doc.table_count !== undefined && doc.table_count > 0 && (
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-muted/40 border border-border/10">
                <Network className="w-3 h-3 text-muted-foreground/60" />
                <span>{t("common.count", { count: doc.table_count, unit: t("common.tables") })}</span>
              </div>
            )}
            {doc.image_count !== undefined && doc.image_count > 0 && (
              <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-muted/40 border border-border/10">
                <ImageIcon className="w-3 h-3 text-muted-foreground/60" />
                <span>{t("common.count", { count: doc.image_count, unit: t("common.images") })}</span>
              </div>
            )}
          </div>

          {/* Doc Number & Type */}
          {(doc.document_number || doc.document_type) && (
            <div className="flex items-center gap-1.5 flex-wrap">
              {doc.document_number && (
                <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-muted/50 text-[10px] font-bold border border-border/50 text-secondary-foreground">
                  <Tag className="w-2.5 h-2.5 text-primary/70" />
                  {doc.document_number}
                </div>
              )}
              {doc.document_type && (
                <div className="flex items-center px-1.5 py-0.5 rounded bg-primary/10 text-[10px] font-bold border border-primary/20 text-primary uppercase">
                  {doc.document_type.name}
                </div>
              )}
            </div>
          )}

          {/* Enhanced Signer Info Display */}
          {displaySigner && (
            <div className="flex flex-col gap-0.5 mt-0.5 px-2 py-1 rounded-lg bg-green-500/5 border border-green-500/10 w-fit max-w-full">
              <div className="flex items-center gap-1.5 text-[11px] text-green-600 dark:text-green-400 font-bold">
                <ShieldCheck className="w-3.5 h-3.5" />
                <span className="truncate">
                  {t("files.metadata.signed_by")}: {displaySigner}
                  {digital_signatures && digital_signatures.length > 1 && (
                    <span className="text-[9px] font-normal opacity-70 ml-1">
                      (+{digital_signatures.length - 1})
                    </span>
                  )}
                </span>
              </div>
              {digital_signatures?.[0]?.organization && (
                <p className="text-[9px] text-muted-foreground/60 truncate ml-5 leading-none">
                  {digital_signatures[0].organization}
                </p>
              )}
            </div>
          )}
          
          {/* Progress pills for active tasks */}
          {(status === "chunking" || status === "embedding" || status === "building_kg") && (
            <div className="mt-1">
              <SubTaskProgress 
                embed_done={doc.embed_done}
                captions_done={doc.captions_done}
                kg_done={doc.kg_done}
              />
            </div>
          )}
        </div>
      </div>
      
      {/* Footer Timestamps */}
      <div className="mt-auto px-4 py-2 border-t border-border/30 bg-muted/10 flex items-center justify-between text-[9px] text-muted-foreground/60 uppercase tracking-tighter rounded-b-xl">
        <span>Uploaded: {formatDate(created_at)}</span>
        <Clock className="w-2.5 h-2.5" />
      </div>
    </motion.div>
  );
});
