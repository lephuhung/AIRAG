import { memo, useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { useTranslation } from "@/hooks/useTranslation";
import {
  Trash2,
  CheckCircle2,
  Loader2,
  Layers,
  ImageIcon,
  Network,
  Play,
  MoreHorizontal,
  Download,
  RefreshCw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { getFileConfig } from "@/components/rag/document-utils";
import type { Document, DocumentStatus } from "@/types";


// ---------------------------------------------------------------------------
// Sub-task progress pills (shown when status = chunking | embedding)
// ---------------------------------------------------------------------------
interface SubTaskProgressProps {
  embed_done?: boolean;
  captions_done?: boolean;
  kg_done?: boolean;
  isNarrow?: boolean;
}

function SubTaskProgress({ embed_done, captions_done, kg_done, isNarrow }: SubTaskProgressProps) {
  const { t } = useTranslation();
  const tasks = [
    { done: embed_done,    label: t("files.tasks.embed"),    Icon: Layers },
    { done: captions_done, label: t("files.tasks.captions"), Icon: ImageIcon },
    { done: kg_done,       label: t("files.tasks.kg"),       Icon: Network },
  ];

  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
      {tasks.map(({ done, label, Icon }) => (
        <span
          key={label}
          title={label}
          className={cn(
            "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border h-5 transition-all duration-200",
            done
              ? "bg-green-500/10 text-green-400 border-green-500/20"
              : "bg-amber-400/10 text-amber-400/70 border-amber-400/20",
          )}
        >
          {done
            ? <CheckCircle2 className="w-2.5 h-2.5 flex-shrink-0" />
            : <Loader2 className="w-2.5 h-2.5 animate-spin flex-shrink-0" />
          }
          <Icon className="w-2.5 h-2.5 flex-shrink-0" />
          {!isNarrow && <span className="truncate max-w-[80px]">{label}</span>}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metadata chips
// ---------------------------------------------------------------------------
function MetadataChips({ doc }: { doc: Document }) {
  const { t } = useTranslation();
  const chips: { label: string; value: number }[] = [];
  if (doc.page_count && doc.page_count > 0) chips.push({ label: t("files.metadata.pages"), value: doc.page_count });
  if (doc.chunk_count > 0) chips.push({ label: t("files.metadata.chunks"), value: doc.chunk_count });
  if (doc.image_count && doc.image_count > 0) chips.push({ label: t("files.metadata.images"), value: doc.image_count });
  if (doc.table_count && doc.table_count > 0) chips.push({ label: t("files.metadata.tables"), value: doc.table_count });

  if (chips.length === 0) return null;

  return (
    <div className="flex items-center gap-2 mt-1">
      {chips.map((c) => (
        <span key={c.label} className="text-xs text-muted-foreground">
          {c.value} {c.label}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// DocumentCard
// ---------------------------------------------------------------------------
interface DocumentCardProps {
  doc: Document;
  selected?: boolean;
  onToggleSelect?: (id: string) => void;
  onDelete: (id: string) => void;
  onProcess: (id: string) => void;
  onReindex?: (id: string) => void;
  onDownload?: (doc: Document) => void;
  onClick?: (doc: Document) => void;
  showSubTasks?: boolean;
  isProcessing?: boolean;
  className?: string;
}

const ACTIVE_STATUSES: DocumentStatus[] = ["parsing", "ocring", "chunking", "embedding", "building_kg"];

export const DocumentCard = memo(({
  doc,
  selected,
  onToggleSelect,
  onDelete,
  onProcess,
  onReindex,
  onDownload,
  onClick,
  showSubTasks: forceShowSubTasks,
  isProcessing,
  className,
}: DocumentCardProps) => {
  const { t } = useTranslation();
  const fileConfig = getFileConfig(doc.file_type);
  const FileIcon = fileConfig.icon;
  const sizeStr = doc.file_size >= 1024 * 1024
    ? `${(doc.file_size / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.round(doc.file_size / 1024)} KB`;

  const isActive = ACTIVE_STATUSES.includes(doc.status);
  const shouldShowSubTasks = forceShowSubTasks ?? (doc.status === "chunking" || doc.status === "embedding");

  // Detection for narrow layout
  const [isNarrow, setIsNarrow] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!cardRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      setIsNarrow(width < 250);
    });
    observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, []);

  // Action dropdown menu (mirrors FileCard's menu)
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

  // Flash animation when user just clicked "Analyze"
  const [justTriggered, setJustTriggered] = useState(false);
  useEffect(() => {
    if (justTriggered) {
      const timer = setTimeout(() => setJustTriggered(false), 1200);
      return () => clearTimeout(timer);
    }
  }, [justTriggered]);

  const triggerProcess = () => {
    setJustTriggered(true);
    onProcess(doc.id);
  };

  const handleProcess = (e: React.MouseEvent) => {
    e.stopPropagation();
    triggerProcess();
  };

  return (
    <motion.div
      ref={cardRef}
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{
        opacity: 1,
        y: 0,
        ...(justTriggered ? { scale: [1, 0.98, 1.01, 1] } : {}),
      }}
      exit={{ opacity: 0, y: -8 }}
      transition={justTriggered ? { duration: 0.4 } : undefined}
      className={cn(
        "group relative rounded-lg border bg-card transition-all duration-200",
        isActive
          ? "border-blue-400/50 shadow-[0_0_12px_-3px_rgba(96,165,250,0.3)]"
          : "border-border hover:shadow-md hover:-translate-y-0.5",
        selected && "border-primary ring-1 ring-primary/30 shadow-sm",
        isMenuOpen && "z-30",
        doc.status === "indexed" || doc.status === "building_kg" ? "cursor-pointer" : "cursor-default",
        justTriggered && "ring-2 ring-blue-400/60",
        className
      )}
      onClick={() => onClick?.(doc)}
    >
      {/* Shimmer overlay for active processing */}
      {isActive && (
        <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
          <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-blue-400/[0.07] to-transparent" />
        </div>
      )}

      <div className="relative px-4 py-3 flex flex-col gap-2">
        {onToggleSelect && (
          <div 
            className="absolute top-3 right-3 z-10 opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={(e) => e.stopPropagation()}
          >
            <input
              type="checkbox"
              checked={selected}
              onChange={() => onToggleSelect(doc.id)}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
            />
          </div>
        )}
        <div className="flex items-start gap-3">
          {/* File icon */}
          <div className={cn(
            "w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors",
            isActive ? "bg-blue-400/10" : "bg-muted/50",
          )}>
            {isActive ? (
              <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />
            ) : (
              <FileIcon className={cn("w-5 h-5", fileConfig.color)} />
            )}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <p className="font-medium text-sm truncate">{doc.original_filename}</p>
            </div>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className="text-[10px] text-muted-foreground font-medium">{sizeStr}</span>
              {doc.parser_version && (
                <span className="text-xs text-muted-foreground/60">{doc.parser_version}</span>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-1 flex-shrink-0">
            {doc.status === "pending" && (
              <button
                onClick={handleProcess}
                disabled={isProcessing}
                className="flex items-center gap-1.5 px-2 py-1 rounded bg-blue-500/10 text-blue-500 hover:bg-blue-500/20 transition-colors disabled:opacity-50"
                title={t("workspace.analyze_all")}
              >
                <Play className="w-3 h-3 fill-current" />
                <span className="font-semibold uppercase tracking-wider text-[10px]">{t("files.status.pending")}</span>
              </button>
            )}

            <div className="relative" ref={menuRef}>
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  setIsMenuOpen(!isMenuOpen);
                }}
                className={cn(
                  "h-7 w-7 transition-opacity",
                  isMenuOpen ? "opacity-100 bg-muted" : "opacity-0 group-hover:opacity-100"
                )}
              >
                <MoreHorizontal className="w-3.5 h-3.5" />
              </Button>

              {isMenuOpen && (
                <div
                  className="absolute right-0 top-full mt-1 w-44 bg-card border rounded-lg shadow-xl z-20 py-1 overflow-hidden"
                  onClick={(e) => e.stopPropagation()}
                >
                  {doc.status === "pending" && (
                    <button
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left font-semibold text-blue-500"
                      onClick={() => {
                        triggerProcess();
                        setIsMenuOpen(false);
                      }}
                      disabled={isProcessing}
                    >
                      <RefreshCw className={cn("w-3.5 h-3.5", isProcessing && "animate-spin")} />
                      {t("files.analyze")}
                    </button>
                  )}
                  {onDownload && (
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
                  )}
                  {onReindex && (
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
                  )}
                  <div className="h-px bg-border my-1" />
                  <button
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm text-destructive hover:bg-destructive/10 transition-colors text-left"
                    onClick={() => {
                      onDelete(doc.id);
                      setIsMenuOpen(false);
                    }}
                    disabled={isProcessing}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                    {t("common.delete")}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Status specific messages */}
        <div className={cn(
          "flex flex-col gap-1 transition-all duration-200",
          isActive ? "pl-11 sm:pl-13" : "pl-11"
        )}>
          {doc.status === "parsing" && (
            <span className="text-xs text-blue-400/80 font-medium animate-pulse">
              {t("files.msg.parsing")}
            </span>
          )}
          {doc.status === "ocring" && (
            <span className="text-xs text-indigo-400/80 font-medium animate-pulse">
              {t("files.msg.ocring")}
            </span>
          )}
          {doc.status === "chunking" && (
            <span className="text-xs text-cyan-400/80 font-medium animate-pulse">
              {t("files.msg.chunking")}
            </span>
          )}
          {doc.status === "embedding" && (
            <span className="text-xs text-amber-400/80 font-medium animate-pulse">
              {t("files.msg.embedding")}
            </span>
          )}
          {doc.status === "building_kg" && (
            <span className="text-xs text-violet-400/80 font-medium animate-pulse">
              {t("files.msg.building_kg")}
            </span>
          )}
          
          <MetadataChips doc={doc} />
          
          {shouldShowSubTasks && (
            <SubTaskProgress
              embed_done={doc.embed_done}
              captions_done={doc.captions_done}
              kg_done={doc.kg_done}
              isNarrow={isNarrow}
            />
          )}
          
          {doc.error_message && (
            <p className="text-xs text-destructive mt-1 truncate">{doc.error_message}</p>
          )}
        </div>
      </div>
    </motion.div>
  );
});
