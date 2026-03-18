import { memo, useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  FileText,
  FileType,
  Presentation,
  FileCode,
  Hash,
  Trash2,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Loader2,
  Clock,
  File,
  Sparkles,
  Layers,
  ImageIcon,
  Network,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Document, DocumentStatus } from "@/types";

// ---------------------------------------------------------------------------
// File-type icon mapping
// ---------------------------------------------------------------------------
const FILE_TYPE_CONFIG: Record<string, { icon: typeof FileText; color: string }> = {
  pdf:  { icon: FileText, color: "text-red-400" },
  docx: { icon: FileType, color: "text-blue-400" },
  pptx: { icon: Presentation, color: "text-orange-400" },
  txt:  { icon: FileCode, color: "text-muted-foreground" },
  md:   { icon: Hash, color: "text-purple-400" },
};

function getFileConfig(fileType: string) {
  const ext = fileType.replace(".", "").toLowerCase();
  return FILE_TYPE_CONFIG[ext] ?? { icon: File, color: "text-muted-foreground" };
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------
const STATUS_CONFIG: Record<DocumentStatus, { label: string; className: string; icon: typeof CheckCircle2 }> = {
  pending:         { label: "Pending",  className: "bg-muted text-muted-foreground",         icon: Clock },
  parsing:         { label: "Parsing",  className: "bg-blue-400/15 text-blue-400",           icon: Loader2 },
  parsed:          { label: "Parsed",   className: "bg-cyan-400/15 text-cyan-400",           icon: Loader2 },
  indexed_partial: { label: "Indexing", className: "bg-amber-400/15 text-amber-400",         icon: Loader2 },
  indexed:         { label: "Indexed",  className: "bg-primary/15 text-primary",             icon: CheckCircle2 },
  failed:          { label: "Failed",   className: "bg-destructive/15 text-destructive",     icon: XCircle },
};

function StatusBadge({ status }: { status: DocumentStatus }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
  const Icon = config.icon;
  const isAnimated = status === "parsing" || status === "parsed" || status === "indexed_partial";

  return (
    <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full", config.className)}>
      <Icon className={cn("w-3 h-3", isAnimated && "animate-spin")} />
      {config.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Sub-task progress pills (shown when status = parsed | indexed_partial)
// ---------------------------------------------------------------------------
interface SubTaskProgressProps {
  embed_done?: boolean;
  captions_done?: boolean;
  kg_done?: boolean;
}

function SubTaskProgress({ embed_done, captions_done, kg_done }: SubTaskProgressProps) {
  const tasks = [
    { done: embed_done,    label: "Embed",    Icon: Layers },
    { done: captions_done, label: "Captions", Icon: ImageIcon },
    { done: kg_done,       label: "KG",       Icon: Network },
  ];

  return (
    <div className="flex items-center gap-1.5 mt-1.5">
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
          {done
            ? <CheckCircle2 className="w-2.5 h-2.5" />
            : <Loader2 className="w-2.5 h-2.5 animate-spin" />
          }
          <Icon className="w-2.5 h-2.5" />
          {label}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metadata chips
// ---------------------------------------------------------------------------
function MetadataChips({ doc }: { doc: Document }) {
  const chips: { label: string; value: number }[] = [];
  if (doc.page_count && doc.page_count > 0) chips.push({ label: "pages", value: doc.page_count });
  if (doc.chunk_count > 0) chips.push({ label: "chunks", value: doc.chunk_count });
  if (doc.image_count && doc.image_count > 0) chips.push({ label: "images", value: doc.image_count });
  if (doc.table_count && doc.table_count > 0) chips.push({ label: "tables", value: doc.table_count });

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
  onDelete: (id: number) => void;
  onReindex: (id: number) => void;
  onProcess: (id: number) => void;
  isProcessing?: boolean;
  onClick?: (doc: Document) => void;
}

const ACTIVE_STATUSES: DocumentStatus[] = ["parsing", "parsed", "indexed_partial"];

export const DocumentCard = memo(function DocumentCard({
  doc,
  selected,
  onDelete,
  onReindex,
  onProcess,
  isProcessing,
  onClick,
}: DocumentCardProps) {
  const fileConfig = getFileConfig(doc.file_type);
  const FileIcon = fileConfig.icon;
  const sizeStr = doc.file_size >= 1024 * 1024
    ? `${(doc.file_size / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.round(doc.file_size / 1024)} KB`;

  const isActive = ACTIVE_STATUSES.includes(doc.status);
  const showSubTasks = doc.status === "parsed" || doc.status === "indexed_partial";
  // KG still running in background after document is indexed
  const kgPending = doc.status === "indexed" && doc.kg_done === false;

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

  return (
    <motion.div
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
        doc.status === "indexed" ? "cursor-pointer" : "cursor-default",
        justTriggered && "ring-2 ring-blue-400/60",
      )}
      onClick={() => onClick?.(doc)}
    >
      {/* Shimmer overlay for active processing */}
      {isActive && (
        <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
          <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-blue-400/[0.07] to-transparent" />
        </div>
      )}

      <div className="relative px-4 py-3 flex items-start gap-3">
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
            <StatusBadge status={doc.status} />
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-xs text-muted-foreground">{sizeStr}</span>
            {doc.parser_version && (
              <span className="text-xs text-muted-foreground/60">{doc.parser_version}</span>
            )}
            {doc.status === "parsing" && (
              <span className="text-xs text-blue-400/80 font-medium animate-pulse">
                Parsing document...
              </span>
            )}
            {doc.status === "parsed" && (
              <span className="text-xs text-cyan-400/80 font-medium animate-pulse">
                Building index...
              </span>
            )}
            {doc.status === "indexed_partial" && (
              <span className="text-xs text-amber-400/80 font-medium animate-pulse">
                Finalizing...
              </span>
            )}
          </div>
          <MetadataChips doc={doc} />
          {showSubTasks && (
            <SubTaskProgress
              embed_done={doc.embed_done}
              captions_done={doc.captions_done}
              kg_done={doc.kg_done}
            />
          )}
          {kgPending && (
            <div className="flex items-center gap-1 mt-1.5">
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium border bg-violet-400/10 text-violet-400/80 border-violet-400/20">
                <Loader2 className="w-2.5 h-2.5 animate-spin" />
                <Network className="w-2.5 h-2.5" />
                Building KG…
              </span>
            </div>
          )}
          {doc.error_message && (
            <p className="text-xs text-destructive mt-1 truncate">{doc.error_message}</p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {(doc.status === "pending" || doc.status === "failed") && (
            <Button
              variant="default"
              size="sm"
              onClick={handleProcess}
              disabled={isProcessing}
              className="h-7 text-xs gap-1.5"
            >
              {isProcessing ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Sparkles className="w-3 h-3" />
              )}
              Analyze
            </Button>
          )}
          {doc.status === "indexed" && (
            <Button
              variant="ghost"
              size="icon"
              onClick={(e) => { e.stopPropagation(); onReindex(doc.id); }}
              className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
              title="Re-analyze"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            onClick={(e) => { e.stopPropagation(); onDelete(doc.id); }}
            className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <Trash2 className="w-3.5 h-3.5 text-destructive" />
          </Button>
        </div>
      </div>
    </motion.div>
  );
});
