import { memo, useState } from "react";
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
        "inline-flex items-center gap-1.5 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded-md border shadow-sm",
        config.className,
        status === "failed" ? "border-destructive/20" : "border-border/50"
      )}
    >
      <Icon className={cn("w-3 h-3", isAnimated && "animate-spin")} />
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

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={cn(
        "group relative flex flex-col bg-card/50 backdrop-blur-sm border rounded-xl overflow-hidden hover:shadow-lg transition-all duration-300",
        status === "failed" ? "border-destructive/30" : "border-border/50",
        className
      )}
    >
      {/* Top Banner: Status & Menu */}
      <div className="flex items-center justify-between px-4 py-3 bg-muted/30">
        <StatusBadge status={status} />
        
        <div className="flex items-center gap-1 ml-auto">
          {/* Action Menu */}
          <div className="relative">
            <Button
              variant="ghost"
              size="icon"
              className="w-8 h-8 rounded-full hover:bg-muted/80"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <MoreHorizontal className="w-4 h-4" />
            </Button>

            <AnimatePresence>
              {isMenuOpen && (
                <>
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 z-10" 
                    onClick={() => setIsMenuOpen(false)}
                  />
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95, y: -10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: -10 }}
                    className="absolute right-0 top-full mt-1 w-48 bg-card border rounded-lg shadow-xl z-20 py-1 overflow-hidden"
                  >
                    <button
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left font-medium text-blue-500"
                      onClick={() => {
                        onProcess(doc.id);
                        setIsMenuOpen(false);
                      }}
                      disabled={isProcessing}
                    >
                      <RefreshCw className={cn("w-4 h-4", isProcessing && "animate-spin")} />
                      {t("files.analyze")}
                    </button>
                    <button
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left"
                      onClick={() => {
                        onDownload(doc);
                        setIsMenuOpen(false);
                      }}
                    >
                      <Download className="w-4 h-4 text-muted-foreground" />
                      {t("common.download")}
                    </button>
                    <button
                      className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left"
                      onClick={() => {
                        onPreview(doc);
                        setIsMenuOpen(false);
                      }}
                    >
                      <Eye className="w-4 h-4 text-muted-foreground" />
                      {t("files.preview")}
                    </button>
                    {onClickEdit && (
                      <button
                        className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted transition-colors text-left font-medium text-primary"
                        onClick={() => {
                          onClickEdit(doc);
                          setIsMenuOpen(false);
                        }}
                      >
                        <Tag className="w-4 h-4" />
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
                      <RefreshCw className="w-4 h-4 text-muted-foreground" />
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
                      <Trash2 className="w-4 h-4" />
                      {t("common.delete")}
                    </button>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Document Content */}
      <div className="p-4 flex flex-col gap-3">
        {/* Main Info */}
        <div className="flex items-start gap-3">
          <div className={cn(
            "p-3 rounded-lg bg-background/80 border shadow-sm",
            fileConfig.color.replace("text-", "bg-").replace("-400", "-400/10")
          )}>
            <FileIcon className={cn("w-6 h-6", fileConfig.color)} />
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 min-w-0">
              <h3 className="font-semibold text-sm truncate leading-tight" title={filename || original_filename}>
                {filename || original_filename}
              </h3>
              {status === "indexed" && (
                <CheckCircle2 className="w-4 h-4 text-primary shrink-0" />
              )}
            </div>
            <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground uppercase font-medium tracking-wider">
              <span>{file_type?.replace(".", "") || "FILE"}</span>
              <span className="w-1 h-1 rounded-full bg-border" />
              <span>{formatFileSize(file_size)}</span>
              {doc.parser_version && (
                <>
                  <span className="w-1 h-1 rounded-full bg-border" />
                  <span>{doc.parser_version}</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Status indicator for failed documents */}
        {status === "failed" && doc.error_message && (
          <div className="p-2 bg-destructive/5 border border-destructive/10 rounded flex items-start gap-2">
            <AlertCircle className="w-3.5 h-3.5 text-destructive shrink-0 mt-0.5" />
            <p className="text-[11px] text-destructive leading-relaxed font-medium">
              {doc.error_message}
            </p>
          </div>
        )}

        {/* Metadata Chips */}
        <div className="flex flex-col gap-2 pt-1 border-t border-border/30">
          {/* Counts */}
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
            <div className="flex items-center gap-1">
              <FileText className="w-3 h-3" />
              <span>{t("count", { count: doc.page_count || 0, unit: t("common.pages") })}</span>
            </div>
            <div className="flex items-center gap-1">
              <Layers className="w-3 h-3" />
              <span>{t("count", { count: doc.chunk_count || 0, unit: t("common.chunks") })}</span>
            </div>
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

          {/* Signer Info */}
          {displaySigner && (
            <div className="flex items-center gap-1.5 text-[11px] text-green-500 font-semibold mt-1">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span className="truncate">{t("files.signed_by", { name: displaySigner })}</span>
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
      <div className="mt-auto px-4 py-2 border-t border-border/30 bg-muted/10 flex items-center justify-between text-[9px] text-muted-foreground/60 uppercase tracking-tighter">
        <span>Uploaded: {formatDate(created_at)}</span>
        <Clock className="w-2.5 h-2.5" />
      </div>
    </motion.div>
  );
});
