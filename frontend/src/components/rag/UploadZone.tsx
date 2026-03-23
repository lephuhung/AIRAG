import { useState, useRef, useCallback, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileUp } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";
import { cn } from "@/lib/utils";

const ACCEPTED_TYPES = ".pdf,.txt,.docx,.md,.pptx";
const ACCEPTED_EXTENSIONS = new Set(["pdf", "txt", "docx", "md", "pptx"]);
const MAX_SIZE_MB = 50;

interface UploadZoneProps {
  onUpload: (file: File) => void;
  isUploading?: boolean;
  compact?: boolean;
  /** Always-visible mini drag-drop zone */
  mini?: boolean;
}

export const UploadZone = memo(function UploadZone({ onUpload, isUploading, compact, mini }: UploadZoneProps) {
  const { t } = useTranslation();
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
    if (!ACCEPTED_EXTENSIONS.has(ext)) return t("workspace.unsupported_format", { ext });
    if (file.size > MAX_SIZE_MB * 1024 * 1024) return t("workspace.file_size_error", { size: MAX_SIZE_MB });
    return null;
  }, [t]);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files) return;
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const error = validateFile(file);
        if (error) {
          // imported in parent — use toast there
          continue;
        }
        onUpload(file);
      }
    },
    [onUpload, validateFile]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  if (mini) {
    return (
      <>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES}
          multiple
          onChange={(e) => { handleFiles(e.target.files); if (inputRef.current) inputRef.current.value = ""; }}
          className="hidden"
        />
        <motion.div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
          animate={isDragOver ? { scale: 1.01 } : { scale: 1 }}
          className={cn(
            "h-full rounded-lg border-2 border-dashed cursor-pointer transition-colors duration-200",
            "flex flex-col items-center justify-center",
            isDragOver
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-muted/30",
            isUploading && "opacity-60 pointer-events-none"
          )}
        >
          <AnimatePresence mode="wait">
            {isDragOver ? (
              <motion.div
                key="drop"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex flex-col items-center"
              >
                <FileUp className="w-6 h-6 text-primary mb-1" />
                <p className="text-xs font-medium text-primary">{t("workspace.drop_here")}</p>
              </motion.div>
            ) : (
              <motion.div
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex flex-col items-center"
              >
                <Upload className={cn("w-6 h-6 text-muted-foreground mb-1", isUploading && "animate-pulse")} />
                <p className="text-xs font-medium">
                  {isUploading ? t("workspace.uploading") : t("workspace.drop_click")}
                </p>
                <p className="text-[10px] text-muted-foreground/60 mt-0.5">
                  PDF, DOCX, PPTX, TXT, MD (max {MAX_SIZE_MB}MB)
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </>
    );
  }

  if (compact) {
    return (
      <>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES}
          multiple
          onChange={(e) => { handleFiles(e.target.files); if (inputRef.current) inputRef.current.value = ""; }}
          className="hidden"
        />
        <motion.div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => inputRef.current?.click()}
          animate={isDragOver ? { scale: 1.01 } : { scale: 1 }}
          className={cn(
            "flex items-center gap-3 px-4 py-5 rounded-lg border border-dashed cursor-pointer transition-colors duration-150",
            isDragOver
              ? "border-primary bg-primary/5 text-primary"
              : "border-border hover:border-primary/40 hover:bg-muted/30 text-muted-foreground",
            isUploading && "opacity-60 pointer-events-none"
          )}
        >
          <Upload className={cn("w-5 h-5 flex-shrink-0", isUploading && "animate-pulse")} />
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-[11px] font-medium truncate">
              {isUploading ? t("workspace.uploading") : isDragOver ? t("workspace.drop_to_upload") : t("workspace.drop_click")}
            </span>
            <span className="text-[10px] opacity-60">PDF · DOCX · PPTX · TXT · MD</span>
          </div>
        </motion.div>
      </>
    );
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES}
        multiple
        onChange={(e) => { handleFiles(e.target.files); if (inputRef.current) inputRef.current.value = ""; }}
        className="hidden"
      />
      <motion.div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => inputRef.current?.click()}
        animate={isDragOver ? { scale: 1.01 } : { scale: 1 }}
        className={cn(
          "relative rounded-lg border-2 border-dashed cursor-pointer transition-colors duration-200",
          "flex flex-col items-center justify-center py-8 px-4",
          isDragOver
            ? "border-primary bg-primary/5"
            : "border-border hover:border-primary/50 hover:bg-muted/30",
          isUploading && "opacity-60 pointer-events-none"
        )}
      >
        <AnimatePresence mode="wait">
          {isDragOver ? (
            <motion.div
              key="drop"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="flex flex-col items-center"
            >
              <FileUp className="w-8 h-8 text-primary mb-2" />
              <p className="text-sm font-medium text-primary">{t("workspace.drop_here")}</p>
            </motion.div>
          ) : (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center"
            >
              <Upload className="w-8 h-8 text-muted-foreground mb-2" />
              <p className="text-sm font-medium">
                {isUploading ? t("workspace.uploading") : t("workspace.drop_click")}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                PDF, DOCX, PPTX, TXT, MD (max {MAX_SIZE_MB}MB)
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </>
  );
});
