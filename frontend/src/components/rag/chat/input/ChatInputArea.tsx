import { useState, useRef, useEffect } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Send,
  Square,
  Loader2,
  FileText,
  FileCode,
  FileSearch,
  Layers,
  CornerDownLeft,
  Paperclip,
  Mic,
  X,
  Image as ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslation } from "@/hooks/useTranslation";
import { getFileConfig } from "@/components/rag/document-utils";
import { formatMentionName } from "@/components/rag/chat/utils";
import type { Document } from "@/types";

// ---------------------------------------------------------------------------
// Document mention dropdown for @ autocomplete
// ---------------------------------------------------------------------------
type MentionDoc = {
  id: string;
  filename: string;
  original_filename?: string;
  file_type?: string;
  document_number?: string | null;
  document_title?: string | null;
  document_type_name?: string | null;
  issuing_agency?: string | null;
  published_date?: string | null;
  page_count?: number | null;
  file_size?: number | null;
};

function DocumentMentionDropdown({
  docs,
  onSelect,
  onClose,
  selectedIndex = 0,
  anchorRef,
}: {
  docs: MentionDoc[];
  onSelect: (doc: { id: string; filename: string }) => void;
  onClose: () => void;
  selectedIndex?: number;
  anchorRef?: React.RefObject<HTMLTextAreaElement | null>;
}) {
  const { t } = useTranslation();
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [coords, setCoords] = useState<{ bottom: number; left: number; width: number } | null>(null);

  // Calculate position from the textarea element to avoid stacking context issues
  useEffect(() => {
    const updatePosition = () => {
      if (anchorRef?.current) {
        const rect = anchorRef.current.getBoundingClientRect();
        // Mention dropdown stays above the textarea as it is tied to the cursor/typing experience
        setCoords({
          bottom: window.innerHeight - rect.top + 6,
          left: rect.left,
          width: Math.max(280, rect.width)
        });
      }
    };

    updatePosition();
    window.addEventListener('resize', updatePosition);
    return () => window.removeEventListener('resize', updatePosition);
  }, [anchorRef]);

  // Click outside listener
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (!dropdownRef.current) return;
      if (dropdownRef.current.contains(e.target as Node)) return;
      onClose();
    };

    const timer = setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
    }, 100);

    return () => {
      clearTimeout(timer);
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [onClose]);

  if (!coords) return null;

  const dropdownStyle: React.CSSProperties = {
    position: "fixed",
    bottom: coords.bottom,
    left: coords.left,
    width: coords.width,
    zIndex: 999999,
  };

  const containerClasses = cn(
    "bg-white dark:bg-zinc-950",
    "border border-zinc-200 dark:border-zinc-800",
    "rounded-xl shadow-[0_12px_48px_-12px_rgba(0,0,0,0.18)]",
    "overflow-hidden flex flex-col"
  );

  const content = (
    <motion.div
      ref={dropdownRef}
      initial={{ opacity: 0, y: 4, scale: 0.995 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.995 }}
      transition={{ duration: 0.12, ease: "easeOut" }}
      style={dropdownStyle}
      className={containerClasses}
    >
      {docs.length === 0 ? (
        <div className="flex items-center justify-center py-4 px-4 text-center gap-2">
          <FileSearch className="w-3.5 h-3.5 text-muted-foreground/40" />
          <p className="text-[11px] text-muted-foreground font-medium">
            {t("chat.no_docs_found")}
          </p>
        </div>
      ) : (
        <>
          {/* Header - Minimalist */}
          <div className="px-4 py-2 shrink-0 flex items-center justify-between bg-zinc-50 dark:bg-white/5 border-b border-zinc-100 dark:border-zinc-800/50">
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-bold uppercase tracking-[0.05em] text-zinc-400 dark:text-zinc-500">
                {t("chat.mention_category_docs")}
              </span>
            </div>
          </div>

          {/* List - compact single-line rows, height-capped */}
          <div className="max-h-[224px] overflow-y-auto py-1 px-1 scrollbar-none flex flex-col gap-px">
            {docs.map((doc, idx) => {
              const ext = (doc.file_type || doc.filename?.split(".").pop() || "").toLowerCase();
              const fileConfig = getFileConfig(ext);
              const FileIcon = fileConfig.icon;
              const isHighlighted = idx === selectedIndex;

              // Primary label: prefer the extracted title/subject, fall back to filename
              const fileName = doc.original_filename || doc.filename;
              const primary = doc.document_title || fileName;
              // Show the filename inline only when it adds info (title differs)
              const showFilename = !!doc.document_title && fileName !== doc.document_title;

              return (
                <button
                  key={doc.id}
                  type="button"
                  onClick={() => onSelect(doc)}
                  title={[doc.document_title, fileName, doc.issuing_agency].filter(Boolean).join(" — ")}
                  className={cn(
                    "w-full flex items-center gap-2 px-2 py-1.5 rounded-lg transition-all duration-150 text-left relative group",
                    isHighlighted
                      ? "bg-primary/[0.06] dark:bg-primary/[0.12] ring-1 ring-inset ring-primary/15"
                      : "hover:bg-zinc-50 dark:hover:bg-zinc-800/40"
                  )}
                >
                  {/* File-type logo */}
                  <div className={cn(
                    "w-7 h-7 rounded-lg flex items-center justify-center shrink-0",
                    fileConfig.bgColor
                  )}>
                    <FileIcon className={cn("w-[15px] h-[15px]", fileConfig.color)} />
                  </div>

                  {/* Title takes the left, flexes to push the rest right */}
                  <span className={cn(
                    "flex-1 min-w-0 text-[12.5px] font-semibold truncate tracking-tight transition-colors duration-150",
                    isHighlighted ? "text-foreground" : "text-foreground/90 group-hover:text-foreground"
                  )}>
                    {primary}
                  </span>

                  {/* Right side: ref number · type · filename · pages, pinned to the right edge */}
                  {doc.document_number && (
                    <span className={cn(
                      "text-[9px] font-bold font-mono tracking-tight shrink-0 px-1.5 py-0.5 rounded-md border",
                      isHighlighted ? "bg-primary/10 border-primary/25 text-primary" : "bg-zinc-100 dark:bg-zinc-800/50 border-zinc-200 dark:border-zinc-800 text-zinc-500 dark:text-zinc-400"
                    )}>
                      {doc.document_number}
                    </span>
                  )}

                  {doc.document_type_name && (
                    <span className="inline-flex items-center gap-0.5 text-[9.5px] font-medium leading-none shrink-0 text-muted-foreground/55 max-w-[110px]">
                      <Layers className="w-2.5 h-2.5 shrink-0 opacity-70" />
                      <span className="truncate">{doc.document_type_name}</span>
                    </span>
                  )}

                  {showFilename && (
                    <span className="text-[10.5px] text-muted-foreground/60 truncate shrink-0 max-w-[35%]">
                      {fileName}
                    </span>
                  )}

                  {doc.page_count ? (
                    <span className="inline-flex items-center gap-0.5 text-[9.5px] font-medium leading-none shrink-0 text-muted-foreground/55">
                      <FileText className="w-2.5 h-2.5 shrink-0 opacity-70" />
                      {doc.page_count} tr.
                    </span>
                  ) : null}

                  {isHighlighted && (
                    <CornerDownLeft className="w-3 h-3 shrink-0 opacity-40" />
                  )}
                </button>
              );
            })}
          </div>

          {/* Footer - Integrated and small */}
          <div className="px-2.5 py-1 bg-zinc-50/30 dark:bg-white/5 border-t border-border/20 flex items-center gap-3 shrink-0">
            <div className="flex items-center gap-1">
              <kbd className="min-w-[14px] h-3.5 px-0.5 rounded border border-border bg-white dark:bg-zinc-900 text-[7px] font-bold flex items-center justify-center opacity-60">↑↓</kbd>
              <span className="text-[7px] text-muted-foreground/40 font-bold uppercase tracking-wider">{t("common.navigate")}</span>
            </div>
            <div className="flex items-center gap-1">
              <kbd className="min-w-[14px] h-3.5 px-0.5 rounded border border-border bg-white dark:bg-zinc-900 text-[7px] font-bold flex items-center justify-center opacity-60">↵</kbd>
              <span className="text-[7px] text-muted-foreground/40 font-bold uppercase tracking-wider">{t("common.select")}</span>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );

  return createPortal(<AnimatePresence mode="wait">{content}</AnimatePresence>, document.body);
}

// Upload-file button for the chat toolbar. The old multi-item "Tools" dropdown
// was collapsed to this single action — voice input already has its own
// dedicated mic button in the action area, so file upload is the only tool left.
function UploadButton({ onClick }: { onClick?: () => void }) {
  const { t } = useTranslation();

  const label = t("chat.tool_upload") || "Tải lên tệp tin";

  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={cn(
        // Icon-only by default; the label slides open on hover so the toolbar
        // stays compact while still being discoverable.
        "group flex items-center justify-center h-9 px-2 hover:px-3 gap-0 hover:gap-1.5 rounded-full transition-all duration-300 text-[13px] font-bold tracking-tight",
        "text-primary bg-primary/8 ring-1 ring-primary/15 hover:bg-primary/15"
      )}
    >
      <Paperclip className="w-4 h-4 shrink-0" />
      <span className="max-w-0 group-hover:max-w-[160px] overflow-hidden whitespace-nowrap transition-all duration-300">
        {label}
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Chat Input Area — Gemini-style floating card
// ---------------------------------------------------------------------------
export interface AttachedFile {
  id: string; // Document database ID
  file: File;
  status: "uploading" | "parsing" | "ready" | "indexed" | "failed";
  progress: number;
  docMetadata?: Document;
}

export function ChatInputArea({
  input,
  setInput,
  isStreaming,
  onSend,
  onCancel,
  attachedFiles,
  onRemoveAttachment,
  inputRef,
  handleKeyDown,
  onPlus,
  onMic,
  micRecording,
  micTranscribing,
  t,
  referencedDocs,
  onRemoveReferencedDoc,
  showMentionDropdown,
  filteredMentionDocs,
  onSelectMentionDoc,
  onCloseMentionDropdown,
  onInputChange,
  mentionSelectedIndex,
}: {
  input: string;
  setInput: (v: string) => void;
  isStreaming: boolean;
  onSend: () => void;
  onCancel: () => void;
  attachedFiles: AttachedFile[];
  onRemoveAttachment: (id: string) => void;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  onPlus?: () => void;
  onMic?: () => void;
  micRecording?: boolean;
  micTranscribing?: boolean;
  t: any;
  referencedDocs?: { id: string; filename: string; original_filename?: string }[];
  onRemoveReferencedDoc?: (docId: string) => void;
  showMentionDropdown?: boolean;
  filteredMentionDocs?: MentionDoc[];
  onSelectMentionDoc?: (doc: { id: string; filename: string; original_filename?: string }) => void;
  onCloseMentionDropdown?: () => void;
  onInputChange?: (text: string, cursorPos: number) => void;
  mentionSelectedIndex?: number;
}) {
  const highlightRef = useRef<HTMLDivElement>(null);

  const renderHighlightedMentions = () => {
    if (!referencedDocs || referencedDocs.length === 0) return input;

    // Use formatted names for matching
    const formattedNames = referencedDocs.map(d => formatMentionName(d.original_filename || d.filename)).sort((a, b) => b.length - a.length);
    if (formattedNames.length === 0) return input;

    const escapeRegExp = (string: string) => string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(@(?:${formattedNames.map(escapeRegExp).join('|')}))`, 'g');
    const parts = input.split(regex);

    return parts.map((part, i) => {
        if (part.startsWith('@') && formattedNames.includes(part.slice(1))) {
            const displayName = part.slice(1);
            const doc = referencedDocs.find(d => formatMentionName(d.original_filename || d.filename) === displayName);
            const ext = doc ? (doc.original_filename || doc.filename).split('.').pop()?.toLowerCase() : 'pdf';

            let Icon = FileText;
            let iconColor = "text-blue-500/80";
            if (ext === 'pdf') { iconColor = "text-red-500/80"; }
            else if (['doc', 'docx'].includes(ext!)) { iconColor = "text-blue-600/80"; }
            else if (['xls', 'xlsx'].includes(ext!)) { iconColor = "text-green-600/80"; }
            else if (['jpg', 'jpeg', 'png', 'svg'].includes(ext!)) { Icon = ImageIcon; iconColor = "text-purple-500/80"; }
            else if (['ts', 'tsx', 'js', 'jsx'].includes(ext!)) { Icon = FileCode; iconColor = "text-amber-500/80"; }

            return (
                <span key={i} className="relative inline">
                    <span className="relative z-10 text-transparent inline-block">@</span>
                    <span className="absolute z-10 left-[0px] top-[50%] -translate-y-[45%] flex items-center justify-center pointer-events-none">
                        <Icon className={cn("w-[13.5px] h-[13.5px]", iconColor)} />
                    </span>
                    <span className="relative z-10 text-foreground" style={{ WebkitTextStroke: "0.2px currentColor" }}>
                        {displayName}
                    </span>
                </span>
            );
        }
        return <span key={i} className="relative z-20 text-foreground/90">{part}</span>;
    });
  };

  return (
    <div className="relative w-full group">
      <div
        data-chat-input-container="true"
        className="relative flex flex-col bg-background/60 backdrop-blur-3xl border border-border/80 rounded-[28px] shadow-[0_12px_40px_rgb(0,0,0,0.08)] transition-all duration-500 focus-within:shadow-primary/10 focus-within:border-primary/30 overflow-hidden ring-1 ring-black/5 dark:ring-white/5"
      >
        {/* Input Text Area */}
        <div className="px-2.5 pt-3.5 pb-1">
          <div className="relative w-full">
             <div
               ref={highlightRef}
               aria-hidden="true"
               className="absolute inset-0 px-1.5 py-1 pointer-events-none whitespace-pre-wrap break-words overflow-hidden text-[15.5px] font-chat leading-relaxed tracking-tight text-foreground/90"
               style={{ wordBreak: 'break-word' }}
            >
              {renderHighlightedMentions()}
              {input.endsWith('\n') ? <br/> : null}
            </div>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (onInputChange) {
                  onInputChange(e.target.value, e.target.selectionStart || 0);
                }
              }}
              onScroll={(e) => {
                if (highlightRef.current) {
                  highlightRef.current.scrollTop = e.currentTarget.scrollTop;
                  highlightRef.current.scrollLeft = e.currentTarget.scrollLeft;
                }
              }}
              onKeyDown={handleKeyDown}
              placeholder={t("chat.input_placeholder") || "Hỏi tôi bất cứ điều gì, hoặc gõ @ để nhắc đến tài liệu..."}
              rows={1}
              className={cn(
                "relative z-10 w-full resize-none bg-transparent px-1.5 py-1 text-[15.5px] placeholder:text-muted-foreground/45 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50",
                "max-h-[200px] min-h-[38px]",
                "font-chat leading-relaxed tracking-tight text-transparent selection:bg-primary/20 selection:text-transparent",
                "caret-foreground"
              )}
              style={{ height: "auto" }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement;
                target.style.height = "auto";
                target.style.height = Math.min(target.scrollHeight, 200) + "px";
              }}
            />
          </div>
        </div>

        {/* Toolbar Row */}
        <div className="flex items-center gap-2 px-2.5 pb-2.5 pt-0.5">
          <div className="flex items-center gap-2 shrink-0">
            {/* Upload file (voice has its own mic button in the action area) */}
            <UploadButton onClick={onPlus} />
          </div>

          {/* Active quote scope: @mentioned docs + uploaded files — compact chips
              inline with the tools so the input box itself stays short. Horizontally
              scrollable when there are many; persists until the user removes a chip. */}
          {(attachedFiles.length > 0 || (referencedDocs && referencedDocs.length > 0)) && (
            <div
              className="flex-1 min-w-0 max-w-[60%] flex items-center gap-1.5 overflow-x-auto scrollbar-none py-0.5"
              style={{
                // Fade the right edge so it's obvious the strip scrolls when full,
                // while never letting chips butt up against the Send button.
                maskImage: "linear-gradient(to right, black calc(100% - 14px), transparent 100%)",
                WebkitMaskImage: "linear-gradient(to right, black calc(100% - 14px), transparent 100%)",
              }}
            >
              <AnimatePresence initial={false}>
                {/* @mentioned document chips */}
                {(referencedDocs || []).map((doc) => {
                  const name = formatMentionName(doc.original_filename || doc.filename);
                  return (
                    <motion.div
                      key={`ref-${doc.id}`}
                      initial={{ opacity: 0, scale: 0.85 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.85 }}
                      className="flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-lg bg-primary/5 border border-primary/20 shrink-0"
                      title={name}
                    >
                      <FileText className="w-3 h-3 text-primary shrink-0" />
                      <span className="text-[11px] font-medium truncate max-w-[110px] text-foreground">
                        {name}
                      </span>
                      <button
                        type="button"
                        onClick={() => onRemoveReferencedDoc?.(doc.id)}
                        aria-label={t("chat.remove") || "Gỡ"}
                        className="w-3.5 h-3.5 rounded-full flex items-center justify-center text-muted-foreground hover:bg-destructive hover:text-destructive-foreground transition-all shrink-0"
                      >
                        <X className="w-2.5 h-2.5" />
                      </button>
                    </motion.div>
                  );
                })}
                {/* uploaded file chips */}
                {attachedFiles.map((file) => (
                  <motion.div
                    key={file.id}
                    initial={{ opacity: 0, scale: 0.85 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.85 }}
                    className="flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-lg bg-white dark:bg-zinc-900 border border-border shrink-0"
                    title={file.file.name}
                  >
                    {file.status === "uploading" || file.status === "parsing" ? (
                      <Loader2 className="w-3 h-3 animate-spin text-primary shrink-0" />
                    ) : (
                      <FileText className={cn(
                        "w-3 h-3 shrink-0",
                        file.status === "failed" ? "text-destructive" : "text-primary"
                      )} />
                    )}
                    <span className="text-[11px] font-medium truncate max-w-[110px] text-foreground">
                      {file.file.name}
                    </span>
                    <button
                      type="button"
                      onClick={() => onRemoveAttachment(file.id)}
                      aria-label={t("chat.remove") || "Gỡ"}
                      className="w-3.5 h-3.5 rounded-full flex items-center justify-center text-muted-foreground hover:bg-destructive hover:text-destructive-foreground transition-all shrink-0"
                    >
                      <X className="w-2.5 h-2.5" />
                    </button>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}

          <div className="flex items-center gap-2 ml-auto shrink-0">
            {/* Action Button (Mic / Send / Stop) */}
            <div className="ml-1">
              {isStreaming ? (
                <button
                  type="button"
                  onClick={() => onCancel()}
                  aria-label={t("chat.cancel") || "Stop"}
                  className="w-10 h-10 rounded-full flex items-center justify-center bg-destructive/10 text-destructive hover:bg-destructive/15 transition-all shadow-sm ring-1 ring-destructive/20 cursor-pointer"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                </button>
              ) : (micRecording || micTranscribing) ? (
                <button
                  type="button"
                  onClick={onMic}
                  disabled={micTranscribing}
                  aria-label={micTranscribing ? t("chat.transcribing") : t("chat.recording")}
                  className={cn(
                    "w-10 h-10 rounded-full flex items-center justify-center transition-all shadow-sm cursor-pointer",
                    micTranscribing
                      ? "bg-primary/10 text-primary ring-1 ring-primary/20"
                      : "bg-destructive text-destructive-foreground ring-1 ring-destructive/30 animate-pulse"
                  )}
                  title={micTranscribing ? t("chat.transcribing") : t("chat.recording")}
                >
                  {micTranscribing ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Square className="w-3.5 h-3.5 fill-current" />
                  )}
                </button>
              ) : (input.trim() || attachedFiles.some(f => f.status === "indexed")) ? (
                <button
                  type="button"
                  onClick={() => onSend()}
                  className="w-10 h-10 rounded-full flex items-center justify-center bg-primary text-primary-foreground hover:bg-primary/90 hover:scale-105 active:scale-95 transition-all shadow-lg shadow-primary/20 cursor-pointer"
                >
                  <Send className="w-4 h-4 translate-x-0.5" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={onMic}
                  aria-label={t("chat.voice")}
                  className="w-10 h-10 rounded-full flex items-center justify-center text-violet-500 bg-violet-500/10 ring-1 ring-violet-500/20 hover:bg-violet-500/20 hover:scale-105 active:scale-95 transition-all shadow-sm cursor-pointer"
                  title={t("chat.voice")}
                >
                  <Mic className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>
        </div>

        {showMentionDropdown && (
          <DocumentMentionDropdown
            docs={filteredMentionDocs || []}
            onSelect={onSelectMentionDoc || (() => { })}
            onClose={onCloseMentionDropdown || (() => { })}
            selectedIndex={mentionSelectedIndex || 0}
            anchorRef={inputRef}
          />
        )}
      </div>
    </div>
  );
}
