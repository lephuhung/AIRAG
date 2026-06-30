import { useState, useRef, useEffect, useCallback, useContext, memo, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  ThumbsUp,
  ThumbsDown,
  ClipboardCheck,
  Copy,
  FileCode,
  FileText,
  Loader2,
  Square,
  Volume2,
  Share2,
  RotateCcw,
  Zap,
  BookOpen,
  X,
  DatabaseZap,
  Sparkles,
  User,
  Image as ImageIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatTime } from "@/lib/format";
import { api, rewritePresignedUrl } from "@/lib/api";
import { useTranslation } from "@/hooks/useTranslation";
import { useTTSStore } from "@/hooks/useTTS";
import { useAuthStore } from "@/stores/authStore";
import { STATUS_CONFIG, getFileConfig } from "@/components/rag/document-utils";
import { StreamingMarkdown } from "@/components/rag/MemoizedMarkdown";
import type { ChatMessage, Document, DocumentStatus } from "@/types";
import { SessionIdCtx } from "@/components/rag/chat/contexts";
import { formatMentionName } from "@/components/rag/chat/utils";
import { stripCitations, markdownToPlainText } from "@/components/rag/chat/markdown/text";
import { MarkdownWithCitations } from "@/components/rag/chat/markdown/MarkdownWithCitations";
import { PeopleCard } from "@/components/rag/chat/people/PeopleCard";
import { SourceItem, KGSourceItem } from "@/components/rag/chat/sources/SourceItem";
import type { RelevanceRating } from "@/components/rag/chat/sources/SourceItem";
import { ImageRefsPanel } from "@/components/rag/chat/sources/ImageRefs";
import { PremiumThinking } from "@/components/rag/chat/thinking/PremiumThinking";

// ---------------------------------------------------------------------------
// Copy message actions — plain text or raw markdown (without citations)
// ---------------------------------------------------------------------------
function AssistantMessageFooter({
  message,
}: {
  message: ChatMessage;
}) {
  const { t } = useTranslation();
  const [copiedMode, setCopiedMode] = useState<"text" | "markdown" | null>(null);
  const [showSourcesPopover, setShowSourcesPopover] = useState(false);
  const ttsActiveId = useTTSStore((s) => s.activeId);
  const ttsStatus = useTTSStore((s) => s.status);
  const ttsPlay = useTTSStore((s) => s.play);
  const ttsActive = ttsActiveId === message.id;
  const [ratings, setRatings] = useState<Record<string, RelevanceRating>>({});
  const popoverRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [popoverCoords, setPopoverCoords] = useState<{ bottom: number; right: number } | null>(null);
  const sessionId = useContext(SessionIdCtx);
  const queryClient = useQueryClient();

  // Close popover when clicking outside
  useEffect(() => {
    if (!showSourcesPopover) return;
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setShowSourcesPopover(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showSourcesPopover]);

  // Update popover coordinates when opening or window resizing
  const updateCoords = useCallback(() => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setPopoverCoords({
        bottom: window.innerHeight - rect.top + 8,
        right: window.innerWidth - rect.right - 50, // Shift 50px right as requested
      });
    }
  }, []);

  useEffect(() => {
    if (showSourcesPopover) {
      updateCoords();
      window.addEventListener("resize", updateCoords);
      window.addEventListener("scroll", updateCoords, true);
      return () => {
        window.removeEventListener("resize", updateCoords);
        window.removeEventListener("scroll", updateCoords, true);
      };
    }
  }, [showSourcesPopover, updateCoords]);

  const rateMutation = useMutation({
    mutationFn: ({
      sessionId,
      messageId,
      sourceIndex,
      rating,
    }: {
      sessionId: string;
      messageId: string;
      sourceIndex: string;
      rating: RelevanceRating;
    }) =>
      api.post(`/rag/chat/${sessionId}/rate`, {
        message_id: messageId,
        source_index: sourceIndex,
        rating: rating,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["chat-history", sessionId] });
    },
  });

  const handleRate = useCallback(
    async (sourceIndex: string, rating: RelevanceRating) => {
      if (!sessionId || !message.id) return;

      const newRating = ratings[sourceIndex] === rating ? "partial" : rating;
      const prev = { ...ratings };
      setRatings((r) => ({ ...r, [sourceIndex]: newRating }));

      try {
        await rateMutation.mutateAsync({
          sessionId,
          messageId: message.id,
          sourceIndex,
          rating: newRating,
        });
      } catch {
        setRatings(prev);
      }
    },
    [sessionId, message.id, ratings, rateMutation],
  );

  const handleCopy = useCallback(
    (mode: "text" | "markdown") => {
      const value =
        mode === "text"
          ? markdownToPlainText(message.content)
          : stripCitations(message.content);
      navigator.clipboard.writeText(value).then(() => {
        setCopiedMode(mode);
        setTimeout(() => setCopiedMode(null), 2000);
      });
    },
    [message.content],
  );

  const hasSources = message.sources && message.sources.length > 0;

  return (
    <>
      <div className="flex items-center justify-between gap-1.5 mt-2 pt-1 border-t border-muted/30">
        {/* Action Icons (Left) */}
        <div className="flex items-center gap-1">
          <button className="p-1 rounded-md text-muted-foreground/40 hover:text-muted-foreground hover:bg-muted/60 transition-all">
            <ThumbsUp className="w-3.5 h-3.5" />
          </button>
          <button className="p-1 rounded-md text-muted-foreground/40 hover:text-muted-foreground hover:bg-muted/60 transition-all">
            <ThumbsDown className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => handleCopy("text")}
            className={cn(
              "p-1 rounded-md transition-all",
              copiedMode === "text"
                ? "text-emerald-500 bg-emerald-500/5"
                : "text-muted-foreground/40 hover:text-muted-foreground hover:bg-muted/60",
            )}
            aria-label={t("chat.copy_text")}
          >
            {copiedMode === "text" ? (
              <ClipboardCheck className="w-3.5 h-3.5" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
          </button>
          <button
            onClick={() => handleCopy("markdown")}
            className={cn(
              "p-1 rounded-md transition-all",
              copiedMode === "markdown"
                ? "text-emerald-500 bg-emerald-500/5"
                : "text-muted-foreground/40 hover:text-muted-foreground hover:bg-muted/60",
            )}
            aria-label={t("chat.copy_markdown")}
          >
            {copiedMode === "markdown" ? (
              <ClipboardCheck className="w-3.5 h-3.5" />
            ) : (
              <FileCode className="w-3.5 h-3.5" />
            )}
          </button>
          <button
            onClick={() => ttsPlay(message.id, markdownToPlainText(message.content))}
            className={cn(
              "p-1 rounded-md transition-all",
              ttsActive
                ? "text-emerald-500 bg-emerald-500/5"
                : "text-muted-foreground/40 hover:text-muted-foreground hover:bg-muted/60",
            )}
            aria-label={t("chat.read_aloud")}
            title={t("chat.read_aloud")}
          >
            {ttsActive && ttsStatus === "loading" ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : ttsActive && ttsStatus === "playing" ? (
              <Square className="w-3.5 h-3.5" />
            ) : (
              <Volume2 className="w-3.5 h-3.5" />
            )}
          </button>
          <button className="p-1 rounded-md text-muted-foreground/40 hover:text-muted-foreground hover:bg-muted/60 transition-all">
            <Share2 className="w-3.5 h-3.5" />
          </button>
          <button className="p-1 rounded-md text-muted-foreground/40 hover:text-muted-foreground hover:bg-muted/60 transition-all">
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Metadata & Sources (Right) */}
        <div className="flex items-center gap-2">

          <div className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-[11px] font-semibold tracking-wide">
            <Zap className="w-3.5 h-3.5 fill-current" />
            <span>Fast</span>
          </div>

          {hasSources && (
            <div className="relative">
              <button
                ref={buttonRef}
                onClick={() => setShowSourcesPopover((v) => !v)}
                className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-primary/10 border border-primary/20 text-primary hover:bg-primary/20 transition-colors text-[10px] font-semibold"
              >
                <BookOpen className="w-3 h-3" />
                <span>
                  {message.sources!.length} {t("rag.sources")}
                </span>
              </button>

              {/* Portal-based Floating Popover — bypasses ChatPanel overflow constraints */}
              {typeof document !== "undefined" && createPortal(
                <AnimatePresence>
                  {showSourcesPopover && popoverCoords && (
                    <motion.div
                      ref={popoverRef}
                      initial={{ opacity: 0, scale: 0.95, y: 8 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95, y: 8 }}
                      transition={{ duration: 0.15, ease: "easeOut" }}
                      className="fixed w-80 max-h-[360px] overflow-hidden bg-background/95 backdrop-blur-sm border rounded-xl shadow-2xl z-[9999] flex flex-col origin-bottom-right"
                      style={{
                        bottom: popoverCoords.bottom,
                        right: popoverCoords.right,
                      }}
                    >
                      <div className="flex-shrink-0 flex items-center justify-between px-3 py-2 border-b bg-muted/30">
                        <div className="flex items-center gap-2">
                          <FileText className="w-3.5 h-3.5 text-primary" />
                          <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/80">{t("rag.sources")}</span>
                          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-primary/10 text-primary font-bold">
                            {message.sources!.length}
                          </span>
                        </div>
                        <button
                          onClick={() => setShowSourcesPopover(false)}
                          className="p-1 rounded-md hover:bg-muted transition-colors"
                        >
                          <X className="w-3 h-3 text-muted-foreground" />
                        </button>
                      </div>

                      <div className="flex-1 overflow-y-auto divide-y divide-muted/50 scrollbar-none">
                        {message.sources!
                          .filter((s) => s.source_type !== "kg")
                          .map((source) => (
                            <SourceItem
                              key={String(source.index)}
                              source={source}
                              messageId={message.id}
                              ratings={ratings}
                              onRate={handleRate}
                              onClosePopover={() => setShowSourcesPopover(false)}
                            />
                          ))}
                        {message.sources!
                          .filter((s) => s.source_type === "kg")
                          .map((source) => (
                            <KGSourceItem
                              key={String(source.index)}
                              source={source}
                              messageId={message.id}
                              ratings={ratings}
                              onRate={handleRate}
                              onClosePopover={() => setShowSourcesPopover(false)}
                            />
                          ))}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>,
                document.body
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// Helper: Add Abbreviation Button
function AddAbbreviationButton({
  shortForm,
  onClick,
}: {
  shortForm: string;
  onClick: (s: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <button
      onClick={() => onClick(shortForm)}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-primary/8 border border-primary/20 text-primary text-[12px] font-medium hover:bg-primary/15 transition-colors shadow-sm suggestion-chip-hover"
    >
      <DatabaseZap className="w-3.5 h-3.5" />
      <span>
        {t("chat.add_abbreviation", { abbr: shortForm })}
      </span>
    </button>
  );
}

// File attachment badge for inline display in message bubbles
function FileAttachmentBadge({ doc }: { doc: { id: string; filename: string; original_filename?: string; file_type?: string; status: string } }) {
  const { t } = useTranslation();
  const displayName = doc.original_filename || doc.filename;
  const fileConfig = getFileConfig(doc.file_type || doc.filename?.split(".").pop() || "");
  const FileIcon = fileConfig.icon;
  const statusConfig = STATUS_CONFIG[doc.status as DocumentStatus] ?? STATUS_CONFIG.pending;
  const StatusIcon = statusConfig.icon;
  const isProcessing = ["parsing", "ocring", "chunking", "embedding", "building_kg"].includes(doc.status);

  return (
    <div className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-muted/60 border border-border/50 text-xs">
      <FileIcon className={cn("w-3.5 h-3.5 shrink-0", fileConfig.color)} />
      <span className="truncate max-w-[120px] font-medium text-foreground/80" title={displayName}>
        {displayName}
      </span>
      {doc.status !== "indexed" && (
        <span className={cn(
          "inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide",
          statusConfig.className,
        )}>
          <StatusIcon className={cn("w-2.5 h-2.5 shrink-0", isProcessing && "animate-spin")} />
          {t(statusConfig.labelKey)}
        </span>
      )}
    </div>
  );
}

export const MessageBubble = memo(function MessageBubble({
  message,
  onAddAbbreviation,
  docMetadataMap,
}: {
  message: ChatMessage;
  onAddAbbreviation: (short: string) => void;
  docMetadataMap?: Map<string, Document>;
}) {
  const { t } = useTranslation();
  const isUser = message.role === "user";
  const user = useAuthStore((s) => s.user);

  const initials = user?.full_name
    ?.split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  const proseClasses = cn(
    "prose max-w-none text-foreground/90 font-chat text-[15px] leading-[1.65]",
    "[&_p]:my-2 [&_p]:text-justify [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-1.5 [&_li]:text-justify",
    "[&_pre]:bg-zinc-950/50 [&_pre]:dark:bg-black/40 [&_pre]:border [&_pre]:border-border/50 [&_pre]:rounded-xl [&_pre]:p-4 [&_pre]:my-4",
    "[&_code]:bg-muted/60 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded-md [&_code]:text-[13px] [&_code]:text-foreground/90 [&_code]:font-medium",
    "[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2 [&_a]:decoration-primary/30 hover:[&_a]:decoration-primary transition-all",
    "[&_strong]:text-foreground [&_strong]:font-bold",
    "[&_h1]:text-foreground [&_h1]:text-lg [&_h1]:font-bold [&_h1]:mt-6 [&_h1]:mb-3 [&_h1]:tracking-tight",
    "[&_h2]:text-foreground [&_h2]:text-base [&_h2]:font-semibold [&_h2]:mt-5 [&_h2]:mb-2.5 [&_h2]:tracking-tight",
    "[&_h3]:text-foreground [&_h3]:text-[15px] [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-2 [&_h3]:tracking-tight",
    "[&_blockquote]:border-l-3 [&_blockquote]:border-primary/40 [&_blockquote]:bg-primary/5 [&_blockquote]:px-4 [&_blockquote]:py-1 [&_blockquote]:italic [&_blockquote]:text-foreground/70 [&_blockquote]:rounded-r-lg",
    "[&_table]:text-[13px] [&_table]:border-collapse [&_th]:border [&_th]:border-border/60 [&_td]:border [&_td]:border-border/60 [&_th]:bg-muted/40 [&_th]:px-3 [&_th]:py-2 [&_td]:px-3 [&_td]:py-2",
    "[&_li]:text-foreground/90",
    "[&_.katex-display]:overflow-x-auto [&_.katex-display]:py-3",
    "[&_.katex]:text-[1.05em]"
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.19, 1, 0.22, 1] }}
      className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}
    >
      {!isUser && (
        <div className="flex-shrink-0 mt-1">
          <div className="w-8 h-8 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
        </div>
      )}

      <div
        className={cn(
          isUser
            ? "max-w-[85%] rounded-2xl px-4 py-3 bg-secondary/40 border border-border/40 shadow-sm"
            : "max-w-[90%] min-w-0 py-1"
        )}
      >
        {/* Modern Unified Reasoning Orchestrator */}
        {!isUser && (
          <div className="flex flex-col gap-2">
            <PremiumThinking
              thinking={message.thinking || message.agentSteps?.find(s => s.thinkingText)?.thinkingText || ""}
              agentSteps={message.agentSteps}
              isStreaming={message.isStreaming}
              hasContent={!!message.content}
            />
          </div>
        )}

        {isUser ? (
          (() => {
            const allAvailableDocs = message.documentIds?.map(id => (docMetadataMap && docMetadataMap.get(id)) || message.attachedDocs?.find(d => d.id === id)).filter(Boolean) as any[] || [];
            const mentionedDocIds = new Set<string>();
            let elements: ReactNode[] = [message.content];

            if (allAvailableDocs.length > 0) {
              for (const doc of allAvailableDocs) {
                const truncatedDisplayName = formatMentionName(doc.original_filename || doc.filename);
                const docIdTag = `<document_id=${doc.id}>`;
                const escapeRegExp = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

                const regexStr = `(?:${escapeRegExp(docIdTag)}|@${escapeRegExp(truncatedDisplayName)})`;
                const mentionRegex = new RegExp(`(${regexStr})`, 'g');

                let foundMatch = false;

                const newElements: ReactNode[] = [];
                elements.forEach((el, index) => {
                  if (typeof el === 'string') {
                    const parts = el.split(mentionRegex);

                    parts.forEach((part, i) => {
                      if (part === docIdTag || part === `@${truncatedDisplayName}`) {
                        foundMatch = true;
                        const ext = (doc.file_type || doc.filename?.split(".").pop() || "").toLowerCase();
                        let Icon = FileText;
                        let iconColor = "text-blue-500/80";
                        if (ext === 'pdf') { iconColor = "text-red-500/80"; }
                        else if (['doc', 'docx'].includes(ext)) { iconColor = "text-blue-600/80"; }
                        else if (['xls', 'xlsx'].includes(ext)) { iconColor = "text-green-600/80"; }
                        else if (['jpg', 'jpeg', 'png', 'svg'].includes(ext)) { Icon = ImageIcon; iconColor = "text-purple-500/80"; }
                        else if (['ts', 'tsx', 'js', 'jsx'].includes(ext)) { Icon = FileCode; iconColor = "text-amber-500/80"; }

                        newElements.push(
                          <span key={`${index}-${i}-mention`} className="inline-flex items-center gap-1 mx-1 px-2 rounded-lg bg-primary/10 border border-primary/20 text-[14px]">
                            <Icon className={cn("w-3.5 h-3.5", iconColor)} />
                            <span className="text-foreground" style={{ WebkitTextStroke: "0.2px currentColor" }}>{truncatedDisplayName}</span>
                          </span>
                        );
                      } else if (part) {
                        newElements.push(part);
                      }
                    });
                  } else {
                    newElements.push(el);
                  }
                });
                elements = newElements;
                if (foundMatch) mentionedDocIds.add(doc.id);
              }
            }

            const remainingDocs = allAvailableDocs.filter(d => !mentionedDocIds.has(d.id));

            return (
              <>
                <p className="text-[15px] leading-[1.65] whitespace-pre-wrap font-chat text-foreground/90">
                  {elements}
                </p>
                {remainingDocs.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-3">
                    {remainingDocs.map(doc => (
                      <FileAttachmentBadge key={doc.id} doc={doc} />
                    ))}
                  </div>
                )}
              </>
            );
          })()
        ) : message.isStreaming ? (
          message.peopleData && message.peopleData.length > 0 ? null : message.content ? (
            <div
              className={cn(proseClasses, "relative")}
              style={{
                maskImage: "linear-gradient(to bottom, black calc(100% - 80px), transparent 100%)",
                WebkitMaskImage: "linear-gradient(to bottom, black calc(100% - 80px), transparent 100%)",
              }}
            >
              <StreamingMarkdown
                content={message.content}
                isStreaming
                renderBlock={(block) => (
                  <MarkdownWithCitations
                    content={block}
                    sources={message.sources || []}
                    relatedEntities={message.relatedEntities || []}
                    imageRefs={message.imageRefs}
                  />
                )}
              />
              <span className="streaming-cursor" />
            </div>
          ) : null
        ) : message.peopleData && message.peopleData.length > 0 ? null : (
          <div className={proseClasses}>
            <MarkdownWithCitations
              content={message.content}
              sources={message.sources || []}
              relatedEntities={message.relatedEntities || []}
              imageRefs={message.imageRefs}
            />
          </div>
        )}

        {/* People Card — structured display for people search results */}
        {!isUser && message.peopleData && message.peopleData.length > 0 && (
          <PeopleCard people={message.peopleData} isLoadingMore={message.isStreaming} />
        )}

        {/* Potential Abbreviation Suggestion Buttons */}
        {!isUser && !message.isStreaming && message.potential_abbreviations && message.potential_abbreviations.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {message.potential_abbreviations.map((abbr) => (
              <AddAbbreviationButton
                key={abbr}
                shortForm={abbr}
                onClick={onAddAbbreviation}
              />
            ))}
          </div>
        )}

        {/* Footer actions for assistant messages */}
        {!isUser && message.content && (
          <AssistantMessageFooter message={message} />
        )}


        {!isUser && !message.isStreaming && message.imageRefs && message.imageRefs.length > 0 && (
          <ImageRefsPanel images={message.imageRefs} />
        )}

        <p
          className={cn(
            "text-[9px] mt-1",
            isUser ? "text-muted-foreground/50" : "text-muted-foreground/50"
          )}
        >
          {formatTime(message.timestamp)}
        </p>
      </div>

      {isUser && (
        <div
          className={cn(
            "w-8 h-8 rounded-full overflow-hidden flex items-center justify-center text-[10px] font-semibold flex-shrink-0 mt-0.5 avatar-ring cursor-pointer transition-all duration-200",
            user?.avatar_url
              ? "ring-2 ring-primary/25 shadow-sm"
              : "bg-secondary/80 border border-border/50 text-muted-foreground"
          )}
          title={user?.full_name || t("common.you")}
        >
          {user?.avatar_url ? (
            <img
              src={rewritePresignedUrl(user.avatar_url)}
              alt={user.full_name || "User"}
              className="w-full h-full object-cover"
            />
          ) : (
            initials || <User className="w-4 h-4" />
          )}
        </div>
      )}
    </motion.div>
  );
});
