import { useContext } from "react";
import { ThumbsUp, ThumbsDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslation } from "@/hooks/useTranslation";
import { useDocument } from "@/hooks/useDocuments";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import type { ChatSourceChunk } from "@/types";
import { DebugCtx } from "@/components/rag/chat/contexts";

export type RelevanceRating = "relevant" | "partial" | "not_relevant";

function SourceRatingButtons({
  sourceIndex,
  currentRating,
  onRate,
}: {
  sourceIndex: string;
  currentRating?: RelevanceRating;
  onRate: (sourceIndex: string, rating: RelevanceRating) => void;
}) {
  return (
    <div
      className="flex items-center gap-0.5 ml-auto flex-shrink-0"
      onClick={(e) => e.stopPropagation()}
    >
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRate(sourceIndex, "relevant");
        }}
        className={cn(
          "p-0.5 rounded transition-colors",
          currentRating === "relevant"
            ? "text-emerald-500"
            : "text-muted-foreground/20 hover:text-emerald-500/60",
        )}
        title="Relevant"
      >
        <ThumbsUp className="w-2.5 h-2.5" />
      </button>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRate(sourceIndex, "not_relevant");
        }}
        className={cn(
          "p-0.5 rounded transition-colors",
          currentRating === "not_relevant"
            ? "text-destructive"
            : "text-muted-foreground/20 hover:text-destructive/60",
        )}
        title="Not relevant"
      >
        <ThumbsDown className="w-2.5 h-2.5" />
      </button>
    </div>
  );
}

// Source item in the sources panel
export function SourceItem({
  source,
  messageId,
  ratings,
  onRate,
  onClosePopover,
}: {
  source: ChatSourceChunk;
  messageId?: string;
  ratings: Record<string, RelevanceRating>;
  onRate: (sourceIndex: string, rating: RelevanceRating) => void;
  onClosePopover?: () => void;
}) {
  const { t } = useTranslation();
  const { activateCitation } = useWorkspaceStore();
  const { data: doc } = useDocument(source.document_id);
  const debugMode = useContext(DebugCtx);

  return (
    <button
      onClick={() => {
        activateCitation(source, [], doc);
        onClosePopover?.();
      }}
      className="w-full text-left px-2.5 py-2 hover:bg-muted/50 transition-colors"
    >
      <div className="flex items-center gap-2 mb-1">
        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 flex-shrink-0" />
        <span className="text-[10px] font-medium text-foreground/80">
          {source.article_label && source.document_number
            ? `${source.article_label} — ${source.document_number}`
            : source.document_number ||
              doc?.original_filename ||
              t("rag.source")}
        </span>
        <span className="text-[10px] text-muted-foreground">p.{source.page_no}</span>
        {source.validity_status === "superseded" && (
          <span
            className="text-[9px] px-1 py-0.5 rounded font-medium bg-destructive/15 text-destructive flex-shrink-0"
            title={
              source.superseded_by
                ? `Đã được thay thế bởi ${source.superseded_by}`
                : "Văn bản đã hết hiệu lực"
            }
          >
            Hết hiệu lực
          </span>
        )}
        {source.validity_status === "partially_amended" && (
          <span
            className="text-[9px] px-1 py-0.5 rounded font-medium bg-amber-400/15 text-amber-600 dark:text-amber-400 flex-shrink-0"
            title="Văn bản đã được sửa đổi/bãi bỏ một phần"
          >
            Đã sửa đổi
          </span>
        )}
        {source.heading_path.length > 0 && (
          <span className="text-[10px] text-muted-foreground/60 truncate">
            {source.heading_path.join(" > ")}
          </span>
        )}
        {messageId && (
          <SourceRatingButtons
            sourceIndex={String(source.index)}
            currentRating={ratings[String(source.index)]}
            onRate={onRate}
          />
        )}
      </div>
      <p className="text-[11px] text-foreground/70 line-clamp-2 leading-relaxed">
        {source.content.slice(0, 150)}
        {source.content.length > 150 ? "..." : ""}
      </p>
      {debugMode && (
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-[8px] px-1 py-0.5 rounded bg-muted font-mono text-muted-foreground/70">
            score: {source.score.toFixed(3)}
          </span>
          <span className="text-[8px] px-1 py-0.5 rounded font-medium bg-blue-400/15 text-blue-400">
            {source.source_type || "vector"}
          </span>
        </div>
      )}
    </button>
  );
}

export function KGSourceItem({
  source,
  messageId,
  ratings,
  onRate,
  onClosePopover,
}: {
  source: ChatSourceChunk;
  messageId?: string;
  ratings: Record<string, RelevanceRating>;
  onRate: (sourceIndex: string, rating: RelevanceRating) => void;
  onClosePopover?: () => void;
}) {
  const { t } = useTranslation();
  const { activateCitationKG } = useWorkspaceStore();
  const { data: doc } = useDocument(source.document_id);
  const debugMode = useContext(DebugCtx);

  return (
    <button
      onClick={() => {
        activateCitationKG(source, [], doc);
        onClosePopover?.();
      }}
      className="w-full text-left px-2.5 py-2 hover:bg-purple-400/5 hover:bg-muted/50 transition-colors"
    >
      <div className="flex items-center gap-2 mb-1">
        <div className="w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
        <span className="text-[10px] font-medium text-purple-600 dark:text-purple-400">
          {t("common.knowledge_graph")}
        </span>
        {messageId && (
          <SourceRatingButtons
            sourceIndex={String(source.index)}
            currentRating={ratings[String(source.index)]}
            onRate={onRate}
          />
        )}
      </div>
      <p className="text-[11px] text-foreground/70 line-clamp-2 leading-relaxed">
        {source.content.slice(0, 150)}
        {source.content.length > 150 ? "..." : ""}
      </p>
      {debugMode && (
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-[8px] px-1 py-0.5 rounded bg-muted font-mono text-muted-foreground/70">
            score: {source.score.toFixed(3)}
          </span>
          <span className="text-[8px] px-1 py-0.5 rounded font-medium bg-purple-400/15 text-purple-400">
            kg
          </span>
        </div>
      )}
    </button>
  );
}
