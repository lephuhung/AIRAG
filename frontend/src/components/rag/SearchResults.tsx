import { memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SearchX, Inbox } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";
import { ResultCard } from "./ResultCard";
import { KGSummary } from "./KGSummary";
import { ImageResultGrid } from "./ImageResultGrid";
import type { RAGQueryResponse } from "@/types";

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------
function ResultSkeleton() {
  return (
    <div className="space-y-3">
      {[1, 2, 3].map((i) => (
        <div key={i} className="rounded-lg border bg-card/40 px-4 py-3 space-y-2 animate-pulse">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded-full bg-muted" />
            <div className="h-3 bg-muted rounded w-32" />
            <div className="h-3 bg-muted rounded w-16 ml-auto" />
          </div>
          <div className="space-y-1.5">
            <div className="h-3 bg-muted rounded w-full" />
            <div className="h-3 bg-muted rounded w-4/5" />
            <div className="h-3 bg-muted rounded w-3/5" />
          </div>
          <div className="h-1 bg-muted rounded-full w-full" />
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty states
// ---------------------------------------------------------------------------
function NoResults({ query }: { query: string }) {
  const { t } = useTranslation();
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-col items-center py-10 text-center"
    >
      <SearchX className="w-10 h-10 text-muted-foreground/40 mb-3" />
      <p className="text-sm font-medium">{t("chat.no_relevant")}</p>
      <p className="text-xs text-muted-foreground mt-1 max-w-xs">
        {t("chat.no_results_desc")} {t("chat.query")}: &quot;{query}&quot;
      </p>
    </motion.div>
  );
}

function SearchPrompt() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center py-10 text-center">
      <Inbox className="w-10 h-10 text-muted-foreground/30 mb-3" />
      <p className="text-sm text-muted-foreground">
        {t("chat.search_prompt")}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SearchResults
// ---------------------------------------------------------------------------
interface SearchResultsProps {
  results: RAGQueryResponse | null;
  isSearching: boolean;
  hasSearched: boolean;
  query: string;
}

export const SearchResults = memo(function SearchResults({
  results,
  isSearching,
  hasSearched,
  query,
}: SearchResultsProps) {
  const { t } = useTranslation();
  if (isSearching) return <ResultSkeleton />;

  if (!hasSearched) return <SearchPrompt />;

  if (results && results.chunks.length === 0) return <NoResults query={query} />;

  if (!results) return null;

  return (
    <div className="space-y-4">
      {/* Summary header */}
      <p className="text-xs text-muted-foreground">
        {t("chat.found_chunks", { count: results.total_chunks, query: results.query })}
      </p>

      {/* Knowledge Graph summary */}
      {results.knowledge_graph_summary && (
        <KGSummary summary={results.knowledge_graph_summary} />
      )}

      {/* Chunk results */}
      <AnimatePresence mode="popLayout">
        <div className="space-y-2">
          {results.chunks.map((chunk, i) => (
            <ResultCard key={chunk.chunk_id} chunk={chunk} index={i} query={query} />
          ))}
        </div>
      </AnimatePresence>

      {/* Image results */}
      {results.image_refs && results.image_refs.length > 0 && (
        <ImageResultGrid images={results.image_refs} />
      )}
    </div>
  );
});
