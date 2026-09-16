// Small string helpers shared across the chat UI (extracted from ChatPanel.tsx).

import type { ChatSourceChunk, PublicCitation } from "@/types";

export const truncateName = (name: string, maxLength = 25) => {
  if (name.length <= maxLength) return name;
  return name.slice(0, maxLength - 8) + "..." + name.slice(-5);
};

export const formatMentionName = (name: string) => {
  const clean = name.replace(/\.[^/.]+$/, "");
  return truncateName(clean, 30);
};

// Shorten filename for citation display (strips extension, adds ellipsis).
export function shortenDocName(filename: string, maxLen = 14): string {
  const name = filename.replace(/\.[^.]+$/, ""); // strip extension
  if (name.length <= maxLen) return name;
  return name.slice(0, maxLen - 1) + "…"; // ellipsis
}

// Project public citations onto the sources presentation model — preserves the
// answer's citation handle (`index`), provenance (`source_type`) and rank
// (`score`); never synthesizes them. Shared by the live `citation` SSE frame
// (useRAGChatStream) and history reload (ChatPanel) so markers resolve to
// badges identically in both paths.
export function citationsToSourceChunks(citations: PublicCitation[]): ChatSourceChunk[] {
  return citations
    .filter((c) => c.document_id && c.chunk_id)
    .map((c) => ({
      index: c.index ?? c.citation_id,
      chunk_id: String(c.chunk_id),
      content: c.content || "",
      document_id: String(c.document_id),
      page_no: c.page_no ?? 0,
      heading_path: c.heading_path || [],
      score: c.score ?? 0,
      source_type: (c.source_type ?? "vector") as ChatSourceChunk["source_type"],
      document_number: c.document_number ?? null,
      article_label: c.article_label ?? null,
      validity_status: c.validity_status ?? null,
      superseded_by: c.superseded_by ?? null,
    }));
}
