import { useState, Children, isValidElement, type ReactNode } from "react";
import { Brain, FileText, Image as ImageIcon } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";
import { useDocument } from "@/hooks/useDocuments";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import type { ChatSourceChunk, ChatImageRef } from "@/types";
import { shortenDocName } from "@/components/rag/chat/utils";

// ---------------------------------------------------------------------------
// Citation badge — clickable [N] marker → icon + docname-P.N
// ---------------------------------------------------------------------------
function CitationLink({
  index,
  source,
  relatedEntities,
}: {
  index: string;
  source: ChatSourceChunk;
  relatedEntities: string[];
}) {
  const { t } = useTranslation();
  const { activateCitation, activateCitationKG } =
    useWorkspaceStore();
  const { data: doc } = useDocument(source.document_id);

  const isKG = source.source_type === "kg";

  const handleContentClick = () => {
    if (isKG) {
      activateCitationKG(source, relatedEntities, doc);
    } else {
      activateCitation(source, relatedEntities, doc);
    }
  };

  const handleKGClick = () => {
    activateCitationKG(source, relatedEntities, doc);
  };

  if (isKG) {
    // KG source — purple chip with Brain emoji
    return (
      <button
        onClick={handleContentClick}
        className="inline-flex items-center gap-0.5 h-[18px] px-1.5 mx-0.5 text-[10px] font-medium rounded-full bg-purple-400/15 text-purple-500 dark:text-purple-400 hover:bg-purple-400/25 transition-colors align-middle whitespace-nowrap"
        title={t("chat.view_kg")}
      >
        <Brain className="w-2.5 h-2.5 flex-shrink-0" />
        <span>KG-{index}</span>
      </button>
    );
  }

  const docName = doc?.original_filename
    ? shortenDocName(doc.original_filename)
    : t("rag.source") + ` ${index}`;
  const label = source.page_no ? `${docName}-P.${source.page_no}` : docName;

  return (
    <span className="inline-flex gap-0.5 mx-0.5 align-middle">
      <button
        onClick={handleContentClick}
        aria-label={t("chat.view_source", { name: doc?.original_filename || "unknown", page: source.page_no })}
        className="inline-flex items-center gap-0.5 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-primary/12 text-primary hover:bg-primary/20 transition-colors whitespace-nowrap"
      >
        <FileText className="w-2.5 h-2.5 flex-shrink-0" />
        <span>{label}</span>
      </button>
      <button
        onClick={handleKGClick}
        aria-label={t("chat.highlight_kg")}
        className="inline-flex items-center justify-center w-[18px] h-[18px] text-[10px] font-bold rounded-full bg-purple-400/15 text-purple-500 dark:text-purple-400 hover:bg-purple-400/25 transition-colors"
      >
        <Brain className="w-2.5 h-2.5" />
      </button>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Memory citation badge — clickable [MEM-N] → Brain icon + text
// ---------------------------------------------------------------------------
function MemoryCitation({ index }: { index?: string }) {
  return (
    <span
      className="inline-flex items-center justify-center w-[18px] h-[18px] mx-0.5 text-[11px] font-medium rounded-full bg-amber-400/15 text-amber-600 dark:text-amber-400 align-middle"
      title={index || "Thông tin cá nhân của bạn"}
    >
      🧠
    </span>
  );
}

// ---------------------------------------------------------------------------
// Inline image badge — clickable [IMG-N] → icon + docname-P.N with preview
// ---------------------------------------------------------------------------
function InlineImageRef({
  imgRefId,
  imageRef,
}: {
  imgRefId: string;
  imageRef: ChatImageRef;
}) {
  const { t } = useTranslation();
  const [showPreview, setShowPreview] = useState(false);
  const { activateImageCitation } = useWorkspaceStore();
  const { data: doc } = useDocument(imageRef.document_id);

  const handleClick = () => {
    setShowPreview((p) => !p);
    activateImageCitation(imageRef, doc);
  };

  const docName = doc?.original_filename
    ? shortenDocName(doc.original_filename)
    : t("rag.image") + ` ${imgRefId}`;
  const label = `${docName}-P.${imageRef.page_no || "?"}`;

  return (
    <span className="inline-flex flex-col mx-0.5">
      <button
        onClick={handleClick}
        className="inline-flex items-center gap-0.5 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-emerald-400/15 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-400/25 transition-colors align-middle whitespace-nowrap"
        title={imageRef.caption || t("common.page_x", { page: imageRef.page_no })}
      >
        <ImageIcon className="w-2.5 h-2.5 flex-shrink-0" />
        <span>{label}</span>
      </button>
      {showPreview && (
        <a
          href={imageRef.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block mt-1 rounded-md overflow-hidden border bg-white max-w-[280px] hover:border-primary/50 transition-colors"
        >
          <img
            src={imageRef.url}
            alt={imageRef.caption || t("common.page_x", { page: imageRef.page_no })}
            className="w-full h-auto max-h-[180px] object-contain"
          />
          {imageRef.caption && (
            <span className="block px-2 py-1 text-[9px] text-muted-foreground leading-tight border-t bg-muted/30">
              p.{imageRef.page_no} — {imageRef.caption}
            </span>
          )}
        </a>
      )}
    </span>
  );
}

// Citation regex — matches:
//   - New format: [a3z9], [IMG-p4f2], [MEM-xxx]
//   - Grouped: [a3z9, b2m7, IMG-p4f2]
//   - Legacy numeric: [1], [2]
// Does NOT match random bracketed text like [id1], [ref2] — those render as plain text.
const CITATION_RE = /(\[\s*(?:(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+)(?:\s*,\s*(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+))*|\d+)(?:\s*,\s*(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+|\d+))*\s*\])/g;

// Render a plain text string, converting literal <br> / <br/> / <br /> tokens
// (which LLMs love to emit inside markdown table cells) into real line breaks.
// We don't enable rehype-raw — only this single, safe element is ever produced.
function renderTextWithBreaks(text: string, keyPrefix: string): ReactNode {
  const segments = text.split(/<br\s*\/?>/gi);
  if (segments.length === 1) return text;
  const out: ReactNode[] = [];
  segments.forEach((seg, i) => {
    if (seg) out.push(seg);
    if (i < segments.length - 1) out.push(<br key={`${keyPrefix}-br-${i}`} />);
  });
  return out;
}

// Process React children to replace [XXXX] and [IMG-XXXX] with interactive
// components. Supports both new [a3x9] and legacy [1] citation formats.
// Also handles grouped brackets like [a3x9, b2m7] by splitting into individual.
export function injectCitations(
  children: ReactNode,
  sources: ChatSourceChunk[],
  relatedEntities: string[],
  imageRefs?: ChatImageRef[],
  fallbackSources?: ChatSourceChunk[],
): ReactNode {
  return Children.map(children, (child) => {
    // Process string nodes — split on citation patterns
    if (typeof child === "string") {
      const parts = child.split(CITATION_RE);
      if (parts.length === 1) return renderTextWithBreaks(child, "t");
      const result: ReactNode[] = [];
      parts.forEach((part, i) => {
        // Check if this part is a bracket group
        const bracketMatch = part.match(/^\[(.+)\]$/);
        if (!bracketMatch) {
          if (part) result.push(renderTextWithBreaks(part, `p${i}`));
          return;
        }
        // Split on commas for grouped citations [a3x9, b2m7]
        const tokens = bracketMatch[1].split(/,\s*/);
        tokens.forEach((token, ti) => {
          const key = `${i}-${ti}`;
          // Image citation: IMG-xxxx
          const imgMatch = token.match(/^IMG-(.+)$/i);
          if (imgMatch && imageRefs && imageRefs.length > 0) {
            const imgId = imgMatch[1];
            // Match by ref_id first, then fallback to legacy numeric index
            const imageRef =
              imageRefs.find((ir) => ir.ref_id === imgId) ??
              imageRefs[parseInt(imgId, 10) - 1]; // legacy 1-indexed
            if (imageRef) {
              result.push(<InlineImageRef key={key} imgRefId={imgId} imageRef={imageRef} />);
              return;
            }
          }
          // Memory citation: MEM-xxxx — 🧠 emoji only for genuine MEM- citations
          const memMatch = token.match(/^MEM-(.+)$/i);
          if (memMatch) {
            const memId = memMatch[1];
            result.push(<MemoryCitation key={key} index={`MEM-${memId}`} />);
            return;
          }
          // Text citation: match source by index (string or numeric)
          // First try current message's sources, then fallback to historical sources
          const cleanToken = token.trim().toLowerCase();
          const source =
            sources.find((s) => String(s.index).toLowerCase() === cleanToken) ??
            (fallbackSources ? fallbackSources.find((s) => String(s.index).toLowerCase() === cleanToken) : undefined);
          if (source) {
            result.push(
              <CitationLink key={key} index={String(source.index)} source={source} relatedEntities={relatedEntities} />
            );
            return;
          }
          // Truly unmatched — render as-is (no 🧠 for non-MEM citations)
          result.push(`[${token}]`);
        });
      });
      return result;
    }
    // Recurse into React elements that have children
    if (isValidElement(child) && child.props && (child.props as { children?: ReactNode }).children) {
      const props = child.props as { children?: ReactNode };
      return Object.assign({}, child, {
        props: {
          ...child.props,
          children: injectCitations(props.children, sources, relatedEntities, imageRefs, fallbackSources),
        },
      });
    }
    return child;
  });
}
