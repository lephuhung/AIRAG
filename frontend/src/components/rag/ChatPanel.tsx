import { useState, useRef, useEffect, useCallback, useMemo, memo, createContext, useContext, Children, isValidElement, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  Send,
  Square,
  User,
  Loader2,
  Sparkles,
  FileText,
  ImageIcon,
  Brain,
  ChevronDown,
  Copy,
  ClipboardCheck,
  FileCode,
  ThumbsUp,
  ThumbsDown,
  DatabaseZap,
  X,
  Share2,
  RotateCcw,
  Zap,
  BookOpen,
  Plus,
  Mic,
  Settings2,
  Music,
  GraduationCap,
  Pencil,
} from "lucide-react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import css from "react-syntax-highlighter/dist/esm/languages/prism/css";
import markup from "react-syntax-highlighter/dist/esm/languages/prism/markup";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import cpp from "react-syntax-highlighter/dist/esm/languages/prism/cpp";
import diff from "react-syntax-highlighter/dist/esm/languages/prism/diff";
import markdown from "react-syntax-highlighter/dist/esm/languages/prism/markdown";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { api, rewritePresignedUrl } from "@/lib/api";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useAuthStore } from "@/stores/authStore";
import { useThemeStore } from "@/stores/useThemeStore";

SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("ts", typescript);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("shell", bash);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("css", css);
SyntaxHighlighter.registerLanguage("html", markup);
SyntaxHighlighter.registerLanguage("xml", markup);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("yml", yaml);
SyntaxHighlighter.registerLanguage("java", java);
SyntaxHighlighter.registerLanguage("go", go);
SyntaxHighlighter.registerLanguage("cpp", cpp);
SyntaxHighlighter.registerLanguage("c", cpp);
SyntaxHighlighter.registerLanguage("diff", diff);
SyntaxHighlighter.registerLanguage("markdown", markdown);
SyntaxHighlighter.registerLanguage("md", markdown);
import { useDocument } from "@/hooks/useDocuments";
import { useChatHistory } from "@/hooks/useChatHistory";
import { useRAGChatStream } from "@/hooks/useRAGChatStream";
import { useTranslation } from "@/hooks/useTranslation";
import { useCreateAbbreviation } from "@/hooks/useAbbreviations";
import { AbbreviationModal } from "@/components/rag/AbbreviationModal";
import { StreamingMarkdown } from "@/components/rag/MemoizedMarkdown";
import { STEP_CONFIG } from "@/components/rag/ThinkingTimeline";
import type {
  ChatMessage,
  ChatImageRef,
  ChatSourceChunk,
  ChatStreamStatus,
  LLMCapabilities,
  AgentStep,
  AgentStepType,
} from "@/types";

// Context to provide sessionId and debugMode to nested components
const SessionIdCtx = createContext<string | null>(null);
const DebugCtx = createContext(false);

// Context: accumulated sources from ALL messages in the conversation.
// Used as fallback when a message references citation IDs from previous turns.
const AllSourcesCtx = createContext<ChatSourceChunk[]>([]);

// ---------------------------------------------------------------------------
// Helper: shorten filename for citation display
// ---------------------------------------------------------------------------
function shortenDocName(filename: string, maxLen = 14): string {
  const name = filename.replace(/\.[^.]+$/, ""); // strip extension
  if (name.length <= maxLen) return name;
  return name.slice(0, maxLen - 1) + "\u2026"; // ellipsis
}

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
  const label = `${docName}-P.${source.page_no || "?"}`;

  return (
    <span className="inline-flex gap-0.5 mx-0.5 align-middle">
      <button
        onClick={handleContentClick}
        className="inline-flex items-center gap-0.5 h-[18px] px-1.5 text-[10px] font-medium rounded-full bg-primary/12 text-primary hover:bg-primary/20 transition-colors whitespace-nowrap"
        title={t("chat.view_source", { name: doc?.original_filename || "unknown", page: source.page_no })}
      >
        <FileText className="w-2.5 h-2.5 flex-shrink-0" />
        <span>{label}</span>
      </button>
      <button
        onClick={handleKGClick}
        className="inline-flex items-center justify-center w-[18px] h-[18px] text-[10px] font-bold rounded-full bg-purple-400/15 text-purple-500 dark:text-purple-400 hover:bg-purple-400/25 transition-colors"
        title={t("chat.highlight_kg")}
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

// ---------------------------------------------------------------------------
// Process React children to replace [XXXX] and [IMG-XXXX] with interactive
// components. Supports both new [a3x9] and legacy [1] citation formats.
// Also handles grouped brackets like [a3x9, b2m7] by splitting into individual.
// ---------------------------------------------------------------------------
// Citation regex — matches:
//   - New format: [a3z9], [IMG-p4f2], [MEM-xxx]
//   - Grouped: [a3z9, b2m7, IMG-p4f2]
//   - Legacy numeric: [1], [2]
// Does NOT match random bracketed text like [id1], [ref2] — those render as plain text.
const CITATION_RE = /(\[\s*(?:(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+)(?:\s*,\s*(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+))*|\d+)(?:\s*,\s*(?:[a-zA-Z0-9]{2,6}|IMG-[a-zA-Z0-9]+|MEM-[a-zA-Z0-9_-]+|\d+))*\s*\])/g;

function injectCitations(
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
      if (parts.length === 1) return child;
      const result: ReactNode[] = [];
      parts.forEach((part, i) => {
        // Check if this part is a bracket group
        const bracketMatch = part.match(/^\[(.+)\]$/);
        if (!bracketMatch) {
          if (part) result.push(part);
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

// ---------------------------------------------------------------------------
// Preprocess markdown: fix common LLM output issues
// ---------------------------------------------------------------------------
function preprocessMarkdown(text: string): string {
  const lines = text.split("\n");
  const result: string[] = [];
  let prevWasTable = false;
  let inCodeFence = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      inCodeFence = !inCodeFence;
    }

    const isTable = (trimmed.startsWith("|") && trimmed.endsWith("|")) ||
      /^\|[\s:|-]+\|$/.test(trimmed);

    // Insert blank line when transitioning from table row to non-table content
    if (prevWasTable && !isTable && trimmed !== "") {
      result.push("");
    }

    // Convert single-line display math $$content$$ to multi-line format
    if (
      !inCodeFence &&
      trimmed.startsWith("$$") &&
      trimmed.endsWith("$$") &&
      trimmed.length > 4 &&
      trimmed !== "$$"
    ) {
      const mathContent = trimmed.slice(2, -2);
      result.push("$$");
      result.push(mathContent);
      result.push("$$");
    } else {
      result.push(line);
    }

    prevWasTable = isTable;
  }

  // Convert memory section markers to a styled markdown heading for ReactMarkdown.
  // Backend emits "[Memory]" (current) — also handle legacy "<memory_section>" tag.
  let processed = result.join("\n");
  processed = processed.replace(/\[Memory\]/gi, "\n---\n🧠 ");
  processed = processed.replace(/<memory_section>/gi, "\n---\n🧠 ");

  return processed;
}

// ---------------------------------------------------------------------------
// Extract raw text from React node tree (for code blocks)
// ---------------------------------------------------------------------------
function extractText(node: ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (!node) return "";
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (isValidElement(node)) {
    const props = node.props as { children?: ReactNode };
    return extractText(props.children);
  }
  return "";
}

// ---------------------------------------------------------------------------
// Code block with syntax highlighting + copy button
// ---------------------------------------------------------------------------
function CodeBlock({
  language,
  children,
}: {
  language: string;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const theme = useThemeStore((s) => s.theme);
  const isDark = theme === "dark";
  const code = extractText(children).replace(/\n$/, "");

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className="group relative my-2">
      {language && (
        <span className="absolute top-2 right-2 text-[9px] uppercase text-muted-foreground/40 font-mono select-none z-10 pointer-events-none">
          {language}
        </span>
      )}
      <button
        onClick={handleCopy}
        className={cn(
          "absolute top-2 left-2 p-1 rounded-md text-muted-foreground/50 hover:text-muted-foreground transition-all opacity-0 group-hover:opacity-100 z-10",
          isDark ? "bg-white/5 hover:bg-white/10" : "bg-black/5 hover:bg-black/10"
        )}
        title={t("chat.copy_code")}
      >
        {copied ? (
          <ClipboardCheck className="w-3 h-3 text-emerald-500" />
        ) : (
          <Copy className="w-3 h-3" />
        )}
      </button>
      <SyntaxHighlighter
        language={language}
        style={isDark ? oneDark : oneLight}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: "8px",
          fontSize: "12px",
          padding: "10px 12px",
          ...(isDark
            ? {
              background: "oklch(0.18 0.015 155)",
              border: "1px solid oklch(0.30 0.025 155)",
            }
            : {
              background: "oklch(0.96 0.008 105)",
              border: "1px solid oklch(0.88 0.018 105)",
            }),
        }}
        codeTagProps={{ style: { fontFamily: '"IBM Plex Mono", "Fira Code", monospace' } }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Markdown renderer with inline citation links + LaTeX + code blocks
// ---------------------------------------------------------------------------
function MarkdownWithCitations({
  content,
  sources,
  relatedEntities,
  imageRefs,
}: {
  content: string;
  sources: ChatSourceChunk[];
  relatedEntities: string[];
  imageRefs?: ChatImageRef[];
}) {
  const processed = preprocessMarkdown(content);

  // Fallback: accumulated sources from all messages in the conversation.
  // When the model references citation IDs from previous answers (e.g. when
  // it didn't call search_documents), we can still render them as links.
  const allSources = useContext(AllSourcesCtx);

  // Create a wrapper component that injects citations into rendered children
  const withCitations = (Tag: string) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return ({ children, ...props }: any) => {
      const injected = injectCitations(children, sources, relatedEntities, imageRefs, allSources);
      return <Tag {...props}>{injected}</Tag>;
    };
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        p: withCitations("p"),
        li: withCitations("li"),
        td: withCitations("td"),
        th: withCitations("th"),
        h1: withCitations("h1"),
        h2: withCitations("h2"),
        h3: withCitations("h3"),
        h4: withCitations("h4"),
        h5: withCitations("h5"),
        h6: withCitations("h6"),
        strong: withCitations("strong"),
        em: withCitations("em"),
        a: ({ href, children, ...props }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
            {injectCitations(children, sources, relatedEntities, imageRefs, allSources)}
          </a>
        ),
        // Code block — delegate to CodeBlock for syntax highlighting
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        code: ({ className, children, ...props }: any) => {
          const langMatch = /language-(\w+)/.exec(className || "");
          // Inline code (no language class)
          if (!langMatch) {
            return <code className={className} {...props}>{children}</code>;
          }
          // Fenced code block → syntax highlighted
          return <CodeBlock language={langMatch[1]}>{children}</CodeBlock>;
        },
      }}
    >
      {processed}
    </ReactMarkdown>
  );
}

// ---------------------------------------------------------------------------
// Source Rating Buttons
// ---------------------------------------------------------------------------
type RelevanceRating = "relevant" | "partial" | "not_relevant";

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

// ---------------------------------------------------------------------------
// Source item in the sources panel
// ---------------------------------------------------------------------------
function SourceItem({
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
          {doc?.original_filename || t("rag.source")}
        </span>
        <span className="text-[10px] text-muted-foreground">p.{source.page_no}</span>
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

function KGSourceItem({
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Image references panel — shows retrieved images in chat
// ---------------------------------------------------------------------------
function ImageRefCard({ img }: { img: ChatImageRef }) {
  const { activateImageCitation } = useWorkspaceStore();
  const { data: doc } = useDocument(img.document_id);
  return (
    <button
      onClick={() => activateImageCitation(img, doc)}
      className="group block rounded-md overflow-hidden border bg-background hover:border-primary/50 transition-colors text-left cursor-pointer"
    >
      <img
        src={img.url}
        alt={img.caption || `Image from page ${img.page_no}`}
        className="w-full h-auto max-h-[200px] object-contain bg-white"
        loading="lazy"
      />
      {img.caption && (
        <p className="px-2 py-1 text-[10px] text-muted-foreground leading-tight line-clamp-2 border-t">
          p.{img.page_no} — {img.caption}
        </p>
      )}
    </button>
  );
}

function ImageRefsPanel({ images }: { images: ChatImageRef[] }) {
  const { t } = useTranslation();

  if (images.length === 0) return null;

  const [expanded, setExpanded] = useState(true);

  return (
    <div className="mt-2 rounded-md border bg-muted/20 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors"
      >
        <ImageIcon className="w-3 h-3" />
        {t("chat.images_from_docs", { count: images.length })}
        <span className="ml-auto text-[10px]">{expanded ? "▲" : "▼"}</span>
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: "auto" }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="p-2 grid gap-2" style={{ gridTemplateColumns: images.length === 1 ? "1fr" : "repeat(auto-fit, minmax(140px, 1fr))" }}>
              {images.map((img) => (
                <ImageRefCard key={img.image_id} img={img} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Thinking panel — collapsible violet-themed thinking process display
// ---------------------------------------------------------------------------
function ThinkingPanel({ thinking }: { thinking: string }) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);

  if (!thinking) return null;

  return (
    <div className="mt-1.5 mb-1 rounded-md border border-violet-500/20 bg-violet-500/5 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-violet-400 hover:text-violet-300 [[data-theme='light']_&]:text-violet-600 [[data-theme='light']_&]:hover:text-violet-700 transition-colors"
      >
        <Brain className="w-3 h-3" />
        {t("chat.thinking_process")}
        <ChevronDown
          className={cn(
            "w-3 h-3 ml-auto transition-transform",
            expanded && "rotate-180"
          )}
        />
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: "auto" }}
            exit={{ height: 0 }}
            className="overflow-hidden"
          >
            <div className="px-2.5 pb-2 border-t border-violet-500/10">
              <pre className="text-[11px] text-violet-300/90 [[data-theme='light']_&]:text-violet-700/90 whitespace-pre-wrap leading-relaxed mt-1.5 max-h-[300px] overflow-y-auto">
                {thinking}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Copy message actions — plain text or raw markdown (without citations)
// ---------------------------------------------------------------------------
const CITATION_STRIP_RE = /\s*\[(?:[a-z0-9]+|IMG-[a-z0-9]+)(?:,\s*(?:[a-z0-9]+|IMG-[a-z0-9]+))*\]/g;

/** Remove citation references like [a3x9], [IMG-p4f2], [a3x9, b2m7] */
function stripCitations(md: string): string {
  return md.replace(CITATION_STRIP_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}

/** Convert markdown to plain text: strip formatting, links, images, code fences */
function markdownToPlainText(md: string): string {
  let text = stripCitations(md);
  text = text.replace(/```[\s\S]*?```/g, (m) => {
    const lines = m.split("\n");
    return lines.slice(1, -1).join("\n");
  });
  text = text.replace(/`([^`]+)`/g, "$1");
  text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1");
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  text = text.replace(/\*\*(.+?)\*\*/g, "$1");
  text = text.replace(/\*(.+?)\*/g, "$1");
  text = text.replace(/__(.+?)__/g, "$1");
  text = text.replace(/_(.+?)_/g, "$1");
  text = text.replace(/^#{1,6}\s+/gm, "");
  text = text.replace(/^[-*_]{3,}\s*$/gm, "");
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim();
}

function AssistantMessageFooter({
  message,
}: {
  message: ChatMessage;
}) {
  const { t } = useTranslation();
  const [copiedMode, setCopiedMode] = useState<"text" | "markdown" | null>(null);
  const [showSourcesPopover, setShowSourcesPopover] = useState(false);
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
          title={t("chat.copy_text")}
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
          title={t("chat.copy_markdown")}
        >
          {copiedMode === "markdown" ? (
            <ClipboardCheck className="w-3.5 h-3.5" />
          ) : (
            <FileCode className="w-3.5 h-3.5" />
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

// ---------------------------------------------------------------------------
// Helper: Add Abbreviation Button
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Single message bubble
// ---------------------------------------------------------------------------
const MessageBubble = memo(function MessageBubble({
  message,
  onAddAbbreviation,
}: {
  message: ChatMessage;
  onAddAbbreviation: (short: string) => void;
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
    "prose max-w-none text-foreground/90 font-chat text-[15.5px] leading-relaxed",
    "[&_p]:my-1.5 [&_p]:text-justify [&_ul]:my-1.5 [&_ol]:my-1.5 [&_li]:my-1 [&_li]:text-justify",
    "[&_pre]:bg-transparent [&_pre]:border-none [&_pre]:p-0 [&_pre]:m-0",
    "[&_code]:bg-muted/50 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-[13px] [&_code]:text-foreground/90 [&_code]:font-[400]",
    "[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2",
    "[&_strong]:text-foreground [&_em]:text-foreground/80",
    "[&_h1]:text-foreground [&_h2]:text-foreground [&_h3]:text-foreground [&_h4]:text-foreground",
    "[&_h1]:text-base [&_h1]:font-bold [&_h1]:mt-4 [&_h1]:mb-2 [&_h1]:tracking-tight",
    "[&_h2]:text-[15px] [&_h2]:font-semibold [&_h2]:mt-3 [&_h2]:mb-1.5 [&_h2]:tracking-tight",
    "[&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2.5 [&_h3]:mb-1 [&_h3]:tracking-tight",
    "[&_blockquote]:border-l-2 [&_blockquote]:border-primary/30 [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-foreground/60",
    "[&_table]:text-[13px] [&_th]:px-2.5 [&_th]:py-1.5 [&_td]:px-2.5 [&_td]:py-1.5 [&_th]:text-foreground/80 [&_td]:text-foreground/80",
    "[&_li]:text-foreground/90",
    "[&_.katex-display]:overflow-x-auto [&_.katex-display]:py-2.5",
    "[&_.katex]:text-[0.95em]"
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("flex gap-2", isUser ? "justify-end" : "justify-start")}
    >
      {/* Assistant: Bot icon with glow ring during streaming */}
      {!isUser && (
        <div className="relative w-6 h-6 flex-shrink-0 mt-1">
          {message.isStreaming && <div className="icon-glow-ring" />}
          <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center overflow-hidden border border-primary/20">
            <img src="/logo.png" alt="HRAG" className="w-4 h-4 object-contain shadow-sm" />
          </div>
        </div>
      )}

      <div
        className={cn(
          isUser
            ? "max-w-[85%] rounded-xl px-3 py-2 bg-secondary/50"
            : "max-w-[90%] min-w-0 py-1"
        )}
      >
        {/* ThinkingTimeline — single instance, never unmounts between streaming→completed */}
        {/* Typing indicator — only when streaming with no steps and no content yet */}
        {!isUser && message.isStreaming && !message.content && !message.agentSteps?.length && (
          <TypingIndicator status="analyzing" />
        )}

        {isUser ? (
          <p className="text-[15.5px] leading-relaxed whitespace-pre-wrap font-chat">
            {message.content}
          </p>
        ) : message.isStreaming ? (
          message.content ? (
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
          ) : message.thinking ? (
            <InlineThinkingPreview text={message.thinking} />
          ) : null
        ) : (
          <div className={proseClasses}>
            <MarkdownWithCitations
              content={message.content}
              sources={message.sources || []}
              relatedEntities={message.relatedEntities || []}
              imageRefs={message.imageRefs}
            />
          </div>
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

        {/* ThinkingPanel — only when no ThinkingTimeline with thinking log (avoid duplication) */}
        {!isUser && message.thinking && !message.isStreaming &&
          !message.agentSteps?.some((s) => s.thinkingText) && (
            <ThinkingPanel thinking={message.thinking} />
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
          {new Date(message.timestamp).toLocaleTimeString("vi-VN", {
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
            timeZone: "Asia/Ho_Chi_Minh",
          })}
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

// ---------------------------------------------------------------------------
// Inline thinking preview — shown in message body while model is thinking
// ---------------------------------------------------------------------------

function InlineThinkingPreview({ text }: { text: string }) {
  const { t } = useTranslation();
  const containerRef = useRef<HTMLDivElement>(null);
  const isUserScrolledRef = useRef(false);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 20;
    isUserScrolledRef.current = !isAtBottom;
  }, []);

  useEffect(() => {
    if (containerRef.current && !isUserScrolledRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [text]);

  return (
    <div className="mt-1">
      <div className="flex items-center gap-1.5 mb-1.5">
        <Brain className="w-3.5 h-3.5 text-violet-400 animate-pulse" />
        <span className="text-xs font-medium text-violet-400">{t("chat.thinking")}</span>
      </div>
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className={cn(
          "text-xs leading-relaxed text-muted-foreground/70 italic",
          "max-h-[200px] overflow-y-auto scrollbar-none",
          "border-l-2 border-violet-500/30 pl-3",
          "whitespace-pre-wrap break-words",
        )}
      >
        {text}
        <span className="animate-pulse text-violet-400 ml-0.5">|</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------
const STATUS_LABELS: Record<string, string> = {
  analyzing: "rag.status.analyzing",
  retrieving: "rag.status.retrieving",
  generating: "rag.status.generating",
};

function TypingIndicator({ status }: { status?: ChatStreamStatus }) {
  const { t } = useTranslation();
  const labelKey = (status && STATUS_LABELS[status]) || "rag.status.default";
  const label = t(labelKey);
  return (
    <div className="flex gap-2 items-start">
      <div className="py-1">
        <div className="flex items-center gap-1.5">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
          <span className="text-xs text-muted-foreground">{label}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Suggestion chips (empty state)
// ---------------------------------------------------------------------------
function SuggestionChips({ onSelect }: { onSelect: (text: string) => void }) {
  const { t } = useTranslation();
  
  const suggestions = [
    { text: t("chat.suggestion_topics"), icon: <ImageIcon className="w-3.5 h-3.5 text-orange-400" /> },
    { text: t("chat.suggestion_entities"), icon: <Music className="w-3.5 h-3.5 text-pink-400" /> },
    { text: t("chat.suggestion_methodology"), icon: <GraduationCap className="w-3.5 h-3.5 text-blue-400" /> },
    { text: "Viết bất cứ thứ gì", icon: <Pencil className="w-3.5 h-3.5 text-gray-400" /> },
  ];

  return (
    <div className="flex flex-wrap gap-2.5 justify-center max-w-[800px] mt-8 animate-in fade-in slide-in-from-bottom-4 duration-700 delay-300 fill-mode-both px-4">
      {suggestions.map((s) => (
        <button
          key={s.text}
          onClick={() => onSelect(s.text)}
          className="flex items-center gap-2.5 text-[13px] px-5 py-2.5 rounded-full bg-secondary/30 hover:bg-secondary/60 border border-transparent hover:border-secondary transition-all duration-300 text-muted-foreground hover:text-foreground font-medium shadow-sm active:scale-95 whitespace-nowrap"
        >
          {s.icon}
          <span>{s.text}</span>
        </button>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Chat Input Area — Gemini-style floating card
// ---------------------------------------------------------------------------
function ChatInputArea({
  input,
  setInput,
  isStreaming,
  onSend,
  onCancel,
  thinkingSupported,
  enableThinking,
  onToggleThinking,
  forceSearch,
  onToggleSearch,
  inputRef,
  handleKeyDown,
  t
}: {
  input: string;
  setInput: (v: string) => void;
  isStreaming: boolean;
  onSend: () => void;
  onCancel: () => void;
  thinkingSupported: boolean;
  enableThinking: boolean;
  onToggleThinking: () => void;
  forceSearch: boolean;
  onToggleSearch: () => void;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  t: any;
}) {
  return (
    <div className="relative flex flex-col bg-background/80 backdrop-blur-3xl border border-border/60 rounded-[22px] shadow-[0_8px_30px_rgb(0,0,0,0.06)] transition-all duration-300 focus-within:shadow-primary/8 focus-within:border-primary/20 overflow-hidden ring-1 ring-black/5 dark:ring-white/5">
      {/* Input Text Area */}
      <div className="px-4 pt-3.5 pb-1">
        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t("chat.input_placeholder")}
          rows={1}
          className={cn(
            "w-full resize-none bg-transparent px-0 py-1 text-[15.5px] placeholder:text-muted-foreground/45 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50",
            "max-h-[200px] min-h-[38px]",
            "font-chat leading-relaxed tracking-tight text-foreground/90 selection:bg-primary/20"
          )}
          style={{ height: "auto" }}
          onInput={(e) => {
            const target = e.target as HTMLTextAreaElement;
            target.style.height = "auto";
            target.style.height = Math.min(target.scrollHeight, 200) + "px";
          }}
        />
      </div>

      {/* Toolbar Row */}
      <div className="flex items-center justify-between px-2.5 pb-2.5 pt-0.5">
        <div className="flex items-center gap-1">
          {/* Add / Upload */}
          <button 
            className="w-9 h-9 rounded-full flex items-center justify-center text-muted-foreground/60 hover:text-primary hover:bg-primary/5 transition-all"
            title={t("chat.upload")}
          >
            <Plus className="w-5 h-5" />
          </button>
          
          {/* Tools Toggle */}
          <button 
            type="button"
            onClick={onToggleSearch}
            className={cn(
              "flex items-center gap-1.5 px-3 h-9 rounded-full transition-all text-[13px] font-medium",
              forceSearch 
                ? "text-amber-600 bg-amber-500/10 hover:bg-amber-500/15" 
                : "text-muted-foreground/60 hover:text-foreground hover:bg-muted/80"
            )}
          >
            <Settings2 className="w-4 h-4" />
            <span>Công cụ</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          {/* Thinking Toggle — Styled as a pill dropdown */}
          {thinkingSupported && (
            <button
              type="button"
              onClick={() => onToggleThinking()}
              className={cn(
                "flex items-center gap-1.5 px-3 h-9 rounded-full transition-all text-xs font-semibold tracking-tight",
                enableThinking
                  ? "text-violet-500 bg-violet-500/10 hover:bg-violet-500/15"
                  : "text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/80"
              )}
            >
              <span>{t("chat.think_toggle")}</span>
              <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", enableThinking && "rotate-180")} />
            </button>
          )}

          {/* Action Button (Mic / Send / Stop) */}
          <div className="ml-1">
            {isStreaming ? (
              <button
                type="button"
                onClick={() => onCancel()}
                className="w-10 h-10 rounded-full flex items-center justify-center bg-destructive/10 text-destructive hover:bg-destructive/15 transition-all shadow-sm ring-1 ring-destructive/20 cursor-pointer"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
              </button>
            ) : input.trim() ? (
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
                className="w-10 h-10 rounded-full flex items-center justify-center text-muted-foreground/60 hover:text-primary hover:bg-primary/5 transition-all cursor-default"
                title={t("chat.voice")}
              >
                <Mic className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChatPanel — main export
// ---------------------------------------------------------------------------


interface ChatPanelProps {
  sessionId: string | null;
  sessionTitle?: string;
}

export const ChatPanel = memo(function ChatPanel({
  sessionId,
  sessionTitle,
}: ChatPanelProps) {
  const { t } = useTranslation();
  const { user } = useAuthStore();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [enableThinking, setEnableThinking] = useState(true);
  const [thinkingDefaultSynced, setThinkingDefaultSynced] = useState(false);

  const [forceSearch, setForceSearch] = useState(false);

  // Reset session state when switching chats/starting a new chat
  useEffect(() => {
    setMessages([]);
    setInput("");
  }, [sessionId]);

  // Abbreviation modal state
  const [isAbbModalOpen, setIsAbbModalOpen] = useState(false);
  const [selectedAbbShort, setSelectedAbbShort] = useState("");
  const createAbb = useCreateAbbreviation();

  const handleOpenAbbModal = useCallback((short: string) => {
    setSelectedAbbShort(short);
    setIsAbbModalOpen(true);
  }, []);

  const handleSaveAbb = async (data: { short_form: string; full_form: string; description?: string }) => {
    try {
      await createAbb.mutateAsync(data);
      toast.success(t("admin.abbreviations.toast.created"));
      setIsAbbModalOpen(false);
    } catch (err: any) {
      toast.error(err.message || t("admin.abbreviations.toast.error"));
    }
  };

  // Load chat history from PostgreSQL
  const { data: historyData, isLoading: historyLoading } = useChatHistory(sessionId);
  const queryClient = useQueryClient();

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const scrollAnimRef = useRef<number | undefined>(undefined);
  const spacerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Debug mode (Ctrl+Shift+D toggle, persisted in localStorage)
  const [debugMode, setDebugMode] = useState(() =>
    localStorage.getItem("hrag-debug-mode") === "true",
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === "D") {
        e.preventDefault();
        setDebugMode((prev) => {
          const next = !prev;
          localStorage.setItem("hrag-debug-mode", String(next));
          toast.success(next ? t("chat.debug_on") : t("chat.debug_off"));
          return next;
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);



  // Check LLM capabilities (thinking support)
  const { data: capabilities } = useQuery<LLMCapabilities>({
    queryKey: ["llm-capabilities"],
    queryFn: () => api.get<LLMCapabilities>("/rag/capabilities"),
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    retry: 1,
  });
  const thinkingSupported = capabilities?.supports_thinking ?? false;
 
  // Auto-focus input for new chat sessions
  useEffect(() => {
    if (messages.length === 0 && !historyLoading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [messages.length, historyLoading]);

  // Sync thinking toggle default from server (once per mount)
  useEffect(() => {
    if (capabilities && !thinkingDefaultSynced) {
      setEnableThinking(capabilities.thinking_default);
      setThinkingDefaultSynced(true);
    }
  }, [capabilities, thinkingDefaultSynced]);

  // Sync DB history → local messages state when data loads.
  // IMPORTANT: preserve agentSteps from local state — they are client-side only (not stored in DB).
  // Without this, queryClient.invalidateQueries after streaming overwrites agentSteps → ThinkingTimeline disappears.
  useEffect(() => {
    if (historyData?.messages) {
      setMessages((prev) => {
        // Build a map of existing agentSteps by message id so we can re-attach them 
        // if they are not yet in DB, or if local live steps are more detailed.
        const stepsMap = new Map<string, AgentStep[]>();
        for (const m of prev) {
          if (m.agentSteps?.length) stepsMap.set(m.id, m.agentSteps);
        }

        const dbMessages = historyData.messages.map((m) => ({
          id: m.message_id,
          role: m.role as "user" | "assistant",
          content: m.content,
          sources: m.sources ?? undefined,
          relatedEntities: m.related_entities ?? undefined,
          imageRefs: m.image_refs ?? undefined,
          thinking: m.thinking ?? undefined,
          timestamp: m.created_at,
          potential_abbreviations: m.potential_abbreviations ?? undefined,
          // Priority: local live steps (from current session) > DB-persisted synthetic steps
          agentSteps: stepsMap.get(m.message_id) ?? (m.agent_steps?.length 
            ? (m.agent_steps as any[]).map((s, i) => ({
                id: s.id || `hist-${m.message_id}-${i}`,
                step: s.step || 'analyzing',
                status: (s.status) || 'completed',
                detail: s.detail || (STEP_CONFIG[s.step as AgentStepType]?.labelKey ? t(STEP_CONFIG[s.step as AgentStepType].labelKey) : 'Processing'),
                timestamp: s.timestamp || (m.created_at ? new Date(m.created_at).getTime() : Date.now()),
                ...s
              })) as AgentStep[] 
            : undefined),
        }));

        // Merge: keep local messages that are NOT in DB yet (e.g. still streaming or just finished background save)
        const dbIds = new Set(dbMessages.map((m) => m.id));
        // Also check by content/role for user messages to avoid duplication if IDs haven't synced yet
        const dbUserContents = new Set(dbMessages.filter(m => m.role === 'user').map(m => m.content));
        // Add safety-net content deduplication for assistant messages too
        const dbAssistantContents = new Set(dbMessages.filter(m => m.role === 'assistant').map(m => m.content));
        
        const localOnly = prev.filter((m) => {
          if (dbIds.has(m.id)) return false;
          if (m.role === 'user' && dbUserContents.has(m.content)) return false;
          if (m.role === 'assistant' && dbAssistantContents.has(m.content)) return false;
          return true;
        });

        if (localOnly.length === 0) return dbMessages;

        // Combine — usually local-only messages (recent stream) belong at the end
        return [...dbMessages, ...localOnly];
      });
    }
  }, [historyData]);

  // SSE streaming chat
  const stream = useRAGChatStream(sessionId);
  const streamingMsgIdRef = useRef<string | null>(null);
  // Snapshot agentSteps into a ref so finalize always has fresh data
  const agentStepsRef = useRef<AgentStep[]>([]);
  useEffect(() => {
    if (stream.agentSteps.length > 0) {
      agentStepsRef.current = stream.agentSteps;
    }
  }, [stream.agentSteps]);

  // Sync server-assigned message ID to local streaming message
  useEffect(() => {
    if (stream.aiMessageId && streamingMsgIdRef.current) {
      const serverId = stream.aiMessageId;
      const localId = streamingMsgIdRef.current;
      if (serverId !== localId) {
        setMessages((prev) =>
          prev.map((m) => (m.id === localId ? { ...m, id: serverId } : m))
        );
        streamingMsgIdRef.current = serverId;
      }
    }
  }, [stream.aiMessageId]);

  // Sync server-assigned user message ID to local message
  useEffect(() => {
    if (stream.userMessageId) {
      const serverId = stream.userMessageId;
      setMessages((prev) => {
        // Find the most recent user message that doesn't have a server-assigned ID (starts with 'msg_')
        const lastUserIdx = [...prev].reverse().findIndex(m => m.role === 'user' && !m.id.startsWith('msg_'));
        if (lastUserIdx === -1) return prev;
        
        const idx = prev.length - 1 - lastUserIdx;
        if (prev[idx].id === serverId) return prev;
        
        const updated = [...prev];
        updated[idx] = { ...updated[idx], id: serverId };
        return updated;
      });
    }
  }, [stream.userMessageId]);

  // Double-rAF + easeOutCubic scroll to bottom
  const scrollToBottom = useCallback((smooth = true) => {
    const container = scrollContainerRef.current;
    if (!container) return;

    // Cancel in-progress animation
    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = undefined;
    }

    // Double rAF: ensure React commit + browser paint before measuring
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const el = scrollContainerRef.current;
        if (!el) return;
        const target = el.scrollHeight - el.clientHeight;
        if (!smooth || Math.abs(target - el.scrollTop) < 10) {
          el.scrollTop = target;
          return;
        }

        const start = el.scrollTop;
        const distance = target - start;
        const duration = 400;
        const startTime = performance.now();

        const scrollEl = el; // capture for closure
        function animate(now: number) {
          const t = Math.min((now - startTime) / duration, 1);
          const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
          scrollEl.scrollTop = start + distance * ease;
          if (t < 1) {
            scrollAnimRef.current = requestAnimationFrame(animate);
          } else {
            scrollAnimRef.current = undefined;
          }
        }

        scrollAnimRef.current = requestAnimationFrame(animate);
      });
    });
  }, []);

  // Scroll user message to top of chat area
  const scrollUserMsgToTop = useCallback((msgId: string) => {
    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = undefined;
    }
    // Double rAF: wait for React commit + browser paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const container = scrollContainerRef.current;
        if (!container) return;

        // Ensure spacer is set before scrolling (useEffect may not have run yet)
        if (spacerRef.current) {
          spacerRef.current.style.height = `${container.clientHeight}px`;
        }

        const el = container.querySelector(`[data-message-id="${msgId}"]`) as HTMLElement | null;
        if (!el) return;

        // Use getBoundingClientRect for accurate position relative to scroll container
        // (offsetTop is relative to offsetParent, not scroll container)
        const containerRect = container.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        const relativeTop = elRect.top - containerRect.top + container.scrollTop;

        const PADDING_TOP = 12;
        const start = container.scrollTop;
        const target = Math.max(0, relativeTop - PADDING_TOP);
        if (Math.abs(target - start) < 5) return;

        const distance = target - start;
        const duration = 380;
        const startTime = performance.now();

        function animate(now: number) {
          const t = Math.min((now - startTime) / duration, 1);
          const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
          container!.scrollTop = start + distance * ease;
          if (t < 1) {
            scrollAnimRef.current = requestAnimationFrame(animate);
          } else {
            scrollAnimRef.current = undefined;
          }
        }
        scrollAnimRef.current = requestAnimationFrame(animate);
      });
    });
  }, []);

  // Only set spacer height during streaming so user message can scroll to top.
  // Otherwise, remove spacer so we don't scroll into an empty void.
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || !spacerRef.current) return;

    if (stream.isStreaming) {
      spacerRef.current.style.height = `${container.clientHeight}px`;
    } else {
      spacerRef.current.style.height = "0px";
    }
  }, [stream.isStreaming]);

  // Track when streaming finishes to avoid spurious scrollToBottom
  const prevIsStreamingRef = useRef(false);
  const justFinishedStreamingRef = useRef(false);
  useEffect(() => {
    if (prevIsStreamingRef.current && !stream.isStreaming) {
      justFinishedStreamingRef.current = true;
    }
    prevIsStreamingRef.current = stream.isStreaming;
  }, [stream.isStreaming]);

  // Auto-scroll only on non-streaming message changes (history load, etc.)
  // Skip when streaming just ended — viewport already shows end of AI response
  useEffect(() => {
    if (!stream.isStreaming) {
      if (justFinishedStreamingRef.current) {
        justFinishedStreamingRef.current = false;
        return;
      }
      scrollToBottom();
    }
  }, [messages, stream.isStreaming, scrollToBottom]);

  // Sync streaming content + agentSteps → messages state for the streaming message
  useEffect(() => {
    if (!streamingMsgIdRef.current) return;
    const id = streamingMsgIdRef.current;
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === id);
      if (idx === -1) return prev;
      const m = prev[idx];

      // Bail out if nothing actually changed — prevents infinite re-render
      const newContent = stream.streamingContent;
      const newSources = stream.pendingSources.length > 0 ? stream.pendingSources : m.sources;
      const newImages = stream.pendingImages.length > 0 ? stream.pendingImages : m.imageRefs;
      const newThinking = stream.thinkingText || m.thinking;
      const newSteps = stream.agentSteps.length > 0 ? stream.agentSteps : m.agentSteps;
      const newPotentials = stream.potentialAbbreviations.length > 0 ? stream.potentialAbbreviations : m.potential_abbreviations;

      if (
        m.content === newContent &&
        m.sources === newSources &&
        m.imageRefs === newImages &&
        m.thinking === newThinking &&
        m.agentSteps === newSteps &&
        m.potential_abbreviations === newPotentials
      ) {
        return prev; // no change → skip setMessages re-render
      }

      const updated = [...prev];
      updated[idx] = {
        ...m,
        content: newContent,
        sources: newSources,
        imageRefs: newImages,
        thinking: newThinking,
        agentSteps: newSteps,
        potential_abbreviations: newPotentials,
      };
      return updated;
    });
  }, [stream.streamingContent, stream.pendingSources, stream.pendingImages, stream.thinkingText, stream.isStreaming, stream.agentSteps]);

  const handleSend = useCallback(
    async (text?: string) => {
      const msg = (text || input).trim();
      if (!msg || stream.isStreaming) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: msg,
        timestamp: new Date().toISOString(),
      };

      // Add placeholder assistant message for streaming
      const assistantId = crypto.randomUUID();
      streamingMsgIdRef.current = assistantId;
      const placeholderMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, placeholderMsg]);
      setInput("");
      // Scroll new user message to top so agent response fills the space below
      scrollUserMsgToTop(userMsg.id);

      // Build history from previous messages (exclude the new user + placeholder)
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const finalMsg = await stream.sendMessage(
        msg,
        history,
        thinkingSupported && enableThinking,
        forceSearch,
      );

      // Finalize the streaming message (prefer finalMsg.agentSteps — directly from SSE loop,
      // fallback to ref snapshot, then to what was synced into the message during streaming)
      if (finalMsg) {
        // Invalidate sessions list query to fetch generated chat title from backend
        queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
        // Invalidate the specific chat history cache to prevent stale messages on remount
        queryClient.invalidateQueries({ queryKey: ["chat-history", sessionId] });

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                ...finalMsg,
                id: finalMsg.id, // Use the official server-assigned ID from finalMsg
                isStreaming: false,
                agentSteps: finalMsg.agentSteps?.length
                  ? finalMsg.agentSteps
                  : agentStepsRef.current.length > 0
                    ? agentStepsRef.current
                    : m.agentSteps,
              }
              : m,
          ),
        );
      } else if (stream.error) {
        toast.error(t("chat.failed", { error: stream.error }));
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                ...m,
                content: m.content || t("chat.error_fallback"),
                isStreaming: false,
              }
              : m,
          ),
        );
      } else {
        // Cancelled — keep partial content
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        );
      }
      streamingMsgIdRef.current = null;
    },
    [input, messages, stream, thinkingSupported, enableThinking, forceSearch, scrollUserMsgToTop],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Collect all sources from all assistant messages for citation fallback.
  // When the model doesn't call search_documents but references citation IDs
  // from earlier answers, this allows those citations to still render as links.
  // NOTE: Must be declared before any early returns to satisfy Rules of Hooks.
  const allSourcesFlat = useMemo(() => {
    const seen = new Set<string>();
    const merged: ChatSourceChunk[] = [];
    for (const m of messages) {
      if (m.role === "assistant" && m.sources) {
        for (const s of m.sources) {
          const key = String(s.index);
          if (!seen.has(key)) {
            seen.add(key);
            merged.push(s);
          }
        }
      }
    }
    return merged;
  }, [messages]);

  if (historyLoading) {
    return (
      <div className="h-full flex items-center justify-center border-r">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <SessionIdCtx.Provider value={sessionId}>
      <DebugCtx.Provider value={debugMode}>
        <AllSourcesCtx.Provider value={allSourcesFlat}>
          <div className="flex flex-col h-full bg-background border-r relative z-0 overflow-hidden">
            {/* Header */}
            {/* Header */}
            <div className="flex-shrink-0 flex items-center justify-between px-4 py-3 bg-background/50 backdrop-blur-md">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl flex items-center justify-center bg-primary/10 border border-primary/15 overflow-hidden shadow-sm">
                  <img src="/logo.png" alt="HRAG" className="w-5 h-5 object-contain" />
                </div>
                <h2 className="text-[14px] font-bold tracking-tight text-foreground line-clamp-1">
                  {sessionTitle || (sessionId ? `${t("chat.session", { id: sessionId })}` : t("chat.select_session"))}
                </h2>
              </div>
            </div>

            {/* Main Content Area */}
            {messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center px-4 overflow-y-auto pb-[10vh] scrollbar-none">
                <div className="w-full max-w-[720px] flex flex-col items-center translate-y-[-4vh]">
                  {/* Greeting */}
                  <div className="mb-10 text-center animate-in fade-in zoom-in-95 duration-1000 ease-out">
                  <div className="inline-flex items-center gap-2 mb-4 px-3 py-1 rounded-full bg-primary/5 border border-primary/10 text-primary">
                    <Sparkles className="w-3.5 h-3.5" />
                    <span className="text-[11px] font-bold uppercase tracking-wider">Assistant</span>
                  </div>
                  <h1 className="text-3xl md:text-5xl font-bold tracking-tight text-foreground mb-4">
                    Xin chào {user?.full_name || "XayDung"}!
                  </h1>
                  <p className="text-lg md:text-2xl text-muted-foreground/60 font-medium">
                    {t("chat.assistant_desc")}
                  </p>
                </div>

                {/* Input Area (Centered) */}
                <div className="w-full max-w-[720px] px-2 mb-6">
                  <ChatInputArea 
                    input={input}
                    setInput={setInput}
                    isStreaming={stream.isStreaming}
                    onSend={handleSend}
                    onCancel={stream.cancel}
                    thinkingSupported={thinkingSupported}
                    enableThinking={enableThinking}
                    onToggleThinking={() => setEnableThinking(!enableThinking)}
                    forceSearch={forceSearch}
                    onToggleSearch={() => setForceSearch(!forceSearch)}
                    inputRef={inputRef}
                    handleKeyDown={handleKeyDown}
                    t={t}
                  />
                </div>

                {/* Suggestions Pills (Below) */}
                <SuggestionChips onSelect={handleSend} />
                </div>
              </div>
            ) : (
              <>
                {/* Messages List */}
                <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto px-3 py-3 space-y-4 relative scrollbar-none">
                  <AnimatePresence mode="popLayout">
                    {messages.map((msg) => (
                      <motion.div 
                        key={msg.id} 
                        data-message-id={msg.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, transition: { duration: 0.1 } }}
                      >
                        <MessageBubble
                          message={msg}
                          onAddAbbreviation={handleOpenAbbModal}
                        />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  <div ref={spacerRef} aria-hidden />
                </div>

                {/* Sticky Input Area (Fixed at bottom) */}
                <div className="flex-shrink-0 p-4 border-t/0 pb-8 last-msg-focus-fix bg-gradient-to-t from-background via-background/80 to-transparent">
                  <div className="w-full max-w-[720px] mx-auto px-2">
                    <ChatInputArea 
                      input={input}
                      setInput={setInput}
                      isStreaming={stream.isStreaming}
                      onSend={handleSend}
                      onCancel={stream.cancel}
                      thinkingSupported={thinkingSupported}
                      enableThinking={enableThinking}
                      onToggleThinking={() => setEnableThinking(!enableThinking)}
                      forceSearch={forceSearch}
                      onToggleSearch={() => setForceSearch(!forceSearch)}
                      inputRef={inputRef}
                      handleKeyDown={handleKeyDown}
                      t={t}
                    />
                    <p className="text-[10px] text-muted-foreground/40 mt-3 text-center font-medium">
                      {t("chat.input_hint")}
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>
        </AllSourcesCtx.Provider>
      </DebugCtx.Provider>

      <AbbreviationModal
        open={isAbbModalOpen}
        onOpenChange={setIsAbbModalOpen}
        abbreviation={null}
        initialShortForm={selectedAbbShort}
        onSave={handleSaveAbb}
        isPending={createAbb.isPending}
      />
    </SessionIdCtx.Provider>
  );
});

