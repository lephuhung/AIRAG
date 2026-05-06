import { useState, useRef, useEffect, useCallback, useMemo, memo, createContext, useContext, Children, isValidElement, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
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
  FileSearch,
  CornerDownLeft,
  LayoutGrid,
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
import { cn, generateId } from "@/lib/utils";
import { api, rewritePresignedUrl } from "@/lib/api";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useAuthStore } from "@/stores/authStore";
import { useThemeStore } from "@/stores/useThemeStore";

const truncateName = (name: string, maxLength = 25) => {
  if (name.length <= maxLength) return name;
  return name.slice(0, maxLength - 8) + "..." + name.slice(-5);
};

const formatMentionName = (name: string) => {
  let clean = name.replace(/\.[^/.]+$/, "");
  return truncateName(clean, 30);
};

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
import { useDocument, useDocuments } from "@/hooks/useDocuments";
import { useChatHistory, useSessionDocuments } from "@/hooks/useChatHistory";
import { useRAGChatStream } from "@/hooks/useRAGChatStream";
import { useTranslation } from "@/hooks/useTranslation";
import { useCreateChatSession, useUpdateSessionTitle } from "@/hooks/useChatSessions";
import { useCreateAbbreviation } from "@/hooks/useAbbreviations";
import { AbbreviationModal } from "@/components/rag/AbbreviationModal";
import { StreamingMarkdown } from "@/components/rag/MemoizedMarkdown";
import { STEP_CONFIG, ThinkingTimeline } from "@/components/rag/ThinkingTimeline";
import { STATUS_CONFIG, getFileConfig } from "@/components/rag/document-utils";
import type {
  ChatMessage,
  ChatImageRef,
  ChatSourceChunk,
  ChatStreamStatus,
  LLMCapabilities,
  AgentStep,
  AgentStepType,
  Document,
  DocumentStatus,
  PeopleRecord,
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
    let processedLine = line;
    // Fix: Headers lacking a space (e.g. "##Title" -> "## Title")
    if (/^#{1,6}[^#\s]/.test(processedLine)) {
      processedLine = processedLine.replace(/^(#{1,6})([^#\s])/, "$1 $2");
    }

    if (processedLine.trim().startsWith("```")) {
      inCodeFence = !inCodeFence;
    }

    const isTable = (processedLine.trim().startsWith("|") && processedLine.trim().endsWith("|")) ||
      /^\|[\s:|-]+\|$/.test(processedLine.trim());

    // Insert blank line before a table if needed
    if (isTable && !prevWasTable && result.length > 0 && result[result.length - 1].trim() !== "") {
      result.push("");
    }

    // Insert blank line after a table if current line is not a table
    if (prevWasTable && !isTable && processedLine.trim() !== "") {
      result.push("");
    }

    // Convert single-line display math $$content$$ to multi-line format
    if (
      !inCodeFence &&
      processedLine.trim().startsWith("$$") &&
      processedLine.trim().endsWith("$$") &&
      processedLine.trim().length > 4 &&
      processedLine.trim() !== "$$"
    ) {
      const mathContent = processedLine.trim().slice(2, -2);
      result.push("$$");
      result.push(mathContent);
      result.push("$$");
    } else {
      result.push(processedLine);
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
        aria-label={t("chat.copy_code")}
        className={cn(
          "absolute top-2 left-2 p-1 rounded-md text-muted-foreground/50 hover:text-muted-foreground transition-all opacity-0 group-hover:opacity-100 z-10",
          isDark ? "bg-white/5 hover:bg-white/10" : "bg-black/5 hover:bg-black/10"
        )}
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
// People Card — structured display for MongoDB people search results
// ---------------------------------------------------------------------------

/** Field display config: maps schema-agnostic keys to display labels and icon */
const FIELD_CONFIG: Record<string, { label: string; icon: ReactNode }> = {
  hoTen: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  HO_TEN: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  TenHoiVien: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  ho_ten: { label: "Họ tên", icon: <User className="w-3 h-3" /> },
  maSoBhxh: { label: "Mã BHXH", icon: <DatabaseZap className="w-3 h-3" /> },
  soTheBhyt: { label: "Số thẻ BHYT", icon: <FileText className="w-3 h-3" /> },
  ngaySinhHienThi: { label: "Ngày sinh", icon: <BookOpen className="w-3 h-3" /> },
  NGAY_SINH: { label: "Ngày sinh", icon: <BookOpen className="w-3 h-3" /> },
  NgaySinh: { label: "Ngày sinh", icon: <BookOpen className="w-3 h-3" /> },
  soCmnd: { label: "Số CMND", icon: <FileSearch className="w-3 h-3" /> },
  cmnd: { label: "Số CMND", icon: <FileSearch className="w-3 h-3" /> },
  SoDinhDanh: { label: "Số định danh", icon: <FileSearch className="w-3 h-3" /> },
  MA_DOI_TUONG: { label: "Mã định danh", icon: <FileSearch className="w-3 h-3" /> },
  dienThoai: { label: "Điện thoại", icon: <Mic className="w-3 h-3" /> },
  SoDienThoai: { label: "Điện thoại", icon: <Mic className="w-3 h-3" /> },
  DIEN_THOAI_ME: { label: "Điện thoại mẹ", icon: <Mic className="w-3 h-3" /> },
  diaChi: { label: "Địa chỉ", icon: <LayoutGrid className="w-3 h-3" /> },
  DiaChi: { label: "Địa chỉ", icon: <LayoutGrid className="w-3 h-3" /> },
  coSoKCB: { label: "CS KCB", icon: <Settings2 className="w-3 h-3" /> },
  trangThaiThe: { label: "Trạng thái", icon: <Zap className="w-3 h-3" /> },
  tyLeBhyt: { label: "Tỷ lệ BHYT", icon: <Sparkles className="w-3 h-3" /> },
  tuNgay: { label: "Từ ngày", icon: <BookOpen className="w-3 h-3" /> },
  denNgay: { label: "Đến ngày", icon: <BookOpen className="w-3 h-3" /> },
  TenHangHoiVien: { label: "Hạng hội viên", icon: <GraduationCap className="w-3 h-3" /> },
  SoTheHoiVien: { label: "Số thẻ hội viên", icon: <FileText className="w-3 h-3" /> },
  TEN_ME: { label: "Tên mẹ", icon: <User className="w-3 h-3" /> },
  GIOI_TINH: { label: "Giới tính", icon: <User className="w-3 h-3" /> },
  PID: { label: "PID", icon: <FileSearch className="w-3 h-3" /> },
};

/** Fields to exclude from display */
const SKIP_FIELDS = new Set(["_id", "_source_schema", "lookup_type", "found", "persons", "display"]);

/** Get name field from a people record (tries multiple possible field names) */
function getNameField(record: Record<string, unknown>): string {
  for (const key of ["hoTen", "HO_TEN", "TenHoiVien", "ho_ten"]) {
    if (record[key] && typeof record[key] === "string") {
      return record[key] as string;
    }
  }
  return "(Không có tên)";
}

/** Schema display name mapping */
const SCHEMA_LABELS: Record<string, string> = {
  bhxh: "BHXH",
  lg: "LG Hội viên",
  vacxin: "Tiêm chủng",
  evn: "Điện lực",
};

function PeopleCard({ people, isLoadingMore }: { people: PeopleRecord[], isLoadingMore?: boolean }) {
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  if (!people || people.length === 0) return null;

  const handleCopyCard = (person: PeopleRecord, idx: number) => {
    const name = getNameField(person);
    const schema = SCHEMA_LABELS[person._source_schema || ""] || person._source_schema || "Unknown";
    const lines = [`Thông tin: ${name}`, `Nguồn dữ liệu: ${schema}`, ""];

    const displayFields = Object.entries(person)
      .filter(([k, v]) => !SKIP_FIELDS.has(k) && v !== undefined && v !== null && v !== "")
      .sort(([a], [b]) => {
        const order = Object.keys(FIELD_CONFIG);
        return order.indexOf(a) - order.indexOf(b);
      });

    for (const [key, val] of displayFields) {
      const config = FIELD_CONFIG[key];
      const label = config?.label || key;
      lines.push(`- ${label}: ${val}`);
    }

    navigator.clipboard.writeText(lines.join("\n")).then(() => {
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx(null), 2000);
    });
  };

  return (
    <div className="my-3 space-y-2">
      {people.map((person, idx) => {
        const name = getNameField(person);
        const schema = SCHEMA_LABELS[person._source_schema || ""] || person._source_schema;
        const displayFields = Object.entries(person)
          .filter(([k, v]) => !SKIP_FIELDS.has(k) && v !== undefined && v !== null && v !== "")
          .sort(([a], [b]) => {
            const order = Object.keys(FIELD_CONFIG);
            return order.indexOf(a) - order.indexOf(b);
          });

        return (
          <div
            key={idx}
            className="relative rounded-xl border border-border/40 bg-zinc-50/50 dark:bg-zinc-900/40 p-4 hover:shadow-md transition-all duration-300"
          >
            {/* Copy button — top right corner */}
            <button
              onClick={() => handleCopyCard(person, idx)}
              className={cn(
                "absolute top-3 right-3 p-2 rounded-lg text-xs transition-all",
                copiedIdx === idx
                  ? "bg-emerald-500/10 text-emerald-600"
                  : "bg-muted/30 hover:bg-muted text-muted-foreground hover:text-foreground"
              )}
              title="Copy"
            >
              {copiedIdx === idx ? (
                <ClipboardCheck className="w-4 h-4" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>

            {/* Header: name + schema badge */}
            <div className="flex items-center gap-3 mb-4 pr-10">
              <div className="w-9 h-9 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary text-sm font-bold shadow-sm">
                {name.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-bold text-[15px] text-foreground tracking-tight truncate">{name}</p>
                {schema && (
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary/60" />
                    <p className="text-[10px] text-muted-foreground font-bold uppercase tracking-wider">{schema}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Fields grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-2.5">
              {displayFields.map(([key, val]) => {
                const config = FIELD_CONFIG[key];
                if (!config) return null;
                return (
                  <div key={key} className="flex items-center gap-2.5">
                    <div className="flex-shrink-0 w-5 h-5 rounded-md bg-muted/40 flex items-center justify-center text-primary/70 scale-90">
                      {config.icon}
                    </div>
                    <div className="min-w-0">
                      <p className="text-[10px] text-muted-foreground font-semibold uppercase tracking-tight leading-none mb-0.5">{config.label}</p>
                      <p className="text-[12px] font-medium text-foreground/90 truncate">{String(val)}</p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
      
      {isLoadingMore && (
        <div className="flex items-center justify-center gap-2 p-3 text-muted-foreground bg-muted/10 rounded-lg border border-dashed">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span className="text-xs font-medium">Đang tìm kiếm thêm cơ sở dữ liệu...</span>
        </div>
      )}
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
            className="overflow-hidden border-t"
          >
            <div className="p-2 flex flex-wrap gap-2">
              {images.map((img, i) => (
                <ImageRefCard key={i} img={img} />
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
const PremiumThinking = memo(({
  thinking,
  agentSteps,
  isStreaming,
  hasContent
}: {
  thinking: string;
  agentSteps?: AgentStep[];
  isStreaming?: boolean;
  hasContent?: boolean
}) => {
  const { t } = useTranslation();
  
  // Show if there's thinking text, agent steps, or we are currently streaming the beginning of a message
  const hasReasoning = (agentSteps && agentSteps.length > 0) || !!thinking || (isStreaming && !hasContent);
  
  // Find active/done steps
  const activeStep = useMemo(() => agentSteps?.find((s) => s.status === "active"), [agentSteps]);
  const doneStep = useMemo(() => agentSteps?.find((s) => s.step === "done" && s.status === "completed"), [agentSteps]);
  
  // Dynamic Label Logic
  const labelText = useMemo(() => {
    if (!!thinking && isStreaming && !hasContent) {
      return t("chat.thinking_process") || "Quá trình suy luận";
    }
    if (hasContent && isStreaming) {
      return t("rag.status.generating") || "Đang tạo câu trả lời...";
    }
    if (isStreaming && !hasContent) {
      if (activeStep) {
        const config = STEP_CONFIG[activeStep.step];
        return activeStep.detail || (config ? t(config.labelKey) : "Đang xử lý...");
      }
      if (doneStep && !thinking) {
        return t("rag.timeline.done") || "Đã xong bước chuẩn bị";
      }
      return t("chat.analyzing_question") || "Đang phân tích câu hỏi...";
    }
    return t("chat.thinking_process") || "Quá trình suy luận";
  }, [activeStep, doneStep, thinking, isStreaming, hasContent, t]);

  const [expanded, setExpanded] = useState(false);
  const [showFull, setShowFull] = useState(false);
  const userToggledRef = useRef(false);

  const handleToggle = useCallback(() => {
    userToggledRef.current = true;
    setExpanded(prev => !prev);
  }, []);

  // Simplified Truncation - much faster than split/join for long strings
  const { displayThinking, isTruncated } = useMemo(() => {
    if (showFull || thinking.length < 1000) {
      return { displayThinking: thinking, isTruncated: false };
    }
    
    // Quick approximation of 200 words using character count (~1000 chars)
    // or finding the 200th space
    let count = 0;
    let index = -1;
    for (let i = 0; i < thinking.length; i++) {
      if (thinking[i] === ' ') {
        count++;
        if (count >= 200) {
          index = i;
          break;
        }
      }
    }
    
    if (index === -1) return { displayThinking: thinking, isTruncated: false };
    return {
      displayThinking: thinking.slice(0, index) + "...",
      isTruncated: true
    };
  }, [thinking, showFull]);

  // Auto-expand/collapse logic
  useEffect(() => {
    if (userToggledRef.current) return;
    
    // Auto-expand only when thinking starts and no final content yet
    if (isStreaming && !!thinking && thinking.length > 5 && !hasContent && !expanded) {
      setExpanded(true);
    }
    
    // Auto-collapse when content starts appearing or when streaming finishes to focus on the answer
    if ((hasContent || !isStreaming) && expanded) {
      setExpanded(false);
    }
  }, [thinking, isStreaming, hasContent, expanded]);

  if (!hasReasoning) return null;

  return (
    <div className="mb-3 group">
      <button
        onClick={handleToggle}
        className={cn(
          "flex items-center gap-2 px-2 py-1.5 rounded-md text-[13.5px] font-bold transition-all duration-200",
          "bg-transparent border-none outline-none select-none",
          "text-slate-700 hover:text-violet-700 hover:bg-violet-600/15",
          expanded && "text-violet-700 bg-violet-600/20"
        )}
      >
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.div
            key={labelText}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -3 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="flex items-center gap-2"
          >
            {isStreaming ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-violet-500" />
            ) : (
              <Sparkles className="w-3.5 h-3.5 text-violet-400" />
            )}
            <span>{labelText}</span>
            <ChevronDown className={cn("w-3.5 h-3.5 transition-transform duration-300", expanded && "rotate-180")} />
          </motion.div>
        </AnimatePresence>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="mt-2 pl-3 border-l-2 border-violet-100/50 space-y-3">
              {!!thinking && (
                <div className="text-[13.5px] text-slate-600 leading-relaxed font-normal">
                  <div className="prose prose-sm max-w-none prose-slate prose-p:leading-relaxed prose-pre:bg-slate-50">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {displayThinking}
                    </ReactMarkdown>
                    {isStreaming && !hasContent && (
                      <div className="flex items-center gap-1.5 mt-2 text-violet-500/60">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        <span className="text-[11px] italic font-medium">Đang suy nghĩ...</span>
                      </div>
                    )}
                  </div>
                  
                  {isTruncated && (
                    <button
                      onClick={() => setShowFull(true)}
                      className="mt-2 text-violet-600 hover:text-violet-700 font-medium text-[12px] underline-offset-4 hover:underline"
                    >
                      {t("common.show_more")}
                    </button>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});


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
// File attachment badge for inline display in message bubbles
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Document mention dropdown for @ autocomplete
// ---------------------------------------------------------------------------
function DocumentMentionDropdown({
  docs,
  onSelect,
  onClose,
  selectedIndex = 0,
  anchorRef,
}: {
  docs: { id: string; filename: string; original_filename?: string; file_type?: string; document_number?: string | null }[];
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

          {/* List - Ultra compact */}
          <div className="max-h-[280px] overflow-y-auto py-0.5 px-0.5 scrollbar-none flex flex-col gap-px">
            {docs.map((doc, idx) => {
              const fileConfig = getFileConfig(doc.file_type || doc.filename?.split(".").pop() || "");
              const FileIcon = fileConfig.icon;
              const isHighlighted = idx === selectedIndex;

              return (
                <button
                  key={doc.id}
                  type="button"
                  onClick={() => onSelect(doc)}
                  className={cn(
                    "w-full flex items-center gap-2 px-2 py-1 rounded-md transition-all duration-150 text-left relative overflow-hidden group",
                    isHighlighted
                      ? "bg-primary/[0.04] dark:bg-primary/[0.1]"
                      : "hover:bg-zinc-50/50 dark:hover:bg-zinc-800/30"
                  )}
                >
                  <div className={cn(
                    "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-all duration-150",
                    isHighlighted
                      ? "bg-white dark:bg-zinc-900 shadow-[0_1px_3px_rgba(0,0,0,0.1)] border border-zinc-200 dark:border-zinc-800"
                      : "bg-zinc-100/50 dark:bg-zinc-800/40"
                  )}>
                    <FileIcon className={cn(
                      "w-4 h-4",
                      fileConfig.color
                    )} />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2 overflow-hidden">
                      <span className={cn(
                        "text-[12px] font-medium truncate tracking-tight transition-colors duration-150",
                        isHighlighted ? "text-foreground" : "text-muted-foreground/80 group-hover:text-foreground"
                      )}>
                        {doc.original_filename || doc.filename}
                      </span>
                      {doc.document_number && (
                        <span className={cn(
                          "text-[9px] font-bold font-mono tracking-tight shrink-0 px-1.5 py-0.5 rounded-md border",
                          isHighlighted ? "bg-primary/5 border-primary/20 text-primary/60" : "bg-zinc-100 dark:bg-zinc-800/50 border-zinc-200 dark:border-zinc-800 text-zinc-400"
                        )}>
                          {doc.document_number}
                        </span>
                      )}
                    </div>
                  </div>

                  {isHighlighted && (
                    <div className="shrink-0 ml-0.5 opacity-30">
                      <CornerDownLeft className="w-2.5 h-2.5" />
                    </div>
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

// ---------------------------------------------------------------------------
// Referenced document badge (shown in input area)
// ---------------------------------------------------------------------------
function ReferencedDocBadge({ doc, onRemove }: { doc: { id: string; filename: string; original_filename?: string }; onRemove: () => void }) {
  const displayName = doc.original_filename || doc.filename;
  const fileConfig = getFileConfig(doc.filename.split(".").pop() || "");
  const FileIcon = fileConfig.icon;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="inline-flex items-center gap-2 p-1 pl-1.5 pr-2.5 bg-white dark:bg-zinc-900/90 border border-zinc-200 dark:border-zinc-800/80 shadow-sm rounded-xl group select-none transition-all hover:bg-zinc-50 hover:border-zinc-300 dark:hover:bg-zinc-800 dark:hover:border-zinc-700"
    >
      <div className={cn("w-7 h-7 rounded-lg flex items-center justify-center shrink-0", fileConfig.bgColor)}>
        <FileIcon className={cn("w-3.5 h-3.5", fileConfig.color)} />
      </div>
      <span className="text-[12px] font-bold text-foreground tracking-tight truncate max-w-[150px]">
        {displayName}
      </span>
      <button
        type="button"
        onClick={onRemove}
        className="w-5 h-5 rounded-full flex items-center justify-center hover:bg-zinc-100 dark:hover:bg-zinc-700/50 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-all ml-1"
      >
        <X className="w-3 h-3" />
      </button>
    </motion.div>
  );
}

function ToolsDropdown({
  onPlus,
  onMic,
  forceSearch,
  onToggleSearch
}: {
  onPlus?: () => void;
  onMic?: () => void;
  forceSearch: boolean;
  onToggleSearch: () => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const [coords, setCoords] = useState({ bottom: 0, left: 0, width: 0 });
  const anchorRef = useRef<HTMLButtonElement>(null);
  const { t } = useTranslation();

  useEffect(() => {
    if (isOpen && anchorRef.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      
      setCoords({
        bottom: window.innerHeight - rect.top + 8,
        left: rect.left,
        width: rect.width
      });
    }
  }, [isOpen]);

  const menuItems = [
    { id: 'word', icon: FileText, label: t("chat.tool_word"), onClick: onPlus },
    { id: 'audio', icon: Mic, label: t("chat.tool_audio"), onClick: onMic },
  ];

  return (
    <div className="relative">
      <button
        ref={anchorRef}
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          "flex items-center gap-1.5 px-3 h-9 rounded-full transition-all text-[13px] font-bold tracking-tight",
          isOpen
            ? "text-primary bg-primary/10 shadow-sm"
            : "text-zinc-500 hover:text-zinc-700 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:text-zinc-200 dark:hover:bg-zinc-900"
        )}
      >
        <LayoutGrid className={cn("w-4 h-4", isOpen && "animate-pulse")} />
        <span>{t("chat.tools_button") || "Công cụ"}</span>
      </button>

      {isOpen && createPortal(
        <>
          <div
            className="fixed inset-0 z-[9998]"
            onClick={() => setIsOpen(false)}
          />
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            style={{
              position: 'fixed',
              bottom: coords.bottom,
              left: coords.left,
              zIndex: 9999
            }}
            className="w-64 bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-2xl shadow-xl overflow-hidden p-1.5"
          >
            <div className="flex flex-col gap-0.5">
              {menuItems.map((item, idx) => {
                const Icon = item.icon;
                const isFirst = idx === 0;
                return (
                  <div key={item.id}>
                    <button
                      type="button"
                      onClick={() => {
                        item.onClick?.();
                        setIsOpen(false);
                      }}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all text-left group hover:bg-zinc-100 dark:hover:bg-zinc-900 text-foreground/80 hover:text-foreground"
                    >
                      <Icon className="w-4 h-4 shrink-0 text-foreground/60 group-hover:text-foreground transition-colors" />
                      <span className="text-[13.5px] font-medium tracking-tight flex-1">{item.label}</span>
                    </button>
                    {isFirst && (
                      <div className="h-[1px] bg-zinc-100 dark:bg-zinc-800 my-1 mx-1" />
                    )}
                  </div>
                );
              })}
            </div>
          </motion.div>
        </>,
        document.body
      )}
    </div>
  );
}

function SearchTooltipWrapper({
  forceSearch,
  onToggleSearch,
  t
}: {
  forceSearch: boolean;
  onToggleSearch: () => void;
  t: any
}) {
  const [isHovered, setIsHovered] = useState(false);
  const [coords, setCoords] = useState({ bottom: 0, left: 0 });
  const anchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isHovered && anchorRef.current) {
      const rect = anchorRef.current.getBoundingClientRect();
      const container = anchorRef.current.closest('[data-chat-input-container]');
      const containerRect = container?.getBoundingClientRect();
      const menuWidth = 200; // estimated tooltip width
      
      let left = rect.left + rect.width / 2;
      if (containerRect && containerRect.left > menuWidth + 20) {
        // If we want to move it to the margin too? 
        // No, tooltips are better centered over the button.
        // But for consistency with the user's request to not cover content, 
        // let's at least make sure it doesn't cover too much.
      }

      setCoords({
        bottom: window.innerHeight - rect.top + 8,
        left: left
      });
    }
  }, [isHovered]);

  return (
    <div
      ref={anchorRef}
      className="relative"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <button
        type="button"
        onClick={onToggleSearch}
        className={cn(
          "w-9 h-9 flex items-center justify-center rounded-full transition-all shadow-sm border",
          forceSearch
            ? "text-amber-600 bg-amber-500/10 border-amber-500/20 hover:bg-amber-500/15"
            : "text-zinc-400 bg-white dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 hover:bg-zinc-50 dark:hover:bg-zinc-800"
        )}
      >
        <Zap className={cn("w-4 h-4", forceSearch && "fill-amber-500")} />
      </button>

      {isHovered && createPortal(
        <motion.div
          initial={{ opacity: 0, y: 5, scale: 0.95, x: '-50%' }}
          animate={{ opacity: 1, y: 0, scale: 1, x: '-50%' }}
          style={{
            position: 'fixed',
            bottom: coords.bottom,
            left: coords.left,
            zIndex: 10000
          }}
          className="px-3 py-1.5 bg-zinc-900 text-white text-[11px] font-medium rounded-lg whitespace-nowrap shadow-xl pointer-events-none"
        >
          {forceSearch ? t("chat.search_on") : t("chat.search_off")}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-x-4 border-x-transparent border-t-4 border-t-zinc-900" />
        </motion.div>,
        document.body
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single message bubble
// ---------------------------------------------------------------------------
const MessageBubble = memo(function MessageBubble({
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
            let elements: React.ReactNode[] = [message.content];

            if (allAvailableDocs.length > 0) {
              for (const doc of allAvailableDocs) {
                const truncatedDisplayName = formatMentionName(doc.original_filename || doc.filename);
                const docIdTag = `<document_id=${doc.id}>`;
                const escapeRegExp = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                
                const regexStr = `(?:${escapeRegExp(docIdTag)}|@${escapeRegExp(truncatedDisplayName)})`;
                const mentionRegex = new RegExp(`(${regexStr})`, 'g');
                
                let foundMatch = false;
                
                const newElements: React.ReactNode[] = [];
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
// Format Upload Progress — animated steps during docx upload/parse
// ---------------------------------------------------------------------------
import type { FormatUploadStep } from "@/types";

const FORMAT_STEPS: { step: FormatUploadStep; icon: string; labelKey: string }[] = [
  { step: "uploading", icon: "📤", labelKey: "chat.format_step_uploading" },
  { step: "extracting", icon: "📄", labelKey: "chat.format_step_extracting" },
  { step: "analyzing", icon: "🔍", labelKey: "chat.format_step_analyzing" },
  { step: "complete", icon: "✅", labelKey: "chat.format_step_complete" },
];

function FormatUploadProgress({ step, message }: { step: FormatUploadStep; message?: string }) {
  const { t } = useTranslation();
  const currentIndex = FORMAT_STEPS.findIndex((s) => s.step === step);

  return (
    <div className="flex flex-col gap-2 py-1">
      <div className="flex items-center gap-1.5">
        <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
        <span className="text-xs text-muted-foreground">{message || t("chat.format_check_loading")}</span>
      </div>
      <div className="flex items-center gap-1.5 pl-1">
        {FORMAT_STEPS.slice(0, -1).map((s, i) => {
          const isDone = i < currentIndex;
          const isActive = i === currentIndex;
          return (
            <div key={s.step} className="flex items-center gap-1">
              <div
                className={cn(
                  "w-5 h-5 rounded-full flex items-center justify-center text-[10px] transition-all duration-300",
                  isDone && "bg-emerald-500/20 text-emerald-500",
                  isActive && "bg-primary/20 text-primary ring-2 ring-primary/30 animate-pulse",
                  !isDone && !isActive && "bg-muted/40 text-muted-foreground/40"
                )}
              >
                {isDone ? "✓" : s.icon}
              </div>
              <span
                className={cn(
                  "text-[10px] transition-all duration-300",
                  isDone && "text-emerald-600 dark:text-emerald-400",
                  isActive && "text-primary font-medium",
                  !isDone && !isActive && "text-muted-foreground/40"
                )}
              >
                {t(s.labelKey)}
              </span>
              {i < FORMAT_STEPS.length - 2 && (
                <div
                  className={cn(
                    "w-4 h-px mx-0.5 transition-all duration-300",
                    i < currentIndex ? "bg-emerald-500/40" : "bg-muted/30"
                  )}
                />
              )}
            </div>
          );
        })}
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
    { text: t("chat.suggestion_any"), icon: <Pencil className="w-3.5 h-3.5 text-gray-400" /> },
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
interface AttachedFile {
  id: string; // Document database ID
  file: File;
  status: "uploading" | "parsing" | "ready" | "indexed" | "failed";
  progress: number;
  docMetadata?: Document;
}

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
  attachedFiles,
  onRemoveAttachment,
  inputRef,
  handleKeyDown,
  onPlus,
  onMic,
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
  thinkingSupported: boolean;
  enableThinking: boolean;
  onToggleThinking: () => void;
  forceSearch: boolean;
  onToggleSearch: () => void;
  attachedFiles: AttachedFile[];
  onRemoveAttachment: (id: string) => void;
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  onPlus?: () => void;
  onMic?: () => void;
  t: any;
  referencedDocs?: { id: string; filename: string; original_filename?: string }[];
  onRemoveReferencedDoc?: (docId: string) => void;
  showMentionDropdown?: boolean;
  filteredMentionDocs?: { id: string; filename: string; original_filename?: string; document_number?: string | null }[];
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
        {/* Attached Files Preview */}
        {attachedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 px-4 pt-3.5 pb-2">
            <AnimatePresence>
              {attachedFiles.map((file) => (
                <motion.div
                  key={file.id}
                  initial={{ opacity: 0, scale: 0.9, y: 5 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.9, y: 5 }}
                  className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-border shadow-sm group relative"
                >
                  <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                    {file.status === "uploading" || file.status === "parsing" ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <FileText className="w-3.5 h-3.5" />
                    )}
                  </div>
                  <div className="flex flex-col min-w-0 pr-4">
                    <span className="text-[11.5px] font-semibold truncate max-w-[140px] text-foreground">
                      {file.file.name}
                    </span>
                    <span className={cn(
                      "text-[9px] font-bold uppercase tracking-wider",
                      file.status === "indexed" ? "text-emerald-500" :
                        file.status === "ready" ? "text-amber-500" :
                          file.status === "failed" ? "text-destructive" :
                            "text-muted-foreground/60"
                    )}>
                      {file.status === "indexed" ? t("chat.status_indexed") || "Sẵn sàng" :
                        file.status === "ready" ? t("chat.status_ready") || "Sẵn sàng truy vấn" :
                          file.status === "parsing" ? t("chat.status_parsing") || "Đang xử lý..." :
                            file.status === "uploading" ? t("chat.status_uploading") || "Đang tải lên..." :
                              t("chat.status_failed") || "Lỗi"}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => onRemoveAttachment(file.id)}
                    className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-zinc-200 dark:bg-zinc-800 border border-border flex items-center justify-center text-muted-foreground hover:bg-destructive hover:text-destructive-foreground transition-all opacity-0 group-hover:opacity-100 shadow-sm"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}



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
        <div className="flex items-center justify-between px-2.5 pb-2.5 pt-0.5">
          <div className="flex items-center gap-2">
            {/* Unified Tools Dropdown */}
            <ToolsDropdown
              onPlus={onPlus}
              onMic={onMic}
              forceSearch={forceSearch}
              onToggleSearch={onToggleSearch}
            />

            {/* Force Search Toggle — Compact with Portal Tooltip */}
            <SearchTooltipWrapper
              forceSearch={forceSearch}
              onToggleSearch={onToggleSearch}
              t={t}
            />
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
                  aria-label={t("chat.cancel") || "Stop"}
                  className="w-10 h-10 rounded-full flex items-center justify-center bg-destructive/10 text-destructive hover:bg-destructive/15 transition-all shadow-sm ring-1 ring-destructive/20 cursor-pointer"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
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
                  className="w-10 h-10 rounded-full flex items-center justify-center text-muted-foreground/60 hover:text-primary hover:bg-primary/5 transition-all cursor-default"
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

interface ChatPanelProps {
  sessionId: string | null;
  sessionTitle?: string;
}

export const ChatPanel = memo(function ChatPanel({
  sessionId,
  sessionTitle,
}: ChatPanelProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createSession = useCreateChatSession();
  const { user } = useAuthStore();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mentionSelectedIndex, setMentionSelectedIndex] = useState(0);
  const [input, setInput] = useState(() => {
    if (!sessionId) return localStorage.getItem("hrag-draft-new") || "";
    return localStorage.getItem(`hrag-draft-${sessionId}`) || "";
  });
  const [enableThinking, setEnableThinking] = useState(true);
  const [thinkingDefaultSynced, setThinkingDefaultSynced] = useState(false);
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const [referencedDocs, setReferencedDocs] = useState<{ id: string; filename: string; original_filename?: string }[]>([]);
  const [docMetadataMap, setDocMetadataMap] = useState<Map<string, Document>>(new Map());
  const skipResetRef = useRef<string | null>(null);

  const [forceSearch, setForceSearch] = useState(true);

  // @docname mention states
  const [showMentionDropdown, setShowMentionDropdown] = useState(false);
  const [mentionSearch, setMentionSearch] = useState("");

  // 2. Personal Workspace Detection
  const { data: workspaces } = useQuery<any[]>({
    queryKey: ["workspaces"],
    queryFn: () => api.get<any[]>("/workspaces"),
  });

  const personalWorkspace = useMemo(() => {
    if (!workspaces) return undefined;
    const ws = workspaces as any;
    return (
      ws.find((item: any) => item.is_default === true) ||
      ws.find((item: any) => item.visibility === "personal") ||
      ws[0]
    );
  }, [workspaces]);

  const currentWorkspaceId = personalWorkspace?.id;

  // Fetch all workspace docs to build metadata map for inline file badges in history messages
  const { data: workspaceDocs } = useDocuments(currentWorkspaceId);

  // Fetch session documents from API for @mention autocomplete
  const { data: sessionDocsData } = useSessionDocuments(sessionId);

  const filteredMentionDocs = useMemo(() => {
    const sessionDocs = (sessionDocsData as any)?.documents || [];
    const workspaceDocsList = workspaceDocs || [];

    // Map locally attached files so they immediately appear in mentions
    // Use docMetadata.original_filename if available (from API response after upload)
    const attachedDocsInfo = attachedFiles
      .map((af) => ({
        id: af.id,
        filename: af.file.name,
        original_filename: af.docMetadata?.original_filename || af.file.name,
        file_type: af.file.name.split('.').pop(),
      }));

    // Map workspace docs - they have original_filename from the API
    const workspaceDocsInfo = workspaceDocsList.map((doc: any) => ({
      id: doc.id,
      filename: doc.filename || doc.original_filename || "Untitled",
      original_filename: doc.original_filename || doc.filename,
      file_type: doc.file_type || (doc.filename?.split('.')?.pop()),
      document_number: doc.document_number
    }));

    const allDocs = [...attachedDocsInfo, ...sessionDocs, ...workspaceDocsInfo];
    if (allDocs.length === 0) return [];
    // Deduplicate by ID
    const uniqueDocs = Array.from(new Map(allDocs.map(d => [d.id, d])).values());

    // If no search term, show the most recent or all
    if (!mentionSearch) {
      return uniqueDocs
        .slice(0, 8)
        .map(doc => ({
          id: doc.id,
          filename: doc.filename || doc.original_filename || "Untitled",
          original_filename: doc.original_filename,
          file_type: doc.file_type || (doc.filename?.split('.').pop()),
          document_number: doc.document_number
        }));
    }

    // Filter by search term
    const search = mentionSearch.toLowerCase();
    return uniqueDocs
      .filter(doc =>
        doc.filename?.toLowerCase().includes(search) ||
        doc.original_filename?.toLowerCase().includes(search) ||
        (doc.document_number && doc.document_number.toLowerCase().includes(search))
      )
      .slice(0, 8)
      .map(doc => ({
        id: String(doc.id),
        filename: doc.filename || doc.original_filename || "Untitled",
        original_filename: doc.original_filename,
        file_type: doc.file_type || (doc.filename?.split('.').pop()),
        document_number: doc.document_number
      }));
  }, [workspaceDocs, sessionDocsData, mentionSearch, attachedFiles]);

  // Sync index on search change
  useEffect(() => {
    setMentionSelectedIndex(0);
  }, [mentionSearch]);

  const handleMentionInput = useCallback((text: string, cursorPos: number) => {
    const textBeforeCursor = text.slice(0, cursorPos);
    const match = textBeforeCursor.match(/(?:^|\s)@([^\s]*)$/);

    if (!match) {
      setShowMentionDropdown(false);
      setMentionSearch("");
      return;
    }

    setMentionSearch(match[1]);
    setShowMentionDropdown(true);
  }, []);  // Helper: insert selected doc into input
  const insertMentionDoc = useCallback((doc: { id: string; filename: string; original_filename?: string }) => {
    const textBeforeCursor = input.slice(0, inputRef.current?.selectionStart || 0);
    const atIndex = textBeforeCursor.lastIndexOf('@');
    if (atIndex === -1) return;

    // Replace the @query part with the formatted document name
    const textBeforeAt = input.slice(0, atIndex);
    const textAfterMention = input.slice(inputRef.current?.selectionStart || 0);

    const displayName = formatMentionName(doc.original_filename || doc.filename);
    const newInput = `${textBeforeAt}@${displayName} ${textAfterMention}`;

    setInput(newInput);
    setShowMentionDropdown(false);
    setMentionSearch("");

    // Set cursor position after the inserted mention
    setTimeout(() => {
      if (inputRef.current) {
        const newCursorPos = textBeforeAt.length + displayName.length + 2; // +1 for @, +1 for space
        inputRef.current.setSelectionRange(newCursorPos, newCursorPos);
        inputRef.current.focus();
      }
    }, 0);

    // Add to referenced docs
    if (!referencedDocs.find(d => d.id === doc.id)) {
      setReferencedDocs(prev => [...prev, { id: doc.id, filename: doc.filename, original_filename: doc.original_filename }]);
    }
  }, [input, referencedDocs]);

  // Helper: remove referenced doc
  const removeReferencedDoc = useCallback((docId: string) => {
    setReferencedDocs(prev => prev.filter(d => d.id !== docId));
  }, []);

  // Build docMetadataMap from workspace docs
  useEffect(() => {
    if (!workspaceDocs) return;
    const map = new Map<string, Document>();
    for (const doc of workspaceDocs) {
      map.set(doc.id, doc);
    }
    setDocMetadataMap(map);
  }, [workspaceDocs]);

  // Reset session state when switching chats/starting a new chat
  useEffect(() => {
    const isDebug = document.documentElement.classList.contains("debug-mode") ||
                    localStorage.getItem("hrag-debug-mode") === "true";

    // Only skip reset if this session was just created (handleSend set this)
    if (skipResetRef.current === sessionId) {
      if (isDebug) console.log(`[Persistence] Skipping reset for NEW session: ${sessionId}`);
      skipResetRef.current = null;
      return;
    }

    // For any other session change, clear the skip flag and reset
    if (skipResetRef.current !== null) {
      skipResetRef.current = null;
    }

    if (isDebug) console.log(`[Persistence] Session ID changed: ${sessionId || "new"}. Loading defaults...`);

    setMessages([]);
    setReferencedDocs([]);
    setAttachedFiles([]);

    const key = sessionId ? `hrag-draft-${sessionId}` : "hrag-draft-new";
    const mentionKey = sessionId ? `hrag-mentions-${sessionId}` : "hrag-mentions-new";

    const savedDraft = localStorage.getItem(key) || "";
    setInput(savedDraft);

    const savedMentions = localStorage.getItem(mentionKey);
    if (savedMentions) {
      try {
        const parsed = JSON.parse(savedMentions);
        setReferencedDocs(parsed);
        if (isDebug) console.log(`[Persistence] Restored ${parsed.length} mentions for ${sessionId || "new"}`);
      } catch (err) {
        console.error("Failed to parse saved mentions:", err);
      }
    }
  }, [sessionId]);

  useEffect(() => {
    const key = sessionId ? `hrag-draft-${sessionId}` : "hrag-draft-new";
    const mentionKey = sessionId ? `hrag-mentions-${sessionId}` : "hrag-mentions-new";

    // Text Draft
    if (input.trim()) {
      localStorage.setItem(key, input);
    } else {
      localStorage.removeItem(key);
    }

    // Mention Draft
    if (referencedDocs.length > 0) {
      localStorage.setItem(mentionKey, JSON.stringify(referencedDocs));
    } else {
      localStorage.removeItem(mentionKey);
    }
  }, [sessionId, input, referencedDocs]);

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
  const docxInputRef = useRef<HTMLInputElement>(null);

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

  // 7. Background File Upload & Polling
  const pollDocumentStatus = useCallback(async (docId: string) => {
    let attempts = 0;
    const maxAttempts = 30;

    const poll = async () => {
      try {
        const doc = await api.get<any>(`/documents/${docId}`);
        const docData = doc as any;
        if (docData.status === "indexed" || docData.status === "building_kg") {
          queryClient.invalidateQueries({ queryKey: ["documents", currentWorkspaceId] });
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "indexed", progress: 100, docMetadata: docData } : f
          ));
          return;
        }
        // Parse done → embedding in background. Allow chat immediately.
        if (docData.status === "chunking" || docData.status === "embedding") {
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "ready", progress: 70, docMetadata: docData } : f
          ));
          return;
        }
        if (docData.status === "failed") {
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "failed" } : f
          ));
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 2000);
        } else {
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "failed" } : f
          ));
          toast.error(t("chat.upload_failed"));
        }
      } catch (err: any) {
        // If document not found (404) or other error, stop polling
        console.error("Polling failed:", err);
        setAttachedFiles(prev => prev.map(f =>
          f.id === docId ? { ...f, status: "failed" } : f
        ));
        // Don't continue polling on error - stop here
        return;
      }
    };

    poll();
  }, []);

  const handleFileSelect = useCallback(
    async (file: File) => {
      if (!currentWorkspaceId) {
        toast.error(t("chat.no_workspace"));
        return;
      }

      if (file.name.toLowerCase().endsWith(".doc")) {
        toast.error(t("chat.unsupported_doc"));
        return;
      }

      const tempId = generateId();
      const newAttachedFile: AttachedFile = {
        id: tempId,
        file,
        status: "uploading",
        progress: 10,
      };

      setAttachedFiles((prev) => [...prev, newAttachedFile]);

      try {
        const response = await api.uploadFile<any>(
          `/documents/upload/${currentWorkspaceId}`,
          file,
          { "X-Chat-Upload": "true" }
        );

        const docId = response.id;

        setAttachedFiles(prev => prev.map(f =>
          f.id === tempId ? { ...f, id: docId, status: "parsing", progress: 40, docMetadata: response } : f
        ));

        // Invalidate workspace docs so mention dropdown gets updated list
        queryClient.invalidateQueries({ queryKey: ["documents", currentWorkspaceId] });

        pollDocumentStatus(docId);
      } catch (err) {
        toast.error(t("chat.upload_failed"));
        setAttachedFiles(prev => prev.map(f =>
          f.id === tempId ? { ...f, status: "failed" } : f
        ));
      }
    },
    [currentWorkspaceId, pollDocumentStatus, queryClient, t]
  );

  const removeAttachment = useCallback((id: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== id));
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
  useEffect(() => {
    if (historyData?.messages) {
      setMessages((prev) => {
        const stepsMap = new Map<string, AgentStep[]>();
        for (const m of prev) {
          if (m.agentSteps?.length) stepsMap.set(m.id, m.agentSteps);
        }

        const dbMessages = historyData.messages.map((m) => ({
          id: m.message_id,
          role: m.role as "user" | "assistant",
          content: m.content,
          documentIds: m.document_ids ?? undefined,
          attachedDocs: m.attached_docs ?? undefined,
          sources: m.sources ?? undefined,
          relatedEntities: m.related_entities ?? undefined,
          imageRefs: m.image_refs ?? undefined,
          thinking: m.thinking ?? undefined,
          timestamp: m.created_at,
          potential_abbreviations: m.potential_abbreviations ?? undefined,
          peopleData: m.people_data ?? undefined,
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

        const dbIds = new Set(dbMessages.map((m) => m.id));
        const dbUserContents = new Set(dbMessages.filter(m => m.role === 'user').map(m => m.content));
        const dbAssistantContents = new Set(dbMessages.filter(m => m.role === 'assistant').map(m => m.content));

        const localOnly = prev.filter((m) => {
          if (dbIds.has(m.id)) return false;
          if (m.role === 'user' && dbUserContents.has(m.content)) return false;
          if (m.role === 'assistant' && dbAssistantContents.has(m.content)) return false;
          return true;
        });

        if (localOnly.length === 0) return dbMessages;

        return [...dbMessages, ...localOnly];
      });
    }
  }, [historyData]);

  // SSE streaming chat
  const updateSessionTitle = useUpdateSessionTitle();
  const stream = useRAGChatStream(
    sessionId,
    useCallback(
      (newTitle: string) => {
        if (sessionId) {
          updateSessionTitle(sessionId, newTitle);
        }
        queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      },
      [sessionId, updateSessionTitle, queryClient],
    ),
  );
  const streamingMsgIdRef = useRef<string | null>(null);
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
        setMessages((prev) => {
          // Find and update the message by LOCAL id, then update ref to server id
          const idx = prev.findIndex((m) => m.id === localId);
          if (idx === -1) return prev;
          const updated = [...prev];
          updated[idx] = { ...updated[idx], id: serverId };
          return updated;
        });
        // Update ref to server ID so sync effect can find it
        streamingMsgIdRef.current = serverId;
      }
    }
  }, [stream.aiMessageId]);

  // Sync server-assigned user message ID to local message
  useEffect(() => {
    if (stream.userMessageId) {
      const serverId = stream.userMessageId;
      setMessages((prev) => {
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

    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = undefined;
    }

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

        const scrollEl = el;
        function animate(now: number) {
          const t = Math.min((now - startTime) / duration, 1);
          const ease = 1 - Math.pow(1 - t, 3);
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
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const container = scrollContainerRef.current;
        if (!container) return;

        if (spacerRef.current) {
          spacerRef.current.style.height = `${container.clientHeight}px`;
        }

        const el = container.querySelector(`[data-message-id="${msgId}"]`) as HTMLElement | null;
        if (!el) return;

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
          const ease = 1 - Math.pow(1 - t, 3);
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

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || !spacerRef.current) return;

    if (stream.isStreaming) {
      spacerRef.current.style.height = `${container.clientHeight}px`;
    } else {
      spacerRef.current.style.height = "0px";
    }
  }, [stream.isStreaming]);

  const prevIsStreamingRef = useRef(false);
  const justFinishedStreamingRef = useRef(false);
  useEffect(() => {
    if (prevIsStreamingRef.current && !stream.isStreaming) {
      justFinishedStreamingRef.current = true;
    }
    prevIsStreamingRef.current = stream.isStreaming;
  }, [stream.isStreaming]);

  useEffect(() => {
    if (!stream.isStreaming) {
      if (justFinishedStreamingRef.current) {
        justFinishedStreamingRef.current = false;
        return;
      }
      scrollToBottom();
    }
  }, [messages, stream.isStreaming, scrollToBottom]);

  useEffect(() => {
    if (!streamingMsgIdRef.current) return;
    const id = streamingMsgIdRef.current;
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === id);
      if (idx === -1) return prev;
      const m = prev[idx];

      const newContent = stream.streamingContent;
      const newSources = stream.pendingSources.length > 0 ? stream.pendingSources : m.sources;
      const newImages = stream.pendingImages.length > 0 ? stream.pendingImages : m.imageRefs;
      const newThinking = stream.thinkingText || m.thinking;
      const newSteps = stream.agentSteps.length > 0 ? stream.agentSteps : m.agentSteps;
      const newPotentials = stream.potentialAbbreviations.length > 0 ? stream.potentialAbbreviations : m.potential_abbreviations;
      // Only use pendingPeople during active streaming; after complete fires, peopleData is already set
      // Don't update peopleData if pendingPeople is empty (might be from sendMessage reset before complete)
      const newPeople = stream.pendingPeople.length > 0
        ? stream.pendingPeople
        : (stream.isStreaming ? m.peopleData : (m.peopleData ?? stream.pendingPeople));

      if (
        m.content === newContent &&
        m.sources === newSources &&
        m.imageRefs === newImages &&
        m.thinking === newThinking &&
        m.agentSteps === newSteps &&
        m.potential_abbreviations === newPotentials &&
        m.peopleData === newPeople &&
        m.isStreaming === stream.isStreaming
      ) {
        return prev;
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
        peopleData: newPeople,
        isStreaming: stream.isStreaming,
      };
      return updated;
    });
  }, [stream.streamingContent, stream.pendingSources, stream.pendingImages, stream.thinkingText, stream.isStreaming, stream.agentSteps, stream.pendingPeople, stream.streamCompleteTick]);

  const handleSend = useCallback(
    async (text?: string) => {
      const msg = (text || input).trim();
      const isStillProcessing = attachedFiles.some(f => f.status === "uploading");
      if (isStillProcessing) {
        toast.info(t("chat.wait_for_files"));
        return;
      }
      if (!msg && attachedFiles.length === 0) return;
      if (stream.isStreaming) return;

      let effectiveSessionId = sessionId;
      if (!effectiveSessionId) {
        try {
          const newSession = await createSession.mutateAsync({ title: msg.slice(0, 30) || t("nav.new_chat") });
          effectiveSessionId = newSession.id;

          // Clear "New Chat" draft/mentions since we are sending it
          localStorage.removeItem("hrag-draft-new");
          localStorage.removeItem("hrag-mentions-new");
          
          if (document.documentElement.classList.contains("debug-mode")) {
            console.log(`[Persistence] Cleared "new" draft for session "${effectiveSessionId}"`);
          }

          skipResetRef.current = newSession.id;
          // Update URL so chat is bookmarkable
          navigate(`/chat/${newSession.id}`, { replace: true });
        } catch (err: any) {
          toast.error(t("chat.create_failed"));
          return;
        }
      }

      // Only send @ mentioned docs + newly attached files (not all session docs)
      // "parsing", "ready" files (parse done, embed in background) are also included —
      // backend fetches markdown directly from MinIO so content is available immediately.
      // Filter mentions to only include docs still present in user text
      const validMentions = referencedDocs.filter(doc => {
        const displayName = formatMentionName(doc.original_filename || doc.filename);
        return msg.includes(`@${displayName}`);
      });

      setReferencedDocs(validMentions); // Clean up unused mentions

      const documentIds = [
        ...validMentions.map(d => d.id),
        ...attachedFiles.filter(f => f.status === "indexed" || f.status === "ready" || f.status === "parsing").map(f => f.id),
      ];
      setAttachedFiles([]);

      const attachedDocs = attachedFiles
        .filter(f => (f.status === "indexed" || f.status === "ready" || f.status === "parsing") && f.docMetadata)
        .map(f => f.docMetadata as Document);

      let msgToBackend = msg;
      validMentions.forEach(doc => {
         const truncatedName = formatMentionName(doc.original_filename || doc.filename);
         const escapedTruncated = truncatedName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
         msgToBackend = msgToBackend.replace(new RegExp(`@${escapedTruncated}`, 'g'), `<document_id=${doc.id}>`);
      });

      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: msgToBackend,
        timestamp: new Date().toISOString(),
        documentIds: documentIds.length > 0 ? documentIds : undefined,
        attachedDocs: attachedDocs.length > 0 ? attachedDocs : undefined,
      };

      const assistantId = generateId();
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
      setReferencedDocs([]); // Clear mentions draft
      localStorage.removeItem(sessionId ? `hrag-draft-${sessionId}` : "hrag-draft-new");
      localStorage.removeItem(sessionId ? `hrag-mentions-${sessionId}` : "hrag-mentions-new");
      // Scroll new user message to top so agent response fills the space below
      scrollUserMsgToTop(userMsg.id);

      // Build history from previous messages (exclude the new user + placeholder)
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const finalMsg = await stream.sendMessage(
        msgToBackend,
        history,
        thinkingSupported && enableThinking,
        forceSearch,
        effectiveSessionId || undefined,
        documentIds
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
                id: finalMsg.id,
                isStreaming: false, // Ensure streaming is false when finalizing
                agentSteps: finalMsg.agentSteps?.length
                  ? finalMsg.agentSteps
                  : agentStepsRef.current.length > 0
                    ? agentStepsRef.current
                    : m.agentSteps,
              }
              : m,
          ),
        );
        // Only clear streaming ref AFTER setMessages completes and isStreaming is false
        // This prevents race condition where sync effect could update wrong message
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
        streamingMsgIdRef.current = null;
      } else {
        // Cancelled — keep partial content
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        );
        streamingMsgIdRef.current = null;
      }
    },
    [input, messages, stream, thinkingSupported, enableThinking, forceSearch, scrollUserMsgToTop, sessionId, navigate, createSession, t, attachedFiles],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showMentionDropdown) {
      if (e.key === "Escape") {
        if (showMentionDropdown) {
          setShowMentionDropdown(false);
          setMentionSearch("");
        } else if (input.trim()) {
          setInput("");
        }
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setMentionSelectedIndex((prev) => (prev + 1) % (filteredMentionDocs.length || 1));
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setMentionSelectedIndex((prev) => (prev - 1 + (filteredMentionDocs.length || 1)) % (filteredMentionDocs.length || 1));
        return;
      }
      if (e.key === "Enter" && filteredMentionDocs.length > 0) {
        e.preventDefault();
        insertMentionDoc(filteredMentionDocs[mentionSelectedIndex]);
        return;
      }
    }

    if (e.key === "Backspace" && inputRef.current && !showMentionDropdown) {
      const cursor = inputRef.current.selectionStart;
      if (cursor === inputRef.current.selectionEnd) {
        const textBefore = input.slice(0, cursor);

        for (const doc of referencedDocs) {
          const displayName = formatMentionName(doc.original_filename || doc.filename);
          const exactMatch1 = textBefore.endsWith(`@${displayName} `) ? `@${displayName} ` : "";
          const exactMatch2 = textBefore.endsWith(`@${displayName}`) ? `@${displayName}` : "";
          const matchStr = exactMatch1 || exactMatch2;
          
          if (matchStr) {
            e.preventDefault();
            const newInput = textBefore.slice(0, -matchStr.length) + input.slice(cursor);
            setInput(newInput);
            
            // Allow state to update, then fix cursor
            setTimeout(() => {
              if (inputRef.current) {
                const newPos = cursor - matchStr.length;
                inputRef.current.setSelectionRange(newPos, newPos);
              }
            }, 0);
            return;
          }
        }
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleMicClick = useCallback(() => {
    toast.info(t("chat.audio_feature_coming") || "Tính năng đang phát triển");
  }, [t]);

  const handlePlusClick = useCallback(() => {
    docxInputRef.current?.click();
  }, []);

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
            <div className="flex-shrink-0 flex items-center justify-between px-6 py-4 bg-background/40 backdrop-blur-xl border-b border-border/40">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-background border border-border/60 overflow-hidden shadow-sm transition-transform hover:scale-105">
                  <img src="/logo.png" alt="AIRAG" className="w-5.5 h-5.5 object-contain" />
                </div>
                <div>
                  <h2 className="text-[14px] font-bold tracking-tight text-foreground line-clamp-1">
                    {sessionTitle || (sessionId ? `${t("chat.session", { id: sessionId })}` : t("chat.select_session"))}
                  </h2>
                  <div className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">{t("chat.assistant_online")}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Main Content Area */}
            {messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center px-4 overflow-y-auto pb-[10vh] scrollbar-none">
                <div className="w-full max-w-[720px] flex flex-col items-center translate-y-[-4vh]">
                  {/* Greeting */}
                  <div className="mb-12 text-center animate-in fade-in zoom-in-95 duration-1000 ease-out">
                    <div className="inline-flex items-center gap-2 mb-6 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary shadow-[0_0_15px_rgba(var(--color-primary),0.1)]">
                      <Sparkles className="w-4 h-4" />
                      <span className="text-[12px] font-bold uppercase tracking-[0.1em]">{t("chat.ai_assistant")}</span>
                    </div>
                    <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-foreground mb-6 bg-gradient-to-b from-foreground to-foreground/60 bg-clip-text text-transparent">
                      {t("chat.greeting", { name: user?.full_name || "XayDung" })}
                    </h1>
                    <p className="text-xl md:text-2xl text-muted-foreground/50 font-medium max-w-[600px] mx-auto leading-relaxed">
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
                      attachedFiles={attachedFiles}
                      onRemoveAttachment={removeAttachment}
                      inputRef={inputRef}
                      handleKeyDown={handleKeyDown}
                      onPlus={handlePlusClick}
                      onMic={handleMicClick}
                      t={t}
                      referencedDocs={referencedDocs}
                      onRemoveReferencedDoc={removeReferencedDoc}
                      showMentionDropdown={showMentionDropdown}
                      filteredMentionDocs={filteredMentionDocs}
                      onSelectMentionDoc={insertMentionDoc}
                      onCloseMentionDropdown={() => {
                        setShowMentionDropdown(false);
                        setMentionSearch("");
                      }}
                      onInputChange={handleMentionInput}
                      mentionSelectedIndex={mentionSelectedIndex}
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
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8, transition: { duration: 0.15 } }}
                        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                      >
                        <MessageBubble
                          message={msg}
                          onAddAbbreviation={handleOpenAbbModal}
                          docMetadataMap={docMetadataMap}
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
                      attachedFiles={attachedFiles}
                      onRemoveAttachment={removeAttachment}
                      inputRef={inputRef}
                      handleKeyDown={handleKeyDown}
                      onPlus={handlePlusClick}
                      onMic={handleMicClick}
                      t={t}
                      referencedDocs={referencedDocs}
                      onRemoveReferencedDoc={removeReferencedDoc}
                      showMentionDropdown={showMentionDropdown}
                      filteredMentionDocs={filteredMentionDocs}
                      onSelectMentionDoc={insertMentionDoc}
                      onCloseMentionDropdown={() => {
                        setShowMentionDropdown(false);
                        setMentionSearch("");
                      }}
                      onInputChange={handleMentionInput}
                      mentionSelectedIndex={mentionSelectedIndex}
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

      {/* Hidden file input for DOCX format checking */}
      <input
        ref={docxInputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md,.pptx"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileSelect(file);
          if (docxInputRef.current) docxInputRef.current.value = "";
        }}
        className="hidden"
      />
    </SessionIdCtx.Provider>
  );
});

