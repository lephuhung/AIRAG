import { useState, useEffect, useRef, useMemo, useCallback, memo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "@/hooks/useTranslation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";
import { FileText, List, ChevronRight, Maximize2, Minimize2, AlignLeft, LayoutTemplate } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import type { Document, ChatSourceChunk } from "@/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface Heading {
  id: string;
  text: string;
  level: number;
}

interface ChunkContextResponse {
  document_id: string;
  target_chunk_index: number;
  chunk_range: [number, number];
  total_chunks: number;
  chunks: Array<{
    chunk_id: string;
    chunk_index: number;
    content: string;
    page_no: number | null;
    heading_path: string;
    source: string;
  }>;
  markdown: string;
}

// ---------------------------------------------------------------------------
// Skeleton loader
// ---------------------------------------------------------------------------
function ViewerSkeleton() {
  return (
    <div className="p-6 space-y-4 animate-pulse">
      <div className="h-6 bg-muted rounded w-3/5" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-4/5" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-2/3" />
      <div className="h-20 bg-muted rounded w-full mt-4" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-3/4" />
      <div className="h-4 bg-muted rounded w-full" />
      <div className="h-4 bg-muted rounded w-1/2" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Error state
// ---------------------------------------------------------------------------
function ViewerError({ message }: { message: string }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      <FileText className="w-10 h-10 text-muted-foreground/40 mb-3" />
      <p className="text-sm font-medium">{t("viewer.error_loading")}</p>
      <p className="text-xs text-muted-foreground mt-1 max-w-xs">{message}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------
function ViewerEmpty() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center justify-center py-16 px-6 text-center">
      <FileText className="w-10 h-10 text-muted-foreground/30 mb-3" />
      <p className="text-sm text-muted-foreground">
        {t("viewer.no_content")}
      </p>
      <p className="text-xs text-muted-foreground/60 mt-1">
        {t("rag.not_processed_message")}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Table of Contents sidebar
// ---------------------------------------------------------------------------
const TOCSidebar = memo(function TOCSidebar({
  headings,
  activeId,
  onSelect,
}: {
  headings: Heading[];
  activeId: string | null;
  onSelect: (id: string) => void;
}) {
  const { t } = useTranslation();
  if (headings.length === 0) return null;

  return (
    <nav className="w-52 flex-shrink-0 border-r overflow-y-auto py-3 px-2 hidden xl:block">
      <div className="flex items-center gap-1.5 px-2 mb-2">
        <List className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">{t("viewer.contents")}</span>
      </div>
      <ul className="space-y-0.5">
        {headings.map((h) => (
          <li key={h.id}>
            <button
              onClick={() => onSelect(h.id)}
              className={cn(
                "w-full text-left text-xs py-1 px-2 rounded-md transition-colors truncate",
                "hover:bg-muted",
                activeId === h.id
                  ? "text-primary font-medium bg-primary/10"
                  : "text-muted-foreground"
              )}
              style={{ paddingLeft: `${(h.level - 1) * 12 + 8}px` }}
              title={h.text}
            >
              {h.text}
            </button>
          </li>
        ))}
      </ul>
    </nav>
  );
});

// ---------------------------------------------------------------------------
// Page divider — inserted between pages in the markdown
// ---------------------------------------------------------------------------
function PageDivider({ pageNo }: { pageNo: number }) {
  const { t } = useTranslation();
  return (
    <div className="flex items-center gap-3 py-6 select-none" data-page={pageNo}>
      <div className="flex-1 border-t border-dashed border-border/40" />
      <span className="text-[10px] font-semibold text-muted-foreground/70 uppercase tracking-wider px-2.5 py-1 rounded-full bg-muted/40 border border-border/40 shadow-sm">
        {t("common.page_x", { page: pageNo })}
      </span>
      <div className="flex-1 border-t border-dashed border-border/40" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Extract headings from markdown for TOC
// ---------------------------------------------------------------------------
function extractHeadings(markdown: string): Heading[] {
  const headings: Heading[] = [];
  const lines = markdown.split("\n");
  const seenCounts = new Map<string, number>();
  let inCodeFence = false;
  for (const line of lines) {
    // Skip fenced code blocks so "# comment" lines don't become phantom TOC
    // entries — that would also drift the dedupe counter out of sync with the DOM.
    if (line.trim().startsWith("```")) {
      inCodeFence = !inCodeFence;
      continue;
    }
    if (inCodeFence) continue;
    const match = line.match(/^(#{1,4})\s+(.+)/);
    if (match) {
      const level = match[1].length;
      const text = match[2].replace(/[*_`#]/g, "").trim();
      // Use the SAME slug + dedupe logic the renderer uses, so TOC ids always
      // resolve to real DOM ids (even with duplicate heading texts).
      const id = dedupeHeadingId(generateHeadingId(text), seenCounts);
      headings.push({ id, text, level });
    }
  }
  return headings;
}

// ---------------------------------------------------------------------------
// Fix unsupported LaTeX commands before KaTeX processing
// ---------------------------------------------------------------------------
// Insert page dividers into markdown text
// ---------------------------------------------------------------------------
function insertPageDividers(markdown: string): string {
  if (!markdown) return "";

  let result = markdown;

  // Step 1 — Standardize OCR format: <!-- page N --> followed by ---
  // We consume the --- if it's within 50 chars of a page marker to avoid double dividers.
  result = result.replace(
    /<!--\s*page\s+(\d+)\s*-->(\s*)\n+---\n+/gi,
    (_, pageNo, whitespace) => `\n<hr data-page="${pageNo}" />\n${whitespace}`
  );

  // Step 2 — Handle remaining standalone <!-- page N --> markers
  result = result.replace(
    /<!--\s*page\s+(\d+)\s*-->/gi,
    (_, pageNo) => `\n<hr data-page="${pageNo}" />\n`
  );

  // Step 3 — Pre-process all horizontal rules (---) that don't have a page number.
  // To avoid side-effects in render, we'll do a sequential pass here.
  let pageCounter = 1;
  const parts = result.split(/(<hr data-page="\d+" \/>)/i);
  
  const finalParts = parts.map(part => {
    const pageMatch = part.match(/<hr data-page="(\d+)" \/>/i);
    if (pageMatch) {
      pageCounter = parseInt(pageMatch[1]);
      return part;
    }
    
    // For plain --- markers, we auto-increment from the last known page
    return part.replace(/\n---\n/g, () => {
      pageCounter += 1;
      return `\n<hr data-page="${pageCounter}" />\n`;
    });
  });

  result = finalParts.join("");

  // Step 4 — De-duplicate adjacent markers (e.g. <hr ... /><hr ... />)
  // We keep only the last one in a cluster as it's usually the most "current" page.
  result = result.replace(
    /(<hr data-page="\d+" \/>\s*){2,}/gi,
    (match) => {
      const markers = match.match(/<hr data-page="\d+" \/>/gi);
      return markers ? markers[markers.length - 1] : match;
    }
  );

  return result;
}

// ---------------------------------------------------------------------------
// Strip OCR layout HTML for the plain-text view (mirrors backend
// strip_ocr_layout): drop the alignment/2-column tags + data-bbox, keep the
// reading-order text, and turn page markers into readable separators.
// ---------------------------------------------------------------------------
function stripOcrLayoutClient(md: string): string {
  if (!md.includes("data-bbox")) return md;
  let t = md.replace(/<!--\s*page\s+(\d+)\s*-->/gi, "\n— Trang $1 —\n");
  t = t.replace(/<br\s*\/?>/gi, "\n");
  t = t.replace(/<\/(div|p|figure)>/gi, "\n");
  t = t.replace(/<(?!!--)[^>]*>/g, "");
  t = t
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, " ");
  t = t.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n");
  return t.trim();
}

// ---------------------------------------------------------------------------
// DocumentViewer
// ---------------------------------------------------------------------------
interface DocumentViewerProps {
  doc: Document;
  scrollToPage?: number | null;
  scrollToHeading?: string | null;
  scrollToImageSrc?: string | null;
  highlightChunks?: ChatSourceChunk[];
  onScrolled?: () => void;
}

export const DocumentViewer = memo(function DocumentViewer({
  doc,
  scrollToPage,
  scrollToHeading,
  scrollToImageSrc,
  highlightChunks,
  onScrolled,
}: DocumentViewerProps) {
  const { t } = useTranslation();
  const contentRef = useRef<HTMLDivElement>(null);
  const [activeHeading, setActiveHeading] = useState<string | null>(null);
  const [showToc, setShowToc] = useState(true);

  // Per-render counter for heading-id deduplication. Reset on every render
  // (below) so the memoized markdown components number duplicate headings in
  // document order — matching the ids produced by extractHeadings() for the TOC.
  const headingIdCounter = useRef<Map<string, number>>(new Map());

  // ---- View mode: chunk (lightweight) vs full (entire markdown) ----
  // Citation clicks open chunk mode; opening a file from the library stays full.
  const [viewMode, setViewMode] = useState<"chunk" | "full">("full");

  // Determine if we should use chunk-context mode
  const hasHighlight = highlightChunks && highlightChunks.length > 0;
  const primaryChunk = hasHighlight ? highlightChunks[0] : null;

  // Whether the highlighted chunk carries enough info to be located in the doc.
  // Without a locator the chunk-context API silently falls back to chunk 0, so
  // such citations (summary / whole-doc chunks) are shown in full view instead.
  const hasChunkLocator = useMemo(() => {
    if (!primaryChunk) return false;
    if (primaryChunk.page_no === 0) return false; // summary / whole-doc chunk
    if (primaryChunk.chunk_id?.includes("summary") || primaryChunk.chunk_id?.includes("full")) return false;
    return (
      /chunk_(\d+)$/.test(primaryChunk.chunk_id ?? "") ||
      !!primaryChunk.page_no ||
      (primaryChunk.heading_path?.length ?? 0) > 0
    );
  }, [primaryChunk?.chunk_id, primaryChunk?.page_no, primaryChunk?.heading_path?.length]);

  // When new highlights arrive, switch to chunk mode automatically — but only
  // when the chunk is actually locatable; otherwise fall back to the full doc.
  useEffect(() => {
    if (hasHighlight) {
      setViewMode(hasChunkLocator ? "chunk" : "full");
    }
  }, [hasHighlight, hasChunkLocator, primaryChunk?.chunk_id, primaryChunk?.page_no]);

  const effectiveMode = hasHighlight && hasChunkLocator ? viewMode : "full";

  // ---- Fetch chunk context (lightweight) ----
  const chunkQueryParams = useMemo(() => {
    if (!primaryChunk || !hasChunkLocator) return "";
    const params = new URLSearchParams();
    // Use chunk_id to extract chunk_index
    const idxMatch = primaryChunk.chunk_id?.match(/chunk_(\d+)$/);
    if (idxMatch) {
      params.set("chunk_index", idxMatch[1]);
    } else if (primaryChunk.page_no) {
      params.set("page_no", String(primaryChunk.page_no));
    }
    if (primaryChunk.heading_path?.length > 0) {
      params.set("heading_path", primaryChunk.heading_path.join(" > "));
    }
    params.set("context_window", "3");
    return params.toString();
  }, [primaryChunk?.chunk_id, primaryChunk?.page_no, primaryChunk?.heading_path?.join(","), hasChunkLocator]);

  const { data: chunkContext, isLoading: isChunkLoading, error: chunkError } = useQuery({
    queryKey: ["document-chunk-context", doc.id, chunkQueryParams],
    queryFn: () => api.get<ChunkContextResponse>(`/documents/${doc.id}/chunk-context?${chunkQueryParams}`),
    enabled: doc.status === "indexed" && effectiveMode === "chunk" && !!chunkQueryParams,
    staleTime: 2 * 60 * 1000,
  });

  // ---- Fetch full markdown content ----
  const { data: fullMarkdown, isLoading: isFullLoading, error: fullError } = useQuery({
    queryKey: ["document-markdown", doc.id],
    queryFn: () => api.getText(`/documents/${doc.id}/markdown`),
    enabled: doc.status === "indexed" && effectiveMode === "full",
    staleTime: 5 * 60 * 1000, // cache 5 min
  });

  // ---- Resolved markdown based on mode ----
  const markdown = effectiveMode === "chunk" ? chunkContext?.markdown : fullMarkdown;
  const isLoading = effectiveMode === "chunk" ? isChunkLoading : isFullLoading;
  const error = effectiveMode === "chunk" ? chunkError : fullError;

  // ---- Extract headings for TOC ----
  const headings = useMemo(
    () => (markdown ? extractHeadings(markdown) : []),
    [markdown]
  );

  // Reset page counter whenever the document changes (not just markdown content).
  // This prevents the counter from leaking between documents when markdown is cached.
  // No side-effects needed here as page numbers are pre-assigned in processedMarkdown

  // ---- OCR layout: detect + plain/rendered toggle ----
  // Reconstructed administrative-layout markdown carries data-bbox attributes;
  // only then do we offer the "văn bản thuần" toggle.
  const hasOcrLayout = useMemo(
    () => !!markdown && markdown.includes("data-bbox"),
    [markdown]
  );
  const [plainMode, setPlainMode] = useState(false);
  const plainText = useMemo(
    () => (hasOcrLayout ? stripOcrLayoutClient(markdown || "") : ""),
    [markdown, hasOcrLayout]
  );

  // ---- Process markdown (insert page dividers) ----
  const processedMarkdown = useMemo(() => {
    return markdown ? insertPageDividers(markdown) : "";
  }, [markdown]);

  // Reset the heading-id counter before the markdown subtree renders this pass.
  // The h1–h4 components (memoized below) consume it in document order, so ids
  // stay deterministic and identical across renders for the same content.
  headingIdCounter.current = new Map();

  // Signer for the OCR signature-block placeholder (Phase 2: real data).
  // Prefers the manual override, then the first digital signature's signer.
  const signerName = useMemo(
    () => doc.signer_name || doc.digital_signatures?.[0]?.signer_name || "",
    [doc.signer_name, doc.digital_signatures]
  );

  // ---- Stable ReactMarkdown components (prevents DOM recreation on re-render) ----
  // Without memoization, inline arrow functions create new references each render,
  // causing React to unmount/remount all heading elements — destroying highlight classes.
  const mdComponents = useMemo<import("react-markdown").Components>(() => ({
    h1: ({ children, ...props }) => {
      const id = dedupeHeadingId(generateHeadingId(getHeadingText(children)), headingIdCounter.current);
      return <h1 id={id} {...props}>{children}</h1>;
    },
    h2: ({ children, ...props }) => {
      const id = dedupeHeadingId(generateHeadingId(getHeadingText(children)), headingIdCounter.current);
      return <h2 id={id} {...props}>{children}</h2>;
    },
    h3: ({ children, ...props }) => {
      const id = dedupeHeadingId(generateHeadingId(getHeadingText(children)), headingIdCounter.current);
      return <h3 id={id} {...props}>{children}</h3>;
    },
    h4: ({ children, ...props }) => {
      const id = dedupeHeadingId(generateHeadingId(getHeadingText(children)), headingIdCounter.current);
      return <h4 id={id} {...props}>{children}</h4>;
    },
    hr: (props) => {
      // data-page is set during insertPageDividers (pre-processing)
      const dp = (props as Record<string, unknown>)["data-page"];
      const pageNo = dp != null ? Number(dp) : 0;
      
      if (pageNo === 0) {
        return <hr className="my-8 border-t border-border/20" />;
      }
      return <PageDivider pageNo={pageNo} />;
    },
    p: ({ children, node, ...props }) => {
      const hasImage = (node as any)?.children?.some(
        (child: any) => child.type === "element" && child.tagName === "img"
      );
      if (hasImage)
        return (
          <div className="mb-3 leading-relaxed text-foreground/80" {...props}>
            {children}
          </div>
        );
      return <p {...props}>{children}</p>;
    },
    img: ({ src, alt, ...props }) => (
      <figure className="my-4">
        <img
          src={src}
          alt={alt || ""}
          loading="lazy"
          className="rounded-lg max-w-full mx-auto border border-border/30"
          style={{ minHeight: 120, objectFit: "contain", background: "var(--muted)" }}
          onLoad={(e) => {
            (e.target as HTMLImageElement).style.minHeight = "auto";
            (e.target as HTMLImageElement).style.background = "none";
          }}
          onError={(e) => {
            (e.target as HTMLImageElement).style.display = "none";
          }}
          {...props}
        />
        {alt && (
          <figcaption className="text-xs text-muted-foreground text-center mt-1.5 italic">
            {alt}
          </figcaption>
        )}
      </figure>
    ),
    // OCR signature/seal region (raw <figure class="ocr-figure">). When the
    // pipeline captured a signer, render a real signature block instead of the
    // bare placeholder.
    figure: ({ children, ...props }) => {
      const raw = (props as Record<string, unknown>).className;
      const cls = Array.isArray(raw) ? raw.join(" ") : String(raw ?? "");
      if (cls.includes("ocr-figure") && signerName) {
        return (
          <div className="ocr-sign-block" data-bbox={(props as Record<string, unknown>)["data-bbox"] as string}>
            <div className="ocr-sign-label">(Đã ký)</div>
            <div className="ocr-sign-name">{signerName}</div>
          </div>
        );
      }
      return <figure {...props}>{children}</figure>;
    },
  }), [signerName]);

  // Stable plugin arrays
  const remarkPlugins = useMemo(() => [remarkGfm, remarkMath], []);
  const rehypePlugins = useMemo(() => [rehypeKatex, rehypeRaw], []);

  // ---- Intersection observer for active heading ----
  useEffect(() => {
    if (!contentRef.current || headings.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveHeading(entry.target.id);
          }
        }
      },
      { root: contentRef.current, rootMargin: "-20% 0px -60% 0px", threshold: 0 }
    );

    const headingElements = contentRef.current.querySelectorAll("h1, h2, h3, h4");
    headingElements.forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, [headings, processedMarkdown]);

  // ---- Scroll-to support (from citation cross-link) ----
  // Manual scrollTop + rAF animation — immune to browser cancelling smooth scroll
  // on layout shifts (lazy images, tab switches, etc.)

  const scrollTo = useCallback(
    (
      target: HTMLElement,
      block: "start" | "center" = "center",
      onDone?: () => void
    ) => {
      // Find the element's REAL scrollable ancestor (the height chain / which
      // element actually scrolls isn't guaranteed to be contentRef), falling
      // back to contentRef. Using offsetParent-walking broke when contentRef
      // wasn't positioned, so measure with getBoundingClientRect instead.
      let container: HTMLElement | null = target.parentElement;
      while (container) {
        const oy = getComputedStyle(container).overflowY;
        if (
          (oy === "auto" || oy === "scroll") &&
          container.scrollHeight > container.clientHeight + 2
        ) {
          break;
        }
        container = container.parentElement;
      }
      if (!container) container = contentRef.current;
      if (!container) return;
      const scroller = container;

      const calcTarget = () => {
        const cRect = scroller.getBoundingClientRect();
        const tRect = target.getBoundingClientRect();
        // target's offset within the scroller's scrollable content
        const rel = tRect.top - cRect.top + scroller.scrollTop;
        const targetH = target.offsetHeight;
        const containerH = scroller.clientHeight;
        const dest =
          block === "center"
            ? rel - containerH / 2 + targetH / 2
            : rel - 16;
        return Math.max(0, Math.min(dest, scroller.scrollHeight - containerH));
      };

      // Animate with rAF (cannot be cancelled by browser unlike smooth scrollIntoView)
      const animate = (dest: number) => {
        const start = scroller.scrollTop;
        const dist = dest - start;
        if (Math.abs(dist) < 1) return;
        // Duration scales with distance so short hops still feel animated and
        // long jumps don't whip past — clamped to a comfortable 320–900ms.
        const duration = Math.min(900, Math.max(320, Math.abs(dist) * 0.45 + 200));
        const t0 = performance.now();
        const step = () => {
          const p = Math.min((performance.now() - t0) / duration, 1);
          // easeInOutCubic — gentle accelerate then settle, reads as a smooth glide.
          const ease = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
          scroller.scrollTop = start + dist * ease;
          if (p < 1) requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
      };

      // Execute: scroll now, then correction after images may have loaded
      animate(calcTarget());
      const correctionTimeout = setTimeout(() => {
        animate(calcTarget());
        onDone?.();
      }, 800);
      return () => clearTimeout(correctionTimeout);
    },
    []
  );

  // Ref to track cleanup for previous scroll operations
  const scrollCleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    // Cleanup previous scroll operation
    scrollCleanupRef.current?.();
    scrollCleanupRef.current = null;

    if (!contentRef.current || !markdown) return;
    if (!scrollToImageSrc && !scrollToHeading && !scrollToPage) return;

    // Double rAF: first waits for React commit, second waits for browser paint.
    // This ensures ReactMarkdown has fully rendered headings/page-dividers/images
    // before we calculate scroll positions — critical after tab switch (KG→Content).
    const rafId = requestAnimationFrame(() => requestAnimationFrame(() => {
      if (!contentRef.current) return;

      // Image citation — scroll to exact image element
      if (scrollToImageSrc) {
        const imgEl = contentRef.current.querySelector(
          `img[src="${CSS.escape(scrollToImageSrc)}"]`
        ) as HTMLElement | null;
        if (imgEl) {
          const figure = (imgEl.closest("figure") || imgEl) as HTMLElement;
          scrollCleanupRef.current = scrollTo(figure, "center", onScrolled) ?? null;
          // Flash highlight on figure
          figure.classList.add("ring-2", "ring-primary/50", "rounded-lg", "transition-all");
          setTimeout(() => {
            figure.classList.remove("ring-2", "ring-primary/50", "rounded-lg", "transition-all");
          }, 2500);
          return;
        }
        // Fallback: scroll to page
        if (scrollToPage) {
          const pageEl = contentRef.current.querySelector(
            `[data-page="${scrollToPage}"]`
          ) as HTMLElement | null;
          if (pageEl) {
            scrollCleanupRef.current = scrollTo(pageEl, "start", onScrolled) ?? null;
          }
        }
        return;
      }

      if (scrollToHeading) {
        const targetId = generateHeadingId(scrollToHeading);
        const el = contentRef.current.querySelector(
          `#${CSS.escape(targetId)}`
        ) as HTMLElement | null;
        if (el) {
          scrollCleanupRef.current = scrollTo(el, "center", onScrolled) ?? null;
          el.classList.add("bg-primary/20", "transition-colors");
          setTimeout(() => el.classList.remove("bg-primary/20"), 2000);
          return;
        }
      }

      if (scrollToPage && scrollToPage > 0) {
        const el = contentRef.current.querySelector(
          `[data-page="${scrollToPage}"]`
        ) as HTMLElement | null;
        if (el) {
          scrollCleanupRef.current = scrollTo(el, "start", onScrolled) ?? null;
        }
      }
    }));

    return () => cancelAnimationFrame(rafId);
  }, [scrollToPage, scrollToHeading, scrollToImageSrc, markdown, onScrolled, scrollTo]);

  // ---- Highlight chunks from citations ----
  // Depends on processedMarkdown so highlights re-apply after document switch
  // (markdown loads async → DOM not ready when highlightChunks first set)
  useEffect(() => {
    if (!contentRef.current) return;

    // Always clear previous highlights first
    contentRef.current.querySelectorAll(".chunk-hl").forEach((el) => {
      (el as HTMLElement).classList.remove(
        "chunk-hl",
        "chunk-hl-heading",
        "chunk-hl-sibling"
      );
    });

    if (!highlightChunks || highlightChunks.length === 0) return;

    for (const chunk of highlightChunks) {
      // Strategy 1: find the heading from heading_path, highlight it + siblings
      const lastHeading =
        chunk.heading_path.length > 0
          ? chunk.heading_path[chunk.heading_path.length - 1]
          : null;

      if (lastHeading) {
        const headingId = generateHeadingId(lastHeading);
        const headingEl = contentRef.current.querySelector(
          `#${CSS.escape(headingId)}`
        );
        if (headingEl) {
          // Highlight heading
          headingEl.classList.add(
            "chunk-hl",
            "chunk-hl-heading"
          );
          // Highlight siblings until next heading
          let sibling = headingEl.nextElementSibling;
          let count = 0;
          while (sibling && !sibling.tagName.match(/^H[1-4]$/) && count < 20) {
            sibling.classList.add(
              "chunk-hl",
              "chunk-hl-sibling"
            );
            sibling = sibling.nextElementSibling;
            count++;
          }
          continue;
        }
      }

      // Strategy 2: No heading_path - find text content matching chunk.content
      // and highlight its container element
      if (chunk.content) {
        // Strip markdown syntax characters before matching against DOM textContent
        const searchText = chunk.content
          .replace(/[#*`_~\[\]()]/g, "") // Remove common markdown chars
          .slice(0, 100)
          .trim();
          
        if (searchText.length > 5) {
          const allElements = contentRef.current.querySelectorAll("p, div, li, td, th");
          for (const el of allElements) {
            // Match against a slightly shorter segment to allow for minor parsing differences
            if (el.textContent && el.textContent.includes(searchText.slice(0, 40))) {
              el.classList.add("chunk-hl", "chunk-hl-sibling");
              break;
            }
          }
        }
      }
    }
    // Scroll is handled by the scroll-to effect above — don't compete here
  }, [highlightChunks, processedMarkdown]);

  // ---- TOC heading click ----
  // Use the rAF scrollTo (same as citations) — native scrollIntoView gets
  // cancelled by layout shifts inside the scroll container.
  const handleTocSelect = useCallback((id: string) => {
    const root = contentRef.current;
    if (!root) return;
    // Resolve the target heading. The TOC slug (from extractHeadings on the raw
    // markdown) and the rendered heading id (computed independently in the h1-h4
    // renderers) can disagree — links/setext/dedupe drift — leaving the click
    // dead. So fall back to POSITION: the headings array and the DOM headings
    // are both in document order, so the Nth TOC entry is the Nth heading.
    let el = root.querySelector(`#${CSS.escape(id)}`) as HTMLElement | null;
    if (!el) {
      const idx = headings.findIndex((h) => h.id === id);
      // Scope to the markdown <article> — contentRef also holds the document
      // title <h2>, which would otherwise offset the position lookup by one.
      const article = root.querySelector("article");
      if (idx >= 0 && article) {
        el = (article.querySelectorAll("h1, h2, h3, h4")[idx] as HTMLElement) ?? null;
      }
    }
    if (el) {
      scrollTo(el, "start");
      setActiveHeading(id);
      // Brief flash so the jump target is obvious
      el.classList.add("bg-primary/15", "transition-colors", "rounded");
      setTimeout(() => el!.classList.remove("bg-primary/15"), 1500);
    }
  }, [scrollTo, headings]);

  // Whether the TOC menu is actually on screen (full mode, toggled on, has
  // headings). Gates the sidebar render below.
  const tocVisible = showToc && effectiveMode === "full" && headings.length > 0;

  // ---- Loading / error / empty states ----
  if (doc.status !== "indexed") {
    return <ViewerEmpty />;
  }
  if (isLoading) return <ViewerSkeleton />;
  if (error) return <ViewerError message={(error as Error).message} />;
  if (!markdown || markdown.trim().length === 0) return <ViewerEmpty />;

  return (
    <div className="flex h-full min-h-0">
      {/* TOC sidebar — only in full mode (chunk content has partial headings, TOC is misleading) */}
      {tocVisible && (
        <TOCSidebar
          headings={headings}
          activeId={activeHeading}
          onSelect={handleTocSelect}
        />
      )}

      {/* Main markdown content.
          NOTE: no `scroll-smooth` here — programmatic scrolling goes through the
          rAF-based scrollTo() which animates scrollTop itself. CSS
          `scroll-behavior: smooth` re-animates every scrollTop write per frame,
          fighting the rAF loop so TOC/citation jumps stall. */}
      <div ref={contentRef} className="flex-1 min-h-0 overflow-y-auto">
        {/* TOC toggle (for smaller screens / when TOC hidden) */}
        {headings.length > 0 && (
          <button
            onClick={() => setShowToc(!showToc)}
            className={cn(
              "sticky top-2 left-2 z-10 p-1.5 rounded-md border bg-background/80 backdrop-blur-sm",
              "hover:bg-muted transition-colors xl:hidden",
              "flex items-center gap-1 text-xs text-muted-foreground"
            )}
          >
            <List className="w-3.5 h-3.5" />
            <ChevronRight className={cn("w-3 h-3 transition-transform", showToc && "rotate-90")} />
          </button>
        )}

        {/* Document fills the full viewer width — no centered max-width column —
            so the content expands to fill the panel (1:1) with no leftover
            whitespace beside the TOC menu in the split-screen viewer. */}
        <div className="w-full min-h-full bg-background">
          <div className="w-full min-h-full px-6 py-6 sm:px-12 sm:py-10 transition-all duration-300">
            {/* Document title header */}
            <div className="relative mb-8 pb-5 border-b border-border/60 pl-4">
              <span className="absolute left-0 top-1 bottom-5 w-1 rounded-full bg-gradient-to-b from-primary to-primary/30" />
              <div className="flex items-center justify-between gap-4">
                <div className="flex flex-col gap-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    {doc.document_type?.name && (
                      <span className="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded-full bg-primary/10 text-primary ring-1 ring-primary/20">
                        {doc.document_type.name}
                      </span>
                    )}
                    {doc.document_number && (
                      <span className="text-[10px] font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded-full">{doc.document_number}</span>
                    )}
                  </div>
                  <h2 className="text-2xl font-bold tracking-tight text-foreground mt-1 leading-snug">{doc.document_title || doc.original_filename}</h2>
                  <span className="text-xs text-muted-foreground/60 truncate">{doc.original_filename}</span>
                </div>
                {/* View toggles */}
                <div className="flex items-center gap-2 flex-shrink-0">
                  {/* OCR layout: rendered ⇄ plain text */}
                  {hasOcrLayout && effectiveMode === "full" && (
                    <button
                      onClick={() => setPlainMode((p) => !p)}
                      className={cn(
                        "flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold rounded-full transition-all flex-shrink-0",
                        plainMode
                          ? "bg-muted text-muted-foreground hover:bg-muted/80"
                          : "bg-primary/10 text-primary ring-1 ring-primary/20 hover:bg-primary/20"
                      )}
                      title={plainMode ? "Xem bản trình bày" : "Xem văn bản thuần"}
                    >
                      {plainMode ? (
                        <><LayoutTemplate className="w-3.5 h-3.5" />Bản trình bày</>
                      ) : (
                        <><AlignLeft className="w-3.5 h-3.5" />Văn bản thuần</>
                      )}
                    </button>
                  )}
                  {/* Chunk/Full mode toggle */}
                  {hasHighlight && hasChunkLocator && (
                    <button
                      onClick={() => setViewMode(effectiveMode === "chunk" ? "full" : "chunk")}
                      className={cn(
                        "flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold rounded-full transition-all flex-shrink-0",
                        effectiveMode === "chunk"
                          ? "bg-primary text-primary-foreground shadow-md hover:shadow-lg"
                          : "bg-muted text-muted-foreground hover:bg-muted/80"
                      )}
                      title={effectiveMode === "chunk" ? t("viewer.view_full") : t("viewer.view_chunk")}
                    >
                      {effectiveMode === "chunk" ? (
                        <><Maximize2 className="w-3.5 h-3.5" />{t("viewer.view_full")}</>
                      ) : (
                        <><Minimize2 className="w-3.5 h-3.5" />{t("viewer.view_chunk")}</>
                      )}
                    </button>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3 mt-3 text-xs text-muted-foreground pt-3">
                {effectiveMode === "chunk" && chunkContext ? (
                  <>
                    <span className="inline-flex items-center gap-1 text-primary font-semibold bg-primary/10 ring-1 ring-primary/20 px-2.5 py-1 rounded-full">
                      📍 Chunk {chunkContext.chunk_range[0]+1}–{chunkContext.chunk_range[1]+1} / {chunkContext.total_chunks}
                    </span>
                    {primaryChunk?.page_no && (
                      <span className="flex items-center gap-1">
                         <span className="w-1 h-1 rounded-full bg-muted-foreground/30" />
                         Trang {primaryChunk.page_no}
                      </span>
                    )}
                  </>
                ) : (
                  <>
                    {doc.page_count && doc.page_count > 0 && <span className="flex items-center gap-1">{doc.page_count} {t("files.metadata.pages")}</span>}
                    {doc.chunk_count > 0 && <span className="flex items-center gap-1"><span className="w-1 h-1 rounded-full bg-muted-foreground/30" />{doc.chunk_count} {t("files.metadata.chunks")}</span>}
                    {doc.parser_version && <span className="flex items-center gap-1"><span className="w-1 h-1 rounded-full bg-muted-foreground/30" />{t("viewer.parsed_by", { version: doc.parser_version })}</span>}
                  </>
                )}
              </div>
            </div>

            {/* Rendered markdown */}
            <article
              className={cn(
                "prose prose-sm max-w-none text-foreground/80",
                // Headings — explicit foreground for light/dark theme support
                "[&_h1]:text-2xl [&_h1]:font-bold [&_h1]:mt-8 [&_h1]:mb-4 [&_h1]:pb-2 [&_h1]:border-b [&_h1]:border-border/50 [&_h1]:scroll-mt-8 [&_h1]:text-foreground [&_h1]:tracking-tight",
                "[&_h2]:text-xl [&_h2]:font-bold [&_h2]:mt-7 [&_h2]:mb-3 [&_h2]:pb-1.5 [&_h2]:border-b [&_h2]:border-border/30 [&_h2]:scroll-mt-8 [&_h2]:text-foreground [&_h2]:tracking-tight",
                "[&_h3]:text-lg [&_h3]:font-semibold [&_h3]:mt-6 [&_h3]:mb-2 [&_h3]:scroll-mt-8 [&_h3]:text-foreground",
                "[&_h4]:text-base [&_h4]:font-semibold [&_h4]:mt-5 [&_h4]:mb-2 [&_h4]:scroll-mt-8 [&_h4]:text-foreground/90",
                // Body text
                "[&_p]:text-[15px] [&_p]:text-foreground/80 [&_p]:leading-[1.75] [&_p]:mb-4",
                "[&_li]:text-[15px] [&_li]:text-foreground/80 [&_li]:leading-[1.75] [&_li]:my-1",
                "[&_ul]:my-4 [&_ol]:my-4 [&_li]:marker:text-primary/50",
                "[&_strong]:text-foreground [&_strong]:font-bold",
                // Tables — rounded card with hover rows
                "[&_table]:w-full [&_table]:my-6 [&_table]:border-collapse [&_table]:text-[13px] [&_table]:rounded-lg [&_table]:overflow-hidden [&_table]:shadow-sm [&_table]:ring-1 [&_table]:ring-border",
                "[&_th]:bg-muted/60 [&_th]:border-b [&_th]:border-border [&_th]:px-3.5 [&_th]:py-2.5 [&_th]:text-left [&_th]:font-bold [&_th]:text-foreground [&_th]:text-[11px] [&_th]:uppercase [&_th]:tracking-wide",
                "[&_td]:border-t [&_td]:border-border/60 [&_td]:px-3.5 [&_td]:py-2 [&_td]:text-foreground/80",
                "[&_tbody_tr]:transition-colors [&_tbody_tr:hover]:bg-muted/25",
                // Code
                "[&_code]:bg-muted [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-[13px] [&_code]:text-primary [&_code]:font-mono",
                "[&_pre]:bg-muted/40 [&_pre]:rounded-xl [&_pre]:p-4 [&_pre]:my-6 [&_pre]:overflow-x-auto [&_pre]:text-[13px] [&_pre]:border [&_pre]:border-border/50",
                // Blockquotes
                "[&_blockquote]:border-l-4 [&_blockquote]:border-primary/40 [&_blockquote]:pl-4 [&_blockquote]:pr-3 [&_blockquote]:italic [&_blockquote]:text-foreground/70 [&_blockquote]:bg-primary/5 [&_blockquote]:py-2 [&_blockquote]:my-5 [&_blockquote]:rounded-r-lg",
                // Images
                "[&_img]:rounded-xl [&_img]:max-w-full [&_img]:my-6 [&_img]:shadow-md [&_img]:ring-1 [&_img]:ring-border/30",
                // Links
                "[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-4 [&_a]:decoration-primary/30 [&_a]:hover:decoration-primary [&_a]:transition-colors",
                // Horizontal rules
                "[&_hr]:border-border/40",
                // KaTeX math blocks
                "[&_.katex-display]:overflow-x-auto [&_.katex-display]:py-4",
                "[&_.katex]:text-[1.05em]",
                // ── OCR layout reconstruction (scanned administrative docs) ──
                // Backend rebuilds the original văn bản hành chính layout from
                // Unlimited-OCR bounding boxes into these classes.
                "[&_.ocr-note]:text-right [&_.ocr-note]:!text-[12px] [&_.ocr-note]:text-muted-foreground [&_.ocr-note]:!my-0 [&_.ocr-note]:leading-tight",
                "[&_.ocr-center]:text-center [&_.ocr-right]:text-right [&_.ocr-left]:text-left",
                "[&_.ocr-title]:font-bold [&_.ocr-title]:uppercase [&_.ocr-title]:tracking-wide [&_.ocr-title]:!text-[16px] [&_.ocr-title]:!my-3",
                // Two-column national heading (Nghị định 30): agency ‖ quốc hiệu
                "[&_.ocr-header-grid]:flex [&_.ocr-header-grid]:justify-between [&_.ocr-header-grid]:items-start [&_.ocr-header-grid]:gap-6 [&_.ocr-header-grid]:my-4",
                "[&_.ocr-header-grid>.ocr-col-left]:flex-1 [&_.ocr-header-grid>.ocr-col-left]:text-center [&_.ocr-header-grid>.ocr-col-left]:font-bold [&_.ocr-header-grid>.ocr-col-left]:uppercase [&_.ocr-header-grid>.ocr-col-left]:text-[13px]",
                "[&_.ocr-header-grid>.ocr-col-right]:flex-1 [&_.ocr-header-grid>.ocr-col-right]:text-center [&_.ocr-header-grid>.ocr-col-right]:font-bold",
                // Reference line: số hiệu (trái) ‖ địa danh, ngày tháng (phải)
                "[&_.ocr-row-2col]:flex [&_.ocr-row-2col]:justify-between [&_.ocr-row-2col]:gap-6 [&_.ocr-row-2col]:my-3",
                "[&_.ocr-row-2col>.ocr-col-left]:text-left [&_.ocr-row-2col>.ocr-col-right]:text-right [&_.ocr-row-2col>.ocr-col-right]:italic",
                // Signature / seal block placeholder
                "[&_.ocr-figure]:text-center [&_.ocr-figure]:!text-[12px] [&_.ocr-figure]:text-muted-foreground/60 [&_.ocr-figure]:italic [&_.ocr-figure]:my-4 [&_.ocr-figure]:border [&_.ocr-figure]:border-dashed [&_.ocr-figure]:border-border/40 [&_.ocr-figure]:rounded-lg [&_.ocr-figure]:py-3",
                // Real signature block (when signer captured)
                "[&_.ocr-sign-block]:ml-auto [&_.ocr-sign-block]:w-fit [&_.ocr-sign-block]:text-center [&_.ocr-sign-block]:my-5 [&_.ocr-sign-block]:mr-6",
                "[&_.ocr-sign-label]:italic [&_.ocr-sign-label]:text-[13px] [&_.ocr-sign-label]:text-muted-foreground [&_.ocr-sign-label]:mb-8",
                "[&_.ocr-sign-name]:font-bold [&_.ocr-sign-name]:text-foreground",
                // Block wrapper for tables / lists / figures (Docling layout path)
                "[&_.ocr-block]:my-3 [&_.ocr-block>figure]:mx-auto [&_.ocr-block>img]:mx-auto"
              )}
            >
              {plainMode && hasOcrLayout ? (
                <pre className="whitespace-pre-wrap break-words font-sans text-[14px] leading-[1.75] text-foreground/80 m-0">
                  {plainText}
                </pre>
              ) : (
                <ReactMarkdown
                  remarkPlugins={remarkPlugins}
                  rehypePlugins={rehypePlugins}
                  components={mdComponents}
                >
                  {processedMarkdown}
                </ReactMarkdown>
              )}
            </article>
          </div>
        </div>
      </div>
    </div>
  );
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function getHeadingText(children: React.ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(getHeadingText).join("");
  if (children && typeof children === "object" && "props" in children) {
    return getHeadingText((children as React.ReactElement<{ children?: React.ReactNode }>).props.children);
  }
  return String(children ?? "");
}

function generateHeadingId(text: string): string {
  return text
    // Transliterate Vietnamese → ASCII so headings produce meaningful, stable,
    // collision-free slugs (e.g. "CỘNG HÒA" → "cong-hoa", not "cng-ha"/empty).
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")   // strip combining diacritic marks
    .replace(/đ/g, "d").replace(/Đ/g, "d")  // đ / Đ
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

// Append an occurrence suffix to a base slug so duplicate heading texts get
// unique, deterministic ids in document order. Shared by the TOC extractor and
// the markdown renderer so both agree.
//   1st occurrence → "intro"      2nd → "intro-1"      3rd → "intro-2"
function dedupeHeadingId(base: string, counts: Map<string, number>): string {
  const n = (counts.get(base) ?? 0) + 1;
  counts.set(base, n);
  return n > 1 ? `${base}-${n - 1}` : base;
}
