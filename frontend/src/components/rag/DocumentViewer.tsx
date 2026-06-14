import { useState, useEffect, useRef, useMemo, useCallback, memo } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "@/hooks/useTranslation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";
import { FileText, List, ChevronRight, Maximize2, Minimize2 } from "lucide-react";
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

  // ---- Process markdown (insert page dividers) ----
  const processedMarkdown = useMemo(() => {
    return markdown ? insertPageDividers(markdown) : "";
  }, [markdown]);

  // Reset the heading-id counter before the markdown subtree renders this pass.
  // The h1–h4 components (memoized below) consume it in document order, so ids
  // stay deterministic and identical across renders for the same content.
  headingIdCounter.current = new Map();

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
  }), []);

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
      const container = contentRef.current;
      if (!container) return;

      const calcTarget = () => {
        // Calculate offset of target relative to scroll container
        let offset = 0;
        let el: HTMLElement | null = target;
        while (el && el !== container) {
          offset += el.offsetTop;
          el = el.offsetParent as HTMLElement | null;
        }
        const targetH = target.offsetHeight;
        const containerH = container.clientHeight;
        let dest =
          block === "center"
            ? offset - containerH / 2 + targetH / 2
            : offset;
        return Math.max(0, Math.min(dest, container.scrollHeight - containerH));
      };

      // Animate with rAF (cannot be cancelled by browser unlike smooth scrollIntoView)
      const animate = (dest: number) => {
        const start = container.scrollTop;
        const dist = dest - start;
        if (Math.abs(dist) < 1) return;
        const duration = Math.min(400, Math.abs(dist) * 0.5 + 150);
        const t0 = performance.now();
        const step = () => {
          const p = Math.min((performance.now() - t0) / duration, 1);
          const ease = 1 - Math.pow(1 - p, 3); // easeOutCubic
          container.scrollTop = start + dist * ease;
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
  const handleTocSelect = useCallback((id: string) => {
    if (!contentRef.current) return;
    const el = contentRef.current.querySelector(`#${CSS.escape(id)}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      setActiveHeading(id);
    }
  }, []);

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
      {showToc && effectiveMode === "full" && (
        <TOCSidebar
          headings={headings}
          activeId={activeHeading}
          onSelect={handleTocSelect}
        />
      )}

      {/* Main markdown content */}
      <div ref={contentRef} className="flex-1 min-h-0 overflow-y-auto scroll-smooth">
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

        {/* Centered A4-like container for better readability on wide screens */}
        <div className="w-full h-full bg-gradient-to-b from-muted/20 to-muted/5">
          <div className="mx-auto max-w-[850px] min-h-full bg-background px-6 py-6 sm:px-14 sm:py-12 sm:my-8 sm:shadow-xl sm:rounded-lg sm:ring-1 sm:ring-border/40 transition-all duration-300">
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
                "[&_.katex]:text-[1.05em]"
              )}
            >
              <ReactMarkdown
                remarkPlugins={remarkPlugins}
                rehypePlugins={rehypePlugins}
                components={mdComponents}
              >
                {processedMarkdown}
              </ReactMarkdown>
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
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
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
