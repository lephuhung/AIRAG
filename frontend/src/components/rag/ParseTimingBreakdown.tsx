import { useMemo, useState } from "react";
import { useTranslation } from "@/hooks/useTranslation";
import { ChevronDown, ChevronUp, Timer } from "lucide-react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Parse timing breakdown — renders Document.parse_timing (JSONB written by the
// parse worker) as a per-stage waterfall so operators can see where a slow
// parse spent its time (Docling convert vs OCR render/infer vs post-steps).
// ---------------------------------------------------------------------------

type Timing = Record<string, unknown>;

interface StageDef {
  key: string;
  group: "download" | "detect" | "docling" | "ocr" | "legacy" | "persist" | "post";
}

// Display order mirrors the pipeline execution order. Stages absent from a
// document's timing dict are skipped (e.g. docling_* keys won't exist on an
// OCR-path document).
const STAGE_ORDER: StageDef[] = [
  { key: "minio_download_ms", group: "download" },
  { key: "digital_signature_ms", group: "download" },
  { key: "detect_scanned_ms", group: "detect" },
  { key: "detect_broken_vn_ms", group: "detect" },
  { key: "docling_convert_ms", group: "docling" },
  { key: "extract_images_ms", group: "docling" },
  { key: "extract_tables_ms", group: "docling" },
  { key: "export_markdown_ms", group: "docling" },
  { key: "export_chunk_markdown_ms", group: "docling" },
  { key: "chunking_ms", group: "docling" },
  { key: "ocr_render_ms", group: "ocr" },
  { key: "ocr_infer_ms", group: "ocr" },
  { key: "ocr_model_load_ms", group: "ocr" },
  { key: "ocr_prep_ms", group: "ocr" },
  { key: "ocr_generate_ms", group: "ocr" },
  { key: "ocr_chunk_ms", group: "ocr" },
  { key: "legacy_load_ms", group: "legacy" },
  { key: "legacy_chunk_ms", group: "legacy" },
  { key: "derive_headings_ms", group: "legacy" },
  { key: "upload_markdown_ms", group: "persist" },
  { key: "persist_structure_ms", group: "persist" },
  { key: "persist_images_tables_ms", group: "persist" },
  { key: "page1_ocr_ms", group: "post" },
  { key: "classify_llm_ms", group: "post" },
  { key: "validity_ms", group: "post" },
];

const GROUP_BAR: Record<StageDef["group"], string> = {
  download: "bg-sky-500/70",
  detect: "bg-slate-400/70",
  docling: "bg-emerald-500/70",
  ocr: "bg-amber-500/80",
  legacy: "bg-slate-400/70",
  persist: "bg-violet-500/70",
  post: "bg-rose-400/70",
};

const GROUP_DOT: Record<StageDef["group"], string> = {
  download: "bg-sky-500",
  detect: "bg-slate-400",
  docling: "bg-emerald-500",
  ocr: "bg-amber-500",
  legacy: "bg-slate-400",
  persist: "bg-violet-500",
  post: "bg-rose-400",
};

function formatMs(ms: number): string {
  if (ms >= 60_000) return `${(ms / 60_000).toFixed(1)}m`;
  if (ms >= 1_000) return `${(ms / 1_000).toFixed(1)}s`;
  return `${ms}ms`;
}

function formatBytes(n: number): string {
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(1)} MB`;
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(0)} KB`;
  return `${n} B`;
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ── Per-page OCR bars ────────────────────────────────────────────────────────

function OcrPageBars({ timing }: { timing: Timing }) {
  const { t } = useTranslation();
  const pageMs = Array.isArray(timing.ocr_page_ms)
    ? (timing.ocr_page_ms as unknown[]).map((v) => num(v) ?? 0)
    : null;
  if (!pageMs || pageMs.length === 0) return null;
  const waitMs = Array.isArray(timing.ocr_page_wait_ms)
    ? (timing.ocr_page_wait_ms as unknown[]).map((v) => num(v) ?? 0)
    : pageMs.map(() => 0);
  const maxPage = Math.max(...pageMs.map((p, i) => p + waitMs[i]), 1);

  return (
    <div className="mt-2 pt-2 border-t border-border/50">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
        {t("parse_timing.per_page", { count: pageMs.length })}
      </p>
      <div className="space-y-0.5">
        {pageMs.map((ms, i) => {
          const wait = waitMs[i] ?? 0;
          const total = ms + wait;
          return (
            <div key={i} className="flex items-center gap-2">
              <span className="w-8 flex-shrink-0 text-right text-[10px] tabular-nums text-muted-foreground">
                p{i + 1}
              </span>
              <div className="relative flex-1 h-2.5 bg-muted/30 rounded-sm overflow-hidden">
                {/* queue wait (semaphore) — muted segment */}
                <div
                  className="absolute top-0 h-full bg-slate-400/50"
                  style={{ width: `${(wait / maxPage) * 100}%` }}
                  title={`${t("parse_timing.wait")}: ${formatMs(wait)}`}
                />
                {/* request time — amber segment after the wait */}
                <div
                  className="absolute top-0 h-full bg-amber-500/80 rounded-r-sm"
                  style={{
                    left: `${(wait / maxPage) * 100}%`,
                    width: `${(ms / maxPage) * 100}%`,
                  }}
                  title={`${t("parse_timing.request")}: ${formatMs(ms)}`}
                />
              </div>
              <span className="w-12 flex-shrink-0 text-right text-[10px] tabular-nums text-muted-foreground">
                {formatMs(total)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-3 mt-1.5 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-slate-400/50" />
          {t("parse_timing.wait")}
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-sm bg-amber-500/80" />
          {t("parse_timing.request")}
        </span>
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

export function ParseTimingBreakdown({ timing }: { timing: Timing }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

  const rows = useMemo(() => {
    const out: Array<{ def: StageDef; ms: number }> = [];
    for (const def of STAGE_ORDER) {
      const ms = num(timing[def.key]);
      if (ms != null) out.push({ def, ms });
    }
    return out;
  }, [timing]);

  const totalMs = num(timing.total_ms) ?? rows.reduce((s, r) => s + r.ms, 0);
  const maxStage = Math.max(...rows.map((r) => r.ms), 1);
  const pages = num(timing.ocr_pages);
  const backend = typeof timing.ocr_backend === "string" ? timing.ocr_backend : null;
  const fileBytes = num(timing.file_bytes);

  if (rows.length === 0) return null;

  return (
    <div className="mt-3 rounded-lg border border-border/60 bg-muted/20 overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs hover:bg-muted/40 transition-colors"
      >
        <Timer className="w-3.5 h-3.5 text-primary/70" />
        <span className="font-medium">{t("parse_timing.title")}</span>
        <span className="tabular-nums text-muted-foreground">
          {formatMs(totalMs)}
        </span>
        {backend && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-600 dark:text-amber-400 font-medium">
            OCR/{backend}
          </span>
        )}
        {pages != null && (
          <span className="text-[10px] text-muted-foreground">
            {pages} {t("parse_timing.pages")}
          </span>
        )}
        {fileBytes != null && (
          <span className="text-[10px] text-muted-foreground">
            {formatBytes(fileBytes)}
          </span>
        )}
        <span className="ml-auto">
          {open ? (
            <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
          )}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 pt-1">
          <div className="space-y-0.5">
            {rows.map(({ def, ms }) => (
              <div key={def.key} className="flex items-center gap-2">
                <span className="w-44 flex-shrink-0 flex items-center gap-1.5 text-[11px] font-mono text-foreground/80 truncate">
                  <span
                    className={cn("w-2 h-2 rounded-sm flex-shrink-0", GROUP_DOT[def.group])}
                  />
                  {t(`parse_timing.stage.${def.key}`) !== `parse_timing.stage.${def.key}`
                    ? t(`parse_timing.stage.${def.key}`)
                    : def.key.replace(/_ms$/, "")}
                </span>
                <div className="relative flex-1 h-3 bg-muted/30 rounded-sm overflow-hidden">
                  <div
                    className={cn("absolute top-0 h-full rounded-sm", GROUP_BAR[def.group])}
                    style={{ width: `${Math.max((ms / maxStage) * 100, 0.5)}%` }}
                  />
                </div>
                <span className="w-14 flex-shrink-0 text-right text-[11px] tabular-nums text-muted-foreground">
                  {formatMs(ms)}
                </span>
                <span className="w-10 flex-shrink-0 text-right text-[10px] tabular-nums text-muted-foreground/60">
                  {totalMs > 0 ? `${((ms / totalMs) * 100).toFixed(0)}%` : "—"}
                </span>
              </div>
            ))}
          </div>
          <OcrPageBars timing={timing} />
        </div>
      )}
    </div>
  );
}
