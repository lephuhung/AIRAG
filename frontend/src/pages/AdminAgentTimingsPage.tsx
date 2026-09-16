import { useEffect, useMemo, useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { useTranslation } from "@/hooks/useTranslation";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  RefreshCw,
  Search,
  Activity,
  ChevronRight,
  AlertTriangle,
  Ban,
  CheckCircle2,
  Loader2,
  Timer,
} from "lucide-react";
import { Button } from "@/components/ui/button";

// ── API shapes (mirror GET /admin/agent/timings*) ────────────────────────────
interface TimingRun {
  run_id: string;
  thread_id: string;
  started_at: string | null;
  wall_ms: number | null;
  span_count: number;
  question: string | null;
  user_email: string | null;
  turn_idx: number;
}

interface TimingSpan {
  span_id: string;
  parent_span: string | null;
  kind: "turn" | "node" | "dispatch" | "stage";
  name: string;
  start_offset_ms: number;
  duration_ms: number;
  status: "ok" | "error" | "cancelled";
  meta: Record<string, unknown> | null;
}

interface TimingDetail {
  run_id: string;
  thread_id: string;
  wall_ms: number;
  question: string | null;
  user_email: string | null;
  spans: TimingSpan[];
}
// ── Presentation ─────────────────────────────────────────────────────────────

const KIND_STYLES: Record<TimingSpan["kind"], string> = {
  turn: "bg-slate-500/80",
  node: "bg-sky-500/80",
  dispatch: "bg-violet-500/80",
  stage: "bg-amber-500/80",
};

const STATUS_RING: Record<TimingSpan["status"], string> = {
  ok: "",
  error: "ring-2 ring-red-500 bg-red-500/80",
  cancelled: "ring-2 ring-amber-500 bg-amber-500/40",
};

function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${ms}ms`;
}

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const s = /Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z";
  return new Date(s).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

/** Depth of a span in the parent chain (turn=0, top-level node=1, …). */
function spanDepth(span: TimingSpan, byLabel: Map<string, TimingSpan>): number {
  let depth = 0;
  let cur = span.parent_span;
  const seen = new Set<string>();
  while (cur && !seen.has(cur)) {
    seen.add(cur);
    const parent = byLabel.get(cur);
    if (!parent) break;
    depth += 1;
    cur = parent.parent_span;
  }
  return depth;
}

function StatusIcon({ status }: { status: TimingSpan["status"] }) {
  if (status === "error")
    return <AlertTriangle className="w-3 h-3 text-red-500" />;
  if (status === "cancelled") return <Ban className="w-3 h-3 text-amber-500" />;
  return <CheckCircle2 className="w-3 h-3 text-emerald-500" />;
}

// ── Waterfall row ────────────────────────────────────────────────────────────

function WaterfallRow({
  span,
  wallMs,
  depth,
}: {
  span: TimingSpan;
  wallMs: number;
  depth: number;
}) {
  const left = wallMs > 0 ? (span.start_offset_ms / wallMs) * 100 : 0;
  const width =
    wallMs > 0 ? Math.max((span.duration_ms / wallMs) * 100, 0.4) : 0.4;
  const taskId =
    span.meta && typeof span.meta["task_id"] === "string"
      ? (span.meta["task_id"] as string)
      : null;

  return (
    <div className="group flex items-center gap-2 py-1 hover:bg-muted/40 rounded px-1">
      <div
        className="flex items-center gap-1.5 min-w-0 w-64 flex-shrink-0"
        style={{ paddingLeft: depth * 14 }}
      >
        <StatusIcon status={span.status} />
        <span
          className={cn(
            "truncate text-xs font-mono",
            span.kind === "turn" && "font-semibold",
            span.kind === "dispatch" && "text-violet-600 dark:text-violet-400",
            span.kind === "stage" && "text-amber-600 dark:text-amber-400",
          )}
          title={taskId ? `${span.name} · ${taskId}` : span.name}
        >
          {span.name}
        </span>
      </div>
      <div className="relative flex-1 h-4 bg-muted/30 rounded overflow-hidden">
        <div
          className={cn(
            "absolute top-0 h-full rounded-sm transition-all",
            KIND_STYLES[span.kind],
            STATUS_RING[span.status],
          )}
          style={{ left: `${left}%`, width: `${width}%` }}
          title={`${span.name}: ${formatMs(span.duration_ms)} @ +${span.start_offset_ms}ms`}
        />
      </div>
      <span className="w-16 flex-shrink-0 text-right text-xs tabular-nums text-muted-foreground">
        {formatMs(span.duration_ms)}
      </span>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export function AdminAgentTimingsPage() {
  const { t } = useTranslation();
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");

  // Debounce the search term so typing doesn't fire a request per keystroke.
  useEffect(() => {
    const id = setTimeout(() => setSearch(searchInput.trim()), 300);
    return () => clearTimeout(id);
  }, [searchInput]);

  const runsQuery = useQuery({
    queryKey: ["admin-agent-timings", search],
    queryFn: () =>
      api.get<{ runs: TimingRun[] }>(
        `/admin/agent/timings?limit=50${search ? `&q=${encodeURIComponent(search)}` : ""}`,
      ),
    refetchInterval: 15000,
    placeholderData: keepPreviousData,
  });

  const detailQuery = useQuery({
    queryKey: ["admin-agent-timing", selectedRun],
    queryFn: () => api.get<TimingDetail>(`/admin/agent/timings/${selectedRun}`),
    enabled: !!selectedRun,
    placeholderData: keepPreviousData,
  });

  const runs = runsQuery.data?.runs ?? [];
  const detail = detailQuery.data;

  const byLabel = useMemo(() => {
    const map = new Map<string, TimingSpan>();
    for (const s of detail?.spans ?? []) map.set(`${s.kind}:${s.name}`, s);
    return map;
  }, [detail]);

  return (
    <div className="flex flex-col h-full p-6 gap-4 overflow-hidden">
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-xl font-semibold flex items-center gap-2">
            <Timer className="w-5 h-5" />
            {t("admin.timings.title")}
          </h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {t("admin.timings.subtitle")}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => runsQuery.refetch()}
          disabled={runsQuery.isFetching}
        >
          {runsQuery.isFetching ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4" />
          )}
          <span className="ml-1.5">{t("admin.timings.refresh")}</span>
        </Button>
      </div>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Runs list */}
        <div className="w-96 flex-shrink-0 flex flex-col border border-border rounded-lg overflow-hidden">
          <div className="px-3 py-2 border-b border-border bg-muted/30 flex items-center gap-2">
            <Search className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
            <input
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder={t("admin.timings.search_placeholder")}
              className="flex-1 min-w-0 bg-transparent text-xs outline-none placeholder:text-muted-foreground"
            />
          </div>
          <div className="flex-1 overflow-y-auto">
            {runsQuery.isLoading && (
              <div className="flex items-center justify-center py-10 text-muted-foreground">
                <Loader2 className="w-5 h-5 animate-spin" />
              </div>
            )}
            {!runsQuery.isLoading && runs.length === 0 && (
              <div className="px-3 py-10 text-center text-sm text-muted-foreground">
                <Activity className="w-6 h-6 mx-auto mb-2 opacity-40" />
                {t("admin.timings.no_runs")}
              </div>
            )}
            {runs.map((run) => (
              <button
                key={run.run_id}
                onClick={() => setSelectedRun(run.run_id)}
                className={cn(
                  "w-full text-left px-3 py-2.5 border-b border-border/50 transition-colors",
                  selectedRun === run.run_id
                    ? "bg-primary/10"
                    : "hover:bg-muted/40",
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <span
                    className={cn(
                      "text-xs leading-snug line-clamp-2 break-words",
                      run.question
                        ? "text-foreground"
                        : "text-muted-foreground italic font-mono",
                    )}
                  >
                    {run.question || run.run_id}
                  </span>
                  <ChevronRight
                    className={cn(
                      "w-3.5 h-3.5 flex-shrink-0 mt-0.5 text-muted-foreground transition-transform",
                      selectedRun === run.run_id && "rotate-90 text-primary",
                    )}
                  />
                </div>
                <div className="flex items-center gap-2 mt-1.5 text-[11px] text-muted-foreground">
                  <span>{formatTime(run.started_at)}</span>
                  <span className="tabular-nums font-medium text-foreground/80">
                    {run.wall_ms != null ? formatMs(run.wall_ms) : "—"}
                  </span>
                  <span>
                    {run.span_count} {t("admin.timings.spans")}
                  </span>
                  {run.user_email && (
                    <span className="truncate ml-auto">{run.user_email}</span>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Waterfall */}
        <div className="flex-1 min-w-0 border border-border rounded-lg flex flex-col overflow-hidden">
          {!selectedRun && (
            <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
              {t("admin.timings.select_run")}
            </div>
          )}
          {selectedRun && detailQuery.isLoading && (
            <div className="flex-1 flex items-center justify-center">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          )}
          {selectedRun && detail && (
            <>
              <div className="px-4 py-2.5 border-b border-border bg-muted/30 text-xs">
                {detail.question && (
                  <p className="text-sm text-foreground mb-1.5 line-clamp-2">
                    {detail.question}
                  </p>
                )}
                <div className="flex items-center gap-4">
                  <span className="font-mono text-muted-foreground">
                    {detail.run_id}
                  </span>
                  {detail.user_email && (
                    <span className="text-muted-foreground">
                      {detail.user_email}
                    </span>
                  )}
                  <span className="text-muted-foreground">
                    {t("admin.timings.wall")}:{" "}
                    <span className="font-semibold text-foreground tabular-nums">
                      {formatMs(detail.wall_ms)}
                    </span>
                  </span>
                  <span className="text-muted-foreground">
                    {detail.spans.length} {t("admin.timings.spans")}
                  </span>
                  <div className="ml-auto flex items-center gap-3">
                    {(["node", "dispatch", "stage"] as const).map((k) => (
                      <span key={k} className="flex items-center gap-1">
                        <span
                          className={cn(
                            "w-2.5 h-2.5 rounded-sm",
                            KIND_STYLES[k],
                          )}
                        />
                        <span className="text-muted-foreground">
                          {t(`admin.timings.kind_${k}`)}
                        </span>
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-2">
                {detail.spans.map((span) => (
                  <WaterfallRow
                    key={span.span_id}
                    span={span}
                    wallMs={detail.wall_ms}
                    depth={spanDepth(span, byLabel)}
                  />
                ))}
              </div>
            </>
          )}
          {selectedRun && detailQuery.isError && (
            <div className="flex-1 flex items-center justify-center text-sm text-red-500">
              {(detailQuery.error as Error).message}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
