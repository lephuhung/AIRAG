import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  Activity,
  CheckCircle2,
  Clock,
  Cpu,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Trash2,
  FileText,
  Layers,
  XCircle,
  Inbox,
  Wifi,
  WifiOff,
  RotateCcw,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { formatRelativeDate, formatProcessingTime } from "@/lib/format";
import type {
  WorkerOverview,
  PipelineDocument,
} from "@/types";

// ---------------------------------------------------------------------------
// Pipeline status config (matches DocumentCard STATUS_CONFIG)
// ---------------------------------------------------------------------------
const PIPELINE_STATUS: Record<
  string,
  { label: string; color: string; bgColor: string; icon: typeof Clock }
> = {
  pending:     { label: "Pending",      color: "text-muted-foreground", bgColor: "bg-muted",                icon: Clock },
  parsing:     { label: "Parsing",      color: "text-blue-400",        bgColor: "bg-blue-400/15",          icon: Loader2 },
  ocring:      { label: "OCR",          color: "text-indigo-400",      bgColor: "bg-indigo-400/15",        icon: Loader2 },
  chunking:    { label: "Chunking",     color: "text-cyan-400",        bgColor: "bg-cyan-400/15",          icon: Loader2 },
  embedding:   { label: "Embedding",    color: "text-amber-400",       bgColor: "bg-amber-400/15",         icon: Loader2 },
  building_kg: { label: "Building KG",  color: "text-violet-400",      bgColor: "bg-violet-400/15",        icon: Loader2 },
  indexed:     { label: "Indexed",      color: "text-primary",         bgColor: "bg-primary/15",           icon: CheckCircle2 },
  failed:      { label: "Failed",       color: "text-destructive",     bgColor: "bg-destructive/15",       icon: XCircle },
};

const PROCESSING_KEYS = ["parsing", "ocring", "chunking", "embedding", "building_kg"] as const;

// ---------------------------------------------------------------------------
// WorkersPage
// ---------------------------------------------------------------------------
export function WorkersPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Data queries
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ["workers-overview"],
    queryFn: () => api.get<WorkerOverview>("/workers/overview"),
    refetchInterval: 5000,
  });

  const { data: pipelineData, isLoading: pipelineLoading } = useQuery({
    queryKey: ["workers-pipeline"],
    queryFn: () => api.get<{ documents: PipelineDocument[] }>("/workers/pipeline"),
    refetchInterval: 5000,
  });

  // Mutations
  const retryAll = useMutation({
    mutationFn: () => api.post<{ retried_count: number }>("/workers/retry-failed"),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["workers-overview"] });
      queryClient.invalidateQueries({ queryKey: ["workers-pipeline"] });
      toast.success(`Retrying ${(data as any)?.retried_count ?? 0} failed documents`);
    },
    onError: () => toast.error("Failed to retry documents"),
  });

  const retrySingle = useMutation({
    mutationFn: (docId: number) => api.post(`/workers/retry-failed/${docId}`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["workers-overview"] });
      queryClient.invalidateQueries({ queryKey: ["workers-pipeline"] });
      toast.success("Document queued for retry");
    },
    onError: () => toast.error("Failed to retry document"),
  });

  const purgeQueue = useMutation({
    mutationFn: (queueName: string) => api.post(`/workers/queues/${queueName}/purge`),
    onSuccess: (_, queueName) => {
      queryClient.invalidateQueries({ queryKey: ["workers-overview"] });
      toast.success(`Queue ${queueName} purged`);
    },
    onError: () => toast.error("Failed to purge queue"),
  });

  // UI state
  const [purgeConfirm, setPurgeConfirm] = useState<string | null>(null);
  const [retryAllConfirm, setRetryAllConfirm] = useState(false);

  // Computed
  const pipeline = overview?.pipeline_summary;
  const failedCount = pipeline?.failed ?? 0;

  const pipelineDocs = pipelineData?.documents ?? [];
  const failedDocs = useMemo(() => pipelineDocs.filter((d) => d.status === "failed"), [pipelineDocs]);
  const activeDocs = useMemo(
    () => pipelineDocs.filter((d) => d.status !== "failed" && d.status !== "indexed"),
    [pipelineDocs],
  );

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* ── Header ── */}
      <div className="flex-shrink-0 border-b px-6 py-4">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
          <button onClick={() => navigate("/")} className="hover:text-foreground transition-colors">
            Dashboard
          </button>
          <span>/</span>
          <span className="text-foreground font-medium">Workers</span>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              Worker Pipeline
            </h1>
            <p className="text-xs text-muted-foreground">
              RabbitMQ queues, processing pipeline, and worker status
            </p>
          </div>
          <div className="flex items-center gap-2">
            {failedCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 text-xs border-destructive/30 text-destructive hover:bg-destructive/10"
                onClick={() => setRetryAllConfirm(true)}
                disabled={retryAll.isPending}
              >
                {retryAll.isPending ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RotateCcw className="w-3.5 h-3.5" />
                )}
                Retry All Failed ({failedCount})
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* ── Content ── */}
      <div className="flex-1 overflow-y-auto px-6 py-5 space-y-6">
        {overviewLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="rounded-xl border bg-card animate-pulse p-4 h-28" />
            ))}
          </div>
        ) : (
          <>
            {/* ── Connection status ── */}
            <div className="flex items-center gap-2">
              {overview?.rabbitmq_connected ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full bg-green-500/10 text-green-400 border border-green-500/20">
                  <Wifi className="w-3 h-3" />
                  RabbitMQ Connected
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full bg-destructive/10 text-destructive border border-destructive/20">
                  <WifiOff className="w-3 h-3" />
                  RabbitMQ Disconnected
                </span>
              )}
            </div>

            {/* ── Pipeline Summary Cards ── */}
            <div>
              <h2 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
                <Layers className="w-4 h-4" />
                Pipeline Summary
              </h2>
              <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3">
                {pipeline &&
                  Object.entries(pipeline).map(([key, count]) => {
                    const config = PIPELINE_STATUS[key] ?? PIPELINE_STATUS.pending;
                    const Icon = config.icon;
                    const isAnimated = PROCESSING_KEYS.includes(key as any) && count > 0;
                    return (
                      <div
                        key={key}
                        className={cn(
                          "rounded-xl border bg-card p-3 flex flex-col items-center gap-1.5 transition-all",
                          count > 0 && key === "failed" && "border-destructive/30",
                          isAnimated && "border-blue-400/30",
                        )}
                      >
                        <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center", config.bgColor)}>
                          <Icon className={cn("w-4 h-4", config.color, isAnimated && "animate-spin")} />
                        </div>
                        <span className="text-lg font-bold tabular-nums">{count}</span>
                        <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">
                          {config.label}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* ── Queue Details ── */}
            {overview && overview.queues.length > 0 && (
              <div>
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
                  <Inbox className="w-4 h-4" />
                  Queue Details
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  {overview.queues.map((q) => (
                    <div key={q.name} className="rounded-xl border bg-card p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium truncate">{q.name}</span>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 flex-shrink-0"
                          title="Purge queue"
                          onClick={() => setPurgeConfirm(q.name)}
                        >
                          <Trash2 className="w-3 h-3 text-destructive" />
                        </Button>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-center">
                          <p className="text-lg font-bold tabular-nums">{q.messages_ready}</p>
                          <p className="text-[10px] text-muted-foreground">Ready</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold tabular-nums">{q.messages_unacked}</p>
                          <p className="text-[10px] text-muted-foreground">Processing</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold tabular-nums text-primary">{q.consumers}</p>
                          <p className="text-[10px] text-muted-foreground">Workers</p>
                        </div>
                      </div>
                      {(q.message_rate_in > 0 || q.message_rate_out > 0) && (
                        <div className="flex items-center justify-between text-[11px] text-muted-foreground pt-1 border-t border-border/50">
                          <span>In: {q.message_rate_in.toFixed(1)}/s</span>
                          <span>Out: {q.message_rate_out.toFixed(1)}/s</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Active Workers ── */}
            {overview && Object.keys(overview.active_workers).length > 0 && (
              <div>
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
                  <Cpu className="w-4 h-4" />
                  Active Workers
                </h2>
                <div className="flex items-center gap-3 flex-wrap">
                  {Object.entries(overview.active_workers).map(([type, count]) => (
                    <div
                      key={type}
                      className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border bg-card"
                    >
                      <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                      <span className="text-sm font-medium capitalize">{type}</span>
                      <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                        {count} consumer{count !== 1 ? "s" : ""}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── Active Documents (Processing) ── */}
            {!pipelineLoading && activeDocs.length > 0 && (
              <div>
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-1.5">
                  <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                  Processing ({activeDocs.length})
                </h2>
                <div className="rounded-xl border bg-card overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/30">
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">File</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Status</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Sub-tasks</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Time</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Updated</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeDocs.map((doc) => {
                        const config = PIPELINE_STATUS[doc.status] ?? PIPELINE_STATUS.pending;
                        const Icon = config.icon;
                        const isAnimated = PROCESSING_KEYS.includes(doc.status as any);
                        return (
                          <tr key={doc.id} className="border-b border-border/50 last:border-0">
                            <td className="px-4 py-2.5">
                              <div className="flex items-center gap-2">
                                <FileText className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
                                <span className="truncate max-w-[200px]" title={doc.filename}>
                                  {doc.filename}
                                </span>
                              </div>
                            </td>
                            <td className="px-4 py-2.5">
                              <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full", config.bgColor, config.color)}>
                                <Icon className={cn("w-3 h-3", isAnimated && "animate-spin")} />
                                {config.label}
                              </span>
                            </td>
                            <td className="px-4 py-2.5">
                              <div className="flex items-center gap-1">
                                <SubTaskPill done={doc.embed_done} label="E" />
                                <SubTaskPill done={doc.captions_done} label="C" />
                                <SubTaskPill done={doc.kg_done} label="K" />
                              </div>
                            </td>
                            <td className="px-4 py-2.5 text-xs text-muted-foreground tabular-nums">
                              {doc.processing_time_ms > 0
                                ? formatProcessingTime(doc.processing_time_ms)
                                : "—"}
                            </td>
                            <td className="px-4 py-2.5 text-xs text-muted-foreground">
                              {doc.updated_at ? formatRelativeDate(doc.updated_at) : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* ── Failed Documents ── */}
            {!pipelineLoading && failedDocs.length > 0 && (
              <div>
                <h2 className="text-sm font-semibold mb-3 flex items-center gap-1.5 text-destructive">
                  <AlertTriangle className="w-4 h-4" />
                  Failed ({failedDocs.length})
                </h2>
                <div className="rounded-xl border border-destructive/20 bg-card overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-destructive/5">
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">File</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Error</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Updated</th>
                        <th className="text-right px-4 py-2 text-xs font-medium text-muted-foreground">Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {failedDocs.map((doc) => (
                        <tr key={doc.id} className="border-b border-border/50 last:border-0">
                          <td className="px-4 py-2.5">
                            <div className="flex items-center gap-2">
                              <XCircle className="w-3.5 h-3.5 text-destructive flex-shrink-0" />
                              <span className="truncate max-w-[200px]" title={doc.filename}>
                                {doc.filename}
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-2.5">
                            <span className="text-xs text-destructive/80 truncate max-w-[300px] block" title={doc.error_message ?? ""}>
                              {doc.error_message || "Unknown error"}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 text-xs text-muted-foreground">
                            {doc.updated_at ? formatRelativeDate(doc.updated_at) : "—"}
                          </td>
                          <td className="px-4 py-2.5 text-right">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs gap-1"
                              onClick={() => retrySingle.mutate(doc.id)}
                              disabled={retrySingle.isPending}
                            >
                              <RefreshCw className={cn("w-3 h-3", retrySingle.isPending && "animate-spin")} />
                              Retry
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* ── Empty state ── */}
            {!pipelineLoading && activeDocs.length === 0 && failedDocs.length === 0 && (
              <div className="flex flex-col items-center justify-center py-16 text-center">
                <CheckCircle2 className="w-10 h-10 text-primary/30 mb-3" />
                <h3 className="text-sm font-medium text-muted-foreground mb-1">
                  All clear
                </h3>
                <p className="text-xs text-muted-foreground/70">
                  No documents currently processing or failed.
                </p>
              </div>
            )}
          </>
        )}
      </div>

      {/* Purge confirmation */}
      <ConfirmDialog
        open={purgeConfirm !== null}
        onConfirm={() => {
          if (purgeConfirm) {
            purgeQueue.mutate(purgeConfirm);
            setPurgeConfirm(null);
          }
        }}
        onCancel={() => setPurgeConfirm(null)}
        title="Purge Queue"
        message={`Are you sure you want to purge "${purgeConfirm}"? All pending messages will be permanently deleted.`}
        confirmLabel="Purge"
        variant="danger"
      />

      {/* Retry All confirmation */}
      <ConfirmDialog
        open={retryAllConfirm}
        onConfirm={() => {
          retryAll.mutate();
          setRetryAllConfirm(false);
        }}
        onCancel={() => setRetryAllConfirm(false)}
        title="Retry All Failed"
        message={`This will reset ${failedCount} failed document${failedCount !== 1 ? "s" : ""} and re-queue them for processing. Continue?`}
        confirmLabel="Retry All"
        variant="default"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// SubTaskPill — tiny indicator for embed/caption/kg status
// ---------------------------------------------------------------------------
function SubTaskPill({
  done,
  label,
}: {
  done: boolean;
  label: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[9px] font-medium border",
        done
          ? "bg-green-500/10 text-green-400 border-green-500/20"
          : "bg-muted/50 text-muted-foreground/50 border-border/50",
      )}
      title={`${label === "E" ? "Embed" : label === "C" ? "Captions" : "KG"}: ${done ? "Done" : "Pending"}`}
    >
      {done ? <CheckCircle2 className="w-2.5 h-2.5" /> : <Clock className="w-2.5 h-2.5" />}
      {label}
    </span>
  );
}
