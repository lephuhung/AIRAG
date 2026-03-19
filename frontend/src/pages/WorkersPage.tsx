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
  Play,
  Square,
  Heart,
  Skull,
  ChevronDown,
  ChevronRight,
  Zap,
  Minus,
  MailWarning,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { formatRelativeDate, formatProcessingTime } from "@/lib/format";
import type {
  WorkerOverview,
  WorkerHealthCheck,
  ManagedWorkerInfo,
  DeadLetterMessage,
  PipelineDocument,
} from "@/types";

// ---------------------------------------------------------------------------
// Pipeline status config
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

const WORKER_TYPES = ["parse", "embed", "caption", "kg"] as const;
type WorkerType = (typeof WORKER_TYPES)[number];

const WORKER_COLORS: Record<WorkerType, string> = {
  parse:   "text-blue-400",
  embed:   "text-purple-400",
  caption: "text-amber-400",
  kg:      "text-cyan-400",
};

// ---------------------------------------------------------------------------
// Helper: format uptime
// ---------------------------------------------------------------------------
function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

// ---------------------------------------------------------------------------
// Collapsible Section
// ---------------------------------------------------------------------------
function Section({
  title,
  icon: Icon,
  badge,
  badgeColor,
  defaultOpen = true,
  children,
}: {
  title: string;
  icon: typeof Activity;
  badge?: string | number;
  badgeColor?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full text-left group mb-3"
      >
        {open ? (
          <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
        )}
        <Icon className="w-4 h-4" />
        <span className="text-sm font-semibold">{title}</span>
        {badge !== undefined && (
          <span className={cn(
            "text-[10px] font-medium px-1.5 py-0.5 rounded-full",
            badgeColor || "bg-muted text-muted-foreground",
          )}>
            {badge}
          </span>
        )}
      </button>
      {open && children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// WorkersPage
// ---------------------------------------------------------------------------
export function WorkersPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // ── Queries ──
  const { data: overview, isLoading: overviewLoading } = useQuery({
    queryKey: ["workers-overview"],
    queryFn: () => api.get<WorkerOverview>("/workers/overview"),
    refetchInterval: 5000,
  });

  const { data: health } = useQuery({
    queryKey: ["workers-health"],
    queryFn: () => api.get<WorkerHealthCheck>("/workers/health"),
    refetchInterval: 10000,
  });

  const { data: managedData } = useQuery({
    queryKey: ["workers-managed"],
    queryFn: () => api.get<{ workers: Record<string, ManagedWorkerInfo[]> }>("/workers/managed"),
    refetchInterval: 5000,
  });

  const { data: pipelineData, isLoading: pipelineLoading } = useQuery({
    queryKey: ["workers-pipeline"],
    queryFn: () => api.get<{ documents: PipelineDocument[] }>("/workers/pipeline"),
    refetchInterval: 5000,
  });

  const { data: dlqData } = useQuery({
    queryKey: ["workers-dlq"],
    queryFn: () => api.get<{ queue: string; count: number; messages: DeadLetterMessage[] }>("/workers/dead-letter"),
    refetchInterval: 15000,
  });

  // ── Invalidation helper ──
  const invalidateAll = () => {
    queryClient.invalidateQueries({ queryKey: ["workers-overview"] });
    queryClient.invalidateQueries({ queryKey: ["workers-pipeline"] });
    queryClient.invalidateQueries({ queryKey: ["workers-health"] });
    queryClient.invalidateQueries({ queryKey: ["workers-managed"] });
    queryClient.invalidateQueries({ queryKey: ["workers-dlq"] });
  };

  // ── Worker management mutations ──
  const startWorker = useMutation({
    mutationFn: (params: { worker_type: string; count?: number }) =>
      api.post("/workers/start", params),
    onSuccess: (_, params) => {
      invalidateAll();
      toast.success(`Started ${params.count || 1} ${params.worker_type} worker(s)`);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const stopWorker = useMutation({
    mutationFn: (workerType: string) => api.post(`/workers/stop/${workerType}`),
    onSuccess: (_, wt) => {
      invalidateAll();
      toast.success(`Stopped ${wt} worker(s)`);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const restartWorker = useMutation({
    mutationFn: (workerType: string) => api.post(`/workers/restart/${workerType}`),
    onSuccess: (_, wt) => {
      invalidateAll();
      toast.success(`Restarted ${wt} worker(s)`);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  // ── Pipeline mutations ──
  const retryAll = useMutation({
    mutationFn: () => api.post<{ retried_count: number }>("/workers/retry-failed"),
    onSuccess: (data) => {
      invalidateAll();
      toast.success(`Retrying ${(data as any)?.retried_count ?? 0} failed documents`);
    },
    onError: () => toast.error("Failed to retry documents"),
  });

  const retrySingle = useMutation({
    mutationFn: (docId: number) => api.post(`/workers/retry-failed/${docId}`),
    onSuccess: () => {
      invalidateAll();
      toast.success("Document queued for retry");
    },
    onError: () => toast.error("Failed to retry document"),
  });

  const purgeQueue = useMutation({
    mutationFn: (queueName: string) => api.post(`/workers/queues/${queueName}/purge`),
    onSuccess: (_, queueName) => {
      invalidateAll();
      toast.success(`Queue ${queueName} purged`);
    },
    onError: () => toast.error("Failed to purge queue"),
  });

  const deleteQueue = useMutation({
    mutationFn: (queueName: string) => api.delete(`/workers/queues/${queueName}`),
    onSuccess: (_, queueName) => {
      invalidateAll();
      toast.success(`Queue ${queueName} deleted`);
    },
    onError: () => toast.error("Failed to delete queue"),
  });

  // ── DLQ mutations ──
  const purgeDlq = useMutation({
    mutationFn: () => api.post("/workers/dead-letter/purge"),
    onSuccess: () => {
      invalidateAll();
      toast.success("Dead letter queue purged");
    },
    onError: () => toast.error("Failed to purge DLQ"),
  });

  const retryDlq = useMutation({
    mutationFn: () => api.post<{ retried: number }>("/workers/dead-letter/retry"),
    onSuccess: (data) => {
      invalidateAll();
      toast.success(`Retried ${(data as any)?.retried ?? 0} messages from DLQ`);
    },
    onError: () => toast.error("Failed to retry DLQ messages"),
  });

  // ── UI state ──
  const [purgeConfirm, setPurgeConfirm] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [retryAllConfirm, setRetryAllConfirm] = useState(false);
  const [stopConfirm, setStopConfirm] = useState<string | null>(null);

  // ── Computed ──
  const pipeline = overview?.pipeline_summary;
  const failedCount = pipeline?.failed ?? 0;
  const pipelineDocs = pipelineData?.documents ?? [];
  const failedDocs = useMemo(() => pipelineDocs.filter((d) => d.status === "failed"), [pipelineDocs]);
  const activeDocs = useMemo(
    () => pipelineDocs.filter((d) => d.status !== "failed" && d.status !== "indexed"),
    [pipelineDocs],
  );
  const dlqCount = dlqData?.count ?? 0;
  const dlqMessages = dlqData?.messages ?? [];
  const managedWorkers = managedData?.workers ?? {};

  // Health status color
  const healthStatus = health?.status ?? "unknown";
  const healthColor =
    healthStatus === "healthy" ? "text-green-400 bg-green-500/10 border-green-500/20" :
    healthStatus === "degraded" ? "text-amber-400 bg-amber-500/10 border-amber-500/20" :
    "text-destructive bg-destructive/10 border-destructive/20";

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
              Manage workers, monitor queues, and inspect the processing pipeline
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Health badge */}
            {health && (
              <span className={cn(
                "inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border",
                healthColor,
              )}>
                <Heart className="w-3 h-3" />
                {healthStatus.charAt(0).toUpperCase() + healthStatus.slice(1)}
              </span>
            )}

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
            <div className="flex items-center gap-2 flex-wrap">
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
              {health?.checks?.rabbitmq?.version && (
                <span className="text-[10px] text-muted-foreground">
                  v{health.checks.rabbitmq.version}
                </span>
              )}
              {dlqCount > 0 && (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  <Skull className="w-3 h-3" />
                  {dlqCount} dead-letter msg{dlqCount !== 1 ? "s" : ""}
                </span>
              )}
            </div>

            {/* ── Worker Management ── */}
            <Section title="Worker Management" icon={Cpu} defaultOpen={true}>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {WORKER_TYPES.map((wtype) => {
                  const rmqConsumers = overview?.active_workers?.[wtype] ?? 0;
                  const managedCount = overview?.managed_workers?.[wtype] ?? 0;
                  const managedList = managedWorkers[wtype] ?? [];
                  const aliveList = managedList.filter((w) => w.alive);
                  const totalConsumers = rmqConsumers;
                  const isRunning = totalConsumers > 0 || aliveList.length > 0;

                  return (
                    <div
                      key={wtype}
                      className={cn(
                        "rounded-xl border bg-card p-4 space-y-3",
                        isRunning && "border-green-500/20",
                      )}
                    >
                      {/* Header */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className={cn(
                            "w-2.5 h-2.5 rounded-full",
                            isRunning ? "bg-green-400 animate-pulse" : "bg-muted-foreground/30",
                          )} />
                          <span className={cn("text-sm font-semibold capitalize", WORKER_COLORS[wtype])}>
                            {wtype}
                          </span>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {totalConsumers} consumer{totalConsumers !== 1 ? "s" : ""}
                        </span>
                      </div>

                      {/* Managed worker details */}
                      {aliveList.length > 0 && (
                        <div className="space-y-1">
                          {aliveList.map((w) => (
                            <div key={w.pid} className="flex items-center justify-between text-[11px] text-muted-foreground">
                              <span>PID {w.pid}</span>
                              <span>{formatUptime(w.uptime_seconds)}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {managedCount > 0 && aliveList.length === 0 && (
                        <p className="text-[11px] text-muted-foreground/50">
                          {managedCount} managed (external)
                        </p>
                      )}

                      {/* Actions */}
                      <div className="flex items-center gap-1.5 pt-1 border-t border-border/50">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs gap-1 flex-1"
                          onClick={() => startWorker.mutate({ worker_type: wtype, count: 1 })}
                          disabled={startWorker.isPending}
                        >
                          <Play className="w-3 h-3" />
                          Start
                        </Button>
                        {isRunning && (
                          <>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs gap-1 flex-1"
                              onClick={() => restartWorker.mutate(wtype)}
                              disabled={restartWorker.isPending}
                            >
                              <RefreshCw className={cn("w-3 h-3", restartWorker.isPending && "animate-spin")} />
                              Restart
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-xs gap-1 text-destructive hover:text-destructive"
                              onClick={() => setStopConfirm(wtype)}
                              disabled={stopWorker.isPending}
                            >
                              <Square className="w-3 h-3" />
                            </Button>
                          </>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Quick actions */}
              <div className="flex items-center gap-2 mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs gap-1.5"
                  onClick={() => {
                    WORKER_TYPES.forEach((wt) =>
                      startWorker.mutate({ worker_type: wt, count: 1 })
                    );
                  }}
                  disabled={startWorker.isPending}
                >
                  <Zap className="w-3 h-3" />
                  Start All Workers
                </Button>
              </div>
            </Section>

            {/* ── Pipeline Summary Cards ── */}
            <Section title="Pipeline Summary" icon={Layers}>
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
            </Section>

            {/* ── Queue Details ── */}
            {overview && overview.queues.length > 0 && (
              <Section title="Queue Details" icon={Inbox} badge={overview.queues.length}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  {overview.queues.map((q) => (
                    <div key={q.name} className="rounded-xl border bg-card p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="text-sm font-medium truncate">{q.name}</span>
                          {q.has_dlx && (
                            <span className="text-[9px] px-1 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20 flex-shrink-0">
                              DLX
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-0.5 flex-shrink-0">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            title="Purge queue"
                            onClick={() => setPurgeConfirm(q.name)}
                          >
                            <Trash2 className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            title="Delete queue"
                            onClick={() => setDeleteConfirm(q.name)}
                          >
                            <Minus className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                          </Button>
                        </div>
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
              </Section>
            )}

            {/* ── Dead Letter Queue ── */}
            {dlqCount > 0 && (
              <Section
                title="Dead Letter Queue"
                icon={MailWarning}
                badge={dlqCount}
                badgeColor="bg-amber-500/10 text-amber-400"
                defaultOpen={false}
              >
                <div className="rounded-xl border border-amber-500/20 bg-card overflow-hidden">
                  {/* DLQ actions */}
                  <div className="flex items-center gap-2 px-4 py-3 border-b border-border/50 bg-amber-500/5">
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs gap-1.5"
                      onClick={() => retryDlq.mutate()}
                      disabled={retryDlq.isPending}
                    >
                      {retryDlq.isPending ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <RotateCcw className="w-3 h-3" />
                      )}
                      Retry All
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs gap-1.5 text-destructive hover:text-destructive"
                      onClick={() => purgeDlq.mutate()}
                      disabled={purgeDlq.isPending}
                    >
                      <Trash2 className="w-3 h-3" />
                      Purge
                    </Button>
                    <span className="text-xs text-muted-foreground ml-auto">
                      Messages that failed after {3 + 1} attempts
                    </span>
                  </div>
                  {/* DLQ messages */}
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/30">
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Exchange</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Routing Key</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Retries</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">Payload</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dlqMessages.map((msg, i) => (
                        <tr key={i} className="border-b border-border/50 last:border-0">
                          <td className="px-4 py-2 text-xs font-mono">{msg.exchange || "—"}</td>
                          <td className="px-4 py-2 text-xs font-mono">{msg.routing_key || "—"}</td>
                          <td className="px-4 py-2 text-xs tabular-nums">
                            {(msg.headers?.["x-retry-count"] as number) ?? "?"}
                          </td>
                          <td className="px-4 py-2">
                            <span className="text-xs text-muted-foreground truncate max-w-[300px] block font-mono">
                              {msg.payload.slice(0, 120)}{msg.payload.length > 120 ? "…" : ""}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Section>
            )}

            {/* ── Active Documents (Processing) ── */}
            {!pipelineLoading && activeDocs.length > 0 && (
              <Section title={`Processing (${activeDocs.length})`} icon={Loader2}>
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
              </Section>
            )}

            {/* ── Failed Documents ── */}
            {!pipelineLoading && failedDocs.length > 0 && (
              <Section
                title={`Failed (${failedDocs.length})`}
                icon={AlertTriangle}
                badge={failedDocs.length}
                badgeColor="bg-destructive/10 text-destructive"
              >
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
              </Section>
            )}

            {/* ── Health Details (collapsed by default) ── */}
            {health && (
              <Section title="Health Details" icon={Heart} defaultOpen={false}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {/* RabbitMQ */}
                  <HealthCard
                    title="RabbitMQ"
                    status={health.checks.rabbitmq.status}
                    details={[
                      health.checks.rabbitmq.version ? `Version: ${health.checks.rabbitmq.version}` : null,
                      health.checks.rabbitmq.cluster ? `Cluster: ${health.checks.rabbitmq.cluster}` : null,
                      health.checks.rabbitmq.error ? `Error: ${health.checks.rabbitmq.error}` : null,
                    ].filter(Boolean) as string[]}
                  />

                  {/* Pipeline */}
                  <HealthCard
                    title="Pipeline"
                    status={health.checks.pipeline.status}
                    details={[
                      `In progress: ${health.checks.pipeline.documents_in_progress}`,
                      `Failed: ${health.checks.pipeline.documents_failed}`,
                    ]}
                  />

                  {/* Dead Letter Queue */}
                  <HealthCard
                    title="Dead Letter Queue"
                    status={health.checks.dead_letter_queue.status}
                    details={[
                      `Messages: ${health.checks.dead_letter_queue.messages}`,
                    ]}
                  />

                  {/* Queue health cards */}
                  {Object.entries(health.checks.queues).map(([qName, qInfo]) => (
                    <HealthCard
                      key={qName}
                      title={qName}
                      status={qInfo.status}
                      details={[
                        `Consumers: ${qInfo.consumers}`,
                        `Ready: ${qInfo.messages_ready}`,
                        qInfo.has_dlx ? "DLX: ✓" : "DLX: ✗",
                        ...qInfo.warnings.map((w) => `⚠ ${w}`),
                      ]}
                    />
                  ))}
                </div>
              </Section>
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

      {/* ── Confirm Dialogs ── */}
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

      <ConfirmDialog
        open={deleteConfirm !== null}
        onConfirm={() => {
          if (deleteConfirm) {
            deleteQueue.mutate(deleteConfirm);
            setDeleteConfirm(null);
          }
        }}
        onCancel={() => setDeleteConfirm(null)}
        title="Delete Queue"
        message={`Delete queue "${deleteConfirm}" entirely? It will be recreated with DLX support when a worker restarts.`}
        confirmLabel="Delete"
        variant="danger"
      />

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

      <ConfirmDialog
        open={stopConfirm !== null}
        onConfirm={() => {
          if (stopConfirm) {
            stopWorker.mutate(stopConfirm);
            setStopConfirm(null);
          }
        }}
        onCancel={() => setStopConfirm(null)}
        title="Stop Workers"
        message={`Stop all managed ${stopConfirm} workers? Any in-progress tasks will be interrupted.`}
        confirmLabel="Stop"
        variant="danger"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// HealthCard — compact card for health section
// ---------------------------------------------------------------------------
function HealthCard({
  title,
  status,
  details,
}: {
  title: string;
  status: string;
  details: string[];
}) {
  const statusColor =
    status === "healthy" ? "text-green-400" :
    status === "warning" ? "text-amber-400" :
    "text-destructive";

  const statusBg =
    status === "healthy" ? "bg-green-500/10" :
    status === "warning" ? "bg-amber-500/10" :
    "bg-destructive/10";

  return (
    <div className="rounded-xl border bg-card p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium truncate">{title}</span>
        <span className={cn("text-[10px] font-medium px-1.5 py-0.5 rounded-full capitalize", statusBg, statusColor)}>
          {status}
        </span>
      </div>
      <div className="space-y-0.5">
        {details.map((d, i) => (
          <p key={i} className="text-[11px] text-muted-foreground">{d}</p>
        ))}
      </div>
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
