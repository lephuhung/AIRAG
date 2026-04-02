import { useState, useMemo } from "react";
import { useTranslation } from "@/hooks/useTranslation";
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
  { labelKey: string; color: string; bgColor: string; icon: typeof Clock }
> = {
  pending:     { labelKey: "files.tabs.pending",      color: "text-muted-foreground", bgColor: "bg-muted",                icon: Clock },
  parsing:     { labelKey: "files.tabs.processing",   color: "text-blue-400",        bgColor: "bg-blue-400/15",          icon: Loader2 },
  ocring:      { labelKey: "files.tabs.processing",   color: "text-indigo-400",      bgColor: "bg-indigo-400/15",        icon: Loader2 },
  chunking:    { labelKey: "files.tabs.processing",   color: "text-cyan-400",        bgColor: "bg-cyan-400/15",          icon: Loader2 },
  embedding:   { labelKey: "files.tabs.processing",   color: "text-amber-400",       bgColor: "bg-amber-400/15",         icon: Loader2 },
  building_kg: { labelKey: "files.tabs.processing",   color: "text-violet-400",      bgColor: "bg-violet-400/15",        icon: Loader2 },
  indexed:     { labelKey: "files.tabs.indexed",      color: "text-primary",         bgColor: "bg-primary/15",           icon: CheckCircle2 },
  failed:      { labelKey: "files.tabs.failed",       color: "text-destructive",     bgColor: "bg-destructive/15",       icon: XCircle },
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
  const { t } = useTranslation();
  const formatUptime = (seconds: number): string => {
    if (seconds < 60) return t("workers.uptime.seconds", { count: Math.round(seconds) });
    if (seconds < 3600) {
      return t("workers.uptime.minutes_seconds", {
        m: Math.floor(seconds / 60),
        s: Math.round(seconds % 60),
      });
    }
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return t("workers.uptime.hours_minutes", { h, m });
  };
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
      toast.success(t("workers.start_success", { count: params.count || 1, type: params.worker_type }));
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const stopWorker = useMutation({
    mutationFn: (workerType: string) => api.post(`/workers/stop/${workerType}`),
    onSuccess: (_, wt) => {
      invalidateAll();
      toast.success(t("workers.stop_success", { type: wt }));
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const restartWorker = useMutation({
    mutationFn: (workerType: string) => api.post(`/workers/restart/${workerType}`),
    onSuccess: (_, wt) => {
      invalidateAll();
      toast.success(t("workers.restart_success", { type: wt }));
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const restartAllWorkers = useMutation({
    mutationFn: () => api.post("/workers/restart-all"),
    onSuccess: () => {
      invalidateAll();
      toast.success(t("workers.restart_all_success"));
    },
    onError: (err: Error) => toast.error(t("workers.restart_all_failed")),
  });

  // ── Pipeline mutations ──
  const retryAll = useMutation({
    mutationFn: () => api.post<{ retried_count: number }>("/workers/retry-failed"),
    onSuccess: (data) => {
      invalidateAll();
      toast.success(t("workers.retry_all_success", { count: (data as any)?.retried_count ?? 0 }));
    },
    onError: () => toast.error(t("workers.retry_all_failed")),
  });

  const retrySingle = useMutation({
    mutationFn: (docId: number) => api.post(`/workers/retry-failed/${docId}`),
    onSuccess: () => {
      invalidateAll();
      toast.success(t("workers.retry_single_success"));
    },
    onError: () => toast.error(t("workers.retry_single_failed")),
  });

  const purgeQueue = useMutation({
    mutationFn: (queueName: string) => api.post(`/workers/queues/${queueName}/purge`),
    onSuccess: (_, queueName) => {
      invalidateAll();
      toast.success(t("workers.queue_purged", { name: queueName }));
    },
    onError: () => toast.error(t("workers.queue_purge_failed")),
  });

  const deleteQueue = useMutation({
    mutationFn: (queueName: string) => api.delete(`/workers/queues/${queueName}`),
    onSuccess: (_, queueName) => {
      invalidateAll();
      toast.success(t("workers.queue_deleted", { name: queueName }));
    },
    onError: () => toast.error(t("workers.queue_delete_failed")),
  });

  // ── DLQ mutations ──
  const purgeDlq = useMutation({
    mutationFn: () => api.post("/workers/dead-letter/purge"),
    onSuccess: () => {
      invalidateAll();
      toast.success(t("workers.dlq_purged"));
    },
    onError: () => toast.error(t("workers.dlq_purge_failed")),
  });

  const retryDlq = useMutation({
    mutationFn: () => api.post<{ retried: number }>("/workers/dead-letter/retry"),
    onSuccess: (data) => {
      invalidateAll();
      toast.success(t("workers.dlq_retry_success", { count: (data as any)?.retried ?? 0 }));
    },
    onError: () => toast.error(t("workers.dlq_retry_failed")),
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

  const getHealthStatusLabel = (status: string) => {
    switch (status) {
      case "healthy": return t("workers.healthy");
      case "degraded": return t("workers.degraded");
      case "unhealthy": return t("workers.unhealthy");
      default: return t("common.unknown");
    }
  };

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* ── Header ── */}
      <div className="flex-shrink-0 border-b px-6 py-4">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-2">
          <button onClick={() => navigate("/")} className="hover:text-foreground transition-colors">
            {t("nav.dashboard")}
          </button>
          <span>/</span>
          <span className="text-foreground font-medium">{t("workers.title_short")}</span>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              {t("workers.title")}
            </h1>
            <p className="text-xs text-muted-foreground">
              {t("workers.desc")}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="default"
              size="sm"
              className="h-8 gap-1.5 text-xs shadow-sm shadow-primary/20"
              onClick={() => {
                WORKER_TYPES.forEach((wt) =>
                  startWorker.mutate({ worker_type: wt, count: 1 })
                );
              }}
              disabled={startWorker.isPending}
            >
              <Zap className={cn("w-3.5 h-3.5", startWorker.isPending && "animate-spin")} />
              {t("workers.start_all")}
            </Button>

            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5 text-xs border-primary/20 text-primary hover:bg-primary/5"
              onClick={() => restartAllWorkers.mutate()}
              disabled={restartAllWorkers.isPending}
            >
              <RefreshCw className={cn("w-3.5 h-3.5", restartAllWorkers.isPending && "animate-spin")} />
              {t("workers.restart_all")}
            </Button>

            <div className="h-6 w-px bg-border mx-1" />

            <div className="flex items-center gap-2">
              {/* Health badge */}
              {health && (
                <span className={cn(
                  "inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border",
                  healthColor,
                )}>
                  <Heart className="w-3 h-3" />
                  {getHealthStatusLabel(healthStatus)}
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
                  {t("workers.retry_all_with_count", { count: failedCount })}
                </Button>
              )}
            </div>
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
                  {t("workers.rabbitmq_connected")}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full bg-destructive/10 text-destructive border border-destructive/20">
                  <WifiOff className="w-3 h-3" />
                  {t("workers.rabbitmq_disconnected")}
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
                  {t(dlqCount === 1 ? 'workers.dlq_msg' : 'workers.dlq_msg_plural', { count: dlqCount })}
                </span>
              )}
            </div>

            {/* ── Worker Management ── */}
            <Section title={t("workers.management")} icon={Cpu} defaultOpen={true}>
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
                          {t("workers.consumers", { count: totalConsumers })}
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
                          {t("workers.managed_external", { count: managedCount })}
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
                          {t("common.start")}
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
                              {t("common.restart")}
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

              {/* DLQ status card — informational only, cannot be started */}
              <div className="mt-3 rounded-xl border bg-card p-4 flex items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className={cn(
                    "w-2.5 h-2.5 rounded-full flex-shrink-0",
                    dlqCount > 0 ? "bg-amber-400" : "bg-muted-foreground/30",
                  )} />
                  <div>
                    <span className="text-sm font-semibold text-muted-foreground">hrag.dead-letter</span>
                    <p className="text-[11px] text-muted-foreground/60 mt-0.5">
                      {t("workers.dlq_desc")}
                    </p>
                  </div>
                </div>
                {dlqCount > 0 ? (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 flex-shrink-0">
                    <Skull className="w-3 h-3" />
                    {t(dlqCount === 1 ? 'workers.dlq_msg' : 'workers.dlq_msg_plural', { count: dlqCount })}
                  </span>
                ) : (
                  <span className="text-xs text-muted-foreground/50 flex-shrink-0">{t("common.empty")}</span>
                )}
              </div>

            </Section>

            {/* ── Pipeline Summary Cards ── */}
            <Section title={t("workers.pipeline_summary")} icon={Layers}>
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
                          {t(config.labelKey)}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </Section>

            {/* ── Queue Details ── */}
            {overview && overview.queues.length > 0 && (() => {
              // Exclude the dead-letter queue — it's shown separately below
              const workerQueues = overview.queues.filter(
                (q) => q.name !== "hrag.dead-letter"
              );
              if (workerQueues.length === 0) return null;
              return (
              <Section title={t("workers.queue_details")} icon={Inbox} badge={workerQueues.length}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  {workerQueues.map((q) => (
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
                            title={t("workers.purge_queue")}
                            onClick={() => setPurgeConfirm(q.name)}
                          >
                            <Trash2 className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            title={t("workers.delete_queue")}
                            onClick={() => setDeleteConfirm(q.name)}
                          >
                            <Minus className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                          </Button>
                        </div>
                      </div>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="text-center">
                          <p className="text-lg font-bold tabular-nums">{q.messages_ready}</p>
                          <p className="text-[10px] text-muted-foreground">{t("workers.ready")}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold tabular-nums">{q.messages_unacked}</p>
                          <p className="text-[10px] text-muted-foreground">{t("workers.processing")}</p>
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold tabular-nums text-primary">{q.consumers}</p>
                          <p className="text-[10px] text-muted-foreground">{t("workers.consumers_label")}</p>
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
              );
            })()}

            {/* ── Dead Letter Queue ── */}
            {dlqCount > 0 && (
              <Section
                title={t("workers.dead_letter_queue")}
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
                      {t("common.retry_all")}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs gap-1.5 text-destructive hover:text-destructive"
                      onClick={() => purgeDlq.mutate()}
                      disabled={purgeDlq.isPending}
                    >
                      <Trash2 className="w-3 h-3" />
                      {t("common.purge")}
                    </Button>
                    <span className="text-xs text-muted-foreground ml-auto">
                      {t("workers.dlq_retry_msg")}
                    </span>
                  </div>
                  {/* DLQ messages */}
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/30">
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.exchange")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.routing_key")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.retries")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.payload")}</th>
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
              <Section title={t("workers.processing_with_count", { count: activeDocs.length })} icon={Loader2}>
                <div className="rounded-xl border bg-card overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/30">
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.file")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.status")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.sub_tasks")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.time")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.updated")}</th>
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
                                {t(config.labelKey)}
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
                title={t("workers.failed_with_count", { count: failedDocs.length })}
                icon={AlertTriangle}
                badge={failedDocs.length}
                badgeColor="bg-destructive/10 text-destructive"
              >
                <div className="rounded-xl border border-destructive/20 bg-card overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-destructive/5">
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.file")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.error")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.updated")}</th>
                        <th className="text-right px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.actions")}</th>
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
                              {t("common.retry")}
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
              <Section title={t("workers.health_details")} icon={Heart} defaultOpen={false}>
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
                      t("workers.health.in_progress", { count: health.checks.pipeline.documents_in_progress }),
                      t("workers.health.failed", { count: health.checks.pipeline.documents_failed }),
                    ]}
                  />

                  {/* Dead Letter Queue */}
                  <HealthCard
                    title="Dead Letter Queue"
                    status={health.checks.dead_letter_queue.status}
                    details={[
                      t("workers.health.messages", { count: health.checks.dead_letter_queue.messages }),
                    ]}
                  />

                  {/* Queue health cards */}
                  {Object.entries(health.checks.queues).map(([qName, qInfo]) => (
                    <HealthCard
                      key={qName}
                      title={qName}
                      status={qInfo.status}
                      details={[
                        t("workers.health.consumers", { count: qInfo.consumers }),
                        t("workers.health.ready", { count: qInfo.messages_ready }),
                        t("workers.health.dlx", { status: qInfo.has_dlx ? "✓" : "✗" }),
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
                  {t("workers.all_clear")}
                </h3>
                <p className="text-xs text-muted-foreground/70">
                  {t("workers.all_clear_desc")}
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
        title={t("workers.purge_queue_title")}
        message={t("workers.purge_queue_msg", { name: purgeConfirm })}
        confirmLabel={t("common.purge")}
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
        title={t("workers.delete_queue_title")}
        message={t("workers.delete_queue_msg", { name: deleteConfirm })}
        confirmLabel={t("common.delete")}
        variant="danger"
      />

      <ConfirmDialog
        open={retryAllConfirm}
        onConfirm={() => {
          retryAll.mutate();
          setRetryAllConfirm(false);
        }}
        onCancel={() => setRetryAllConfirm(false)}
        title={t("workers.retry_all_title")}
        message={t("workers.retry_all_msg", { count: failedCount })}
        confirmLabel={t("common.retry_all")}
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
        title={t("workers.stop_workers_title")}
        message={t("workers.stop_workers_msg", { type: stopConfirm })}
        confirmLabel={t("common.stop")}
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
