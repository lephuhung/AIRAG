import { useState, useMemo } from "react";
import { useTranslation } from "@/hooks/useTranslation";
import { useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useCancelPipeline } from "@/hooks/useWorkers";
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
  Gauge,
  Thermometer,
  MemoryStick,
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
  QueueInfo,
  GpuOverview,
  GpuInfo,
} from "@/types";

// ---------------------------------------------------------------------------
// Pipeline status config
// ---------------------------------------------------------------------------
const PIPELINE_STATUS: Record<
  string,
  { labelKey: string; color: string; bgColor: string; icon: typeof Clock }
> = {
  pending:     { labelKey: "workers.status.pending",     color: "text-muted-foreground", bgColor: "bg-muted",                icon: Clock },
  parsing:     { labelKey: "workers.status.parsing",     color: "text-blue-400",        bgColor: "bg-blue-400/15",          icon: Loader2 },
  ocring:      { labelKey: "workers.status.ocring",      color: "text-indigo-400",      bgColor: "bg-indigo-400/15",        icon: Loader2 },
  chunking:    { labelKey: "workers.status.chunking",    color: "text-cyan-400",        bgColor: "bg-cyan-400/15",          icon: Loader2 },
  embedding:   { labelKey: "workers.status.embedding",   color: "text-amber-400",       bgColor: "bg-amber-400/15",         icon: Loader2 },
  building_kg: { labelKey: "workers.status.building_kg", color: "text-violet-400",      bgColor: "bg-violet-400/15",        icon: Loader2 },
  indexed:     { labelKey: "workers.status.indexed",     color: "text-primary",         bgColor: "bg-primary/15",           icon: CheckCircle2 },
  failed:      { labelKey: "workers.status.failed",      color: "text-destructive",     bgColor: "bg-destructive/15",       icon: XCircle },
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
// Queue grouping — turn raw RabbitMQ queue names into something readable.
// Worker queues map to one of the 4 worker types (KG is split per-workspace,
// so all `hrag.kg.*` collapse into a single logical "kg" group). Everything
// else under `hrag.*` (retry / memory) is "system" infra; the DLQ is hidden
// here because it has its own dedicated section.
// ---------------------------------------------------------------------------
function queueGroupOf(name: string): WorkerType | "system" | null {
  if (name === "hrag.dead-letter") return null;
  if (name === "hrag.parse") return "parse";
  if (name === "hrag.embed") return "embed";
  if (name === "hrag.caption") return "caption";
  if (name === "hrag.kg" || name.startsWith("hrag.kg.")) return "kg";
  return "system";
}

type QueueAgg = {
  messages_ready: number;
  messages_unacked: number;
  consumers: number;
  message_rate_in: number;
  message_rate_out: number;
  has_dlx: boolean;
};

function aggregateQueues(queues: QueueInfo[]): QueueAgg {
  return queues.reduce<QueueAgg>(
    (a, q) => ({
      messages_ready: a.messages_ready + q.messages_ready,
      messages_unacked: a.messages_unacked + q.messages_unacked,
      consumers: a.consumers + q.consumers,
      message_rate_in: a.message_rate_in + q.message_rate_in,
      message_rate_out: a.message_rate_out + q.message_rate_out,
      has_dlx: a.has_dlx || !!q.has_dlx,
    }),
    { messages_ready: 0, messages_unacked: 0, consumers: 0, message_rate_in: 0, message_rate_out: 0, has_dlx: false },
  );
}

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

  const { data: gpuData } = useQuery({
    queryKey: ["workers-gpu"],
    queryFn: () => api.get<GpuOverview>("/workers/gpu"),
    refetchInterval: 3000,
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
      toast.success(t("workers.start_success", { 
        count: params.count || 1, 
        type: t(`workers.types.${params.worker_type}`) || params.worker_type 
      }));
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const stopWorker = useMutation({
    mutationFn: (workerType: string) => api.post(`/workers/stop/${workerType}`),
    onSuccess: (_, wt) => {
      invalidateAll();
      toast.success(t("workers.stop_success", { type: t(`workers.types.${wt}`) || wt }));
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const restartWorker = useMutation({
    mutationFn: (workerType: string) => api.post(`/workers/restart/${workerType}`),
    onSuccess: (_, wt) => {
      invalidateAll();
      toast.success(t("workers.restart_success", { type: t(`workers.types.${wt}`) || wt }));
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const restartAllWorkers = useMutation({
    mutationFn: () => api.post("/workers/restart-all"),
    onSuccess: () => {
      invalidateAll();
      toast.success(t("workers.restart_all_success"));
    },
    onError: () => toast.error(t("workers.restart_all_failed")),
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

  const clearStuck = useMutation({
    mutationFn: () => api.post<{ retried_count: number }>("/workers/retry-stuck"),
    onSuccess: (data) => {
      invalidateAll();
      toast.success(t("workers.clear_stuck_success", { count: (data as any)?.retried_count ?? 0 }));
    },
    onError: () => toast.error(t("workers.clear_stuck_failed")),
  });

  const retrySingle = useMutation({
    mutationFn: (docId: string) => api.post(`/workers/retry-failed/${docId}`),
    onSuccess: () => {
      invalidateAll();
      toast.success(t("workers.retry_single_success"));
    },
    onError: () => toast.error(t("workers.retry_single_failed")),
  });

  const deleteFailedDoc = useMutation({
    mutationFn: (docId: string) => api.delete(`/documents/${docId}`),
    onSuccess: () => {
      invalidateAll();
      toast.success(t("workers.delete_doc_success"));
    },
    onError: () => toast.error(t("workers.delete_doc_failed")),
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

  const cancelDoc = useCancelPipeline({
    onSuccess: () => {
      invalidateAll();
      toast.success(t("workers.cancel_success"));
    },
    onError: () => toast.error(t("workers.cancel_failed")),
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
  const [cancelConfirmDoc, setCancelConfirmDoc] = useState<string | null>(null);
  const [deleteFailedDocId, setDeleteFailedDocId] = useState<string | null>(null);

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

  // Group raw queues by worker type; collect retry/memory infra separately
  const { workerQueueGroups, systemQueues } = useMemo(() => {
    const groups: Record<WorkerType, QueueInfo[]> = { parse: [], embed: [], caption: [], kg: [] };
    const system: QueueInfo[] = [];
    for (const q of overview?.queues ?? []) {
      const g = queueGroupOf(q.name);
      if (g === null) continue;
      if (g === "system") system.push(q);
      else groups[g].push(q);
    }
    return { workerQueueGroups: groups, systemQueues: system };
  }, [overview?.queues]);

  // Health status color
  const healthStatus = health?.status ?? "unknown";
  const healthColor =
    healthStatus === "healthy" ? "text-green-400 bg-green-500/10 border-green-500/20" :
    healthStatus === "degraded" ? "text-amber-400 bg-amber-500/10 border-amber-500/20" :
    "text-destructive bg-destructive/10 border-destructive/20";

  const getHealthStatusLabel = (status: string) => {
    return t(`workers.health_status.${status}`) || status;
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
            {/* Control-plane commands — work for container workers via RabbitMQ */}
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
              {t("workers.resume_all")}
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

              {pipeline?.ocring != null && pipeline?.ocring > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1.5 text-xs border-indigo-400/30 text-indigo-400 hover:bg-indigo-400/10"
                  onClick={() => clearStuck.mutate()}
                  disabled={clearStuck.isPending}
                  title={t("workers.clear_stuck_desc")}
                >
                  {clearStuck.isPending ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Layers className="w-3.5 h-3.5" />
                  )}
                  {t("workers.clear_stuck")}
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
                  {t("workers.labels.version")}: {health.checks.rabbitmq.version}
                </span>
              )}
              {dlqCount > 0 && (
                <button
                  onClick={() => {
                    const el = document.getElementById('dlq-section');
                    if (el) el.scrollIntoView({ behavior: 'smooth' });
                  }}
                  title={t("workers.dlq_desc")}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/20 transition-colors"
                >
                  <Skull className="w-3 h-3" />
                  {t(dlqCount === 1 ? 'workers.dlq_msg' : 'workers.dlq_msg_plural', { count: dlqCount })}
                </button>
              )}
            </div>

            {/* ── Worker Management ── */}
            <Section title={t("workers.management")} icon={Cpu} defaultOpen={true}>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {WORKER_TYPES.map((wtype) => {
                  const rmqConsumers = overview?.active_workers?.[wtype] ?? 0;
                  const containerCount = overview?.container_workers?.[wtype] ?? 0;
                  const managedList = managedWorkers[wtype] ?? [];
                  // Live container replicas (healthy or deliberately paused)
                  const aliveList = managedList.filter((w) => w.status === "healthy" || w.status === "paused");
                  // Prefer container count for Docker workers; fall back to consumer count
                  const workerCount = containerCount > 0 ? containerCount : rmqConsumers;
                  const isRunning = workerCount > 0 || aliveList.length > 0;
                  const isPaused = aliveList.length > 0 && aliveList.every((w) => w.paused);

                  return (
                    <div
                      key={wtype}
                      className={cn(
                        "rounded-xl border bg-card p-4 space-y-3 flex flex-col justify-between",
                        isRunning && "border-green-500/20",
                      )}
                    >
                      <div>
                        {/* Header */}
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <div className={cn(
                              "w-2.5 h-2.5 rounded-full flex-shrink-0",
                              isPaused ? "bg-amber-400" :
                              isRunning ? "bg-green-400 animate-pulse" : "bg-muted-foreground/30",
                            )} />
                            <span className={cn("text-sm font-semibold truncate", WORKER_COLORS[wtype])}>
                              {t(`workers.types.${wtype}`) || wtype}
                            </span>
                            {isPaused && (
                              <span className="text-[10px] font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-full px-1.5 py-0.5">
                                {t("workers.paused")}
                              </span>
                            )}
                          </div>
                          <span className="text-xs font-bold tabular-nums text-muted-foreground ml-2">
                            {workerCount}
                          </span>
                        </div>

                        {/* Container replica details */}
                        {aliveList.length > 0 && (
                          <div className="space-y-1 mb-2">
                            {aliveList.slice(0, 2).map((w) => (
                              <div key={w.container} className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                                <span className={cn("truncate", w.paused && "text-amber-400/80")} title={w.container}>
                                  {w.container}
                                </span>
                                <span className="flex-shrink-0">{formatUptime(w.uptime_seconds ?? 0)}</span>
                              </div>
                            ))}
                            {aliveList.length > 2 && (
                              <div className="text-[10px] text-muted-foreground/50 text-center">
                                +{aliveList.length - 2} {t("common.show_more")}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Control-plane commands (pause / resume / restart via RabbitMQ) */}
                      <div className="flex items-center gap-1.5 pt-2 border-t border-border/50">
                        {isPaused ? (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-[11px] gap-1 flex-1 px-1 text-green-500 hover:text-green-400"
                            onClick={() => startWorker.mutate({ worker_type: wtype, count: 1 })}
                            disabled={startWorker.isPending}
                          >
                            <Play className="w-3 h-3" />
                            {t("workers.resume")}
                          </Button>
                        ) : (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 text-[11px] gap-1 flex-1 px-1"
                            onClick={() => setStopConfirm(wtype)}
                            disabled={stopWorker.isPending || !isRunning}
                          >
                            <Square className="w-3 h-3" />
                            {t("workers.pause")}
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 text-[11px] gap-1 flex-1 px-1"
                          onClick={() => restartWorker.mutate(wtype)}
                          disabled={restartWorker.isPending || !isRunning}
                        >
                          <RefreshCw className={cn("w-3 h-3", restartWorker.isPending && "animate-spin")} />
                          {t("common.restart")}
                        </Button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </Section>


            {/* ── LLM Services ── */}
            {health && (
              <Section title={t("workers.llm_service")} icon={Cpu} defaultOpen={true}>
                <div className="rounded-xl border bg-card overflow-hidden">
                  <div className="flex flex-col md:flex-row md:divide-x divide-border/50">
                    {/* OCR Service */}
                    {health.checks.llm_services?.ocr && (
                      <div className="flex-1 p-3 flex items-center justify-between gap-3 min-w-0">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <div className={cn(
                            "w-2 h-2 rounded-full flex-shrink-0",
                            health.checks.llm_services.ocr.status === "healthy" ? "bg-green-400" :
                            health.checks.llm_services.ocr.status === "warning" ? "bg-amber-400" : "bg-destructive"
                          )} title={health.checks.llm_services.ocr.status} />
                          <div className="flex flex-col min-w-0">
                            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none mb-1">
                              {t("workers.health.ocr_service")}
                            </span>
                            <span className="text-xs font-semibold truncate" title={health.checks.llm_services.ocr.model}>
                              {health.checks.llm_services.ocr.model || "—"}
                            </span>
                          </div>
                        </div>
                        {health.checks.llm_services.ocr.error && (
                          <span className="text-[10px] text-destructive font-medium truncate max-w-[100px]" title={health.checks.llm_services.ocr.error}>
                            {health.checks.llm_services.ocr.error}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Qwen Memory */}
                    {health.checks.llm_services?.memory && (
                      <div className="flex-1 p-3 flex items-center justify-between gap-3 min-w-0">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <div className={cn(
                            "w-2 h-2 rounded-full flex-shrink-0",
                            health.checks.llm_services.memory.status === "healthy" ? "bg-green-400" :
                            health.checks.llm_services.memory.status === "warning" ? "bg-amber-400" : "bg-destructive"
                          )} title={health.checks.llm_services.memory.status} />
                          <div className="flex flex-col min-w-0">
                            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none mb-1">
                              {t("workers.health.qwen_memory")}
                            </span>
                            <span className="text-xs font-semibold truncate" title={health.checks.llm_services.memory.model}>
                              {health.checks.llm_services.memory.model || "—"}
                            </span>
                          </div>
                        </div>
                        {health.checks.llm_services.memory.error && (
                          <span className="text-[10px] text-destructive font-medium truncate max-w-[100px]" title={health.checks.llm_services.memory.error}>
                            {health.checks.llm_services.memory.error}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Main LLM */}
                    {health.checks.llm_services?.main_llm && (
                      <div className="flex-1 p-3 flex items-center justify-between gap-3 min-w-0">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <div className={cn(
                            "w-2 h-2 rounded-full flex-shrink-0",
                            health.checks.llm_services.main_llm.status === "healthy" ? "bg-green-400" :
                            health.checks.llm_services.main_llm.status === "warning" ? "bg-amber-400" : "bg-destructive"
                          )} title={health.checks.llm_services.main_llm.status} />
                          <div className="flex flex-col min-w-0">
                            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none mb-1">
                              {t("workers.health.main_llm")}
                            </span>
                            <span className="text-xs font-semibold truncate" title={health.checks.llm_services.main_llm.model}>
                              {health.checks.llm_services.main_llm.model || "—"}
                            </span>
                          </div>
                        </div>
                        {health.checks.llm_services.main_llm.error && (
                          <span className="text-[10px] text-destructive font-medium truncate max-w-[100px]" title={health.checks.llm_services.main_llm.error}>
                            {health.checks.llm_services.main_llm.error}
                          </span>
                        )}
                      </div>
                    )}

                    {/* Embed + Rerank service */}
                    {health.checks.llm_services?.embed_rerank && (
                      <div className="flex-1 p-3 flex items-center justify-between gap-3 min-w-0">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <div className={cn(
                            "w-2 h-2 rounded-full flex-shrink-0",
                            health.checks.llm_services.embed_rerank.status === "healthy" ? "bg-green-400" :
                            health.checks.llm_services.embed_rerank.status === "warning" ? "bg-amber-400" : "bg-destructive"
                          )} title={health.checks.llm_services.embed_rerank.status} />
                          <div className="flex flex-col min-w-0">
                            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none mb-1">
                              {t("workers.health.embed_rerank")}
                            </span>
                            <span className="text-xs font-semibold truncate" title={health.checks.llm_services.embed_rerank.model}>
                              {health.checks.llm_services.embed_rerank.model || "—"}
                            </span>
                          </div>
                        </div>
                        {health.checks.llm_services.embed_rerank.error && (
                          <span className="text-[10px] text-destructive font-medium truncate max-w-[100px]" title={health.checks.llm_services.embed_rerank.error}>
                            {health.checks.llm_services.embed_rerank.error}
                          </span>
                        )}
                      </div>
                    )}

                    {/* STT (Whisper) service */}
                    {health.checks.llm_services?.stt && (
                      <div className="flex-1 p-3 flex items-center justify-between gap-3 min-w-0">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <div className={cn(
                            "w-2 h-2 rounded-full flex-shrink-0",
                            health.checks.llm_services.stt.status === "healthy" ? "bg-green-400" :
                            health.checks.llm_services.stt.status === "warning" ? "bg-amber-400" : "bg-destructive"
                          )} title={health.checks.llm_services.stt.status} />
                          <div className="flex flex-col min-w-0">
                            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none mb-1">
                              {t("workers.health.stt")}
                            </span>
                            <span className="text-xs font-semibold truncate" title={health.checks.llm_services.stt.model}>
                              {health.checks.llm_services.stt.model || "—"}
                            </span>
                          </div>
                        </div>
                        {health.checks.llm_services.stt.error && (
                          <span className="text-[10px] text-destructive font-medium truncate max-w-[100px]" title={health.checks.llm_services.stt.error}>
                            {health.checks.llm_services.stt.error}
                          </span>
                        )}
                      </div>
                    )}

                    {/* TTS (OmniVoice) service */}
                    {health.checks.llm_services?.tts && (
                      <div className="flex-1 p-3 flex items-center justify-between gap-3 min-w-0">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <div className={cn(
                            "w-2 h-2 rounded-full flex-shrink-0",
                            health.checks.llm_services.tts.status === "healthy" ? "bg-green-400" :
                            health.checks.llm_services.tts.status === "warning" ? "bg-amber-400" : "bg-destructive"
                          )} title={health.checks.llm_services.tts.status} />
                          <div className="flex flex-col min-w-0">
                            <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground/70 leading-none mb-1">
                              {t("workers.health.tts")}
                            </span>
                            <span className="text-xs font-semibold truncate" title={health.checks.llm_services.tts.model}>
                              {health.checks.llm_services.tts.model || "—"}
                            </span>
                          </div>
                        </div>
                        {health.checks.llm_services.tts.error && (
                          <span className="text-[10px] text-destructive font-medium truncate max-w-[100px]" title={health.checks.llm_services.tts.error}>
                            {health.checks.llm_services.tts.error}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </Section>
            )}

            {/* ── GPU / VRAM (real-time, 3s poll) ── */}
            {gpuData?.available && gpuData.gpus.length > 0 && (
              <Section
                title={t("workers.gpu.title")}
                icon={Gauge}
                badge={gpuData.gpus.length > 1 ? gpuData.gpus.length : undefined}
                defaultOpen={true}
              >
                <div className="space-y-3">
                  {gpuData.gpus.map((gpu) => (
                    <GpuCard key={gpu.index} gpu={gpu} />
                  ))}
                </div>
              </Section>
            )}

            {/* ── Pipeline Summary Cards ── */}
            <Section title={t("workers.pipeline_summary")} icon={Layers}>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 xl:grid-cols-8 gap-2.5">
                {pipeline &&
                  Object.entries(pipeline).map(([key, count]) => {
                    const config = PIPELINE_STATUS[key] ?? PIPELINE_STATUS.pending;
                    const Icon = config.icon;
                    const isAnimated = PROCESSING_KEYS.includes(key as any) && count > 0;
                    return (
                      <div
                        key={key}
                        className={cn(
                          "rounded-xl border bg-card p-2.5 flex flex-col items-center gap-1.5 transition-all hover:shadow-sm hover:border-primary/20 group relative overflow-hidden",
                          count > 0 && key === "failed" && "border-destructive/30 bg-destructive/5",
                          isAnimated && "border-blue-400/30 bg-blue-400/5",
                        )}
                        title={t(`workers.status_desc.${key}`)}
                      >
                        <div className={cn("w-7 h-7 rounded-lg flex items-center justify-center transition-transform group-hover:scale-110", config.bgColor)}>
                          <Icon className={cn("w-3.5 h-3.5", config.color, isAnimated && "animate-spin")} />
                        </div>
                        <span className="text-base font-bold tabular-nums leading-none">{count}</span>
                        <div className="flex flex-col items-center gap-0.5 text-center w-full">
                          <span className="text-[9px] text-muted-foreground font-bold uppercase tracking-tighter truncate w-full">
                            {t(config.labelKey)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
              </div>
            </Section>


            {/* ── Queue Details (grouped by worker type) ── */}
            {overview && (() => {
              const activeGroups = WORKER_TYPES.filter((wt) => workerQueueGroups[wt].length > 0);
              if (activeGroups.length === 0 && systemQueues.length === 0) return null;
              return (
              <Section title={t("workers.queue_details")} icon={Inbox} badge={activeGroups.length}>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                  {activeGroups.map((wt) => {
                    const queues = workerQueueGroups[wt];
                    const agg = aggregateQueues(queues);
                    // Only offer purge/delete when the group maps to a single concrete
                    // queue (parse/embed/caption, or a single-workspace KG queue).
                    const singleQueue = queues.length === 1 ? queues[0].name : null;
                    return (
                      <div key={wt} className="rounded-xl border bg-card p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <Inbox className={cn("w-3.5 h-3.5 flex-shrink-0", WORKER_COLORS[wt])} />
                            <span className={cn("text-sm font-semibold truncate", WORKER_COLORS[wt])}>
                              {t(`workers.types.${wt}`) || wt}
                            </span>
                            {wt === "kg" && queues.length > 1 && (
                              <span className="text-[10px] text-muted-foreground flex-shrink-0">
                                {t("workers.kg_queue_count", { count: queues.length })}
                              </span>
                            )}
                            {agg.has_dlx && (
                              <span
                                className="text-[9px] px-1 py-0.5 rounded bg-green-500/10 text-green-400 border border-green-500/20 flex-shrink-0"
                                title={t("workers.dlx_tip")}
                              >
                                DLX
                              </span>
                            )}
                          </div>
                          {singleQueue && (
                            <div className="flex items-center gap-0.5 flex-shrink-0">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6"
                                title={t("workers.purge_queue")}
                                onClick={() => setPurgeConfirm(singleQueue)}
                              >
                                <Trash2 className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6"
                                title={t("workers.delete_queue")}
                                onClick={() => setDeleteConfirm(singleQueue)}
                              >
                                <Minus className="w-3 h-3 text-muted-foreground hover:text-destructive" />
                              </Button>
                            </div>
                          )}
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                          <div className="text-center" title={t("workers.queue_tip_ready")}>
                            <p className="text-lg font-bold tabular-nums">{agg.messages_ready}</p>
                            <p className="text-[10px] text-muted-foreground">{t("workers.ready")}</p>
                          </div>
                          <div className="text-center" title={t("workers.queue_tip_processing")}>
                            <p className="text-lg font-bold tabular-nums text-amber-400">{agg.messages_unacked}</p>
                            <p className="text-[10px] text-muted-foreground">{t("workers.processing")}</p>
                          </div>
                          <div className="text-center" title={t("workers.queue_tip_consumers")}>
                            <p className="text-lg font-bold tabular-nums text-primary">{agg.consumers}</p>
                            <p className="text-[10px] text-muted-foreground">{t("workers.consumers_label")}</p>
                          </div>
                        </div>
                        {(agg.message_rate_in > 0 || agg.message_rate_out > 0) && (
                          <div className="flex items-center justify-between text-[11px] text-muted-foreground pt-1 border-t border-border/50">
                            <span>{t("workers.labels.in")}: {agg.message_rate_in.toFixed(1)}{t("workers.labels.per_second")}</span>
                            <span>{t("workers.labels.out")}: {agg.message_rate_out.toFixed(1)}{t("workers.labels.per_second")}</span>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* System queues (retry / memory) — collapsed, advanced */}
                {systemQueues.length > 0 && (
                  <div className="mt-4">
                    <Section
                      title={t("workers.system_queues")}
                      icon={Layers}
                      badge={systemQueues.length}
                      defaultOpen={false}
                    >
                      <p className="text-[11px] text-muted-foreground/70 mb-2">
                        {t("workers.system_queues_desc")}
                      </p>
                      <div className="rounded-xl border bg-card overflow-hidden divide-y divide-border/50">
                        {systemQueues.map((q) => (
                          <div key={q.name} className="flex items-center justify-between gap-3 px-4 py-2.5">
                            <span className="text-xs font-mono truncate min-w-0" title={q.name}>
                              {q.name}
                            </span>
                            <div className="flex items-center gap-4 flex-shrink-0 text-xs tabular-nums text-muted-foreground">
                              <span title={t("workers.queue_tip_ready")}>
                                {t("workers.ready")} {q.messages_ready}
                              </span>
                              <span title={t("workers.queue_tip_processing")}>
                                {t("workers.processing")} {q.messages_unacked}
                              </span>
                              <div className="flex items-center gap-0.5">
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
                          </div>
                        ))}
                      </div>
                    </Section>
                  </div>
                )}
              </Section>
              );
            })()}

            {/* ── Dead Letter Queue ── */}
            <div id="dlq-section">
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
          </div>

            {/* ── Active Documents (Processing) ── */}
            {!pipelineLoading && activeDocs.length > 0 && (
              <Section title={t("workers.processing_with_count", { count: activeDocs.length })} icon={Loader2}>
                <div className="rounded-xl border bg-card overflow-hidden">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b bg-muted/30">
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.file")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("common.status")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.sub_tasks")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.time")}</th>
                        <th className="text-left px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.updated")}</th>
                        <th className="text-right px-4 py-2 text-xs font-medium text-muted-foreground">{t("workers.actions")}</th>
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
                            <td className="px-4 py-2.5 text-right">
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 text-xs gap-1 text-destructive hover:text-destructive"
                                onClick={() => setCancelConfirmDoc(String(doc.id))}
                                disabled={cancelDoc.isPending}
                              >
                                <XCircle className="w-3 h-3" />
                                {t("common.cancel")}
                              </Button>
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
                              {doc.error_message || t("workers.errors.unknown")}
                            </span>
                          </td>
                          <td className="px-4 py-2.5 text-xs text-muted-foreground">
                            {doc.updated_at ? formatRelativeDate(doc.updated_at) : "—"}
                          </td>
                          <td className="px-4 py-2.5 text-right">
                            <div className="flex items-center justify-end gap-1">
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
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-7 text-xs gap-1 text-destructive hover:text-destructive"
                                onClick={() => setDeleteFailedDocId(String(doc.id))}
                                disabled={deleteFailedDoc.isPending}
                              >
                                <Trash2 className="w-3 h-3" />
                              </Button>
                            </div>
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
                    title={t("workers.health.rabbitmq")}
                    status={health.checks.rabbitmq.status}
                    details={[
                      health.checks.rabbitmq.version ? `${t("workers.labels.version")}: ${health.checks.rabbitmq.version}` : null,
                      health.checks.rabbitmq.cluster ? `${t("workers.labels.cluster")}: ${health.checks.rabbitmq.cluster}` : null,
                      health.checks.rabbitmq.error ? `${t("workers.labels.error")}: ${health.checks.rabbitmq.error}` : null,
                    ].filter(Boolean) as string[]}
                  />

                  {/* Pipeline */}
                  <HealthCard
                    title={t("workers.health.pipeline")}
                    status={health.checks.pipeline.status}
                    details={[
                      t("workers.health.in_progress", { count: health.checks.pipeline.documents_in_progress }),
                      t("workers.health.failed", { count: health.checks.pipeline.documents_failed }),
                    ]}
                  />

                  {/* Dead Letter Queue */}
                  <HealthCard
                    title={t("workers.health.dlq")}
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

      <ConfirmDialog
        open={cancelConfirmDoc !== null}
        onConfirm={() => {
          if (cancelConfirmDoc) {
            cancelDoc.mutate(cancelConfirmDoc);
            setCancelConfirmDoc(null);
          }
        }}
        onCancel={() => setCancelConfirmDoc(null)}
        title={t("workers.cancel_title")}
        message={t("workers.cancel_msg")}
        confirmLabel={t("common.cancel")}
        variant="danger"
      />

      <ConfirmDialog
        open={deleteFailedDocId !== null}
        onConfirm={() => {
          if (deleteFailedDocId) {
            deleteFailedDoc.mutate(deleteFailedDocId);
            setDeleteFailedDocId(null);
          }
        }}
        onCancel={() => setDeleteFailedDocId(null)}
        title={t("workers.delete_doc_title")}
        message={t("workers.delete_doc_msg")}
        confirmLabel={t("common.delete")}
        variant="danger"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// GpuCard — real-time VRAM / utilization / temperature for one GPU.
// The VRAM bar is stacked: one colored segment per GPU process, a grey
// "other" segment for memory NVML can't attribute (CUDA contexts, etc.),
// and the remaining background = free.
// ---------------------------------------------------------------------------
const GPU_PROC_COLORS = [
  "bg-blue-400",
  "bg-violet-400",
  "bg-amber-400",
  "bg-cyan-400",
  "bg-pink-400",
  "bg-emerald-400",
  "bg-orange-400",
  "bg-indigo-400",
];

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const { t } = useTranslation();
  const pct = gpu.memory_pct;
  const pctColor =
    pct >= 90 ? "text-destructive" :
    pct >= 70 ? "text-amber-400" :
    "text-green-400";
  const toGb = (mb: number) => (mb / 1024).toFixed(1);

  const procs = gpu.processes ?? [];
  const procSumMb = procs.reduce((s, p) => s + p.memory_mb, 0);
  const otherMb = Math.max(0, gpu.memory_used_mb - procSumMb);
  const widthPct = (mb: number) => (mb / gpu.memory_total_mb) * 100;

  return (
    <div className="rounded-xl border bg-card p-4 space-y-3">
      {/* Header: GPU name + util/temp + VRAM % */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2 min-w-0">
          <MemoryStick className="w-3.5 h-3.5 text-primary flex-shrink-0" />
          <span className="text-sm font-semibold truncate" title={gpu.name}>
            {gpu.name}
          </span>
          <span className="text-[10px] text-muted-foreground flex-shrink-0">
            #{gpu.index}
          </span>
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          {gpu.utilization_pct != null && (
            <span className="inline-flex items-center gap-1.5" title={t("workers.gpu.utilization")}>
              <Gauge className="w-3 h-3" />
              <span className="font-semibold tabular-nums text-foreground">{gpu.utilization_pct}%</span>
            </span>
          )}
          {gpu.temperature_c != null && (
            <span className="inline-flex items-center gap-1.5" title={t("workers.gpu.temperature")}>
              <Thermometer className="w-3 h-3" />
              <span className={cn(
                "font-semibold tabular-nums",
                gpu.temperature_c >= 85 ? "text-destructive" :
                gpu.temperature_c >= 75 ? "text-amber-400" : "text-foreground",
              )}>
                {gpu.temperature_c}°C
              </span>
            </span>
          )}
          <span className={cn("text-sm font-bold tabular-nums", pctColor)}>
            {pct.toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Stacked VRAM bar — one segment per process */}
      <div>
        <div className="h-3 rounded-full bg-muted overflow-hidden flex">
          {procs.map((p, i) => (
            <div
              key={p.pid}
              className={cn(
                "h-full transition-all duration-700 flex-shrink-0",
                GPU_PROC_COLORS[i % GPU_PROC_COLORS.length],
              )}
              style={{ width: `${widthPct(p.memory_mb)}%` }}
              title={`${p.label} — ${toGb(p.memory_mb)} GB (PID ${p.pid})`}
            />
          ))}
          {otherMb > 0 && (
            <div
              className="h-full bg-muted-foreground/40 transition-all duration-700 flex-shrink-0"
              style={{ width: `${widthPct(otherMb)}%` }}
              title={`${t("workers.gpu.other")} — ${toGb(otherMb)} GB`}
            />
          )}
        </div>
        <div className="flex items-center justify-between mt-1.5 text-[11px] text-muted-foreground tabular-nums">
          <span>
            {t("workers.gpu.vram")}: {toGb(gpu.memory_used_mb)} / {toGb(gpu.memory_total_mb)} GB
          </span>
          <span>
            {t("workers.gpu.free")}: {toGb(gpu.memory_total_mb - gpu.memory_used_mb)} GB
          </span>
        </div>
      </div>

      {/* Process legend — horizontal chips, wraps to screen width */}
      {procs.length > 0 && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 pt-2 border-t border-border/50">
          {procs.map((p, i) => (
            <span
              key={p.pid}
              className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground"
              title={`PID ${p.pid}`}
            >
              <span className={cn(
                "w-2 h-2 rounded-sm flex-shrink-0",
                GPU_PROC_COLORS[i % GPU_PROC_COLORS.length],
              )} />
              <span className="font-medium text-foreground">{p.label}</span>
              <span className="tabular-nums">{toGb(p.memory_mb)} GB</span>
            </span>
          ))}
          {otherMb > 256 && (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <span className="w-2 h-2 rounded-sm flex-shrink-0 bg-muted-foreground/40" />
              <span>{t("workers.gpu.other")}</span>
              <span className="tabular-nums">{toGb(otherMb)} GB</span>
            </span>
          )}
        </div>
      )}
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
  const { t } = useTranslation();
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
        <span className={cn("text-[10px] font-medium px-1.5 py-0.5 rounded-full", statusBg, statusColor)}>
          {t(`workers.health_status.${status}`) || status}
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
  const { t } = useTranslation();
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 px-1 py-0.5 rounded text-[9px] font-medium border",
        done
          ? "bg-green-500/10 text-green-400 border-green-500/20"
          : "bg-muted/50 text-muted-foreground/50 border-border/50",
      )}
      title={`${label === "E" ? t("workers.types.embed") : label === "C" ? t("workers.types.caption") : t("workers.types.kg")}: ${done ? t("common.completed") : t("pipeline.stats.ready")}`}
    >
      {done ? <CheckCircle2 className="w-2.5 h-2.5" /> : <Clock className="w-2.5 h-2.5" />}
      {label}
    </span>
  );
}
