import { memo, useState, useCallback } from "react";
import { useTranslation } from "@/hooks/useTranslation";
import {
  ArrowRight,
  RefreshCw,
  Trash2,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Wifi,
  WifiOff,
  Server,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  useWorkerOverview,
  usePipelineDocuments,
  usePurgeQueue,
  useRetryFailed,
  useRetryDocument,
} from "@/hooks/useWorkers";
import type { QueueInfo, PipelineDocument, DocumentStatus } from "@/types";

// ---------------------------------------------------------------------------
// Queue Health Card
// ---------------------------------------------------------------------------
function QueueCard({ queue }: { queue: QueueInfo }) {
  const { t } = useTranslation();
  const hasConsumers = queue.consumers > 0;
  const hasBacklog = queue.messages_ready > 10;

  const statusColor = !hasConsumers
    ? "border-destructive/30 bg-destructive/5"
    : hasBacklog
      ? "border-amber-400/30 bg-amber-400/5"
      : "border-primary/20 bg-primary/5";

  const dotColor = !hasConsumers
    ? "bg-destructive"
    : hasBacklog
      ? "bg-amber-400"
      : "bg-primary";

  // Extract short name: "nexusrag.parse" → "parse"
  const shortName = queue.name.replace("hrag.", "").split(".")[0];

  return (
    <div className={cn("rounded-lg border px-3 py-2.5 space-y-1.5", statusColor)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <div className={cn("w-1.5 h-1.5 rounded-full", dotColor)} />
          <span className="text-xs font-semibold uppercase tracking-wider">
            {shortName}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Server className="w-3 h-3 text-muted-foreground" />
          <span className="text-xs font-bold">{queue.consumers}</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
        <div className="flex justify-between">
          <span className="text-muted-foreground">{t("pipeline.stats.ready")}</span>
          <span className="font-medium">{queue.messages_ready}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">{t("pipeline.stats.inflight")}</span>
          <span className="font-medium">{queue.messages_unacked}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">{t("pipeline.stats.in_s")}</span>
          <span className="font-medium">{queue.message_rate_in.toFixed(1)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">{t("pipeline.stats.out_s")}</span>
          <span className="font-medium">{queue.message_rate_out.toFixed(1)}</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pipeline Status Flow
// ---------------------------------------------------------------------------
function PipelineFlow({ summary }: { summary: Record<string, number> | { [K: string]: number } }) {
  const { t } = useTranslation();
  const PIPELINE_STAGES: { key: string; label: string; color: string }[] = [
    { key: "pending", label: t("common.pending"), color: "bg-muted text-muted-foreground" },
    { key: "parsing", label: t("files.status.parsing"), color: "bg-blue-400/15 text-blue-400" },
    { key: "ocring", label: t("files.status.ocring"), color: "bg-indigo-400/15 text-indigo-400" },
    { key: "chunking", label: t("files.status.chunking"), color: "bg-cyan-400/15 text-cyan-400" },
    { key: "embedding", label: t("files.status.embedding"), color: "bg-amber-400/15 text-amber-400" },
    { key: "building_kg", label: t("files.status.building_kg"), color: "bg-violet-400/15 text-violet-400" },
    { key: "indexed", label: t("files.status.indexed"), color: "bg-primary/15 text-primary" },
    { key: "failed", label: t("common.error"), color: "bg-destructive/15 text-destructive" },
  ];

  return (
    <div className="flex items-center gap-1 flex-wrap">
      {PIPELINE_STAGES.map((stage, i) => {
        const count = summary[stage.key] ?? 0;
        return (
          <div key={stage.key} className="flex items-center gap-1">
            {i > 0 && i < PIPELINE_STAGES.length - 1 && (
              <ArrowRight className="w-3 h-3 text-muted-foreground/40 flex-shrink-0" />
            )}
            {i === PIPELINE_STAGES.length - 1 && (
              <span className="text-muted-foreground/40 text-[10px] mx-0.5">/</span>
            )}
            <div
              className={cn(
                "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium",
                stage.color,
                count === 0 && "opacity-40"
              )}
            >
              {stage.label}
              <span className="font-bold">{count}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Status badge for pipeline documents
// ---------------------------------------------------------------------------
const STATUS_ICON: Record<string, typeof CheckCircle2> = {
  pending: Clock,
  parsing: Loader2,
  ocring: Loader2,
  chunking: Loader2,
  embedding: Loader2,
  building_kg: Loader2,
  indexed: CheckCircle2,
  failed: XCircle,
};

const STATUS_CLASS: Record<string, string> = {
  pending: "text-muted-foreground",
  parsing: "text-blue-400",
  ocring: "text-indigo-400",
  chunking: "text-cyan-400",
  embedding: "text-amber-400",
  building_kg: "text-violet-400",
  indexed: "text-primary",
  failed: "text-destructive",
};

const ANIMATING: Set<string> = new Set([
  "parsing", "ocring", "chunking", "embedding", "building_kg",
]);

function DocStatusBadge({ status }: { status: DocumentStatus }) {
  const Icon = STATUS_ICON[status] || Clock;
  const cls = STATUS_CLASS[status] || "text-muted-foreground";
  const animate = ANIMATING.has(status);

  return (
    <div className={cn("flex items-center gap-1", cls)}>
      <Icon className={cn("w-3 h-3", animate && "animate-spin")} />
      <span className="text-[10px] font-medium capitalize">
        {status.replace("_", " ")}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-task pills
// ---------------------------------------------------------------------------
function SubTaskPills({ doc }: { doc: PipelineDocument }) {
  const pills = [
    { label: "Embed", done: doc.embed_done },
    { label: "Captions", done: doc.captions_done },
    { label: "KG", done: doc.kg_done },
  ];
  return (
    <div className="flex items-center gap-1">
      {pills.map((p) => (
        <span
          key={p.label}
          className={cn(
            "text-[9px] px-1.5 py-0.5 rounded-full font-medium",
            p.done
              ? "bg-primary/10 text-primary"
              : "bg-muted text-muted-foreground"
          )}
        >
          {p.label} {p.done ? "✓" : "⟳"}
        </span>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Active Documents Table
// ---------------------------------------------------------------------------
function ActiveDocuments({
  documents,
  onRetry,
  retryingId,
}: {
  documents: PipelineDocument[];
  onRetry: (id: string) => void;
  retryingId: string | null;
}) {
  const { t } = useTranslation();
  if (documents.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
        <CheckCircle2 className="w-8 h-8 mb-2 opacity-40" />
        <p className="text-xs">{t("pipeline.no_active")}</p>
      </div>
    );
  }

  return (
    <div className="space-y-1">
      {documents.map((doc) => (
        <div
          key={doc.id}
          className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-muted/50 transition-colors text-xs"
        >
          {/* Filename + status */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-medium truncate max-w-[200px]">
                {doc.filename}
              </span>
              <DocStatusBadge status={doc.status} />
            </div>
            {/* Sub-tasks shown for chunking+ */}
            {["chunking", "embedding", "building_kg"].includes(doc.status) && (
              <div className="mt-1">
                <SubTaskPills doc={doc} />
              </div>
            )}
          </div>

          {/* Error message */}
          {doc.error_message && (
            <span className="text-destructive text-[10px] max-w-[150px] truncate" title={doc.error_message}>
              {doc.error_message}
            </span>
          )}

          {/* Time */}
          {doc.processing_time_ms != null && doc.processing_time_ms > 0 && (
            <span className="text-[10px] text-muted-foreground whitespace-nowrap">
              {(doc.processing_time_ms / 1000).toFixed(1)}s
            </span>
          )}

          {/* Retry button for failed docs */}
          {doc.status === "failed" && (
            <button
              onClick={() => onRetry(doc.id)}
              disabled={retryingId === doc.id}
              className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium bg-destructive/10 text-destructive hover:bg-destructive/20 transition-colors disabled:opacity-50"
            >
              {retryingId === doc.id ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <RefreshCw className="w-3 h-3" />
              )}
              {t("pipeline.retry")}
            </button>
          )}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// PipelinePanel — main component
// ---------------------------------------------------------------------------
interface PipelinePanelProps {
  workspaceId: string;
}

export const PipelinePanel = memo(function PipelinePanel({
  workspaceId,
}: PipelinePanelProps) {
  const { t } = useTranslation();
  const { data: overview, isLoading: overviewLoading } = useWorkerOverview();
  const { data: pipelineData, isLoading: pipelineLoading } =
    usePipelineDocuments(workspaceId);
  const purgeQueue = usePurgeQueue();
  const retryFailed = useRetryFailed();
  const retryDocument = useRetryDocument();

  const [retryingDocId, setRetryingDocId] = useState<string | null>(null);
  const [purgeTarget, setPurgeTarget] = useState<string | null>(null);

  const handleRetryDoc = useCallback(
    (docId: string) => {
      setRetryingDocId(docId);
      retryDocument.mutate(docId, {
        onSettled: () => setRetryingDocId(null),
      });
    },
    [retryDocument]
  );

  const handleRetryAll = useCallback(() => {
    retryFailed.mutate(workspaceId);
  }, [retryFailed, workspaceId]);

  const handlePurge = useCallback(
    (queueName: string) => {
      if (!confirm(t("pipeline.purge_confirm", { queue: queueName }))) {
        return;
      }
      setPurgeTarget(queueName);
      purgeQueue.mutate(queueName, {
        onSettled: () => setPurgeTarget(null),
      });
    },
    [purgeQueue]
  );

  const isLoading = overviewLoading || pipelineLoading;
  const documents = pipelineData?.documents ?? [];
  const failedCount = overview?.pipeline_summary?.failed ?? 0;

  if (isLoading && !overview) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col min-h-0 overflow-y-auto">
      <div className="p-3 space-y-4">
        {/* Connection status */}
        <div className="flex items-center gap-2">
          {overview?.rabbitmq_connected ? (
            <div className="flex items-center gap-1.5 text-primary text-xs">
              <Wifi className="w-3.5 h-3.5" />
              <span className="font-medium">{t("pipeline.rabbitmq_connected")}</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-destructive text-xs">
              <WifiOff className="w-3.5 h-3.5" />
              <span className="font-medium">{t("pipeline.rabbitmq_disconnected")}</span>
            </div>
          )}
          <span className="text-[10px] text-muted-foreground ml-auto">
            {t("pipeline.auto_refresh", { seconds: 5 })}
          </span>
        </div>

        {/* Section 1: Queue Health */}
        {overview?.queues && overview.queues.length > 0 && (
          <div>
            <h3 className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              {t("pipeline.queue_health")}
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {overview.queues.map((q) => (
                <QueueCard key={q.name} queue={q} />
              ))}
            </div>
          </div>
        )}

        {/* Section 2: Pipeline Status */}
        {overview?.pipeline_summary && (
          <div>
            <h3 className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              {t("pipeline.pipeline_status")}
            </h3>
            <PipelineFlow summary={overview.pipeline_summary as unknown as Record<string, number>} />
          </div>
        )}

        {/* Section 3: Active Documents */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {t("pipeline.active_docs")}
            </h3>
            {failedCount > 0 && (
              <button
                onClick={handleRetryAll}
                disabled={retryFailed.isPending}
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium bg-destructive/10 text-destructive hover:bg-destructive/20 transition-colors disabled:opacity-50"
              >
                {retryFailed.isPending ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <RefreshCw className="w-3 h-3" />
                )}
                {t("pipeline.retry_all", { count: failedCount })}
              </button>
            )}
          </div>
          <ActiveDocuments
            documents={documents}
            onRetry={handleRetryDoc}
            retryingId={retryingDocId}
          />
        </div>

        {/* Section 4: Queue Actions */}
        {overview?.queues && overview.queues.length > 0 && (
          <div>
            <h3 className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              {t("pipeline.queue_actions")}
            </h3>
            <div className="flex flex-wrap gap-2">
              {overview.queues.map((q) => (
                <button
                  key={q.name}
                  onClick={() => handlePurge(q.name)}
                  disabled={purgeTarget === q.name || q.messages_ready === 0}
                  className="flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium border border-muted hover:bg-muted/50 transition-colors disabled:opacity-30"
                >
                  {purgeTarget === q.name ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Trash2 className="w-3 h-3" />
                  )}
                  {t("common.purge")} {q.name.replace("hrag.", "")}
                  {q.messages_ready > 0 && (
                    <span className="bg-muted px-1 rounded-full">{q.messages_ready}</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Warning for no RabbitMQ */}
        {overview && !overview.rabbitmq_connected && (
          <div className="flex items-start gap-2 p-3 rounded-lg border border-amber-400/30 bg-amber-400/5">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-amber-400">
              <p className="font-medium">{t("pipeline.rabbitmq_error")}</p>
              <p className="mt-0.5 opacity-80">
                {t("pipeline.rabbitmq_error_desc")}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
