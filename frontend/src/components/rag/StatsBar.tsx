import { useTranslation } from "@/hooks/useTranslation";
import { type RAGStats } from "@/types";
import { Database, FileText, CheckCircle2, Loader2, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";

interface StatsBarProps {
  stats: RAGStats | undefined; // Align with useQuery's return type correctly
  processingCount: number;
}

export function StatsBar({ stats, processingCount }: StatsBarProps) {
  const { t } = useTranslation();

  const items = [
    {
      label: t("stats.total_docs"),
      value: stats?.total_documents || 0,
      icon: FileText,
      color: "text-indigo-500",
      bg: "bg-indigo-500/10",
    },
    {
      label: t("stats.indexed"),
      value: stats?.indexed_documents || 0,
      icon: CheckCircle2,
      color: "text-green-500",
      bg: "bg-green-500/10",
    },
    {
      label: t("stats.chunks"),
      value: stats?.total_chunks || 0,
      icon: Database,
      color: "text-blue-500",
      bg: "bg-blue-500/10",
    },
    {
      label: t("stats.images"),
      value: stats?.image_count || 0,
      icon: ImageIcon,
      color: "text-amber-500",
      bg: "bg-amber-500/10",
    },
  ];

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        {items.map((item) => (
          <div key={item.label} className={cn("p-2 rounded-lg border", item.bg)}>
            <div className="flex items-center gap-1.5 mb-1">
              <item.icon className={cn("w-3 h-3", item.color)} />
              <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-tight">
                {item.label}
              </span>
            </div>
            <p className="text-xs font-bold tabular-nums">{item.value}</p>
          </div>
        ))}
      </div>

      {processingCount > 0 && (
        <div className="flex items-center gap-2 p-1.5 px-2 bg-blue-500/5 border border-blue-200/20 rounded-md">
          <Loader2 className="w-3 h-3 text-blue-500 animate-spin" />
          <span className="text-[10px] text-blue-600 font-medium">
            {t("stats.processing_msg", { count: processingCount })}
          </span>
        </div>
      )}
    </div>
  );
}
