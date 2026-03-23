import { memo } from "react";
import { Search, X } from "lucide-react";
import { useTranslation } from "@/hooks/useTranslation";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "@/types";

type FilterStatus = "all" | DocumentStatus;

interface DocumentFiltersProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  statusFilter: FilterStatus;
  onStatusChange: (s: FilterStatus) => void;
  counts: Record<FilterStatus, number>;
}

export type { FilterStatus };

export const DocumentFilters = memo(({
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusChange,
  counts,
}: DocumentFiltersProps) => {
  const { t } = useTranslation();

  const TABS = [
    { id: "all", label: t("common.all"), count: counts.all },
    { id: "indexed", label: t("files.tabs.indexed"), color: "bg-green-500", count: counts.indexed },
    { id: "processing", label: t("files.tabs.processing"), color: "bg-blue-500", count: counts.parsing },
    { id: "failed", label: t("files.tabs.failed"), color: "bg-destructive", count: counts.failed },
  ];

  return (
    <div className="flex items-center gap-3 flex-wrap">
      {/* Search */}
      <div className="relative flex-1 min-w-[180px] max-w-xs">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
        <input
          type="text"
          placeholder={t("workspace.filter_placeholder")}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="w-full bg-muted/50 border-none rounded-md py-1.5 pl-8 pr-3 text-xs focus:ring-1 focus:ring-primary/30 transition-all"
        />
        {searchQuery && (
          <button
            onClick={() => onSearchChange("")}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 hover:bg-muted rounded"
          >
            <X className="w-3 h-3 text-muted-foreground" />
          </button>
        )}
      </div>

      {/* Status tabs */}
      <div className="flex items-center gap-1 bg-muted/40 rounded-lg p-0.5">
        {TABS.map((tab) => {
          const isActive = statusFilter === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onStatusChange(tab.id as FilterStatus)}
              className={cn(
                "flex items-center gap-2 px-2 py-1.5 rounded-md text-[11px] transition-colors",
                isActive
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-muted"
              )}
            >
              {tab.color && <div className={cn("w-1.5 h-1.5 rounded-full", tab.color)} />}
              {tab.label}
              {tab.count > 0 && (
                <span className={cn(
                  "ml-1 text-[10px]",
                  isActive ? "text-primary" : "text-muted-foreground/60"
                )}>
                  {tab.count}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
});
