import { useState } from "react";
import { useQuery, keepPreviousData } from "@tanstack/react-query";
import { useTranslation } from "@/hooks/useTranslation";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  History,
  User as UserIcon,
  ChevronDown,
  ShieldCheck,
  Cpu,
} from "lucide-react";
import { Select } from "@/components/ui/select";
import { Button } from "@/components/ui/button";

interface AuditLog {
  id: string;
  actor_id: string | null;
  actor_email: string | null;
  actor_name: string | null;
  action: string;
  resource_type: string;
  resource_id: string | null;
  resource_label: string | null;
  summary: string | null;
  extra: Record<string, unknown> | null;
  method: string | null;
  path: string | null;
  status_code: number | null;
  ip_address: string | null;
  source: string;
  created_at: string;
}

interface AuditLogList {
  items: AuditLog[];
  total: number;
  page: number;
  per_page: number;
}

// 5 domains → one tab each. The tenant tab spans its sub-resources.
const DOMAINS: { key: string; types: string[] }[] = [
  { key: "abbreviation", types: ["abbreviation"] },
  { key: "user", types: ["user"] },
  { key: "tenant", types: ["tenant", "tenant_member", "tenant_invite"] },
  { key: "workspace", types: ["workspace"] },
  { key: "document_type", types: ["document_type"] },
];

const ACTIONS = [
  "create",
  "update",
  "delete",
  "set_default",
  "set_admin",
  "approve",
  "reject",
  "update_role",
  "update_prompt",
  "delete_prompt",
  "reset_password",
];

const ACTION_STYLES: Record<string, string> = {
  create: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
  approve: "bg-emerald-500/10 text-emerald-500 border-emerald-500/20",
  set_admin: "bg-violet-500/10 text-violet-500 border-violet-500/20",
  set_default: "bg-sky-500/10 text-sky-500 border-sky-500/20",
  update: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  update_role: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  update_prompt: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  delete: "bg-red-500/10 text-red-500 border-red-500/20",
  reject: "bg-red-500/10 text-red-500 border-red-500/20",
  delete_prompt: "bg-red-500/10 text-red-500 border-red-500/20",
  reset_password: "bg-amber-500/10 text-amber-500 border-amber-500/20",
};

const PER_PAGE = 50;

function formatTime(iso: string): string {
  // Backend timestamps are naive UTC — normalise to a real UTC instant.
  const s = /Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z";
  const d = new Date(s);
  return d.toLocaleString([], {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function SystemLogsPage() {
  const { t } = useTranslation();
  const [domain, setDomain] = useState(DOMAINS[0].key);
  const [action, setAction] = useState("");
  const [page, setPage] = useState(1);

  const activeDomain = DOMAINS.find((d) => d.key === domain) ?? DOMAINS[0];

  const query = useQuery({
    queryKey: ["audit-logs", domain, action, page],
    queryFn: () => {
      const params = new URLSearchParams();
      params.set("page", String(page));
      params.set("per_page", String(PER_PAGE));
      activeDomain.types.forEach((rt) => params.append("resource_type", rt));
      if (action) params.set("action", action);
      return api.get<AuditLogList>(`/audit-logs?${params.toString()}`);
    },
    placeholderData: keepPreviousData,
    refetchInterval: 15000,
  });

  const data = query.data;
  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));

  const tr = (key: string, fallback: string) => {
    const v = t(key);
    return v === key ? fallback : v;
  };
  const resName = (type: string) => tr(`audit.resource.${type}`, type);
  const actName = (a: string) => tr(`audit.action_label.${a}`, a);

  const changeDomain = (key: string) => {
    setDomain(key);
    setAction("");
    setPage(1);
  };

  return (
    <div className="flex flex-col h-full bg-background overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-border bg-card/50 backdrop-blur-sm px-6 pt-4">
        <div className="max-w-screen-2xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="p-2 rounded-xl bg-primary/10 text-primary">
                <History className="w-5 h-5" />
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight">{t("audit.title")}</h1>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-[11px] text-muted-foreground">{t("audit.subtitle")}</span>
                  <span className="text-[10px] text-muted-foreground/50">•</span>
                  <span className="text-[11px] font-medium text-muted-foreground">
                    {total.toLocaleString()} {t("audit.total")}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Action filter */}
              <div className="relative">
                <Select
                  value={action}
                  onChange={(e) => {
                    setAction(e.target.value);
                    setPage(1);
                  }}
                  className="pl-3 pr-8 h-10 w-[160px] text-sm font-medium bg-muted/20 border-transparent hover:bg-muted/30 rounded-lg"
                >
                  <option value="">{t("audit.all_actions")}</option>
                  {ACTIONS.map((a) => (
                    <option key={a} value={a}>
                      {actName(a)}
                    </option>
                  ))}
                </Select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
              </div>

              <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10 rounded-lg text-muted-foreground hover:bg-muted"
                onClick={() => query.refetch()}
                title={t("audit.refresh")}
              >
                <RefreshCw className={cn("w-4 h-4", query.isFetching && "animate-spin")} />
              </Button>
            </div>
          </div>

          {/* Domain tabs */}
          <div className="flex items-center gap-1 mt-3 -mb-px overflow-x-auto">
            {DOMAINS.map((d) => {
              const isActive = d.key === domain;
              return (
                <button
                  key={d.key}
                  onClick={() => changeDomain(d.key)}
                  className={cn(
                    "px-4 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 transition-colors",
                    isActive
                      ? "border-primary text-primary"
                      : "border-transparent text-muted-foreground hover:text-foreground hover:border-border"
                  )}
                >
                  {resName(d.key)}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-screen-2xl mx-auto w-full px-6 py-4">
          {items.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-24 text-muted-foreground/40">
              <History className="w-14 h-14 opacity-20 mb-4" />
              <p className="text-sm font-medium uppercase tracking-wide opacity-60">
                {t("audit.empty")}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto rounded-xl border border-border/60">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-muted/30 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
                    <th className="px-4 py-2.5 font-semibold whitespace-nowrap">{t("audit.col_time")}</th>
                    <th className="px-4 py-2.5 font-semibold whitespace-nowrap">{t("audit.col_actor")}</th>
                    <th className="px-4 py-2.5 font-semibold whitespace-nowrap">{t("audit.col_action")}</th>
                    <th className="px-4 py-2.5 font-semibold whitespace-nowrap">{t("audit.col_resource")}</th>
                    <th className="px-4 py-2.5 font-semibold">{t("audit.col_summary")}</th>
                    <th className="px-4 py-2.5 font-semibold whitespace-nowrap">{t("audit.col_source")}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {items.map((log) => (
                    <tr key={log.id} className="hover:bg-muted/30 transition-colors align-top">
                      <td className="px-4 py-3 whitespace-nowrap text-[12px] text-muted-foreground tabular-nums">
                        {formatTime(log.created_at)}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center shrink-0">
                            <UserIcon className="w-3.5 h-3.5" />
                          </div>
                          <div className="leading-tight">
                            <div className="font-medium text-foreground">
                              {log.actor_name || t("audit.system")}
                            </div>
                            {log.actor_email && (
                              <div className="text-[11px] text-muted-foreground">{log.actor_email}</div>
                            )}
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span
                          className={cn(
                            "inline-block px-2 py-0.5 rounded-md border text-[11px] font-semibold",
                            ACTION_STYLES[log.action] ||
                              "bg-slate-500/10 text-slate-400 border-slate-500/20"
                          )}
                        >
                          {actName(log.action)}
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="leading-tight">
                          <span className="inline-block px-1.5 py-0.5 rounded bg-muted/50 text-[11px] font-medium text-muted-foreground">
                            {resName(log.resource_type)}
                          </span>
                          {log.resource_label && (
                            <div className="mt-1 font-medium text-foreground max-w-[200px] truncate" title={log.resource_label}>
                              {log.resource_label}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-foreground/90 max-w-[420px]">
                        <span className="break-words">{log.summary || "—"}</span>
                        {log.ip_address && (
                          <span className="ml-2 text-[10px] text-muted-foreground/60">({log.ip_address})</span>
                        )}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap">
                        <span
                          className={cn(
                            "inline-flex items-center gap-1 text-[11px] font-medium",
                            log.source === "explicit" ? "text-emerald-500" : "text-muted-foreground"
                          )}
                          title={log.source}
                        >
                          {log.source === "explicit" ? (
                            <ShieldCheck className="w-3.5 h-3.5" />
                          ) : (
                            <Cpu className="w-3.5 h-3.5" />
                          )}
                          {log.source === "explicit" ? t("audit.explicit") : t("audit.auto")}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Footer / pagination */}
      <div className="flex-shrink-0 border-t border-border bg-card/50 backdrop-blur-sm px-6 py-2.5">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between text-[12px] text-muted-foreground">
          <span>
            {t("audit.page")} {page} / {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-8 gap-1"
              disabled={page <= 1 || query.isFetching}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft className="w-4 h-4" /> {t("audit.prev")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 gap-1"
              disabled={page >= totalPages || query.isFetching}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              {t("audit.next")} <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
