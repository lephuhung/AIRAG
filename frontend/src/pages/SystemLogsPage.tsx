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
  ChevronDown,
  ShieldCheck,
  Cpu,
  Send,
  Type,
  Users,
  Building2,
  LayoutGrid,
  FileType,
  Hash,
  Inbox,
  type LucideIcon,
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

interface TelegramLinkAdmin {
  telegram_chat_id: string;
  telegram_user_id: string | null;
  telegram_username: string | null;
  user_id: string;
  user_email: string | null;
  user_name: string | null;
  created_at: string;
}

// Pseudo-domain for the Telegram directory tab (not an audit resource_type).
const TELEGRAM_TAB = "telegram";

// 5 domains → one tab each. The tenant tab spans its sub-resources.
const DOMAINS: { key: string; types: string[] }[] = [
  { key: "abbreviation", types: ["abbreviation"] },
  { key: "user", types: ["user"] },
  { key: "tenant", types: ["tenant", "tenant_member", "tenant_invite"] },
  { key: "workspace", types: ["workspace"] },
  { key: "document_type", types: ["document_type"] },
];

// Icon per tab — gives the segmented control a quick visual anchor.
const TAB_ICONS: Record<string, LucideIcon> = {
  abbreviation: Type,
  user: Users,
  tenant: Building2,
  workspace: LayoutGrid,
  document_type: FileType,
  [TELEGRAM_TAB]: Send,
};

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
  create: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
  approve: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
  set_admin: "bg-violet-500/10 text-violet-600 dark:text-violet-400 border-violet-500/20",
  set_default: "bg-sky-500/10 text-sky-600 dark:text-sky-400 border-sky-500/20",
  update: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
  update_role: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
  update_prompt: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
  delete: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  reject: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  delete_prompt: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  reset_password: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
};

const PER_PAGE = 50;

// Backend timestamps are naive UTC — normalise to a real UTC instant, then split
// into date + time so rows can stack them for a cleaner two-line look.
function formatParts(iso: string): { date: string; time: string } {
  const s = /Z|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + "Z";
  const d = new Date(s);
  return {
    date: d.toLocaleDateString([], { day: "2-digit", month: "2-digit", year: "numeric" }),
    time: d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }),
  };
}

function initials(name: string | null | undefined): string {
  if (!name) return "?";
  return name
    .trim()
    .split(/\s+/)
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

export function SystemLogsPage() {
  const { t } = useTranslation();
  const [domain, setDomain] = useState(DOMAINS[0].key);
  const [action, setAction] = useState("");
  const [page, setPage] = useState(1);

  const isTelegram = domain === TELEGRAM_TAB;
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
    enabled: !isTelegram,
  });

  // Telegram directory — all linked accounts across users (superadmin endpoint).
  const tgQuery = useQuery({
    queryKey: ["telegram-links-all"],
    queryFn: () => api.get<TelegramLinkAdmin[]>("/integrations/telegram/links/all"),
    enabled: isTelegram,
    refetchInterval: 15000,
  });
  const tgLinks = tgQuery.data ?? [];

  const data = query.data;
  const items = data?.items ?? [];
  const total = isTelegram ? tgLinks.length : data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil((data?.total ?? 0) / PER_PAGE));
  const activeFetching = isTelegram ? tgQuery.isFetching : query.isFetching;
  const isEmpty = isTelegram ? tgLinks.length === 0 : items.length === 0;

  const tr = (key: string, fallback: string) => {
    const v = t(key);
    return v === key ? fallback : v;
  };
  const resName = (type: string) => tr(`audit.resource.${type}`, type);
  const actName = (a: string) => tr(`audit.action_label.${a}`, a);
  const tabName = (key: string) =>
    key === TELEGRAM_TAB ? tr("telegram_links.tab", "Telegram") : resName(key);

  const changeDomain = (key: string) => {
    setDomain(key);
    setAction("");
    setPage(1);
  };

  const allTabs = [...DOMAINS.map((d) => d.key), TELEGRAM_TAB];

  return (
    <div className="flex flex-col h-full bg-background overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-border/70 bg-card/40 backdrop-blur-sm px-6 pt-5">
        <div className="max-w-screen-2xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="flex items-center gap-3.5">
              <div className="grid place-items-center w-11 h-11 rounded-2xl bg-gradient-to-br from-primary/20 to-primary/5 text-primary ring-1 ring-primary/10">
                <History className="w-5 h-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight">{t("audit.title")}</h1>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className="text-xs text-muted-foreground">{t("audit.subtitle")}</span>
                  <span className="w-1 h-1 rounded-full bg-muted-foreground/30" />
                  <span className="text-xs font-medium text-foreground/70 tabular-nums">
                    {total.toLocaleString()} {isTelegram ? tr("telegram_links.total", "liên kết") : t("audit.total")}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2.5">
              {/* Action filter (audit tabs only) */}
              {!isTelegram && (
                <div className="relative">
                  <Select
                    value={action}
                    onChange={(e) => {
                      setAction(e.target.value);
                      setPage(1);
                    }}
                    className="pl-3.5 pr-9 h-10 w-[170px] text-sm font-medium bg-muted/30 border-border/50 hover:bg-muted/50 rounded-xl transition-colors"
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
              )}

              <Button
                variant="ghost"
                size="icon"
                className="h-10 w-10 rounded-xl text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                onClick={() => (isTelegram ? tgQuery.refetch() : query.refetch())}
                title={t("audit.refresh")}
              >
                <RefreshCw className={cn("w-4 h-4", activeFetching && "animate-spin")} />
              </Button>
            </div>
          </div>

          {/* Segmented tabs */}
          <div className="mt-4 pb-4 overflow-x-auto">
            <div className="inline-flex items-center gap-1 p-1 rounded-xl bg-muted/40 ring-1 ring-border/40">
              {allTabs.map((key) => {
                const Icon = TAB_ICONS[key] ?? History;
                const isActive = key === domain;
                return (
                  <button
                    key={key}
                    onClick={() => changeDomain(key)}
                    className={cn(
                      "flex items-center gap-1.5 px-3.5 py-1.5 text-[13px] font-medium rounded-lg whitespace-nowrap transition-all",
                      isActive
                        ? "bg-card text-foreground shadow-sm ring-1 ring-border/60"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <Icon className={cn("w-3.5 h-3.5", isActive ? "text-primary" : "opacity-70")} />
                    {tabName(key)}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-screen-2xl mx-auto w-full px-6 py-5">
          {isEmpty ? (
            <div className="flex flex-col items-center justify-center py-28 text-center">
              <div className="grid place-items-center w-16 h-16 rounded-2xl bg-muted/40 text-muted-foreground/40 mb-4">
                {isTelegram ? <Inbox className="w-8 h-8" /> : <History className="w-8 h-8" />}
              </div>
              <p className="text-sm font-medium text-muted-foreground/70">
                {isTelegram ? tr("telegram_links.empty", "Chưa có tài khoản nào liên kết") : t("audit.empty")}
              </p>
            </div>
          ) : (
            <div className="rounded-2xl border border-border/60 bg-card/30 overflow-hidden shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  {isTelegram ? (
                    <>
                      <thead>
                        <tr className="bg-muted/40 text-left text-[10.5px] uppercase tracking-wider text-muted-foreground/80">
                          <th className="px-5 py-3 font-semibold whitespace-nowrap">{tr("telegram_links.col_user", "Người dùng")}</th>
                          <th className="px-5 py-3 font-semibold whitespace-nowrap">{tr("telegram_links.col_telegram", "Telegram")}</th>
                          <th className="px-5 py-3 font-semibold whitespace-nowrap">{tr("telegram_links.col_chat", "Chat ID")}</th>
                          <th className="px-5 py-3 font-semibold whitespace-nowrap text-right">{tr("telegram_links.col_linked", "Liên kết lúc")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tgLinks.map((link) => {
                          const ts = formatParts(link.created_at);
                          return (
                            <tr
                              key={link.telegram_chat_id}
                              className="border-t border-border/40 hover:bg-muted/30 transition-colors"
                            >
                              <td className="px-5 py-3.5 whitespace-nowrap">
                                <div className="flex items-center gap-3">
                                  <div className="grid place-items-center w-9 h-9 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 text-primary text-[11px] font-bold ring-1 ring-primary/10 shrink-0">
                                    {initials(link.user_name)}
                                  </div>
                                  <div className="leading-tight">
                                    <div className="font-medium text-foreground">{link.user_name || "—"}</div>
                                    {link.user_email && (
                                      <div className="text-[11.5px] text-muted-foreground">{link.user_email}</div>
                                    )}
                                  </div>
                                </div>
                              </td>
                              <td className="px-5 py-3.5 whitespace-nowrap">
                                <div className="flex items-center gap-2.5">
                                  <div className="grid place-items-center w-7 h-7 rounded-lg bg-sky-500/10 text-sky-500 shrink-0">
                                    <Send className="w-3.5 h-3.5" />
                                  </div>
                                  <div className="leading-tight">
                                    <div className="font-medium text-foreground">
                                      {link.telegram_username ? `@${link.telegram_username}` : "—"}
                                    </div>
                                    {link.telegram_user_id && (
                                      <div className="text-[11.5px] text-muted-foreground tabular-nums">ID {link.telegram_user_id}</div>
                                    )}
                                  </div>
                                </div>
                              </td>
                              <td className="px-5 py-3.5 whitespace-nowrap">
                                <span className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-muted/60 font-mono text-[11.5px] font-medium text-muted-foreground tabular-nums">
                                  <Hash className="w-3 h-3 opacity-60" />
                                  {link.telegram_chat_id}
                                </span>
                              </td>
                              <td className="px-5 py-3.5 whitespace-nowrap text-right">
                                <div className="font-medium text-foreground/80 tabular-nums">{ts.date}</div>
                                <div className="text-[11.5px] text-muted-foreground tabular-nums">{ts.time}</div>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </>
                  ) : (
                    <>
                      <thead>
                        <tr className="bg-muted/40 text-left text-[10.5px] uppercase tracking-wider text-muted-foreground/80">
                          <th className="px-5 py-3 font-semibold whitespace-nowrap">{t("audit.col_time")}</th>
                          <th className="px-5 py-3 font-semibold whitespace-nowrap">{t("audit.col_actor")}</th>
                          <th className="px-5 py-3 font-semibold whitespace-nowrap">{t("audit.col_action")}</th>
                          <th className="px-5 py-3 font-semibold whitespace-nowrap">{t("audit.col_resource")}</th>
                          <th className="px-5 py-3 font-semibold">{t("audit.col_summary")}</th>
                          <th className="px-5 py-3 font-semibold whitespace-nowrap text-right">{t("audit.col_source")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {items.map((log) => {
                          const ts = formatParts(log.created_at);
                          return (
                            <tr
                              key={log.id}
                              className="border-t border-border/40 hover:bg-muted/30 transition-colors align-top"
                            >
                              <td className="px-5 py-3.5 whitespace-nowrap">
                                <div className="font-medium text-foreground/80 tabular-nums">{ts.date}</div>
                                <div className="text-[11.5px] text-muted-foreground tabular-nums">{ts.time}</div>
                              </td>
                              <td className="px-5 py-3.5 whitespace-nowrap">
                                <div className="flex items-center gap-3">
                                  <div className="grid place-items-center w-9 h-9 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 text-primary text-[11px] font-bold ring-1 ring-primary/10 shrink-0">
                                    {log.actor_name ? initials(log.actor_name) : <Cpu className="w-4 h-4" />}
                                  </div>
                                  <div className="leading-tight">
                                    <div className="font-medium text-foreground">
                                      {log.actor_name || t("audit.system")}
                                    </div>
                                    {log.actor_email && (
                                      <div className="text-[11.5px] text-muted-foreground">{log.actor_email}</div>
                                    )}
                                  </div>
                                </div>
                              </td>
                              <td className="px-5 py-3.5 whitespace-nowrap">
                                <span
                                  className={cn(
                                    "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-semibold",
                                    ACTION_STYLES[log.action] ||
                                      "bg-slate-500/10 text-slate-500 border-slate-500/20"
                                  )}
                                >
                                  <span className="w-1.5 h-1.5 rounded-full bg-current" />
                                  {actName(log.action)}
                                </span>
                              </td>
                              <td className="px-5 py-3.5 whitespace-nowrap">
                                <div className="leading-tight">
                                  <span className="inline-block px-2 py-0.5 rounded-md bg-muted/60 text-[11px] font-medium text-muted-foreground">
                                    {resName(log.resource_type)}
                                  </span>
                                  {log.resource_label && (
                                    <div
                                      className="mt-1 font-medium text-foreground max-w-[200px] truncate"
                                      title={log.resource_label}
                                    >
                                      {log.resource_label}
                                    </div>
                                  )}
                                </div>
                              </td>
                              <td className="px-5 py-3.5 text-foreground/80 max-w-[440px]">
                                <span className="break-words">{log.summary || "—"}</span>
                                {log.ip_address && (
                                  <span className="ml-2 inline-block font-mono text-[10px] text-muted-foreground/60">
                                    {log.ip_address}
                                  </span>
                                )}
                              </td>
                              <td className="px-5 py-3.5 whitespace-nowrap text-right">
                                <span
                                  className={cn(
                                    "inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium",
                                    log.source === "explicit"
                                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                                      : "bg-muted/60 text-muted-foreground"
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
                          );
                        })}
                      </tbody>
                    </>
                  )}
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer / pagination */}
      <div className="flex-shrink-0 border-t border-border/70 bg-card/40 backdrop-blur-sm px-6 py-3">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between text-xs text-muted-foreground">
          {isTelegram ? (
            <span className="tabular-nums">
              {total.toLocaleString()} {tr("telegram_links.total", "tài khoản liên kết")}
            </span>
          ) : (
            <>
              <span className="tabular-nums">
                {t("audit.page")} {page} <span className="text-muted-foreground/50">/ {totalPages}</span>
              </span>
              <div className="flex items-center gap-1.5">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1 rounded-lg"
                  disabled={page <= 1 || query.isFetching}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                >
                  <ChevronLeft className="w-4 h-4" /> {t("audit.prev")}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1 rounded-lg"
                  disabled={page >= totalPages || query.isFetching}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                >
                  {t("audit.next")} <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
