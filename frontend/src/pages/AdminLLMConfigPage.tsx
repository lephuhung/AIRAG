import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  Bot,
  Brain,
  Eye,
  Layers,
  MessageSquare,
  Network,
  Mic,
  Volume2,
  Database,
  ArrowUpDown,
  Loader2,
  RotateCcw,
  Save,
  PlugZap,
  Cpu,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Cable,
  Plus,
  Pencil,
  Trash2,
  KeyRound,
  ChevronDown,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { cn } from "@/lib/utils";
import { useTranslation } from "@/hooks/useTranslation";
import {
  useLlmConfig,
  useUpdateLlmConfig,
  useDeleteLlmConfigOverride,
  useSaveConnection,
  useDeleteConnection,
} from "@/hooks/useLlmConfig";
import { api } from "@/lib/api";
import type {
  ConnectionInfo,
  LlmRole,
  LlmTestResult,
  RoleAssignmentStatus,
} from "@/types/llmConfig";

// ── Shared helpers ───────────────────────────────────────────────────────────

/** Slugify a connection display name into a stable conn_id. */
function slugify(input: string): string {
  return input
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
}

const PROVIDERS = [
  { value: "openai_compatible", label: "OpenAI-compatible (vLLM / DeepSeek / OpenRouter…)" },
  { value: "gemini", label: "Google Gemini" },
  { value: "ollama", label: "Ollama (local)" },
];

/** Quick presets — ONLY prefill the form, never save. */
const PRESETS = [
  { label: "Local vLLM", provider: "openai_compatible", baseUrl: "" },
  { label: "Google Gemini", provider: "gemini", baseUrl: "" },
  { label: "Ollama Local", provider: "ollama", baseUrl: "http://localhost:11434" },
  {
    label: "DeepSeek",
    provider: "openai_compatible",
    baseUrl: "https://api.deepseek.com/v1",
  },
  { label: "OpenAI-compatible", provider: "openai_compatible", baseUrl: "" },
];

// ── Role metadata ────────────────────────────────────────────────────────────

interface RoleMeta {
  key: LlmRole;
  icon: typeof MessageSquare;
  accent: string;
}

const ROLE_META: RoleMeta[] = [
  { key: "main", icon: MessageSquare, accent: "text-sky-500 bg-sky-500/10" },
  { key: "vision", icon: Eye, accent: "text-violet-500 bg-violet-500/10" },
  { key: "thinking", icon: Brain, accent: "text-amber-500 bg-amber-500/10" },
  { key: "memory_agent", icon: Bot, accent: "text-emerald-500 bg-emerald-500/10" },
  { key: "kg_extract", icon: Network, accent: "text-rose-500 bg-rose-500/10" },
  { key: "graphiti", icon: Layers, accent: "text-teal-500 bg-teal-500/10" },
  { key: "stt", icon: Mic, accent: "text-cyan-500 bg-cyan-500/10" },
  { key: "tts", icon: Volume2, accent: "text-fuchsia-500 bg-fuchsia-500/10" },
  { key: "embedding", icon: Database, accent: "text-indigo-500 bg-indigo-500/10" },
  { key: "rerank", icon: ArrowUpDown, accent: "text-orange-500 bg-orange-500/10" },
];

// ── Model catalogue cache (shared between sections) ────────────────────────

interface CatalogEntry {
  models: string[];
  failed: boolean;
}

function useModelCatalog() {
  const [catalog, setCatalog] = useState<Record<string, CatalogEntry>>({});
  const inflight = useRef<Set<string>>(new Set());

  const load = useCallback(async (connId: string) => {
    if (inflight.current.has(connId)) return;
    inflight.current.add(connId);
    try {
      const res = await api.listLlmModelsByConn(connId);
      setCatalog((c) => ({
        ...c,
        [connId]: {
          models: res.ok ? res.models : [],
          failed: !res.ok || res.models.length === 0,
        },
      }));
    } catch {
      setCatalog((c) => ({ ...c, [connId]: { models: [], failed: true } }));
    } finally {
      inflight.current.delete(connId);
    }
  }, []);

  return { catalog, load };
}

// ── Connection form modal (create / edit) ───────────────────────────────────

interface ConnForm {
  name: string;
  provider: string;
  baseUrl: string;
  apiKey: string;
  isVllm: boolean;
  /** Service kind for non-OpenAI endpoints (embed_rerank / stt). */
  kind: string;
}

function ConnectionFormModal({
  editId,
  editConn,
  onClose,
}: {
  editId: string | null;
  editConn: ConnectionInfo | null;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const saveMutation = useSaveConnection();

  const [form, setForm] = useState<ConnForm>({
    name: editConn?.name ?? "",
    provider: editConn?.provider ?? "openai_compatible",
    baseUrl: editConn?.base_url ?? "",
    apiKey: "",
    isVllm: Boolean(editConn?.extra?.is_vllm),
    kind: String(editConn?.extra?.kind ?? ""),
  });
  const [models, setModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsFailed, setModelsFailed] = useState(false);
  const [testResult, setTestResult] = useState<LlmTestResult | null>(null);
  const [testing, setTesting] = useState(false);

  // AUTO-LOAD catalogue: debounced 600ms once url (+ key when needed) look valid.
  useEffect(() => {
    const url = form.baseUrl.trim();
    const key = form.apiKey.trim();
    const needsKey = form.provider !== "ollama"; // Ollama needs no auth
    if (!url || (needsKey && !key)) return;

    const timer = setTimeout(async () => {
      setModelsLoading(true);
      try {
        const res = await api.listLlmModels({
          provider: form.provider,
          base_url: url,
          api_key: key || undefined,
        });
        if (res.ok && res.models.length > 0) {
          setModels(res.models);
          setModelsFailed(false);
        } else {
          setModels([]);
          setModelsFailed(true);
        }
      } catch {
        setModels([]);
        setModelsFailed(true);
      } finally {
        setModelsLoading(false);
      }
    }, 600);
    return () => clearTimeout(timer);
  }, [form.provider, form.baseUrl, form.apiKey]);

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await api.testLlmConfig({
        provider: form.provider,
        base_url: form.baseUrl.trim(),
        api_key: form.apiKey.trim(),
        is_vllm: form.isVllm,
        kind: form.kind,
      });
      setTestResult(res);
      if (res.ok && res.models_list_available && res.models.length > 0) {
        setModels(res.models);
        setModelsFailed(false);
      }
    } catch (err) {
      setTestResult({
        ok: false,
        models: [],
        models_list_available: false,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setTesting(false);
    }
  };

  const canSave =
    !!form.name.trim() &&
    !!form.baseUrl.trim() &&
    testResult?.ok === true &&
    !saveMutation.isPending;

  const handleSave = async () => {
    const connId =
      editId ?? (slugify(form.name) || `conn-${Date.now().toString(36)}`);
    try {
      await saveMutation.mutateAsync({
        connId,
        data: {
          name: form.name.trim(),
          provider: form.provider,
          base_url: form.baseUrl.trim(),
          ...(form.apiKey.trim() ? { api_key: form.apiKey.trim() } : {}),
          extra: { ...(editConn?.extra ?? {}), is_vllm: form.isVllm, ...(form.kind ? { kind: form.kind } : {}) },
        },
      });
      toast.success(t("admin.llm.v2.conn_saved"));
      onClose();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("admin.llm.toast.save_failed"));
    }
  };

  const inputCls =
    "w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative bg-card border rounded-xl shadow-xl w-full max-w-lg max-h-[90vh] overflow-y-auto p-5">
        <h3 className="font-semibold text-sm mb-4">
          {editId
            ? t("admin.llm.v2.edit_conn")
            : t("admin.llm.v2.add_conn")}
        </h3>

        {/* Quick presets */}
        {!editId && (
          <div className="flex flex-wrap items-center gap-1.5 mb-4">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mr-1">
              {t("admin.llm.presets.label")}
            </span>
            {PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() =>
                  setForm((f) => ({
                    ...f,
                    provider: p.provider,
                    baseUrl: p.baseUrl || f.baseUrl,
                  }))
                }
                className="text-xs px-2 py-1 rounded-md border bg-background hover:bg-muted/60 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        )}

        <div className="space-y-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">
              {t("admin.llm.v2.name")}
            </label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
              placeholder="DeepSeek, vLLM nội bộ, …"
              className={inputCls}
            />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">
              {t("admin.llm.provider")}
            </label>
            <select
              value={form.provider}
              onChange={(e) => {
                setForm((f) => ({ ...f, provider: e.target.value }));
                setTestResult(null);
              }}
              className={inputCls}
            >
              {PROVIDERS.map((p) => (
                <option key={p.value} value={p.value}>
                  {p.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">
              {t("admin.llm.base_url")}
            </label>
            <input
              type="text"
              value={form.baseUrl}
              onChange={(e) => {
                setForm((f) => ({ ...f, baseUrl: e.target.value }));
                setTestResult(null);
              }}
              placeholder={
                form.provider === "ollama"
                  ? "http://localhost:11434"
                  : "https://api.deepseek.com/v1"
              }
              className={cn(inputCls, "font-mono")}
            />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground block mb-1">
              {t("admin.llm.api_key")}
            </label>
            <input
              type="password"
              value={form.apiKey}
              onChange={(e) => {
                setForm((f) => ({ ...f, apiKey: e.target.value }));
                setTestResult(null);
              }}
              placeholder={
                editConn?.has_api_key && editConn.masked_api_key
                  ? `${editConn.masked_api_key} (${t("admin.llm.api_key.stored")})`
                  : form.provider === "ollama"
                    ? t("admin.llm.api_key.not_needed")
                    : t("common.optional")
              }
              autoComplete="new-password"
              className={inputCls}
            />
          </div>

          {/* Toggle: server is a vLLM instance (gates chat_template_kwargs) */}
          {form.provider === "openai_compatible" && (
            <label className="flex items-start gap-2 cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-border mt-0.5"
                checked={form.isVllm}
                onChange={(e) => {
                  setForm((f) => ({ ...f, isVllm: e.target.checked }));
                  setTestResult(null);
                }}
              />
              <span className="text-xs">
                <span className="font-medium">{t("admin.llm.v2.is_vllm")}</span>
                <span className="block text-muted-foreground mt-0.5">
                  {t("admin.llm.v2.is_vllm_hint")}
                </span>
              </span>
            </label>
          )}
        </div>

        {/* Auto-loaded catalogue preview */}
        <div className="mt-3 min-h-[20px]">
          {modelsLoading ? (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <Loader2 className="w-3 h-3 animate-spin" />
              {t("admin.llm.v2.loading_models")}
            </span>
          ) : modelsFailed ? (
            <span className="inline-flex items-center gap-1 text-[11px] text-amber-600 dark:text-amber-400">
              <AlertTriangle className="w-3 h-3" />
              {t("admin.llm.models.failed_badge")}
            </span>
          ) : models.length > 0 ? (
            <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              <CheckCircle2 className="w-3 h-3 text-emerald-500" />
              {t("admin.llm.models.loaded", { count: models.length })}
              <span className="font-mono text-[10px] opacity-70 truncate max-w-[240px]">
                {models.slice(0, 3).join(", ")}
                {models.length > 3 ? ", …" : ""}
              </span>
            </span>
          ) : null}
        </div>

        {/* Actions */}
        <div className="flex items-center justify-between gap-2 border-t pt-3 mt-3">
          <Button
            size="sm"
            variant="outline"
            onClick={handleTest}
            disabled={testing || !form.baseUrl.trim()}
          >
            {testing ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin mr-1" />
            ) : (
              <PlugZap className="w-3.5 h-3.5 mr-1" />
            )}
            {t("admin.llm.test.button")}
          </Button>
          <div className="flex items-center gap-2">
            <Button size="sm" variant="ghost" onClick={onClose}>
              {t("common.cancel")}
            </Button>
            <Button size="sm" onClick={handleSave} disabled={!canSave}>
              {saveMutation.isPending ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin mr-1" />
              ) : (
                <Save className="w-3.5 h-3.5 mr-1" />
              )}
              {t("common.save")}
            </Button>
          </div>
        </div>

        {/* Test result */}
        {testResult && (
          <div
            className={cn(
              "mt-3 text-xs rounded-lg px-3 py-2 border flex items-start gap-2",
              testResult.ok
                ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-700 dark:text-emerald-400"
                : "bg-destructive/10 border-destructive/20 text-destructive",
            )}
          >
            {testResult.ok ? (
              <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            ) : (
              <XCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
            )}
            <span className="min-w-0 break-words">
              {testResult.ok
                ? t("admin.llm.test.ok", {
                    latency: Math.round(testResult.latency_ms ?? 0),
                    count: testResult.models?.length ?? 0,
                  })
                : t("admin.llm.test.fail", {
                    error: testResult.error || "unknown",
                  })}
            </span>
          </div>
        )}
        {testResult?.ok === true && (
          <p className="mt-2 text-[11px] text-muted-foreground">
            {t("admin.llm.test.required_hint")}
          </p>
        )}
      </div>
    </div>
  );
}

// ── Connection card ──────────────────────────────────────────────────────────

function ConnectionCard({
  connId,
  conn,
  catalog,
  referencedBy,
  onEdit,
  reloadCatalog,
}: {
  connId: string;
  conn: ConnectionInfo;
  catalog?: CatalogEntry;
  referencedBy: LlmRole[];
  onEdit: () => void;
  reloadCatalog: (id: string) => void;
}) {
  const { t } = useTranslation();
  const deleteMutation = useDeleteConnection();

  const [confirmDelete, setConfirmDelete] = useState(false);
  const [testState, setTestState] = useState<
    { loading: boolean; ok: boolean | null }
  >({ loading: false, ok: null });

  const handleTest = async () => {
    setTestState({ loading: true, ok: null });
    try {
      // Saved connections are probed via conn_id so the server resolves the
      // stored credentials itself (plaintext never leaves the backend).
      const res = await api.listLlmModelsByConn(connId);
      setTestState({ loading: false, ok: res.ok && res.models.length > 0 });
      if (res.ok) reloadCatalog(connId);
    } catch {
      setTestState({ loading: false, ok: false });
    }
  };

  const handleDelete = async (force: boolean) => {
    try {
      const res = await deleteMutation.mutateAsync({ connId, force });
      if (!res.deleted && res.referencing_roles.length > 0) {
        toast.error(
          t("admin.llm.v2.conn_delete_blocked", {
            roles: res.referencing_roles.join(", "),
          }),
        );
        setConfirmDelete(false);
        return;
      }
      toast.success(t("admin.llm.v2.conn_deleted"));
      setConfirmDelete(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("admin.llm.v2.delete_failed"));
      setConfirmDelete(false);
    }
  };

  return (
    <div className="rounded-xl border bg-card p-4">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
            <Cable className="w-4 h-4 text-primary" />
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-sm truncate">{conn.name}</h3>
            <span className="inline-block text-[10px] px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground font-medium mt-0.5">
              {conn.provider}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1 flex-shrink-0">
          <Button
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0"
            onClick={onEdit}
            title={t("admin.llm.v2.edit_conn")}
          >
            <Pencil className="w-3.5 h-3.5" />
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 w-7 p-0 text-destructive hover:bg-destructive/10"
            onClick={() => setConfirmDelete(true)}
            title={t("admin.llm.v2.delete_conn")}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      <p className="text-xs font-mono text-muted-foreground truncate">{conn.base_url}</p>
      <p className="text-[11px] text-muted-foreground mt-1 inline-flex items-center gap-1">
        <KeyRound className="w-3 h-3" />
        {conn.has_api_key && conn.masked_api_key
          ? conn.masked_api_key
          : t("admin.llm.api_key.not_needed")}
      </p>
      {referencedBy.length > 0 && (
        <p className="text-[10px] text-muted-foreground/80 mt-1">
          {t("admin.llm.v2.used_by", { roles: referencedBy.join(", ") })}
        </p>
      )}

      <div className="flex items-center justify-between gap-2 mt-3 pt-2 border-t">
        <Button
          size="sm"
          variant="outline"
          className="h-7 text-xs"
          onClick={handleTest}
          disabled={testState.loading}
        >
          {testState.loading ? (
            <Loader2 className="w-3 h-3 animate-spin mr-1" />
          ) : (
            <PlugZap className="w-3 h-3 mr-1" />
          )}
          {t("admin.llm.test.button")}
        </Button>
        {testState.ok === true && (
          <span className="inline-flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="w-3 h-3" />
            OK{catalog?.models.length ? ` · ${catalog.models.length}` : ""}
          </span>
        )}
        {testState.ok === false && (
          <span className="inline-flex items-center gap-1 text-[11px] text-destructive">
            <XCircle className="w-3 h-3" />
            {t("admin.llm.test.fail_short")}
          </span>
        )}
      </div>

      <ConfirmDialog
        open={confirmDelete}
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => handleDelete(false)}
        title={t("admin.llm.v2.delete_confirm_title")}
        message={t("admin.llm.v2.delete_confirm_msg", { name: conn.name })}
        confirmLabel={t("admin.llm.v2.delete_conn")}
        variant="danger"
      />
    </div>
  );
}

// ── Assignment row (Section 2) ──────────────────────────────────────────────

function AssignmentRow({
  meta,
  status,
  connections,
  catalog,
  loadCatalog,
}: {
  meta: RoleMeta;
  status: RoleAssignmentStatus;
  connections: Record<string, ConnectionInfo>;
  catalog: Record<string, CatalogEntry>;
  loadCatalog: (id: string) => void;
}) {
  const { t } = useTranslation();
  const updateMutation = useUpdateLlmConfig();
  // MUST go through this mutation (not a bare api call) so the react-query
  // cache is invalidated — otherwise the row keeps stale server state after
  // reset and re-selection appears frozen.
  const deleteMutation = useDeleteLlmConfigOverride();

  // Single-selector state: combobox lists "model" entries grouped by
  // connection; picking one sets BOTH the connection target and the model.
  const [connId, setConnId] = useState(
    status.conn_id && status.conn_id !== "@env" ? status.conn_id : "",
  );
  const [model, setModel] = useState(
    status.source === "db" ? status.model : "",
  );
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState("");

  // Preload catalogues for every connection so the selector lists all models.
  useEffect(() => {
    Object.keys(connections).forEach((cid) => {
      if (!catalog[cid]) loadCatalog(cid);
    });
  }, [connections, catalog, loadCatalog]);

  const Icon = meta.icon;
  const savedConnId =
    status.conn_id && status.conn_id !== "@env" ? status.conn_id : "";
  const savedModel = status.source === "db" ? status.model : "";
  const dirty = connId !== savedConnId || model !== savedModel;

  // NOTE: server-side changes (save/reset) arrive via the parent's remount
  // key (`updated_at`) — no local sync effect needed here.

  // Grouped model catalogue. When a connection declares `default_models`,
  // ONLY those are offered (the live catalogue may contain unrelated aliases
  // like tts-1/tts-1-hd that have no real backing). Otherwise the full live
  // catalogue is shown; local services with neither fall back to free-text.
  const groups = useMemo(() => {
    return Object.entries(connections).map(([cid, c]) => {
      const cat = catalog[cid];
      const defaults = (c.extra?.default_models as string[] | undefined) ?? [];
      const models = defaults.length
        ? defaults
        : cat?.models?.length
          ? cat.models
          : [];
      return {
        cid,
        name: c.name as string,
        models,
      };
    });
  }, [connections, catalog]);

  const applyPick = (cid: string, m: string) => {
    setConnId(cid);
    setModel(m);
    setOpen(false);
    setFilter("");
  };

  const handleSave = async () => {
    try {
      await updateMutation.mutateAsync({
        role: meta.key,
        data: { conn_id: connId, model: model.trim() },
      });
      toast.success(t("admin.llm.v2.assigned"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("admin.llm.toast.save_failed"));
    }
  };

  const handleReset = async () => {
    try {
      await deleteMutation.mutateAsync(meta.key);
      toast.success(t("admin.llm.toast.reset"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("admin.llm.toast.reset_failed"));
    }
  };

  const inputCls =
    "w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30";

  return (
    <div className="border-b last:border-b-0 px-4 py-3">
      <div className="flex items-center gap-3 flex-wrap lg:flex-nowrap">
        {/* Role identity */}
        <div className="flex items-center gap-2.5 min-w-[180px] flex-1">
          <div
            className={cn(
              "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0",
              meta.accent,
            )}
          >
            <Icon className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-xs flex items-center gap-1.5">
              {t(`admin.llm.role.${meta.key}.title`)}
              {status.source === "db" ? (
                <span className="text-[9px] px-1 py-0.5 rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-medium">
                  {t("admin.llm.badge.override")}
                </span>
              ) : (
                <span className="text-[9px] px-1 py-0.5 rounded-full bg-muted text-muted-foreground font-medium">
                  {t("admin.llm.badge.env")}
                </span>
              )}
            </h3>
            <p className="text-[11px] text-muted-foreground truncate">
              {t(`admin.llm.role.${meta.key}.desc`)}
            </p>
          </div>
        </div>

        {/* Single selector with a themed dropdown panel (native datalist
            popups can't be styled). Typing filters; picking assigns both the
            connection and the model. */}
        <div
          className="min-w-[210px] w-full lg:w-[300px] relative"
          onBlur={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node)) {
              // Free-typed text commits to the current connection on leave.
              if (filter.trim() && connId) setModel(filter.trim());
              setOpen(false);
              setFilter("");
            }
          }}
        >
          <div className="relative">
            <input
              list={undefined}
              autoComplete="off"
              value={open ? filter : connId && model ? `${model} · ${connections[connId]?.name ?? connId}` : ""}
              onFocus={() => { setOpen(true); setFilter(""); }}
              onChange={(e) => setFilter(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") { setOpen(false); (e.target as HTMLInputElement).blur(); }
                if (e.key === "Enter") {
                  const q = filter.toLowerCase();
                  let picked = false;
                  for (const g of groups) {
                    const hit = g.models.find((m) => m.toLowerCase().includes(q));
                    if (hit) { applyPick(g.cid, hit); picked = true; break; }
                  }
                  // No catalogue match → commit typed text to current conn.
                  if (!picked && filter.trim() && connId) {
                    applyPick(connId, filter.trim());
                  }
                }
              }}
              placeholder={t("admin.llm.v2.pick_model")}
              className={cn(
                inputCls,
                "h-8 px-2.5 pr-7 text-xs rounded-md cursor-pointer hover:border-primary/40",
              )}
            />
            <button
              type="button"
              tabIndex={-1}
              onClick={() => { setOpen((o) => !o); setFilter(""); }}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <ChevronDown className={cn("w-3.5 h-3.5 transition-transform", open && "rotate-180")} />
            </button>
          </div>
          {open && (
            <div className="absolute z-50 mt-1 w-full max-h-64 overflow-auto rounded-md border border-input bg-background shadow-md py-1">
              {groups.map((g) => {
                const q = filter.toLowerCase();
                const shown = g.models.filter((m) => m.toLowerCase().includes(q));
                if (!shown.length && q) return null;
                return (
                  <div key={g.cid}>
                    <div className="px-2.5 pt-1.5 pb-0.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground truncate">
                      {g.name}
                    </div>
                    {shown.length === 0 ? (
                      <button
                        key={`${g.cid}-empty`}
                        type="button"
                        onMouseDown={(e) => { e.preventDefault(); applyPick(g.cid, ""); }}
                        className="w-full text-left px-2.5 py-1.5 text-xs text-muted-foreground hover:bg-accent"
                      >
                        {t("admin.llm.v2.type_model")}
                      </button>
                    ) : (
                      shown.map((m) => {
                        const active = connId === g.cid && model === m;
                        return (
                          <button
                            key={`${g.cid}-${m}`}
                            type="button"
                            title={`${m} · ${g.name}`}
                            onMouseDown={(e) => { e.preventDefault(); applyPick(g.cid, m); }}
                            className={cn(
                              "w-full text-left px-2.5 py-1.5 text-xs flex items-center justify-between gap-2",
                              active ? "bg-primary/10 text-primary" : "hover:bg-accent",
                            )}
                          >
                            <span className="truncate">{m}</span>
                            {active && <CheckCircle2 className="w-3 h-3 shrink-0" />}
                          </button>
                        );
                      })
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {connId && !model.trim() ? (
            <p className="text-[10px] text-muted-foreground mt-0.5 truncate">
              {t("admin.llm.v2.type_model")}
            </p>
          ) : null}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 ml-auto">
          {status.source === "db" && (
            <Button
              size="sm"
              variant="ghost"
              className="h-8 w-8 p-0 text-destructive hover:bg-destructive/10"
              onClick={handleReset}
              title={t("admin.llm.reset")}
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </Button>
          )}
          <Button
            size="sm"
            className="h-8"
            disabled={!dirty || updateMutation.isPending || !connId || !model.trim()}
            onClick={handleSave}
          >
            {updateMutation.isPending ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin mr-1" />
            ) : (
              <Save className="w-3.5 h-3.5 mr-1" />
            )}
            {t("common.save")}
          </Button>
        </div>
      </div>

      {/* Restart / reindex warnings */}
      {meta.key === "embedding" && (
        <p className="mt-2 text-[11px] rounded-lg bg-red-500/10 border border-red-500/20 text-red-700 dark:text-red-400 px-3 py-1.5 inline-flex items-start gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          {t("admin.llm.v2.banner_embedding")}
        </p>
      )}
      {meta.key === "rerank" && (
        <p className="mt-2 text-[11px] rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-700 dark:text-amber-400 px-3 py-1.5 inline-flex items-start gap-1.5">
          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          {t("admin.llm.v2.banner_rerank")}
        </p>
      )}
    </div>
  );
}

function catalogLoading(entry?: CatalogEntry): boolean {
  return entry === undefined;
}

// ── Page ─────────────────────────────────────────────────────────────────────

export function AdminLLMConfigPage() {
  const { t } = useTranslation();
  const { data, isLoading, isError, error } = useLlmConfig();
  const { catalog, load } = useModelCatalog();

  const [modal, setModal] = useState<
    | { mode: "create" }
    | { mode: "edit"; connId: string; conn: ConnectionInfo }
    | null
  >(null);

  // Roles currently referencing each connection (for badges + delete guard).
  const refsByConn = useMemo(() => {
    const map: Record<string, LlmRole[]> = {};
    if (!data) return map;
    for (const [role, st] of Object.entries(data.roles) as [LlmRole, RoleAssignmentStatus][]) {
      if (st.conn_id && st.conn_id !== "@env") {
        (map[st.conn_id] ??= []).push(role);
      }
    }
    return map;
  }, [data]);

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Cpu className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold">{t("admin.llm.title")}</h1>
              <p className="text-sm text-muted-foreground">{t("admin.llm.subtitle")}</p>
            </div>
          </div>
          {data && (
            <span className="text-xs text-muted-foreground font-mono">
              v{data.version}
            </span>
          )}
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : isError || !data ? (
          <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
            <AlertTriangle className="w-10 h-10 mb-3 opacity-30" />
            <p className="text-sm">
              {error instanceof Error ? error.message : t("admin.llm.load_failed")}
            </p>
          </div>
        ) : (
          <>
            {/* SECTION 1 — Connections */}
            <div className="mb-3 flex items-center justify-between">
              <div>
                <h2 className="font-semibold text-sm flex items-center gap-2">
                  <Cable className="w-4 h-4 text-primary" />
                  {t("admin.llm.v2.conn_title")}
                </h2>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {t("admin.llm.v2.conn_desc")}
                </p>
              </div>
              <Button size="sm" onClick={() => setModal({ mode: "create" })}>
                <Plus className="w-3.5 h-3.5 mr-1" />
                {t("admin.llm.v2.add_conn")}
              </Button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 mb-8">
              {Object.entries(data.connections).length === 0 && (
                <p className="text-xs text-muted-foreground border border-dashed rounded-xl px-4 py-6 col-span-full text-center">
                  {t("admin.llm.v2.no_conns")}
                </p>
              )}
              {Object.entries(data.connections).map(([cid, conn]) => (
                <ConnectionCard
                  key={cid}
                  connId={cid}
                  conn={conn}
                  catalog={catalog[cid]}
                  referencedBy={refsByConn[cid] ?? []}
                  onEdit={() => setModal({ mode: "edit", connId: cid, conn })}
                  reloadCatalog={load}
                />
              ))}
            </div>

            {/* SECTION 2 — Assignments */}
            <div className="mb-3">
              <h2 className="font-semibold text-sm flex items-center gap-2">
                <Bot className="w-4 h-4 text-primary" />
                {t("admin.llm.v2.assign_title")}
              </h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                {t("admin.llm.v2.assign_desc")}
              </p>
            </div>
            <div className="rounded-xl border bg-card overflow-hidden">
              {ROLE_META.map((meta) => (
                <AssignmentRow
                  key={`${meta.key}-${data.roles[meta.key]?.updated_at ?? "env"}`}
                  meta={meta}
                  status={
                    data.roles[meta.key] ?? {
                      conn_id: "@env",
                      model: "",
                      source: "env",
                      resolved: { provider: "", base_url: "", model: "" },
                    }
                  }
                  connections={data.connections}
                  catalog={catalog}
                  loadCatalog={load}
                />
              ))}
            </div>
          </>
        )}
      </div>

      {modal && (
        <ConnectionFormModal
          editId={modal.mode === "edit" ? modal.connId : null}
          editConn={modal.mode === "edit" ? modal.conn : null}
          onClose={() => setModal(null)}
        />
      )}
    </div>
  );
}
