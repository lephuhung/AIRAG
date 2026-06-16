/**
 * ApiKeysTab
 * ==========
 * Lists, creates and revokes third-party API keys (used by Telegram bot backends,
 * Zalo, n8n, Slack, ...). A freshly created key's plaintext is shown exactly once.
 */
import { useState } from "react";
import { Copy, KeyRound, Loader2, Plus, Trash2, Check } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import {
  useApiKeys,
  useCreateApiKey,
  useRevokeApiKey,
} from "@/hooks/useIntegrations";
import type { ApiKeyInfo } from "@/types";

export function ApiKeysTab() {
  const { data: keys, isLoading } = useApiKeys();
  const createKey = useCreateApiKey();
  const revokeKey = useRevokeApiKey();

  const [name, setName] = useState("");
  const [newKey, setNewKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [revokeTarget, setRevokeTarget] = useState<ApiKeyInfo | null>(null);

  const handleCreate = async () => {
    if (!name.trim()) {
      toast.error("Please enter a name for the key");
      return;
    }
    try {
      const created = await createKey.mutateAsync({ name: name.trim() });
      setNewKey(created.key);
      setName("");
      toast.success("API key created");
    } catch (err: any) {
      toast.error(err.message || "Failed to create API key");
    }
  };

  const handleCopy = async () => {
    if (!newKey) return;
    try {
      await navigator.clipboard.writeText(newKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Could not copy to clipboard");
    }
  };

  const handleRevoke = async () => {
    if (!revokeTarget) return;
    try {
      await revokeKey.mutateAsync(revokeTarget.id);
      toast.success("API key revoked");
    } catch (err: any) {
      toast.error(err.message || "Failed to revoke key");
    } finally {
      setRevokeTarget(null);
    }
  };

  const activeKeys = (keys ?? []).filter((k) => !k.revoked);

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        API keys authenticate third-party clients (e.g. a Telegram bot backend, n8n,
        Zalo) as your account. Send the key in the{" "}
        <code className="px-1 rounded bg-muted">X-API-Key</code> header.
      </p>

      {/* Create */}
      <div className="flex items-end gap-2">
        <div className="flex-1 space-y-1.5">
          <label className="text-xs font-medium text-muted-foreground">
            New key name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            placeholder="e.g. Telegram bot"
            className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
          />
        </div>
        <Button size="sm" onClick={handleCreate} disabled={createKey.isPending}>
          {createKey.isPending ? (
            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
          ) : (
            <Plus className="w-3.5 h-3.5 mr-1.5" />
          )}
          Create
        </Button>
      </div>

      {/* One-time plaintext key reveal */}
      {newKey && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 space-y-2">
          <p className="text-xs font-medium text-amber-700 dark:text-amber-400">
            Copy this key now — it will not be shown again.
          </p>
          <div className="flex items-center gap-2">
            <code className="flex-1 px-2 py-1.5 text-xs rounded bg-background border break-all">
              {newKey}
            </code>
            <button
              onClick={handleCopy}
              className="rounded-md p-1.5 hover:bg-muted transition-colors shrink-0"
              title="Copy"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-600" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>
          </div>
          <button
            onClick={() => setNewKey(null)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* List */}
      <div className="space-y-2">
        {isLoading ? (
          <div className="flex justify-center py-6">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : activeKeys.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-6">
            No API keys yet.
          </p>
        ) : (
          activeKeys.map((k) => (
            <div
              key={k.id}
              className="flex items-center gap-3 rounded-lg border px-3 py-2"
            >
              <KeyRound className="w-4 h-4 text-muted-foreground shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{k.name}</p>
                <p className="text-xs text-muted-foreground truncate">
                  <code>{k.prefix}…</code> · created{" "}
                  {new Date(k.created_at).toLocaleDateString()}
                  {k.last_used_at
                    ? ` · last used ${new Date(k.last_used_at).toLocaleDateString()}`
                    : " · never used"}
                </p>
              </div>
              <button
                onClick={() => setRevokeTarget(k)}
                className="rounded-md p-1.5 hover:bg-destructive/10 text-destructive transition-colors shrink-0"
                title="Revoke"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))
        )}
      </div>

      <ConfirmDialog
        open={revokeTarget !== null}
        onConfirm={handleRevoke}
        onCancel={() => setRevokeTarget(null)}
        title="Revoke API key"
        message={`Revoke "${revokeTarget?.name}"? Any client using it will stop working immediately.`}
        confirmLabel="Revoke"
        variant="danger"
      />
    </div>
  );
}
