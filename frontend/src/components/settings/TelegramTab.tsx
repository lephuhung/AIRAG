/**
 * TelegramTab
 * ===========
 * Link/unlink Telegram chats to this account. The web mints a one-time code; the
 * user redeems it in the bot with `/start <code>` (or via the deep link). Once
 * linked, the bot chats with the user's exact permissions.
 */
import { useState } from "react";
import { Copy, Check, Loader2, Send, Trash2, ExternalLink } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useAuthStore } from "@/stores/authStore";
import { TelegramBotSetup } from "@/components/settings/TelegramBotSetup";
import {
  useCreateLinkCode,
  useTelegramLinks,
  useUnlinkTelegram,
} from "@/hooks/useIntegrations";
import { copyToClipboard } from "@/lib/clipboard";
import type { TelegramLinkCode, TelegramLinkInfo } from "@/types";

export function TelegramTab() {
  const isSuperadmin = useAuthStore((s) => s.user?.is_superadmin ?? false);
  const { data: links, isLoading } = useTelegramLinks();
  const createCode = useCreateLinkCode();
  const unlink = useUnlinkTelegram();

  const [code, setCode] = useState<TelegramLinkCode | null>(null);
  const [copied, setCopied] = useState(false);
  const [unlinkTarget, setUnlinkTarget] = useState<TelegramLinkInfo | null>(null);

  const handleGenerate = async () => {
    try {
      const c = await createCode.mutateAsync();
      setCode(c);
    } catch (err: any) {
      toast.error(err.message || "Failed to generate code");
    }
  };

  const handleCopy = async () => {
    if (!code) return;
    if (await copyToClipboard(`/start ${code.code}`)) {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } else {
      toast.error("Could not copy to clipboard");
    }
  };

  const handleUnlink = async () => {
    if (!unlinkTarget) return;
    try {
      await unlink.mutateAsync(unlinkTarget.telegram_chat_id);
      toast.success("Telegram chat unlinked");
    } catch (err: any) {
      toast.error(err.message || "Failed to unlink");
    } finally {
      setUnlinkTarget(null);
    }
  };

  return (
    <div className="space-y-4">
      {/* Admin-only: system-wide bot configuration */}
      {isSuperadmin && <TelegramBotSetup />}

      <p className="text-xs text-muted-foreground">
        Link a Telegram chat to ask the knowledge base from Telegram. Generate a
        code below, then send it to the bot as{" "}
        <code className="px-1 rounded bg-muted">/start &lt;code&gt;</code>.
      </p>

      {/* Generate code */}
      <Button size="sm" onClick={handleGenerate} disabled={createCode.isPending}>
        {createCode.isPending ? (
          <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
        ) : (
          <Send className="w-3.5 h-3.5 mr-1.5" />
        )}
        Generate linking code
      </Button>

      {code && (
        <div className="rounded-lg border border-primary/40 bg-primary/5 p-3 space-y-2">
          <p className="text-xs text-muted-foreground">
            Send this to the bot (valid for {code.ttl_minutes} minutes):
          </p>
          <div className="flex items-center gap-2">
            <code className="flex-1 px-2 py-1.5 text-sm font-semibold tracking-wider rounded bg-background border text-center">
              /start {code.code}
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
          {code.deep_link && (
            <a
              href={code.deep_link}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1.5 text-xs text-primary hover:underline"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              Open in Telegram
            </a>
          )}
        </div>
      )}

      {/* Linked chats */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">Linked chats</p>
        {isLoading ? (
          <div className="flex justify-center py-6">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : (links ?? []).length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-6">
            No Telegram chats linked yet.
          </p>
        ) : (
          links!.map((l) => (
            <div
              key={l.telegram_chat_id}
              className="flex items-center gap-3 rounded-lg border px-3 py-2"
            >
              <Send className="w-4 h-4 text-muted-foreground shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">
                  {l.telegram_username ? `@${l.telegram_username}` : "Telegram chat"}
                </p>
                <p className="text-xs text-muted-foreground truncate">
                  chat {l.telegram_chat_id} · linked{" "}
                  {new Date(l.created_at).toLocaleDateString()}
                </p>
              </div>
              <button
                onClick={() => setUnlinkTarget(l)}
                className="rounded-md p-1.5 hover:bg-destructive/10 text-destructive transition-colors shrink-0"
                title="Unlink"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))
        )}
      </div>

      <ConfirmDialog
        open={unlinkTarget !== null}
        onConfirm={handleUnlink}
        onCancel={() => setUnlinkTarget(null)}
        title="Unlink Telegram"
        message="Unlink this chat? It will no longer be able to query the knowledge base until re-linked."
        confirmLabel="Unlink"
        variant="danger"
      />
    </div>
  );
}
