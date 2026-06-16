/**
 * TelegramBotSetup (superadmin only)
 * ==================================
 * System-wide Telegram bot configuration done entirely from the UI — no .env.
 * Lets a superadmin paste the bot token (validated via getMe), set/confirm the
 * webhook URL, register the webhook with Telegram, and test the connection.
 */
import { useEffect, useState } from "react";
import { Loader2, Save, Webhook, Plug, CheckCircle2, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  useTelegramConfig,
  useUpdateTelegramConfig,
  useRegisterTelegramWebhook,
  useTestTelegramConnection,
} from "@/hooks/useIntegrations";

export function TelegramBotSetup() {
  const { data: config, isLoading } = useTelegramConfig();
  const update = useUpdateTelegramConfig();
  const register = useRegisterTelegramWebhook();
  const test = useTestTelegramConnection();

  const [token, setToken] = useState("");
  const [webhookUrl, setWebhookUrl] = useState("");
  const [enabled, setEnabled] = useState(true);
  const [result, setResult] = useState<{ ok: boolean; detail: string } | null>(null);

  // Sync local form when the config loads / changes.
  useEffect(() => {
    if (!config) return;
    setWebhookUrl(config.webhook_url || config.suggested_webhook_url || "");
    setEnabled(config.enabled);
  }, [config]);

  const handleSave = async () => {
    try {
      await update.mutateAsync({
        ...(token.trim() ? { bot_token: token.trim() } : {}),
        webhook_url: webhookUrl.trim(),
        enabled,
      });
      setToken("");
      toast.success("Đã lưu cấu hình bot");
    } catch (err: any) {
      toast.error(err.message || "Lưu cấu hình thất bại");
    }
  };

  const handleRegister = async () => {
    try {
      const res = await register.mutateAsync();
      setResult(res);
      res.ok ? toast.success("Đã đăng ký webhook") : toast.error("Đăng ký webhook thất bại");
    } catch (err: any) {
      toast.error(err.message || "Đăng ký webhook thất bại");
    }
  };

  const handleTest = async () => {
    try {
      const res = await test.mutateAsync();
      setResult(res);
    } catch (err: any) {
      toast.error(err.message || "Kiểm tra kết nối thất bại");
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-6">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-muted/20 p-3 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Cấu hình Bot (Admin)
        </p>
        {config?.bot_username && (
          <span className="text-xs text-muted-foreground">
            @{config.bot_username}
            {config.bot_id ? ` · id ${config.bot_id}` : ""}
          </span>
        )}
      </div>

      {/* Bot token */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">
          Bot token (từ @BotFather)
        </label>
        <input
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          placeholder={
            config?.has_token
              ? `Đã lưu (••••${config.token_hint ?? ""}) — nhập để thay`
              : "123456:ABC-DEF..."
          }
          className="w-full px-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
        />
      </div>

      {/* Webhook URL */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-muted-foreground">Webhook URL</label>
        <input
          type="text"
          value={webhookUrl}
          onChange={(e) => setWebhookUrl(e.target.value)}
          placeholder="https://service.hatinh.local/api/v1/integrations/telegram/webhook"
          className="w-full px-3 py-2 text-sm rounded-lg border bg-background font-mono text-xs focus:outline-none focus:ring-2 focus:ring-primary/30"
        />
        <p className="text-[11px] text-muted-foreground">
          Secret token được tạo & quản lý tự động khi đăng ký webhook.
        </p>
      </div>

      {/* Enabled */}
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
          className="rounded border-border"
        />
        Bật bot (nhận & trả lời tin nhắn)
      </label>

      {/* Actions */}
      <div className="flex flex-wrap gap-2 pt-1">
        <Button size="sm" onClick={handleSave} disabled={update.isPending}>
          {update.isPending ? (
            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
          ) : (
            <Save className="w-3.5 h-3.5 mr-1.5" />
          )}
          Lưu
        </Button>
        <Button
          size="sm"
          variant="secondary"
          onClick={handleRegister}
          disabled={register.isPending || !config?.has_token}
        >
          {register.isPending ? (
            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
          ) : (
            <Webhook className="w-3.5 h-3.5 mr-1.5" />
          )}
          Đăng ký webhook
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={handleTest}
          disabled={test.isPending || !config?.has_token}
        >
          {test.isPending ? (
            <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
          ) : (
            <Plug className="w-3.5 h-3.5 mr-1.5" />
          )}
          Kiểm tra kết nối
        </Button>
      </div>

      {/* Result */}
      {result && (
        <div
          className={`flex items-start gap-2 rounded-md p-2 text-xs ${
            result.ok
              ? "bg-green-500/10 text-green-700 dark:text-green-400"
              : "bg-destructive/10 text-destructive"
          }`}
        >
          {result.ok ? (
            <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
          ) : (
            <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
          )}
          <span className="break-words">{result.detail}</span>
        </div>
      )}
    </div>
  );
}
