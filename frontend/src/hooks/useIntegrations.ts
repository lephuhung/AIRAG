/**
 * Hooks for the integrations features: third-party API keys and Telegram linking.
 * Mirrors the query/mutation pattern in useAbbreviations.ts.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type {
  ApiKeyCreated,
  ApiKeyInfo,
  TelegramConfigInfo,
  TelegramConfigUpdate,
  TelegramLinkCode,
  TelegramLinkInfo,
  WebhookActionResult,
} from "@/types";

// ── API keys ────────────────────────────────────────────────────────────────

export function useApiKeys() {
  return useQuery({
    queryKey: ["api-keys"],
    queryFn: () => api.get<ApiKeyInfo[]>("/integrations/api-keys"),
  });
}

export function useCreateApiKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { name: string; scopes?: string[] }) =>
      api.post<ApiKeyCreated>("/integrations/api-keys", data),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["api-keys"] }),
  });
}

export function useRevokeApiKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.delete(`/integrations/api-keys/${id}`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["api-keys"] }),
  });
}

// ── Telegram linking ──────────────────────────────────────────────────────────

export function useTelegramLinks() {
  return useQuery({
    queryKey: ["telegram-links"],
    queryFn: () => api.get<TelegramLinkInfo[]>("/integrations/telegram/links"),
  });
}

export function useCreateLinkCode() {
  return useMutation({
    // Requires a current 2FA code (account must have 2FA enabled).
    mutationFn: (totpCode: string) =>
      api.post<TelegramLinkCode>("/integrations/telegram/link-code", {
        totp_code: totpCode,
      }),
  });
}

export function useUnlinkTelegram() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (chatId: string) =>
      api.delete(`/integrations/telegram/links/${chatId}`),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["telegram-links"] }),
  });
}

// ── Telegram bot config (superadmin) ───────────────────────────────────────────

export function useTelegramConfig(enabled = true) {
  return useQuery({
    queryKey: ["telegram-config"],
    queryFn: () => api.get<TelegramConfigInfo>("/integrations/telegram/config"),
    enabled,
  });
}

export function useUpdateTelegramConfig() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: TelegramConfigUpdate) =>
      api.put<TelegramConfigInfo>("/integrations/telegram/config", data),
    onSuccess: (data) =>
      queryClient.setQueryData(["telegram-config"], data),
  });
}

export function useRegisterTelegramWebhook() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.post<WebhookActionResult>("/integrations/telegram/config/register-webhook"),
    onSuccess: (res) => {
      if (res.config) queryClient.setQueryData(["telegram-config"], res.config);
    },
  });
}

export function useTestTelegramConnection() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      api.post<WebhookActionResult>("/integrations/telegram/config/test"),
    onSuccess: (res) => {
      if (res.config) queryClient.setQueryData(["telegram-config"], res.config);
    },
  });
}
