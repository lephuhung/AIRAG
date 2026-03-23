import { translate } from "@/hooks/useTranslation";
import { useI18nStore } from "@/stores/i18nStore";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/vi";

dayjs.extend(relativeTime);

/**
 * Shared formatting utilities.
 */

/**
 * Format a date string as a human-readable relative date.
 */
export function formatRelativeDate(dateStr: string): string {
  const lang = useI18nStore.getState().language;
  return dayjs(dateStr).locale(lang).fromNow();
}

/**
 * Format a date string as a fixed date (DD/MM/YYYY).
 */
export function formatDate(dateStr: string): string {
  return dayjs(dateStr).format("DD/MM/YYYY");
}

/**
 * Format a byte count as a human-readable file size.
 */
export function formatFileSize(bytes: number): string {
  const lang = useI18nStore.getState().language;
  const t = (k: string) => translate(k, lang);
  
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} ${t("common.mb")}`;
  }
  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)} ${t("common.kb")}`;
  }
  return `${bytes} ${t("common.b")}`;
}

/**
 * Format processing time in milliseconds as a human-readable string.
 */
export function formatProcessingTime(ms: number): string {
  const lang = useI18nStore.getState().language;
  const t = (k: string) => translate(k, lang);

  if (ms < 1000) return `${Math.round(ms)}${t("common.ms")}`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}${t("common.s")}`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}${t("common.m")} ${seconds}${t("common.s")}`;
}
