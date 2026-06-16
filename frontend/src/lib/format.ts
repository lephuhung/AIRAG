import { translate } from "@/hooks/useTranslation";
import { useI18nStore } from "@/stores/i18nStore";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import timezone from "dayjs/plugin/timezone";
import utc from "dayjs/plugin/utc";
import "dayjs/locale/vi";

dayjs.extend(relativeTime);
dayjs.extend(utc);
dayjs.extend(timezone);

const VIETNAM_TZ = "Asia/Ho_Chi_Minh";

/**
 * Shared formatting utilities.
 */

/**
 * Clean a chat session title for display.
 *
 * Titles auto-generated from the first message can leak artifacts when the user
 * attaches a file and mentions it: the raw `@FileName` mention token and the
 * backend's `<document_id=...>` tag. This strips those plus other special
 * characters, keeping letters/digits/spaces and basic punctuation (Vietnamese
 * accents preserved via the Unicode `\p{L}` class).
 */
export function cleanChatTitle(raw?: string | null): string {
  if (!raw) return "";
  let s = raw;
  // Drop document-id tags and @-mention tokens inserted by file attachments.
  s = s.replace(/<document_id=[^>]+>/gi, " ");
  s = s.replace(/@\S+/g, " ");
  // Remove special characters, keep word chars, whitespace and basic punctuation.
  s = s.replace(/[^\p{L}\p{N}\s.,!?()\-:/]/gu, " ");
  // Collapse whitespace.
  s = s.replace(/\s+/g, " ").trim();
  // Capitalize the first letter.
  return s ? s.charAt(0).toLocaleUpperCase() + s.slice(1) : s;
}

/**
 * Parse a timestamp coming from the backend into a dayjs object in Vietnam time.
 *
 * The backend stores and sends time in UTC. Numeric values are epoch
 * milliseconds (unambiguous). ISO strings WITHOUT a timezone designator (no
 * trailing "Z" or "+hh:mm") are naive-UTC: the browser would otherwise assume
 * they are local time and the displayed value would be off by the local offset
 * (e.g. -7h in Vietnam). We force such strings to be interpreted as UTC, then
 * convert to GMT+7 for display.
 */
export function parseServerDate(value: string | number | Date): dayjs.Dayjs {
  if (typeof value === "string") {
    const hasTz = /(?:Z|[+-]\d{2}:?\d{2})$/.test(value.trim());
    return (hasTz ? dayjs(value) : dayjs.utc(value)).tz(VIETNAM_TZ);
  }
  return dayjs(value).tz(VIETNAM_TZ);
}

/**
 * Format a server timestamp as HH:mm (24h) in Vietnam time.
 */
export function formatTime(value: string | number | Date): string {
  return parseServerDate(value).format("HH:mm");
}

/**
 * Format a date string as a human-readable relative date.
 */
export function formatRelativeDate(dateStr: string): string {
  const lang = useI18nStore.getState().language;
  return parseServerDate(dateStr).locale(lang).fromNow();
}

/**
 * Format a date string as a fixed date (DD/MM/YYYY).
 */
export function formatDate(dateStr: string): string {
  return parseServerDate(dateStr).format("DD/MM/YYYY");
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
