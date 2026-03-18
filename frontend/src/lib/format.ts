/**
 * Shared formatting utilities.
 * Extracted from KnowledgeBasesPage + DocumentCard to avoid duplication.
 */

/**
 * Format a date string as a human-readable relative date.
 * "Today", "Yesterday", "3 days ago", "Mar 15", "Dec 3, 2024"
 */
export function formatRelativeDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();

  // Zero-out time to compare calendar days
  const dateDay = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const diffMs = today.getTime() - dateDay.getTime();
  const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;

  const month = date.toLocaleString("en-US", { month: "short" });
  const day = date.getDate();

  if (date.getFullYear() === now.getFullYear()) return `${month} ${day}`;
  return `${month} ${day}, ${date.getFullYear()}`;
}

/**
 * Format a byte count as a human-readable file size.
 * "2.4 MB", "128 KB", "512 B"
 */
export function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (bytes >= 1024) {
    return `${Math.round(bytes / 1024)} KB`;
  }
  return `${bytes} B`;
}

/**
 * Format processing time in milliseconds as a human-readable string.
 * "3.2s", "150ms", "1m 23s"
 */
export function formatProcessingTime(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}
