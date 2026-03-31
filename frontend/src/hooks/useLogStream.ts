/**
 * useLogStream — SSE streaming hook for system log viewing.
 *
 * Connects to GET /logs/stream?files=backend.log,worker_parse.log,...
 * and yields log lines as they are written.
 */
import { useState, useRef, useCallback, useEffect } from "react";
import { useAuthStore } from "@/stores/authStore";

const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

export interface LogLine {
  filename: string;
  line: string;
  level: "info" | "warning" | "error" | "debug" | "success";
  timestamp: number;
}

export interface LogStreamState {
  lines: LogLine[];
  isStreaming: boolean;
  error: string | null;
  selectedFiles: string[];
  start: (files: string[]) => void;
  stop: () => void;
  clear: () => void;
}

export function useLogStream(): LogStreamState {
  const [lines, setLines] = useState<LogLine[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);

  const abortRef = useRef<AbortController | null>(null);
  const readerRef = useRef<ReadableStreamDefaultReader | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      readerRef.current?.cancel();
    };
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    readerRef.current?.cancel();
    abortRef.current = null;
    readerRef.current = null;
    setIsStreaming(false);
  }, []);

  const clear = useCallback(() => {
    setLines([]);
  }, []);

  const start = useCallback((files: string[]) => {
    // Stop any existing stream
    stop();
    setSelectedFiles(files);
    setError(null);

    abortRef.current = new AbortController();
    setIsStreaming(true);

    const token = useAuthStore.getState().token;
    const filesParam = files.join(",");

    fetch(`${BASE_URL}/logs/stream?files=${encodeURIComponent(filesParam)}`, {
      method: "GET",
      headers: {
        Accept: "text/event-stream",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      signal: abortRef.current.signal,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        if (!response.body) {
          throw new Error("No response body");
        }
        return response.body.getReader();
      })
      .then((reader) => {
        readerRef.current = reader;
        const decoder = new TextDecoder();
        let sseBuffer = "";
        let currentEventType = "";

        function processChunk(value: Uint8Array) {
          sseBuffer += decoder.decode(value, { stream: true });
          const lines = sseBuffer.split("\n");
          sseBuffer = lines.pop() || "";

          for (const line of lines) {
            // Skip heartbeat comments
            if (line.startsWith(":")) continue;

            if (line.startsWith("event: ")) {
              currentEventType = line.slice(7).trim();
              continue;
            }

            if (line.startsWith("data: ")) {
              const jsonStr = line.slice(6).trim();
              if (!jsonStr) continue;

              try {
                const data = JSON.parse(jsonStr);

                if (currentEventType === "log_line") {
                  const logLine: LogLine = {
                    filename: data.filename || "",
                    line: data.line || "",
                    level: data.level || "info",
                    timestamp: Date.now(),
                  };
                  setLines((prev) => {
                    // Keep max 5000 lines to prevent memory issues
                    const next = [...prev, logLine];
                    if (next.length > 5000) {
                      return next.slice(-4000);
                    }
                    return next;
                  });
                } else if (currentEventType === "error") {
                  setError((data.message || "Unknown error"));
                }
              } catch {
                // Ignore malformed JSON
              }
            }
          }
        }

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) return;
            processChunk(value);
            read();
          }).catch((err) => {
            if ((err as Error)?.name === "AbortError") return;
            setError(`Stream read error: ${err}`);
            setIsStreaming(false);
          });
        }

        read();
      })
      .catch((err) => {
        if ((err as Error)?.name === "AbortError") return;
        setError(`Connection error: ${err}`);
        setIsStreaming(false);
      });
  }, [stop]);

  return {
    lines,
    isStreaming,
    error,
    selectedFiles,
    start,
    stop,
    clear,
  };
}
