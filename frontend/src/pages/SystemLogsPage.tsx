import { useEffect, useRef, useState, useCallback } from "react";
import { useTranslation } from "@/hooks/useTranslation";
import { useLogStream, type LogLine } from "@/hooks/useLogStream";
import { api } from "@/lib/api";
import { useQuery } from "@tanstack/react-query";
import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  AlertCircle,
  Info,
  Search,
  Trash2,
  Download,
  Zap,
  Terminal as TerminalIcon,
  ChevronDown,
} from "lucide-react";
import { Select } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

const LOG_FILES = [
  { name: "backend.log", label: "Backend" },
  { name: "backend_restart.log", label: "Backend Restart" },
  { name: "worker_parse.log", label: "Parse Worker" },
  { name: "worker_embed.log", label: "Embed Worker" },
  { name: "worker_caption.log", label: "Caption Worker" },
  { name: "worker_kg.log", label: "KG Worker" },
  { name: "workers.log", label: "Workers" },
  { name: "workers_restart.log", label: "Workers Restart" },
  { name: "ocr_vllm.log", label: "OCR vLLM" },
  { name: "qwen_vllm.log", label: "Qwen vLLM" },
];

const LOG_LEVEL_STYLES: Record<string, { icon: typeof Info; color: string; bg: string }> = {
  error: { icon: AlertCircle, color: "text-red-400", bg: "bg-red-400/10" },
  warning: { icon: AlertTriangle, color: "text-amber-400", bg: "bg-amber-400/10" },
  info: { icon: Info, color: "text-blue-400", bg: "bg-blue-400/10" },
  success: { icon: Info, color: "text-emerald-400", bg: "bg-emerald-400/10" },
  debug: { icon: Info, color: "text-slate-500", bg: "bg-slate-500/10" },
};

interface LogFileInfo {
  name: string;
  size: number;
  modified: number | null;
  exists?: boolean;
}

interface HistoricalLogResponse {
  filename: string;
  total_lines: number;
  lines: string[];
}

function parseLogLevel(line: string): LogLine["level"] | "success" {
  const upper = line.toUpperCase();
  if (upper.includes("[ERROR]") || upper.includes("[CRITICAL]") || upper.includes("EXCEPTION") || upper.includes("FAILED")) return "error";
  if (upper.includes("[WARNING]") || upper.includes("[WARN]")) return "warning";
  if (upper.includes("[SUCCESS]") || upper.includes("DONE") || upper.includes("COMPLETED") || upper.includes("OK")) return "success";
  if (upper.includes("[DEBUG]") || upper.includes("[TRACE]")) return "debug";
  return "info";
}

export function SystemLogsPage() {
  const { t } = useTranslation();
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [selectedFile, setSelectedFile] = useState<string>("backend.log");
  const [historicalLines, setHistoricalLines] = useState<LogLine[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);

  const { data: filesInfo } = useQuery({
    queryKey: ["log-files"],
    queryFn: () => api.get<{ files: LogFileInfo[] }>("/logs/list"),
  });

  const {
    lines: streamLines,
    isStreaming,
    error,
    start,
    stop,
  } = useLogStream();

  // Load historical logs when file changes
  const loadHistoricalLogs = useCallback(async (filename: string) => {
    try {
      const data = await api.get<HistoricalLogResponse>(`/logs/${encodeURIComponent(filename)}?lines=500`);
      const parsed: LogLine[] = (data.lines || []).map((line, idx) => ({
        filename,
        line,
        level: parseLogLevel(line),
        timestamp: Date.now() - ((data.lines?.length || 0) - idx) * 1000,
      }));
      setHistoricalLines(parsed);
    } catch (err) {
      console.error("Failed to load historical logs:", err);
      setHistoricalLines([]);
    }
  }, []);

  // Handle file change
  const handleFileChange = useCallback((newFile: string) => {
    setSelectedFile(newFile);
    loadHistoricalLogs(newFile);
  }, [loadHistoricalLogs]);

  // Initial load
  useEffect(() => {
    loadHistoricalLogs(selectedFile);
  }, []);

  // Start streaming when file is selected
  useEffect(() => {
    start([selectedFile]);
    return () => stop();
  }, [selectedFile, start, stop]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [historicalLines, streamLines, autoScroll]);

  // Combine historical + stream lines
  const allLines = [...historicalLines, ...streamLines];
  
  const displayLines = allLines.filter(line => 
    line.line.toLowerCase().includes(searchQuery.toLowerCase())
  );


  const getLogIcon = (line: LogLine) => {
    const styles = LOG_LEVEL_STYLES[line.level] || LOG_LEVEL_STYLES.info;
    const Icon = styles.icon;
    return <Icon className={cn("w-3.5 h-3.5 shrink-0 mt-1", styles.color)} />;
  };

  const handleClear = () => {
    setHistoricalLines([]);
    // Note: stream lines are managed by the hook, maybe add a clear to hook too
  };

  const handleDownload = () => {
    const content = displayLines.map(l => `[${new Date(l.timestamp).toISOString()}] ${l.line}`).join("\n");
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${selectedFile}_${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col h-full bg-background overflow-hidden selection:bg-primary/30">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-border bg-card/50 backdrop-blur-sm px-6 py-4">
        <div className="max-w-screen-2xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="p-2 rounded-xl bg-primary/10 text-primary">
              <TerminalIcon className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">{t("nav.system_logs")}</h1>
              <div className="flex items-center gap-2 mt-0.5">
                {isStreaming ? (
                  <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider font-bold text-emerald-500">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    {t("logs.streaming")}
                  </span>
                ) : (
                  <span className="text-[10px] uppercase tracking-wider font-bold text-muted-foreground">
                    {t("logs.offline")}
                  </span>
                )}
                <span className="text-[10px] text-muted-foreground/50">•</span>
                <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                  {displayLines.length.toLocaleString()} {t("logs.lines")}
                </span>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative group">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
              <Input
                placeholder={t("common.search")}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 w-[200px] lg:w-[300px] bg-muted/20 border-transparent focus:bg-background transition-all"
              />
            </div>

            <div className="h-8 w-px bg-border/50 hidden md:block" />

            <div className="flex items-center gap-2">
              <div className="relative">
                <Select
                  value={selectedFile}
                  onChange={(e) => handleFileChange(e.target.value)}
                  className="pl-3 pr-8 py-1.5 h-10 w-[180px] text-sm font-medium bg-muted/20 border-transparent hover:bg-muted/30 transition-colors rounded-lg appearance-none"
                >
                  {LOG_FILES.map((file) => {
                    const fileInfo = filesInfo?.files?.find((f) => f.name === file.name);
                    return (
                      <option key={file.name} value={file.name} className="bg-card">
                        {file.label}
                        {fileInfo && fileInfo.size > 0 && ` (${(fileInfo.size / 1024).toFixed(0)}KB)`}
                      </option>
                    );
                  })}
                </Select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
              </div>

              <div className="flex items-center p-1 bg-muted/20 rounded-lg border border-border/50">
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-8 w-8 rounded-md transition-all",
                    autoScroll ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:bg-muted"
                  )}
                  onClick={() => setAutoScroll(!autoScroll)}
                  title="Auto-scroll"
                >
                  <Zap className={cn("w-4 h-4", autoScroll && "fill-current")} />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 rounded-md text-muted-foreground hover:bg-muted"
                  onClick={handleClear}
                  title="Clear view"
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 rounded-md text-muted-foreground hover:bg-muted"
                  onClick={handleDownload}
                  title="Export logs"
                >
                  <Download className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-screen-2xl mx-auto w-full px-6">
        {/* Error display */}
        {error && (
          <div className="mt-4 p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-sm flex items-center gap-3 animate-in fade-in slide-in-from-top-2">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <p className="font-medium">{error}</p>
          </div>
        )}
      </div>

      {/* Log display */}
      <div
        ref={logContainerRef}
        className="flex-1 overflow-y-auto font-mono text-[13px] leading-relaxed scroll-smooth"
      >
        <div className="max-w-screen-2xl mx-auto">
          {displayLines.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground/30 animate-in fade-in duration-500">
              <div className="p-6 rounded-full bg-muted/5 mb-4 italic">
                <TerminalIcon className="w-16 h-16 opacity-10" />
              </div>
              <p className="text-sm font-medium tracking-wide uppercase opacity-50">
                {searchQuery ? t("common.no_results") : t("logs.waiting")}
              </p>
            </div>
          ) : (
            <div className="py-4">
              {displayLines.map((logLine, idx) => {
                const levelStyle = LOG_LEVEL_STYLES[logLine.level] || LOG_LEVEL_STYLES.info;
                const timestamp = new Date(logLine.timestamp).toLocaleTimeString([], { 
                  hour12: false, 
                  hour: '2-digit', 
                  minute: '2-digit', 
                  second: '2-digit' 
                });

                return (
                  <div
                    key={`${logLine.filename}-${idx}-${logLine.timestamp}`}
                    className={cn(
                      "group flex items-start px-6 py-0.5 hover:bg-muted/50 transition-colors border-l-2 border-transparent relative",
                      logLine.level === 'error' && "bg-red-500/5 border-red-500/50",
                      logLine.level === 'warning' && "bg-amber-500/5 border-amber-500/30"
                    )}
                  >
                    {/* Line Number */}
                    <span className="w-10 shrink-0 text-[10px] text-muted-foreground/30 select-none text-right pr-4 font-normal group-hover:text-muted-foreground transition-colors">
                      {idx + 1}
                    </span>

                    {/* Timestamp */}
                    <span className="text-[11px] text-muted-foreground/40 w-20 shrink-0 select-none mr-2">
                      {timestamp}
                    </span>

                    {/* Level Icon */}
                    <div className="w-6 shrink-0 flex justify-center pt-1 mr-1">
                      {getLogIcon(logLine)}
                    </div>

                    {/* File Tag (shown on hover or compact) */}
                    <span className="hidden lg:block opacity-0 group-hover:opacity-100 transition-opacity absolute right-4 top-1 py-0.5 px-2 rounded bg-muted/20 text-[10px] text-muted-foreground font-medium select-none z-10 pointer-events-none">
                      {logLine.filename.replace(".log", "")}
                    </span>

                    {/* Log Content */}
                    <span className={cn(
                      "flex-1 break-all whitespace-pre-wrap",
                      levelStyle.color,
                      logLine.level === 'info' && "text-foreground opacity-90"
                    )}>
                      {/* Basic highlighting for common patterns */}
                      {logLine.line.split(/([ \t]|\[.*?\]|\".*?\"|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|GET|POST|PUT|DELETE|FETCH|OK|FAILED|ERROR|WARN|INFO|DEBUG|SUCCESS)/g).map((part, i) => {
                        if (/^\[.*?\]$/.test(part)) return <span key={i} className="text-muted-foreground/60">{part}</span>;
                        if (/^(GET|POST|PUT|DELETE)$/.test(part)) return <span key={i} className="text-sky-500 font-bold">{part}</span>;
                        if (/^(OK|SUCCESS)$/.test(part)) return <span key={i} className="text-emerald-600 font-bold">{part}</span>;
                        if (/^(FAILED|ERROR)$/.test(part)) return <span key={i} className="text-destructive font-bold">{part}</span>;
                        if (/^(WARN|WARNING)$/.test(part)) return <span key={i} className="text-amber-600 font-bold">{part}</span>;
                        if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(part)) return <span key={i} className="text-purple-500 opacity-80">{part}</span>;
                        if (/^\".*?\"$/.test(part)) return <span key={i} className="text-amber-500/80 italic">{part}</span>;
                        return part;
                      })}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Footer status */}
      <div className="flex-shrink-0 border-t border-border bg-card/50 backdrop-blur-sm px-6 py-2.5">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between text-[11px] font-medium tracking-wide uppercase text-muted-foreground/60">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-primary" />
              {LOG_FILES.find(f => f.name === selectedFile)?.label || selectedFile}
            </span>
            {streamLines.length > 0 && (
              <span className="flex items-center gap-2 text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded-full font-bold">
                <Zap className="w-3 h-3 fill-current" />
                {streamLines.length} {t("logs.new_incoming")}
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              {t("logs.status")}:
              <span className="text-emerald-500">{isStreaming ? "ONLINE" : "OFFLINE"}</span>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
