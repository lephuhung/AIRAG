import { motion, AnimatePresence } from "framer-motion";
import {
  Brain,
  Lightbulb,
  Search,
  Database,
  PenLine,
  CheckCircle2,
  AlertCircle,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslation } from "@/hooks/useTranslation";
import type { AgentStep, AgentStepType } from "@/types";

// ---------------------------------------------------------------------------
// Step Configuration with Premium Colors
// ---------------------------------------------------------------------------

interface StepConfig {
  icon: LucideIcon;
  labelKey: string;
  color: string;
  bg: string;
  glow: string;
}

export const STEP_CONFIG: Record<AgentStepType, StepConfig> = {
  analyzing: { 
    icon: Brain, 
    labelKey: "rag.timeline.analyzing", 
    color: "text-indigo-500", 
    bg: "bg-indigo-500/10",
    glow: "shadow-[0_0_15px_-3px_rgba(99,102,241,0.4)]"
  },
  understood: { 
    icon: Lightbulb, 
    labelKey: "rag.timeline.understood", 
    color: "text-amber-500", 
    bg: "bg-amber-500/10",
    glow: "shadow-[0_0_15px_-3px_rgba(245,158,11,0.4)]"
  },
  retrieving: { 
    icon: Search, 
    labelKey: "rag.timeline.retrieving", 
    color: "text-blue-500", 
    bg: "bg-blue-500/10",
    glow: "shadow-[0_0_15px_-3px_rgba(59,130,246,0.4)]"
  },
  sources_found: { 
    icon: Database, 
    labelKey: "rag.timeline.sources_found", 
    color: "text-emerald-500", 
    bg: "bg-emerald-500/10",
    glow: "shadow-[0_0_15px_-3px_rgba(16,185,129,0.4)]"
  },
  generating: { 
    icon: PenLine, 
    labelKey: "rag.timeline.generating", 
    color: "text-fuchsia-500", 
    bg: "bg-fuchsia-500/10",
    glow: "shadow-[0_0_15px_-3px_rgba(217,70,239,0.4)]"
  },
  done: { 
    icon: CheckCircle2, 
    labelKey: "rag.timeline.done", 
    color: "text-emerald-500", 
    bg: "bg-emerald-500/10",
    glow: "shadow-[0_0_15px_-3px_rgba(16,185,129,0.4)]"
  },
  error: { 
    icon: AlertCircle, 
    labelKey: "rag.timeline.error", 
    color: "text-destructive", 
    bg: "bg-destructive/10",
    glow: "shadow-[0_0_15px_-3px_rgba(239,68,68,0.4)]"
  },
};

// ---------------------------------------------------------------------------
// Step-detail humanizer
//
// Backend status events sometimes carry internal node / tool identifiers
// (e.g. "memory_recall", "resolve_doc", or a react-executor dump like
// "Đang tra cứu: resolve_document_reference, search_documents"). These must
// never be shown verbatim to end users. We map known identifiers to friendly
// Vietnamese phrases and collapse any other bare identifier to a generic
// "thinking" message.
// ---------------------------------------------------------------------------

const GENERIC_THINKING = "Đang suy nghĩ...";

const INTERNAL_STEP_LABELS: Record<string, string> = {
  // Graph nodes / agents
  memory_recall: "Đang nhớ lại ngữ cảnh",
  query_analyzer: "Đang phân tích câu hỏi",
  query_enricher: "Đang bổ sung ngữ cảnh",
  supervisor: "Đang điều phối",
  intent_classifier: "Đang phân loại yêu cầu",
  resolve_doc: "Đang tìm văn bản",
  resolve_doc_agent: "Đang tìm văn bản",
  answer_generator: "Đang soạn câu trả lời",
  result_evaluator: "Đang kiểm tra kết quả",
  mongo_formatter: "Đang tổng hợp thông tin",
  react_executor: "Đang tra cứu thông tin",
  rag: "Đang tìm kiếm tài liệu",
  rag_agent: "Đang tìm kiếm tài liệu",
  people: "Đang tra cứu nhân sự",
  people_agent: "Đang tra cứu nhân sự",
  write: "Đang xử lý văn bản",
  write_agent: "Đang xử lý văn bản",
  direct: "Đang trả lời",
  clarification: "Đang làm rõ yêu cầu",
  // Tools
  resolve_document_reference: "Đang xác định văn bản",
  search_documents: "Đang tìm kiếm tài liệu",
  search_document: "Đang tìm kiếm tài liệu",
  get_document_content: "Đang đọc nội dung văn bản",
  list_documents: "Đang liệt kê tài liệu",
  recall_memory: "Đang nhớ lại ngữ cảnh",
  save_memory: "Đang ghi nhớ",
  search_people: "Đang tra cứu nhân sự",
  search_abbreviation: "Đang tra cứu viết tắt",
};

// A bare internal identifier: snake_case (has "_") or lowerCamelCase function name.
const IDENTIFIER_RE = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$|^[a-z]+[A-Z][a-zA-Z0-9]*$/;

/**
 * Convert a raw backend step `detail` into user-safe display text.
 * Never leaks internal node / function identifiers.
 */
export function humanizeStepDetail(detail?: string | null, fallback?: string): string {
  const raw = (detail ?? "").trim();
  if (!raw) return fallback || GENERIC_THINKING;

  // Whole string is a known internal identifier.
  const exact = INTERNAL_STEP_LABELS[raw.toLowerCase()];
  if (exact) return exact;

  // Identifier tokens hidden inside the string (e.g. tool-name dumps).
  const tokens = raw.match(/[A-Za-z_][A-Za-z0-9_]*/g) || [];
  const idTokens = tokens.filter(
    (tok) => IDENTIFIER_RE.test(tok) || tok.toLowerCase() in INTERNAL_STEP_LABELS
  );
  if (idTokens.length) {
    const mapped = idTokens
      .map((tok) => INTERNAL_STEP_LABELS[tok.toLowerCase()])
      .find(Boolean);
    return mapped || fallback || GENERIC_THINKING;
  }

  // Already human-readable text — keep as-is.
  return raw;
}

function formatMs(ms: number): string {
  const absMs = Math.abs(ms);
  if (absMs < 1000) return `${absMs}ms`;
  if (absMs < 60000) return `${(absMs / 1000).toFixed(1)}s`;
  return `${Math.floor(absMs / 60000)}m ${Math.floor((absMs % 60000) / 1000)}s`;
}

// ---------------------------------------------------------------------------
// PremiumStepIndicator — shows active step with pulsing aura
// ---------------------------------------------------------------------------

function PremiumStepIndicator({ step }: { step: AgentStep }) {
  const { t } = useTranslation();
  const config = STEP_CONFIG[step.step] || STEP_CONFIG.analyzing;
  const Icon = config.icon;
  const isActive = step.status === "active";

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95, y: -10 }}
      className="flex items-center gap-3 py-1"
    >
      <div className="relative">
        {/* Animated Rings for Active State */}
        {isActive && (
          <>
            <motion.div
              className={cn("absolute inset-0 rounded-xl", config.bg)}
              animate={{ scale: [1, 1.4, 1], opacity: [0.3, 0, 0.3] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            />
            <motion.div
              className={cn("absolute inset-0 rounded-xl", config.bg)}
              animate={{ scale: [1, 1.8, 1], opacity: [0.2, 0, 0.2] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut", delay: 0.5 }}
            />
          </>
        )}

        {/* Icon Container */}
        <div
          className={cn(
            "relative w-9 h-9 rounded-xl flex items-center justify-center transition-all duration-500",
            config.bg,
            config.glow,
            "backdrop-blur-md border border-white/10 dark:border-white/5"
          )}
        >
          {isActive ? (
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
              className="absolute inset-0 rounded-xl border border-dashed border-current opacity-20"
              style={{ color: "inherit" }}
            />
          ) : null}
          <Icon className={cn("w-5 h-5", config.color)} />
        </div>
      </div>

      <div className="flex flex-col">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold tracking-tight text-foreground/90">
            {humanizeStepDetail(step.detail, t(config.labelKey))}
          </span>
          <AnimatePresence>
            {isActive && (
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center gap-1"
              >
                <span className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.span
                      key={i}
                      className="w-1 h-1 rounded-full bg-primary/40"
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1, repeat: Infinity, delay: i * 0.2 }}
                    />
                  ))}
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <span className="text-[11px] text-muted-foreground/60 font-medium">
          {isActive ? "Đang xử lý..." : "Đã hoàn tất"}
        </span>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Main ThinkingTimeline component
// ---------------------------------------------------------------------------

interface ThinkingTimelineProps {
  steps: AgentStep[];
  className?: string;
  autoCollapse?: boolean;
}

export function ThinkingTimeline({
  steps,
  className,
}: ThinkingTimelineProps) {
  const { t } = useTranslation();
  if (steps.length === 0) return null;

  const activeStep = steps.find((s) => s.status === "active");
  const lastCompletedStep = [...steps].reverse().find((s) => s.status === "completed" && s.step !== "done");
  const doneStep = steps.find((s) => s.step === "done" && s.status === "completed");

  const displayStep = activeStep || doneStep || lastCompletedStep;

  if (!displayStep) return null;

  return (
    <div className={cn("py-2", className)}>
      <AnimatePresence mode="wait">
        <motion.div
          key={displayStep.id}
          initial={{ opacity: 0, y: 5 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -5 }}
          className="inline-flex flex-col"
        >
          {/* Header Label - Optional, very subtle */}
          <div className="flex items-center gap-1.5 mb-2 px-1">
            <Sparkles className="w-3 h-3 text-primary/50" />
            <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground/40">
              {t("rag.timeline.analyzing") || "Processing Engine"}
            </span>
          </div>

          {/* The Active/Display Step */}
          <div className={cn(
            "relative group rounded-2xl transition-all duration-500",
            "bg-gradient-to-br from-muted/40 to-muted/10",
            "border border-white/10 dark:border-white/5",
            "p-3 pr-6 shadow-sm hover:shadow-md",
            "backdrop-blur-xl"
          )}>
            <PremiumStepIndicator step={displayStep} />
            
            {/* Minimal Progress Bar at bottom */}
            {activeStep && (
              <div className="absolute bottom-0 left-0 right-0 h-[2px] overflow-hidden rounded-b-2xl">
                <motion.div 
                  className="h-full bg-gradient-to-r from-primary/20 via-primary to-primary/20"
                  animate={{ x: ["-100%", "100%"] }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  style={{ width: "50%" }}
                />
              </div>
            )}
          </div>

          {/* Mini History Dots - Show progress through steps */}
          <div className="flex items-center gap-1.5 mt-3 px-3">
            {steps.filter(s => s.step !== 'done').map((s) => (
              <div 
                key={s.id}
                className={cn(
                  "h-1 rounded-full transition-all duration-500",
                  s.status === 'completed' ? "w-4 bg-primary/30" : 
                  s.status === 'active' ? "w-8 bg-primary shadow-[0_0_8px_rgba(var(--primary),0.5)]" : 
                  "w-2 bg-muted-foreground/20"
                )}
              />
            ))}
            {doneStep && (
              <motion.span 
                initial={{ opacity: 0, scale: 0.5 }}
                animate={{ opacity: 1, scale: 1 }}
                className="text-[10px] font-mono text-emerald-500 font-bold ml-2"
              >
                {formatMs(doneStep.durationMs || 0)}
              </motion.span>
            )}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}