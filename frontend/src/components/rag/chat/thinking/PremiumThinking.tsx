import { useState, useRef, useEffect, useCallback, useMemo, memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Loader2, Sparkles, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTranslation } from "@/hooks/useTranslation";
import { STEP_CONFIG, humanizeStepDetail } from "@/components/rag/ThinkingTimeline";
import type { AgentStep } from "@/types";

// Thinking panel — collapsible violet-themed thinking process display
export const PremiumThinking = memo(({
  thinking,
  agentSteps,
  isStreaming,
  hasContent
}: {
  thinking: string;
  agentSteps?: AgentStep[];
  isStreaming?: boolean;
  hasContent?: boolean
}) => {
  const { t } = useTranslation();

  // Show if there's thinking text, agent steps, or we are currently streaming the beginning of a message
  const hasReasoning = (agentSteps && agentSteps.length > 0) || !!thinking || (isStreaming && !hasContent);

  // Find active/done steps
  const activeStep = useMemo(() => agentSteps?.find((s) => s.status === "active"), [agentSteps]);
  const doneStep = useMemo(() => agentSteps?.find((s) => s.step === "done" && s.status === "completed"), [agentSteps]);

  // Dynamic Label Logic
  const labelText = useMemo(() => {
    if (!!thinking && isStreaming && !hasContent) {
      return t("chat.thinking_process") || "Quá trình suy luận";
    }
    if (hasContent && isStreaming) {
      return t("rag.status.generating") || "Đang tạo câu trả lời...";
    }
    if (isStreaming && !hasContent) {
      if (activeStep) {
        const config = STEP_CONFIG[activeStep.step];
        return humanizeStepDetail(activeStep.detail, config ? t(config.labelKey) : "Đang xử lý...");
      }
      if (doneStep && !thinking) {
        return t("rag.timeline.done") || "Đã xong bước chuẩn bị";
      }
      return t("chat.analyzing_question") || "Đang phân tích câu hỏi...";
    }
    return t("chat.thinking_process") || "Quá trình suy luận";
  }, [activeStep, doneStep, thinking, isStreaming, hasContent, t]);

  const [expanded, setExpanded] = useState(false);
  const [showFull, setShowFull] = useState(false);
  const userToggledRef = useRef(false);

  const handleToggle = useCallback(() => {
    userToggledRef.current = true;
    setExpanded(prev => !prev);
  }, []);

  // Simplified Truncation - much faster than split/join for long strings
  const { displayThinking, isTruncated } = useMemo(() => {
    if (showFull || thinking.length < 1000) {
      return { displayThinking: thinking, isTruncated: false };
    }

    // Quick approximation of 200 words using character count (~1000 chars)
    // or finding the 200th space
    let count = 0;
    let index = -1;
    for (let i = 0; i < thinking.length; i++) {
      if (thinking[i] === ' ') {
        count++;
        if (count >= 200) {
          index = i;
          break;
        }
      }
    }

    if (index === -1) return { displayThinking: thinking, isTruncated: false };
    return {
      displayThinking: thinking.slice(0, index) + "...",
      isTruncated: true
    };
  }, [thinking, showFull]);

  // Auto-expand/collapse logic
  useEffect(() => {
    if (userToggledRef.current) return;

    // Auto-expand only when thinking starts and no final content yet
    if (isStreaming && !!thinking && thinking.length > 5 && !hasContent && !expanded) {
      setExpanded(true);
    }

    // Auto-collapse when content starts appearing or when streaming finishes to focus on the answer
    if ((hasContent || !isStreaming) && expanded) {
      setExpanded(false);
    }
  }, [thinking, isStreaming, hasContent, expanded]);

  if (!hasReasoning) return null;

  return (
    <div className="mb-3 group">
      <button
        onClick={handleToggle}
        className={cn(
          "flex items-center gap-2 px-2 py-1.5 rounded-md text-[13.5px] font-bold transition-all duration-200",
          "bg-transparent border-none outline-none select-none",
          "text-slate-700 hover:text-violet-700 hover:bg-violet-600/15",
          expanded && "text-violet-700 bg-violet-600/20"
        )}
      >
        <AnimatePresence mode="popLayout" initial={false}>
          <motion.div
            key={labelText}
            initial={{ opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -3 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
            className="flex items-center gap-2"
          >
            {isStreaming ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-violet-500" />
            ) : (
              <Sparkles className="w-3.5 h-3.5 text-violet-400" />
            )}
            <span>{labelText}</span>
            <ChevronDown className={cn("w-3.5 h-3.5 transition-transform duration-300", expanded && "rotate-180")} />
          </motion.div>
        </AnimatePresence>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="mt-2 pl-3 border-l-2 border-violet-100/50 space-y-3">
              {!!thinking && (
                <div className="text-[13.5px] text-slate-600 leading-relaxed font-normal">
                  <div className="prose prose-sm max-w-none prose-slate prose-p:leading-relaxed prose-pre:bg-slate-50">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {displayThinking}
                    </ReactMarkdown>
                    {isStreaming && !hasContent && (
                      <div className="flex items-center gap-1.5 mt-2 text-violet-500/60">
                        <Loader2 className="w-3 h-3 animate-spin" />
                        <span className="text-[11px] italic font-medium">Đang suy nghĩ...</span>
                      </div>
                    )}
                  </div>

                  {isTruncated && (
                    <button
                      onClick={() => setShowFull(true)}
                      className="mt-2 text-violet-600 hover:text-violet-700 font-medium text-[12px] underline-offset-4 hover:underline"
                    >
                      {t("common.show_more")}
                    </button>
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
});
