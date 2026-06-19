import { useEffect, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useParams } from "react-router-dom";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

import { useChatSessions } from "@/hooks/useChatSessions";
import { ChatPanel } from "@/components/rag/ChatPanel";
import { DocumentViewer } from "@/components/rag/DocumentViewer";
import { BrandLogo } from "@/components/layout/BrandLogo";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useTranslation } from "@/hooks/useTranslation";

export function ChatPage() {
  const { sessionId: sessionIdStr } = useParams<{ sessionId: string }>();
  const { t } = useTranslation();

  const currentSessionId = sessionIdStr || null;
  // -- Store --
  const {
    selectedDoc,
    selectDoc,
    reset: resetStore,
    highlightChunks,
    scrollToPage,
    scrollToHeading,
    scrollToImageSrc,
    clearScrollTarget,
  } = useWorkspaceStore();

  // Reset store khi chuyển đổi phiên chat (session) hoặc khi rời trang (unmount)
  useEffect(() => {
    resetStore();
    return () => {
      resetStore();
    };
  }, [currentSessionId, resetStore]);

  // -- Queries & Mutations --
  const { data: sessions } = useChatSessions();

  const currentSession = useMemo(
    () => sessions?.find((s: any) => String(s.id) === currentSessionId) ?? null,
    [sessions, currentSessionId]
  );
  const sessionTitle = currentSession?.title;

  return (
    <div className="h-full overflow-hidden flex flex-col">
      {/* Mobile header (hidden on md) */}
      <div className="md:hidden flex h-14 items-center gap-3 border-b border-border/60 bg-background/80 backdrop-blur-xl px-4 z-10">
        <BrandLogo size={28} glow />
        <span className="font-semibold text-sm tracking-tight bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
          {t("chat.mobile_title")}
        </span>
      </div>

      <div className="flex-1 flex overflow-hidden relative justify-center bg-background">

        {/* Middle: Chat Panel */}
        <motion.div 
          layout
          initial={false}
          className={cn(
            "h-full min-w-[320px] relative z-10 shrink-0",
            selectedDoc 
              ? "w-full md:w-1/2 max-w-[850px]" 
              : "flex-1 w-full max-w-2xl lg:max-w-3xl xl:max-w-4xl"
          )}
          transition={{ 
            type: "spring", 
            stiffness: 300, 
            damping: 32,
            mass: 1
          }}
        >
          <ChatPanel 
            sessionId={currentSessionId} 
            sessionTitle={sessionTitle} 
          />
        </motion.div>

        <AnimatePresence mode="popLayout" initial={false}>
          {selectedDoc && (
            <motion.div 
              key={selectedDoc.id}
              initial={{ x: "100%" }}
              animate={{ x: 0 }}
              exit={{ x: "100%", transition: { duration: 0.25, ease: "easeInOut" } }}
              transition={{ 
                type: "spring", 
                stiffness: 320, 
                damping: 34,
                mass: 1
              }}
              className="absolute md:relative right-0 inset-y-0 w-full md:w-1/2 max-w-[950px] h-full bg-background flex flex-col z-20 border-l border-border/60 shadow-[-10px_0_40px_-18px_rgba(0,0,0,0.25)]"
            >
              {/* Header của viewer - Cải tiến hiển thị tên file và hỗ trợ Glassmorphism */}
              <div className="relative h-12 border-b border-border/60 flex items-center justify-between gap-3 px-4 bg-background/70 backdrop-blur-xl sticky top-0 shrink-0 z-10">
                {/* Gradient accent line at the bottom edge */}
                <span
                  aria-hidden
                  className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent"
                />
                <span className="flex items-center gap-2.5 min-w-0">
                  <span className="h-6 w-1 rounded-full bg-primary/70 shrink-0" />
                  <span
                    className="text-sm font-semibold tracking-tight text-foreground/85 truncate"
                    title={selectedDoc.original_filename}
                  >
                    {selectedDoc.original_filename}
                  </span>
                </span>
                <button
                  onClick={() => selectDoc(null)}
                  className="p-2 rounded-full hover:bg-muted text-muted-foreground transition-all duration-200 hover:rotate-90 active:scale-95 flex-shrink-0"
                  aria-label={t("common.close")}
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              {/* The actual viewer, taking remaining height */}
              <div className="flex-1 overflow-hidden relative bg-muted/10">
                <DocumentViewer
                  doc={selectedDoc}
                  highlightChunks={highlightChunks}
                  scrollToPage={scrollToPage}
                  scrollToHeading={scrollToHeading}
                  scrollToImageSrc={scrollToImageSrc}
                  onScrolled={clearScrollTarget}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
