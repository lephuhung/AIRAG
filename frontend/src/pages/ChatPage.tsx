import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { MessageSquare, X } from "lucide-react";
import { cn } from "@/lib/utils";

import { useChatSessions, useCreateChatSession } from "@/hooks/useChatSessions";
import { ChatPanel } from "@/components/rag/ChatPanel";
import { DocumentViewer } from "@/components/rag/DocumentViewer";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useTranslation } from "@/hooks/useTranslation";

export function ChatPage() {
  const { sessionId: sessionIdStr } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
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

  // Reset store when switching sessions
  useEffect(() => {
    resetStore();
  }, [currentSessionId, resetStore]);

  // -- Queries & Mutations --
  const { data: sessions } = useChatSessions();
  const createSession = useCreateChatSession();

  // Redirect to first session if none selected and sessions exist
  useEffect(() => {
    if (!currentSessionId && sessions && sessions.length > 0) {
      navigate(`/chat/${sessions[0].id}`, { replace: true });
    }
  }, [currentSessionId, sessions, navigate]);

  const currentSession = sessions?.find(s => s.id === currentSessionId);
  const sessionTitle = currentSession?.title;

  // -- Handlers --
  const handleNewSession = async () => {
    try {
      const newSession = await createSession.mutateAsync({ title: t("nav.new_chat") });
      navigate(`/chat/${newSession.id}`);
    } catch (error) {
      toast.error(t("chat.create_failed"));
    }
  };

  return (
    <div className="h-full overflow-hidden flex flex-col">
      {/* Mobile header (hidden on md) */}
      <div className="md:hidden flex h-14 items-center gap-3 border-b bg-background px-4 z-10">
        <MessageSquare className="w-5 h-5 text-primary" />
        <span className="font-semibold text-sm">{t("chat.mobile_title")}</span>
      </div>

      <div className="flex-1 flex overflow-hidden relative">
        {/* Middle: Chat Panel */}
        <motion.div 
          layout
          initial={false}
          className={cn(
            "flex-1 h-full min-w-[320px] relative z-10",
            selectedDoc ? "w-1/3" : "w-full max-w-2xl lg:max-w-3xl xl:max-w-4xl mx-auto"
          )}
          transition={{ 
            type: "spring", 
            stiffness: 300, 
            damping: 34,
            mass: 0.8
          }}
        >
          {currentSessionId ? (
            <ChatPanel sessionId={currentSessionId} sessionTitle={sessionTitle} />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center px-4 space-y-4">
              <MessageSquare className="w-12 h-12 text-muted-foreground/30" />
              <div className="space-y-1">
                <h3 className="font-medium text-lg">{t("chat.welcome_title")}</h3>
                <p className="text-sm text-muted-foreground max-w-sm">
                  {t("chat.welcome_description")}
                </p>
              </div>
              <Button onClick={handleNewSession}>{t("nav.new_chat")}</Button>
            </div>
          )}
        </motion.div>

        <AnimatePresence mode="popLayout" initial={false}>
          {selectedDoc && (
            <motion.div 
              key={selectedDoc.id}
              initial={{ x: "100%", opacity: 0.5 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: "100%", opacity: 0, transition: { duration: 0.2, ease: "easeInOut" } }}
              transition={{ 
                type: "spring", 
                stiffness: 350, 
                damping: 38,
                mass: 0.8
              }}
              className="w-2/3 h-full border-l bg-background flex flex-col z-20 shadow-2xl relative"
            >
              <div className="absolute inset-y-0 -left-6 w-6 bg-gradient-to-r from-transparent to-black/[0.03] pointer-events-none" />
              
              {/* Header with close button */}
              <div className="h-10 border-b flex items-center justify-between px-3 bg-muted/20 shrink-0">
                <span className="text-xs font-semibold truncate text-foreground/70">
                  {t("chat.doc_source")}
                </span>
                <button
                  onClick={() => selectDoc(null)}
                  className="p-1.5 rounded-full hover:bg-muted text-muted-foreground transition-all duration-200 hover:rotate-90 active:scale-95"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              {/* The actual viewer, taking remaining height */}
              <div className="flex-1 overflow-hidden relative">
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
