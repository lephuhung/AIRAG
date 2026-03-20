import { useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { MessageSquare, X } from "lucide-react";
import { cn } from "@/lib/utils";

import { useChatSessions, useCreateChatSession } from "@/hooks/useChatSessions";
import { ChatPanel } from "@/components/rag/ChatPanel";
import { DocumentViewer } from "@/components/rag/DocumentViewer";
import { useWorkspaceStore } from "@/stores/workspaceStore";

export function ChatPage() {
  const { sessionId: sessionIdStr } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();

  const currentSessionId = sessionIdStr ? Number(sessionIdStr) : null;

  // -- Store --
  const {
    selectedDoc,
    selectDoc,
    reset: resetStore,
    highlightChunks,
    scrollToPage,
    scrollToHeading,
    scrollToImageSrc,
  } = useWorkspaceStore();

  // Reset store when switching sessions
  useEffect(() => {
    resetStore();
  }, [currentSessionId, resetStore]);

  // -- Queries & Mutations --
  const { data: sessions, isLoading: loadingSessions } = useChatSessions();
  const createSession = useCreateChatSession();

  // Redirect to first session if none selected and sessions exist
  useEffect(() => {
    if (!currentSessionId && sessions && sessions.length > 0) {
      navigate(`/chat/${sessions[0].id}`, { replace: true });
    }
  }, [currentSessionId, sessions, navigate]);

  // -- Handlers --
  const handleNewSession = async () => {
    try {
      const newSession = await createSession.mutateAsync({ title: "New Chat" });
      navigate(`/chat/${newSession.id}`);
    } catch (error) {
      toast.error("Failed to create chat session");
    }
  };

  return (
    <div className="h-full overflow-hidden flex flex-col">
      {/* Mobile header (hidden on md) */}
      <div className="md:hidden flex h-14 items-center gap-3 border-b bg-background px-4 z-10">
        <MessageSquare className="w-5 h-5 text-primary" />
        <span className="font-semibold text-sm">NexusRAG Chat</span>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Middle: Chat Panel */}
        <div className={cn(
          "flex-1 h-full min-w-[320px] transition-all",
          selectedDoc ? "max-w-[50%]" : "max-w-7xl mx-auto"
        )}>
          {currentSessionId ? (
            <ChatPanel sessionId={currentSessionId} />
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-center px-4 space-y-4">
              <MessageSquare className="w-12 h-12 text-muted-foreground/30" />
              <div className="space-y-1">
                <h3 className="font-medium text-lg">NexusRAG Global Chat</h3>
                <p className="text-sm text-muted-foreground max-w-sm">
                  Create a new chat or select an existing one to start exploring all your accessible knowledge bases.
                </p>
              </div>
              <Button onClick={handleNewSession}>Start New Chat</Button>
            </div>
          )}
        </div>

        {/* Right: Document Viewer (conditionally rendered) */}
        {selectedDoc && (
          <div className="w-[50%] h-full border-l bg-background flex flex-col z-20">
            {/* Header with close button */}
            <div className="h-10 border-b flex items-center justify-between px-3 bg-muted/10 shrink-0">
              <span className="text-xs font-semibold truncate text-muted-foreground">
                Document Source
              </span>
              <button
                onClick={() => selectDoc(null)}
                className="p-1.5 rounded-md hover:bg-muted text-muted-foreground transition-colors"
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
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
