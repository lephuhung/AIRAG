import { useState, useRef, useEffect, useCallback, useMemo, memo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import "katex/dist/katex.min.css";
import {
  Loader2,
  Sparkles,
  ChevronDown,
} from "lucide-react";
import { toast } from "sonner";
import { generateId } from "@/lib/utils";
import { cleanChatTitle } from "@/lib/format";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";
import { BrandLogo } from "@/components/layout/BrandLogo";

import { useDocuments } from "@/hooks/useDocuments";
import { useChatHistory, useSessionDocuments } from "@/hooks/useChatHistory";
import { useRAGChatStream } from "@/hooks/useRAGChatStream";
import { useTranslation } from "@/hooks/useTranslation";
import { useCreateChatSession, useUpdateSessionTitle } from "@/hooks/useChatSessions";
import { useCreateAbbreviation } from "@/hooks/useAbbreviations";
import { useSTT } from "@/hooks/useSTT";
import { AbbreviationModal } from "@/components/rag/AbbreviationModal";
import { STEP_CONFIG } from "@/components/rag/ThinkingTimeline";
import { formatMentionName } from "@/components/rag/chat/utils";
import { SessionIdCtx, DebugCtx, AllSourcesCtx } from "@/components/rag/chat/contexts";
import type {
  ChatMessage,
  ChatSourceChunk,
  AgentStep,
  AgentStepType,
  Document,
  ChatHistoryResponse,
  PersistedChatMessage,
} from "@/types";
import { MessageBubble } from "@/components/rag/chat/message/MessageBubble";
import { ChatInputArea } from "@/components/rag/chat/input/ChatInputArea";
import type { AttachedFile } from "@/components/rag/chat/input/ChatInputArea";
import { SuggestionChips } from "@/components/rag/chat/input/SuggestionChips";

interface ChatPanelProps {
  sessionId: string | null;
  sessionTitle?: string;
}

export const ChatPanel = memo(function ChatPanel({
  sessionId,
  sessionTitle,
}: ChatPanelProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const createSession = useCreateChatSession();
  const { user } = useAuthStore();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mentionSelectedIndex, setMentionSelectedIndex] = useState(0);
  const [input, setInput] = useState(() => {
    if (!sessionId) return localStorage.getItem("hrag-draft-new") || "";
    return localStorage.getItem(`hrag-draft-${sessionId}`) || "";
  });
  const [attachedFiles, setAttachedFiles] = useState<AttachedFile[]>([]);
  const [referencedDocs, setReferencedDocs] = useState<{ id: string; filename: string; original_filename?: string }[]>([]);
  const [docMetadataMap, setDocMetadataMap] = useState<Map<string, Document>>(new Map());
  const skipResetRef = useRef<string | null>(null);

  // @docname mention states
  const [showMentionDropdown, setShowMentionDropdown] = useState(false);
  const [mentionSearch, setMentionSearch] = useState("");

  // 2. Personal Workspace Detection
  const { data: workspaces } = useQuery<any[]>({
    queryKey: ["workspaces"],
    queryFn: () => api.get<any[]>("/workspaces"),
  });

  const personalWorkspace = useMemo(() => {
    if (!workspaces) return undefined;
    const ws = workspaces as any;
    return (
      ws.find((item: any) => item.is_default === true) ||
      ws.find((item: any) => item.visibility === "personal") ||
      ws[0]
    );
  }, [workspaces]);

  const currentWorkspaceId = personalWorkspace?.id;

  // Fetch all workspace docs to build metadata map for inline file badges in history messages
  const { data: workspaceDocs } = useDocuments(currentWorkspaceId);

  // Fetch session documents from API for @mention autocomplete
  const { data: sessionDocsData } = useSessionDocuments(sessionId);

  const filteredMentionDocs = useMemo(() => {
    const sessionDocs = (sessionDocsData as any)?.documents || [];
    const workspaceDocsList = workspaceDocs || [];

    // Normalize any doc source into a single enriched shape so the dropdown can
    // surface meaningful hints (title, type, issuing agency, pages, size) and not
    // just a bare filename.
    const toMentionDoc = (doc: any) => ({
      id: String(doc.id),
      filename: doc.filename || doc.original_filename || "Untitled",
      original_filename: doc.original_filename || doc.filename,
      file_type: doc.file_type || (doc.filename?.split('.')?.pop()),
      document_number: doc.document_number ?? null,
      document_title: doc.document_title ?? null,
      document_type_name: doc.document_type?.name ?? null,
      issuing_agency: doc.issuing_agency ?? null,
      published_date: doc.published_date ?? null,
      page_count: doc.page_count ?? null,
      file_size: doc.file_size ?? null,
    });

    // Map locally attached files so they immediately appear in mentions
    // Use docMetadata.original_filename if available (from API response after upload)
    const attachedDocsInfo = attachedFiles.map((af) =>
      toMentionDoc({
        id: af.id,
        filename: af.file.name,
        original_filename: af.docMetadata?.original_filename || af.file.name,
        file_type: af.file.name.split('.').pop(),
        file_size: af.file.size,
      })
    );

    const allDocs = [
      ...attachedDocsInfo,
      ...sessionDocs.map(toMentionDoc),
      ...workspaceDocsList.map(toMentionDoc),
    ];
    if (allDocs.length === 0) return [];
    // Deduplicate by ID (prefer the first / richest occurrence)
    const uniqueDocs = Array.from(new Map(allDocs.map(d => [d.id, d])).values());

    // If no search term, show the most recent or all
    if (!mentionSearch) {
      return uniqueDocs.slice(0, 8);
    }

    // Filter by search term — match across filename, title, ref number and agency
    const search = mentionSearch.toLowerCase();
    return uniqueDocs
      .filter(doc =>
        doc.filename?.toLowerCase().includes(search) ||
        doc.original_filename?.toLowerCase().includes(search) ||
        doc.document_title?.toLowerCase().includes(search) ||
        doc.issuing_agency?.toLowerCase().includes(search) ||
        (doc.document_number && doc.document_number.toLowerCase().includes(search))
      )
      .slice(0, 8);
  }, [workspaceDocs, sessionDocsData, mentionSearch, attachedFiles]);

  // Sync index on search change
  useEffect(() => {
    setMentionSelectedIndex(0);
  }, [mentionSearch]);

  const handleMentionInput = useCallback((text: string, cursorPos: number) => {
    const textBeforeCursor = text.slice(0, cursorPos);
    const match = textBeforeCursor.match(/(?:^|\s)@([^\s]*)$/);

    if (!match) {
      setShowMentionDropdown(false);
      setMentionSearch("");
      return;
    }

    setMentionSearch(match[1]);
    setShowMentionDropdown(true);
  }, []);  // Helper: insert selected doc into input
  const insertMentionDoc = useCallback((doc: { id: string; filename: string; original_filename?: string }) => {
    const textBeforeCursor = input.slice(0, inputRef.current?.selectionStart || 0);
    const atIndex = textBeforeCursor.lastIndexOf('@');
    if (atIndex === -1) return;

    // Replace the @query part with the formatted document name
    const textBeforeAt = input.slice(0, atIndex);
    const textAfterMention = input.slice(inputRef.current?.selectionStart || 0);

    const displayName = formatMentionName(doc.original_filename || doc.filename);

    // Only inline the @mention TEXT when it is part of an actual sentence. If the
    // user hasn't typed anything besides the @query (empty sentence), don't litter
    // the input — the quoted file lives only as a persistent "Đang hỏi trong" chip
    // at the top. This avoids showing the same file in two places.
    const restOfSentence = `${textBeforeAt}${textAfterMention}`;
    const inlineMention = restOfSentence.trim().length > 0;
    const newInput = inlineMention
      ? `${textBeforeAt}@${displayName} ${textAfterMention}`
      : ""; // drop the lone @query — scope is shown only as the top chip

    setInput(newInput);
    setShowMentionDropdown(false);
    setMentionSearch("");

    // Set cursor position after the inserted mention (or just refocus when inline-less)
    setTimeout(() => {
      if (!inputRef.current) return;
      if (inlineMention) {
        const newCursorPos = textBeforeAt.length + displayName.length + 2; // +1 for @, +1 for space
        inputRef.current.setSelectionRange(newCursorPos, newCursorPos);
      }
      inputRef.current.focus();
    }, 0);

    // Add to referenced docs (persistent quote scope) in both cases
    if (!referencedDocs.find(d => d.id === doc.id)) {
      setReferencedDocs(prev => [...prev, { id: doc.id, filename: doc.filename, original_filename: doc.original_filename }]);
    }
  }, [input, referencedDocs]);

  // Helper: remove referenced doc
  const removeReferencedDoc = useCallback((docId: string) => {
    setReferencedDocs(prev => prev.filter(d => d.id !== docId));
  }, []);

  // Build docMetadataMap from workspace docs
  useEffect(() => {
    if (!workspaceDocs) return;
    const map = new Map<string, Document>();
    for (const doc of workspaceDocs) {
      map.set(doc.id, doc);
    }
    setDocMetadataMap(map);
  }, [workspaceDocs]);

  // Reset session state when switching chats/starting a new chat
  useEffect(() => {
    const isDebug = document.documentElement.classList.contains("debug-mode") ||
                    localStorage.getItem("hrag-debug-mode") === "true";

    // Only skip reset if this session was just created (handleSend set this)
    if (skipResetRef.current === sessionId) {
      if (isDebug) console.log(`[Persistence] Skipping reset for NEW session: ${sessionId}`);
      skipResetRef.current = null;
      return;
    }

    // For any other session change, clear the skip flag and reset
    if (skipResetRef.current !== null) {
      skipResetRef.current = null;
    }

    if (isDebug) console.log(`[Persistence] Session ID changed: ${sessionId || "new"}. Loading defaults...`);

    setMessages([]);
    setReferencedDocs([]);
    setAttachedFiles([]);

    const key = sessionId ? `hrag-draft-${sessionId}` : "hrag-draft-new";
    const mentionKey = sessionId ? `hrag-mentions-${sessionId}` : "hrag-mentions-new";

    const savedDraft = localStorage.getItem(key) || "";
    setInput(savedDraft);

    const savedMentions = localStorage.getItem(mentionKey);
    if (savedMentions) {
      try {
        const parsed = JSON.parse(savedMentions);
        setReferencedDocs(parsed);
        if (isDebug) console.log(`[Persistence] Restored ${parsed.length} mentions for ${sessionId || "new"}`);
      } catch (err) {
        console.error("Failed to parse saved mentions:", err);
      }
    }
  }, [sessionId]);

  useEffect(() => {
    const key = sessionId ? `hrag-draft-${sessionId}` : "hrag-draft-new";
    const mentionKey = sessionId ? `hrag-mentions-${sessionId}` : "hrag-mentions-new";

    // Text Draft
    if (input.trim()) {
      localStorage.setItem(key, input);
    } else {
      localStorage.removeItem(key);
    }

    // Mention Draft
    if (referencedDocs.length > 0) {
      localStorage.setItem(mentionKey, JSON.stringify(referencedDocs));
    } else {
      localStorage.removeItem(mentionKey);
    }
  }, [sessionId, input, referencedDocs]);

  // Abbreviation modal state
  const [isAbbModalOpen, setIsAbbModalOpen] = useState(false);
  const [selectedAbbShort, setSelectedAbbShort] = useState("");
  const createAbb = useCreateAbbreviation();

  const handleOpenAbbModal = useCallback((short: string) => {
    setSelectedAbbShort(short);
    setIsAbbModalOpen(true);
  }, []);

  const handleSaveAbb = async (data: { short_form: string; full_form: string; description?: string }) => {
    try {
      await createAbb.mutateAsync(data);
      toast.success(t("admin.abbreviations.toast.created"));
      setIsAbbModalOpen(false);
    } catch (err: any) {
      toast.error(err.message || t("admin.abbreviations.toast.error"));
    }
  };

  // Load chat history from PostgreSQL. Polls every 10s (see useChatHistory) so
  // messages sent on another channel (e.g. Telegram) appear here; isStreamingRef
  // pauses that polling while a web stream is in flight.
  const isStreamingRef = useRef(false);
  const { data: historyData, isLoading: historyLoading } = useChatHistory(
    sessionId,
    { isStreamingRef },
  );
  const queryClient = useQueryClient();

  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const scrollAnimRef = useRef<number | undefined>(undefined);
  const spacerRef = useRef<HTMLDivElement>(null);
  // Whether the user is following the latest content (near the bottom). When
  // false (they scrolled up to read), we never auto-scroll and show a jump button.
  const pinnedToBottomRef = useRef(true);
  const [showScrollDown, setShowScrollDown] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const docxInputRef = useRef<HTMLInputElement>(null);

  // Debug mode (Ctrl+Shift+D toggle, persisted in localStorage)
  const [debugMode, setDebugMode] = useState(() =>
    localStorage.getItem("hrag-debug-mode") === "true",
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === "D") {
        e.preventDefault();
        setDebugMode((prev) => {
          const next = !prev;
          localStorage.setItem("hrag-debug-mode", String(next));
          toast.success(next ? t("chat.debug_on") : t("chat.debug_off"));
          return next;
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // 7. Background File Upload & Polling
  const pollDocumentStatus = useCallback(async (docId: string) => {
    let attempts = 0;
    const maxAttempts = 30;

    const poll = async () => {
      try {
        const doc = await api.get<any>(`/documents/${docId}`);
        const docData = doc as any;
        if (docData.status === "indexed" || docData.status === "building_kg") {
          queryClient.invalidateQueries({ queryKey: ["documents", currentWorkspaceId] });
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "indexed", progress: 100, docMetadata: docData } : f
          ));
          return;
        }
        // Parse done → embedding in background. Allow chat immediately.
        if (docData.status === "chunking" || docData.status === "embedding") {
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "ready", progress: 70, docMetadata: docData } : f
          ));
          return;
        }
        if (docData.status === "failed") {
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "failed" } : f
          ));
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 2000);
        } else {
          setAttachedFiles(prev => prev.map(f =>
            f.id === docId ? { ...f, status: "failed" } : f
          ));
          toast.error(t("chat.upload_failed"));
        }
      } catch (err: any) {
        // If document not found (404) or other error, stop polling
        console.error("Polling failed:", err);
        setAttachedFiles(prev => prev.map(f =>
          f.id === docId ? { ...f, status: "failed" } : f
        ));
        // Don't continue polling on error - stop here
        return;
      }
    };

    poll();
  }, []);

  const handleFileSelect = useCallback(
    async (file: File) => {
      if (!currentWorkspaceId) {
        toast.error(t("chat.no_workspace"));
        return;
      }

      if (file.name.toLowerCase().endsWith(".doc")) {
        toast.error(t("chat.unsupported_doc"));
        return;
      }

      const tempId = generateId();
      const newAttachedFile: AttachedFile = {
        id: tempId,
        file,
        status: "uploading",
        progress: 10,
      };

      setAttachedFiles((prev) => [...prev, newAttachedFile]);

      try {
        const response = await api.uploadFile<any>(
          `/documents/upload/${currentWorkspaceId}`,
          file,
          { "X-Chat-Upload": "true" }
        );

        const docId = response.id;

        setAttachedFiles(prev => prev.map(f =>
          f.id === tempId ? { ...f, id: docId, status: "parsing", progress: 40, docMetadata: response } : f
        ));

        // Invalidate workspace docs so mention dropdown gets updated list
        queryClient.invalidateQueries({ queryKey: ["documents", currentWorkspaceId] });

        pollDocumentStatus(docId);
      } catch (err) {
        toast.error(t("chat.upload_failed"));
        setAttachedFiles(prev => prev.map(f =>
          f.id === tempId ? { ...f, status: "failed" } : f
        ));
      }
    },
    [currentWorkspaceId, pollDocumentStatus, queryClient, t]
  );

  const removeAttachment = useCallback((id: string) => {
    setAttachedFiles((prev) => prev.filter((f) => f.id !== id));
  }, []);

  // Auto-focus input for new chat sessions
  useEffect(() => {
    if (messages.length === 0 && !historyLoading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [messages.length, historyLoading]);


  // Sync DB history → local messages state when data loads.
  useEffect(() => {
    if (!sessionId) {
      setMessages([]);
      return;
    }

    if (historyData?.messages && historyData.session_id === sessionId) {
      setMessages((prev) => {
        const stepsMap = new Map<string, AgentStep[]>();
        for (const m of prev) {
          if (m.agentSteps?.length) stepsMap.set(m.id, m.agentSteps);
        }

        const dbMessages = historyData.messages.map((m) => ({
          id: m.message_id,
          role: m.role as "user" | "assistant",
          content: m.content,
          documentIds: m.document_ids ?? undefined,
          attachedDocs: m.attached_docs ?? undefined,
          sources: m.sources ?? undefined,
          relatedEntities: m.related_entities ?? undefined,
          imageRefs: m.image_refs ?? undefined,
          thinking: m.thinking ?? undefined,
          timestamp: m.created_at,
          potential_abbreviations: m.potential_abbreviations ?? undefined,
          peopleData: m.people_data ?? undefined,
          agentSteps: stepsMap.get(m.message_id) ?? (m.agent_steps?.length
            ? (m.agent_steps as any[]).map((s, i) => ({
              id: s.id || `hist-${m.message_id}-${i}`,
              step: s.step || 'analyzing',
              status: (s.status) || 'completed',
              detail: s.detail || (STEP_CONFIG[s.step as AgentStepType]?.labelKey ? t(STEP_CONFIG[s.step as AgentStepType].labelKey) : 'Processing'),
              timestamp: s.timestamp || (m.created_at ? new Date(m.created_at).getTime() : Date.now()),
              ...s
            })) as AgentStep[]
            : undefined),
        }));

        const dbIds = new Set(dbMessages.map((m) => m.id));
        const dbUserContents = new Set(dbMessages.filter(m => m.role === 'user').map(m => m.content));
        const dbAssistantContents = new Set(dbMessages.filter(m => m.role === 'assistant').map(m => m.content));

        const localOnly = prev.filter((m) => {
          if (dbIds.has(m.id)) return false;
          if (m.role === 'user' && dbUserContents.has(m.content)) return false;
          if (m.role === 'assistant' && dbAssistantContents.has(m.content)) return false;
          return true;
        });

        if (localOnly.length === 0) return dbMessages;

        return [...dbMessages, ...localOnly];
      });
    }
  }, [historyData]);

  // SSE streaming chat
  const updateSessionTitle = useUpdateSessionTitle();
  const stream = useRAGChatStream(
    sessionId,
    useCallback(
      (newTitle: string) => {
        if (sessionId) {
          updateSessionTitle(sessionId, newTitle);
        }
        queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      },
      [sessionId, updateSessionTitle, queryClient],
    ),
  );

  // Mirror live streaming state into the ref that gates history polling, so a
  // background refetch never lands on top of a half-streamed turn.
  useEffect(() => {
    isStreamingRef.current = stream.isStreaming;
  }, [stream.isStreaming]);
  const streamingMsgIdRef = useRef<string | null>(null);
  const agentStepsRef = useRef<AgentStep[]>([]);
  // Synchronous in-flight guard. `stream.isStreaming` only flips to true AFTER
  // the async session-creation + sendMessage call, leaving a window where rapid
  // Enter presses each pass the isStreaming check and spawn duplicate sessions.
  // This ref is set synchronously at the very top of handleSend to close that gap.
  const isSendingRef = useRef(false);
  useEffect(() => {
    if (stream.agentSteps.length > 0) {
      agentStepsRef.current = stream.agentSteps;
    }
  }, [stream.agentSteps]);

  // Sync server-assigned message ID to local streaming message
  useEffect(() => {
    if (stream.aiMessageId && streamingMsgIdRef.current) {
      const serverId = stream.aiMessageId;
      const localId = streamingMsgIdRef.current;
      if (serverId !== localId) {
        setMessages((prev) => {
          // Find and update the message by LOCAL id, then update ref to server id
          const idx = prev.findIndex((m) => m.id === localId);
          if (idx === -1) return prev;
          const updated = [...prev];
          updated[idx] = { ...updated[idx], id: serverId };
          return updated;
        });
        // Update ref to server ID so sync effect can find it
        streamingMsgIdRef.current = serverId;
      }
    }
  }, [stream.aiMessageId]);

  // Sync server-assigned user message ID to local message
  useEffect(() => {
    if (stream.userMessageId) {
      const serverId = stream.userMessageId;
      setMessages((prev) => {
        const lastUserIdx = [...prev].reverse().findIndex(m => m.role === 'user' && !m.id.startsWith('msg_'));
        if (lastUserIdx === -1) return prev;

        const idx = prev.length - 1 - lastUserIdx;
        if (prev[idx].id === serverId) return prev;

        const updated = [...prev];
        updated[idx] = { ...updated[idx], id: serverId };
        return updated;
      });
    }
  }, [stream.userMessageId]);

  // Double-rAF + easeOutCubic scroll to bottom
  const scrollToBottom = useCallback((smooth = true) => {
    const container = scrollContainerRef.current;
    if (!container) return;

    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = undefined;
    }

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const el = scrollContainerRef.current;
        if (!el) return;
        // Exclude the streaming spacer — target the end of real content, not the
        // empty room kept below it during streaming.
        const spacerH = spacerRef.current?.offsetHeight ?? 0;
        const target = Math.max(0, el.scrollHeight - spacerH - el.clientHeight);
        if (!smooth || Math.abs(target - el.scrollTop) < 10) {
          el.scrollTop = target;
          return;
        }

        const start = el.scrollTop;
        const distance = target - start;
        const duration = 400;
        const startTime = performance.now();

        const scrollEl = el;
        function animate(now: number) {
          const t = Math.min((now - startTime) / duration, 1);
          const ease = 1 - Math.pow(1 - t, 3);
          scrollEl.scrollTop = start + distance * ease;
          if (t < 1) {
            scrollAnimRef.current = requestAnimationFrame(animate);
          } else {
            scrollAnimRef.current = undefined;
          }
        }

        scrollAnimRef.current = requestAnimationFrame(animate);
      });
    });
  }, []);

  // Recompute whether the user is near the bottom (pinned) + toggle the jump
  // button. The streaming spacer is excluded so its empty room never counts as
  // unread content.
  const updateScrollState = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const spacerH = spacerRef.current?.offsetHeight ?? 0;
    const unreadBelow = el.scrollHeight - spacerH - el.scrollTop - el.clientHeight;
    const pinned = unreadBelow <= 80;
    pinnedToBottomRef.current = pinned;
    setShowScrollDown(!pinned);
  }, []);

  // Scroll user message to top of chat area
  const scrollUserMsgToTop = useCallback((msgId: string) => {
    if (scrollAnimRef.current) {
      cancelAnimationFrame(scrollAnimRef.current);
      scrollAnimRef.current = undefined;
    }
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const container = scrollContainerRef.current;
        if (!container) return;

        if (spacerRef.current) {
          spacerRef.current.style.height = `${container.clientHeight}px`;
        }

        const el = container.querySelector(`[data-message-id="${msgId}"]`) as HTMLElement | null;
        if (!el) return;

        const containerRect = container.getBoundingClientRect();
        const elRect = el.getBoundingClientRect();
        const relativeTop = elRect.top - containerRect.top + container.scrollTop;

        const PADDING_TOP = 12;
        const start = container.scrollTop;
        const target = Math.max(0, relativeTop - PADDING_TOP);
        if (Math.abs(target - start) < 5) return;

        const distance = target - start;
        const duration = 380;
        const startTime = performance.now();

        function animate(now: number) {
          const t = Math.min((now - startTime) / duration, 1);
          const ease = 1 - Math.pow(1 - t, 3);
          container!.scrollTop = start + distance * ease;
          if (t < 1) {
            scrollAnimRef.current = requestAnimationFrame(animate);
          } else {
            scrollAnimRef.current = undefined;
          }
        }
        scrollAnimRef.current = requestAnimationFrame(animate);
      });
    });
  }, []);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || !spacerRef.current) return;

    if (stream.isStreaming) {
      spacerRef.current.style.height = `${container.clientHeight}px`;
    } else {
      spacerRef.current.style.height = "0px";
    }
  }, [stream.isStreaming]);

  // Auto-scroll to the latest content ONLY when the user is already near the
  // bottom. If they've scrolled up to read, leave their position alone — the
  // jump button lets them catch up. This also prevents the jarring jump-to-end
  // right after streaming finishes.
  useEffect(() => {
    if (stream.isStreaming) return;
    if (pinnedToBottomRef.current) scrollToBottom();
  }, [messages, stream.isStreaming, scrollToBottom]);

  // Keep the pinned-state / jump-button fresh as content grows (streaming) and
  // as layout settles (e.g. the spacer collapsing when streaming ends).
  useEffect(() => {
    const id = requestAnimationFrame(updateScrollState);
    return () => cancelAnimationFrame(id);
  }, [messages, stream.streamingContent, stream.isStreaming, updateScrollState]);

  useEffect(() => {
    if (!streamingMsgIdRef.current) return;
    const id = streamingMsgIdRef.current;
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === id);
      if (idx === -1) return prev;
      const m = prev[idx];

      const newContent = stream.streamingContent;
      const newSources = stream.pendingSources.length > 0 ? stream.pendingSources : m.sources;
      const newImages = stream.pendingImages.length > 0 ? stream.pendingImages : m.imageRefs;
      const newThinking = stream.thinkingText || m.thinking;
      const newSteps = stream.agentSteps.length > 0 ? stream.agentSteps : m.agentSteps;
      const newPotentials = stream.potentialAbbreviations.length > 0 ? stream.potentialAbbreviations : m.potential_abbreviations;
      // Only use pendingPeople during active streaming; after complete fires, peopleData is already set
      // Don't update peopleData if pendingPeople is empty (might be from sendMessage reset before complete)
      const newPeople = stream.pendingPeople.length > 0
        ? stream.pendingPeople
        : (stream.isStreaming ? m.peopleData : (m.peopleData ?? stream.pendingPeople));

      if (
        m.content === newContent &&
        m.sources === newSources &&
        m.imageRefs === newImages &&
        m.thinking === newThinking &&
        m.agentSteps === newSteps &&
        m.potential_abbreviations === newPotentials &&
        m.peopleData === newPeople &&
        m.isStreaming === stream.isStreaming
      ) {
        return prev;
      }

      const updated = [...prev];
      updated[idx] = {
        ...m,
        content: newContent,
        sources: newSources,
        imageRefs: newImages,
        thinking: newThinking,
        agentSteps: newSteps,
        potential_abbreviations: newPotentials,
        peopleData: newPeople,
        isStreaming: stream.isStreaming,
      };
      return updated;
    });
  }, [stream.streamingContent, stream.pendingSources, stream.pendingImages, stream.thinkingText, stream.isStreaming, stream.agentSteps, stream.pendingPeople, stream.streamCompleteTick]);

  const handleSend = useCallback(
    async (text?: string) => {
      const msg = (text || input).trim();
      // Block send while any file is still uploading OR parsing — its markdown is not
      // yet in MinIO, so the backend would silently drop it. "ready"/"indexed" are safe.
      const isStillProcessing = attachedFiles.some(f => f.status === "uploading" || f.status === "parsing");
      if (isStillProcessing) {
        toast.info(t("chat.wait_for_files"));
        return;
      }
      if (!msg && attachedFiles.length === 0) return;
      if (stream.isStreaming) return;
      // Reentrancy guard: block a second send while the first is still in its
      // async setup (session creation → sendMessage), before isStreaming flips.
      // Without this, mashing Enter on a fresh chat creates duplicate sessions.
      if (isSendingRef.current) return;
      isSendingRef.current = true;

      try {
      let effectiveSessionId = sessionId;
      if (!effectiveSessionId) {
        try {
          const newSession = await createSession.mutateAsync({ title: cleanChatTitle(msg).slice(0, 30) || t("nav.new_chat") });
          effectiveSessionId = newSession.id;

          // Clear "New Chat" draft/mentions since we are sending it
          localStorage.removeItem("hrag-draft-new");
          localStorage.removeItem("hrag-mentions-new");

          if (document.documentElement.classList.contains("debug-mode")) {
            console.log(`[Persistence] Cleared "new" draft for session "${effectiveSessionId}"`);
          }

          skipResetRef.current = newSession.id;
          // Update URL so chat is bookmarkable
          navigate(`/chat/${newSession.id}`, { replace: true });
        } catch (err: any) {
          toast.error(t("chat.create_failed"));
          return;
        }
      }

      // Persistent quote scope: the user's active quoted docs (@mentions) AND
      // uploaded files stay in scope on EVERY turn until they explicitly remove the
      // chip (X) — we deliberately no longer drop a mention just because its @text
      // isn't typed in this particular message. This is what makes a bare follow-up
      // ("địa chỉ mail để báo cáo hằng tuần là gì") keep pointing at the quoted file
      // instead of drifting to the whole workspace. Dedup by id.
      const documentIds = Array.from(new Set([
        ...referencedDocs.map(d => d.id),
        // Only send files whose parse is confirmed done (markdown exists in MinIO).
        // "parsing" is excluded: markdown_s3_key may still be NULL → backend direct-fetch
        // skips the file and it becomes invisible to the agent (race condition fix).
        ...attachedFiles.filter(f => f.status === "indexed" || f.status === "ready").map(f => f.id),
      ]));
      // [debug] Trace which docs/files are actually sent to the backend + their status.
      // Enable via Ctrl+Shift+D (toggles document.documentElement "debug-mode").
      if (document.documentElement.classList.contains("debug-mode")) {
        console.log("[ChatPanel/handleSend] sending documentIds:", documentIds, {
          mentions: referencedDocs.map(d => ({ id: d.id, name: d.original_filename || d.filename })),
          attached: attachedFiles.map(f => ({ id: f.id, status: f.status, name: f.file?.name })),
        });
      }

      const attachedDocs = attachedFiles
        .filter(f => (f.status === "indexed" || f.status === "ready" || f.status === "parsing") && f.docMetadata)
        .map(f => f.docMetadata as Document);

      let msgToBackend = msg;
      // Replace any @mention TEXT that is actually present in this message with the
      // <document_id=…> tag (no-op for active scope docs the user didn't re-type).
      referencedDocs.forEach(doc => {
         const truncatedName = formatMentionName(doc.original_filename || doc.filename);
         const escapedTruncated = truncatedName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
         msgToBackend = msgToBackend.replace(new RegExp(`@${escapedTruncated}`, 'g'), `<document_id=${doc.id}>`);
      });

      const userMsg: ChatMessage = {
        id: generateId(),
        role: "user",
        content: msgToBackend,
        timestamp: new Date().toISOString(),
        documentIds: documentIds.length > 0 ? documentIds : undefined,
        attachedDocs: attachedDocs.length > 0 ? attachedDocs : undefined,
      };

      const assistantId = generateId();
      streamingMsgIdRef.current = assistantId;
      const placeholderMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, placeholderMsg]);
      setInput("");
      // Keep the active quote scope (referencedDocs + attachedFiles) across turns —
      // it is only cleared when the user removes a chip (X) or switches/creates a
      // chat. Only the typed draft text is cleared here.
      localStorage.removeItem(sessionId ? `hrag-draft-${sessionId}` : "hrag-draft-new");
      // Scroll new user message to top so agent response fills the space below
      scrollUserMsgToTop(userMsg.id);

      // Build history from previous messages (exclude the new user + placeholder)
      const history = messages.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const finalMsg = await stream.sendMessage(
        msgToBackend,
        history,
        false, // thinking is decided server-side by query complexity
        false, // search is no longer forced from the UI — the agent decides when to search
        effectiveSessionId || undefined,
        documentIds
      );

      // Finalize the streaming message (prefer finalMsg.agentSteps — directly from SSE loop,
      // fallback to ref snapshot, then to what was synced into the message during streaming)
      if (finalMsg) {
        // Invalidate sessions list query to fetch generated chat title from backend
        queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });

        setMessages((prev) => {
          const next = prev.map((m) =>
            m.id === assistantId
              ? {
                ...finalMsg,
                id: finalMsg.id,
                isStreaming: false, // Ensure streaming is false when finalizing
                agentSteps: finalMsg.agentSteps?.length
                  ? finalMsg.agentSteps
                  : agentStepsRef.current.length > 0
                    ? agentStepsRef.current
                    : m.agentSteps,
              }
              : m,
          );

          // Persistence runs in a background task on the server, so the DB may not
          // have this answer yet. Instead of force-refetching chat-history (which
          // would cache a pre-save, answer-less history and make the reply vanish
          // when switching tabs), sync the cache from local state so the full
          // exchange survives a remount without needing a reload.
          if (effectiveSessionId) {
            const persisted: PersistedChatMessage[] = next.map((m) => ({
              id: m.id,
              message_id: m.id,
              role: m.role,
              content: m.content,
              document_ids: m.documentIds ?? null,
              attached_docs: m.attachedDocs ?? null,
              sources: m.sources ?? null,
              related_entities: m.relatedEntities ?? null,
              image_refs: m.imageRefs ?? null,
              thinking: m.thinking ?? null,
              agent_steps: m.agentSteps ?? null,
              potential_abbreviations: m.potential_abbreviations ?? null,
              people_data: m.peopleData ?? null,
              created_at: m.timestamp ?? new Date().toISOString(),
            }));
            const cacheSessionId = effectiveSessionId;
            queueMicrotask(() =>
              queryClient.setQueryData<ChatHistoryResponse>(["chat-history", cacheSessionId], {
                session_id: cacheSessionId,
                messages: persisted,
                total: persisted.length,
              }),
            );
          }

          return next;
        });
        // Only clear streaming ref AFTER setMessages completes and isStreaming is false
        // This prevents race condition where sync effect could update wrong message
      } else if (stream.error) {
        toast.error(t("chat.failed", { error: stream.error }));
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                ...m,
                content: m.content || t("chat.error_fallback"),
                isStreaming: false,
              }
              : m,
          ),
        );
        streamingMsgIdRef.current = null;
      } else {
        // Cancelled — keep partial content
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, isStreaming: false } : m,
          ),
        );
        streamingMsgIdRef.current = null;
      }
      } finally {
        // Release the reentrancy guard once setup is done (stream.isStreaming now
        // owns the in-flight state) or whenever an early return/throw bails out.
        isSendingRef.current = false;
      }
    },
    [input, messages, stream, scrollUserMsgToTop, sessionId, navigate, createSession, t, attachedFiles],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showMentionDropdown) {
      if (e.key === "Escape") {
        if (showMentionDropdown) {
          setShowMentionDropdown(false);
          setMentionSearch("");
        } else if (input.trim()) {
          setInput("");
        }
        return;
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setMentionSelectedIndex((prev) => (prev + 1) % (filteredMentionDocs.length || 1));
        return;
      }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setMentionSelectedIndex((prev) => (prev - 1 + (filteredMentionDocs.length || 1)) % (filteredMentionDocs.length || 1));
        return;
      }
      if (e.key === "Enter" && filteredMentionDocs.length > 0) {
        e.preventDefault();
        insertMentionDoc(filteredMentionDocs[mentionSelectedIndex]);
        return;
      }
    }

    if (e.key === "Backspace" && inputRef.current && !showMentionDropdown) {
      const cursor = inputRef.current.selectionStart;
      if (cursor === inputRef.current.selectionEnd) {
        const textBefore = input.slice(0, cursor);

        for (const doc of referencedDocs) {
          const displayName = formatMentionName(doc.original_filename || doc.filename);
          const exactMatch1 = textBefore.endsWith(`@${displayName} `) ? `@${displayName} ` : "";
          const exactMatch2 = textBefore.endsWith(`@${displayName}`) ? `@${displayName}` : "";
          const matchStr = exactMatch1 || exactMatch2;
          
          if (matchStr) {
            e.preventDefault();
            const newInput = textBefore.slice(0, -matchStr.length) + input.slice(cursor);
            setInput(newInput);
            
            // Allow state to update, then fix cursor
            setTimeout(() => {
              if (inputRef.current) {
                const newPos = cursor - matchStr.length;
                inputRef.current.setSelectionRange(newPos, newPos);
              }
            }, 0);
            return;
          }
        }
      }
    }

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleVoiceTranscript = useCallback((text: string) => {
    setInput((prev) => {
      const sep = prev && !/\s$/.test(prev) ? " " : "";
      return prev + sep + text;
    });
    // Re-focus the textarea so the user can review/edit before sending.
    setTimeout(() => inputRef.current?.focus(), 0);
  }, [setInput]);

  const {
    isRecording: isVoiceRecording,
    isTranscribing: isVoiceTranscribing,
    toggleRecording: handleMicClick,
  } = useSTT({ onTranscript: handleVoiceTranscript, t });

  const handlePlusClick = useCallback(() => {
    docxInputRef.current?.click();
  }, []);

  // Collect all sources from all assistant messages for citation fallback.
  // When the model doesn't call search_documents but references citation IDs
  // from earlier answers, this allows those citations to still render as links.
  // NOTE: Must be declared before any early returns to satisfy Rules of Hooks.
  const allSourcesFlat = useMemo(() => {
    const seen = new Set<string>();
    const merged: ChatSourceChunk[] = [];
    for (const m of messages) {
      if (m.role === "assistant" && m.sources) {
        for (const s of m.sources) {
          const key = String(s.index);
          if (!seen.has(key)) {
            seen.add(key);
            merged.push(s);
          }
        }
      }
    }
    return merged;
  }, [messages]);

  if (historyLoading) {
    return (
      <div className="h-full flex items-center justify-center border-r">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <SessionIdCtx.Provider value={sessionId}>
      <DebugCtx.Provider value={debugMode}>
        <AllSourcesCtx.Provider value={allSourcesFlat}>
          <div className="flex flex-col h-full bg-background border-r relative z-0 overflow-hidden">
            {/* Header */}
            {/* Header */}
            <div className="flex-shrink-0 flex items-center justify-between px-6 py-4 bg-background/40 backdrop-blur-xl border-b border-border/40">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-background border border-border/60 overflow-hidden shadow-sm transition-transform hover:scale-105 shrink-0">
                  <img src="/logo.png" alt="AIRAG" className="w-5.5 h-5.5 object-contain" />
                </div>
                <div className="min-w-0 flex-1 relative">
                  {(() => {
                    const displayTitle =
                      cleanChatTitle(sessionTitle) ||
                      (sessionId ? `${t("chat.session", { id: sessionId })}` : t("chat.select_session"));
                    return (
                      <AnimatePresence mode="wait" initial={false}>
                        <motion.h2
                          key={displayTitle}
                          initial={{ opacity: 0, y: 5, filter: "blur(4px)" }}
                          animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                          exit={{ opacity: 0, y: -5, filter: "blur(4px)" }}
                          transition={{ duration: 0.3, ease: "easeOut" }}
                          className="text-[14px] font-bold tracking-tight text-foreground break-words leading-snug"
                        >
                          {displayTitle}
                        </motion.h2>
                      </AnimatePresence>
                    );
                  })()}
                </div>
              </div>
            </div>

            {/* Main Content Area */}
            {messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center px-4 overflow-y-auto pb-[10vh] scrollbar-none">
                <div className="w-full max-w-[720px] flex flex-col items-center translate-y-[-4vh]">
                  {/* Greeting */}
                  <div className="mb-12 text-center animate-in fade-in zoom-in-95 duration-1000 ease-out">
                    {/* Brand logo — swap via src/lib/brand.ts */}
                    <div className="mb-7 flex justify-center">
                      <BrandLogo
                        size={72}
                        glow
                        className="drop-shadow-[0_8px_24px_rgba(0,0,0,0.12)]"
                      />
                    </div>
                    <div className="inline-flex items-center gap-2 mb-6 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary shadow-[0_0_15px_rgba(var(--color-primary),0.1)]">
                      <Sparkles className="w-4 h-4" />
                      <span className="text-[12px] font-bold uppercase tracking-[0.1em]">{t("chat.ai_assistant")}</span>
                    </div>
                    <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-foreground mb-6 bg-gradient-to-b from-foreground to-foreground/60 bg-clip-text text-transparent">
                      {t("chat.greeting", { name: user?.full_name || "XayDung" })}
                    </h1>
                    <p className="text-xl md:text-2xl text-muted-foreground/50 font-medium max-w-[600px] mx-auto leading-relaxed">
                      {t("chat.assistant_desc")}
                    </p>
                  </div>

                  {/* Input Area (Centered) */}
                  <div className="w-full max-w-[720px] px-2 mb-6">
                    <ChatInputArea
                      input={input}
                      setInput={setInput}
                      isStreaming={stream.isStreaming}
                      onSend={handleSend}
                      onCancel={stream.cancel}
                      attachedFiles={attachedFiles}
                      onRemoveAttachment={removeAttachment}
                      inputRef={inputRef}
                      handleKeyDown={handleKeyDown}
                      onPlus={handlePlusClick}
                      onMic={handleMicClick}
                      micRecording={isVoiceRecording}
                      micTranscribing={isVoiceTranscribing}
                      t={t}
                      referencedDocs={referencedDocs}
                      onRemoveReferencedDoc={removeReferencedDoc}
                      showMentionDropdown={showMentionDropdown}
                      filteredMentionDocs={filteredMentionDocs}
                      onSelectMentionDoc={insertMentionDoc}
                      onCloseMentionDropdown={() => {
                        setShowMentionDropdown(false);
                        setMentionSearch("");
                      }}
                      onInputChange={handleMentionInput}
                      mentionSelectedIndex={mentionSelectedIndex}
                    />
                  </div>

                  {/* Suggestions Pills (Below) */}
                  <SuggestionChips onSelect={handleSend} />
                </div>
              </div>
            ) : (
              <>
                {/* Messages List */}
                <div ref={scrollContainerRef} onScroll={updateScrollState} className="flex-1 min-h-0 overflow-y-auto px-3 py-3 space-y-4 relative scrollbar-none">
                  <AnimatePresence mode="popLayout">
                    {messages.map((msg) => (
                      <motion.div
                        key={msg.id}
                        data-message-id={msg.id}
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8, transition: { duration: 0.15 } }}
                        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                      >
                        <MessageBubble
                          message={msg}
                          onAddAbbreviation={handleOpenAbbModal}
                          docMetadataMap={docMetadataMap}
                        />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                  <div ref={spacerRef} aria-hidden />
                </div>

                {/* Sticky Input Area (Fixed at bottom) */}
                <div className="relative flex-shrink-0 p-4 border-t/0 pb-8 last-msg-focus-fix bg-gradient-to-t from-background via-background/80 to-transparent">
                  {/* Jump-to-latest button — shown when the user has scrolled up
                      and there is unread content below. */}
                  {showScrollDown && (
                    <button
                      type="button"
                      onClick={() => scrollToBottom()}
                      aria-label={t("chat.scroll_to_latest")}
                      title={t("chat.scroll_to_latest")}
                      className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 z-20 flex items-center justify-center w-9 h-9 rounded-full bg-background/90 backdrop-blur border border-muted shadow-md text-muted-foreground hover:text-foreground hover:bg-muted transition-all animate-in fade-in slide-in-from-bottom-2 duration-200"
                    >
                      <ChevronDown className="w-5 h-5" />
                    </button>
                  )}
                  <div className="w-full max-w-[720px] mx-auto px-2">
                    <ChatInputArea
                      input={input}
                      setInput={setInput}
                      isStreaming={stream.isStreaming}
                      onSend={handleSend}
                      onCancel={stream.cancel}
                      attachedFiles={attachedFiles}
                      onRemoveAttachment={removeAttachment}
                      inputRef={inputRef}
                      handleKeyDown={handleKeyDown}
                      onPlus={handlePlusClick}
                      onMic={handleMicClick}
                      micRecording={isVoiceRecording}
                      micTranscribing={isVoiceTranscribing}
                      t={t}
                      referencedDocs={referencedDocs}
                      onRemoveReferencedDoc={removeReferencedDoc}
                      showMentionDropdown={showMentionDropdown}
                      filteredMentionDocs={filteredMentionDocs}
                      onSelectMentionDoc={insertMentionDoc}
                      onCloseMentionDropdown={() => {
                        setShowMentionDropdown(false);
                        setMentionSearch("");
                      }}
                      onInputChange={handleMentionInput}
                      mentionSelectedIndex={mentionSelectedIndex}
                    />
                    <p className="text-[10px] text-muted-foreground/40 mt-3 text-center font-medium">
                      {t("chat.input_hint")}
                    </p>
                  </div>
                </div>
              </>
            )}
          </div>
        </AllSourcesCtx.Provider>
      </DebugCtx.Provider>

      <AbbreviationModal
        open={isAbbModalOpen}
        onOpenChange={setIsAbbModalOpen}
        abbreviation={null}
        initialShortForm={selectedAbbShort}
        onSave={handleSaveAbb}
        isPending={createAbb.isPending}
      />

      {/* Hidden file input for DOCX format checking */}
      <input
        ref={docxInputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md,.pptx"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFileSelect(file);
          if (docxInputRef.current) docxInputRef.current.value = "";
        }}
        className="hidden"
      />
    </SessionIdCtx.Provider>
  );
});

