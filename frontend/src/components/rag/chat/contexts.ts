import { createContext } from "react";
import type { ChatSourceChunk } from "@/types";

// Context to provide sessionId and debugMode to nested components.
export const SessionIdCtx = createContext<string | null>(null);
export const DebugCtx = createContext(false);

// Context: accumulated sources from ALL messages in the conversation.
// Used as fallback when a message references citation IDs from previous turns.
export const AllSourcesCtx = createContext<ChatSourceChunk[]>([]);
